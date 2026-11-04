"""Explicit, verified installation of a complete Windows FFmpeg/ffprobe pair.

Discovery/probe checks NEVER access the network. Only install_windows_runtime
(called by an explicit Repair action or the Windows setup script) downloads.
No recordings are read or uploaded. No system PATH/registry changes.
Compatible with v0.5.3 media_tools for the small in-place repair package.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict
from fractions import Fraction
from functools import lru_cache
from pathlib import Path, PurePosixPath
from uuid import uuid4

from oculostudio import media_tools as mt
from oculostudio.errors import DependencyError, ValidationError
from oculostudio.util import CancellationToken, atomic_write_json, emit

# Exact upstream release and checksum; not an unauthenticated 'latest' URL.
# SHA-256 checked against Gyan's .zip.sha256 publication on 2026-09-26.
BUILD = '8.1.2-essentials_build'
BUILD_URL = 'https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-8.1.2-essentials_build.zip'
BUILD_SHA256 = 'db580001caa24ac104c8cb856cd113a87b0a443f7bdf47d8c12b1d740584a2ec'
PROVIDER_PAGE = 'https://www.gyan.dev/ffmpeg/builds/'
CHECKSUM_URL = BUILD_URL + '.sha256'
MAX_ARCHIVE_BYTES = 512 * 1024**2
MAX_EXPANDED_BYTES = 2 * 1024**3
_ALLOWED_HOSTS = {'www.gyan.dev', 'gyan.dev'}
_INSTALL_LOCK = threading.Lock()
ENV_OVERRIDES = ('OCULOSTUDIO_FFMPEG_PATH', 'OCULOSTUDIO_FFMPEG',
                 'OCULOSTUDIO_FFPROBE_PATH', 'OCULOSTUDIO_FFPROBE', 'IMAGEIO_FFMPEG_EXE')


def _flags() -> int:
    return subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0


def _supported_windows() -> bool:
    return (sys.platform == 'win32' and struct.calcsize('P') == 8
            and platform.machine().lower() in {'amd64', 'x86_64'})


def runtime_root() -> Path:
    return mt.config_path().parent / 'runtimes'


@lru_cache(maxsize=32)
def _probe_smoke(path: str, size: int, mtime_ns: int) -> dict:
    """Exercise the precise -show_frames/JSON path used by Native clock indexing."""
    fixture = Path(__file__).with_name('resources') / 'media_probe_test.mp4'
    if not fixture.is_file():
        raise DependencyError('Media probe test fixture is missing; re-extract the complete update package.')
    command = [path, '-v', 'error', '-threads', '1', '-select_streams', 'v:0',
               '-show_frames', '-show_streams', '-show_entries',
               'frame=pts,best_effort_timestamp,duration,pkt_duration:'
               'stream=time_base,start_pts,duration_ts,width,height', '-of', 'json', str(fixture)]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=20, creationflags=_flags())
        if proc.returncode:
            raise ValueError(proc.stderr[-1400:])
        data = json.loads(proc.stdout)
        frames, streams = data.get('frames', []), data.get('streams', [])
        if len(frames) != 3 or len(streams) != 1:
            raise ValueError('Expected 3 video frames and one video stream in the synthetic MP4.')
        timestamps = [int(f.get('pts', f.get('best_effort_timestamp'))) for f in frames]
        tb = Fraction(streams[0]['time_base'])
        if not all(b > a for a,b in zip(timestamps, timestamps[1:])) or tb <= 0:
            raise ValueError('Frame timestamps/time base are invalid.')
        if int(streams[0]['duration_ts']) <= 0:
            raise ValueError('Stream duration_ts is missing or non-positive.')
        return {'status': 'passed', 'synthetic_frames': 3,
                'timestamps_strictly_increasing': True, 'time_base': str(tb)}
    except (OSError, ValueError, TypeError, KeyError, subprocess.TimeoutExpired) as exc:
        raise DependencyError(f'ffprobe was found but actual MP4 frame/JSON reading failed: {exc}') from exc


def probe_preflight(ffprobe_path: str | None = None, *, ffmpeg_path: str | None = None) -> dict:
    tool = mt.resolve_media_tool('ffprobe', ffprobe_path, sibling_hint=ffmpeg_path)
    s = Path(tool.path).stat()
    return {'ffprobe': asdict(tool), 'frame_probe_test': dict(_probe_smoke(tool.path, s.st_size, s.st_mtime_ns))}


def complete_preflight(ffmpeg_path: str | None = None, ffprobe_path: str | None = None,
                       *, preserve_audio: bool = True) -> dict:
    # A good FFmpeg must never make a missing ffprobe appear 'ready'.
    probe = probe_preflight(ffprobe_path, ffmpeg_path=ffmpeg_path)
    encoder = mt.privacy_encoder_preflight(ffmpeg_path, ffprobe_path=probe['ffprobe']['path'],
                                          preserve_audio=preserve_audio)
    return {**encoder, **probe}


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parsed = urllib.parse.urlsplit(newurl)
        if parsed.scheme != 'https' or parsed.hostname not in _ALLOWED_HOSTS:
            raise DependencyError('The media download redirected outside the approved HTTPS build host.')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download(destination: Path, token: CancellationToken, progress=None) -> dict:
    """Stream to disk with TLS validation, bounded size, timeout and cancellation."""
    request = urllib.request.Request(BUILD_URL, headers={'User-Agent': 'OculoStudio-MediaRepair/0.5.4'})
    opener = urllib.request.build_opener(_SafeRedirect())
    digest = hashlib.sha256(); total = 0; start = time.monotonic(); next_notice = 0.0
    try:
        with opener.open(request, timeout=30) as response, destination.open('wb') as output:
            expected = int(response.headers.get('Content-Length') or 0)
            if expected > MAX_ARCHIVE_BYTES:
                raise DependencyError('Media archive exceeds the download size limit.')
            while True:
                token.raise_if_cancelled()
                block = response.read(1024**2)
                if not block:
                    break
                total += len(block)
                if total > MAX_ARCHIVE_BYTES or time.monotonic() - start > 3600:
                    raise DependencyError('Media download exceeded its size or one-hour time limit.')
                output.write(block); digest.update(block)
                now = time.monotonic()
                if now >= next_notice:
                    fraction = min(.70, .06 + .64 * total / expected) if expected else .10
                    suffix = f' / {expected/1024**2:.1f} MiB' if expected else ' MiB total unknown'
                    emit(progress, fraction, f'Downloading FFmpeg + ffprobe: {total/1024**2:.1f}{suffix}')
                    next_notice = now + .25
            if expected and total != expected:
                raise DependencyError(f'Incomplete download: got {total} of {expected} bytes.')
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise DependencyError('Could not download the complete media runtime. Check internet/proxy access to '
                              'www.gyan.dev; retry Repair, or choose an existing full FFmpeg bin folder. '
                              'No executable was installed and no recording was changed. ' + str(exc)) from exc
    if digest.hexdigest() != BUILD_SHA256:
        raise DependencyError('FFmpeg download SHA-256 mismatch. Refusing to unpack or run it; '
                              'no previous working tools were changed. Retry or use a trusted local bin folder.')
    return {'url': BUILD_URL, 'sha256': digest.hexdigest(), 'bytes': total}


def _verify_local_archive(path: Path, token: CancellationToken) -> dict:
    h = hashlib.sha256(); size = 0
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4*1024**2), b''):
            token.raise_if_cancelled(); size += len(block)
            if size > MAX_ARCHIVE_BYTES: raise DependencyError('Media archive too large.')
            h.update(block)
    if h.hexdigest() != BUILD_SHA256:
        raise DependencyError('Offline ZIP is not the pinned, verified FFmpeg 8.1.2 essentials ZIP. '
                              'For another trusted build, select its extracted bin directory in Video tools.')
    return {'local_archive': str(path), 'sha256': h.hexdigest(), 'bytes': size}


def _safe_unpack(archive: Path, destination: Path, token: CancellationToken) -> dict[str, Path]:
    """Validate EVERY member before writing; retain upstream LICENSE/README/docs."""
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist(); seen = set(); total = 0
        if len(infos) > 50000: raise DependencyError('Too many members in media archive.')
        for info in infos:
            text = info.filename.replace('\\', '/')
            p = PurePosixPath(text)
            key = text.rstrip('/').casefold()
            if (not p.parts or p.is_absolute() or '..' in p.parts or any(':' in v for v in p.parts)
                or key in seen or stat.S_ISLNK(info.external_attr >> 16)):
                raise DependencyError(f'Unsafe or duplicate media ZIP member: {text}')
            if any(v.rstrip(' .') != v for v in p.parts):
                raise DependencyError(f'Unsafe Windows archive member: {text}')
            if any(re.fullmatch(r'(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?', v, re.I) for v in p.parts):
                raise DependencyError('Windows device path in media archive.')
            seen.add(key); total += info.file_size
            if total > MAX_EXPANDED_BYTES: raise DependencyError('Expanded media archive too large.')
        executables = {}
        for name in ('ffmpeg', 'ffprobe'):
            matches = [i for i in infos if PurePosixPath(i.filename).name.lower() == name + '.exe']
            if len(matches) != 1:
                raise DependencyError(f'Full build must contain exactly one {name}.exe, found {len(matches)}.')
            executables[name] = destination / matches[0].filename
        if executables['ffmpeg'].parent != executables['ffprobe'].parent:
            raise DependencyError('FFmpeg and ffprobe must belong to the same bin directory.')
        for info in infos:
            token.raise_if_cancelled()
            target = destination / info.filename.replace('\\', '/')
            if info.is_dir(): target.mkdir(parents=True, exist_ok=True); continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open('wb') as dst:
                while True:
                    token.raise_if_cancelled(); block = src.read(4*1024**2)
                    if not block: break
                    dst.write(block)
            # Permission only helps local Linux test fixtures; normal Windows uses PE execution.
            if target.suffix.lower() == '.exe': target.chmod(0o700)
        return executables


def _check_pair(paths: dict[str, Path]) -> dict:
    return complete_preflight(str(paths['ffmpeg']), str(paths['ffprobe']))


def _activate_existing_pair(paths: dict[str, Path]) -> dict:
    checks = _check_pair(paths)
    cfg = mt.config_path(); before = cfg.read_bytes() if cfg.exists() else None
    if before is not None:
        backup = cfg.with_name('media_tools.before-repair-' + uuid4().hex[:8] + '.json')
        backup.write_bytes(before)
    else:
        backup = None
    try:
        mt.save_tool_settings({k:str(v) for k,v in paths.items()})
        checks = complete_preflight()
    except Exception:
        if before is None: cfg.unlink(missing_ok=True)
        else:
            tmp = cfg.with_name(cfg.name + '.restore-' + uuid4().hex)
            tmp.write_bytes(before); os.replace(tmp, cfg)
        raise
    return {'ready': True, 'downloaded': False, 'action': 'activated_application_local_pair',
            'settings_file': str(cfg), 'checks': checks, 'settings_backup': str(backup) if backup else None}


def _application_pairs():
    for folder in mt._portable_directories():
        for base in (folder, folder/'bin'):
            paths = {name: base/(name+'.exe') for name in ('ffmpeg','ffprobe')}
            if all(p.is_file() for p in paths.values()):
                yield paths


def install_windows_runtime(*, archive_path: str | None = None, progress=None,
                            cancellation: CancellationToken | None = None, force: bool = False) -> dict:
    token = cancellation or CancellationToken.create()
    token.raise_if_cancelled()
    if not _INSTALL_LOCK.acquire(blocking=False):
        raise DependencyError('A media repair is already running. Wait for it to finish.')
    try:
        emit(progress, .01, 'Checking BOTH media tools before installation')
        if not force and not archive_path:
            try:
                checks = complete_preflight()
            except (DependencyError, OSError):
                pass
            else:
                emit(progress, 1.0, 'Ready: FFmpeg encoding AND ffprobe frame reading passed; no download needed')
                return {'ready': True, 'downloaded': False, 'checks': checks, 'action': 'reused_existing_tools'}
        if not _supported_windows():
            raise DependencyError('Automatic complete-runtime download supports Windows x64 only. '
                                  'On Linux/macOS/ARM, configure an existing full FFmpeg installation.')
        active_overrides = [k for k in ENV_OVERRIDES if os.environ.get(k)]
        if active_overrides:
            raise DependencyError('Environment variables override saved media paths: ' + ', '.join(active_overrides) +
                                  '. Clear these overrides or fix their executables before automatic repair.')
        if not archive_path:
            for paths in _application_pairs():
                try:
                    emit(progress, .05, 'Testing the already-bundled offline pair; no download')
                    result = _activate_existing_pair(paths)
                except (DependencyError, OSError):
                    continue
                emit(progress, 1.0, 'Ready: bundled FFmpeg + ffprobe activated and tested')
                return result
        root = runtime_root(); root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.install-', dir=root) as temporary:
            stage = Path(temporary)
            zip_path = Path(archive_path).expanduser().resolve() if archive_path else stage / 'download.zip'
            source = _verify_local_archive(zip_path, token) if archive_path else _download(zip_path, token, progress)
            emit(progress, .73, 'SHA-256 verified. Extracting full tool pair and license documents')
            unpacked = stage / 'build'; unpacked.mkdir()
            paths = _safe_unpack(zip_path, unpacked, token)
            token.raise_if_cancelled()
            emit(progress, .83, 'Testing ffprobe actual frame JSON and FFmpeg H.264 / AAC encoding')
            checks = _check_pair(paths)
            token.raise_if_cancelled()
            final = root / ('ffmpeg-' + BUILD + '-' + BUILD_SHA256[:12] + '-' + uuid4().hex[:8])
            relative = {name: p.relative_to(unpacked) for name,p in paths.items()}
            unpacked.rename(final)
            final_paths = {k: final / v for k,v in relative.items()}
            cfg = mt.config_path(); old_bytes = cfg.read_bytes() if cfg.exists() else None
            backup = None; settings_committed = False
            try:
                # Explicit probes above bypass stale settings. Repair intentionally selects the verified pair.
                token.raise_if_cancelled()
                checks = _check_pair(final_paths)
                if old_bytes is not None:
                    backup = cfg.with_name('media_tools.before-repair-' + uuid4().hex[:8] + '.json')
                    backup.write_bytes(old_bytes)
                mt.save_tool_settings({name: str(path) for name,path in final_paths.items()})
                settings_committed = True
                # Check discovery through the same path the application will use next time.
                checks = complete_preflight()
                manifest = {'schema_version': 'oculostudio-media-runtime-1.0', 'build': BUILD,
                            'provider_page': PROVIDER_PAGE, 'checksum_url': CHECKSUM_URL,
                            'download': source, 'activated_at_unix': int(time.time()),
                            'paths': {k:str(v) for k,v in final_paths.items()}, 'checks': checks,
                            'settings_backup': str(backup) if backup else None}
                atomic_write_json(final / 'oculostudio_runtime.json', manifest)
            except Exception:
                if settings_committed:
                    if old_bytes is None: cfg.unlink(missing_ok=True)
                    else:
                        tmp = cfg.with_name(cfg.name+'.restore-'+uuid4().hex)
                        tmp.write_bytes(old_bytes); os.replace(tmp,cfg)
                shutil.rmtree(final, ignore_errors=True)
                raise
            emit(progress, 1.0, 'Ready: complete FFmpeg + ffprobe installed and both functional tests passed')
            return {'ready': True, 'downloaded': not bool(archive_path), 'action': 'installed_verified_pair',
                    'runtime_dir': str(final), 'settings_file': str(cfg), 'checks': checks,
                    'manifest': str(final/'oculostudio_runtime.json')}
    finally:
        _INSTALL_LOCK.release()


def install_runtime_job(store, *, session_id: str, progress=None, cancellation=None) -> dict:
    """Thin adapter for Studio's existing cancellable job queue; never changes project assets."""
    if session_id not in store.snapshot().sessions:
        raise ValidationError('Unknown session for media-install job')
    return install_windows_runtime(progress=progress, cancellation=cancellation)
