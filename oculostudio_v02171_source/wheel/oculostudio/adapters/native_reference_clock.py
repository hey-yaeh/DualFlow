"""A native scene MP4 presentation clock paired with its own .time file.

No nominal FPS, Cloud frame count, first gaze, or session/global identity map
is used to infer this relationship. Only the selected scene part is extracted.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import tempfile
import time
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, Iterator
from uuid import uuid4

import numpy as np

from oculostudio.adapters.archive import SourceView
from oculostudio.errors import AssetError, ProcessingError, ValidationError
from oculostudio.util import CancellationToken, atomic_write_json, emit, find_executable, sha256_file

CACHE_VERSION = 'native-reference-clock-1.0'


def round_ratio(num: int, den: int) -> int:
    """Nearest integer, ties away from zero; never cast epoch ns to float."""
    if den <= 0:
        raise ValidationError('Non-positive timestamp denominator')
    sign = -1 if num < 0 else 1
    q, r = divmod(abs(int(num)), int(den))
    return sign * (q + int(2 * r >= den))


def ticks_ns(value: int, time_base: Fraction) -> int:
    return round_ratio(int(value) * time_base.numerator * 1_000_000_000, time_base.denominator)


def source_stamp(path: str | Path) -> dict[str, Any]:
    """Change detection stamp, NOT presented as a cryptographic checksum."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise AssetError(f'Source not found: {p}')
    if p.is_file():
        s = p.stat()
        return {'path': str(p), 'size': s.st_size, 'mtime_ns': s.st_mtime_ns}
    rows = []
    for file in sorted(p.rglob('*')):
        if file.is_file():
            s = file.stat()
            rows.append([file.relative_to(p).as_posix(), s.st_size, s.st_mtime_ns])
    return {'path': str(p), 'files': rows}


def stamp_key(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@contextmanager
def open_member(source: str | Path, name: str) -> Iterator[Any]:
    """Stream a member without unpacking an entire archive."""
    p = Path(source).expanduser().resolve()
    member = PurePosixPath(name.replace('\\', '/'))
    if member.is_absolute() or '..' in member.parts or any(':' in x for x in member.parts):
        raise AssetError(f'Unsafe member path: {name}')
    if p.is_dir():
        target = (p / Path(*member.parts)).resolve()
        if p not in target.parents:
            raise AssetError('Member escapes source directory')
        with target.open('rb') as handle:
            yield handle
    elif zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as archive:
            info = archive.getinfo(name)
            if info.file_size > 100 * 1024**3:
                raise AssetError(f'Member is larger than the 100 GiB safety limit: {name}')
            with archive.open(info) as handle:
                yield handle
    else:
        if p.name != name:
            raise AssetError('Plain file has no requested member')
        with p.open('rb') as handle:
            yield handle


def native_options(source: str | Path) -> list[dict[str, Any]]:
    """Enumerate EVERY scene part; never select the first video and time blindly."""
    rows = []
    with SourceView(source) as view:
        names = view.names()
        if len(names) != len(set(names)):
            raise AssetError('Duplicate ZIP member names; a unique source inventory is required')
        for info_name in sorted(n for n in names if PurePosixPath(n).name.lower() == 'info.json'):
            root = PurePosixPath(info_name).parent
            info = view.read_json(info_name)
            rid = info.get('recording_id') or info.get('recording id')
            if not rid:
                continue
            for video in sorted(names):
                base = PurePosixPath(video)
                if base.parent != root or not re.search(r'neon scene camera.*\.mp4$', base.name, re.I):
                    continue
                time_name = str(base.with_suffix('.time'))
                if time_name not in names:
                    continue
                rows.append({'recording_id': str(rid), 'root': '' if str(root) == '.' else str(root),
                             'info_member': info_name, 'video_member': video, 'time_member': time_name,
                             'recording_start_utc_ns': str(info.get('start_time', ''))})
    if not rows:
        raise AssetError('No Neon scene MP4 with matching .time and recording info was found in NRD')
    return rows


def select_native(source: str | Path, video_member: str | None = None) -> dict[str, Any]:
    options = native_options(source)
    if video_member:
        options = [o for o in options if o['video_member'] == video_member]
    if len(options) != 1:
        raise ValidationError('Select the exact NRD scene video part used for your cut; '
                              f'{len(options)} matching parts found. Start is local to that MP4.')
    return options[0]


def run_probe(path: Path, ffprobe_path: str | None, token: CancellationToken, progress=None) -> dict:
    executable = find_executable('ffprobe', ffprobe_path)
    command = [executable, '-v', 'error', '-threads', '2', '-select_streams', 'v:0',
               '-show_frames', '-show_streams', '-show_entries',
               'frame=pts,best_effort_timestamp,duration,pkt_duration:'
               'stream=time_base,start_pts,duration_ts,nb_frames,width,height',
               '-of', 'json', str(path)]
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    # File streams avoid stdout/stderr pipe deadlocks on large recordings.
    with tempfile.TemporaryFile(mode='w+b') as stdout, tempfile.TemporaryFile(mode='w+b') as stderr:
        proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, creationflags=flags)
        try:
            started = time.monotonic()
            next_notice = started + 5
            while proc.poll() is None:
                token.raise_if_cancelled()
                if time.monotonic() - started > 7200:
                    raise ProcessingError('ffprobe timed out after two hours')
                if time.monotonic() >= next_notice:
                    emit(progress, 0.20, f'Indexing Native frame PTS · {int(time.monotonic()-started)} s elapsed; first run only')
                    next_notice += 5
                time.sleep(0.1)
            stderr.seek(0)
            if proc.returncode:
                raise ProcessingError(f'ffprobe failed ({proc.returncode}): ' +
                                      stderr.read(8000).decode('utf-8', errors='replace'))
            stdout.seek(0)
            return json.load(stdout)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


def parse_probe(payload: dict) -> tuple[np.ndarray, int, dict]:
    streams = payload.get('streams', [])
    if len(streams) != 1:
        raise ValidationError('Expected exactly one selected scene-video stream')
    stream = streams[0]
    base = Fraction(stream['time_base'])
    frames = payload.get('frames', [])
    pts = []
    for row in frames:
        raw = row.get('pts', row.get('best_effort_timestamp'))
        if raw is None or str(raw) == 'N/A':
            raise ValidationError('A decoded scene frame has no presentation timestamp')
        pts.append(int(raw))
    if not pts:
        raise ValidationError('Scene video has no decoded frames')
    if any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValidationError('Scene PTS is not strictly increasing; refusing an ambiguous frame index')
    origin = pts[0]
    media_ns = np.asarray([ticks_ns(x - origin, base) for x in pts], dtype=np.int64)
    last_duration = frames[-1].get('duration') or frames[-1].get('pkt_duration')
    stream_duration = stream.get('duration_ts')
    # Container stream duration is authoritative for the playable end. Audio
    # duration and count/FPS estimates are intentionally not used.
    if stream_duration not in (None, 'N/A'):
        end_tick = int(stream.get('start_pts', origin)) + int(stream_duration)
        end_ns = ticks_ns(end_tick - origin, base)
        end_method = 'video_stream_duration_ts'
    elif last_duration not in (None, 'N/A', '0', 0):
        end_ns = ticks_ns(pts[-1] + int(last_duration) - origin, base)
        end_method = 'last_frame_packet_duration'
    else:
        raise ValidationError('No reliable video end timestamp; refusing to use nominal FPS')
    if end_ns <= int(media_ns[-1]):
        raise ValidationError('Video end time is not after its last frame')
    return media_ns, end_ns, {'time_base': str(base), 'first_pts_tick': origin,
                            'decoded_frames': len(pts), 'duration_method': end_method,
                            'width': int(stream.get('width', 0)), 'height': int(stream.get('height', 0))}


@dataclass
class NativeClock:
    media_ns: np.ndarray
    capture_ns: np.ndarray
    end_media_ns: int
    metadata: dict
    index_path: Path | None = None

    def __post_init__(self):
        if len(self.media_ns) != len(self.capture_ns) or not len(self.media_ns):
            raise ValidationError('Native MP4 decoded-frame count differs from scene .time row count')
        if int(self.media_ns[0]) != 0:
            raise ValidationError('Native media clock must be relative to first displayed frame')
        if np.any(np.diff(self.media_ns) <= 0) or np.any(np.diff(self.capture_ns) <= 0):
            raise ValidationError('Native .time / PTS contains duplicate or reversed timestamps')
        if self.end_media_ns <= int(self.media_ns[-1]):
            raise ValidationError('Invalid native video duration')

    @property
    def end_utc_ns(self) -> int:
        return int(self.capture_ns[-1]) + self.end_media_ns - int(self.media_ns[-1])

    def _map(self, value: int, xs: np.ndarray, ys: np.ndarray, x_end: int, y_end: int) -> int:
        t = int(value)
        if t < int(xs[0]) or t > x_end:
            raise ValidationError('Timestamp lies outside the selected NRD video part')
        i = int(np.searchsorted(xs, t, side='right')) - 1
        x0, y0 = int(xs[i]), int(ys[i])
        if t == x0:
            return y0
        x1, y1 = (int(xs[i+1]), int(ys[i+1])) if i+1 < len(xs) else (x_end, y_end)
        return y0 + round_ratio((t - x0) * (y1 - y0), x1 - x0)

    def utc_at(self, media_ns: int) -> int:
        return self._map(media_ns, self.media_ns, self.capture_ns, self.end_media_ns, self.end_utc_ns)

    def media_at(self, utc_ns: int) -> int:
        return self._map(utc_ns, self.capture_ns, self.media_ns, self.end_utc_ns, self.end_media_ns)

    def overlapping_frames(self, start: int, end: int) -> np.ndarray:
        ends = np.r_[self.media_ns[1:], np.int64(self.end_media_ns)]
        return np.flatnonzero((self.media_ns < end) & (ends > start))

    def nearest_capture(self, utc_ns: int) -> tuple[int, int]:
        j = int(np.searchsorted(self.capture_ns, int(utc_ns)))
        options = [i for i in (j-1, j) if 0 <= i < len(self.capture_ns)]
        i = min(options, key=lambda k: (abs(int(self.capture_ns[k]) - int(utc_ns)), k))
        return i, abs(int(self.capture_ns[i]) - int(utc_ns))

    def face_match(self, utc_ns: int, tolerance_ns: int) -> tuple[int, int, str]:
        i, error = self.nearest_capture(utc_ns)
        # A nearest candidate is only usable within tolerance and if no second
        # nearby frame could satisfy the same tolerance.
        if error > tolerance_ns:
            return -1, error, 'outside_tolerance'
        neighbors = [j for j in (i-1, i+1) if 0 <= j < len(self.capture_ns)]
        if any(abs(int(self.capture_ns[j]) - utc_ns) <= tolerance_ns for j in neighbors):
            return -1, error, 'ambiguous_frame'
        return i, error, 'matched'


def load_clock(index_path: str | Path) -> NativeClock:
    p = Path(index_path)
    with np.load(p, allow_pickle=False) as z:
        media = z['media_ns'].copy()
        capture = z['capture_ns'].copy()
        end = int(z['end_media_ns'])
    metadata = json.loads(p.with_suffix('.json').read_text('utf-8'))
    return NativeClock(media, capture, end, metadata, p)


def build_clock(source: str | Path, cache_root: Path, *, video_member: str | None = None,
                ffprobe_path: str | None = None, cancellation: CancellationToken | None = None,
                progress=None) -> NativeClock:
    token = cancellation or CancellationToken.create()
    selected = select_native(source, video_member)
    stamp = source_stamp(source)
    key = stamp_key({'stamp': stamp, 'selection': selected, 'version': CACHE_VERSION})
    cache_root = Path(cache_root) / 'native_reference' / key
    cache_root.mkdir(parents=True, exist_ok=True)
    final_index = cache_root / 'clock.npz'
    if final_index.exists() and final_index.with_suffix('.json').exists():
        result = load_clock(final_index)
        # A complete cache can be reused without another ffprobe installation.
        if result.metadata.get('source_stamp') == stamp:
            emit(progress, 0.34, 'Reusing verified Native frame-clock index')
            return result
    token.raise_if_cancelled()
    from oculostudio.media_runtime import probe_preflight
    emit(progress, 0.01, 'Checking ffprobe before reading or extracting the scene video')
    check = probe_preflight(ffprobe_path)
    ffprobe_path = check['ffprobe']['path']
    with SourceView(source) as view:
        raw = view.read_bytes(selected['time_member'])
    if not raw or len(raw) % 8:
        raise ValidationError('Native scene .time must contain complete little-endian 64-bit timestamps')
    capture_ns = np.frombuffer(raw, dtype='<u8')
    if np.any(capture_ns > np.iinfo(np.int64).max):
        raise ValidationError('Native timestamps exceed int64 range')
    capture_ns = capture_ns.astype(np.int64)
    # Extract the scene video ONLY. Sensor video, raw gaze and log ZIPs are not
    # unpacked or modified, even for multi-gigabyte Native archives.
    p = Path(source).resolve()
    temp_video = cache_root / ('scene_' + uuid4().hex + '.mp4')
    transient = p.is_file()
    video = (p / selected['video_member']).resolve() if p.is_dir() else temp_video
    video_sha = None
    try:
        if transient:
            with open_member(source, selected['video_member']) as src, video.open('wb') as dst:
                digest = hashlib.sha256()
                total = 0
                while True:
                    token.raise_if_cancelled()
                    block = src.read(4*1024*1024)
                    if not block:
                        break
                    dst.write(block); digest.update(block); total += len(block)
                    emit(progress, 0.10, f'Reading Native scene video only · {total//(1024*1024)} MiB')
                video_sha = digest.hexdigest()
        emit(progress, 0.17, 'Indexing actual MP4 frame PTS; not estimating from FPS')
        payload = run_probe(video, ffprobe_path, token, progress)
        media_ns, end_media_ns, meta = parse_probe(payload)
        clock = NativeClock(media_ns, capture_ns, end_media_ns, {})
        residual = capture_ns - int(capture_ns[0]) - media_ns
        meta.update({'schema_version': CACHE_VERSION, **selected,
                     'source_stamp': stamp, 'video_sha256': video_sha or sha256_file(video),
                     'time_sha256': hashlib.sha256(raw).hexdigest(),
                     'capture_vs_anchored_pts_max_abs_ns': int(np.max(np.abs(residual))),
                     'duration_ns': str(end_media_ns),
                     'first_capture_utc_ns': str(int(capture_ns[0])),
                     'last_capture_utc_ns': str(int(capture_ns[-1])),
                     'utc_clock_model': 'piecewise_integer_PTS_to_capture',
                     'last_frame_tail_model': 'unit_slope_after_last_capture'})
        if source_stamp(source) != stamp:
            raise ValidationError('Native source changed during frame indexing')
        # Cache publication is atomic; cancelled or failed decodes never leave
        # a usable partial index.
        with tempfile.NamedTemporaryFile(dir=cache_root, suffix='.npz', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            np.savez_compressed(tmp_path, media_ns=media_ns, capture_ns=capture_ns,
                                end_media_ns=np.int64(end_media_ns))
            token.raise_if_cancelled()
            atomic_write_json(final_index.with_suffix('.json'), meta)
            os.replace(tmp_path, final_index)
        finally:
            tmp_path.unlink(missing_ok=True)
        clock.metadata = meta; clock.index_path = final_index
        emit(progress, 0.35, f'Native clock ready · {len(media_ns):,} decoded frames')
        return clock
    finally:
        if transient:
            temp_video.unlink(missing_ok=True)
