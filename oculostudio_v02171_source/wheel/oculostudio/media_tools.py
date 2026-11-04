"""Machine-local media tools. No network calls, PATH mutation or shell commands.

FFmpeg (encoding) and ffprobe (frame indexing) are deliberately independent.
The imageio-ffmpeg wheel supplies FFmpeg, NOT ffprobe.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from oculostudio.errors import DependencyError, ValidationError

_LOCK = threading.RLock()
TOOLS = {"ffmpeg", "ffprobe"}


def config_path() -> Path:
    override = os.environ.get("OCULOSTUDIO_CONFIG_DIR")
    if override:
        return Path(override).expanduser().resolve() / "media_tools.json"
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
        return base / "OculoStudio" / "media_tools.json"
    return Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "OculoStudio" / "media_tools.json"


def read_tool_settings() -> dict[str, str]:
    path = config_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("settings must be an object")
        return {k: str(v) for k, v in data.items() if k in TOOLS and v}
    except (ValueError, OSError) as exc:
        raise DependencyError(f"Cannot read media-tool settings {path}: {exc}") from exc


def _names(name: str) -> tuple[str, ...]:
    return (name + ".exe", name) if os.name == "nt" else (name,)


def _in_directory(directory: Path, name: str):
    for base in (directory, directory / "bin"):
        for filename in _names(name):
            yield base / filename


def _explicit_file(name: str, value: str | Path) -> Path:
    text = os.path.expandvars(str(value).strip().strip('"').strip("'"))
    if not text:
        raise DependencyError(f"Empty {name} path")
    candidate = Path(text).expanduser()
    if candidate.is_dir():
        for entry in _in_directory(candidate, name):
            if entry.is_file():
                return entry.resolve()
        raise DependencyError(f"No {name} executable in {candidate} or its bin folder. Select {name}.exe, not a ZIP.")
    if candidate.is_file():
        return candidate.resolve()
    # Bare program names are allowed, but command-line arguments are not.
    located = shutil.which(text)
    if located:
        return Path(located).resolve()
    raise DependencyError(f"{name} path does not exist: {text}. Select the executable, not its download ZIP.")


def _flags() -> int:
    return subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


@lru_cache(maxsize=64)
def _version(name: str, path: str, size: int, mtime_ns: int) -> str:
    try:
        proc = subprocess.run([path, "-version"], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=10,
                              creationflags=_flags())
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise DependencyError(f"Cannot run {name}: {path}. {exc}") from exc
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if proc.returncode or not first.lower().startswith(name + " version "):
        raise DependencyError(f"This is not a working {name} executable: {path}. {first[:350]}")
    return first


@dataclass(frozen=True)
class MediaTool:
    name: str
    path: str
    source: str
    version: str


def _validated(name: str, path: Path, source: str) -> MediaTool:
    path = path.resolve()
    stat = path.stat()
    if not path.is_file():
        raise DependencyError(f"Not an executable file: {path}")
    return MediaTool(name, str(path), source,
                     _version(name, str(path), stat.st_size, stat.st_mtime_ns))


def _portable_directories():
    # Only trusted application locations, never arbitrary source-data folders.
    # A wheel installed into the short Windows runtime cannot infer the extracted
    # application folder from __file__, so launch scripts provide it explicitly.
    roots: list[Path] = []
    app_root = os.environ.get("OCULOSTUDIO_APP_ROOT", "").strip().strip('"').strip("'")
    if app_root:
        roots.append(Path(os.path.expandvars(app_root)).expanduser().resolve())
    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
    if getattr(sys, "_MEIPASS", None):
        roots.append(Path(sys._MEIPASS))
    roots.append(Path(__file__).resolve().parents[2])
    seen: set[str] = set()
    for root in roots:
        key = os.path.normcase(str(root))
        if key in seen:
            continue
        seen.add(key)
        yield root / "bin"
        yield root / "tools" / "ffmpeg"


def _imageio_binary() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError):
        return None


def resolve_media_tool(name: str, explicit: str | Path | None = None,
                       *, sibling_hint: str | Path | None = None) -> MediaTool:
    if name not in TOOLS:
        raise ValueError(f"Not a media tool: {name}")
    if explicit and str(explicit).strip():
        return _validated(name, _explicit_file(name, explicit), "explicit")
    env_names = [f"OCULOSTUDIO_{name.upper()}_PATH", f"OCULOSTUDIO_{name.upper()}"]
    if name == "ffmpeg":
        env_names.append("IMAGEIO_FFMPEG_EXE")
    for variable in env_names:
        if os.environ.get(variable):
            return _validated(name, _explicit_file(name, os.environ[variable]), f"env:{variable}")
    settings = read_tool_settings()
    if name in settings:
        # A user's saved choice must not silently fall back to another version.
        return _validated(name, _explicit_file(name, settings[name]), "saved")
    other = "ffprobe" if name == "ffmpeg" else "ffmpeg"
    other_hint = sibling_hint or settings.get(other) or os.environ.get(f"OCULOSTUDIO_{other.upper()}_PATH")
    candidates: list[tuple[Path, str]] = []
    if other_hint:
        p = Path(os.path.expandvars(str(other_hint).strip().strip('"'))).expanduser()
        folder = p if p.is_dir() else p.parent
        candidates.extend((p, f"beside {other}") for p in _in_directory(folder, name))
    for folder in _portable_directories():
        candidates.extend((p, "application-local") for p in _in_directory(folder, name))
    located = shutil.which(name)
    if located:
        candidates.append((Path(located), "PATH"))
    if name == "ffmpeg":
        packaged = _imageio_binary()
        if packaged:
            candidates.append((Path(packaged), "imageio-ffmpeg"))
    errors, seen = [], set()
    for path, source in candidates:
        if str(path) in seen or not path.is_file():
            continue
        seen.add(str(path))
        try:
            return _validated(name, path, source)
        except (DependencyError, OSError) as exc:
            errors.append(str(exc))
    solution = (
        "Close Studio and run REPAIR_MEDIA_TOOLS_WINDOWS.bat in the application folder, "
        "or use Privacy > Video tools > Configure paths to select ffmpeg.exe. "
        "Installing ffmpeg-python does not install the encoder."
        if name == "ffmpeg" else
        "Run REPAIR_FFPROBE_WINDOWS.bat (full offline pair), or select ffprobe.exe in Video tools / "
        "Clips > Advanced options. The bundled tools/ffmpeg/bin folder contains BOTH tools; imageio-ffmpeg alone does NOT include ffprobe."
    )
    suffix = " Last checks: " + " | ".join(errors[-2:]) if errors else ""
    raise DependencyError(f"Required executable '{name}' was not found or could not run. {solution}{suffix}")


def save_tool_settings(changes: dict[str, str | None]) -> dict[str, str]:
    if set(changes) - TOOLS:
        raise ValidationError("Only ffmpeg and ffprobe paths can be configured")
    with _LOCK:
        settings = read_tool_settings()
        # Validate all choices before changing anything on disk.
        for name, value in changes.items():
            if value is None:
                continue
            if not str(value).strip():
                settings.pop(name, None)
            else:
                tool = _validated(name, _explicit_file(name, value), "explicit")
                settings[name] = tool.path
        path = config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        from oculostudio.util import atomic_write_json
        atomic_write_json(path, settings)
        return settings


def media_tools_status() -> dict[str, Any]:
    result: dict[str, Any] = {"settings_file": str(config_path()), "settings": {}, "tools": {}}
    try:
        result["settings"] = read_tool_settings()
    except DependencyError as exc:
        result["settings_error"] = str(exc)
    for name in ("ffmpeg", "ffprobe"):
        try:
            tool = resolve_media_tool(name)
            result["tools"][name] = {"available": True, **asdict(tool)}
        except (DependencyError, OSError) as exc:
            result["tools"][name] = {"available": False, "name": name, "path": None, "error": str(exc)}
    return result


@lru_cache(maxsize=32)
def _encode_smoke(path: str, size: int, mtime_ns: int, preserve_audio: bool,
                  preset: str, crf: int) -> dict[str, Any]:
    # Probe the same H.264/MP4 (+ AAC) encoders before a long face-mask run.
    with tempfile.TemporaryDirectory(prefix="oculo_encoder_check_") as temporary:
        output = Path(temporary) / "check.mp4"
        command = [path, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
                   "-i", "color=c=black:s=32x32:r=30:d=0.1"]
        if preserve_audio:
            command += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono"]
        command += ["-t", "0.1", "-map", "0:v:0", "-c:v", "libx264", "-threads", "1",
                    "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", "-fps_mode", "passthrough"]
        command += (["-map", "1:a:0", "-c:a", "aac", "-b:a", "192k", "-shortest"]
                    if preserve_audio else ["-an"])
        command += ["-movflags", "+faststart", str(output)]
        try:
            proc = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=20, creationflags=_flags())
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DependencyError(f"FFmpeg encoder test failed before video processing: {exc}") from exc
        if proc.returncode or not output.exists() or output.stat().st_size == 0:
            raise DependencyError("FFmpeg was found but H.264/MP4" + (" + AAC" if preserve_audio else "") +
                                  " encoding failed. Choose a working FFmpeg build. " + proc.stderr[-1600:])
        return {"status": "passed", "video_encoder": "libx264", "audio_encoder": "aac" if preserve_audio else None,
                "container": "mp4", "test_bytes": output.stat().st_size}


def privacy_encoder_preflight(ffmpeg_path: str | None = None, *, preserve_audio: bool = True,
                              preset: str = "fast", crf: int = 20,
                              ffprobe_path: str | None = None) -> dict[str, Any]:
    tool = resolve_media_tool("ffmpeg", ffmpeg_path, sibling_hint=ffprobe_path)
    stat = Path(tool.path).stat()
    smoke = _encode_smoke(tool.path, stat.st_size, stat.st_mtime_ns, preserve_audio, preset, crf)
    return {"ffmpeg": asdict(tool), "encoding_test": dict(smoke)}
