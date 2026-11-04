"""File selection shared by the desktop bridge and the browser fallback.

Never create a Tk root in a FastAPI worker. In browser mode a short-lived helper
runs Tk on its own main thread. The browser file input does not need Tk at all.
"""
from __future__ import annotations
import json
import subprocess
import sys
import threading
from typing import Any

DEFAULT_FILTERS = [
    ["OculoStudio data", "*.zip *.c3d *.npz *.csv *.tsv *.parquet *.feather *.mp4 *.mov *.mkv *.avi *.m4v *.wav *.flac *.mp3 *.json *.oculocal *.g3dcal"],
    ["ZIP archives", "*.zip"],
    ["All files", "*.*"],
]
_DIALOG_LOCK = threading.Lock()


def normalize_filters(filetypes=None) -> list[tuple[str, tuple[str, ...]]]:
    rows = filetypes or DEFAULT_FILTERS
    normalized = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError("Each file filter must contain a description and a pattern")
        label, pattern = row
        if not isinstance(label, str) or not isinstance(pattern, str):
            raise ValueError("File filter names and patterns must be text")
        if any(ch in label for ch in "()\r\n"):
            raise ValueError("File filter names cannot contain parentheses or newlines")
        masks = tuple(p for p in pattern.replace(';', ' ').split() if p)
        if not masks:
            raise ValueError("A file filter needs at least one pattern")
        if any('/' in p or '\\' in p or '(' in p or ')' in p for p in masks):
            raise ValueError("Invalid file filter pattern")
        normalized.append((label, masks))
    return normalized


def webview_filters(filetypes=None) -> tuple[str, ...]:
    return tuple(f"{label} ({';'.join(masks)})" for label, masks in normalize_filters(filetypes))


def dialog_result(values) -> dict[str, Any]:
    paths = [] if not values else ([values] if isinstance(values, str) else list(values))
    paths = [str(x) for x in paths if x]
    return {"path": paths[0] if paths else None, "paths": paths, "cancelled": not paths}


class DesktopFileBridge:
    """Only file-dialog methods are exposed to the local webview page."""
    def __init__(self):
        self._window = None

    def bind(self, window):
        self._window = window

    def choose_paths(self, kind='file', title='Select files', filetypes=None, multiple=True):
        if kind not in {'file', 'directory', 'save'}:
            raise ValueError('Unknown dialog type')
        if self._window is None:
            raise RuntimeError('Desktop file bridge is not ready')
        import webview
        modern = getattr(webview, 'FileDialog', None)
        if modern is not None:
            dtype = getattr(modern, {'file':'OPEN', 'directory':'FOLDER', 'save':'SAVE'}[kind])
        else:
            dtype = getattr(webview, {'file':'OPEN_DIALOG', 'directory':'FOLDER_DIALOG', 'save':'SAVE_DIALOG'}[kind])
        filters = () if kind == 'directory' else webview_filters(filetypes)
        if not _DIALOG_LOCK.acquire(blocking=False):
            raise RuntimeError('A file picker is already open; close it before opening another')
        try:
            values = self._window.create_file_dialog(
                dtype, allow_multiple=bool(multiple and kind == 'file'), file_types=filters,
            )
            return {**dialog_result(values), 'backend': 'pywebview'}
        finally:
            _DIALOG_LOCK.release()


def isolated_dialog(kind='file', title='Select files', filetypes=None, multiple=False):
    normalize_filters(filetypes)  # Validate in parent; never send None as a Tk option.
    payload = {'kind':kind, 'title':title, 'filetypes':filetypes or DEFAULT_FILTERS, 'multiple':multiple}
    if not _DIALOG_LOCK.acquire(blocking=False):
        raise ValueError('A file picker is already open; close it before opening another')
    try:
        if getattr(sys, 'frozen', False):
            command = [sys.executable, '--oculo-file-dialog']
        else:
            command = [sys.executable, '-m', 'oculostudio.file_dialogs', '--helper']
        result = subprocess.run(command, input=json.dumps(payload), text=True, encoding='utf-8',
                                capture_output=True, timeout=300, check=False)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or 'File picker failed')
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('File picker timed out after five minutes; choose files again') from exc
    finally:
        _DIALOG_LOCK.release()


def run_helper():
    import tkinter as tk
    from tkinter import filedialog
    payload = json.load(sys.stdin)
    root = tk.Tk()
    try:
        root.withdraw()
        root.attributes('-topmost', True)
        root.update()
        options = {'parent':root, 'title':payload.get('title', 'Select files')}
        kind = payload.get('kind', 'file')
        if kind == 'directory':
            value = filedialog.askdirectory(**options)
        else:
            options['filetypes'] = normalize_filters(payload.get('filetypes'))
            if kind == 'save':
                value = filedialog.asksaveasfilename(**options)
            elif payload.get('multiple'):
                value = filedialog.askopenfilenames(**options)
            else:
                value = filedialog.askopenfilename(**options)
        print(json.dumps({**dialog_result(value), 'backend':'isolated_tk'}, ensure_ascii=True))
    finally:
        root.destroy()


if __name__ == '__main__':
    run_helper()
