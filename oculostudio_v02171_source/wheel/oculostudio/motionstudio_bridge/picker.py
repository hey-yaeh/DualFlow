from __future__ import annotations
import json
import os
import subprocess
import sys


def native_pick(kind: str = "scene", multiple: bool = False) -> list[str]:
    """Open a Windows-native picker in a dedicated subprocess main thread."""
    if os.name != "nt":
        raise RuntimeError("Native picker is available on Windows; paste the full path on other systems")
    script = r'''
import json, sys, tkinter as tk
from tkinter import filedialog
kind = sys.argv[1]
multiple = sys.argv[2] == "1"
root = tk.Tk()
root.withdraw()
try:
    root.attributes("-topmost", True)
except Exception:
    pass
filters = {
    "scene": [("MotionStudio scene manifest", "*_oculostudio_scene.json"), ("JSON", "*.json")],
    "catalog": [("OculoStudio scene catalog", "motionstudio_scene_catalog.json"), ("JSON", "*.json")],
    "rays": [("Gaze rays", "*.csv *.parquet"), ("CSV", "*.csv"), ("Parquet", "*.parquet")],
}.get(kind, [("All files", "*.*")])
if kind == "folder":
    value = filedialog.askdirectory(parent=root)
elif multiple:
    value = filedialog.askopenfilenames(parent=root, filetypes=filters)
else:
    value = filedialog.askopenfilename(parent=root, filetypes=filters)
root.destroy()
if isinstance(value, (tuple, list)):
    print(json.dumps(list(value)))
else:
    print(json.dumps([value] if value else []))
'''
    proc = subprocess.run(
        [sys.executable, "-c", script, kind, "1" if multiple else "0"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "File picker failed")
    return json.loads(proc.stdout.strip() or "[]")
