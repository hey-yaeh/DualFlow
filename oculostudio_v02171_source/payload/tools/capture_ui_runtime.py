"""Compatibility entry point for the live Native-reference UI test.

Usage: python tools/capture_ui_runtime.py --output native_trim_ui.png
This now creates synthetic inputs and exercises real preview/export API calls.
It no longer accepts an existing study project or returns canned API responses.
"""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).with_name('check_native_trim_ui.py')), run_name='__main__')
