from __future__ import annotations

import os
import socket
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

import uvicorn

from oculostudio.server import app, runtime
from oculostudio.file_dialogs import DesktopFileBridge


def free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def wait_for_server(url: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=0.5) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # pragma: no cover - platform/network timing
            last_error = exc
        time.sleep(0.08)
    raise RuntimeError(f"OculoStudio server did not start within {timeout:.1f}s: {last_error}")


def verify_ui_assets(url: str) -> None:
    """Fail before opening a desktop window if CSS or JavaScript is not served."""
    expected = {
        "/": "text/html",
        "/static/styles.css": "text/css",
        "/static/app.js": "javascript",
        "/static/native_trim.js": "javascript",
        "/static/asset_import.js": "javascript",
    }
    failures: list[str] = []
    for path, content_hint in expected.items():
        try:
            request = urllib.request.Request(f"{url}{path}", headers={"Cache-Control": "no-cache"})
            with urllib.request.urlopen(request, timeout=3.0) as response:
                content_type = response.headers.get("Content-Type", "").lower()
                payload = response.read(256)
                if response.status != 200:
                    failures.append(f"{path}: HTTP {response.status}")
                elif content_hint not in content_type:
                    failures.append(f"{path}: unexpected content type {content_type!r}")
                elif not payload:
                    failures.append(f"{path}: empty response")
        except Exception as exc:
            failures.append(f"{path}: {exc}")
    if failures:
        raise RuntimeError("OculoStudio UI asset check failed: " + "; ".join(failures))


def launch(
    *,
    project_path: str | Path | None = None,
    browser: bool = False,
    host: str = "127.0.0.1",
    port: int | None = None,
    debug: bool = False,
) -> int:
    """Launch the local OculoStudio service and desktop shell.

    A pywebview window is preferred when available. The exact same local UI opens
    in the system browser as a robust fallback; the processing core is identical.
    """
    if project_path:
        runtime.open(project_path)
    port = port or free_port(host)
    config = uvicorn.Config(app, host=host, port=port, log_level="debug" if debug else "warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="oculostudio-server", daemon=True)
    thread.start()
    url = f"http://{host}:{port}"
    wait_for_server(url)
    verify_ui_assets(url)

    used_webview = False
    webview_error: Exception | None = None
    if not browser:
        try:
            import webview  # type: ignore

            bridge = DesktopFileBridge()
            window = webview.create_window(
                "OculoStudio",
                url,
                width=1560,
                height=980,
                min_size=(1100, 720),
                background_color="#454927",
                text_select=True,
                js_api=bridge,
            )
            bridge.bind(window)
            if os.name == "nt":
                # Modern CSS grid, custom properties and ES2020 JavaScript require
                # the Chromium-based WebView2 backend. If it is unavailable, use
                # the browser fallback instead of silently rendering a broken UI.
                webview.start(gui="edgechromium", debug=debug, private_mode=False)
            else:
                webview.start(debug=debug, private_mode=False)
            used_webview = True
        except Exception as exc:  # pragma: no cover - platform specific
            webview_error = exc
    if not used_webview:
        if webview_error is not None:
            print(f"OculoStudio desktop WebView was unavailable: {webview_error}")
            print("Opening the same interface in the default browser instead.")
        webbrowser.open(url)
        print(f"OculoStudio is running at {url}")
        print("Press Ctrl+C in this terminal to stop it.")
        try:
            while thread.is_alive():
                thread.join(0.5)
        except KeyboardInterrupt:
            pass
    server.should_exit = True
    thread.join(timeout=5)
    return 0
