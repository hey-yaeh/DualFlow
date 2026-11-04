from __future__ import annotations
import argparse, socket, threading, time, urllib.request, tempfile
from pathlib import Path
import uvicorn
from oculostudio.server import create_app
from oculostudio.workflows import ProjectController

def free_port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]
def wait(url):
    for _ in range(200):
        try:
            if urllib.request.urlopen(url+'/api/health',timeout=1).status==200:return
        except Exception:time.sleep(.05)
    raise RuntimeError('server did not start')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default='docs/OculoStudio_v0_12_0_MotionStudio_Bridge.png');a=ap.parse_args()
    with tempfile.TemporaryDirectory(prefix='oculo_msbridge_ui_') as td:
        c=ProjectController();c.create(str(Path(td)/'project'),'MotionStudio Bridge Demo','S001')
        port=free_port();url=f'http://127.0.0.1:{port}';server=uvicorn.Server(uvicorn.Config(create_app(controller=c),host='127.0.0.1',port=port,log_level='error'));th=threading.Thread(target=server.run,daemon=True);th.start();wait(url)
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser=p.chromium.launch(headless=True)
                page=browser.new_page(viewport={'width':1720,'height':1040},device_scale_factor=1)
                errors=[];page.on('pageerror',lambda e:errors.append(str(e)));page.goto(url,wait_until='networkidle');page.get_by_text('MotionStudio',exact=True).click();page.wait_for_timeout(500);page.screenshot(path=a.output,full_page=False);assert not errors,errors;browser.close()
        finally:server.should_exit=True;th.join(timeout=5)
    return 0
if __name__=='__main__':raise SystemExit(main())
