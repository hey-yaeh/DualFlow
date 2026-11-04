"""Render and validate the packaged v0.8 ETM assignment regression UI with a live backend."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import socket
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tools'))


def _free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));return int(sock.getsockname()[1])


def _wait(url):
    deadline=time.time()+30
    while time.time()<deadline:
        try:
            if urllib.request.urlopen(url+'/api/health',timeout=1).status==200:return
        except Exception:time.sleep(.1)
    raise RuntimeError('test server did not start')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--transport',choices=['http','python'],default='python')
    parser.add_argument('--output',default=str(ROOT/'docs/OculoStudio_v0_8_0_etm_assignment.png'))
    args=parser.parse_args()
    import uvicorn
    from playwright.sync_api import sync_playwright
    from create_etm_assignment_demo import build
    from oculostudio.desktop import verify_ui_assets
    from oculostudio.server import app,runtime

    report={'release':'0.8.0','transport':args.transport,'page_errors':[],'console_errors':[],'checks':[]}
    output=Path(args.output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='oculo_v070_ui_') as temp:
        project=Path(temp)/'project';build(project,overwrite=True);runtime.open(project)
        port=_free_port();url=f'http://127.0.0.1:{port}'
        server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='error'))
        thread=threading.Thread(target=server.run,daemon=True);thread.start()
        try:
            _wait(url);verify_ui_assets(url)
            with sync_playwright() as pw:
                opts={'headless':True,'args':['--no-sandbox']}
                exe=shutil.which('chromium') or shutil.which('google-chrome')
                if exe:opts['executable_path']=exe
                browser=pw.chromium.launch(**opts);page=browser.new_page(viewport={'width':1680,'height':1050})
                page.on('pageerror',lambda e:report['page_errors'].append(str(e)))
                page.on('console',lambda m:report['console_errors'].append(m.text) if m.type=='error' else None)
                if args.transport=='http':
                    page.goto(url,wait_until='networkidle')
                else:
                    static=ROOT/'src/oculostudio/static'
                    html_text=re.sub(r'<script[^>]+src=[^>]+></script>','',(static/'index.html').read_text('utf-8'))
                    html_text=re.sub(r'<link[^>]+rel="stylesheet"[^>]*>','',html_text)
                    page.set_content(html_text);page.add_style_tag(content=(static/'styles.css').read_text('utf-8'))
                    def relay(path,method,headers,body):
                        request=urllib.request.Request(url+path,method=method,headers=headers or {},data=body.encode() if body is not None else None)
                        try:
                            with urllib.request.urlopen(request,timeout=120) as response:return {'status':response.status,'body':response.read().decode(),'type':response.headers.get('Content-Type','application/json')}
                        except urllib.error.HTTPError as error:return {'status':error.code,'body':error.read().decode(),'type':error.headers.get('Content-Type','application/json')}
                    page.expose_function('relayStudioHTTP',relay)
                    page.evaluate("""() => { window.fetch=async(path,init={})=>{const r=await window.relayStudioHTTP(String(path),init.method||'GET',init.headers||{},init.body||null);return new Response(r.body,{status:r.status,headers:{'Content-Type':r.type}})}}""")
                    for name in ('native_trim.js','asset_import.js','app.js'):page.add_script_tag(content=(static/name).read_text('utf-8'))
                    page.evaluate("appState.page='gaze3d'; document.dispatchEvent(new Event('DOMContentLoaded'))")
                page.wait_for_selector('[data-page="gaze3d"]');page.click('[data-page="gaze3d"]')
                page.wait_for_selector('#run-etm-assignment')
                page.wait_for_selector('.gaze3d-assignment-grid')
                text=page.locator('.gaze3d-assignment-grid').inner_text()
                assert 'R01 → ETM_B' in text and 'R02 → ETM_A' in text,text
                assert page.input_value('#etm-body-r01') and page.input_value('#etm-body-r02')
                assert page.input_value('#etm-candidate-a') and page.input_value('#etm-candidate-b')
                assert page.locator('text=resolved').count()>=1
                report['checks'] += ['resolved assignment rendered','filename-based input suggestions rendered','Gaze 3D assignment controls available']
                report['document_scroll_width']=page.evaluate('document.documentElement.scrollWidth')
                assert report['document_scroll_width']<=1680
                page.screenshot(path=str(output));browser.close()
            unexpected=[x for x in report['console_errors'] if 'favicon' not in x.lower()]
            assert not report['page_errors'],report['page_errors'];assert not unexpected,unexpected
            report['ok']=True
        finally:
            server.should_exit=True;thread.join(timeout=10);runtime.close()
    output.with_suffix('.validation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),'utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False));return 0


if __name__=='__main__':raise SystemExit(main())
