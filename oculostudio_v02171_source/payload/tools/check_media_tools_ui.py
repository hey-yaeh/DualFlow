"""Reproducible browser check of real media-tool endpoints and Privacy job.

Synthetic project only. --transport python is a transparent Python HTTP relay
for restricted test sandboxes; no backend responses or encoder results are mocked.
"""
from __future__ import annotations
import argparse, json, os, re, shutil, socket, sys, tempfile, threading, time
import urllib.request, urllib.error
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--transport',choices=['http','python'],default='http')
    parser.add_argument('--output',default=str(ROOT/'docs/OculoStudio_v0_5_3_media_tools.png'))
    args=parser.parse_args()
    output=Path(args.output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    import uvicorn, imageio_ffmpeg
    from playwright.sync_api import sync_playwright
    from oculostudio.demo import create_demo_project
    from oculostudio.server import app,runtime
    from oculostudio.desktop import verify_ui_assets
    report={'transport':args.transport,'page_errors':[], 'console_errors':[], 'checks':[]}
    with tempfile.TemporaryDirectory(prefix='oculo_media_ui_') as temp:
        previous_config=os.environ.get('OCULOSTUDIO_CONFIG_DIR')
        os.environ['OCULOSTUDIO_CONFIG_DIR']=str(Path(temp)/'settings')
        demo=create_demo_project(Path(temp)/'project',run_processing=False)
        store=runtime.open(Path(temp)/'project')
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        url=f'http://127.0.0.1:{port}'
        server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='error'))
        thread=threading.Thread(target=server.run,daemon=True);thread.start()
        try:
            for _ in range(150):
                try:urllib.request.urlopen(url+'/api/health').read();break
                except Exception:time.sleep(.1)
            verify_ui_assets(url)
            with sync_playwright() as pw:
                options={'headless':True,'args':['--no-sandbox']}
                if shutil.which('chromium'):options['executable_path']=shutil.which('chromium')
                browser=pw.chromium.launch(**options)
                page=browser.new_page(viewport={'width':1560,'height':980})
                page.on('pageerror',lambda e:report['page_errors'].append(str(e)))
                page.on('console',lambda m:report['console_errors'].append(m.text) if m.type=='error' else None)
                if args.transport=='http':
                    page.goto(url,wait_until='networkidle')
                    page.click('[data-page="privacy"]')
                else:
                    static=ROOT/'src/oculostudio/static'
                    html=re.sub(r'<script[^>]+src=[^>]+></script>','',(static/'index.html').read_text())
                    html=re.sub(r'<link[^>]+rel="stylesheet"[^>]*>','',html)
                    page.set_content(html);page.add_style_tag(content=(static/'styles.css').read_text())
                    def relay(path,method,headers,body):
                        if not path.startswith('/api/'):raise ValueError('Only local API calls')
                        req=urllib.request.Request(url+path,method=method,headers=headers,
                            data=body.encode() if body is not None else None)
                        try:
                            with urllib.request.urlopen(req,timeout=60) as response:
                                return {'status':response.status,'body':response.read().decode(),'type':response.headers.get('Content-Type','application/json')}
                        except urllib.error.HTTPError as err:
                            return {'status':err.code,'body':err.read().decode(),'type':err.headers.get('Content-Type','application/json')}
                    page.expose_function('relayStudioHTTP',relay)
                    page.evaluate('''() => {window.fetch=async(path,init={})=>{const r=await window.relayStudioHTTP(String(path),init.method||'GET',init.headers||{},init.body||null);return new Response(r.body,{status:r.status,headers:{'Content-Type':r.type}})}}''')
                    # Serve thumbnails/demo SVG through real local server bytes too.
                    def route_asset(route):
                        path=route.request.url.split('studio.local',1)[-1]
                        try:
                            with urllib.request.urlopen(url+path) as resp:
                                route.fulfill(status=resp.status,body=resp.read(),content_type=resp.headers.get('Content-Type','image/jpeg'))
                        except Exception:route.fulfill(status=404,body='not found')
                    page.route('http://studio.local/**',route_asset)
                    page.evaluate("document.head.insertAdjacentHTML('afterbegin','<base href=\"http://studio.local/\">')")
                    for name in ['native_trim.js','asset_import.js','app.js']:
                        page.add_script_tag(content=(static/name).read_text())
                    page.evaluate("appState.page='privacy';document.dispatchEvent(new Event('DOMContentLoaded'))")
                page.wait_for_selector('#check-media-tools')
                page.fill('#privacy-padding','33')
                page.click('#configure-media-tools')
                page.fill('#media-ffmpeg',str(Path(temp)/'not installed.exe'))
                page.click('#media-dialog-check')
                page.wait_for_function("document.querySelector('#media-dialog-result').classList.contains('media-tools-error')")
                assert len(store.snapshot().jobs)==0
                report['checks'].append('invalid executable gives inline error without queueing work')
                exe=imageio_ffmpeg.get_ffmpeg_exe()
                page.fill('#media-ffmpeg',exe)
                page.click('#media-dialog-check')
                page.wait_for_function("document.querySelector('#media-dialog-result').textContent.startsWith('PASS')")
                page.click('#media-dialog-save')
                page.wait_for_selector('#media-dialog-save',state='detached')
                assert page.input_value('#privacy-padding')=='33'
                report['checks'].append('actual encoder test, saved path, active form retained')
                page.click('#check-media-tools')
                page.wait_for_function("document.querySelector('#media-tools-message').textContent.startsWith('PASS')")
                page.select_option('#privacy-native',demo['assets']['native'])
                page.select_option('#privacy-face',demo['assets']['facemapper'])
                page.select_option('#privacy-clip',demo['clip_id'])
                page.click('#run-privacy')
                for _ in range(150):
                    jobs=list(store.snapshot().jobs.values())
                    if jobs and jobs[-1].status in {'completed','failed','cancelled'}:break
                    page.wait_for_timeout(200)
                assert jobs and jobs[-1].status=='completed',[(j.status,j.error) for j in jobs]
                report['checks'].append('real Privacy job completed after UI submission')
                result=jobs[-1].result
                report['output_frame_count']=result['report']['qc']['output_frame_count']
                report['encoder_source']=result['report']['encoder_preflight']['ffmpeg']['source']
                page.evaluate('refreshState({quiet:false})');page.wait_for_timeout(700)
                for width,height in [(1366,768),(1560,980),(1920,1080)]:
                    page.set_viewport_size({'width':width,'height':height});page.wait_for_timeout(150)
                    assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth')
                page.set_viewport_size({'width':1560,'height':980})
                page.locator('#media-tools-panel').evaluate("e => e.scrollIntoView({block: 'start'})")
                page.wait_for_timeout(200)
                page.screenshot(path=str(output))
                page.click('#configure-media-tools')
                page.wait_for_selector('#media-ffmpeg')
                page.screenshot(path=str(output.with_name(output.stem+'_settings.png')))
                browser.close()
            assert not report['page_errors'],report
            report['passed']=True
        finally:
            server.should_exit=True;thread.join(timeout=5);runtime.close()
            if previous_config is None:os.environ.pop('OCULOSTUDIO_CONFIG_DIR',None)
            else:os.environ['OCULOSTUDIO_CONFIG_DIR']=previous_config
    output.with_suffix('.validation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),'utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
