"""Chromium UI import regression against the LIVE local HTTP server.

Uses disposable projects and synthetic ZIPs by default. --real-inputs additionally
imports the supplied I05 NRD/TSD/FM (never included in release). Native OS dialog
is contract-tested separately; injected bridge here tests frontend dispatch only.
"""
from __future__ import annotations
import argparse,hashlib,json,socket,sys,tempfile,threading,time,zipfile,base64,re,shutil,os
import urllib.request,urllib.error
from pathlib import Path
from urllib.request import urlopen
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from oculostudio.server import app,runtime
from oculostudio.desktop import verify_ui_assets
from playwright.sync_api import sync_playwright
import uvicorn

def make_zip(path,mode='tsd'):
    with zipfile.ZipFile(path,'w') as z:
        if mode=='tsd':
            z.writestr('gaze.csv','recording id,timestamp [ns],gaze x [px],gaze y [px]\nrec-001,1700000000000000000,1,2\n')
            z.writestr('world_timestamps.csv','timestamp [ns]\n1700000000000000000\n')
            z.writestr('info.json','{"recording_id":"rec-001"}')
        else:
            z.writestr('face_detections.csv','recording id,timestamp [ns],x,y,width,height\nrec-001,1700000000000000000,1,2,3,4\n')
            z.writestr('gaze_on_face.csv','recording id,timestamp [ns],gaze on face\nrec-001,1700000000000000000,true\n')
            z.writestr('sections.csv','recording id,section id\nrec-001,section-001\n')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(ROOT/'docs/OculoStudio_v0_5_1_import_runtime.png'))
    parser.add_argument('--real-inputs',action='store_true')
    parser.add_argument('--real-input-root',default=os.environ.get('OCULOSTUDIO_REAL_INPUT_ROOT'),help='Directory containing the optional real NRD/TSD/FM validation archives')
    parser.add_argument('--chromium',default=None,help='Optional Chromium executable; otherwise use PATH or Playwright default')
    parser.add_argument('--transport',choices=['http','python'],default='http')
    args=parser.parse_args()
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    report={'transport':('live local HTTP in Chromium' if args.transport=='http' else 'Packaged UI in Chromium; fetch proxied through Python to a LIVE local HTTP server because this environment blocks Chromium localhost navigation. No fabricated import responses; real upload bytes and processing.' ),'page_errors':[], 'expected_errors':[], 'checks':[], 'real_input_checks':[], 'windows_native_dialog':'not executed here; Python bridge unit tests only'}
    runtime.uploads.cleanup();runtime.close()
    with tempfile.TemporaryDirectory(prefix='oculo051_ui_') as temp:
        base=Path(temp);tsd=base/'眼动 TSD(1).zip';fm=base/'FM multiple.zip';make_zip(tsd);make_zip(fm,'fm')
        bad=base/'corrupt.zip';bad.write_bytes(b'not a zip')
        nrd_fake=base/'Native ZIP(2).zip';make_zip(nrd_fake)
        large=base/'large synthetic.csv'
        with large.open('wb') as f:
            f.write(b'timestamp_ns,value\n')
            f.write(b'0,1\n'* (3*1024*1024)) # >8 MiB; cancellation after first chunk
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='error'))
        th=threading.Thread(target=server.run,daemon=True);th.start();url=f'http://127.0.0.1:{port}'
        for _ in range(150):
            try:urlopen(url+'/api/health').read();break
            except Exception:time.sleep(.1)
        verify_ui_assets(url)
        try:
            with sync_playwright() as pw:
                executable=args.chromium or shutil.which('chromium') or shutil.which('google-chrome')
                options={'headless':True,'args':['--no-sandbox']}
                if executable:options['executable_path']=executable
                browser=pw.chromium.launch(**options)
                page=browser.new_page(viewport={'width':1560,'height':980})
                page.on('pageerror',lambda err:report['page_errors'].append(str(err)))
                transport_state={'delay':False}
                if args.transport=='http':
                    page.goto(url,wait_until='networkidle')
                else:
                    def transport(path,method,headers,body,encoded):
                        if not str(path).startswith('/api/'):
                            raise ValueError('Only local API requests are proxied')
                        if transport_state['delay'] and method=='PUT':time.sleep(.25)
                        data=base64.b64decode(body) if encoded else (body.encode() if body is not None else None)
                        request=urllib.request.Request(url+path,data=data,headers=headers,method=method)
                        try:
                            with urllib.request.urlopen(request,timeout=60) as resp:
                                return {'status':resp.status,'body':resp.read().decode(),'type':resp.headers.get('Content-Type','application/json')}
                        except urllib.error.HTTPError as err:
                            return {'status':err.code,'body':err.read().decode(),'type':err.headers.get('Content-Type','application/json')}
                    static=ROOT/'src/oculostudio/static'
                    html=re.sub(r'<script[^>]+src=[^>]+></script>','',(static/'index.html').read_text())
                    html=re.sub(r'<link[^>]+rel="stylesheet"[^>]*>','',html)
                    page.set_content(html)
                    page.add_style_tag(content=(static/'styles.css').read_text())
                    page.expose_function('localImportTransport',transport)
                    page.evaluate(r'''() => {window.fetch=async(path,init={})=>{
                      let body=init.body??null,encoded=false;
                      if(body instanceof Blob){
                        const a=new Uint8Array(await body.arrayBuffer());let b='';
                        for(let i=0;i<a.length;i+=32768)b+=String.fromCharCode(...a.subarray(i,i+32768));
                        body=btoa(b);encoded=true;
                      }
                      if(init.signal?.aborted)throw new DOMException('Aborted','AbortError');
                      const work=window.localImportTransport(String(path),init.method||'GET',init.headers||{},body,encoded);
                      const result=init.signal?await Promise.race([work,new Promise((_,rej)=>init.signal.addEventListener('abort',()=>rej(new DOMException('Aborted','AbortError')),{once:true}))]):await work;
                      return new Response(result.body,{status:result.status,headers:{'content-type':result.type}});
                    }}''')
                    for name in ['native_trim.js','asset_import.js','app.js']:page.add_script_tag(content=(static/name).read_text())
                    page.evaluate("document.dispatchEvent(new Event('DOMContentLoaded'))")
                original_page_url=page.url
                page.click('#import-asset-sidebar')
                page.wait_for_selector('#import-create-project')
                report['checks'].append('Import without project opens a create/open prompt')
                page.click('#import-create-project')
                page.fill('#create-root',str(base/'test project 中文'))
                page.fill('#create-name','OculoStudio import regression')
                page.click('#create-submit')
                page.wait_for_selector('#import-choose-files')
                report['checks'].append('Creating project resumes import instead of losing selection context')
                with page.expect_file_chooser() as picker:
                    page.click('#import-choose-files')
                picker.value.set_files([str(tsd),str(fm)])
                page.wait_for_function("document.querySelectorAll('.import-file-row').length===2")
                page.click('#import-run')
                page.wait_for_function("document.querySelectorAll('.import-file-row[data-status=done]').length===2",timeout=60000)
                page.wait_for_function('assets().length===2');count=page.evaluate('assets().length');assert count==2,count
                assert page.evaluate("assets().every(a=>a.metadata.import.storage_mode==='project_copy')")
                report['checks'].append('Browser Choose ZIP/files opens a real HTML file picker; two ZIPs copied and registered')
                # Drop uses actual File handles from a browser input, not fabricated file paths.
                page.locator('#oculo-import-file-input').set_input_files(str(nrd_fake))
                page.wait_for_function("document.querySelectorAll('.import-file-row').length===3")
                # Remove newly queued row, then dispatch a drop containing that same real File.
                page.locator('.import-file-row[data-status=ready] .import-remove').click()
                page.evaluate('''() => {const dt=new DataTransfer();for(const f of document.querySelector('#oculo-import-file-input').files)dt.items.add(f);
                  document.querySelector('#import-drop-zone').dispatchEvent(new DragEvent('dragover',{bubbles:true,cancelable:true,dataTransfer:dt}));
                  const drop=new DragEvent('drop',{bubbles:true,cancelable:true,dataTransfer:dt});
                  document.querySelector('#import-drop-zone').dispatchEvent(drop); if(!drop.defaultPrevented)throw new Error('drop navigation was not prevented');}''')
                page.wait_for_function("document.querySelectorAll('.import-file-row').length===3")
                page.click('#import-run')
                page.wait_for_function("document.querySelectorAll('.import-file-row[data-status=done]').length===3",timeout=60000)
                page.wait_for_function('assets().length===3');assert page.evaluate('assets().length')==3
                assert page.url==original_page_url
                report['checks'].append('File dragover/drop prevents navigation and actually copies/registers a ZIP')
                # native bridge frontend contract uses reference path backend, not browser copying
                native=base/'Native local reference.zip';make_zip(native)
                page.evaluate('path=>{window.pywebview={api:{choose_paths:async(kind,title,filters,multi)=>{window.pickerContract={kind,filters,multi};return {paths:[path],path,cancelled:false}}}}}',str(native))
                page.click('#import-choose-files');page.wait_for_function("document.querySelectorAll('.import-file-row').length===4")
                page.click('#import-run');page.wait_for_function("document.querySelectorAll('.import-file-row[data-status=done]').length===4",timeout=60000)
                assert page.evaluate("pickerContract.multi===true && pickerContract.filters.some(r=>r[1]==='*.zip')")
                a=next(a for a in runtime.require_store().project.sessions[runtime.require_store().project.selected_session_id].assets.values() if a.source_path==str(native))
                assert a.metadata['import']['storage_mode']=='local_reference'
                report['checks'].append('Desktop bridge contract dispatches multi-file ZIP picker, references original path without copying')
                page.evaluate("window.pywebview.api.choose_paths=async()=>({paths:[],path:null,cancelled:true})")
                page.click('#import-choose-files');page.wait_for_function("document.querySelector('#import-feedback').textContent.includes('Selection cancelled')")
                assert page.locator('.import-file-row').count()==4
                page.evaluate('delete window.pywebview')
                report['checks'].append('Cancelling file picker produces feedback and creates no asset')
                # Persistent corrupt-ZIP failure under its own row.
                with page.expect_file_chooser() as chooser:page.click('#import-browser-files')
                chooser.value.set_files(str(bad));page.click('#import-run')
                page.wait_for_selector('.import-file-row[data-status=failed]',timeout=60000)
                assert 'ZIP' in page.locator('.import-file-row[data-status=failed]').inner_text()
                assert page.evaluate('assets().length')==4
                page.locator('.import-file-row[data-status=failed] .import-remove').click()
                report['checks'].append('Corrupt ZIP fails visibly; neither original nor registry is modified')
                # Cancel before all chunks sent (slowed real local request, not mocked response).
                def delayed(route):
                    time.sleep(.25);route.continue_()
                if args.transport=='http':page.route('**/api/imports/uploads/*?offset=*',delayed)
                else:transport_state['delay']=True
                with page.expect_file_chooser() as chooser:page.click('#import-browser-files')
                chooser.value.set_files(str(large));page.click('#import-run')
                page.wait_for_function("document.querySelector('.import-file-row[data-status=copying]')!==null")
                page.click('#import-cancel')
                page.wait_for_function("!document.querySelector('#import-cancel').disabled===false",timeout=60000)
                page.wait_for_selector('.import-file-row[data-status=cancelled]',timeout=60000)
                assert page.evaluate('assets().length')==4
                assert not runtime.uploads.has_pending()
                if args.transport=='http':page.unroute('**/api/imports/uploads/*?offset=*')
                else:transport_state['delay']=False
                report['checks'].append('Cancelling a multi-chunk upload removes staging and registers no partial file')
                page.click('#import-clear-finished')
                # screenshot settled picker with real data listed only synthetic user inputs
                with page.expect_file_chooser() as chooser:page.click('#import-choose-files')
                chooser.value.set_files([str(tsd),str(fm)])
                page.click('#import-run')
                page.wait_for_function("document.querySelectorAll('.import-file-row[data-status=done]').length===2",timeout=60000)
                page.wait_for_timeout(3800)
                for width,height in [(1366,768),(1560,980),(1920,1080)]:
                    page.set_viewport_size({'width':width,'height':height});page.wait_for_timeout(150)
                    assert page.evaluate('document.documentElement.scrollWidth')<=width
                    box=page.locator('#import-run').bounding_box()
                    page.locator('#import-run').scroll_into_view_if_needed()
                    if (width,height)==(1560,980):page.screenshot(path=str(output))
                report['checks'].append('Import dialog usable at 1366x768, 1560x980, 1920x1080, no horizontal overflow')
                page.click('.modal-close');page.click('[data-page=clips]')
                page.wait_for_selector('#native-trim-start')
                report['checks'].append('Clips → NRD video reference form still available after imports')
                # Real source test, full browser upload of NRD + TSD + FM.
                if args.real_inputs:
                    if not args.real_input_root:
                        raise ValueError('--real-inputs requires --real-input-root or OCULOSTUDIO_REAL_INPUT_ROOT')
                    real_root=Path(args.real_input_root).expanduser().resolve()
                    paths=[real_root/'S001_I05_R01_NRD_untrimmed.zip',real_root/'S001_I05_R01_TSD_untrimmed.zip',real_root/'S001_R01_FM_untrimmed(1).zip']
                    missing=[str(path) for path in paths if not path.is_file()]
                    if missing:
                        raise FileNotFoundError('Missing optional real validation inputs: '+', '.join(missing))
                    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
                    page.click('#import-asset-sidebar');page.click('#import-clear-finished')
                    with page.expect_file_chooser() as chooser:page.click('#import-browser-files')
                    chooser.value.set_files([str(p) for p in paths]);page.click('#import-run')
                    page.wait_for_function("document.querySelectorAll('.import-file-row[data-status=done]').length===3",timeout=240000)
                    assert page.locator('.import-file-row[data-status=failed]').count()==0
                    store=runtime.require_store();sid=store.project.selected_session_id
                    imported=[a for a in store.project.sessions[sid].assets.values() if Path(a.source_path).name in hashes]
                    assert len(imported)==3
                    for a in imported:
                        expected=hashes[Path(a.source_path).name]
                        assert a.source_sha256==expected
                        assert hashlib.sha256(Path(a.source_path).read_bytes()).hexdigest()==expected
                        report['real_input_checks'].append({'file':Path(a.source_path).name,'kind':a.kind,'size':a.size_bytes,'sha256':expected,'byte_identical':True})
                    assert all(hashlib.sha256(p.read_bytes()).hexdigest()==hashes[p.name] for p in paths)
                    page.click('.modal-close');page.click('[data-page=clips]')
                    opts={key:page.locator(key).inner_text() for key in ['#native-trim-nrd','#native-trim-tsd','#native-trim-fm']}
                    assert 'NRD_untrimmed' in opts['#native-trim-nrd']
                    assert 'TSD_untrimmed' in opts['#native-trim-tsd']
                    assert 'FM_untrimmed' in opts['#native-trim-fm']
                    report['checks'].append('Real I05 NRD (294 MB), TSD and six-recording FM imported through browser picker into correct clip selectors; SHA-256 unchanged')
                browser.close()
        finally:
            server.should_exit=True;th.join(5);runtime.uploads.cleanup()
    report['ok']=not report['page_errors']
    output.with_suffix('.validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    assert report['ok'],report

if __name__=='__main__':main()
