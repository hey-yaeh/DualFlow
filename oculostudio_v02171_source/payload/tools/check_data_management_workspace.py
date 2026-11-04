from __future__ import annotations
import argparse, json, re, shutil, socket, tempfile, threading, time, urllib.error, urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def free_port():
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));return int(sock.getsockname()[1])
def wait(url):
    deadline=time.time()+40
    while time.time()<deadline:
        try:
            with urllib.request.urlopen(url+'/api/health',timeout=1) as response:
                if response.status==200:return
        except Exception:time.sleep(.1)
    raise RuntimeError('OculoStudio test server did not start')
def request_json(url,path,method='GET',body=None,data=None,headers=None):
    if body is not None:
        data=json.dumps(body).encode();headers={'Content-Type':'application/json',**(headers or {})}
    request=urllib.request.Request(url+path,method=method,data=data,headers=headers or {})
    with urllib.request.urlopen(request,timeout=180) as response:return json.loads(response.read().decode())
def poll(url,job_id):
    deadline=time.time()+60
    while time.time()<deadline:
        job=request_json(url,f'/api/imports/jobs/{job_id}')['job']
        if job['status'] in {'completed','failed','cancelled'}:return job
        time.sleep(.2)
    raise TimeoutError(job_id)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-output',default=str(ROOT/'docs'/'OculoStudio_v0_21_70_DataManagementWorkspaces.png'))
    parser.add_argument('--dialog-output',default=str(ROOT/'docs'/'OculoStudio_v0_21_70_CalibrationImportDialog.png'))
    parser.add_argument('--calibration-output',default=str(ROOT/'docs'/'OculoStudio_v0_21_70_CalibrationOperationsOnly.png'))
    parser.add_argument('--json',default=str(ROOT/'docs'/'OculoStudio_v0_21_70_DataManagementValidation.json'))
    args=parser.parse_args();outputs=[Path(args.data_output),Path(args.dialog_output),Path(args.calibration_output),Path(args.json)]
    for path in outputs:path.parent.mkdir(parents=True,exist_ok=True)

    import uvicorn, oculostudio
    from oculostudio.server import app, runtime
    from oculostudio.services.asset_import import import_local_asset
    from playwright.sync_api import sync_playwright
    report={'release':'OculoStudio v0.21.71 — Scope-Separated Data Management','application_version':oculostudio.__version__,'checks':{},'page_errors':[],'console_errors':[]}
    checks=report['checks'];checks['installed_version_0_21_70']=oculostudio.__version__=='0.21.71'
    with tempfile.TemporaryDirectory(prefix='oculo_02170_data_') as temp:
        temp_root=Path(temp);parent=temp_root/'projects';parent.mkdir();board=temp_root/'calibration_board_S004-S007.json';interaction_i01=temp_root/'S004_I01_R01_signal.csv';interaction_i02=temp_root/'S004_I02_R01_signal.csv';interaction_i01.write_text('time,value\n0,1\n','utf-8');interaction_i02.write_text('time,value\n0,2\n','utf-8');board.write_text(json.dumps({'tag_corners_mm':{'0':[[0,0,0],[10,0,0],[10,10,0],[0,10,0]]}}),'utf-8')
        port=free_port();url=f'http://127.0.0.1:{port}'
        server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='error'));thread=threading.Thread(target=server.run,daemon=True);thread.start()
        try:
            wait(url);static=Path(oculostudio.__file__).resolve().parent/'static'
            with sync_playwright() as pw:
                options={'headless':True,'args':['--no-sandbox']};executable=shutil.which('chromium') or shutil.which('google-chrome')
                if executable:options['executable_path']=executable
                browser=pw.chromium.launch(**options);page=browser.new_page(viewport={'width':1760,'height':1100});page.set_default_timeout(30000)
                page.on('pageerror',lambda error:report['page_errors'].append(str(error)))
                page.on('console',lambda message:report['console_errors'].append(message.text) if message.type=='error' else None)
                html=re.sub(r'<script[^>]+src=[^>]+></script>','',(static/'index.html').read_text('utf-8'));html=re.sub(r'<link[^>]+rel="stylesheet"[^>]*>','',html);page.set_content(html)
                for css in ('styles.css','motionstudio_bridge.css','interaction_workspace.css'):page.add_style_tag(content=(static/css).read_text('utf-8'))
                def relay(path,method,headers,body):
                    request=urllib.request.Request(url+path,method=method,headers=headers or {},data=body.encode() if isinstance(body,str) else body)
                    try:
                        with urllib.request.urlopen(request,timeout=180) as response:return {'status':response.status,'body':response.read().decode(),'type':response.headers.get('Content-Type','application/json')}
                    except urllib.error.HTTPError as error:return {'status':error.code,'body':error.read().decode(),'type':error.headers.get('Content-Type','application/json')}
                page.expose_function('relayStudioHTTP',relay);page.evaluate("""() => {window.fetch=async(path,init={})=>{const r=await window.relayStudioHTTP(String(path),init.method||'GET',init.headers||{},typeof init.body==='string'?init.body:null);return new Response(r.body,{status:r.status,headers:{'Content-Type':r.type}})}}""")
                for name in ('native_trim.js','asset_import.js','aoi_review.js','gaze3d_preview.js','calibration_preview.js','app.js','interaction_workspace.js','motionstudio_bridge.js'):page.add_script_tag(content=(static/name).read_text('utf-8'))
                page.evaluate("document.dispatchEvent(new Event('DOMContentLoaded'))")
                page.wait_for_selector('#welcome-create');page.click('#welcome-create');page.fill('#create-parent',str(parent));page.fill('#create-folder','OCUMO_S004_S007');page.fill('#create-name','OCUMO S004–S007');page.click('#create-next');page.fill('#create-sessions-spec','S004-S007');page.fill('#create-interactions-spec','I01-I02');page.click('#create-next');page.click('#create-submit');page.wait_for_selector('.wh-session-card',timeout=30000)
                store=runtime.require_store();sid=store.snapshot().selected_session_id
                import_local_asset(store,session_id=sid,path=interaction_i01,checksum=False)
                import_local_asset(store,session_id=sid,path=interaction_i02,checksum=False)
                page.evaluate("refreshState({quiet:false,fullRender:true})");page.wait_for_timeout(350)
                page.evaluate("navigate('data')");page.wait_for_selector('#data-calibration-workspace')
                checks['two_scope_workspaces']=page.locator('.data-scope-workspace').count()==2
                checks['old_global_dropzone_removed']=page.locator('#data-drop-zone').count()==0
                checks['calibration_dropzone_visible']=page.locator('#calibration-data-drop-zone').is_visible()
                checks['interaction_dropzone_visible']=page.locator('#interaction-data-drop-zone').is_visible()
                checks['both_upload_targets_explicit']=all('upload target' in page.locator(selector).inner_text().lower() for selector in ('#data-calibration-workspace','#data-interaction-workspace'))
                checks['calibration_has_inventory_and_inspector']=all(token in page.locator('#data-calibration-workspace').inner_text() for token in ('Registered Calibration sources','Calibration source inspector'))
                checks['interaction_has_inventory_and_inspector']=all(token in page.locator('#data-interaction-workspace').inner_text() for token in ('Registered Interaction assets','Interaction asset inspector'))
                interaction_text=page.locator('.data-table-card').inner_text();checks['current_interaction_filter_hides_other_i_codes']='S004_I01_R01_signal.csv' in interaction_text and 'S004_I02_R01_signal.csv' not in interaction_text
                page.screenshot(path=str(outputs[0]),full_page=True)
                page.click('#data-calibration-choose-files');page.wait_for_selector('.calibration-import-dialog')
                checks['calibration_import_stays_on_data']=page.evaluate('appState.page')=='data'
                checks['calibration_modal_project_scope']='Project-level scope' in page.locator('#calimport-scope').inner_text()
                page.set_input_files('#calimport-browser-input',str(board));page.wait_for_selector('.calibration-import-table tbody tr')
                checks['browser_file_reviewed_before_import']=page.locator('.calibration-import-table tbody tr').count()==1
                checks['board_definition_detected']='Board definition' in page.locator('.calibration-import-table').inner_text()
                checks['confirm_before_mutation']='Nothing is registered until you confirm' in page.locator('.calibration-import-review-panel').inner_text()
                page.screenshot(path=str(outputs[1]),full_page=True);page.click('#calimport-close')

                state=request_json(url,'/api/state');project=state['project'];project_id=project['id'];session_id=project['selected_session_id'];set_code=next(iter(project['calibration_sets'].values()))['code'];raw=board.read_bytes()
                start=request_json(url,'/api/calibration-assets/uploads','POST',{'project_id':project_id,'session_id':session_id,'calibration_set_id':set_code,'filename':board.name,'size':len(raw)})
                request_json(url,f"/api/imports/uploads/{start['upload_id']}?offset=0",'PUT',data=raw,headers={'Content-Type':'application/octet-stream'})
                finish=request_json(url,f"/api/calibration-assets/uploads/{start['upload_id']}/finish",'POST',{'project_id':project_id,'calibration_set_id':set_code})
                job=poll(url,finish['job']['id']);checks['browser_upload_backend_completed']=job['status']=='completed'
                page.evaluate("refreshState({quiet:false,fullRender:true})");page.wait_for_timeout(500);page.wait_for_selector('.calibration-source-table tbody tr')
                checks['calibration_source_visible_in_set']=page.locator('.calibration-source-table tbody tr').count()==1
                checks['calibration_source_excluded_from_interaction_inventory']=page.locator('.data-table-card tbody tr').count()==1 and 'calibration_board' not in page.locator('.data-table-card').inner_text().lower()
                page.evaluate("navigate('calibration')");page.wait_for_selector('.calibration-input-readiness')
                text=page.locator('body').inner_text();checks['calibration_page_is_operations_only']='Open Data → Calibration Data' in text and 'This page performs wearer resolution' in text
                checks['calibration_page_has_no_data_dropzone']=page.locator('#calibration-data-drop-zone').count()==0
                page.screenshot(path=str(outputs[2]),full_page=True);browser.close()
        finally:
            server.should_exit=True;thread.join(timeout=10)
    checks['page_errors_absent']=not report['page_errors'];checks['console_errors_absent']=not report['console_errors'];report['passed_count']=sum(value is True for value in checks.values());report['check_count']=len(checks);report['failed']=[key for key,value in checks.items() if value is not True];report['all_pass']=not report['failed'] and not report['page_errors'] and not report['console_errors'];outputs[3].write_text(json.dumps(report,indent=2),'utf-8');print(json.dumps(report,indent=2));return 0 if report['all_pass'] else 1
if __name__=='__main__':raise SystemExit(main())
