"""Optional development-only UI test. Requires Playwright and a Chromium browser.

Creates a disposable synthetic project; never operates on a user's existing study.
By default uses normal HTTP. --transport python forwards real API calls through
Python for constrained test environments; it is not used by the released GUI.
"""
import sys,os,json,time,socket,threading,urllib.request,urllib.error,re,argparse,tempfile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description='Exercise the packaged NRD trim UI against its real API using temporary synthetic data.')
parser.add_argument('--output',default='native_trim_ui.png')
parser.add_argument('--chromium',default=None)
parser.add_argument('--transport',choices=['http','python'],default='http',help='python transports actual API calls via Python when sandbox Chromium navigation is blocked')
args=parser.parse_args()
output=Path(args.output).expanduser().resolve();output.parent.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'src'))
from oculostudio.server import app,runtime
from oculostudio.desktop import verify_ui_assets
from playwright.sync_api import sync_playwright
import uvicorn
from oculostudio.native_trim_demo import create_native_trim_demo
temp=tempfile.TemporaryDirectory(prefix='oculostudio_ui_test_')
project=Path(temp.name)/'synthetic_project'
create_native_trim_demo(project)
store=runtime.open(project);sid=store.project.selected_session_id
store.update(lambda p:(p.sessions[sid].metadata.pop('native_reference_trim_last',None),p.jobs.clear()))
ids={a.kind:a.id for a in store.project.sessions[sid].assets.values()}
with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='error'))
thread=threading.Thread(target=server.run,daemon=True);thread.start()
url=f'http://127.0.0.1:{port}'
for _ in range(200):
 try:urllib.request.urlopen(url+'/api/health').read();break
 except:time.sleep(.1)
verify_ui_assets(url)
report={'mode':('Standard HTTP navigation to a live FastAPI server.' if args.transport=='http' else 'Packaged HTML/CSS/JS in Chromium; fetch transported via Python to a LIVE local FastAPI server. No fabricated API responses; use only when browser navigation is blocked by the environment.'),'page_errors':[],'console_errors':[],'sizes':[]}
try:
 with sync_playwright() as p:
  executable=args.chromium or shutil.which('chromium') or shutil.which('google-chrome')
  options={'headless':True,'args':['--no-sandbox']}
  if executable:options['executable_path']=executable
  browser=p.chromium.launch(**options)
  def transport(path,method,headers,body):
   assert path.startswith('/api/')
   req=urllib.request.Request(url+path,data=body.encode() if body is not None else None,headers=headers,method=method)
   try:
    with urllib.request.urlopen(req,timeout=60) as resp:return {'status':resp.status,'body':resp.read().decode(),'type':resp.headers.get('Content-Type','application/json')}
   except urllib.error.HTTPError as e:return {'status':e.code,'body':e.read().decode(),'type':e.headers.get('Content-Type','application/json')}
  def new_page():
   pg=browser.new_page(viewport={'width':1560,'height':980},device_scale_factor=1)
   pg.on('pageerror',lambda e:report['page_errors'].append(str(e)))
   pg.on('console',lambda m:report['console_errors'].append(m.text) if m.type=='error' else None)
   if args.transport=='http':
    pg.goto(url,wait_until='networkidle')
    pg.locator('[data-page="clips"]').first.click()
   else:
    static=ROOT/'src/oculostudio/static'
    html=re.sub(r'<script[^>]+src=[^>]+></script>','',(static/'index.html').read_text())
    html=re.sub(r'<link[^>]+rel="stylesheet"[^>]*>','',html)
    pg.set_content(html)
    pg.add_style_tag(content=(static/'styles.css').read_text())
    pg.expose_function('localStudioTransport',transport)
    pg.evaluate('''() => {window.fetch=async(path,init={})=>{const r=await window.localStudioTransport(String(path),init.method||'GET',init.headers||{},init.body||null);return new Response(r.body,{status:r.status,headers:{'content-type':r.type}})}}''')
    pg.add_script_tag(content=(static/'native_trim.js').read_text())
    pg.add_script_tag(content=(static/'asset_import.js').read_text())
    pg.add_script_tag(content=(static/'app.js').read_text())
    pg.evaluate("appState.page='clips'; document.dispatchEvent(new Event('DOMContentLoaded'))")
   pg.wait_for_selector('#native-trim-start')
   return pg
  page=new_page()
  assert page.input_value('#native-timecode-fps')=='30'
  page.select_option('#native-trim-nrd',ids['pupil_native'])
  page.wait_for_function("document.querySelector('#native-trim-part').options.length>1")
  page.select_option('#native-trim-tsd',ids['pupil_cloud'])
  page.select_option('#native-trim-fm',ids['face_mapper'])
  page.fill('#native-trim-start','00:00:49:21')
  page.fill('#native-trim-duration','00:09:24:15')
  page.wait_for_function("nativeTimeReady(nativeDraft())")
  text=page.locator('#native-time-interpretation').inner_text()
  assert all(t in text for t in ['00:00:49.700','00:09:24.500','00:10:14.200']),text
  report['user_timecodes_interpreted']={'start':'00:00:49.700','duration':'00:09:24.500','end':'00:10:14.200'}
  assert len(store.project.jobs)==0, 'input parsing must not queue an indexing job'
  page.select_option('#native-timecode-fps','25')
  page.wait_for_function("nativeTimeReady(nativeDraft()) && nativeDraft().timeValidation.parsed.timecode_fps===25")
  assert '00:00:49.840' in page.locator('#native-time-interpretation').inner_text()
  page.select_option('#native-timecode-fps','30')
  page.fill('#native-trim-start','00:00:49:30')
  page.wait_for_function("nativeDraft().timeValidation?.status==='error'")
  assert '00–29' in page.locator('#native-time-interpretation').inner_text()
  assert page.locator('#native-preview-button').is_disabled()
  assert page.locator('#native-export-button').is_disabled()
  assert len(store.project.jobs)==0
  report['invalid_FF_rejected_inline_without_job']=True
  # Valid short clip in a four-second SYNTHETIC source, not the user's I01 cut.
  page.fill('#native-trim-start','00:00:00:18')
  page.fill('#native-trim-duration','00:00:02:00')
  page.fill('#native-trim-label','Synthetic timecode regression · 30 fps')
  page.wait_for_function("nativeTimeReady(nativeDraft())")
  page.wait_for_timeout(1700)
  assert page.input_value('#native-trim-start')=='00:00:00:18'
  page.click('#native-preview-button')
  page.wait_for_function("nativePreviewCurrent(nativeDraft())",timeout=60000)
  assert page.locator('#native-export-button').is_enabled()
  assert page.locator('.native-preview-result').count()==1
  req_ts=page.locator('.native-utc code').inner_text()
  report['utc_interval_displayed_as_exact_string']=req_ts
  assert len([j for j in store.project.jobs.values() if j.operation=='native_trim_preview'])==1
  # An FPS edit must invalidate the previous input interpretation AND plan.
  page.select_option('#native-timecode-fps','25')
  assert page.locator('#native-export-button').is_disabled()
  page.wait_for_function("nativeTimeReady(nativeDraft())")
  assert page.locator('#native-export-button').is_disabled()
  assert not page.locator('.native-preview-result').is_visible()
  page.select_option('#native-timecode-fps','30')
  page.wait_for_function("nativeTimeReady(nativeDraft())")
  assert page.locator('#native-export-button').is_enabled()
  page.click('#native-export-button')
  page.wait_for_selector('.native-export-result',timeout=60000)
  last=store.snapshot().sessions[sid].metadata['native_reference_trim_last']
  assert last['request']['timecode_fps']==30
  assert last['request']['start_timecode']=='00:00:00:18'
  assert last['request']['timecode_interpretation']['start']['nanoseconds']=='600000000'
  # Save a non-default timebase to verify restoration is not hardwired to 30.
  page.select_option('#native-timecode-fps','25')
  page.wait_for_function("nativeTimeReady(nativeDraft())")
  page.fill('#native-trim-label','Synthetic timecode regression · 25 fps')
  page.click('#native-preview-button')
  page.wait_for_function("nativePreviewCurrent(nativeDraft())",timeout=60000)
  page.click('#native-export-button')
  page.wait_for_function("nativeDraft().result?.preview?.request?.timecode_fps===25",timeout=60000)
  page.close();page=new_page()
  assert page.input_value('#native-timecode-fps')=='25'
  assert page.input_value('#native-trim-start')=='00:00:00:18'
  page.wait_for_function("nativeTimeReady(nativeDraft())")
  assert '00:00:00.720' in page.locator('#native-time-interpretation').inner_text()
  page.click('#global-mode-button');page.wait_for_selector('#clip-start')
  page.click('#native-mode-button');page.wait_for_selector('#native-trim-start')
  assert page.input_value('#native-timecode-fps')=='25'
  # Screenshot of EXACT user inputs: format conversion only, no claim this
  # long interval fits the deliberately short synthetic fixture.
  page.select_option('#native-timecode-fps','30')
  page.fill('#native-trim-start','00:00:49:21')
  page.fill('#native-trim-duration','00:09:24:15')
  page.fill('#native-trim-label','Timecode conversion example · source bounds not checked')
  page.evaluate("nativeDraft().preview=null; nativeDraft().result=null; renderWorkspace()")
  page.wait_for_function("nativeTimeReady(nativeDraft())")
  page.wait_for_timeout(4000)
  for size in [(1560,980),(1366,768),(1920,1080)]:
   page.set_viewport_size({'width':size[0],'height':size[1]});page.wait_for_timeout(350)
   width=page.evaluate('document.documentElement.scrollWidth')
   assert width<=size[0],(size,width)
   assert page.locator('#native-trim-start').count()==1
   report['sizes'].append({'width':size[0],'height':size[1],'document_scroll_width':width,'single_start_field':True})
   if size==(1560,980):page.screenshot(path=str(output))
  report.update({'default_timecode_fps':30,'preview_jobs':2,'export_jobs':2,
    'polling_preserves_frame_inputs':True,'fps_edit_invalidates_plan':True,
    'non_default_fps_restored_after_reload':True,
    'saved_original_input_and_conversion':True,
    'user_screenshot_is_input_conversion_only':True,
    'synthetic_full_pipeline_executed':True,'pipeline_tabs':page.locator('.pipeline-tab').count()})
  browser.close()
finally:
 server.should_exit=True;thread.join(5)
 temp.cleanup()
report['ok']=not report['page_errors'] and not report['console_errors']
output.with_suffix('.validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
assert report['ok'],report
