"""Live UI validation for OculoStudio v0.21.71 Body AOI human review."""
from __future__ import annotations

import argparse, hashlib, json, re, shutil, socket, sys, tempfile, threading, time, urllib.error, urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))


def sha(path: Path) -> str:
    h=hashlib.sha256();
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def free_port():
    with socket.socket() as s: s.bind(('127.0.0.1',0)); return s.getsockname()[1]

def wait(url):
    deadline=time.time()+45
    while time.time()<deadline:
        try:
            with urllib.request.urlopen(url+'/api/health',timeout=1) as r:
                if r.status==200:return
        except Exception:time.sleep(.1)
    raise RuntimeError('server did not start')

def artifact_file(artifact,name):
    for item in artifact.files:
        p=Path(item.path)
        if p.name==name:return p
    raise RuntimeError(name)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(ROOT/'docs'/'OculoStudio_v0_21_50_BodyHumanReviewRegression.png'));args=parser.parse_args()
    import uvicorn, oculostudio
    from create_privacy_aoi_demo import build
    from oculostudio.desktop import verify_ui_assets
    from oculostudio.domain.models import ArtifactKind
    from oculostudio.server import app,runtime
    from oculostudio.services import aoi_review
    from oculostudio.services.aoi_review import review_manifest
    from playwright.sync_api import sync_playwright

    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    report={'release':'0.21.71','feature':'Body AOI human review entry and editor','checks':{},'page_errors':[],'console_errors':[]}
    checks=report['checks']
    with tempfile.TemporaryDirectory(prefix='oculo_2148_body_review_ui_') as td:
        project=Path(td)/'project';build(project,overwrite=True)
        store=runtime.open(project);session=store.selected_session()
        body=next(x for x in session.artifacts.values() if x.kind==ArtifactKind.BODY_AOI)
        video=next(x for x in session.artifacts.values() if x.kind==ArtifactKind.PRIVACY_CLIP)
        body_path=artifact_file(body,'body_aoi_samples.csv')
        samples=pd.read_csv(body_path,low_memory=False)
        flagged=list(samples.index[:3])
        samples.loc[:,'review_required']=False
        samples.loc[flagged,'review_required']=True
        samples.loc[flagged,'assignment_status']='validation_uncertain_body_target'
        samples.loc[flagged,'body_confidence']=0.44
        # Ensure a stable visible automatic target for the editor.
        samples.loc[flagged,'body_entity_id']='P02'
        samples.loc[flagged,'body_relation']='partner'
        samples.loc[flagged,'body_raw_region']='torso'
        samples.loc[flagged,'body_coarse_region']='torso'
        samples.loc[flagged,'body_region']='torso'
        samples.loc[flagged,'body_social_region']='partner_torso'
        samples.to_csv(body_path,index=False)
        body.qc.update({'review_required_fraction':len(flagged)/len(samples),'review_required_count':len(flagged),'output_rows':len(samples)})
        store.save();aoi_review._CACHE.clear()
        manifest=review_manifest(store,session_id=session.id,mode='body',artifact_id=body.id,video_artifact_id=video.id)
        first_pending=int(manifest['event_frames']['pending_review'][0])
        source_hash=sha(body_path)

        port=free_port();url=f'http://127.0.0.1:{port}'
        server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='error'))
        thread=threading.Thread(target=server.run,daemon=True);thread.start()
        try:
            wait(url);verify_ui_assets(url)
            static=Path(oculostudio.__file__).resolve().parent/'static'
            with sync_playwright() as pw:
                opts={'headless':True,'args':['--no-sandbox','--autoplay-policy=no-user-gesture-required']}
                exe=shutil.which('chromium') or shutil.which('google-chrome')
                if exe:opts['executable_path']=exe
                browser=pw.chromium.launch(**opts);page=browser.new_page(viewport={'width':1720,'height':1100})
                page.on('pageerror',lambda e:report['page_errors'].append(str(e)))
                page.on('console',lambda m:report['console_errors'].append(m.text) if m.type=='error' else None)
                html=re.sub(r'<script[^>]+src=[^>]+></script>','',(static/'index.html').read_text('utf-8'))
                html=re.sub(r'<link[^>]+rel="stylesheet"[^>]*>','',html)
                page.set_content(html);page.add_style_tag(content=(static/'styles.css').read_text('utf-8'))
                def relay(path,method,headers,body_text):
                    req=urllib.request.Request(url+path,method=method,headers=headers or {},data=body_text.encode() if body_text is not None else None)
                    try:
                        with urllib.request.urlopen(req,timeout=180) as r:return {'status':r.status,'body':r.read().decode(),'type':r.headers.get('Content-Type','application/json')}
                    except urllib.error.HTTPError as e:return {'status':e.code,'body':e.read().decode(),'type':e.headers.get('Content-Type','application/json')}
                page.expose_function('relayStudioHTTP',relay)
                page.evaluate("""() => { window.fetch=async(path,init={})=>{const r=await window.relayStudioHTTP(String(path),init.method||'GET',init.headers||{},init.body||null);return new Response(r.body,{status:r.status,headers:{'Content-Type':r.type}})}}""")
                def route_local(route):
                    rel=route.request.url.split('studio.local',1)[-1];headers={}
                    if route.request.headers.get('range'):headers['Range']=route.request.headers['range']
                    req=urllib.request.Request(url+rel,headers=headers)
                    try:
                        with urllib.request.urlopen(req,timeout=180) as r:
                            hs={k:v for k,v in r.headers.items() if k.lower() in {'content-type','content-length','content-range','accept-ranges'}}
                            route.fulfill(status=r.status,body=r.read(),headers=hs)
                    except urllib.error.HTTPError as e:route.fulfill(status=e.code,body=e.read())
                page.route('http://studio.local/**',route_local)
                page.evaluate("document.head.insertAdjacentHTML('afterbegin','<base href=\"http://studio.local/\">')")
                for name in ('native_trim.js','asset_import.js','aoi_review.js','app.js'):page.add_script_tag(content=(static/name).read_text('utf-8'))
                page.evaluate("appState.page='targets'; document.dispatchEvent(new Event('DOMContentLoaded'))")
                page.wait_for_selector('#aoi-review-shell',timeout=60_000)
                page.select_option('#aoi-review-mode','body')
                page.wait_for_function("v=>document.querySelector('#aoi-review-artifact')?.value===v",arg=body.id,timeout=60_000)
                page.wait_for_function("document.querySelector('#aoi-review-shell')?.classList.contains('body-review-active')",timeout=60_000)
                before=page.evaluate("""() => ({
                  entryVisible:document.querySelector('#aoi-body-human-review-entry')?.getBoundingClientRect().height>0,
                  buttonText:document.querySelector('#aoi-review-open-body-human-review')?.textContent?.trim(),
                  count:document.querySelector('#aoi-body-human-review-count')?.textContent?.trim(),
                  sideVisible:document.querySelector('#open-body-human-review-side')?.getBoundingClientRect().height>0,
                  retuneVisible:document.querySelector('#reduce-existing-body-review')?.getBoundingClientRect().height>0,
                  retunePolicy:document.querySelector('#body-review-retune-policy')?.value,
                  mode:document.querySelector('#aoi-review-mode')?.value
                })""")
                checks['body_entry_visible_in_body_mode']=bool(before['entryVisible']) and before['mode']=='body'
                checks['body_pending_count_visible']='3 events' in str(before['count']) and '3 samples' in str(before['count'])
                checks['right_property_entry_visible']=bool(before['sideVisible'])
                checks['review_only_reduction_entry_visible']=bool(before['retuneVisible']) and before['retunePolicy'] == 'semantic_guided'
                page.click('#aoi-review-open-body-human-review')
                page.wait_for_function("document.querySelector('#body-manual-review')?.hidden===false",timeout=60_000)
                page.wait_for_function("document.querySelector('#body-review-progress')?.textContent.includes('review events resolved')",timeout=60_000)
                page.wait_for_function("document.querySelector('#body-review-guidance')?.textContent.includes('flagged')",timeout=60_000)
                page.wait_for_timeout(400)
                after=page.evaluate("""() => ({
                  panelHidden:document.querySelector('#body-manual-review')?.hidden,
                  progress:document.querySelector('#body-review-progress')?.textContent?.trim(),
                  status:document.querySelector('#body-review-status')?.textContent?.trim(),
                  guidance:document.querySelector('#body-review-guidance')?.textContent?.trim(),
                  auto:document.querySelector('#body-review-auto')?.textContent?.trim(),
                  relationOptions:document.querySelector('#body-review-relation')?.options.length,
                  regionOptions:document.querySelector('#body-review-region')?.options.length,
                  noBody:Boolean([...document.querySelectorAll('[data-body-review-action]')].find(x=>x.textContent.includes('Mark no body'))),
                  apply:Boolean(document.querySelector('#body-review-apply')),
                  currentFrame:Number((document.querySelector('#aoi-review-summary')?.innerText.match(/Video frame\\s+(\\d+)/)||[])[1]),
                  pageWidth:document.documentElement.scrollWidth
                })""")
                checks['body_editor_open']=after['panelHidden'] is False and 'review events resolved' in after['progress'] and '3 samples remain' in after['progress']
                checks['body_editor_fields']=int(after['relationOptions'] or 0)>=4 and int(after['regionOptions'] or 0)>=10
                checks['body_actions_available']=bool(after['noBody'] and after['apply'])
                checks['jumps_to_first_pending_frame']=after['currentFrame']==first_pending
                checks['body_automatic_target_visible']='Automatic target' in str(after['auto']) and 'Partner' in str(after['auto'])
                checks['responsive_without_horizontal_overflow']=int(after['pageWidth'])<=1720
                # Save an acceptance through the actual UI endpoint.
                page.click('[data-body-review-action="accept_auto"]')
                page.wait_for_function("document.querySelector('#body-review-progress')?.textContent.includes('2 samples remain')",timeout=60_000)
                checks['ui_decision_saves']= '2 samples remain' in page.locator('#body-review-progress').inner_text()
                page.screenshot(path=str(output),full_page=False)
                report['before']=before;report['after']=after
                browser.close()
            checks['source_body_aoi_immutable']=sha(body_path)==source_hash
            unexpected=[x for x in report['console_errors'] if 'favicon' not in x.lower()]
            checks['no_page_errors']=not report['page_errors'];checks['no_console_errors']=not unexpected
            checks['installed_version']=oculostudio.__version__=='0.21.71'
            failed=[k for k,v in checks.items() if not v];report['failed']=failed;report['ok']=not failed
        finally:
            server.should_exit=True;thread.join(timeout=10);runtime.close()
    output.with_suffix('.validation.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),'utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False));return 0 if report.get('ok') else 1

if __name__=='__main__':raise SystemExit(main())
