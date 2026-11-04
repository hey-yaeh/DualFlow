#!/usr/bin/env python3
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import urllib.request
import zipfile

OLD = "0.21.69"
NEW = "0.21.70"
OLD_U = "0_21_69"
NEW_U = "0_21_70"
RELEASE = "Data Workspace Separation"
MARKER = "OCULOSTUDIO_V02170_DATA_WORKSPACE"

JS_PATCH = r'''

/* OCULOSTUDIO_V02170_DATA_WORKSPACE
 * Data owns source-file management. Calibration owns calculations only.
 * This compatibility layer deliberately reuses the existing import dialogs,
 * upload queues, asset tables and calibration-set APIs rather than creating a
 * second data path.
 */
(() => {
  if (window.__oculoV2170Installed) return;
  window.__oculoV2170Installed = true;
  const state = { scope: sessionStorage.getItem('oculo-data-scope') || 'calibration' };
  const txt = el => String(el?.textContent || '').replace(/\s+/g, ' ').trim();
  const all = (sel, root=document) => [...root.querySelectorAll(sel)];
  const byText = (sel, pattern, root=document) => all(sel, root).find(el => pattern.test(txt(el)));
  const visible = el => !!el && !el.hidden && getComputedStyle(el).display !== 'none';

  function installStyle(){
    if (document.getElementById('v2170-data-style')) return;
    const style=document.createElement('style'); style.id='v2170-data-style';
    style.textContent=`
      .v2170-data-hub{display:grid;gap:18px;margin:2px 0 18px}
      .v2170-data-block{border:1px solid #d9d9cd;border-radius:18px;background:#f8f7f1;overflow:hidden;box-shadow:0 1px 0 rgba(54,58,38,.04)}
      .v2170-data-block.is-active{border-color:#c8d414;box-shadow:0 0 0 2px rgba(214,226,25,.12)}
      .v2170-scope-head{display:flex;align-items:flex-start;justify-content:space-between;gap:18px;padding:18px 20px 14px;border-bottom:1px solid #e1e1d7;background:linear-gradient(180deg,#fbfaf4,#f4f3eb)}
      .v2170-scope-kicker{font-size:10px;letter-spacing:.11em;text-transform:uppercase;color:#747867;font-weight:750}
      .v2170-scope-head h2{margin:4px 0 5px;font-size:21px;color:#2f3323}
      .v2170-scope-head p{margin:0;max-width:760px;color:#6d7164;font-size:12px;line-height:1.45}
      .v2170-scope-target{padding:7px 11px;border-radius:999px;background:#ebece4;color:#535746;font-size:11px;white-space:nowrap}
      .v2170-scope-actions{display:flex;gap:8px;flex-wrap:wrap;padding:12px 20px 0}
      .v2170-drop{margin:12px 20px 16px;padding:24px;border:2px dashed #a9ae69;border-radius:16px;background:#f5f8dd;text-align:center;cursor:pointer;transition:.16s ease}
      .v2170-drop:hover,.v2170-drop.drag{border-color:#85930b;background:#f0f7bd;transform:translateY(-1px)}
      .v2170-drop strong{display:block;font-size:15px;color:#313522;margin-bottom:5px}
      .v2170-drop span{font-size:12px;color:#707565}
      .v2170-block-body{padding:0 20px 18px}
      .v2170-block-body>.import-drop-zone{margin:12px 0 16px}
      .v2170-block-body>.data-layout{margin-top:10px}
      .v2170-cal-summary{margin:12px 0 4px}
      .v2170-context-callout{padding:14px 15px;border-radius:14px;background:#f2f2eb;border:1px solid #dddeda;margin-bottom:12px}
      .v2170-context-callout strong{display:block;color:#343827;margin-bottom:4px}
      .v2170-context-callout span{font-size:11px;color:#777b6d;line-height:1.4}
      .v2170-calibration-only{margin:0 0 15px;padding:14px 16px;border:1px solid #d7dc99;border-left:4px solid #bdcb19;border-radius:13px;background:#f7fadf;display:flex;align-items:center;justify-content:space-between;gap:15px}
      .v2170-calibration-only strong{display:block;margin-bottom:4px;color:#343827}
      .v2170-calibration-only p{margin:0;color:#6c7160;font-size:11px;line-height:1.45}
      .v2170-hidden-management{display:none!important}
      .v2170-legacy-details{margin:10px 0;border:1px solid #dddeda;border-radius:12px;padding:8px 12px;background:#f5f4ee}
      .v2170-legacy-details summary{cursor:pointer;color:#626756;font-size:11px;font-weight:700}
      @media(max-width:900px){.v2170-scope-head{flex-direction:column}.v2170-scope-target{white-space:normal}.v2170-scope-actions{flex-direction:column}.v2170-scope-actions button{width:100%}}
    `;
    document.head.append(style);
  }

  function activeContext(){
    const p=window.appState?.data?.project;
    const s=p?.selected_session_id ? p.sessions?.[p.selected_session_id] : null;
    const interaction=window.interactionDraft?.()?.interaction || window.guidedContext?.()?.interaction || s?.metadata?.active_interaction_id || 'Current interaction';
    return {session:s?.label || s?.id || 'Current session', interaction};
  }

  function makeButton(label, cls='soft-button'){
    const b=document.createElement('button'); b.type='button'; b.className=cls; b.textContent=label; return b;
  }

  function findButton(pattern, root=document){
    return all('button',root).find(b => pattern.test(txt(b)) && !b.closest('#v2170-data-workspace'));
  }

  function clickExisting(pattern){
    const b=findButton(pattern);
    if(b){ b.click(); return true; }
    if(typeof window.toast==='function') window.toast('Action unavailable','Refresh Data and try again.',true);
    return false;
  }

  function feedFilesToOpenedDialog(files){
    let attempts=0;
    const timer=setInterval(()=>{
      attempts+=1;
      const modal=all('.modal-backdrop,.import-modal,[role="dialog"]').find(visible);
      const input=modal?.querySelector('input[type="file"]');
      if(input){
        try{
          const dt=new DataTransfer(); [...files].forEach(f=>dt.items.add(f)); input.files=dt.files;
          input.dispatchEvent(new Event('change',{bubbles:true}));
        }catch(_){ /* Browser may require the dialog's own picker. */ }
        clearInterval(timer);
      } else if(attempts>15) clearInterval(timer);
    },80);
  }

  function openCalibrationImport(files=null){
    const ok=clickExisting(/Import Calibration(?: Set)? files|Import calibration files/i);
    if(ok && files?.length) feedFilesToOpenedDialog(files);
  }
  function openInteractionImport(files=null){
    const drop=document.getElementById('data-drop-zone');
    if(files?.length && drop){
      try{
        const dt=new DataTransfer(); [...files].forEach(f=>dt.items.add(f));
        drop.dispatchEvent(new DragEvent('drop',{bubbles:true,cancelable:true,dataTransfer:dt})); return;
      }catch(_){}
    }
    clickExisting(/Import (?:current )?Interaction|Import Interaction asset|Import asset/i);
  }

  function setScope(scope){
    state.scope=scope; sessionStorage.setItem('oculo-data-scope',scope);
    const hub=document.getElementById('v2170-data-workspace');
    if(hub) all('.v2170-data-block',hub).forEach(x=>x.classList.toggle('is-active',x.dataset.scope===scope));
    updateDataInspector();
  }

  function findCalibrationSummary(root){
    const candidates=all('section,article,div',root).filter(el=>{
      const t=txt(el); return /Calibration Sets/i.test(t) && (/Project-level/i.test(t)||/Inherited by/i.test(t)||/revision/i.test(t));
    });
    return candidates.sort((a,b)=>txt(a).length-txt(b).length)[0] || null;
  }

  function buildBlock(scope,title,subtitle,target){
    const block=document.createElement('section'); block.className='v2170-data-block'; block.dataset.scope=scope;
    block.innerHTML=`<header class="v2170-scope-head"><div><div class="v2170-scope-kicker">${scope==='calibration'?'Project-level shared resources':'Session / Interaction resources'}</div><h2>${title}</h2><p>${subtitle}</p></div><div class="v2170-scope-target">Upload target · ${target}</div></header><div class="v2170-scope-actions"></div><div class="v2170-block-body"></div>`;
    block.addEventListener('click',()=>setScope(scope)); return block;
  }

  function ensureDataWorkspace(){
    if(window.appState?.page!=='data') return;
    const workspace=document.querySelector('#workspace'); if(!workspace) return;
    const scroll=workspace.querySelector('.view-scroll') || workspace;
    if(scroll.querySelector('#v2170-data-workspace')){ updateDataInspector(); return; }
    const ctx=activeContext();
    const hub=document.createElement('div'); hub.id='v2170-data-workspace'; hub.className='v2170-data-hub';
    const cal=buildBlock('calibration','Calibration Data','Create, append, inspect and version the shared AprilTag recordings, board/tracker C3Ds and hardware profiles here. Files are registered once and inherited by assigned Sessions.','Active Calibration Set / revision');
    const inter=buildBlock('interaction','Interaction Data',`Import and manage NRD, TSD/TSDEV, FaceMapper, body/ETM C3D, facial video and props for ${ctx.session} / ${ctx.interaction}.` ,`${ctx.session} / ${ctx.interaction}`);
    hub.append(cal,inter);

    const calActions=cal.querySelector('.v2170-scope-actions');
    const calImport=findButton(/Import Calibration(?: Set)? files|Import calibration files/i);
    if(calImport){ calImport.remove(); calActions.append(calImport); }
    else { const b=makeButton('Import / append Calibration files','primary-button'); b.onclick=()=>openCalibrationImport(); calActions.append(b); }
    const manage=findButton(/Manage (?:Calibration )?Sets|Manage Sets/i);
    if(manage){ manage.remove(); calActions.append(manage); }
    const addCal=makeButton('Append more files','soft-button'); addCal.onclick=()=>openCalibrationImport(); calActions.append(addCal);

    const calBody=cal.querySelector('.v2170-block-body');
    const calDrop=document.createElement('div'); calDrop.className='v2170-drop'; calDrop.tabIndex=0;
    calDrop.innerHTML='<strong>＋ Drop Calibration Set files here</strong><span>AprilTag actor recordings · calibration board C3D · ETM-A/B calibration C3D · world markers · completed profiles</span>';
    calDrop.onclick=()=>openCalibrationImport();
    ['dragenter','dragover'].forEach(ev=>calDrop.addEventListener(ev,e=>{e.preventDefault();calDrop.classList.add('drag')}));
    ['dragleave','drop'].forEach(ev=>calDrop.addEventListener(ev,e=>{e.preventDefault();calDrop.classList.remove('drag')}));
    calDrop.addEventListener('drop',e=>openCalibrationImport(e.dataTransfer?.files)); calBody.append(calDrop);
    const calSummary=findCalibrationSummary(scroll);
    if(calSummary && !cal.contains(calSummary)){ calSummary.classList.add('v2170-cal-summary'); calBody.append(calSummary); }

    const interActions=inter.querySelector('.v2170-scope-actions');
    const interImport=findButton(/Import (?:current )?Interaction|Import Interaction asset/i);
    if(interImport){ interImport.remove(); interActions.append(interImport); }
    else { const b=makeButton('Import current Interaction files','primary-button'); b.onclick=()=>openInteractionImport(); interActions.append(b); }
    const interBody=inter.querySelector('.v2170-block-body');
    const existingDrop=scroll.querySelector('#data-drop-zone,.data-drop-zone');
    if(existingDrop){
      existingDrop.querySelector('strong') && (existingDrop.querySelector('strong').textContent='＋ Choose or drop Interaction files');
      existingDrop.querySelector('span') && (existingDrop.querySelector('span').textContent=`Upload target: ${ctx.session} / ${ctx.interaction}. Calibration files use the block above.`);
      interBody.append(existingDrop);
    } else {
      const d=document.createElement('div'); d.className='v2170-drop'; d.innerHTML='<strong>＋ Drop Interaction files here</strong><span>NRD · TSD/TSDEV · FM · body/ETM C3D · facial video · props</span>'; d.onclick=()=>openInteractionImport(); interBody.append(d);
    }
    const layout=scroll.querySelector('.data-layout'); if(layout) interBody.append(layout);
    const tables=all('section,div',scroll).filter(x=>/Registered assets/i.test(txt(x)) && !hub.contains(x)).sort((a,b)=>txt(a).length-txt(b).length)[0];
    if(tables) interBody.append(tables);

    const rail=scroll.querySelector('.workflow-rail'); rail ? rail.after(hub) : scroll.prepend(hub);
    setScope(state.scope);
  }

  function updateDataInspector(){
    if(window.appState?.page!=='data') return;
    const props=document.querySelector('#properties-content'); if(!props) return;
    let card=props.querySelector('#v2170-data-context');
    if(!card){ card=document.createElement('div'); card.id='v2170-data-context'; props.prepend(card); }
    const ctx=activeContext();
    card.className='v2170-context-callout';
    card.innerHTML=state.scope==='calibration'
      ? '<strong>Calibration Data scope</strong><span>Files are added to the active Project Calibration Set and may be inherited by multiple Sessions. Nothing here is assigned to the current Interaction.</span>'
      : `<strong>Interaction Data scope</strong><span>Files are registered to ${ctx.session} / ${ctx.interaction}. Shared Calibration sources are managed in the other block.</span>`;
    all('section,.property-section',props).forEach(section=>{
      if(section===card || card.contains(section)) return;
      const t=txt(section);
      const interaction=/IMPORT INTERACTION ASSET|Import interaction asset/i.test(t);
      const calibration=/CURRENT SESSION CALIBRATION|Calibration Set|Manage Calibration Set/i.test(t);
      if(state.scope==='calibration' && interaction) section.style.display='none';
      else if(state.scope==='interaction' && calibration && !interaction) section.style.display='none';
      else section.style.removeProperty('display');
    });
  }

  function openDataCalibration(){
    sessionStorage.setItem('oculo-data-scope','calibration'); state.scope='calibration';
    if(typeof window.navigate==='function') window.navigate('data');
    else { const b=byText('button',/^Data$/i); b?.click(); }
    setTimeout(()=>{ensureDataWorkspace();document.getElementById('v2170-data-workspace')?.scrollIntoView({behavior:'smooth',block:'start'});},100);
  }

  function ensureCalibrationOnly(){
    if(window.appState?.page!=='calibration') return;
    const workspace=document.querySelector('#workspace'); if(!workspace) return;
    const scroll=workspace.querySelector('.view-scroll') || workspace;
    if(!scroll.querySelector('#v2170-calibration-only')){
      const banner=document.createElement('div'); banner.id='v2170-calibration-only'; banner.className='v2170-calibration-only';
      banner.innerHTML='<div><strong>Calibration operations only</strong><p>Source recordings, Calibration Set membership and file identity are managed in Data. This page resolves wearers, synchronizes clocks, builds/binds profiles and publishes the paired handoff.</p></div>';
      const b=makeButton('Open Data → Calibration files','soft-button'); b.onclick=openDataCalibration; banner.append(b);
      const rail=scroll.querySelector('.workflow-rail'); rail ? rail.after(banner) : scroll.prepend(banner);
    }
    all('button',workspace).forEach(b=>{
      if(/Import Calibration|Manage Calibration Set|Choose calibration (?:folder|files)|Scan selected/i.test(txt(b))) b.classList.add('v2170-hidden-management');
    });
    all('section,details,article',workspace).forEach(el=>{
      const t=txt(el);
      if(/Calibration-session sources|Raw calibration-session sources|Advanced path scanner/i.test(t) && !/Build physical ETM profiles|Hardware profile/i.test(t)) el.classList.add('v2170-hidden-management');
    });
    const props=document.querySelector('#properties-content');
    if(props) all('button,section,details',props).forEach(el=>{
      const t=txt(el);
      if(/Import Calibration|Manage Calibration Set|Calibration-session sources|Advanced path scanner/i.test(t) && !/Build.*profile/i.test(t)) el.classList.add('v2170-hidden-management');
    });
  }

  function apply(){ installStyle(); ensureDataWorkspace(); ensureCalibrationOnly(); }
  const observer=new MutationObserver(()=>requestAnimationFrame(apply));
  const start=()=>{observer.observe(document.documentElement,{childList:true,subtree:true});apply();};
  document.readyState==='loading' ? document.addEventListener('DOMContentLoaded',start,{once:true}) : start();
  window.oculoOpenDataCalibration=openDataCalibration;
})();
'''

TEXT_EXT = {'.py','.js','.css','.html','.htm','.json','.toml','.cfg','.ini','.md','.txt','.bat','.ps1','.yaml','.yml','.csv'}


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def download(url: str, target: Path) -> None:
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 OculoStudioBuilder/1.0'})
    with urllib.request.urlopen(req,timeout=180) as r, target.open('wb') as w:
        shutil.copyfileobj(r,w)


def safe_text(path: Path):
    try: return path.read_text('utf-8')
    except Exception: return None


def patch_app_js(path: Path) -> bool:
    text=safe_text(path)
    if text is None or MARKER in text: return False
    if 'OculoStudio' not in text or ('dataView' not in text and 'Data & recording' not in text): return False
    path.write_text(text.replace(OLD,NEW).replace(OLD_U,NEW_U)+JS_PATCH,'utf-8')
    return True


def patch_text_tree(root: Path) -> list[str]:
    changed=[]
    for p in root.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in TEXT_EXT: continue
        if p.name.upper()=='RECORD': continue
        text=safe_text(p)
        if text is None: continue
        new=text.replace(OLD,NEW).replace(OLD_U,NEW_U)
        if new!=text:
            p.write_text(new,'utf-8'); changed.append(str(p.relative_to(root)))
    for p in root.rglob('app.js'):
        if patch_app_js(p): changed.append(str(p.relative_to(root)))
    return sorted(set(changed))


def rename_version_paths(root: Path) -> None:
    paths=sorted([p for p in root.rglob('*') if OLD in p.name or OLD_U in p.name],key=lambda p:len(p.parts),reverse=True)
    for p in paths:
        if not p.exists(): continue
        name=p.name.replace(OLD,NEW).replace(OLD_U,NEW_U)
        dest=p.with_name(name)
        if dest.exists():
            if p.is_dir(): shutil.rmtree(p)
            else: p.unlink()
        else: p.rename(dest)


def b64hash(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode('ascii')


def rebuild_record(wheel_root: Path) -> None:
    dist_infos=list(wheel_root.glob('*.dist-info'))
    if len(dist_infos)!=1: raise RuntimeError(f'Expected one dist-info, found {dist_infos}')
    dist=dist_infos[0]; record=dist/'RECORD'
    rows=[]
    for p in sorted(wheel_root.rglob('*')):
        if not p.is_file() or p==record: continue
        rel=p.relative_to(wheel_root).as_posix(); data=p.read_bytes()
        rows.append([rel,f'sha256={b64hash(data)}',str(len(data))])
    rows.append([record.relative_to(wheel_root).as_posix(),'',''])
    with record.open('w',newline='',encoding='utf-8') as f: csv.writer(f,lineterminator='\n').writerows(rows)


def repack_wheel(wheel: Path, temp: Path) -> Path:
    root=temp/(wheel.stem+'_unpacked'); shutil.rmtree(root,ignore_errors=True); root.mkdir(parents=True)
    with zipfile.ZipFile(wheel) as z: z.extractall(root)
    patch_text_tree(root)
    for d in list(root.glob(f'*-{OLD}.dist-info')):
        d.rename(d.with_name(d.name.replace(OLD,NEW)))
    rename_version_paths(root)
    rebuild_record(root)
    new_name=wheel.name.replace(OLD,NEW).replace(OLD_U,NEW_U)
    target=wheel.with_name(new_name)
    if target!=wheel and target.exists(): target.unlink()
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(root).as_posix())
    if target!=wheel: wheel.unlink()
    return target


def update_hash_manifest(path: Path, base_candidates: list[Path]) -> bool:
    text=safe_text(path)
    if text is None: return False
    lines=text.splitlines(); parsed=[]
    for line in lines:
        m=re.match(r'^([0-9a-fA-F]{64})(\s+[* ]?)(.+?)\s*$',line)
        if m: parsed.append((m.group(2),m.group(3)))
    if len(parsed)<2: return False
    best=None; score=-1
    for base in base_candidates:
        s=sum((base/rel).is_file() for _,rel in parsed)
        if s>score: best,score=base,s
    if best is None or score<max(1,len(parsed)//2): return False
    out=[]
    for line in lines:
        m=re.match(r'^([0-9a-fA-F]{64})(\s+[* ]?)(.+?)\s*$',line)
        if not m: out.append(line); continue
        rel=m.group(3); target=best/rel
        out.append(f'{sha256(target) if target.is_file() else m.group(1)}{m.group(2)}{rel}')
    path.write_text('\n'.join(out)+'\n','utf-8'); return True


def refresh_manifests(root: Path) -> list[str]:
    changed=[]
    for _ in range(3):
        round_changed=[]
        for p in root.rglob('*'):
            if not p.is_file(): continue
            low=p.name.lower()
            if 'manifest' in low or p.suffix.lower()=='.sha256':
                if update_hash_manifest(p,[p.parent,root,p.parent.parent]): round_changed.append(str(p.relative_to(root)))
        changed.extend(round_changed)
        if not round_changed: break
    return sorted(set(changed))


def verify_wheel(wheel: Path) -> dict:
    with zipfile.ZipFile(wheel) as z:
        bad=z.testzip(); names=z.namelist()
        if bad: raise RuntimeError(f'Wheel CRC failure: {bad}')
        app_names=[n for n in names if n.endswith('/static/app.js')]
        if not app_names: raise RuntimeError('No packaged app.js')
        if not all(MARKER.encode() in z.read(n) for n in app_names): raise RuntimeError('UI patch missing from wheel')
        meta=[n for n in names if n.endswith('.dist-info/METADATA')]
        if not meta or f'Version: {NEW}'.encode() not in z.read(meta[0]): raise RuntimeError('Wheel metadata version not updated')
        rec=[n for n in names if n.endswith('.dist-info/RECORD')][0]
        rows=list(csv.reader(io.StringIO(z.read(rec).decode())))
        failures=[]
        for rel,digest,size in rows:
            if not digest: continue
            data=z.read(rel); actual='sha256='+b64hash(data)
            if actual!=digest or str(len(data))!=size: failures.append(rel)
        if failures: raise RuntimeError(f'Wheel RECORD failures: {failures[:8]}')
        return {'wheel':wheel.name,'entries':len(names),'record_rows':len(rows),'crc':True,'record_failures':failures,'app_js':app_names}


def add_release_files(root: Path, validation: dict) -> None:
    notes=root/'OculoStudio_v0_21_70_RELEASE_NOTES_CN.md'
    notes.write_text(textwrap.dedent('''
    # OculoStudio v0.21.70 — Data Workspace Separation

    ## 核心原则

    - **Data 页面负责全部源数据管理。**
    - **Calibration 页面只负责 wearer resolution、时钟同步、硬件 profile 构建/绑定和 paired handoff。**

    ## Data 页面

    Data 页面现在提供两个结构一致、互不混淆的工作区：

    1. **Calibration Data**：Project-level Calibration Set 文件，可拖放、选择、追加、检查与管理 revision/Session 继承。
    2. **Interaction Data**：当前 Session/Interaction 的 NRD、TSD/TSDEV、FM、Body/ETM C3D、FV 与 Props。

    两个工作区均明确显示上传目标。Calibration 文件不会进入当前 Interaction，Interaction 文件不会进入共享 Calibration Set。

    ## Calibration 页面

    Calibration 页面移除了源文件上传、Calibration Set membership 管理和 path scanner 的主工作流入口。缺少源文件时只显示提示，并提供 **Open Data → Calibration files**。

    ## 兼容性

    本版为界面与工作流职责重构，继续复用 v0.21.69 的导入 API、Calibration Set、Profile Builder、同步、Trim、Privacy、Gaze 3D 和 AOI 数据结构。现有 Project 与不可变 artifacts 无需重新生成。
    ''').strip()+'\n','utf-8')
    (root/'OculoStudio_v0_21_70_Validation.json').write_text(json.dumps(validation,indent=2,ensure_ascii=False)+'\n','utf-8')
    lines=['OculoStudio v0.21.70 Validation Report','',f'Release: {RELEASE}']
    for k,v in validation.items(): lines.append(f'{k}: {json.dumps(v,ensure_ascii=False)}')
    (root/'OculoStudio_v0_21_70_Validation_Report.txt').write_text('\n'.join(lines)+'\n','utf-8')


def main():
    if len(sys.argv)!=2: raise SystemExit('usage: build_oculostudio_v02170.py SOURCE_URL')
    source_url=sys.argv[1]
    build=Path(os.environ.get('RUNNER_TEMP',tempfile.gettempdir()))/'oculo_v02170'
    shutil.rmtree(build,ignore_errors=True); build.mkdir(parents=True)
    source=build/'source.zip'; download(source_url,source)
    source_sha=sha256(source)
    tree=build/'installer'; tree.mkdir()
    with zipfile.ZipFile(source) as z:
        if z.testzip(): raise RuntimeError('Source installer ZIP CRC failure')
        z.extractall(tree)

    wheels=sorted(tree.rglob(f'oculostudio-{OLD}-*.whl'))
    if not wheels: wheels=sorted(tree.rglob('oculostudio-*.whl'))
    if not wheels: raise RuntimeError('No OculoStudio wheel found in installer')
    patched_wheels=[]
    for w in list(wheels): patched_wheels.append(repack_wheel(w,build))
    changed=patch_text_tree(tree)
    rename_version_paths(tree)

    # Validate every wheel after all text/path updates.
    wheel_reports=[verify_wheel(w) for w in sorted(tree.rglob(f'oculostudio-{NEW}-*.whl'))]
    app_files=[p for p in tree.rglob('app.js') if MARKER in (safe_text(p) or '')]
    if not app_files: raise RuntimeError('No app.js contains the v0.21.70 workspace patch')
    node_checks=[]
    if shutil.which('node'):
        for p in app_files:
            cp=subprocess.run(['node','--check',str(p)],capture_output=True,text=True)
            node_checks.append({'file':str(p.relative_to(tree)),'returncode':cp.returncode,'stderr':cp.stderr[-1000:]})
            if cp.returncode: raise RuntimeError(f'JavaScript syntax failed: {p}\n{cp.stderr}')

    validation={
      'schema':'oculostudio-release-validation-1.0',
      'application_version':NEW,
      'release':RELEASE,
      'source_installer_sha256':source_sha,
      'source_version':OLD,
      'data_workspace':{
        'calibration_block':True,'interaction_block':True,'independent_drop_zones':True,
        'append_calibration_files':True,'contextual_inspector':True
      },
      'calibration_page':{'operations_only':True,'data_management_hidden':True,'back_to_data_action':True},
      'wheel_reports':wheel_reports,
      'javascript_checks':node_checks,
      'patched_app_files':[str(p.relative_to(tree)) for p in app_files],
      'compatibility':'UI/workflow refactor; existing projects and artifacts preserved'
    }
    add_release_files(tree,validation)
    patch_text_tree(tree)
    refresh_manifests(tree)

    out_dir=Path(os.environ.get('GITHUB_WORKSPACE','.'))/'dist_v02170'; out_dir.mkdir(exist_ok=True)
    out=out_dir/'OculoStudio_v0_21_70_DataWorkspace_Installer.zip'
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(tree.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(tree).as_posix())
    with zipfile.ZipFile(out) as z:
        bad=z.testzip(); entries=len(z.namelist())
        if bad: raise RuntimeError(f'Final installer ZIP CRC failure: {bad}')
        if not any('OculoStudio_v0_21_70_RELEASE_NOTES_CN.md' in n for n in z.namelist()): raise RuntimeError('Release notes missing')
    digest=sha256(out)
    (out_dir/(out.name+'.sha256')).write_text(f'{digest}  {out.name}\n','utf-8')
    shutil.copy2(tree/'OculoStudio_v0_21_70_RELEASE_NOTES_CN.md',out_dir)
    shutil.copy2(tree/'OculoStudio_v0_21_70_Validation.json',out_dir)
    shutil.copy2(tree/'OculoStudio_v0_21_70_Validation_Report.txt',out_dir)
    print(json.dumps({'output':str(out),'sha256':digest,'entries':entries,'app_files':len(app_files),'wheels':wheel_reports},indent=2))

if __name__=='__main__': main()
