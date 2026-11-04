/* OculoStudio v0.21.72 · Session Shared Data
 * Adds a third Data scope for Session-level FaceMapper packages and keeps
 * Interaction append distinct from Project Calibration data.
 */
(() => {
  'use strict';
  const state={files:[],busy:false,serial:0,roleId:'R01',localPaths:[]};
  const sleep=ms=>new Promise(resolve=>setTimeout(resolve,ms));
  const nativeApi=()=>window.pywebview?.api;
  const workspace=()=>appState.data.session_shared_workspace||{sources:[],coverage:[],unassigned_sections:[],counts:{},paths:{}};
  const scopeOf=asset=>String(asset?.metadata?.asset_scope||'interaction');
  const activeRole=()=>String(activeWorkflowContext()?.roleId||'R01').toUpperCase();
  const sessionCode=()=>sessionWorkspace()?.session_code||session()?.label||'Session';

  function coverageTable(){
    const ws=workspace(),rows=ws.coverage||[];
    if(!rows.length)return '<div class="empty-tree">No planned or imported Interactions yet. Add Interaction files and the recording-ID coverage matrix will appear here.</div>';
    const cell=(row,role)=>{const value=row.roles?.[role]||{status:'missing',message:'No match'};const cls=value.status==='ready'?'good':value.status==='missing'?'bad':value.status==='ambiguous'?'warn':'';return `<td><span class="status-pill ${cls}" title="${esc(value.message||'')}">${esc(value.status||'missing')}</span><small class="session-shared-cell-note">${esc(value.message||'')}</small></td>`;};
    return `<div class="asset-table-scroll"><table class="asset-table session-shared-coverage"><thead><tr><th>Interaction</th><th>R01 FaceMapper</th><th>R02 FaceMapper</th></tr></thead><tbody>${rows.map(row=>`<tr><td><strong>${esc(row.interaction_id)}</strong></td>${cell(row,'R01')}${cell(row,'R02')}</tr>`).join('')}</tbody></table></div>`;
  }

  function sourceTable(){
    const rows=workspace().sources||[];
    if(!rows.length)return '<div class="empty-tree">No Session-shared FaceMapper source is registered. Drop one R01 and/or R02 Session package above.</div>';
    return `<div class="asset-table-scroll"><table class="asset-table"><thead><tr><th>Source</th><th>Role</th><th>Revision</th><th>Sections</th><th>Status</th><th></th></tr></thead><tbody>${rows.map(row=>`<tr><td><span class="table-label">${esc(row.label)}</span><span class="table-path" title="${esc(row.source_path)}">${esc(row.source_path)}</span></td><td><span class="kind-pill">${esc(row.role_id||'—')}</span></td><td>${esc(row.revision_number||1)}</td><td>${esc(row.section_count||0)}</td><td><span class="status-pill ${row.available?'good':'bad'}">${row.available?'Available':'Missing'}</span></td><td><button class="row-button session-shared-open-source" data-path="${esc(row.source_path)}" title="Reveal source">↗</button></td></tr>`).join('')}</tbody></table></div>`;
  }

  function block(){
    const ws=workspace(),counts=ws.counts||{},ready=counts.ready||0,missing=counts.missing||0,ambiguous=counts.ambiguous||0;
    return `<section id="data-session-shared-workspace" class="data-scope-workspace session-shared-data-workspace">
      <header class="data-scope-header"><div><span class="eyebrow">Session-level shared data</span><h2>Session Shared Data</h2><p>Upload one FaceMapper package per wearer role for the whole Session. OculoStudio indexes every recording section and resolves the correct Interaction by exact <code>recording_id</code>; raw ZIP bytes are never copied into every I-code.</p></div><div class="data-scope-context"><span>Upload target</span><strong>${esc(sessionCode())}</strong><small>Shared across Interactions · not part of the Calibration Set</small></div></header>
      <div id="session-shared-data-drop-zone" class="import-drop-zone data-scope-drop session-shared-drop-zone" role="button" tabindex="0" aria-label="Drop Session-shared FaceMapper files"><strong>＋ Drop Session FaceMapper files here</strong><span>Use the complete Session-level R01 or R02 FaceMapper ZIP. The section index is built before the source becomes active.</span></div>
      <div class="data-scope-actions"><button class="primary-button" id="data-session-shared-choose-files">Choose Session files…</button><button class="soft-button" id="data-session-shared-choose-folder">Choose Session folder…</button><button class="soft-button" id="data-session-shared-append">Append / add revision</button><button class="soft-button" id="data-session-shared-promote">Promote existing FaceMapper</button><button class="soft-button" id="data-session-shared-open-folder">Open Session shared folder</button></div>
      <div class="session-shared-summary"><span class="status-pill ${ready?'good':''}">${ready} ready</span><span class="status-pill ${missing?'bad':''}">${missing} missing</span><span class="status-pill ${ambiguous?'warn':''}">${ambiguous} ambiguous</span><span class="status-pill">${(ws.sources||[]).length} active source${(ws.sources||[]).length===1?'':'s'}</span></div>
      <div class="data-scope-content session-shared-content"><section class="table-card"><div class="iw-section-head compact"><div><span class="eyebrow">Session sources</span><h3>FaceMapper packages</h3></div></div>${sourceTable()}</section><section class="table-card"><div class="iw-section-head compact"><div><span class="eyebrow">Exact recording identity</span><h3>Interaction coverage</h3></div><span class="status-pill">recording_id</span></div>${coverageTable()}</section></div>
      ${(ws.unassigned_sections||[]).length?`<details class="metadata-details session-unassigned"><summary>${ws.unassigned_sections.length} FaceMapper section(s) are not assigned to an Interaction</summary><pre class="inspector-code">${esc(JSON.stringify(ws.unassigned_sections,null,2))}</pre></details>`:''}
    </section>`;
  }

  function installDataView(){
    if(typeof dataView!=='function'||dataView.__sessionSharedWrapped)return;
    const original=dataView;
    const wrapped=function(){
      let html=original();
      html=html.replace('<section id="data-interaction-workspace"',block()+'\n<section id="data-interaction-workspace"');
      html=html.replace('Manage NRD, TSD/TSDEV, FaceMapper, body and ETM C3D, facial video, props and other media for one Interaction.','Manage NRD, TSD/TSDEV, body and ETM C3D, facial video, props and other media for one Interaction. Session-wide FaceMapper packages belong in Session Shared Data.');
      html=html.replace('NRD / TSD / TSDEV / FM / body or ETM C3D / facial video / props.','NRD / TSD / TSDEV / body or ETM C3D / facial video / props. FaceMapper uses Session Shared Data.');
      html=html.replace('Choose Interaction files…','Append Interaction files…');
      html=html.replace('Choose Interaction folder…','Append Interaction folder…');
      return html;
    };
    wrapped.__sessionSharedWrapped=true;
    dataView=wrapped;
  }

  function modalRows(){
    return state.files.length?state.files.map(row=>`<div class="import-file-row"><div class="import-row-head"><strong>${esc(row.name)}</strong><span>${esc(row.status||'ready')}</span></div><div class="import-row-meta">${row.mode==='project_copy'?'Project copy':'Reference original'}${row.size!=null?' · '+fmtBytes(row.size):''}</div><progress value="${row.percent||0}" max="100"></progress><div class="import-row-result ${row.status==='failed'?'error':''}">${esc(row.message||'Ready')}</div></div>`).join(''):'<div class="empty-tree">Choose or drop a Session-level FaceMapper ZIP.</div>';
  }

  function updateModal(){
    const queue=document.querySelector('#session-shared-import-queue');if(queue)queue.innerHTML=modalRows();
    const run=document.querySelector('#session-shared-import-run');if(run){const pending=state.files.filter(row=>['ready','failed'].includes(row.status));run.disabled=state.busy||!pending.length;run.textContent=state.busy?'Importing…':`Import ${pending.length} file${pending.length===1?'':'s'} into ${sessionCode()}`;}
  }

  function addBrowserFiles(files){
    for(const file of Array.from(files||[])){
      if(!(file instanceof File))continue;
      if(state.files.some(row=>row.name===file.name&&row.size===file.size&&row.status!=='cancelled'))continue;
      state.files.push({id:`ss_${++state.serial}`,name:file.name,file,size:file.size,mode:'project_copy',status:'ready',percent:0,message:'Ready'});
    }
    updateModal();
  }
  function addPaths(paths){
    for(const raw of paths||[]){const path=String(raw||'').trim().replace(/^(["'])(.*)\1$/,'$2');if(!path)continue;const name=path.replaceAll('\\','/').split('/').filter(Boolean).at(-1)||path;if(state.files.some(row=>row.path===path&&row.status!=='cancelled'))continue;state.files.push({id:`ss_${++state.serial}`,name,path,mode:'local_reference',status:'ready',percent:0,message:'Ready'});}updateModal();
  }

  async function waitJob(row){
    for(;;){const {job}=await api('/api/imports/jobs/'+row.jobId);row.percent=80+20*Number(job.progress||0);row.message=job.error||job.stage||job.status;updateModal();if(job.status==='completed')return job.result;if(job.status==='failed')throw new Error(job.error||'Import failed');if(job.status==='cancelled')throw new Error('Import cancelled');await sleep(400);}
  }

  async function uploadRow(row,roleId){
    const p=project(),sid=selectedSessionId();
    const start=await api('/api/session-shared/uploads',{method:'POST',body:{project_id:p.id,session_id:sid,role_id:roleId,filename:row.file.name,size:row.file.size}});
    row.status='copying';row.message='Copying to Session shared storage…';updateModal();
    for(let offset=0;offset<row.file.size;){const blob=row.file.slice(offset,Math.min(offset+start.chunk_bytes,row.file.size));const response=await fetch(`/api/imports/uploads/${start.upload_id}?offset=${offset}`,{method:'PUT',body:blob,headers:{'Content-Type':'application/octet-stream'}});const value=await response.json();if(!response.ok)throw new Error(value.message||value.detail||'Upload failed');offset=value.offset;row.percent=80*offset/row.file.size;updateModal();}
    const final=await api(`/api/session-shared/uploads/${start.upload_id}/finish`,{method:'POST',body:{project_id:p.id,session_id:sid,role_id:roleId,conflict_policy:'add_revision'}});row.jobId=final.job.id;row.status='inspecting';row.message='Indexing recording sections…';return waitJob(row);
  }

  async function pathRow(row,roleId){
    const result=await api('/api/session-shared/import-paths',{method:'POST',body:{project_id:project().id,session_id:selectedSessionId(),role_id:roleId,paths:[row.path],conflict_policy:'add_revision',checksum:true}});row.jobId=result.job.id;row.status='inspecting';row.percent=5;return waitJob(row);
  }

  async function run(){
    if(state.busy)return;const pending=state.files.filter(row=>['ready','failed'].includes(row.status));if(!pending.length)return;state.busy=true;updateModal();
    for(const row of pending){try{row.status='inspecting';row.message='Starting…';row.percent=0;updateModal();if(row.mode==='project_copy')await uploadRow(row,state.roleId);else await pathRow(row,state.roleId);row.status='done';row.percent=100;row.message='Registered and indexed';}catch(error){row.status='failed';row.message=error.message||String(error);}updateModal();}
    state.busy=false;await refreshState({quiet:false});updateModal();toast('Session Shared Data updated','FaceMapper sections were indexed by recording ID.');
  }

  function open(files=null){
    state.files=[];state.busy=false;state.roleId=activeRole();
    modal('Import Session Shared Data',`<div class="session-shared-import-ui"><div class="scope-banner"><strong>${esc(sessionCode())} · Session Shared Data</strong><span>One FaceMapper package can cover many Interactions. Choose its wearer role; no I-code is assigned to the raw ZIP.</span></div><div class="field"><label>Wearer role</label><select id="session-shared-role"><option value="R01" ${state.roleId==='R01'?'selected':''}>R01</option><option value="R02" ${state.roleId==='R02'?'selected':''}>R02</option></select></div><div id="session-shared-modal-drop" class="import-drop-zone"><strong>Drop Session FaceMapper ZIPs here</strong><span>Files are indexed before becoming the active Session revision.</span></div><div class="import-buttons"><button class="primary-button" id="session-shared-browser">Choose files…</button><button class="soft-button" id="session-shared-native">Choose local files…</button><button class="soft-button" id="session-shared-folder">Choose folder…</button></div><div id="session-shared-import-queue" class="import-queue"></div><div class="form-actions"><button class="primary-button" id="session-shared-import-run">Import</button></div></div>`,(root)=>{
      root.querySelector('.modal').classList.add('asset-import-modal');
      const input=document.createElement('input');input.type='file';input.multiple=true;input.accept='.zip';input.hidden=true;root.append(input);input.onchange=()=>addBrowserFiles(input.files);
      root.querySelector('#session-shared-role').onchange=e=>{state.roleId=e.target.value;};
      root.querySelector('#session-shared-browser').onclick=()=>{input.value='';input.click();};
      root.querySelector('#session-shared-native').onclick=async()=>{const res=nativeApi()?.choose_paths?await nativeApi().choose_paths('file','Select Session FaceMapper ZIPs',[['FaceMapper ZIP','*.zip']],true):null;if(res?.paths)addPaths(res.paths);};
      root.querySelector('#session-shared-folder').onclick=async()=>{const path=await choosePath('directory','Select a Session FaceMapper folder');if(path)addPaths([path]);};
      const drop=root.querySelector('#session-shared-modal-drop');drop.ondragover=e=>{e.preventDefault();};drop.ondrop=e=>{e.preventDefault();addBrowserFiles(e.dataTransfer.files);};
      root.querySelector('#session-shared-import-run').onclick=run;
      if(files)addBrowserFiles(files);updateModal();
    });
  }

  async function promote(){
    const candidates=allAssets(['face_mapper']).filter(asset=>scopeOf(asset)!=='session_shared');
    if(!candidates.length){toast('No legacy FaceMapper source','Import a Session FaceMapper package or select an older Interaction FaceMapper source first.',true);return;}
    modal('Promote existing FaceMapper to Session Shared Data',`<div class="field"><label>Source</label><select id="promote-fm-asset">${candidates.map(row=>`<option value="${esc(row.id)}">${esc(row.label)}</option>`).join('')}</select></div><div class="field"><label>Wearer role</label><select id="promote-fm-role"><option>R01</option><option>R02</option></select></div><p class="form-help">No raw bytes are moved or copied. OculoStudio changes the registry scope, indexes all recording sections and preserves the original provenance.</p><div class="form-actions"><button class="primary-button" id="promote-fm-run">Promote and index</button></div>`,(root,close)=>{root.querySelector('#promote-fm-role').value=activeRole();root.querySelector('#promote-fm-run').onclick=async()=>{try{await api(`/api/session-shared/${encodeURIComponent(selectedSessionId())}/promote/${encodeURIComponent(root.querySelector('#promote-fm-asset').value)}`,{method:'POST',body:{role_id:root.querySelector('#promote-fm-role').value,conflict_policy:'add_revision'}});close();await refreshState({quiet:false});toast('FaceMapper promoted','The source is now Session Shared and indexed by recording ID.');}catch(error){showError(error,'Could not promote FaceMapper');}};});
  }

  function bind(){
    if(appState.page!=='data')return;
    const choose=document.querySelector('#data-session-shared-choose-files'),append=document.querySelector('#data-session-shared-append'),folder=document.querySelector('#data-session-shared-choose-folder'),promoteButton=document.querySelector('#data-session-shared-promote'),openFolder=document.querySelector('#data-session-shared-open-folder'),drop=document.querySelector('#session-shared-data-drop-zone');
    if(choose)choose.onclick=()=>open();if(append)append.onclick=()=>open();if(folder)folder.onclick=async()=>{const path=await choosePath('directory','Select Session Shared Data folder');if(path){open();addPaths([path]);}};if(promoteButton)promoteButton.onclick=promote;if(openFolder)openFolder.onclick=()=>api('/api/open-path',{method:'POST',body:{path:workspace().paths?.root||sessionStoragePath('root')}}).catch(error=>showError(error,'Could not open Session folder'));
    if(drop){drop.onclick=()=>open();drop.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();open();}};drop.ondragover=e=>e.preventDefault();drop.ondrop=e=>{e.preventDefault();open(e.dataTransfer.files);};}
    document.querySelectorAll('.session-shared-open-source').forEach(button=>button.onclick=()=>api('/api/open-path',{method:'POST',body:{path:button.dataset.path}}).catch(error=>showError(error,'Could not reveal source')));
  }

  function installBind(){
    if(typeof bindPageActions!=='function'||bindPageActions.__sessionSharedWrapped)return;
    const original=bindPageActions;
    const wrapped=function(){original();bind();};wrapped.__sessionSharedWrapped=true;bindPageActions=wrapped;
  }

  function installInteractionAppend(){
    // Existing importer already supports repeated batches. Expose that fact explicitly.
    window.OculoInteractionAppend={open:()=>window.OculoImport?.requestOpen?.()};
  }

  function install(){installDataView();installBind();installInteractionAppend();}
  install();
  window.OculoSessionSharedImport={open,openWithFiles:files=>open(files),promote,workspace};
})();
