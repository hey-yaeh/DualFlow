/* v0.5.2 import controller. File pickers and drag/drop feed the SAME queue.
   Ordinary browsers do not reveal absolute File paths. Those files are copied
   in bounded chunks to the local project's imports directory, never guessed. */
(() => {
  'use strict';
  const state = {rows: [], busy: false, projectPending: false, pickerBusy: false, active: null, defaultParticipant: '', dragScope: ''};
  const accept = '.zip,.c3d,.npz,.csv,.tsv,.parquet,.feather,.mp4,.mov,.mkv,.avi,.m4v,.wav,.mp3,.flac,.json,.oculocal,.g3dcal';
  const filters = [['OculoStudio data', accept.split(',').map(e=>'*'+e).join(' ')], ['ZIP archives','*.zip'], ['All files','*.*']];
  let serial=0, dragDepth=0;
  const nativeApi = () => window.pywebview?.api;
  const pending = () => state.rows.filter(r=>['ready','failed','cancelled'].includes(r.status));
  const canImport = () => !!project() && !!selectedSessionId();
  const sleep = ms => new Promise(resolve=>setTimeout(resolve,ms));

  function rowName(path) {return String(path).replaceAll('\\','/').split('/').filter(Boolean).at(-1)||String(path);}
  function ensureProject() {
    if (canImport()) return true;
    state.projectPending=true;
    modal('Create or open a project first · 请先创建或打开项目', `<p>Imported assets need a project and session. Your selected files will remain in the import queue.</p>
      <p>导入文件需要项目。已经选择或拖入的文件会留在队列中；建立项目后继续。</p>
      <div class="form-actions"><button class="primary-button" id="import-create-project">Create project</button><button class="soft-button" id="import-open-project">Open project</button></div>`,(root)=>{
        root.querySelector('#import-create-project').onclick=()=>openCreateProject();
        root.querySelector('#import-open-project').onclick=()=>openOpenProject();
      });
    return false;
  }
  function projectReady() {
    if(state.projectPending && canImport()) {state.projectPending=false; requestOpen();}
  }
  function requestOpen() {
    if(!ensureProject()) return;
    const pageDefault=document.querySelector('#asset-participant')?.value?.trim();
    if(pageDefault!==undefined && pageDefault!==null && pageDefault!=='') state.defaultParticipant=pageDefault;
    navigate('data');
    modal('Import current Interaction files · 导入当前 Interaction 数据', `<div id="asset-import-ui">
      <p class="form-help" id="import-session-note"></p>
      <div id="import-drop-zone" class="import-drop-zone" tabindex="0" role="button" aria-label="Choose or drop ZIP files">
        <strong>Drop current-Interaction files here · 拖入当前 Interaction 数据</strong>
        <span>NRD / TSD / FM / body or ETM C3D / facial video / props. Calibration Set files use the separate Data-page importer.</span>
      </div>
      <div class="import-buttons"><button class="primary-button" id="import-choose-files">Choose ZIP / files…</button>
        <button class="soft-button" id="import-browser-files">Browser file picker…</button>
        <button class="soft-button" id="import-choose-folder">Choose folder…</button></div>
      <p class="import-storage-note">Desktop “Choose files” and pasted paths reference originals (no copy). Browser selection and drag/drop copy files to the selected Session's <code>sessions/S###/imports/</code> folder in 8 MiB pieces, on this computer only. 原文件不会修改。</p>
      <details class="import-path-details"><summary>Use local paths without copying · 粘贴路径（不复制）</summary>
        <label for="import-paths">One file/folder per line. Windows “Copy as path” quotes are accepted.</label>
        <textarea id="import-paths" rows="3" placeholder="C:\\Data\\S001_I05_R01_NRD_untrimmed.zip"></textarea>
        <button class="soft-button" id="import-add-paths">Add paths to queue</button>
      </details>
      <label class="import-participant-label" for="import-participant">Default participant ID for this batch (optional) <input id="import-participant" placeholder="P001" value="${esc(state.defaultParticipant||'')}"></label><p class="form-help">Each queued file can override or clear this value. ETM_A/ETM_B should normally stay blank until wearer resolution.</p>
      <div id="import-queue" class="import-queue" aria-live="polite"></div>
      <div id="import-feedback" class="import-feedback" role="status"></div>
      <div class="form-actions import-actions"><button class="soft-button" id="import-clear-finished">Clear finished</button>
        <button class="soft-button" id="import-cancel">Cancel remaining</button>
        <button class="primary-button" id="import-run">Import selected</button></div>
    </div>`,(root)=>{
      root.querySelector('.modal').classList.add('asset-import-modal');
      root.querySelector('#import-choose-files').onclick=()=>chooseFiles();
      root.querySelector('#import-browser-files').onclick=()=>chooseBrowserFiles();
      root.querySelector('#import-choose-folder').onclick=()=>chooseFolder();
      const drop=root.querySelector('#import-drop-zone');
      drop.onclick=()=>chooseFiles();
      drop.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();chooseFiles();}};
      root.querySelector('#import-add-paths').onclick=()=>{
        const el=root.querySelector('#import-paths');addPaths(el.value.split(/\r?\n/).filter(x=>x.trim()));el.value='';
      };
      root.querySelector('#import-run').onclick=()=>run();
      root.querySelector('#import-cancel').onclick=()=>cancelAll();
      root.querySelector('#import-clear-finished').onclick=()=>{
        state.rows=state.rows.filter(r=>!['done','cancelled'].includes(r.status));update();
      };
      update();
    });
  }
  function feedback(message,error=false) {
    const el=document.querySelector('#import-feedback');
    if(el){el.textContent=message;el.classList.toggle('error',error);}
    else toast(error?'Import issue':'Import',message,error);
  }
  function update() {
    const el=document.querySelector('#import-queue');if(!el)return;
    const note=document.querySelector('#import-session-note');
    if(note){const ctx=activeWorkflowContext();note.textContent=`Interaction scope · Project: ${project()?.name||'—'} / Session: ${session()?.label||'—'} / Interaction: ${ctx.interactionId||'not selected'} / Folder: ${sessionStoragePath('imports')||'legacy imports'}`;}
    el.innerHTML=state.rows.length?state.rows.map(r=>`<div class="import-file-row" data-import-row="${r.id}" data-status="${r.status}">
      <div class="import-row-head"><strong title="${esc(r.path||r.name)}">${esc(r.name)}</strong><span>${esc(r.status)}</span></div>
      <div class="import-row-meta">${r.mode==='local_reference'?'Reference original · 不复制':'Project copy · 项目副本'}${r.size!=null?' · '+fmtBytes(r.size):''}</div>
      <label class="import-row-identity">Participant ID <input data-import-participant="${r.id}" value="${esc(r.participant??state.defaultParticipant??'')}" placeholder="blank / P01 / P02" ${['copying','inspecting','done'].includes(r.status)?'disabled':''}></label>
      <progress value="${r.percent||0}" max="100"></progress>
      <div class="import-row-result ${r.status==='failed'?'error':''}">${esc(r.message||'Ready to import')}</div>
      ${!['copying','inspecting','done'].includes(r.status)?`<button class="row-button import-remove" data-remove-import="${r.id}" title="Remove from queue">×</button>`:''}</div>`).join(''):'<div class="empty-tree">No files selected. Choose files or drop ZIPs above.</div>';
    el.querySelectorAll('[data-remove-import]').forEach(b=>b.onclick=()=>{state.rows=state.rows.filter(r=>r.id!==b.dataset.removeImport);update();});
    el.querySelectorAll('[data-import-participant]').forEach(input=>input.oninput=()=>{const row=state.rows.find(r=>r.id===input.dataset.importParticipant);if(row)row.participant=input.value.trim();});
    const runBtn=document.querySelector('#import-run');
    if(runBtn){runBtn.disabled=state.busy||pending().length===0;runBtn.textContent=state.busy?'Importing…':`Import selected (${pending().length})`;}
    for(const id of ['import-choose-files','import-browser-files','import-choose-folder','import-add-paths','import-participant']){
      const b=document.getElementById(id);if(b)b.disabled=state.busy||state.pickerBusy;
    }
    const cancel=document.querySelector('#import-cancel');if(cancel)cancel.disabled=!state.busy;
  }
  function addPaths(paths, {label=null,participant=null}={}) {
    for(let path of paths){
      path=String(path).trim().replace(/^(["'])(.*)\1$/, '$2');if(!path)continue;
      if(state.rows.some(r=>r.path===path&&r.status!=='cancelled'))continue;
      state.rows.push({id:`import_${++serial}`,name:label||rowName(path),path,mode:'local_reference',status:'ready',percent:0,label,participant:participant??state.defaultParticipant??''});
    }
    update();
  }
  function addFiles(files) {
    const pageDefault=document.querySelector('#asset-participant')?.value?.trim();
    if(pageDefault)state.defaultParticipant=pageDefault;
    for(const file of Array.from(files||[])){
      if(!(file instanceof File))continue;
      const same=state.rows.some(r=>r.file&&r.name===file.name&&r.size===file.size&&r.file.lastModified===file.lastModified&&r.status!=='cancelled');
      if(same)continue;
      state.rows.push({id:`import_${++serial}`,name:file.name,file,size:file.size,mode:'project_copy',status:'ready',percent:0,participant:state.defaultParticipant||''});
    }
    requestOpen();
  }
  function chooseBrowserFiles() {
    if(state.busy)return;
    const input=document.getElementById('oculo-import-file-input');
    input.value='';input.click();
  }
  async function chooseFiles() {
    if(state.busy||state.pickerBusy)return;
    if(!nativeApi()?.choose_paths){chooseBrowserFiles();return;}
    state.pickerBusy=true;update();feedback('Opening the desktop file picker…');
    try {
      const res=await nativeApi().choose_paths('file','Select NRD, TSD and Face Mapper ZIPs',filters,true);
      if(res?.cancelled||!res?.paths?.length){feedback('Selection cancelled. No file was imported.');return;}
      addPaths(res.paths);feedback('Selected files added. Click Import selected to inspect and register.');
    }catch(error){
      feedback(`Desktop picker failed: ${error.message||error}. Use “Browser file picker” instead.`,true);
    }finally{state.pickerBusy=false;update();}
  }
  async function chooseFolder() {
    if(state.busy)return;
    const path=await choosePath('directory','Select an extracted recording folder');
    if(path)addPaths([path]);
  }
  async function waitForJob(row) {
    for(;;){
      const {job}=await api('/api/imports/jobs/'+row.jobId);
      row.message=job.error||job.stage||job.status;
      row.percent=row.mode==='project_copy'?80+20*(job.progress||0):100*(job.progress||0);update();
      if(job.status==='completed'){row.asset=job.result?.asset;row.reused=job.result?.reused;return;}
      if(job.status==='failed')throw new Error(job.error||'Asset inspection failed');
      if(job.status==='cancelled'){const e=new Error('Import cancelled');e.name='AbortError';throw e;}
      await sleep(400);
    }
  }
  async function copyAndInspect(row, projectId,sid,participant) {
    const result=await api('/api/imports/uploads',{method:'POST',body:{project_id:projectId,session_id:sid,filename:row.file.name,size:row.file.size}});
    row.uploadId=result.upload_id;row.status='copying';row.message='Copying to the local project…';row.percent=0;update();
    try{
      for(let offset=0;offset<row.file.size;){
        if(row.cancelRequested)throw new DOMException('Import cancelled','AbortError');
        row.controller=new AbortController();
        const blob=row.file.slice(offset,Math.min(offset+result.chunk_bytes,row.file.size));
        const response=await fetch(`/api/imports/uploads/${row.uploadId}?offset=${offset}`,{
          method:'PUT',body:blob,headers:{'Content-Type':'application/octet-stream'},signal:row.controller.signal,
        });
        const value=await response.json();
        if(!response.ok)throw new Error(value.message||value.detail||'Local file transfer failed');
        offset=value.offset;row.percent=80*offset/row.file.size;
        row.message=`Copied ${fmtBytes(offset)} / ${fmtBytes(row.file.size)} locally`;update();
      }
      row.controller=null;
      if(row.cancelRequested)throw new DOMException('Import cancelled','AbortError');
      const final=await api(`/api/imports/uploads/${row.uploadId}/finish`,{method:'POST',body:{project_id:projectId,participant_id:participant||null}});
      row.jobId=final.job.id;row.status='inspecting';row.message='Inspecting ZIP and registering streams…';update();
      if(row.cancelRequested)await api(`/api/jobs/${row.jobId}/cancel`,{method:'POST'});
      await waitForJob(row);
    }catch(error){
      if(row.uploadId&&!row.jobId){try{await api(`/api/imports/uploads/${row.uploadId}`,{method:'DELETE'});}catch(_){}}
      throw error;
    }
  }
  function inferredRole(name){const m=String(name||'').match(/(?:^|[^A-Za-z0-9])(R\d{1,6})(?![A-Za-z0-9])/i);return m?m[1].toUpperCase():'';}
  function confirmIdentityAssignments(rows, fallback){
    const assignments=rows.map(r=>({name:r.name,participant:String(r.participant??fallback??'').trim(),role:inferredRole(r.name),etm:/ETM[_ -]?[AB](?:\.|_|$)/i.test(r.name)}));
    const participantSet=new Set(assignments.map(x=>x.participant).filter(Boolean));
    const roleSet=new Set(assignments.map(x=>x.role).filter(Boolean));
    const risky=(participantSet.size===1 && roleSet.size>1) || assignments.some(x=>x.etm&&x.participant);
    if(!risky)return true;
    const lines=assignments.map(x=>`${x.name}: ${x.participant||'(blank)'}`).join('\n');
    return window.confirm(`Identity check: this batch contains mixed roles or ETM files.\n\n${lines}\n\nContinue with these assignments?`);
  }
  async function run() {
    if(state.busy||!ensureProject())return;
    const rows=pending();if(!rows.length)return;
    const p=project().id,sid=selectedSessionId();
    const participant=document.querySelector('#import-participant')?.value?.trim()||null;
    state.defaultParticipant=participant||'';
    if(!confirmIdentityAssignments(rows,participant))return;
    state.busy=true;for(const r of rows){r.cancelRequested=false;r.jobId=null;r.uploadId=null;}update();
    let ok=0,failed=0;
    try{
      for(const row of rows){
        if(row.cancelRequested){row.status='cancelled';row.message='Cancelled before import';update();continue;}
        state.active=row;row.status=row.mode==='project_copy'?'copying':'inspecting';row.message='Starting import…';row.percent=0;update();
        try{
          if(row.mode==='project_copy')await copyAndInspect(row,p,sid,(row.participant!==undefined?row.participant:participant)||null);
          else {
            const result=await api('/api/imports/paths',{method:'POST',body:{project_id:p,session_id:sid,paths:[row.path],label:row.label||null,participant_id:((row.participant!==undefined?row.participant:participant)||null),checksum:true}});
            row.jobId=result.jobs[0].id;
            if(row.cancelRequested)await api(`/api/jobs/${row.jobId}/cancel`,{method:'POST'});
            await waitForJob(row);
          }
          row.status='done';row.percent=100;row.message=`${row.reused?'Already registered':'Registered'}: ${row.asset?.kind||'asset'}${row.asset?.recording_id?' · '+row.asset.recording_id:''}`;ok++;
          // Release successful browser File objects; no raw contents are kept in JS.
          row.file=null;
        }catch(error){row.status=error.name==='AbortError'?'cancelled':'failed';row.message=error.message||String(error);failed+=row.status==='failed'?1:0;}
        finally{row.controller=null;state.active=null;update();}
        await refreshState({quiet:true,fullRender:false});
      }
    }finally{
      state.busy=false;update();await refreshState({quiet:false});update();
      feedback(`${ok} registered · ${failed} failed. ${failed?'See the error under each file; no failed file is counted as imported.':'Assets are available in Data and Clips → NRD video reference.'}`,failed>0);
    }
  }
  async function cancelAll() {
    for(const row of state.rows){
      if(['ready','copying','inspecting'].includes(row.status)){row.cancelRequested=true;if(row.status==='ready'){row.status='cancelled';row.message='Cancelled';}}
    }
    const row=state.active;
    if(row){row.controller?.abort();if(row.jobId){try{await api(`/api/jobs/${row.jobId}/cancel`,{method:'POST'});}catch(error){feedback(error.message,true);}}}
    update();
  }
  function queuePathFromForm() {
    if(!ensureProject())return;
    const path=document.querySelector('#asset-path')?.value?.trim();
    if(!path){requestOpen();return;}
    const label=document.querySelector('#asset-label')?.value?.trim();
    const participant=document.querySelector('#asset-participant')?.value?.trim();
    addPaths([path],{label,participant});requestOpen();
  }
  function bindPage() {
    const el=document.querySelector('#interaction-data-drop-zone');
    if(el){el.onclick=requestOpen;el.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();requestOpen();}};}
  }
  function mount() {
    const input=document.createElement('input');input.type='file';input.id='oculo-import-file-input';input.multiple=true;input.accept=accept;input.hidden=true;
    document.body.append(input);input.addEventListener('change',()=>{if(input.files?.length)addFiles(input.files);});
    input.addEventListener('cancel',()=>feedback('Selection cancelled. No file was imported.'));
    const overlay=document.createElement('div');overlay.id='import-drag-overlay';overlay.hidden=true;
    overlay.innerHTML='<strong>Drop to add to the import queue</strong><span>ZIP / NRD / TSD / Face Mapper · 松开鼠标后加入导入队列</span>';
    document.body.append(overlay);
    const hasFiles=e=>Array.from(e.dataTransfer?.types||[]).includes('Files');
    const pointInside=(element,x,y)=>{if(!element)return false;const rect=element.getBoundingClientRect();return x>=rect.left&&x<=rect.right&&y>=rect.top&&y<=rect.bottom;};
    const scopeAtPoint=(x,y)=>{
      if(appState?.page!=='data')return '';
      if(pointInside(document.querySelector('#calibration-data-drop-zone'),x,y))return 'calibration';
      if(pointInside(document.querySelector('#interaction-data-drop-zone'),x,y))return 'interaction';
      return '';
    };
    const updateOverlay=scope=>{const strong=overlay.querySelector('strong'),span=overlay.querySelector('span');if(scope==='calibration'){strong.textContent='Drop into Calibration Data';span.textContent='Files will be appended to the selected Project-level Calibration Set';}else if(scope==='interaction'){strong.textContent='Drop into Interaction Data';span.textContent='Files will be queued for the active Session / Interaction';}else{strong.textContent='Choose a Data block';span.textContent='Move over Calibration Data or Interaction Data before dropping';}};
    window.addEventListener('dragenter',e=>{if(!hasFiles(e))return;e.preventDefault();dragDepth++;state.dragScope=scopeAtPoint(e.clientX,e.clientY)||state.dragScope;updateOverlay(state.dragScope);overlay.hidden=false;},true);
    window.addEventListener('dragover',e=>{if(!hasFiles(e))return;e.preventDefault();const scope=scopeAtPoint(e.clientX,e.clientY);if(scope)state.dragScope=scope;updateOverlay(state.dragScope);if(e.dataTransfer)e.dataTransfer.dropEffect=state.busy?'none':state.dragScope?'copy':'none';},true);
    window.addEventListener('dragleave',e=>{if(!hasFiles(e))return;dragDepth=Math.max(0,dragDepth-1);if(!dragDepth){overlay.hidden=true;state.dragScope='';}},true);
    window.addEventListener('drop',e=>{
      if(!hasFiles(e))return;
      e.preventDefault();e.stopPropagation();const scope=scopeAtPoint(e.clientX,e.clientY)||state.dragScope;dragDepth=0;overlay.hidden=true;state.dragScope='';
      if(state.busy){feedback('An import is active. Wait or cancel before adding more files.',true);return;}
      const items=Array.from(e.dataTransfer.items||[]);
      if(items.some(item=>item.webkitGetAsEntry?.()?.isDirectory)){
        toast('Folder dropped','Use the Choose folder button in the intended Data block. Folder drag/drop is not supported; ordinary files can be dropped directly.',true);return;
      }
      if(scope==='calibration'){
        const setCode=(window.activeCalibrationSetRecord?.()||window.calibrationSetRows?.()?.[0])?.code||'';
        if(!window.OculoCalibrationImport?.openWithFiles){toast('Calibration importer unavailable','Reload OculoStudio and try again.',true);return;}
        window.OculoCalibrationImport.openWithFiles(e.dataTransfer.files,setCode);return;
      }
      if(scope==='interaction'){addFiles(e.dataTransfer.files);return;}
      toast('Choose an import scope','Drop files directly on Calibration Data or Interaction Data.',true);
    },true);
    window.addEventListener('blur',()=>{dragDepth=0;overlay.hidden=true;});
    window.addEventListener('beforeunload',e=>{if(state.busy){e.preventDefault();e.returnValue='An import is active';}});
  }
  window.OculoImport={requestOpen,projectReady,chooseFiles,chooseBrowserFiles,chooseFolder,queuePathFromForm,bindPage};
  document.addEventListener('DOMContentLoaded',mount);
})();
