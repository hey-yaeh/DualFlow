/* OculoStudio Workflow Hub.
 *
 * The default interface is organised by Project -> Session -> Interaction.
 * All scientific services remain unchanged and are still available in Expert Tools.
 */
(() => {
  const legacy = {
    navigate,
    renderTabs,
    renderSidebar,
    renderWorkspace,
    renderProperties,
    bindGlobalActions,
    bindPageActions,
  };

  const GUIDED_PIPELINE = [
    {id:'project', label:'Home', icon:'⌂', desc:'Clean-project overview and the next required action'},
    {id:'interactions', label:'Interactions', icon:'◫', desc:'Process R01 and R02 through one ordered interaction workflow'},
    {id:'review', label:'Review', icon:'▶', desc:'Review privacy, gaze, AOI and dyadic results'},
    {id:'qc', label:'Results', icon:'✓', desc:'Browse and export processed results'},
  ];
  const EXPERT_PAGE_IDS = new Set(PIPELINE.map(row=>row.id));
  const GUIDED_PAGE_IDS = new Set(GUIDED_PIPELINE.map(row=>row.id));
  const storedMode=(()=>{try{return localStorage.getItem('oculo.uiMode')}catch(_){return ''}})();
  const state = {
    mode: storedMode === 'expert' ? 'expert' : 'guided',
    pendingStep: '',
    activeStage: (()=>{try{return localStorage.getItem('oculo.workflowStage')||''}catch(_){return ''}})(),
    reviewLaunch: null,
    calibrationView: (()=>{try{const value=localStorage.getItem('oculo.calibrationView')||'paired';return ['paired','R01','R02'].includes(value)?value:'paired'}catch(_){return 'paired'}})(),
  };

  const STAGE_DEFS = [
    {id:'source', label:'1 · Data', short:'Data', desc:'Import and validate the original R01/R02 NRD, TSD, FaceMapper and Vicon sources.'},
    {id:'calibration', label:'2 · Calibration & Clips', short:'Calibration & Clips', desc:'Resolve the physical ETM, verify Pupil↔Vicon timing and derive both roles’ clips from the common body-C3D interval.'},
    {id:'face', label:'3 · Privacy', short:'Privacy', desc:'Review the automatic intervals, filter FaceMapper and create the final synchronized privacy video.'},
    {id:'scene', label:'4 · Gaze & Targets', short:'Gaze & Targets', desc:'Generate calibrated Gaze 3D and create Face/Body/Object targets.'},
    {id:'dyadic', label:'5 · Dyadic Attention', short:'Dyadic Attention', desc:'Generate mutual gaze and joint-attention episodes on the shared physical interval.'},
  ];

  function workspaceData(){return appState.data.interaction_workspace || {interactions:[],unassigned:[],counts:{}};}
  function projectData(){return appState.data.project_workspace || {sessions:[],counts:{},calibration_profiles:[]};}
  function interactions(){return workspaceData().interactions || [];}
  function context(){
    const data=workspaceData();
    return {
      interactionId:data.selected_interaction_id || interactions()[0]?.interaction_id || '',
      roleId:data.selected_role_id || 'R01',
      workflow:data.selected_workflow || 'face',
    };
  }
  function selectedInteraction(){const id=context().interactionId;return interactions().find(row=>row.interaction_id===id)||interactions()[0]||null;}
  function roleInfo(roleId=context().roleId){return selectedInteraction()?.roles?.[roleId]||null;}
  function focusRoleSource(interactionId,roleId){
    const rows=Object.values(session()?.assets||{}).filter(asset=>{
      const identity=assetIdentity(asset);
      return identity.interaction_id===interactionId && identity.role_id===roleId;
    });
    const priority=['pupil_native','pupil_cloud','face_mapper','c3d'];
    rows.sort((a,b)=>priority.indexOf(a.kind)-priority.indexOf(b.kind));
    const selected=rows[0]||null;
    appState.selectedAssetId=selected?.id||null;
    appState.selectedAssetIds=selected?[selected.id]:[];
  }
  function stage(info,name){return info?.stages?.[name]||{state:'missing',count:0,labels:[]};}
  function isDone(info,name){return stage(info,name).state==='complete';}
  function sharedStage(name){return selectedInteraction()?.shared?.stages?.[name]||{state:'missing',count:0,labels:[]};}
  function modePipeline(){return state.mode==='guided'?GUIDED_PIPELINE:PIPELINE;}
  function setBodyMode(){document.body.classList.toggle('guided-mode',state.mode==='guided');document.body.classList.toggle('expert-mode',state.mode==='expert');}
  function hardwareProfiles(){return artifacts(['calibration_profile']).sort(byCreated).reverse();}
  function acceptanceArtifacts(){return artifacts(['calibration_acceptance']).sort(byCreated).reverse();}
  function sharedAudioReadyForRole(roleId){
    const role=String(roleId||'').toUpperCase(),iid=String(context().interactionId||'').toLowerCase();
    return artifacts(['time_sync']).some(artifact=>{
      if(String(artifact.parameters?.sync_type||artifact.qc?.sync_type||'')!=='shared_audio_video')return false;
      if(!Boolean(artifact.parameters?.visual_review_confirmed||artifact.qc?.visual_review_confirmed))return false;
      const artifactRole=String(artifact.parameters?.wearer_role_id||artifact.qc?.wearer_role_id||'').toUpperCase();
      if(role&&artifactRole&&artifactRole!==role)return false;
      const text=recordText(artifact);return !iid||!(/i\d+/i.test(text))||text.includes(iid);
    });
  }

  function recordText(record){
    const id=record?.id;const asset=session()?.assets?.[id];const artifact=session()?.artifacts?.[id];
    const row=asset||artifact||record||{};
    return JSON.stringify(row).toLowerCase();
  }
  function latestRoleTrim(interactionId,roleId){
    const iid=String(interactionId||'').toLowerCase(),rid=String(roleId||'').toLowerCase();
    return artifacts(['native_reference_trim']).filter(artifact=>{
      const text=recordText(artifact);
      return (!iid||text.includes(iid))&&(!rid||text.includes(rid));
    }).sort(byCreated).slice(-1)[0]||null;
  }
  function latestRoleArtifact(kinds,interactionId,roleId){
    const iid=String(interactionId||'').toLowerCase(),rid=String(roleId||'').toLowerCase();
    const trim=latestRoleTrim(interactionId,roleId);
    return artifacts(kinds).filter(artifact=>{
      const text=recordText(artifact);
      if((iid&&!text.includes(iid))||(rid&&!text.includes(rid)))return false;
      return !trim||typeof nativeTrimIdForRecord!=='function'||nativeTrimIdForRecord(artifact)===trim.id;
    }).sort(byCreated).slice(-1)[0]||null;
  }
  function rowMatchesContext(record,{role=true}={}){
    const ctx=context(); if(!ctx.interactionId)return true;
    const id=record?.id;const source=session()?.assets?.[id]||session()?.artifacts?.[id]||record||{};
    const text=recordText(record);const hasInteraction=/i\d+/i.test(text);
    const reusable=['face_mapper','calibration','calibration_profile'].includes(String(source.kind||''));
    if(hasInteraction&&!text.includes(ctx.interactionId.toLowerCase()))return false;
    if(!hasInteraction&&!reusable)return false;
    if(role && ctx.roleId && /r\d+/i.test(text) && !text.includes(ctx.roleId.toLowerCase()))return false;
    return true;
  }
  function statusBadge(done,label=''){
    return `<span class="iw-status ${done?'done':'missing'}"><i>${done?'✓':'○'}</i>${esc(label|| (done?'Ready':'Not run'))}</span>`;
  }
  function stateChip(value,label){
    const done=value==='complete'||value===true;
    const warn=value==='warning'||value==='review';
    return `<span class="wh-chip ${done?'done':warn?'warning':'missing'}"><i>${done?'✓':warn?'!':'○'}</i>${esc(label)}</span>`;
  }
  function progressText(done,total){return `${done}/${total}`;}
  function participantForRole(row,roleId){
    const role=row?.roles?.[roleId]||{};
    return role.participant_id || role.participant_ids?.[0] || (roleId==='R01'?'P01':roleId==='R02'?'P02':'');
  }

  function rolePhaseStatus(row,roleId,phase){
    const role=row?.roles?.[roleId]||{};
    if(phase==='source'){
      const names=['native','timeseries','facemapper'];
      const complete=names.filter(name=>isDone(role,name)).length;
      return {complete,total:names.length,done:complete===names.length};
    }
    if(phase==='face'){
      const names=['face_filter','privacy_clip'];
      const complete=names.filter(name=>isDone(role,name)).length;
      return {complete,total:names.length,done:complete===names.length};
    }
    if(phase==='scene'){
      const names=['gaze3d','participant_association','face_aoi','body_aoi','fusion'];
      const complete=names.filter(name=>isDone(role,name)).length;
      return {complete,total:names.length,done:complete===names.length};
    }
    return {complete:0,total:1,done:row?.summary?.dyadic_ready===true};
  }
  function sharedSceneReady(){
    return sharedStage('etm_assignment').state==='complete';
  }
  function interactionOverall(row){
    const phases=['source','face','scene'];
    let complete=0,total=0;
    for(const roleId of ['R01','R02'])for(const phase of phases){const x=rolePhaseStatus(row,roleId,phase);complete+=x.complete;total+=x.total;}
    total+=1;{const cal=calibrationWorkflowState();if(cal.allRolesReady&&cal.alignedClipsReady)complete+=1;}
    if(row?.summary?.dyadic_ready)complete+=1;total+=1;
    return {complete,total,ratio:total?complete/total:0};
  }
  function nextRequiredStep(row=selectedInteraction()){
    if(!row)return {label:'Import source data',page:'data',roleId:'R01',stage:'source'};

    // Advance both roles through the same dependency before moving to the next
    // stage. This makes the workflow predictable: after R01 retained clip, the
    // next action is R02 retained clip—not R01 privacy.
    const sourceChecks=[
      ['native','Import NRD source','data','source'],
      ['timeseries','Import TSD source','data','source'],
      ['facemapper','Import Face Mapper source','data','source'],
    ];
    for(const [key,label,page,phase] of sourceChecks){
      for(const roleId of ['R01','R02']){
        const role=row.roles?.[roleId]||{};
        if(!isDone(role,key))return {label:`${roleId}: ${label}`,page,roleId,stage:phase};
      }
    }
    const calibrationState=calibrationWorkflowState();
    if(!calibrationState.allRolesReady)return {label:'Complete Calibration: profiles, wearer and verified clock sync',page:'calibration',roleId:'R01',stage:'calibration'};
    if(!calibrationState.alignedClipsReady)return {label:'Create automatic R01 + R02 clips from the common body-C3D interval',page:'calibration',roleId:'R01',stage:'calibration'};
    const downstreamChecks=[
      ['trim','Review C3D-defined aligned clip','clips','source'],
      ['face_filter','Filter FaceMapper candidates','privacy','face'],
      ['privacy_clip','Create synchronized privacy video','privacy','face'],
    ];
    for(const [key,label,page,phase] of downstreamChecks){
      for(const roleId of ['R01','R02']){
        const role=row.roles?.[roleId]||{};
        if(!isDone(role,key))return {label:`${roleId}: ${label}`,page,roleId,stage:phase};
      }
    }
    for(const roleId of ['R01','R02']){
      const role=row.roles?.[roleId]||{};
      const checks3=[
        ['gaze3d','Generate trim-scoped Gaze 3D','gaze3d'],
        ['participant_association','Associate partner face','targets'],
        ['face_aoi','Generate partner-aware Face AOI','targets'],
        ['body_aoi','Generate Body AOI','targets'],
        ['fusion','Fuse target evidence','targets'],
      ];
      for(const [key,label,page] of checks3)if(!isDone(role,key))return {label:`${roleId}: ${label}`,page,roleId,stage:'scene'};
    }
    if(!row.summary?.dyadic_ready)return {label:'Generate mutual gaze and joint attention',page:'dyadic',roleId:'R01',stage:'dyadic'};
    return {label:'Review completed interaction',page:'review',roleId:'R01',stage:'dyadic'};
  }

  async function saveContext(interactionId, roleId=context().roleId, workflow=context().workflow, {renderAfter=true}={}){
    if(!selectedSessionId()||!interactionId)return;
    try{
      const result=await api(`/api/interactions/${encodeURIComponent(selectedSessionId())}/context`,{method:'POST',body:{interaction_id:interactionId,role_id:roleId,workflow}});
      appState.data.interaction_workspace=result;
      if(renderAfter)render();
    }catch(error){showError(error,'Could not select interaction context');}
  }

  function setStage(stageId,{renderAfter=true}={}){
    state.activeStage=stageId;try{localStorage.setItem('oculo.workflowStage',stageId)}catch(_){}
    const workflow=stageId==='scene'?'scene':stageId==='dyadic'?'dyadic':stageId==='calibration'?'scene':'face';
    const c=context();
    if(c.interactionId)saveContext(c.interactionId,c.roleId,workflow,{renderAfter});else if(renderAfter)render();
  }

  function setMode(mode,{page=null}={}){
    state.mode=mode==='expert'?'expert':'guided';try{localStorage.setItem('oculo.uiMode',state.mode)}catch(_){}setBodyMode();
    if(page)appState.page=page;
    else if(state.mode==='guided'&&!GUIDED_PAGE_IDS.has(appState.page))appState.page='interactions';
    else if(state.mode==='expert'&&!EXPERT_PAGE_IDS.has(appState.page))appState.page='project';
    render();
  }

  function setCalibrationView(value,{renderAfter=true}={}){
    const next=['paired','R01','R02'].includes(value)?value:'paired';
    if(typeof captureCalibrationDraft==='function'&&appState.page==='calibration'&&state.calibrationView!=='paired'){
      try{captureCalibrationDraft()}catch(_){}
    }
    state.calibrationView=next;
    try{localStorage.setItem('oculo.calibrationView',next)}catch(_){}
    if(renderAfter)render();
  }
  function calibrationView(){return state.calibrationView;}
  function calibrationRoleOptionLabel(roleId){
    try{
      const stage=calibrationWorkflowState(roleId);
      const row=stage.rows?.find(item=>item.role===roleId);
      const etm=row?.wearerMapping?.etm_id||row?.mapping?.etm_id||'';
      return etm?`${roleId} details · ${etm}`:`${roleId} details`;
    }catch(_){return `${roleId} details`;}
  }

  async function openStep(page,{roleId=context().roleId,workflow=context().workflow,special=''}={}){
    const interactionId=context().interactionId;
    if(interactionId)await saveContext(interactionId,roleId,workflow,{renderAfter:false});
    state.pendingStep=special||page;
    if(special==='motionstudio'){
      if(window.OculoMotionStudioBridge){window.OculoMotionStudioBridge.open();return;}
      toast('MotionStudio bridge unavailable','The bridge JavaScript did not load.',true);return;
    }
    setMode('expert',{page});
    setTimeout(()=>applyContextDefaults(),0);
  }

  navigate = function(page){
    const allowed=state.mode==='guided'?GUIDED_PAGE_IDS:new Set([...EXPERT_PAGE_IDS,'motionstudio']);
    if(!allowed.has(page)){
      if(state.mode==='guided'&&EXPERT_PAGE_IDS.has(page)){openStep(page);return;}
      return;
    }
    appState.page=page;render();
  };

  renderTabs = function(){
    const rows=modePipeline().filter(item=>state.mode==='guided'||!item.hidden);
    if(state.mode==='guided'){
      $('#pipeline-tabs').innerHTML=`<span class="pipeline-primary-tabs guided-tabs">${rows.map(item=>`<button type="button" class="pipeline-tab ${appState.page===item.id?'active':''}" data-page="${item.id}"><span class="tab-icon">${item.icon}</span><span class="tab-label">${esc(item.label)}</span></button>`).join('')}</span>`;
    }else{
      const primary=rows.filter(item=>!item.utility);
      const utility=rows.filter(item=>item.utility);
      const button=item=>`<button type="button" class="pipeline-tab ${item.optional?'optional-tab ':''}${item.utility?'utility-tab ':''}${appState.page===item.id?'active':''}" data-page="${item.id}"><span class="tab-icon">${item.icon}</span><span class="tab-label">${esc(item.label)}</span>${item.optional?'<span class="tab-badge">Optional</span>':''}</button>`;
      $('#pipeline-tabs').innerHTML=`<span class="pipeline-primary-tabs">${primary.map(button).join('')}</span><span class="pipeline-utility-tabs">${utility.map(button).join('')}</span>`;
    }
    $$('.pipeline-tab').forEach(button=>button.onclick=()=>navigate(button.dataset.page));
    renderOverviewNav();
  };

  function renderOverviewNav(){
    const root=$('#overview-nav');if(!root)return;
    const rows=state.mode==='guided'?
      [['project','⌂','Home'],['interactions','◫','Interactions'],['review','▶','Review'],['qc','✓','Results']]:
      [['project','⌂','Dashboard'],['data','▤','Assets'],['calibration','◎','Calibration'],['clips','▦','Trim'],['privacy','◉','Privacy'],['gaze3d','✣','Gaze 3D'],['targets','⌾','Targets'],['dyadic','∞','Dyadic'],['qc','✓','Results'],['advanced','⋯','Advanced']];
    root.innerHTML=rows.map(([id,icon,label])=>`<button class="side-link ${appState.page===id?'active':''}" data-side-page="${id}"><span>${icon}</span>${esc(label)}</button>`).join('');
    $$('.side-link',root).forEach(button=>button.onclick=()=>navigate(button.dataset.sidePage));
  }

  function guidedSidebarSummary(){
    const row=selectedInteraction();const ctx=context();const ws=workspaceData();
    const r01=row?.roles?.R01||{},r02=row?.roles?.R02||{};
    const next=nextRequiredStep(row);
    return `<div class="wh-sidebar-summary">
      <div class="wh-side-context"><span>Current context</span><strong>${esc(ws.session_code||session()?.label||'')} / ${esc(ctx.interactionId||'No interaction')}</strong><small>${esc(ctx.roleId)} · ${esc(participantForRole(row,ctx.roleId)||'participant unassigned')}</small></div>
      <div class="wh-side-mini"><b>R01</b><span>Face ${r01.face_progress||0}/${r01.face_total||6}</span><span>3D ${r01.scene_progress||0}/${r01.scene_total||5}</span></div>
      <div class="wh-side-mini"><b>R02</b><span>Face ${r02.face_progress||0}/${r02.face_total||6}</span><span>3D ${r02.scene_progress||0}/${r02.scene_total||5}</span></div>
      <button class="wh-next-mini" data-iw-next="true"><span>Next required</span><strong>${esc(next.label)}</strong></button>
      ${ws.counts?.unassigned?`<button class="wh-unassigned" data-iw-go="data">${ws.counts.unassigned} unassigned records</button>`:''}
    </div>`;
  }

  renderSidebar = function(){
    legacy.renderSidebar();renderOverviewNav();
    const sidebar=$('.sidebar-scroll');
    sidebar?.querySelector('.iw-sidebar-context')?.remove();
    if(project()&&state.mode==='guided'){
      const marker=document.createElement('div');marker.className='iw-sidebar-context wh-context';marker.innerHTML=guidedSidebarSummary();sidebar?.prepend(marker);
      const assetTree=$('#asset-tree');if(assetTree){assetTree.innerHTML=`<button class="wh-tree-summary" data-iw-go="data"><span>Source assets</span><strong>${workspaceData().counts?.assets||0}</strong><small>Manage in Expert Tools</small></button>`;}
      const artifactTree=$('#artifact-tree');if(artifactTree){artifactTree.innerHTML=`<button class="wh-tree-summary" data-iw-go="qc"><span>Processed results</span><strong>${workspaceData().counts?.artifacts||0}</strong><small>Browse by interaction</small></button>`;}
      $('#asset-refresh-all')?.classList.add('guided-hidden');
    }
  };

  function interactionCard(row){
    const roles=row.roles||{};const r01=roles.R01||{};const r02=roles.R02||{};const selected=row.interaction_id===context().interactionId;const overall=interactionOverall(row);
    const status=row.summary?.dyadic_ready?'Complete':overall.complete?'In progress':'Not started';
    return `<button class="iw-interaction-card wh-interaction-card ${selected?'selected':''}" data-iw-interaction="${esc(row.interaction_id)}">
      <div class="iw-interaction-head"><span class="iw-interaction-id">${esc(row.interaction_id)}</span><span class="wh-state-label ${status.replaceAll(' ','-').toLowerCase()}">${esc(status)}</span></div>
      <div class="wh-progress"><i style="width:${Math.round(overall.ratio*100)}%"></i></div>
      <div class="iw-role-progress"><span>R01 · ${esc(participantForRole(row,'R01')||'—')}</span><b>Face ${progressText(r01.face_progress||0,r01.face_total||6)} · 3D ${progressText(r01.scene_progress||0,r01.scene_total||5)}</b></div>
      <div class="iw-role-progress"><span>R02 · ${esc(participantForRole(row,'R02')||'—')}</span><b>Face ${progressText(r02.face_progress||0,r02.face_total||6)} · 3D ${progressText(r02.scene_progress||0,r02.scene_total||5)}</b></div>
      <div class="iw-interaction-foot"><span>${row.summary?.record_count||0} linked records</span><span>${selected?'Continue →':'Open →'}</span></div>
    </button>`;
  }

  function guidedProjectView(){
    if(!project())return welcomeView();
    const p=project(),ws=workspaceData(),pws=projectData(),row=selectedInteraction(),next=nextRequiredStep(row),ready=interactions().filter(x=>x.summary?.dyadic_ready).length;
    const sessionCards=(pws.sessions||[]).map(item=>`<article class="wh-session-card ${item.selected?'selected':''}"><header><div><span>${esc(item.session_code)}</span><strong>${esc(item.label)}</strong></div><span class="wh-state-label ${item.selected?'complete':'not-started'}">${item.selected?'Current':'Session'}</span></header><div class="wh-session-stat-grid"><div><b>${item.interaction_count}</b><span>Interactions</span></div><div><b>${item.asset_count}</b><span>Assets</span></div><div><b>${item.artifact_count}</b><span>Results</span></div></div><code>${esc(item.folder)}</code><div class="form-actions"><button class="${item.selected?'soft-button':'primary-button'} compact" data-project-session-select="${esc(item.session_id)}">${item.selected?'Open':'Switch to Session'}</button><button class="soft-button compact" data-project-folder="${esc(item.folder)}">Folder</button></div></article>`).join('');
    return viewShell('Project home',p.name,`
      <section class="wh-home-hero clean-guided-hero"><div><div class="eyebrow">${pws.counts?.sessions||Object.keys(p.sessions||{}).length} Sessions · Project-level folder map</div><h2>One Project for S004–S007, with each Session and Interaction kept explicit.</h2><p>Session folders separate imports, derived outputs, reports and annotations. ETM hardware profiles can be shared through the Project library, while every Interaction retains independent wearer resolution, offset, drift and Trim.</p></div><div class="wh-home-next"><span>Next required action · ${esc(ws.session_code||session()?.label||'')}</span><strong>${esc(next.label)}</strong><button class="primary-button" data-iw-next="true">Continue</button></div></section>
      <div class="iw-summary-grid wh-summary-grid"><article><span>Sessions</span><strong>${pws.counts?.sessions||0}</strong><small>One Project workspace</small></article><article><span>Interactions</span><strong>${pws.counts?.interactions||0}</strong><small>${ready} complete in current Session</small></article><article><span>Shared Calibration</span><strong>${pws.counts?.calibration_profiles||0}</strong><small>Profile records across Sessions</small></article><article><span>Results</span><strong>${pws.counts?.artifacts||0}</strong><small>Immutable processing records</small></article></div>
      <section class="iw-section"><div class="iw-section-head"><div><div class="eyebrow">Project structure</div><h2>Sessions</h2></div><div class="wh-head-actions"><button class="soft-button" data-project-open-root="true">Open Project folder</button><button class="primary-button" data-project-manage="true">Manage Sessions</button></div></div><div class="wh-session-grid">${sessionCards||'<div class="iw-empty"><strong>No Session exists.</strong></div>'}</div></section>
      <section class="iw-section"><div class="iw-section-head"><div><div class="eyebrow">Selected Session</div><h2>${esc(ws.session_code||session()?.label||'Session')} interactions</h2></div><div class="wh-head-actions"><button class="soft-button" data-iw-go="calibration">Calibration setup</button><button class="soft-button" data-iw-go="data">Manage source data</button><button class="soft-button" data-project-manage="true">＋ Interaction plan</button></div></div><div class="iw-interaction-grid">${interactions().length?interactions().map(interactionCard).join(''):'<div class="iw-empty"><strong>No interaction planned or detected.</strong><p>Add I01/I02… in Session Manager or import corrected files containing S/I/R identifiers.</p><button class="primary-button" data-project-manage="true">Plan Interactions</button></div>'}</div></section>`);
  }

  function stageStatus(row,stageId){
    if(!row)return {done:false,partial:false,label:'Not started'};
    if(stageId==='dyadic')return {done:row.summary?.dyadic_ready===true,partial:false,label:row.summary?.dyadic_ready?'Complete':'Locked'};
    if(stageId==='calibration'){
      const cal=calibrationWorkflowState();
      const partial=cal.profiles.length>0||Boolean(cal.assignment);
      const done=cal.allRolesReady&&cal.alignedClipsReady;
      return {done,partial:partial||cal.allRolesReady||cal.alignedClipsReady,label:done?'Complete':cal.allRolesReady?'Create aligned clips':partial?'In progress':'Not started'};
    }
    const a=rolePhaseStatus(row,'R01',stageId),b=rolePhaseStatus(row,'R02',stageId);
    const done=a.done&&b.done;
    const partial=(a.complete+b.complete)>0;
    return {done,partial,label:done?'Complete':partial?'In progress':'Not started'};
  }

  function sourceStage(row){
    const roleCard=roleId=>{
      const role=row.roles?.[roleId]||{};const participant=participantForRole(row,roleId)||'Unassigned';
      const items=[['NRD scene recording','native','data'],['TSD eye data','timeseries','data'],['FaceMapper','facemapper','data']];
      return `<article class="wh-role-column"><header><div><span>${esc(roleId)}</span><strong>${esc(participant)}</strong></div>${stateChip(rolePhaseStatus(row,roleId,'source').done,'Data ready')}</header><div class="wh-check-list">${items.map(([label,key,page])=>stepLine(label,isDone(role,key),page,roleId,'face',stage(role,key).labels?.[0])).join('')}</div></article>`;
    };
    return `<div class="wh-stage-layout"><section><div class="wh-stage-intro"><div><div class="eyebrow">Stage 1</div><h2>Original source data</h2><p>Import the original NRD, TSD and FaceMapper exports for both roles together with the correct body and ETM C3D files. Do not type clip start/end times in the normal workflow.</p></div><button class="soft-button" data-iw-go="data">Manage corrected assets</button></div><div class="wh-role-grid">${roleCard('R01')}${roleCard('R02')}</div></section><aside class="wh-stage-aside"><strong>Required before Calibration</strong><span>NRD, TSD and FaceMapper for both roles</span><span>Correct R01/R02 body C3D</span><span>Correct full-length ETM-A/ETM-B C3D</span><small>The common body-C3D interval is converted into Pupil time only after the verified TimeGraph exists.</small></aside></div>`;
  }

  function faceStage(row){
    const roleCard=roleId=>{
      const role=row.roles?.[roleId]||{};
      const items=[
        ['FaceMapper QC & Filter',isDone(role,'face_filter'),'privacy',stage(role,'face_filter').labels?.[0]||'Filter false-positive candidates before masking'],
        ['Final synchronized Privacy video',isDone(role,'privacy_clip'),'privacy',stage(role,'privacy_clip').labels?.[0]],
      ];
      return `<article class="wh-role-column"><header><div><span>${esc(roleId)}</span><strong>${esc(participantForRole(row,roleId)||'Participant')}</strong></div>${stateChip(rolePhaseStatus(row,roleId,'face').done,'Privacy ready')}</header><div class="wh-check-list">${items.map(([label,done,page,detail])=>stepLine(label,done,page,roleId,'face',detail)).join('')}</div></article>`;
    };
    return `<div class="wh-stage-layout"><section><div class="wh-stage-intro"><div><div class="eyebrow">Stage 3</div><h2>Privacy</h2><p>Review the automatically derived clip boundaries, then run FaceMapper quality control and create the final privacy video once. Face, body and partner identity AOIs use the same immutable physical interval.</p></div><button class="soft-button" data-iw-go="review">Open privacy review</button></div><div class="wh-role-grid">${roleCard('R01')}${roleCard('R02')}</div></section><aside class="wh-stage-aside"><strong>One-pass output</strong><span>Filtered FaceMapper</span><span>C3D-defined trim-aware masking and target FPS</span><span>Audio and frame-map provenance</span><small>No second “trim blurred video” step in the normal workflow.</small></aside></div>`;
  }

  function calibrationStage(row){
    const cal=calibrationWorkflowState();
    const facialCards=['R01','R02'].map(roleId=>{
      const role=row.roles?.[roleId]||{};const syncReady=sharedAudioReadyForRole(roleId),exportReady=isDone(role,'aligned_facial_video');
      return `<article class="wh-role-column"><header><div><span>${esc(roleId)} facial video · optional</span><strong>${esc(participantForRole(row,roleId)||'Participant')}</strong></div>${stateChip(exportReady,exportReady?'Aligned FV ready':syncReady?'Ready to export':'Sync optional')}</header><div class="wh-check-list">${stepLine('Shared-audio FV ↔ ego TimeGraph',syncReady,'calibration',roleId,'calibration',syncReady?'Visual-verified mapping':'Run only when an FV source was recorded')}${stepLine('Export drift-corrected aligned facial video',exportReady,'clips',roleId,'calibration',exportReady?'Clean MP4 + per-frame lineage':syncReady?'Uses the automatic C3D-defined interval':'Available after accepted shared-audio sync')}</div></article>`;
    }).join('');
    return `<div class="wh-stage-layout"><section><div class="wh-stage-intro"><div><div class="eyebrow">Stage 2</div><h2>Calibration, clock sync & automatic clips</h2><p>Resolve R01/R02 to ETM-A/ETM-B, verify the Pupil↔Vicon affine TimeGraph, then convert the shared R01/R02 body-C3D interval into exact NRD/TSD/FaceMapper intervals for both roles. Optional facial-expression videos can join the same TimeGraph through distributed shared-audio matching and be exported on exactly the same physical interval.</p></div><button class="primary-button" data-iw-go="calibration">Open Calibration</button></div><article class="wh-shared-card"><header><div><span>Interaction calibration and clip handoff</span><strong>${esc(row.scene_id||`${workspaceData().session_code}_${row.interaction_id}`)}</strong></div>${stateChip(cal.allRolesReady&&cal.alignedClipsReady,'Calibration + clips ready')}</header><div class="calibration-readiness-grid compact">${calibrationReadinessRows(cal)}</div></article><div class="wh-role-grid"><article class="wh-role-column"><header><div><span>Verified calibration</span><strong>${cal.profiles.length} profile(s)</strong></div>${stateChip(cal.allRolesReady,'TimeGraph ready')}</header><div class="wh-check-list">${stepLine('Profiles, wearer and clock synchronization',cal.allRolesReady,'calibration','R01','calibration',cal.assignment?.qc?.status||'Not run')}</div></article><article class="wh-role-column"><header><div><span>C3D-defined clips</span><strong>${Object.values(cal.alignedRoleTrims||{}).filter(Boolean).length}/2</strong></div>${stateChip(cal.alignedClipsReady,'Aligned clips')}</header><div class="wh-check-list">${stepLine('Create paired R01/R02 clips from common body-C3D overlap',cal.alignedClipsReady,'calibration','R01','calibration',cal.alignedClipsReady?'Published transactionally':'No manual timecodes required')}</div></article></div><div class="wh-role-grid" style="margin-top:12px">${facialCards}</div></section><aside class="wh-stage-aside"><strong>Required before Privacy and Gaze 3D</strong><span>Compatible hardware profile for each ETM</span><span>Verified offset + drift for both Pupil clocks</span><span>Two paired automatic clip artifacts</span><strong style="margin-top:10px">Optional facial-expression path</strong><span>Visual-verified shared-audio FV ↔ ego mapping</span><span>Clean drift-corrected aligned FV export</span><small>Facial export reuses the accepted TimeGraph and automatic clip interval; it never asks for another start/end time.</small></aside></div>`;
  }

  function sceneStage(row){
    const roleCard=roleId=>{
      const role=row.roles?.[roleId]||{};
      const participant=participantForRole(row,roleId)||'Participant';
      const faceArtifact=latestRoleArtifact(['face_aoi'],row.interaction_id,roleId);
      const reviewFraction=Number(faceArtifact?.qc?.review_required_fraction||0);
      const reviewPending=Number(faceArtifact?.qc?.review_samples_remaining??faceArtifact?.qc?.manual_review_pending_sample_count??0);
      const reviewNeeded=reviewPending>0||reviewFraction>0;
      const items=[
        ['Trim-scoped Gaze 3D','gaze3d','gaze3d'],
        ['Partner association','participant_association','targets'],
        ['Partner-aware Face AOI','face_aoi','targets'],
        ['Body AOI','body_aoi','targets'],
        ['Full target fusion','fusion','targets'],
      ];
      const lines=items.map(([label,key,page])=>{
        const base=stepLine(label,isDone(role,key),page,roleId,'scene',stage(role,key).labels?.[0]);
        if(key!=='face_aoi'||!faceArtifact)return base;
        const reviewText=reviewNeeded
          ?`${reviewPending>0?`${reviewPending} sample${reviewPending===1?'':'s'} remain in the manual queue.`:'Automatic Face AOI contains samples marked for human review.'}`
          :'Open the frame-level player to verify the generated Face AOI or audit completed decisions.';
        return `${base}<div class="wh-face-review-entry"><div><strong>${reviewNeeded?'Human review required':'Face AOI review'}</strong><small>${esc(reviewText)}</small></div><button class="primary-button compact" data-iw-face-review="${esc(faceArtifact.id)}" data-iw-review-role="${esc(roleId)}" data-iw-review-interaction="${esc(row.interaction_id)}">${reviewNeeded?'Review flagged samples':'Open human review'}</button></div>`;
      }).join('');
      return `<article class="wh-role-column"><header><div><span>${esc(roleId)}</span><strong>${esc(participant)}</strong></div>${stateChip(rolePhaseStatus(row,roleId,'scene').done,'Targets ready')}</header><div class="wh-check-list">${lines}</div><div class="wh-optional-line">Object AOI is optional and can be added before fusion when reliable prop C3D is available.</div></article>`;
    };
    return `<div class="wh-stage-layout"><section><div class="wh-stage-intro"><div><div class="eyebrow">Stage 4</div><h2>Gaze & Targets</h2><p>Consume the calibration-complete handoff and paired C3D-defined clips, generate gaze-sample-clock rays on the verified TimeGraph, then create partner-aware Face AOI, Body AOI and fused targets for both roles.</p></div><button class="soft-button" data-iw-go="gaze3d">Open Gaze 3D</button></div><div class="wh-role-grid">${roleCard('R01')}${roleCard('R02')}</div></section><aside class="wh-stage-aside"><strong>Normal alignment</strong><span>Common body-C3D physical interval</span><span>Inverse verified Pupil↔Vicon TimeGraph</span><span>Exact NRD frame timestamps and immutable clip lineage</span><small>Manual Timing Recovery is available only in Advanced when the automatic contract cannot be satisfied.</small></aside></div>`;
  }

  function dyadicStage(row){
    const readyR01=isDone(row.roles?.R01,'fusion'),readyR02=isDone(row.roles?.R02,'fusion'),done=row.summary?.dyadic_ready;
    return `<div class="wh-stage-layout"><section><div class="wh-stage-intro"><div><div class="eyebrow">Stage 5</div><h2>Dyadic Attention</h2><p>Combine both full target streams on the same C3D-defined physical interval. The two role-specific local clips retain their shared Vicon lineage through the accepted TimeGraphs.</p></div></div><div class="wh-dyadic-card"><div class="wh-dyadic-prereqs">${stateChip(readyR01,'R01 full target')}${stateChip(readyR02,'R02 full target')}${stateChip(readyR01&&readyR02,'Shared C3D-defined interval')}</div><button class="primary-button" data-iw-step="dyadic" data-iw-role="R01" data-iw-workflow="dyadic" ${!readyR01||!readyR02?'disabled':''}>${done?'Review dyadic attention':'Generate mutual gaze & joint attention'}</button></div></section><aside class="wh-stage-aside"><strong>Outputs</strong><span>Mutual face gaze and eye contact</span><span>Shared social and object attention</span><span>Initiator, follower and latency episodes</span><small>The aligned clip-set artifact proves that both role streams cover the same Vicon interval.</small></aside></div>`;
  }

  function stepLine(label,done,page,roleId,workflow,detail='',options={}){
    const special=options.special||'';const disabled=options.disabled?'disabled':'';
    return `<div class="wh-check-row ${done?'done':''}"><span class="wh-check-icon">${done?'✓':'○'}</span><div><strong>${esc(label)}</strong>${detail?`<small>${esc(detail)}</small>`:''}</div><button class="${done||options.soft?'soft-button':'primary-button'} compact" data-iw-step="${esc(page)}" data-iw-special="${esc(special)}" data-iw-role="${esc(roleId)}" data-iw-workflow="${esc(workflow)}" ${disabled}>${done?'Review':'Open'}</button></div>`;
  }

  function stagePanel(row,stageId){
    if(stageId==='source')return sourceStage(row);
    if(stageId==='face')return faceStage(row);
    if(stageId==='calibration')return calibrationStage(row);
    if(stageId==='scene')return sceneStage(row);
    return dyadicStage(row);
  }

  function interactionWorkspaceView(){
    const row=selectedInteraction();if(!row)return guidedProjectView();const ctx=context();
    let active=state.activeStage;
    if(!STAGE_DEFS.some(x=>x.id===active)){
      const next=nextRequiredStep(row);active=next.stage||'source';state.activeStage=active;
    }
    const next=nextRequiredStep(row);const participantR01=participantForRole(row,'R01')||'Unassigned';const participantR02=participantForRole(row,'R02')||'Unassigned';
    return viewShell('Interaction workflow',`${workspaceData().session_code||session()?.label||''} / ${row.interaction_id}`,`
      <div class="iw-interaction-rail">${interactions().map(item=>`<button class="iw-interaction-pill ${item.interaction_id===row.interaction_id?'active':''}" data-iw-interaction="${esc(item.interaction_id)}"><b>${esc(item.interaction_id)}</b><span>${interactionOverall(item).complete}/${interactionOverall(item).total} steps</span></button>`).join('')}</div>
      <section class="wh-context-header"><div><div class="iw-breadcrumb">${esc(workspaceData().session_code||'Session')} / ${esc(row.interaction_id)}</div><h2>One interaction, five ordered stages.</h2><p>The current S/I/R context drives source selection and output labels. Calibration first verifies both clocks, then the common body-C3D interval automatically defines each role’s Native-reference clip.</p></div><div class="wh-context-identities"><article><span>R01</span><strong>${esc(participantR01)}</strong><small>${isDone(row.roles?.R01,'gaze3d')?'Trim-scoped rays ready':'Awaiting Gaze 3D'}</small></article><article><span>R02</span><strong>${esc(participantR02)}</strong><small>${isDone(row.roles?.R02,'gaze3d')?'Trim-scoped rays ready':'Awaiting Gaze 3D'}</small></article><article><span>Calibration + clips</span><strong>${(()=>{const c=calibrationWorkflowState();return c.allRolesReady&&c.alignedClipsReady?'Ready':'Pending'})()}</strong><small>${hardwareProfiles().length} profile(s) · ${sharedStage('etm_assignment').state==='complete'?'ETM resolved':'ETM pending'}</small></article></div></section>
      <div class="wh-next-banner"><div><span>Next required action</span><strong>${esc(next.label)}</strong></div><button class="primary-button" data-iw-next="true">Continue</button></div>
      <nav class="wh-stage-nav">${STAGE_DEFS.map(def=>{const s=stageStatus(row,def.id);return `<button class="${active===def.id?'active':''} ${s.done?'done':s.partial?'partial':''}" data-iw-stage="${def.id}"><span>${s.done?'✓':STAGE_DEFS.indexOf(def)+1}</span><div><strong>${esc(def.short)}</strong><small>${esc(s.label)}</small></div></button>`}).join('')}</nav>
      <section class="wh-stage-panel">${stagePanel(row,active)}</section>`);
  }

  function profileHardware(profile){return profile.parameters?.hardware_display_name||profile.qc?.hardware_display_name||profile.parameters?.hardware_id||profile.qc?.hardware_id||'Unspecified hardware';}
  function profileMount(profile){return profile.parameters?.mount_version||profile.qc?.mount_version||'mount unspecified';}
  function profileSource(profile){return profile.parameters?.calibration_source||profile.qc?.calibration_source||'calibration profile';}
  function profileSessions(profile){const raw=profile.parameters?.valid_sessions||profile.qc?.valid_sessions||[];return Array.isArray(raw)?raw:[];}
  function guidedHardwareView(){
    const profiles=hardwareProfiles();const acceptances=acceptanceArtifacts();
    const cards=profiles.map(profile=>`<article class="wh-profile-card"><div class="wh-profile-icon">${String(profileHardware(profile)).includes('ETM_A')||String(profileHardware(profile)).includes('ETM-A')?'A':String(profileHardware(profile)).includes('ETM_B')||String(profileHardware(profile)).includes('ETM-B')?'B':'◎'}</div><div><span>${esc(profileHardware(profile))}</span><strong>${esc(profile.label)}</strong><small>${esc(profileMount(profile))} · ${esc(profileSource(profile).replaceAll('_',' '))}</small><small>${profileSessions(profile).length?`Valid: ${esc(profileSessions(profile).join(', '))}`:'Validity not declared'}</small></div><button class="soft-button compact" data-iw-artifact="${esc(profile.id)}">View</button></article>`).join('');
    return viewShell('Project-level hardware','Hardware & Calibration',`
      <section class="wh-hardware-hero"><div><div class="eyebrow">Reusable project resources</div><h2>Measure once, validate for each interaction.</h2><p>Calibration profiles belong to ETM hardware and mount versions—not to P01/P02 or R01/R02.</p></div><div class="wh-hardware-actions"><button class="primary-button" data-iw-expert-page="calibration">Create / import calibration</button><button class="soft-button" data-iw-special-open="motionstudio">Body model & MotionStudio</button></div></section>
      <div class="wh-hardware-grid"><section><div class="iw-section-head compact"><div><div class="eyebrow">Hardware profile library</div><h2>ETM profiles</h2></div><span class="wh-count">${profiles.length}</span></div><div class="wh-profile-list">${cards||'<div class="iw-empty"><strong>No hardware profile registered.</strong><p>Create a measured profile or import a completed .g3dcal.</p></div>'}</div></section><aside><div class="wh-setting-card"><span>Calibration acceptance</span><strong>${acceptances.length}</strong><small>Interaction-specific transfer gates</small><button class="soft-button full" data-iw-expert-page="calibration">Open acceptance tools</button></div><div class="wh-setting-card"><span>Body model root</span><strong>Project setting</strong><small>Used only for fitted-mesh Body AOI</small><button class="soft-button full" data-iw-special-open="motionstudio">Open MotionStudio setup</button></div><div class="wh-setting-card"><span>Hardware naming</span><strong>ETM-A / ETM-B</strong><small>Legacy eye_tracker_1/2 marker aliases remain preserved.</small></div></aside></div>`);
  }

  function guidedReviewView(){
    const ctx=context();
    const interaction=selectedInteraction();
    const activeTrim=latestRoleTrim(ctx.interactionId,ctx.roleId);
    const inContext=artifact=>{
      const text=recordText(artifact);
      if(ctx.interactionId&&!text.includes(ctx.interactionId.toLowerCase()))return false;
      if(ctx.roleId&&!text.includes(ctx.roleId.toLowerCase()))return false;
      if(activeTrim&&typeof nativeTrimIdForRecord==='function'&&nativeTrimIdForRecord(artifact)!==activeTrim.id)return false;
      return true;
    };
    const face=artifacts(['face_aoi']).filter(inContext);
    const body=artifacts(['body_aoi']).filter(inContext);
    const fusion=artifacts(['target_fusion','semantic_targets']).filter(inContext);
    const videos=artifacts(['privacy_clip']).filter(inContext);
    const reviewVideos=artifacts(['aoi_review_video']).filter(inContext);
    const launch=state.reviewLaunch&&state.reviewLaunch.interactionId===ctx.interactionId&&state.reviewLaunch.roleId===ctx.roleId?state.reviewLaunch:null;
    const markup=window.OculoReview?OculoReview.markup({
      faceArtifacts:face,bodyArtifacts:body,fusionArtifacts:fusion,privacyClips:videos,reviewVideos,
      initialMode:launch?.mode||'',initialArtifactId:launch?.artifactId||'',initialVideoArtifactId:launch?.videoArtifactId||'',
    }):'<div class="warning-card">AOI Review Player unavailable.</div>';
    return viewShell('Quality review',interaction?`${interaction.interaction_id} · ${ctx.roleId} Review`:'Review',`<div class="wh-review-toolbar"><div><span>${esc(workspaceData().session_code||'')}</span><strong>${esc(ctx.interactionId||'All interactions')}</strong><span>${esc(ctx.roleId)}</span></div><div><button class="soft-button compact" data-iw-go="interactions">Back to workflow</button><button class="soft-button compact" data-iw-expert-page="advanced">Advanced setup</button></div></div>${markup}`);
  }

  function guidedResultsView(){
    const row=selectedInteraction();const ctx=context();const all=artifacts().filter(a=>!ctx.interactionId||recordText(a).includes(ctx.interactionId.toLowerCase()));
    const buckets=[
      ['R01',all.filter(a=>recordText(a).includes('r01'))],
      ['R02',all.filter(a=>recordText(a).includes('r02'))],
      ['Shared Scene',all.filter(a=>!recordText(a).includes('r01')&&!recordText(a).includes('r02')&&!['dyadic_attention'].includes(a.kind))],
      ['Dyadic',all.filter(a=>a.kind==='dyadic_attention')],
    ];
    const cards=buckets.map(([label,rows])=>`<article class="wh-result-bucket"><header><strong>${esc(label)}</strong><span>${rows.length}</span></header>${rows.length?rows.slice(-8).reverse().map(a=>`<button data-iw-artifact="${esc(a.id)}"><i>${esc(a.kind.replaceAll('_',' '))}</i><span>${esc(a.label)}</span></button>`).join(''):'<p>No results yet.</p>'}</article>`).join('');
    return viewShell('Processed results',row?`${row.interaction_id} · Results`:'Results',`<div class="wh-results-head"><div><div class="eyebrow">Human-readable result map</div><h2>${esc(workspaceData().session_code||'')} / ${esc(ctx.interactionId||'All interactions')}</h2><p>Internal artifact IDs remain available in Technical Details, but the default view follows R01, R02, Shared Scene and Dyadic outputs.</p></div><button class="primary-button" data-iw-expert-page="qc">Open full Results browser</button></div><div class="wh-result-grid">${cards}</div>`);
  }

  function guidedProjectProperties(){
    const pws=projectData();const options=(pws.sessions||[]).map(row=>`<option value="${esc(row.session_id)}" ${row.selected?'selected':''}>${esc(row.session_code)} · ${esc(row.label)}</option>`).join('');
    const current=(pws.sessions||[]).find(row=>row.selected)||null;
    return `<section class="property-section"><div class="section-label">Project structure</div><div class="property-stack"><div class="field"><label>Session</label><select id="guided-project-session">${options}</select></div><div class="form-help">Project → Session → Interaction → R01/R02. Changing Session never moves or merges its files.</div><button class="primary-button full" data-project-manage="true">Manage Sessions & Interactions</button><button class="soft-button full" data-project-folder="${esc(current?.folder||project()?.root||'')}">Open selected Session folder</button></div></section><section class="property-section"><div class="section-label">Shared Calibration</div><div class="property-stack">${stateChip((pws.counts?.calibration_profiles||0)>0,`${pws.counts?.calibration_profiles||0} profile records`)}<div class="form-help">Share only hardware camera-to-tracker profiles. Re-run wearer assignment and clock synchronization for every Interaction.</div><button class="soft-button full" data-project-manage="true">Open shared Calibration manager</button></div></section><button class="soft-button full" data-iw-mode="expert">Open Expert Tools</button>`;
  }

  function interactionsProperties(){
    const row=selectedInteraction(),ctx=context(),next=nextRequiredStep(row);
    return `<section class="property-section"><div class="section-label">Current context</div><div class="property-stack"><div class="field"><label>Interaction</label><select id="iw-interaction-select">${interactions().map(item=>`<option value="${esc(item.interaction_id)}" ${item.interaction_id===ctx.interactionId?'selected':''}>${esc(item.interaction_id)}</option>`).join('')}</select></div><div class="field"><label>Focused role</label><select id="iw-role-select"><option ${ctx.roleId==='R01'?'selected':''}>R01</option><option ${ctx.roleId==='R02'?'selected':''}>R02</option></select></div></div></section><section class="property-section"><div class="section-label">Next required action</div><div class="property-stack"><div class="wh-property-next"><strong>${esc(next.label)}</strong><button class="primary-button full" data-iw-next="true">Continue</button></div></div></section><section class="property-section"><div class="section-label">Readiness</div><div class="property-stack">${statusBadge(row?.summary?.face_ready,'Both Face AOIs')}${statusBadge(row?.summary?.scene_ready,'Both full targets')}${statusBadge(row?.summary?.dyadic_ready,'Dyadic output')}</div></section><button class="soft-button full" data-iw-mode="expert">Open Expert Tools</button>`;
  }
  function hardwareProperties(){
    return `<section class="property-section"><div class="section-label">Hardware library</div><div class="property-stack">${stateChip(hardwareProfiles().length>=2,`${hardwareProfiles().length} profiles registered`)}${stateChip(acceptanceArtifacts().length>0,`${acceptanceArtifacts().length} acceptance results`)}</div></section><section class="property-section"><div class="section-label">Rules</div><div class="form-help">ETM-A / ETM-B are hardware identities. P01/P02 are participants. R01/R02 are interaction roles. The app resolves these layers separately.</div></section><button class="primary-button full" data-iw-expert-page="calibration">Open Calibration tools</button>`;
  }
  function reviewProperties(){
    const ctx=context();
    const face=latestRoleArtifact(['face_aoi'],ctx.interactionId,ctx.roleId);
    const pending=Number(face?.qc?.review_samples_remaining??face?.qc?.manual_review_pending_sample_count??0);
    const reviewButton=face?`<button class="primary-button full" data-iw-face-review="${esc(face.id)}" data-iw-review-role="${esc(ctx.roleId)}" data-iw-review-interaction="${esc(ctx.interactionId)}">${pending>0?`Continue human review · ${pending} pending`:'Open Face AOI human review'}</button>`:`<button class="primary-button full" disabled>Generate Face AOI before human review</button>`;
    return `<section class="property-section"><div class="section-label">Human review</div><div class="property-stack">${reviewButton}<div class="form-help">The entry remains available even while the player is showing Body AOI or fused targets. Opening it switches the review player to the matching Face AOI only.</div></div></section><section class="property-section"><div class="section-label">Review context</div><div class="property-stack"><div class="form-help">Dynamic overlays never modify the immutable analysis tables.</div><button class="soft-button full" data-iw-go="interactions">Return to interaction</button><button class="soft-button full" data-iw-expert-page="targets">Target parameters</button></div></section>`;
  }

  renderWorkspace = function(){
    if(!project()){legacy.renderWorkspace();return;}
    if(state.mode==='guided'){
      const root=$('#workspace');
      if(appState.page==='project')root.innerHTML=guidedProjectView();
      else if(appState.page==='interactions')root.innerHTML=interactionWorkspaceView();
      else if(appState.page==='review')root.innerHTML=guidedReviewView();
      else if(appState.page==='qc')root.innerHTML=guidedResultsView();
      else {appState.page='interactions';root.innerHTML=interactionWorkspaceView();}
      bindPageActions();return;
    }
    primeContextDrafts();legacy.renderWorkspace();injectContextRibbon();
  };

  renderProperties = function(){
    if(!project()){legacy.renderProperties();return;}
    if(state.mode==='guided'){
      const item=GUIDED_PIPELINE.find(row=>row.id===appState.page)||GUIDED_PIPELINE[0];
      $('#module-card').innerHTML=`<div class="module-wrap"><div class="section-label">Workflow Hub</div><div class="module-card"><span class="module-icon">${item.icon}</span><span><b>${esc(item.label)}</b><small>${esc(item.desc)}</small></span></div></div>`;
      const root=$('#properties-content');
      if(appState.page==='project')root.innerHTML=guidedProjectProperties();
      else if(appState.page==='review')root.innerHTML=reviewProperties();
      else if(appState.page==='qc')root.innerHTML=`<section class="property-section"><div class="section-label">Results</div><div class="form-help">Select a result to open the full file browser or use Expert Tools for organized export packages.</div><button class="primary-button full" data-iw-expert-page="qc">Open full Results browser</button></section>`;
      else root.innerHTML=interactionsProperties();
      bindPropertyActions();return;
    }
    try{
      legacy.renderProperties();
      applyContextDefaults();
    }catch(error){
      console.error(error);
      const root=$('#properties-content');
      if(root)root.innerHTML=`<section class="property-section"><div class="section-label">Module panel unavailable</div><div class="warning-card"><strong>The current module could not be rendered.</strong><p>${esc(error?.message||String(error))}</p></div><button class="soft-button full" id="retry-properties-render">Retry module panel</button></section>`;
      $('#retry-properties-render')?.addEventListener('click',()=>render());
      toast('Module panel could not be rendered',error?.message||String(error),true);
    }
  };

  function returnToWorkflowHub(){
    state.mode='guided';
    try{localStorage.setItem('oculo.uiMode','guided')}catch(_){}
    setBodyMode();
    appState.page='interactions';

    // Render immediately so the control always feels responsive, then request a
    // full state refresh so newly completed artifacts update the next-action card.
    render();
    refreshState({quiet:false,fullRender:true});
  }

  function injectContextRibbon(){
    const view=$('#workspace .view');if(!view||!context().interactionId)return;
    view.querySelector('.iw-expert-ribbon')?.remove();
    const ribbon=document.createElement('div');
    ribbon.className='iw-expert-ribbon';
    const calibration=appState.page==='calibration';
    const contextText=calibration
      ?`${workspaceData().session_code||''} / ${context().interactionId} / Paired calibration`
      :`${workspaceData().session_code||''} / ${context().interactionId} / ${context().roleId}`;
    const selector=calibration
      ?`<label class="iw-ribbon-select"><span>Calibration view</span><select id="iw-calibration-view-ribbon"><option value="paired" ${state.calibrationView==='paired'?'selected':''}>Paired summary</option><option value="R01" ${state.calibrationView==='R01'?'selected':''}>${esc(calibrationRoleOptionLabel('R01'))}</option><option value="R02" ${state.calibrationView==='R02'?'selected':''}>${esc(calibrationRoleOptionLabel('R02'))}</option></select></label>`
      :`<label class="iw-ribbon-select"><span>Active role</span><select id="iw-role-view-select"><option value="R01" ${context().roleId==='R01'?'selected':''}>R01</option><option value="R02" ${context().roleId==='R02'?'selected':''}>R02</option></select></label>`;
    ribbon.innerHTML=`<button class="soft-button compact" data-iw-back>← Workflow Hub</button><span>Expert tool</span><strong>${esc(contextText)}</strong>${selector}`;
    view.prepend(ribbon);
    ribbon.querySelector('[data-iw-back]').onclick=returnToWorkflowHub;
    const calibrationSelect=ribbon.querySelector('#iw-calibration-view-ribbon');
    if(calibrationSelect)calibrationSelect.onchange=()=>setCalibrationView(calibrationSelect.value);
    const roleSelect=ribbon.querySelector('#iw-role-view-select');
    if(roleSelect)roleSelect.onchange=async()=>{
      const current=context(),target=String(roleSelect.value||'R01').toUpperCase();
      roleSelect.disabled=true;
      focusRoleSource(current.interactionId,target);
      await saveContext(current.interactionId,target,current.workflow);
      if(roleSelect.isConnected)roleSelect.disabled=false;
    };
  }

  function textMatchOption(select,interactionId,roleId,{roleOptional=false}={}){
    if(!select)return;
    const lower=value=>String(value||'').toLowerCase();
    const iid=lower(interactionId),rid=lower(roleId);
    const options=[...select.options].filter(o=>o.value);
    const matches=(option,{allowRoleless=roleOptional}={})=>{
      const text=lower(option.textContent);
      const hasInteraction=/(?:^|[^a-z0-9])i\d+(?![a-z0-9])/i.test(text);
      const hasRole=/(?:^|[^a-z0-9])r\d+(?![a-z0-9])/i.test(text);
      if(iid&&hasInteraction&&!text.includes(iid))return false;
      if(rid&&hasRole&&!text.includes(rid))return false;
      if(rid&&!hasRole&&!allowRoleless)return false;
      return true;
    };
    const current=select.selectedOptions?.[0];
    if(current?.value&&matches(current))return;
    let candidates=options.filter(option=>matches(option));
    if(!candidates.length&&roleOptional)candidates=options.filter(option=>{
      const text=lower(option.textContent);
      const hasInteraction=/(?:^|[^a-z0-9])i\d+(?![a-z0-9])/i.test(text);
      return !iid||!hasInteraction||text.includes(iid);
    });
    if(candidates.length){
      select.value=candidates[0].value;
      select.dispatchEvent(new Event('change',{bubbles:true}));
    }
  }
  function suggestedLabel(stageName){const c=context();return `${workspaceData().session_code||'S'}_${c.interactionId}_${c.roleId}_${stageName}`.replace(/_+/g,'_');}
  function setDefaultValue(id,value,generic=[]){const el=$(`#${id}`);if(!el)return;const current=String(el.value||'').trim();if(!current||generic.includes(current))el.value=value;}
  function primeContextDrafts(){
    if(appState.page!=='clips'||typeof nativeDraft!=='function')return;
    const d=nativeDraft(),ctx=context();if(!ctx.interactionId)return;
    const find=(kinds,role=true,{interactionRequired=true}={})=>assets(kinds).filter(a=>{const text=recordText(a);return (!interactionRequired||text.includes(ctx.interactionId.toLowerCase()))&&(!role||text.includes(ctx.roleId.toLowerCase()));});
    const n=find(['pupil_native']),t=find(['pupil_cloud']),f=find(['face_mapper'],true,{interactionRequired:false});
    if(!d.native&&n.length===1)d.native=n[0].id;if(!d.tsd&&t.length===1)d.tsd=t[0].id;if(!d.fm&&f.length===1)d.fm=f[0].id;
    if(!d.label||d.label==='Native data clip')d.label=suggestedLabel('trim');
  }
  function applyContextDefaults(){
    const ctx=context();if(!ctx.interactionId)return;
    const pairs=[['fm-filter-trim',false],['privacy-native',false],['privacy-filter',false],['privacy-face',true],['privacy-native-trim-direct',false],['privacy-native-trim',false],['privacy-clip-source',false],['association-trim',false],['association-filter',false],['association-etm',true],['face-aoi-trim',false],['face-aoi-filter',false],['face-aoi-association',false],['face-aoi-privacy',false],['ray-cloud',false],['ray-assignment',true],['fusion-face',false],['fusion-body',false],['fusion-object',false],['dyadic-artifact-a',true],['dyadic-artifact-b',true]];
    for(const [id,roleOptional] of pairs)textMatchOption($(`#${id}`),ctx.interactionId,ctx.roleId,{roleOptional});
    const roleSelect=$('#ray-role');if(roleSelect)roleSelect.value=ctx.roleId;
    setDefaultValue('fm-filter-label',suggestedLabel('facemapper_filter'),['Filtered Face Mapper']);
    setDefaultValue('privacy-clip-label',suggestedLabel('privacy_trimmed'),['privacy_trimmed']);
    setDefaultValue('association-label',suggestedLabel('participant_association'),['participant_association']);
    setDefaultValue('face-aoi-label',suggestedLabel('face_aoi'),['face_aoi']);
    setDefaultValue('etm-assignment-label',`${workspaceData().session_code||'S'}_${ctx.interactionId}_ETM_assignment`,['ETM role assignment']);
  }

  async function runNext(){
    const next=nextRequiredStep();
    if(next.stage)setStage(next.stage,{renderAfter:false});
    if(next.page==='review'){navigate('review');return;}
    await openStep(next.page,{roleId:next.roleId||context().roleId,workflow:next.stage==='scene'?'scene':next.stage==='dyadic'?'dyadic':'face',special:next.special||''});
  }
  async function openFaceReview(artifactId,interactionId,roleId){
    if(!artifactId)return;
    const iid=interactionId||context().interactionId;
    const rid=roleId||context().roleId;
    state.reviewLaunch={mode:'face',artifactId,interactionId:iid,roleId:rid};
    if(iid&&(context().interactionId!==iid||context().roleId!==rid)){
      await saveContext(iid,rid,'scene',{renderAfter:false});
    }
    navigate('review');
  }

  function openArtifact(id){
    if(!id)return;appState.selectedArtifactId=id;setMode('expert',{page:'qc'});
  }

  bindGlobalActions = function(){
    legacy.bindGlobalActions();
    const modeSelect=$('#ui-mode-select');if(modeSelect){
      modeSelect.value=state.mode;
      modeSelect.title=state.mode==='guided'?'Guided workflow is active':'Expert tools are active';
      modeSelect.onchange=()=>setMode(modeSelect.value);
    }
    // Compatibility control retained for older validation scripts; it is not visible.
    const toggle=$('#ui-mode-toggle');if(toggle){
      const guided=state.mode==='guided';
      toggle.textContent=guided?'Guided mode':'Expert mode';
      toggle.dataset.activeMode=state.mode;
      toggle.title=guided?'Guided mode is active. Click to open Expert Tools.':'Expert Tools are active. Click to return to the Guided Workflow.';
      toggle.setAttribute('aria-label',toggle.title);
      toggle.onclick=()=>setMode(guided?'expert':'guided');
    }
    const badge=$('.clean-badge');if(badge){badge.textContent='Clean Core';badge.title='OculoStudio clean-room application core';}
    setBodyMode();
  };

  bindPageActions = function(){
    legacy.bindPageActions();
    $$('[data-iw-interaction]').forEach(button=>button.onclick=()=>saveContext(button.dataset.iwInteraction,context().roleId,context().workflow));
    $$('[data-project-session-select]').forEach(button=>button.onclick=async()=>{try{await api(`/api/session/${encodeURIComponent(button.dataset.projectSessionSelect)}/select`,{method:'POST'});await refreshState({quiet:false,fullRender:true});}catch(error){showError(error,'Could not switch Session')}});
    $$('[data-project-folder]').forEach(button=>button.onclick=()=>openLocalPath(button.dataset.projectFolder));
    $$('[data-project-open-root]').forEach(button=>button.onclick=()=>openLocalPath(project().root));
    $$('[data-project-manage]').forEach(button=>button.onclick=()=>openProjectSessionManager({focus:'sessions'}));
    $$('[data-iw-stage]').forEach(button=>button.onclick=()=>setStage(button.dataset.iwStage));
    $$('[data-iw-step]').forEach(button=>button.onclick=()=>openStep(button.dataset.iwStep,{roleId:button.dataset.iwRole||context().roleId,workflow:button.dataset.iwWorkflow||context().workflow,special:button.dataset.iwSpecial||''}));
    $$('[data-iw-go]').forEach(button=>button.onclick=()=>{const page=button.dataset.iwGo;if(GUIDED_PAGE_IDS.has(page))navigate(page);else openStep(page);});
    $$('[data-iw-mode]').forEach(button=>button.onclick=()=>setMode(button.dataset.iwMode||'expert',{page:button.dataset.iwExpertPage||null}));
    $$('[data-iw-expert-page]').forEach(button=>button.onclick=()=>setMode('expert',{page:button.dataset.iwExpertPage}));
    $$('[data-iw-special-open]').forEach(button=>button.onclick=()=>openStep('project',{special:button.dataset.iwSpecialOpen}));
    $$('[data-iw-artifact]').forEach(button=>button.onclick=()=>openArtifact(button.dataset.iwArtifact));
    $$('[data-iw-face-review]').forEach(button=>button.onclick=()=>openFaceReview(button.dataset.iwFaceReview,button.dataset.iwReviewInteraction,button.dataset.iwReviewRole));
    $$('[data-iw-next]').forEach(button=>button.onclick=runNext);
    $$('[data-calibration-view]').forEach(button=>button.onclick=()=>setCalibrationView(button.dataset.calibrationView));
    $('[data-iw-back]')?.addEventListener('click',returnToWorkflowHub);
    // The Expert ribbon binds its role switch when it is injected. Binding it a
    // second time here submitted duplicate context requests and made role changes
    // susceptible to races with module re-rendering.
    if(state.mode==='guided'&&appState.page==='review'&&window.OculoReview){
      const launch=state.reviewLaunch&&state.reviewLaunch.interactionId===context().interactionId&&state.reviewLaunch.roleId===context().roleId?state.reviewLaunch:null;
      const reviewTrim=latestRoleTrim(context().interactionId,context().roleId);
      OculoReview.bind({sessionId:selectedSessionId(),interactionId:context().interactionId,roleId:context().roleId,nativeTrimArtifactId:reviewTrim?.id||'',initialMode:launch?.mode||'',initialArtifactId:launch?.artifactId||'',initialVideoArtifactId:launch?.videoArtifactId||''});
    }
  };

  const legacyBindPropertyActions=bindPropertyActions;
  bindPropertyActions = function(){
    legacyBindPropertyActions();
    $('#guided-project-session')?.addEventListener('change',async event=>{try{await api(`/api/session/${encodeURIComponent(event.target.value)}/select`,{method:'POST'});await refreshState({quiet:false,fullRender:true});}catch(error){showError(error,'Could not switch Session')}});
    $$('[data-project-folder]').forEach(button=>button.onclick=()=>openLocalPath(button.dataset.projectFolder));
    $$('[data-project-manage]').forEach(button=>button.onclick=()=>openProjectSessionManager({focus:'sessions'}));
    $('#iw-interaction-select')?.addEventListener('change',e=>saveContext(e.target.value,$('#iw-role-select')?.value||'R01',context().workflow));
    $('#iw-role-select')?.addEventListener('change',e=>saveContext($('#iw-interaction-select')?.value||context().interactionId,e.target.value,context().workflow));
    $$('[data-iw-mode]').forEach(button=>button.onclick=()=>setMode(button.dataset.iwMode||'expert',{page:button.dataset.iwExpertPage||null}));
    $$('[data-iw-go]').forEach(button=>button.onclick=()=>GUIDED_PAGE_IDS.has(button.dataset.iwGo)?navigate(button.dataset.iwGo):openStep(button.dataset.iwGo));
    $$('[data-iw-expert-page]').forEach(button=>button.onclick=()=>setMode('expert',{page:button.dataset.iwExpertPage}));
    $$('[data-iw-face-review]').forEach(button=>button.onclick=()=>openFaceReview(button.dataset.iwFaceReview,button.dataset.iwReviewInteraction,button.dataset.iwReviewRole));
    $$('[data-iw-next]').forEach(button=>button.onclick=runNext);
    $$('[data-calibration-view]').forEach(button=>button.onclick=()=>setCalibrationView(button.dataset.calibrationView));
  };

  document.addEventListener('DOMContentLoaded',()=>setBodyMode());
  window.OculoInteractionWorkspace={mode:()=>state.mode,setMode,openStep,context,setStage,nextRequiredStep,returnToWorkflowHub,calibrationView,setCalibrationView};
})();
