/* OculoStudio CleanCore desktop client.
 * Deliberately dependency-free: all processing is performed by the Python
 * CleanCore API. No legacy application, executable, or source module is loaded.
 */

const PIPELINE = [
  { id: 'project', label: 'Project', icon: '▣', desc: 'Create a clean project and follow the processing checklist' },
  { id: 'data', label: 'Data', icon: '▤', desc: 'Import and identify Pupil, FaceMapper and Vicon sources' },
  { id: 'calibration', label: 'Calibration', icon: '◎', desc: 'Resolve ETM wearer, synchronize clocks and derive C3D-defined Pupil clips' },
  { id: 'clips', label: 'Trim', icon: '▦', desc: 'Review the automatic C3D-defined Native-reference intervals or use manual recovery' },
  { id: 'privacy', label: 'Privacy', icon: '◉', desc: 'Filter FaceMapper and create the final synchronized privacy video' },
  { id: 'gaze3d', label: 'Gaze 3D', icon: '✣', desc: 'Generate world-space gaze rays from calibration-complete role inputs' },
  { id: 'targets', label: 'Gaze Targets', icon: '⌾', desc: 'Generate Face, Body and optional Object AOIs, then fuse evidence' },
  { id: 'dyadic', label: 'Dyadic Attention', icon: '∞', desc: 'Annotate mutual gaze, joint attention and gaze following' },
  { id: 'qc', label: 'Results & QC', icon: '✓', desc: 'Review quality gates and export processed results' },
  { id: 'advanced', label: 'Advanced', icon: '⋯', desc: 'Timing recovery, diagnostics and legacy project tools', utility: true },
  // Timing recovery remains available for raw or mismatched recordings, but it
  // is not a stage in the normal pre-trimmed shared-start workflow.
  { id: 'synchronize', label: 'Timing Recovery', icon: '⟳', desc: 'Build a TimeGraph only for raw or mismatched recordings', hidden: true },
];

const ASSET_GROUPS = [
  { label: 'Eye tracking', kinds: ['pupil_native', 'pupil_cloud', 'face_mapper'] },
  { label: 'External semantic evidence', kinds: ['densepose_export'] },
  { label: 'Motion capture', kinds: ['c3d', 'mocap_npz'] },
  { label: 'Media & tables', kinds: ['video', 'audio', 'table'] },
  { label: 'Calibration & other', kinds: ['calibration', 'archive', 'directory', 'unknown'] },
];

const KIND_ICON = {
  pupil_native: 'N', pupil_cloud: 'C', face_mapper: 'F', densepose_export: 'DP', c3d: '3D', mocap_npz: 'M',
  video: '▶', audio: '♪', table: 'T', calibration: '◎', archive: 'Z', directory: 'D', unknown: '?'
};

const appState = {
  page: 'project',
  server: { version: '0.21.71' },
  calibrationSetId: '',
  calibrationSourcePath: '',
  calibrationRecursive: false,
  calibrationScan: null,
  calibrationDrafts: {},
  pendingCalibrationImportJobId: null,
  pendingCalibrationSetProfileJobId: null,
  deduplicationNotice: null,
  data: { project: null, qc: { metrics: [], warnings: [] } },
  selectedAssetId: null,
  selectedCalibrationAssetId: null,
  selectedAssetIds: [],
  dataScrollTarget: '',
  selectedArtifactId: null,
  artifactCleanupPlan: null,
  artifactCleanupPlanSessionId: '',
  artifactCleanupScanOrphans: false,
  artifactCleanupSelection: [],
  artifactCleanupLoading: false,
  resultGroupState: {},
  selectedClipId: null,
  timeAnchors: [{ source_ns: 0, target_ns: 0, label: 'Anchor 1', weight: 1 }],
  timer: null,
  polling: false,
  pendingPrivacyJobId: null,
  pendingETMAssignmentJobId: null,
  pendingCalibrationBindingJobId: null,
  pendingC3DDefinedClipsJobId: null,
  pendingGaze3DJobId: null,
  pendingBodyAOIJobId: null,
  pendingAudioVideoSyncJobId: null,
  audioVideoSyncCandidate: null,
  audioVideoSyncRole: 'R01',
  audioVideoSyncSourceId: '',
  audioVideoSyncReferenceId: '',
  audioVideoSyncMaximumOffsetSec: 300,
  aoiReviewInitialMode: '',
  aoiReviewInitialArtifactId: '',
  aoiReviewInitialVideoArtifactId: '',
  aoiReviewScrollRequested: false,
  aoiReviewOpenManualMode: '',
  gazePreviewMode: null,
  gazePreviewArtifactIds: {R01:'', R02:''},
  gazePreviewToleranceMs: '',
  privacyPreviewMode: 'auto',
  syncPrefill: null,
  autoSyncCandidate: null,
  calibrationAutoSyncCandidates: {R01:null,R02:null},
  calibrationSyncSelections: {R01:'',R02:''},
  calibrationSyncSettings: {maximumOffsetSec:120,sampleRateHz:120,saveIfHigh:true},
  calibrationAutoSyncRunningRole: '',
  calibrationClipSelections: {
    R01:{native_asset_id:'',timeseries_asset_id:'',facemapper_asset_id:''},
    R02:{native_asset_id:'',timeseries_asset_id:'',facemapper_asset_id:''},
  },
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const esc = (value) => String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
const fmtBytes = (bytes) => {
  if (bytes == null || !Number.isFinite(Number(bytes))) return '—';
  const units = ['B','KB','MB','GB','TB']; let n = Number(bytes); let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i += 1; }
  return `${n >= 100 ? n.toFixed(0) : n >= 10 ? n.toFixed(1) : n.toFixed(2)} ${units[i]}`;
};
const compactC3DFrameSource = (value) => {
  const raw=String(value||'').trim();
  if(!raw)return '—';
  if(/HOM_FIRST_FRAME.*HOM_LAST_FRAME/i.test(raw))return 'Vicon long-frame metadata';
  if(/ACTUAL_START_FIELD.*ACTUAL_END_FIELD/i.test(raw))return 'C3D trial range';
  if(/LONG_FRAMES/i.test(raw))return 'C3D long-frame metadata';
  if(/physical.*capacity/i.test(raw))return 'Physical data capacity';
  return raw;
};
const safeProjectFolderName = (value) => String(value||'OCUMO_Clean_Project')
  .trim().replace(/[<>:"\/\\|?*\u0000-\u001F]+/g,'_').replace(/[. ]+$/g,'').replace(/\s+/g,'_') || 'OCUMO_Clean_Project';
const joinLocalPath = (parent, child) => {
  const base=String(parent||'').trim().replace(/[\\/]+$/,'');
  const leaf=String(child||'').trim().replace(/^[\\/]+/,'');
  if(!base)return leaf;
  const sep=base.includes('\\')?'\\':'/';
  return `${base}${sep}${leaf}`;
};

const projectWorkspace = () => appState.data.project_workspace || {sessions:[],counts:{},calibration_profiles:[],calibration_sets:[]};
const calibrationSetRows = () => projectWorkspace().calibration_sets || [];
const calibrationSetRecord = value => {
  const token=String(value||'').trim().toLowerCase();
  return calibrationSetRows().find(row=>String(row.id||'').toLowerCase()===token||String(row.code||'').toLowerCase()===token)||null;
};
const sessionCalibrationBinding = (sessionId=selectedSessionId()) => sessionWorkspace(sessionId)?.calibration_binding || project()?.sessions?.[sessionId]?.metadata?.calibration_binding || null;
const sessionWorkspace = (sessionId=selectedSessionId()) => projectWorkspace().sessions?.find(row=>row.session_id===sessionId)||null;
const sessionStoragePath = (key,sessionId=selectedSessionId()) => sessionWorkspace(sessionId)?.paths?.[key] || '';
const parseCodeSpecification = (value,prefix) => {
  const text=String(value||'').toUpperCase().replace(/[–—]/g,'-');
  const tokens=text.split(/[\s,;]+/).map(row=>row.trim()).filter(Boolean);
  const output=[];const seen=new Set();
  const add=(code)=>{if(!seen.has(code)){seen.add(code);output.push(code);}};
  for(const token of tokens){
    const range=token.match(new RegExp(`^${prefix}(\\d{1,6})-${prefix}?(\\d{1,6})$`,'i'));
    if(range){
      const start=Number(range[1]),end=Number(range[2]);
      if(end<start||end-start>199)throw new Error(`${prefix} range must be ascending and contain at most 200 entries`);
      const width=Math.max(range[1].length,range[2].length,2);
      for(let index=start;index<=end;index++)add(`${prefix}${String(index).padStart(width,'0')}`);
      continue;
    }
    const single=token.match(new RegExp(`^${prefix}(\\d{1,6})$`,'i'));
    if(!single)throw new Error(`Use ${prefix} codes such as ${prefix}004-${prefix}007 or ${prefix}004, ${prefix}005`);
    add(`${prefix}${single[1]}`);
  }
  return output;
};
const parseSessionSpecification = value => parseCodeSpecification(value,'S');
const parseInteractionSpecification = value => parseCodeSpecification(value,'I');
const fmtNs = (value) => {
  const ns = Number(value || 0); const totalMs = Math.max(0, Math.round(ns / 1e6));
  const ms = totalMs % 1000; const totalSec = Math.floor(totalMs / 1000);
  const sec = totalSec % 60; const totalMin = Math.floor(totalSec / 60);
  const min = totalMin % 60; const hr = Math.floor(totalMin / 60);
  return `${String(hr).padStart(2,'0')}:${String(min).padStart(2,'0')}:${String(sec).padStart(2,'0')}.${String(ms).padStart(3,'0')}`;
};
const deepGet = (obj, path, fallback = undefined) => path.split('.').reduce((cur, key) => cur?.[key], obj) ?? fallback;
const byCreated = (a, b) => String(a.created_at || '').localeCompare(String(b.created_at || ''));
const codeFromText = (text, prefix) => {
  const m=String(text||'').match(new RegExp(`(?:^|[^A-Za-z0-9])(${prefix}\\d{1,6})(?![A-Za-z0-9])`,'i'));
  return m?m[1].toUpperCase():'';
};
const assetIdentity = (asset) => {
  const identity=(asset?.metadata?.identity&&typeof asset.metadata.identity==='object')?asset.metadata.identity:{};
  const text=[identity.session_id,identity.interaction_id,identity.role_id,asset?.label,asset?.source_path].filter(Boolean).join(' ');
  return {
    participant_id:asset?.participant_id||identity.participant_id||'',
    session_id:identity.session_id||codeFromText(text,'S')||session()?.label||'',
    interaction_id:identity.interaction_id||codeFromText(text,'I')||'',
    role_id:identity.role_id||codeFromText(text,'R')||'',
    source:identity.source||(asset?.participant_id?'import_user':'filename_inference'),
    updated_at:identity.updated_at||'',
  };
};

const inflightApiRequests = new Map();
function api(path, options = {}) {
  const init = { headers: { 'Content-Type': 'application/json', ...(options.headers || {}) }, ...options };
  if (init.body && typeof init.body !== 'string') init.body = JSON.stringify(init.body);
  const method=String(init.method||'GET').toUpperCase();
  const dedupeRequest=method==='POST' && (path.startsWith('/api/jobs/') || path.startsWith('/api/motionstudio/'));
  const requestKey=dedupeRequest?`${method} ${path} ${init.body||''}`:'';
  if(dedupeRequest && inflightApiRequests.has(requestKey))return inflightApiRequests.get(requestKey);
  const promise=(async()=>{
    const response = await fetch(path, init);
    let payload = null;
    const contentType = response.headers.get('content-type') || '';
    try { payload = contentType.includes('json') ? await response.json() : await response.text(); }
    catch (_) { payload = null; }
    if (!response.ok) {
      const message = payload?.message || payload?.detail || payload?.error || response.statusText;
      throw new Error(typeof message === 'string' ? message : JSON.stringify(message));
    }
    if(payload?.deduplication?.reused){
      appState.deduplicationNotice={...payload.deduplication,expires:Date.now()+1800};
      const title=payload.deduplication.reason==='active_job'?'Identical job already running':'Existing result reused';
      setTimeout(()=>toast(title,payload.deduplication.message||'No duplicate files were created.'),0);
    }
    return payload;
  })();
  if(dedupeRequest){
    inflightApiRequests.set(requestKey,promise);
    promise.finally(()=>inflightApiRequests.delete(requestKey));
  }
  return promise;
}

function toast(title, message = '', error = false) {
  const notice=appState.deduplicationNotice;
  if(!error && notice && notice.expires>Date.now() && /queued|submitted|started/i.test(String(title))){
    appState.deduplicationNotice=null;
    return;
  }
  const stack = $('#toast-stack');
  const el = document.createElement('div');
  el.className = `toast${error ? ' error' : ''}`;
  el.innerHTML = `<strong>${esc(title)}</strong>${message ? `<span>${esc(message)}</span>` : ''}`;
  stack.append(el);
  setTimeout(() => el.remove(), error ? 6000 : 3500);
}

function showError(error, context = 'Operation failed') {
  console.error(error);
  toast(context, error?.message || String(error), true);
}

function modal(title, body, onReady) {
  const root = $('#modal-root');
  root.innerHTML = `<div class="modal-backdrop"><section class="modal" role="dialog" aria-modal="true">
    <header class="modal-head"><h2>${esc(title)}</h2><button class="modal-close" type="button">×</button></header>
    <div class="modal-body">${body}</div></section></div>`;
  const close = () => { root.innerHTML = ''; };
  $('.modal-close', root).onclick = close;
  $('.modal-backdrop', root).addEventListener('click', e => { if (e.target.classList.contains('modal-backdrop')) close(); });
  if (onReady) onReady(root, close);
}

async function choosePath(kind = 'file', title = 'Select a path', filetypes = []) {
  try {
    const result = window.pywebview?.api?.choose_paths
      ? await window.pywebview.api.choose_paths(kind, title, filetypes, false)
      : await api('/api/dialog', { method: 'POST', body: { kind, title, filetypes, multiple:false } });
    if (result.cancelled) toast('Selection cancelled', 'No file or folder was selected.');
    return result.path || '';
  } catch (error) {
    toast('File picker unavailable', String(error.message || error) + ' — For ZIP data use Import asset → Browser file picker, or paste a full path.', true);
    return '';
  }
}

function project() { return appState.data.project; }
function session() {
  const p = project();
  return p?.selected_session_id ? p.sessions?.[p.selected_session_id] : null;
}
function values(object) { return Object.values(object || {}); }
function allAssets(kinds = null) { const rows = values(session()?.assets); return kinds ? rows.filter(a => kinds.includes(a.kind)) : rows; }
function assetScope(asset) { return String(asset?.metadata?.asset_scope || 'interaction'); }
function assets(kinds = null) {
  const rows = allAssets(kinds).filter(asset => assetScope(asset) !== 'calibration');
  return rows;
}
function currentInteractionAssets(kinds = null, {includeUnassigned=true} = {}) {
  const ctx=activeWorkflowContext();
  const rows=assets(kinds);
  if(!ctx.interactionId)return rows;
  return rows.filter(asset=>{
    const identity=assetIdentity(asset);
    if(identity.interaction_id===ctx.interactionId)return true;
    return includeUnassigned&&!identity.interaction_id;
  });
}
function calibrationAssets(kinds = null, setId = null) {
  let rows = allAssets(kinds).filter(asset => assetScope(asset) === 'calibration');
  if (setId) rows = rows.filter(asset => String(asset.metadata?.calibration_set_id || '') === String(setId));
  return rows;
}
function calibrationSetIds() {
  const projectCodes=calibrationSetRows().map(row=>String(row.code||row.id||'')).filter(Boolean);
  const assetCodes=calibrationAssets().map(asset=>String(asset.metadata?.calibration_set_id||'')).filter(Boolean);
  return [...new Set([...projectCodes,...assetCodes])].sort();
}
function activeCalibrationSetId() {
  const rows=calibrationSetRows();
  const binding=sessionCalibrationBinding();
  const bound=rows.find(row=>row.id===binding?.calibration_set_id)||null;
  const current=calibrationSetRecord(appState.calibrationSetId);
  const chosen=current||bound||rows[0]||null;
  if(chosen){appState.calibrationSetId=chosen.code||chosen.id;return appState.calibrationSetId;}
  const ids=calibrationSetIds();
  const legacy=String(appState.calibrationSetId||'');
  if(legacy&&(ids.includes(legacy)||appState.calibrationScan?.calibration_set_id===legacy))return legacy;
  appState.calibrationSetId=ids[0]||'CALIBRATION';
  return appState.calibrationSetId;
}
function activeCalibrationSetRecord(){return calibrationSetRecord(activeCalibrationSetId());}
function calibrationProfileHardware(profile) {
  const token=[profile?.parameters?.hardware_id,profile?.qc?.hardware_id,profile?.parameters?.hardware_display_name,profile?.qc?.hardware_display_name,profile?.label].filter(Boolean).join(' ').toUpperCase().replace(/[^A-Z0-9]+/g,'');
  if(token.includes('ETMA')||token.includes('EYETRACKERA')||token.includes('EYETRACKER1'))return 'ETM_A';
  if(token.includes('ETMB')||token.includes('EYETRACKERB')||token.includes('EYETRACKER2'))return 'ETM_B';
  return '';
}
function calibrationProfilesForActiveSet() {
  const row=activeCalibrationSetRecord();
  const revisionId=row?.active_revision_id||sessionCalibrationBinding()?.revision_id||'';
  const code=String(row?.code||activeCalibrationSetId()||'');
  const recordId=String(row?.id||'');
  return artifacts(['calibration_profile']).filter(profile=>{
    if(String(profile?.parameters?.calibration_set_profile_build_status||'')==='failed_uncommitted')return false;
    const pCode=String(profile?.parameters?.calibration_set_id||profile?.qc?.calibration_set_id||'');
    const pRecord=String(profile?.parameters?.calibration_set_record_id||profile?.qc?.calibration_set_record_id||'');
    const pRevision=String(profile?.parameters?.calibration_revision_id||profile?.qc?.calibration_revision_id||'');
    if(row && !(pCode===code||pRecord===recordId))return false;
    if(revisionId && pRevision && pRevision!==revisionId)return false;
    return Boolean(calibrationProfileHardware(profile));
  }).slice().sort(byCreated);
}
function calibrationSetProfileForHardware(hardware) {
  const key=String(hardware||'').toUpperCase().replace('-','_');
  return calibrationProfilesForActiveSet().filter(profile=>calibrationProfileHardware(profile)===key).slice(-1)[0]||null;
}
function calibrationRoleLabel(role) {
  return ({pupil_calibration:'Pupil calibration', calibration_board:'Calibration board', tracker_candidate:'ETM tracker', world_reference:'World reference', board_definition:'Board definition', completed_profile:'Completed profile', calibration_source:'Calibration source'})[String(role||'')] || String(role||'Calibration source').replaceAll('_',' ');
}

function calibrationDisplayMode() {
  const value=window.OculoInteractionWorkspace?.calibrationView?.() || 'paired';
  return ['paired','R01','R02'].includes(value)?value:'paired';
}

function calibrationContextRole() {
  const display=calibrationDisplayMode();
  if(appState.page==='calibration' && (display==='R01'||display==='R02'))return display;
  return String(activeWorkflowContext()?.roleId || 'R01').trim().toUpperCase() || 'R01';
}

function calibrationParticipantForRole(roleId = calibrationContextRole()) {
  const ctx=activeWorkflowContext();
  const role=String(roleId||'').trim().toUpperCase();
  const counts=new Map();
  for(const asset of assets()){
    const identity=assetIdentity(asset);
    if(ctx.interactionId && identity.interaction_id && identity.interaction_id!==ctx.interactionId)continue;
    if(role && identity.role_id && identity.role_id!==role)continue;
    const participant=String(identity.participant_id||'').trim().toUpperCase();
    if(participant)counts.set(participant,(counts.get(participant)||0)+1);
  }
  const ranked=[...counts.entries()].sort((a,b)=>b[1]-a[1]||a[0].localeCompare(b[0]));
  if(ranked.length)return ranked[0][0];
  return role==='R02'?'P02':'P01';
}

function calibrationParticipantText(asset) {
  const identity=(asset?.metadata?.identity&&typeof asset.metadata.identity==='object')?asset.metadata.identity:{};
  return [asset?.participant_id,identity.participant_id,asset?.metadata?.participant_id,asset?.label,asset?.source_path]
    .filter(Boolean).join(' ').toUpperCase();
}

function calibrationNativeForParticipant(rows, participantId) {
  const participant=String(participantId||'').trim().toUpperCase();
  if(!participant)return rows?.[0]||null;
  return (rows||[]).find(asset=>calibrationParticipantText(asset).includes(participant)) || rows?.[0] || null;
}

function calibrationProfileParticipant(profile) {
  return String(profile?.parameters?.participant_id || profile?.qc?.participant_id || profile?.parameters?.calibration_wearer_id || profile?.qc?.calibration_wearer_id || '').trim().toUpperCase();
}

function expandSessionRange(startText,endText) {
  const start=Number(String(startText||'').replace(/\D/g,''));
  const end=Number(String(endText||'').replace(/\D/g,''));
  if(!Number.isFinite(start)||!Number.isFinite(end)||end<start||end-start>999)return [];
  const width=Math.max(String(startText||'').replace(/\D/g,'').length,String(endText||'').replace(/\D/g,'').length,3);
  return Array.from({length:end-start+1},(_,index)=>`S${String(start+index).padStart(width,'0')}`);
}

function calibrationDefaultValidSessions(activeSet, rows=[]) {
  const text=[activeSet,...(rows||[]).flatMap(row=>[row?.label,row?.source_path])].filter(Boolean).join(' ');
  const range=text.match(/S(\d{1,6})\s*[-_]\s*S?(\d{1,6})/i);
  if(range){
    const expanded=expandSessionRange(range[1],range[2]);
    if(expanded.length)return expanded.join(',');
  }
  const sessions=[...new Set((text.match(/S\d{1,6}/gi)||[]).map(value=>value.toUpperCase()))];
  return sessions.join(',') || session()?.label || 'S001';
}

function calibrationDraftKey(activeSet=activeCalibrationSetId(), roleId=calibrationContextRole()) {
  return [project()?.id||'project',selectedSessionId()||'session',activeSet||'set',String(roleId||'R01').toUpperCase()].join('::');
}

function calibrationDraft(activeSet=activeCalibrationSetId(), roleId=calibrationContextRole()) {
  const key=calibrationDraftKey(activeSet,roleId);
  if(!appState.calibrationDrafts[key])appState.calibrationDrafts[key]={};
  return appState.calibrationDrafts[key];
}

function toggleState(id, fallback=false) {
  const element=$(`#${id}`);
  return element?element.dataset.on==='true':fallback;
}

function captureCalibrationDraft() {
  if(appState.page!=='calibration')return;
  const activeSet=activeCalibrationSetId(),roleId=calibrationContextRole(),draft=calibrationDraft(activeSet,roleId);
  const value=id=>$(`#${id}`)?.value;
  Object.assign(draft,{
    nativeId:value('cal-native')||draft.nativeId||'',
    boardId:value('cal-board')||draft.boardId||'',
    worldId:value('cal-world')||draft.worldId||'',
    trackerIds:$$('.cal-tracker-check:checked').map(input=>input.value),
    boardMode:value('cal-board-mode')||draft.boardMode||'auto_legacy_16_marker',
    boardDefinition:value('cal-board-def')??draft.boardDefinition??'',
    intrinsics:value('cal-intrinsics')??draft.intrinsics??'',
    sampleEvery:value('cal-sample-every')??draft.sampleEvery??'2',
    maxDetections:value('cal-max-detections')??draft.maxDetections??'400',
    identityConfidence:value('cal-identity-confidence')||draft.identityConfidence||'MEDIUM',
    rawIdentity:toggleState('cal-raw-identity',Boolean(draft.rawIdentity)),
    participantId:value('cal-participant')||draft.participantId||calibrationParticipantForRole(roleId),
    mountVersion:value('cal-mount')||draft.mountVersion||'mount_v1',
    validSessions:value('cal-valid-sessions')||draft.validSessions||'',
    autoRegister:toggleState('cal-auto-register',draft.autoRegister!==false),
    autoName:toggleState('cal-auto-name',draft.autoName!==false),
    hardwareId:value('cal-hardware')??draft.hardwareId??'',
    hardwareOverride:toggleState('cal-hardware-override',Boolean(draft.hardwareOverride)),
    profileName:value('cal-name')??draft.profileName??'',
    outputG3dcal:toggleState('cal-output-g3dcal',draft.outputG3dcal!==false),
  });
}
function clips() { return values(session()?.clips).sort(byCreated); }
function artifacts(kinds = null) { const rows = values(session()?.artifacts).sort(byCreated); return kinds ? rows.filter(a => kinds.includes(a.kind)) : rows; }
function domains({includeCalibration=false} = {}) {
  const calibrationAssetIds = new Set(calibrationAssets().map(asset => asset.id));
  return values(session()?.time_domains).filter(domain => includeCalibration || !calibrationAssetIds.has(domain.asset_id));
}
function maps() { return values(session()?.time_maps); }
function jobs() { return values(project()?.jobs).sort((a,b) => String(b.created_at).localeCompare(String(a.created_at))); }
function selectedSessionId() { return project()?.selected_session_id || ''; }
function jobResultArtifactId(job) {
  const result=job?.result||{};
  if(result.artifact&&typeof result.artifact==='object'&&result.artifact.id)return String(result.artifact.id);
  for(const key of ['artifact_id','body_aoi_artifact_id','face_aoi_artifact_id','target_fusion_artifact_id','output_artifact_id']){
    if(result[key])return String(result[key]);
  }
  return '';
}
function bodyGeometryIntegrityForArtifact(artifact) {
  const qc=artifact?.qc||{};
  const applySampleClockGate=raw=>{
    const report={...(raw||{})};
    const clock=String(qc.body_geometry_clock||qc.timebase_summary?.body_geometry_clock||artifact?.parameters?.body_geometry_clock||report.body_geometry_clock||'').trim();
    report.body_geometry_clock=clock||'legacy_video_frame_clock';
    report.sample_clock_verified=clock==='gaze_sample_timestamp';
    if(report.sample_clock_verified)return report;
    const failures=[...(report.failures||[])];
    if(!failures.some(row=>row?.code==='legacy_video_frame_body_geometry_clock'))failures.push({code:'legacy_video_frame_body_geometry_clock',message:'This Body AOI used video-frame body snapshots rather than self/partner C3D interpolated at every gaze timestamp. Regenerate it with v0.21.52 or later.',entity_relation:'all',entity_id:'all'});
    return {...report,status:'failed',review_ready:false,fusion_ready:false,enforcement_outcome:'sample_clock_regeneration_required',failure_count:failures.length,failures,recommended_actions:[...(report.recommended_actions||[]),'Regenerate Body AOI using the gaze-sample geometry clock.']};
  };
  const current=qc.body_geometry_integrity;
  if(current&&typeof current==='object'&&current.schema_version)return applySampleClockGate(current);
  const diagnostics=Array.isArray(qc.body_diagnostics)?qc.body_diagnostics:[];
  const expected=['partner',...(qc.include_self_body?['self']:[])];
  const minimumTorso=0.80,minimumBody=0.90;
  if(!diagnostics.length){
    return {schema_version:'oculostudio-body-geometry-integrity-1.0',policy:'legacy',legacy:true,status:'failed',enforcement_outcome:'legacy_not_review_ready',gate_passed:false,review_ready:false,fusion_ready:false,minimum_torso_coverage:minimumTorso,minimum_body_frame_coverage:minimumBody,expected_relations:expected,failure_count:1,warning_count:0,failures:[{code:'legacy_geometry_diagnostics_missing',message:'This Body AOI predates verifiable torso geometry diagnostics.',entity_relation:'unknown',entity_id:'unknown'}],warnings:[],entities:[],recommended_actions:['Regenerate Body AOI with the current torso integrity gate.']};
  }
  const failures=[],warnings=[];
  const seen=new Set();
  const safeInt=value=>Math.max(0,Number.parseInt(value||0,10)||0);
  const coverage=(a,b)=>b>0?a/b:0;
  const entities=diagnostics.map(raw=>{
    const relation=String(raw?.entity_relation||'other').toLowerCase();
    const required=expected.includes(relation);if(required)seen.add(relation);
    const entityId=String(raw?.entity_id||'unknown_entity');
    const expectedFrames=safeInt(raw?.scene_frame_count),anyFrames=safeInt(raw?.frames_with_any_primitive),torsoFrames=safeInt(raw?.torso_primitive_frame_count),pelvisFrames=safeInt(raw?.pelvis_primitive_frame_count),torsoDerived=safeInt(raw?.torso_derived_frame_count),torsoDirect=safeInt(raw?.torso_direct_frame_count);
    const bodyCoverage=coverage(anyFrames,expectedFrames),torsoCoverage=coverage(torsoFrames,expectedFrames),pelvisCoverage=coverage(pelvisFrames,expectedFrames),fallback=coverage(torsoDerived,torsoFrames);
    const issues=[];
    const fail=(code,message)=>{issues.push(code);failures.push({code,message,entity_id:entityId,entity_relation:relation,source_asset_id:raw?.source_asset_id||null});};
    const warn=(code,message)=>{issues.push(code);warnings.push({code,message,entity_id:entityId,entity_relation:relation,source_asset_id:raw?.source_asset_id||null});};
    if(required){
      if(expectedFrames<=0)fail('no_expected_scene_frames',`${relation} ${entityId} has no aligned scene frames.`);
      else if(anyFrames<=0)fail('no_body_geometry',`${relation} ${entityId} produced no body primitives.`);
      else if(bodyCoverage<minimumBody)fail('body_timeline_coverage_below_threshold',`${relation} ${entityId} body timeline coverage is ${(bodyCoverage*100).toFixed(1)}%; minimum is ${(minimumBody*100).toFixed(0)}%.`);
      if(torsoFrames<=0)fail('torso_missing',`${relation} ${entityId} produced no torso primitive.`);
      else if(torsoCoverage<minimumTorso)fail('torso_coverage_below_threshold',`${relation} ${entityId} torso coverage is ${(torsoCoverage*100).toFixed(1)}%; minimum is ${(minimumTorso*100).toFixed(0)}%.`);
    }
    if(torsoFrames>0&&pelvisFrames<=0)warn('pelvis_surface_missing',`${relation} ${entityId} has torso but no pelvis surface.`);
    if(torsoFrames>0&&fallback>=0.95)warn('torso_almost_entirely_derived',`${relation} ${entityId} torso is almost entirely reconstructed from fallback landmarks.`);
    return {entity_id:entityId,entity_relation:relation,source_asset_id:raw?.source_asset_id||null,source_asset_label:raw?.source_asset_label||null,source_file_name:raw?.source_file_name||null,required,status:issues.some(code=>['no_expected_scene_frames','no_body_geometry','body_timeline_coverage_below_threshold','torso_missing','torso_coverage_below_threshold'].includes(code))?'failed':issues.length?'warning':'pass',expected_frame_count:expectedFrames,frames_with_any_primitive:anyFrames,body_frame_coverage_fraction:bodyCoverage,torso_primitive_frame_count:torsoFrames,torso_coverage_fraction:torsoCoverage,torso_direct_frame_count:torsoDirect,torso_derived_frame_count:torsoDerived,torso_fallback_fraction:fallback,pelvis_primitive_frame_count:pelvisFrames,pelvis_coverage_fraction:pelvisCoverage,issue_codes:issues,unresolved_landmarks:raw?.unresolved_landmarks||{},landmark_matches:raw?.landmark_matches||{},landmark_valid_frame_counts:raw?.landmark_valid_frame_counts||{},landmark_fallback_counts:raw?.landmark_fallback_counts||{},marker_protocol:raw?.marker_protocol||{},marker_prefix_used:raw?.marker_prefix_used||null};
  });
  expected.filter(relation=>!seen.has(relation)).forEach(relation=>{const message=`No geometry diagnostics were produced for required relation ${relation}.`;failures.push({code:'required_relation_geometry_missing',message,entity_id:`missing_${relation}`,entity_relation:relation,source_asset_id:null});entities.push({entity_id:`missing_${relation}`,entity_relation:relation,required:true,status:'failed',expected_frame_count:0,frames_with_any_primitive:0,body_frame_coverage_fraction:0,torso_primitive_frame_count:0,torso_coverage_fraction:0,torso_direct_frame_count:0,torso_derived_frame_count:0,torso_fallback_fraction:0,pelvis_primitive_frame_count:0,pelvis_coverage_fraction:0,issue_codes:['required_relation_geometry_missing']});});
  const gatePassed=failures.length===0;
  return applySampleClockGate({schema_version:'oculostudio-body-geometry-integrity-1.0',policy:'legacy',legacy:true,status:failures.length?'failed':warnings.length?'warning':'pass',enforcement_outcome:gatePassed?'passed':'legacy_not_review_ready',gate_passed:gatePassed,review_ready:gatePassed,fusion_ready:gatePassed,minimum_torso_coverage:minimumTorso,minimum_body_frame_coverage:minimumBody,expected_relations:expected,failure_count:failures.length,warning_count:warnings.length,failures,warnings,entities,recommended_actions:gatePassed?[]:['Verify self/partner C3D assignment and marker prefixes.','Regenerate Body AOI before human review or target fusion.']});
}
function bodyArtifactReviewReady(artifact) { return Boolean(bodyGeometryIntegrityForArtifact(artifact).review_ready); }
function bodyArtifactFusionReady(artifact) { return Boolean(bodyGeometryIntegrityForArtifact(artifact).fusion_ready); }
function bodyArtifactById(id) { return id?artifacts(['body_aoi']).find(row=>String(row.id)===String(id))||null:null; }
function scheduleAOIReviewScroll() {
  const reveal=()=>document.querySelector('#aoi-review-shell')?.scrollIntoView({behavior:'smooth',block:'start'});
  requestAnimationFrame(()=>{reveal();setTimeout(reveal,180);});
}
function openAOIReviewResult(mode,artifactId,{scroll=true,openManual=false}={}) {
  const id=String(artifactId||'');
  if(!id)return;
  const requestedMode=String(mode||'');
  appState.aoiReviewInitialMode=requestedMode;
  appState.aoiReviewInitialArtifactId=id;
  appState.aoiReviewScrollRequested=Boolean(scroll);
  appState.aoiReviewOpenManualMode=openManual?requestedMode:'';
  appState.selectedArtifactId=id;
  appState.page='targets';
  render();
}


const RESULT_STAGE = {
  native_reference_trim:'01 Trim', trimmed_native:'01 Trim', trimmed_timeseries:'01 Trim', trimmed_face_mapper:'01 Trim',
  face_mapper_filter:'02 FaceMapper QC', privacy_mask_preview:'02 Privacy mask preview', aligned_clip_package:'02 Aligned clip export', aligned_facial_video:'02 Aligned facial video', privacy_video:'03 Privacy', privacy_clip:'03 Privacy', participant_association:'04 Participant Association', face_aoi:'05 Face AOI', frame_map:'06 Sync',
  calibration_profile:'04 Calibration profile', etm_role_assignment:'05 Calibration assignment', calibration_acceptance:'06 Calibration acceptance', gaze_rays:'07 Gaze 3D',
  body_aoi:'09 Body AOI', object_aoi:'10 Object AOI', target_fusion:'11 Fusion', semantic_targets:'11 Fusion',
  dyadic_attention:'12 Dyadic Attention', aoi_review_video:'13 Review video', qc_report:'99 QC', results_package:'99 Export package'
};

function artifactStage(kind) { return RESULT_STAGE[kind] || '98 Other'; }
function dirnameLocal(path) {
  const text=String(path||'').replace(/[\\/]+$/,'');
  const index=Math.max(text.lastIndexOf('\\'),text.lastIndexOf('/'));
  return index>0 ? text.slice(0,index) : text;
}
function commonArtifactRoot(artifact) {
  const dirs=(artifact?.files||[]).map(f=>dirnameLocal(f.path)).filter(Boolean);
  if(!dirs.length)return '';
  const sep=dirs[0].includes('\\')?'\\':'/';
  const parts=dirs.map(x=>x.split(/[\\/]+/));
  const common=[];
  for(let i=0;i<Math.min(...parts.map(x=>x.length));i++){
    const value=parts[0][i];
    if(parts.every(row=>String(row[i]).toLowerCase()===String(value).toLowerCase()))common.push(value);else break;
  }
  if(!common.length)return dirs[0];
  return common.join(sep) || sep;
}
function artifactPrimaryFile(artifact) {
  const files=artifact?.files||[];
  const ranked=[/README\.html$/i,/report.*\.html$/i,/overlay.*\.mp4$/i,/\.mp4$/i,/\.parquet$/i,/\.csv$/i,/\.html$/i,/manifest.*\.json$/i,/\.json$/i];
  for(const pattern of ranked){const hit=files.find(f=>pattern.test(f.path||''));if(hit)return hit;}
  return files[0]||null;
}
function extractCode(text,prefix){const m=String(text||'').match(new RegExp(`(?:^|[^A-Za-z0-9])(${prefix}\\d{1,6})(?![A-Za-z0-9])`,'i'));return m?m[1].toUpperCase():'';}
function inferredResultCodes(rows=[]) {
  const texts=[session()?.label||'',...rows.flatMap(a=>[a.label||'',a.clip_id||'',...((a.files||[]).map(f=>f.path||''))])];
  return {s:texts.map(x=>extractCode(x,'S')).find(Boolean)||'',i:texts.map(x=>extractCode(x,'I')).find(Boolean)||'',r:texts.map(x=>extractCode(x,'R')).find(Boolean)||''};
}
function resultArtifactRows(){return artifacts().filter(a=>!['thumbnail','results_package'].includes(a.kind));}
function artifactScalarTexts(value,depth=0){
  if(value==null||depth>3)return [];
  if(Array.isArray(value))return value.flatMap(item=>artifactScalarTexts(item,depth+1));
  if(typeof value==='object')return Object.values(value).flatMap(item=>artifactScalarTexts(item,depth+1));
  return ['string','number'].includes(typeof value)?[String(value)]:[];
}
function artifactRoleCode(artifact){
  if(!artifact)return 'SHARED';
  const primaryTexts=[artifact.role_id,artifact.participant_id,artifact.label,artifact.clip_id,artifact.root_path,...((artifact.files||[]).map(file=>file.path||''))].filter(Boolean);
  const primaryRoles=[...new Set(primaryTexts.map(text=>extractCode(text,'R')).filter(Boolean))];
  if(primaryRoles.length===1)return primaryRoles[0];
  if(primaryRoles.length>1)return 'SHARED';
  const secondaryTexts=[...artifactScalarTexts(artifact.parameters),...artifactScalarTexts(artifact.provenance)];
  const secondaryRoles=[...new Set(secondaryTexts.map(text=>extractCode(text,'R')).filter(Boolean))];
  return secondaryRoles.length===1?secondaryRoles[0]:'SHARED';
}
function artifactRoleLabel(role){return role==='SHARED'?'Shared / session':role;}
function artifactSizeBytes(artifact){
  if(Number.isFinite(Number(artifact?.registered_size_bytes)))return Number(artifact.registered_size_bytes)||0;
  return (artifact?.files||[]).reduce((sum,file)=>sum+(Number(file.size_bytes)||0),0);
}
function groupedArtifacts(rows=[]){
  const roles=new Map();
  rows.forEach(row=>{
    const role=artifactRoleCode(row),stage=artifactStage(row.kind);
    if(!roles.has(role))roles.set(role,new Map());
    const stages=roles.get(role);if(!stages.has(stage))stages.set(stage,[]);stages.get(stage).push(row);
  });
  const roleOrder=(a,b)=>{
    if(a==='SHARED')return b==='SHARED'?0:1;if(b==='SHARED')return -1;
    return a.localeCompare(b,undefined,{numeric:true,sensitivity:'base'});
  };
  return [...roles.entries()].sort(([a],[b])=>roleOrder(a,b)).map(([role,stages])=>({
    role,rows:[...stages.values()].flat(),
    stages:[...stages.entries()].sort(([a],[b])=>a.localeCompare(b,undefined,{numeric:true})).map(([stage,stageRows])=>({stage,rows:stageRows})),
  }));
}
function resultGroupIsOpen(scope,key,defaultOpen=false){
  const token=`${scope}:${key}`,state=appState.resultGroupState||{};
  return Object.prototype.hasOwnProperty.call(state,token)?Boolean(state[token]):Boolean(defaultOpen);
}
function resultGroupOpenAttr(scope,key,defaultOpen=false){return resultGroupIsOpen(scope,key,defaultOpen)?'open':'';}
function resultGroupingToolbar(scope,label='Grouped by role and workflow stage'){
  return `<div class="result-group-toolbar"><span><strong>${esc(label)}</strong><small>R01 / R02 / Shared</small></span><div><button class="row-button" data-result-group-action="expand" data-result-group-scope="${esc(scope)}">Expand all</button><button class="row-button" data-result-group-action="collapse" data-result-group-scope="${esc(scope)}">Collapse all</button></div></div>`;
}
function resultArtifactBrowserMarkup(rows,selected){
  if(!rows.length)return '<div class="empty-tree">No processed outputs yet.</div>';
  const groups=groupedArtifacts(rows),selectedRole=artifactRoleCode(selected);
  return `${resultGroupingToolbar('browser')}<div class="result-artifact-groups">${groups.map((group,index)=>{
    const count=group.rows.length,size=group.rows.reduce((sum,row)=>sum+artifactSizeBytes(row),0);
    const defaultOpen=group.role===selectedRole||(!selected&&index===0)||groups.length===1;
    return `<details class="result-role-group" data-result-group-scope="browser" data-result-group-key="${esc(group.role)}" ${resultGroupOpenAttr('browser',group.role,defaultOpen)}><summary><span class="result-role-summary-main"><b>${esc(artifactRoleLabel(group.role))}</b><small>${group.stages.length} workflow stage${group.stages.length===1?'':'s'}</small></span><span class="result-role-summary-meta">${count} artifact${count===1?'':'s'} · ${fmtBytes(size)}</span></summary><div class="result-role-body">${group.stages.map(stage=>{
      const stageSize=stage.rows.reduce((sum,row)=>sum+artifactSizeBytes(row),0);
      return `<section class="result-stage-section"><div class="result-stage-heading"><span>${esc(stage.stage)}</span><small>${stage.rows.length} · ${fmtBytes(stageSize)}</small></div>${stage.rows.map(a=>`<button class="result-artifact-row ${a.id===selected?.id?'selected':''}" data-result-artifact="${esc(a.id)}"><span class="result-stage">${esc(a.kind.replaceAll('_',' '))}</span><span class="result-artifact-copy"><strong>${esc(a.label)}</strong><small>${a.files?.length||0} files · ${fmtBytes(artifactSizeBytes(a))}</small></span><span>›</span></button>`).join('')}</section>`;
    }).join('')}</div></details>`;
  }).join('')}</div>`;
}
function resultPackageChecklistMarkup(rows,selected){
  if(!rows.length)return '<span class="muted small">No derived artifacts are available.</span>';
  const groups=groupedArtifacts(rows),selectedRole=artifactRoleCode(selected);
  return `${resultGroupingToolbar('package','Grouped export selection')}<div class="result-package-groups">${groups.map((group,index)=>{
    const defaultOpen=group.role===selectedRole||(!selected&&index===0)||groups.length===1;
    return `<details class="result-package-role-group" data-result-group-scope="package" data-result-group-key="${esc(group.role)}" ${resultGroupOpenAttr('package',group.role,defaultOpen)}><summary><span><b>${esc(artifactRoleLabel(group.role))}</b><small>${group.rows.length} artifact${group.rows.length===1?'':'s'}</small></span><span>${fmtBytes(group.rows.reduce((sum,row)=>sum+artifactSizeBytes(row),0))}</span></summary><div class="result-package-role-body">${group.stages.map(stage=>`<section class="result-package-stage"><div class="result-package-stage-label"><span>${esc(stage.stage)}</span><small>${stage.rows.length}</small></div>${stage.rows.map(a=>`<label class="check-row result-package-row"><input type="checkbox" class="result-package-check" value="${esc(a.id)}" ${a.id===selected?.id?'checked':''}><span><strong>${esc(a.label)}</strong><small>${esc(a.kind.replaceAll('_',' '))} · ${fmtBytes(artifactSizeBytes(a))}</small></span></label>`).join('')}</section>`).join('')}</div></details>`;
  }).join('')}</div>`;
}
function bindResultGroupControls(root=document){
  $$('details[data-result-group-scope]',root).forEach(details=>{details.ontoggle=()=>{const token=`${details.dataset.resultGroupScope}:${details.dataset.resultGroupKey}`;appState.resultGroupState={...(appState.resultGroupState||{}),[token]:details.open};};});
  $$('[data-result-group-action]',root).forEach(button=>{button.onclick=()=>{const scope=button.dataset.resultGroupScope,open=button.dataset.resultGroupAction==='expand';$$('details[data-result-group-scope]',root).filter(details=>details.dataset.resultGroupScope===scope).forEach(details=>{details.open=open;const token=`${scope}:${details.dataset.resultGroupKey}`;appState.resultGroupState={...(appState.resultGroupState||{}),[token]:open};});};});
}

function cleanupSelectionSet(){return new Set(appState.artifactCleanupSelection||[]);}
function cleanupEntryKey(type,value){return `${type}:${value}`;}
function cleanupArtifactReason(row){
  const reasons=row?.suggested_reasons||[];
  if(reasons.length)return reasons.join(' ');
  if(row?.dependent_ids?.length)return `Used by ${row.dependent_ids.length} downstream artifact${row.dependent_ids.length===1?'':'s'}.`;
  return 'Leaf artifact. Safe to move when no downstream result is needed.';
}
function artifactCleanupMarkup(){
  const plan=appState.artifactCleanupPlan;
  if(appState.artifactCleanupLoading)return `<div class="cleanup-loading"><span class="mini-spinner"></span><strong>Analysing artifact lineage and project-controlled folders…</strong></div>`;
  if(!plan)return `<div class="empty-tree">Cleanup analysis has not been loaded.</div>`;
  const selected=cleanupSelectionSet(),summary=plan.summary||{},trash=plan.trash||{};
  const rows=plan.artifacts||[];
  const selectedArtifactIds=[...selected].filter(key=>key.startsWith('artifact:')).map(key=>key.slice('artifact:'.length));
  const selectedPlanRows=rows.filter(row=>selectedArtifactIds.includes(row.id));
  const selectedViewRow=rows.find(row=>row.id===appState.selectedArtifactId);
  const preferredRoles=new Set((selectedPlanRows.length?selectedPlanRows:(selectedViewRow?[selectedViewRow]:[])).map(artifactRoleCode));
  const artifactGroups=rows.length?`${resultGroupingToolbar('cleanup','Grouped cleanup inventory')}<div class="cleanup-artifact-groups">${groupedArtifacts(rows).map((group,index)=>{
    const suggestedCount=group.rows.filter(row=>row.suggested).length,blockedCount=group.rows.filter(row=>!row.safe_to_remove).length;
    const defaultOpen=preferredRoles.has(group.role)||(!preferredRoles.size&&index===0)||groupedArtifacts(rows).length===1;
    return `<details class="cleanup-role-group" data-result-group-scope="cleanup" data-result-group-key="${esc(group.role)}" ${resultGroupOpenAttr('cleanup',group.role,defaultOpen)}><summary><span class="cleanup-role-summary-main"><b>${esc(artifactRoleLabel(group.role))}</b><small>${group.stages.length} workflow stage${group.stages.length===1?'':'s'}</small></span><span class="cleanup-role-summary-meta">${group.rows.length} artifacts · ${fmtBytes(group.rows.reduce((sum,row)=>sum+artifactSizeBytes(row),0))}${suggestedCount?` · ${suggestedCount} suggested`:''}${blockedCount?` · ${blockedCount} linked`:''}</span></summary><div class="cleanup-table-wrap"><table class="cleanup-table"><thead><tr><th></th><th>Processed artifact</th><th>Status</th><th>Disk size</th><th>Lineage</th></tr></thead><tbody>${group.stages.map(stage=>`<tr class="cleanup-stage-row"><td colspan="5"><span>${esc(stage.stage)}</span><small>${stage.rows.length} artifact${stage.rows.length===1?'':'s'} · ${fmtBytes(stage.rows.reduce((sum,row)=>sum+artifactSizeBytes(row),0))}</small></td></tr>${stage.rows.map(row=>{
      const key=cleanupEntryKey('artifact',row.id),checked=selected.has(key)?'checked':'';
      const badges=[row.suggested?'<span class="cleanup-badge suggested">Suggested</span>':'',row.safe_to_remove?'<span class="cleanup-badge safe">Leaf</span>':`<span class="cleanup-badge blocked">Used by ${row.dependent_ids?.length||0}</span>`,row.missing_file_count?`<span class="cleanup-badge missing">${row.missing_file_count} missing</span>`:''].join('');
      const dependents=(row.dependents||[]).map(dep=>dep.label).slice(0,3).join(', ');
      return `<tr class="cleanup-row ${row.suggested?'is-suggested':''}"><td><input type="checkbox" class="cleanup-check" data-cleanup-key="${esc(key)}" ${checked}></td><td><strong>${esc(row.label)}</strong><small>${esc(row.kind.replaceAll('_',' '))} · ${esc(row.producer_version||'unknown version')}</small><span class="cleanup-reason">${esc(cleanupArtifactReason(row))}</span>${dependents?`<span class="cleanup-dependents">Downstream: ${esc(dependents)}${(row.dependents||[]).length>3?'…':''}</span>`:''}</td><td>${badges}</td><td><strong>${fmtBytes(row.registered_size_bytes)}</strong><small>${row.registered_file_count||0} files</small></td><td>${row.dependent_ids?.length?`<button class="row-button cleanup-select-chain" data-artifact-id="${esc(row.id)}">Select chain</button>`:'<span class="muted small">—</span>'}</td></tr>`;
    }).join('')}`).join('')}</tbody></table></div></details>`;
  }).join('')}</div>`:'<div class="empty-tree">No active artifacts.</div>';
  const orphans=plan.orphans||[];
  const orphanRows=orphans.length?orphans.map(row=>{
    const key=cleanupEntryKey('orphan',row.path),checked=selected.has(key)?'checked':'';
    return `<tr class="cleanup-row orphan-row"><td><input type="checkbox" class="cleanup-check" data-cleanup-key="${esc(key)}" ${checked}></td><td><strong>${esc(row.name)}</strong><small>${esc(row.relative_path)} · ${esc(row.entry_type)}</small><span class="cleanup-reason">${esc(row.reason||'Unreferenced project output')}</span></td><td><span class="cleanup-badge orphan">Orphan</span></td><td><strong>${fmtBytes(row.size_bytes)}</strong><small>${row.file_count||0} files</small></td><td><button class="row-button result-reveal-file" data-path="${esc(row.path)}">Show</button></td></tr>`;
  }).join(''):`<tr><td colspan="5"><div class="empty-tree">${plan.scan_orphans?'No unreferenced files or folders were found.':'Run “Scan output folders” to find unregistered folders from failed or abandoned processing.'}</div></td></tr>`;
  const selectedCount=selected.size;
  return `<div class="cleanup-summary-grid">
    <div class="cleanup-summary"><span>Active artifacts</span><strong>${summary.artifact_count||0}</strong><small>${fmtBytes(summary.artifact_size_bytes||0)} registered</small></div>
    <div class="cleanup-summary"><span>Suggested cleanup</span><strong>${summary.suggested_artifact_count||0}</strong><small>duplicates or broken records</small></div>
    <div class="cleanup-summary"><span>Unreferenced outputs</span><strong>${plan.scan_orphans?(summary.orphan_count||0):'—'}</strong><small>${plan.scan_orphans?fmtBytes(summary.orphan_size_bytes||0):'not scanned'}</small></div>
    <div class="cleanup-summary"><span>Project Trash</span><strong>${trash.entry_count||0}</strong><small>${fmtBytes(trash.size_bytes||0)}</small></div>
  </div>
  <div class="cleanup-toolbar"><div class="cleanup-toolbar-group"><button class="soft-button" id="cleanup-refresh">Refresh lineage</button><button class="soft-button" id="cleanup-scan-orphans">Scan output folders</button><button class="row-button" id="cleanup-select-suggested">Select suggested</button><button class="row-button" id="cleanup-clear-selection">Clear</button></div><div class="cleanup-toolbar-group"><button class="soft-button" id="cleanup-open-trash">Open Project Trash</button><button class="danger-button" id="cleanup-purge-trash" ${trash.entry_count?'':'disabled'}>Permanently empty Trash</button></div></div>
  <div class="cleanup-warning"><strong>Raw recordings are never deleted.</strong> Moving an artifact removes its active registry record and moves project-controlled files from <code>derived/</code>, <code>reports/</code> or <code>exports/</code> into <code>.trash</code>. Upstream artifacts are blocked unless their full downstream chain is selected.</div>
  ${artifactGroups}
  <details class="cleanup-orphans" ${plan.scan_orphans&&orphans.length?'open':''}><summary>Unreferenced files and folders ${plan.scan_orphans?`(${orphans.length})`:''}</summary><div class="cleanup-table-wrap"><table class="cleanup-table"><thead><tr><th></th><th>Project output</th><th>Status</th><th>Disk size</th><th>Action</th></tr></thead><tbody>${orphanRows}</tbody></table></div></details>
  <div class="cleanup-footer"><span><strong>${selectedCount}</strong> selected</span><button class="danger-button" id="cleanup-move-selected" ${selectedCount?'':'disabled'}>Move selected to Project Trash</button></div>`;
}
function renderArtifactCleanupPanel(){
  const root=$('#artifact-cleanup-root');if(!root)return;
  root.innerHTML=artifactCleanupMarkup();
  bindArtifactCleanupActions();
}
async function loadArtifactCleanupPlan(scanOrphans=appState.artifactCleanupScanOrphans){
  const sessionId=selectedSessionId();if(!sessionId)return;
  appState.artifactCleanupLoading=true;renderArtifactCleanupPanel();
  try{
    const plan=await api(`/api/artifacts/${encodeURIComponent(sessionId)}/cleanup-plan?scan_orphans=${scanOrphans?'true':'false'}`);
    if(sessionId!==selectedSessionId())return;
    appState.artifactCleanupPlan=plan;appState.artifactCleanupPlanSessionId=sessionId;appState.artifactCleanupScanOrphans=Boolean(scanOrphans);
    const valid=new Set([...(plan.artifacts||[]).map(row=>cleanupEntryKey('artifact',row.id)),...(plan.orphans||[]).map(row=>cleanupEntryKey('orphan',row.path))]);
    appState.artifactCleanupSelection=(appState.artifactCleanupSelection||[]).filter(key=>valid.has(key));
  }catch(error){showError(error,'Could not analyse processed artifacts');}
  finally{appState.artifactCleanupLoading=false;renderArtifactCleanupPanel();}
}
function syncCleanupSelection(){
  appState.artifactCleanupSelection=$$('.cleanup-check:checked').map(input=>input.dataset.cleanupKey);
  renderArtifactCleanupPanel();
}
function selectCleanupChain(artifactId){
  const row=(appState.artifactCleanupPlan?.artifacts||[]).find(item=>item.id===artifactId);if(!row)return;
  const selected=cleanupSelectionSet();selected.add(cleanupEntryKey('artifact',artifactId));
  (row.transitive_dependent_ids||[]).forEach(id=>selected.add(cleanupEntryKey('artifact',id)));
  appState.artifactCleanupSelection=[...selected];renderArtifactCleanupPanel();
}
function confirmArtifactCleanup(){
  const selected=cleanupSelectionSet();
  const artifactIds=[...selected].filter(key=>key.startsWith('artifact:')).map(key=>key.slice('artifact:'.length));
  const orphanPaths=[...selected].filter(key=>key.startsWith('orphan:')).map(key=>key.slice('orphan:'.length));
  if(!artifactIds.length&&!orphanPaths.length)return;
  const plan=appState.artifactCleanupPlan||{};
  const artifactLabels=(plan.artifacts||[]).filter(row=>artifactIds.includes(row.id)).map(row=>row.label);
  modal('Move processed outputs to Project Trash',`<div class="warning-card"><strong>This does not touch raw source recordings.</strong><p>${artifactIds.length} artifact record${artifactIds.length===1?'':'s'} and ${orphanPaths.length} unreferenced path${orphanPaths.length===1?'':'s'} will be removed from the active project and moved to <code>.trash</code>.</p></div>${artifactLabels.length?`<div class="cleanup-confirm-list">${artifactLabels.slice(0,12).map(label=>`<span>${esc(label)}</span>`).join('')}${artifactLabels.length>12?`<span>…and ${artifactLabels.length-12} more</span>`:''}</div>`:''}<p>OculoStudio will refuse the operation if any active downstream artifact is not included.</p><div class="form-actions"><button class="soft-button" id="cleanup-cancel">Cancel</button><button class="danger-button" id="cleanup-confirm">Move to Project Trash</button></div>`,(root,close)=>{
    $('#cleanup-cancel',root).onclick=close;
    $('#cleanup-confirm',root).onclick=async()=>{try{const result=await api('/api/artifacts/cleanup',{method:'POST',body:{session_id:selectedSessionId(),artifact_ids:artifactIds,orphan_paths:orphanPaths}});close();appState.artifactCleanupSelection=[];if(artifactIds.includes(appState.selectedArtifactId))appState.selectedArtifactId=null;toast('Processed outputs moved to Project Trash',`${result.removed_artifacts?.length||0} artifacts · ${result.removed_orphans?.length||0} orphan paths`);await refreshState({quiet:false,fullRender:true});await loadArtifactCleanupPlan(appState.artifactCleanupScanOrphans);}catch(error){showError(error,'Cleanup was blocked');}};
  });
}
function confirmPurgeProjectTrash(){
  const projectName=project()?.name||'';
  modal('Permanently empty Project Trash',`<div class="warning-card"><strong>This operation cannot be undone inside OculoStudio.</strong><p>Raw recordings remain untouched, but all files already moved into <code>.trash</code> will be permanently deleted.</p></div><div class="field"><label>Type the exact project name to confirm</label><input id="purge-trash-confirm" placeholder="${esc(projectName)}"></div><div class="form-actions"><button class="soft-button" id="purge-trash-cancel">Cancel</button><button class="danger-button" id="purge-trash-submit" disabled>Permanently delete Trash</button></div>`,(root,close)=>{
    const input=$('#purge-trash-confirm',root),button=$('#purge-trash-submit',root);input.oninput=()=>button.disabled=input.value!==projectName;$('#purge-trash-cancel',root).onclick=close;
    button.onclick=async()=>{try{const result=await api('/api/trash/purge',{method:'POST',body:{confirmation:input.value}});close();toast('Project Trash emptied',`${fmtBytes(result.purged?.size_bytes||0)} permanently removed`);await loadArtifactCleanupPlan(appState.artifactCleanupScanOrphans);}catch(error){showError(error,'Could not empty Project Trash');}};
  });
}
function bindArtifactCleanupActions(){
  bindResultGroupControls($('#artifact-cleanup-root')||document);
  $$('.cleanup-check').forEach(input=>input.onchange=syncCleanupSelection);
  $$('.cleanup-select-chain').forEach(button=>button.onclick=()=>selectCleanupChain(button.dataset.artifactId));
  $$('.cleanup-orphans .result-reveal-file').forEach(button=>button.onclick=()=>revealLocalPath(button.dataset.path));
  $('#cleanup-refresh')?.addEventListener('click',()=>loadArtifactCleanupPlan(appState.artifactCleanupScanOrphans));
  $('#cleanup-scan-orphans')?.addEventListener('click',()=>loadArtifactCleanupPlan(true));
  $('#cleanup-select-suggested')?.addEventListener('click',()=>{const selected=cleanupSelectionSet();(appState.artifactCleanupPlan?.artifacts||[]).filter(row=>row.suggested&&row.safe_to_remove).forEach(row=>selected.add(cleanupEntryKey('artifact',row.id)));(appState.artifactCleanupPlan?.orphans||[]).forEach(row=>selected.add(cleanupEntryKey('orphan',row.path)));appState.artifactCleanupSelection=[...selected];renderArtifactCleanupPanel();});
  $('#cleanup-clear-selection')?.addEventListener('click',()=>{appState.artifactCleanupSelection=[];renderArtifactCleanupPanel();});
  $('#cleanup-move-selected')?.addEventListener('click',confirmArtifactCleanup);
  $('#cleanup-open-trash')?.addEventListener('click',()=>openLocalPath(appState.artifactCleanupPlan?.trash?.path||joinLocalPath(project()?.root,'.trash')));
  $('#cleanup-purge-trash')?.addEventListener('click',confirmPurgeProjectTrash);
}

function optionRows(rows, selected = '', empty = 'Select…', labelFn = x => x.label) {
  return `<option value="">${esc(empty)}</option>` + rows.map(row => `<option value="${esc(row.id)}" ${row.id === selected ? 'selected' : ''}>${esc(labelFn(row))}</option>`).join('');
}

async function refreshState({quiet = true, fullRender = false} = {}) {
  if (appState.polling) {
    if (fullRender || !quiet) appState.pendingFullRefresh = true;
    return;
  }
  appState.polling = true;
  try {
    appState.data = await api('/api/state');
    let completedPrivacyJob = null;
    let completedCalibrationImportJob = null;
    let completedCalibrationSetProfileJob = null;
    let completedBodyAOIJob = null;
    let completedAudioVideoSyncJob = null;
    if (appState.pendingPrivacyJobId) {
      const tracked = project()?.jobs?.[appState.pendingPrivacyJobId] || null;
      if (tracked && ['completed','failed','cancelled'].includes(String(tracked.status))) {
        completedPrivacyJob = tracked;
        appState.pendingPrivacyJobId = null;
      }
    }
    if (appState.pendingCalibrationImportJobId) {
      const tracked = project()?.jobs?.[appState.pendingCalibrationImportJobId] || null;
      if (tracked && ['completed','failed','cancelled'].includes(String(tracked.status))) {
        completedCalibrationImportJob = tracked;
        appState.pendingCalibrationImportJobId = null;
      }
    }
    if (appState.pendingCalibrationSetProfileJobId) {
      const tracked = project()?.jobs?.[appState.pendingCalibrationSetProfileJobId] || null;
      if (tracked && ['completed','failed','cancelled'].includes(String(tracked.status))) {
        completedCalibrationSetProfileJob = tracked;
        appState.pendingCalibrationSetProfileJobId = null;
      }
    }
    if (appState.pendingBodyAOIJobId) {
      const tracked = project()?.jobs?.[appState.pendingBodyAOIJobId] || null;
      if (tracked && ['completed','failed','cancelled'].includes(String(tracked.status))) {
        completedBodyAOIJob = tracked;
        appState.pendingBodyAOIJobId = null;
      }
    }
    if (appState.pendingAudioVideoSyncJobId) {
      const tracked = project()?.jobs?.[appState.pendingAudioVideoSyncJobId] || null;
      if (tracked && ['completed','failed','cancelled'].includes(String(tracked.status))) {
        completedAudioVideoSyncJob = tracked;
        appState.pendingAudioVideoSyncJobId = null;
      }
    }
    const completedGeometryJobs = [];
    for (const [stateKey, kind] of [['pendingETMAssignmentJobId','assignment'],['pendingCalibrationBindingJobId','binding'],['pendingC3DDefinedClipsJobId','clips'],['pendingGaze3DJobId','gaze']]) {
      const jobId=appState[stateKey];
      if(!jobId)continue;
      const tracked=project()?.jobs?.[jobId]||null;
      if(tracked && ['completed','failed','cancelled'].includes(String(tracked.status))){
        completedGeometryJobs.push({kind,job:tracked});
        appState[stateKey]=null;
      }
    }
    const s = session();
    if (s) {
      if (appState.selectedAssetId && !s.assets[appState.selectedAssetId]) appState.selectedAssetId = null;
      appState.selectedAssetIds=(appState.selectedAssetIds||[]).filter(id=>!!s.assets[id]);
      if (!appState.selectedAssetId) appState.selectedAssetId = values(s.assets)[0]?.id || null;
      if (appState.selectedClipId && !s.clips[appState.selectedClipId]) appState.selectedClipId = null;
      if (!appState.selectedClipId) appState.selectedClipId = clips()[0]?.id || null;
      if (appState.selectedArtifactId && !s.artifacts[appState.selectedArtifactId]) appState.selectedArtifactId = null;
      if (appState.artifactCleanupPlanSessionId && appState.artifactCleanupPlanSessionId !== s.id) {
        appState.artifactCleanupPlan = null;
        appState.artifactCleanupPlanSessionId = '';
        appState.artifactCleanupSelection = [];
        appState.artifactCleanupScanOrphans = false;
      }
    }

    if(completedBodyAOIJob?.status==='completed'){
      const artifactId=jobResultArtifactId(completedBodyAOIJob);
      if(artifactId){
        appState.aoiReviewInitialMode='body';
        appState.aoiReviewInitialArtifactId=artifactId;
        appState.selectedArtifactId=artifactId;
        if(appState.page==='targets'){
          appState.aoiReviewScrollRequested=true;
          fullRender=true;
        }
      }
    }

    // A background poll must never rebuild the active form: doing so would erase
    // text, selections, sliders and file paths while the user is editing them.
    // Polls therefore refresh only read-only status surfaces. Explicit actions
    // and the initial load request a full render.
    if (fullRender || !quiet) {
      render();
    } else {
      renderSessionCard();
      renderSidebar();
      renderMetrics();
      renderJobs();
    }
    if (completedPrivacyJob) {
      if (completedPrivacyJob.status === 'completed') {
        if (!(fullRender || !quiet) && appState.page === 'privacy') {
          const panel=$('#properties-content');
          const scrollTop=panel?.scrollTop||0;
          renderProperties();
          requestAnimationFrame(()=>{const next=$('#properties-content');if(next)next.scrollTop=scrollTop;});
        }
        const operation=String(completedPrivacyJob.operation||'');
        if(operation==='privacy_mask_preview')toast('Accepted + repair mask preview ready','Review green accepted, cyan internal repairs and orange edge repairs before generating the final Privacy video.');
        else if(operation==='privacy_final')toast('Synchronized Privacy video ready','The final fixed-FPS video and Native frame map are ready for AOI and attention analysis.');
        else if(operation==='privacy')toast('Reusable Privacy source ready','Open Advanced to derive one or more synchronized clips from it.');
        else toast('Derived Privacy clip ready','The synchronized clip is ready for downstream analysis.');
      } else {
        toast('Privacy processing did not complete',completedPrivacyJob.error||completedPrivacyJob.stage||'Review the failed job in Job Queue.',true);
      }
    }
    if (completedCalibrationImportJob) {
      if (completedCalibrationImportJob.status === 'completed') {
        try {
          if (appState.calibrationSourcePath) await scanCalibrationSourceSelection({render:false,quiet:true});
        } catch (_) { }
        if (!(fullRender || !quiet) && appState.page === 'calibration') {
          const panel=$('#properties-content');const scrollTop=panel?.scrollTop||0;
          renderWorkspace();renderProperties();
          requestAnimationFrame(()=>{const next=$('#properties-content');if(next)next.scrollTop=scrollTop;});
        }
        const result=completedCalibrationImportJob.result||{};
        const detail=`${result.imported_count||0} imported · ${result.reused_count||0} already registered${result.failed_count?` · ${result.failed_count} failed`:''}`;
        toast(result.failed_count?'Calibration import completed with warnings':'Calibration set imported',detail,Boolean(result.failed_count));
      } else {
        toast('Calibration import did not complete',completedCalibrationImportJob.error||completedCalibrationImportJob.stage||'Review the failed batch job.',true);
      }
    }
    if (completedCalibrationSetProfileJob) {
      if (completedCalibrationSetProfileJob.status === 'completed') {
        toast('Physical ETM profiles ready','ETM-A and ETM-B were built from the shared Calibration Set and propagated to assigned Sessions.');
      } else {
        toast('Calibration Set profile build failed',completedCalibrationSetProfileJob.error||completedCalibrationSetProfileJob.stage||'Open the profile builder for the exact failed stage.',true);
      }
    }
    if(completedBodyAOIJob){
      if(completedBodyAOIJob.status==='completed'){
        const artifactId=jobResultArtifactId(completedBodyAOIJob);
        if(artifactId){
          toast('Body AOI result ready',appState.page==='targets'?'The new artifact is selected in Body Review.':'Open Gaze Targets or use Review Body AOI in Job Queue.');
        }else{
          toast('Body AOI completed without a review link','The artifact exists, but this legacy job did not expose its result ID. Open Gaze Targets and select the newest Body AOI artifact.',true);
        }
      }else{
        toast('Body AOI did not complete',completedBodyAOIJob.error||completedBodyAOIJob.stage||'Review the failed job in Job Queue.',true);
      }
    }
    if(completedGeometryJobs.length){
      if(!(fullRender||!quiet) && ['gaze3d','calibration'].includes(appState.page)){
        const panel=$('#properties-content');
        const scrollTop=panel?.scrollTop||0;
        renderWorkspace();
        renderProperties();
        requestAnimationFrame(()=>{const next=$('#properties-content');if(next)next.scrollTop=scrollTop;});
      }
      completedGeometryJobs.forEach(({kind,job})=>{
        if(job.status==='completed'){
          if(kind==='assignment')toast('ETM wearer assignment ready','Phase A is complete. This result is independent of calibration profiles.');
          else if(kind==='binding')toast('Calibration handoff ready','Physical ETM profiles were bound to the immutable wearer assignment.');
          else if(kind==='clips')toast('C3D-defined aligned clips ready','R01 and R02 Native-reference intervals were created from the common body-C3D window; no manual start/end values were used.');
          else toast('Calibrated gaze rays ready','The interactive preview now includes sampled world-space gaze rays.');
        }else{
          toast(kind==='assignment'?'ETM assignment did not complete':kind==='binding'?'Profile binding did not complete':kind==='clips'?'C3D-defined clips did not complete':'Gaze 3D did not complete',job.error||job.stage||'Review the failed job in Job Queue.',true);
        }
      });
    }
    if(completedAudioVideoSyncJob){
      if(completedAudioVideoSyncJob.status==='completed'){
        const result=completedAudioVideoSyncJob.result||{};
        const candidate=result.candidate||null;
        appState.audioVideoSyncCandidate=candidate;
        if(candidate?.candidate_id){
          toast(candidate.confidence==='HIGH'?'Shared-audio mapping ready':'Shared-audio mapping needs review',
            'Review the aligned videos at the start, middle and end before publishing the TimeGraph.');
          setTimeout(()=>openAudioVideoSyncReview({candidateId:candidate.candidate_id}),0);
        }else{
          toast('Shared-audio matching returned no review candidate',result.message||'No TimeGraph was changed.',true);
        }
      }else{
        toast('Shared-audio video synchronization did not complete',completedAudioVideoSyncJob.error||completedAudioVideoSyncJob.stage||'Review the failed job in Job Queue.',true);
      }
      if(!(fullRender||!quiet) && ['calibration','synchronize'].includes(appState.page)){
        const panel=$('#properties-content');const scrollTop=panel?.scrollTop||0;
        renderWorkspace();renderProperties();
        requestAnimationFrame(()=>{const next=$('#properties-content');if(next)next.scrollTop=scrollTop;});
      }
    }
    updateNativeTrimJobs();
  } catch (error) {
    if (!quiet) showError(error, 'Could not refresh OculoStudio');
  } finally {
    appState.polling = false;
    if (appState.pendingFullRefresh) {
      appState.pendingFullRefresh = false;
      setTimeout(()=>refreshState({quiet:false, fullRender:true}),0);
    }
  }
}

async function loadRuntime() {
  try { appState.server = await api('/api/runtime'); }
  catch (error) { showError(error, 'OculoStudio backend is unavailable'); }
}

function navigate(page) {
  if (!PIPELINE.some(x => x.id === page)) return;
  appState.page = page;
  render();
}

function scrollDataWorkspace(scope='interaction') {
  appState.dataScrollTarget=scope;
  if(appState.page!=='data')navigate('data');
  const run=()=>{
    const id=scope==='calibration'?'data-calibration-workspace':'data-interaction-workspace';
    const target=document.getElementById(id);
    if(target){target.scrollIntoView({behavior:'smooth',block:'start'});target.classList.add('data-scope-focus');setTimeout(()=>target.classList.remove('data-scope-focus'),1400);}
  };
  requestAnimationFrame(()=>requestAnimationFrame(run));
}

function render() {
  renderTabs();
  renderSessionCard();
  renderSidebar();
  renderWorkspace();
  renderProperties();
  renderMetrics();
  renderJobs();
  bindGlobalActions();
}

function pipelineTabMarkup(item) {
  return `<button type="button" class="pipeline-tab ${item.optional?'optional-tab ':''}${item.utility?'utility-tab ':''}${appState.page === item.id ? 'active' : ''}" data-page="${item.id}">
    <span class="tab-icon">${item.icon}</span><span class="tab-label">${esc(item.label)}</span>${item.optional?'<span class="tab-badge">Optional</span>':''}</button>`;
}

function renderTabs() {
  const rows=PIPELINE.filter(item=>!item.hidden);
  const primary=rows.filter(item=>!item.utility);
  const utility=rows.filter(item=>item.utility);
  $('#pipeline-tabs').innerHTML = `<span class="pipeline-primary-tabs">${primary.map(pipelineTabMarkup).join('')}</span><span class="pipeline-utility-tabs">${utility.map(pipelineTabMarkup).join('')}</span>`;
  $$('.pipeline-tab').forEach(button => button.onclick = () => navigate(button.dataset.page));
  $$('.side-link').forEach(button => button.classList.toggle('active', button.dataset.sidePage === appState.page));
}

function renderSessionCard() {
  const p = project(); const s = session();
  if (!p || !s) {
    $('#session-card').innerHTML = `<div class="session-wrap"><div class="section-label">Session</div><div class="session-card">
      <div class="session-main"><span class="status-dot" style="background:#aaa"></span><strong>No project open</strong></div>
      <div class="session-meta">Create or open a project</div></div></div>`;
    return;
  }
  const sessionOptions = values(p.sessions).map(row => `<option value="${esc(row.id)}" ${row.id === s.id ? 'selected' : ''}>${esc(row.label)}</option>`).join('');
  $('#session-card').innerHTML = `<div class="session-wrap"><div class="section-label">Session</div><div class="session-card">
    <div class="session-main"><span class="status-dot"></span><strong>${esc(s.label)}</strong></div>
    <div class="session-meta" title="${esc(p.root)}">${esc(p.name)} · revision ${p.revision}</div>
    <select id="session-select" class="session-select" aria-label="Selected session">${sessionOptions}</select>
  </div></div>`;
  $('#session-select').onchange = async e => {
    try { await api(`/api/session/${encodeURIComponent(e.target.value)}/select`, {method:'POST'}); await refreshState({quiet:false}); }
    catch (error) { showError(error, 'Could not select session'); }
  };
}

function renderSidebar() {
  const s = session();
  const tree = $('#asset-tree'); const artifactTree = $('#artifact-tree');
  if (!s) {
    tree.innerHTML = `<div class="empty-tree">No assets yet.</div>`;
    artifactTree.innerHTML = `<div class="empty-tree">No artifacts yet.</div>`;
    $('#storage-label').textContent = '—'; $('#storage-progress').style.width = '0%';
    return;
  }
  const groups = ASSET_GROUPS.map(group => {
    const rows = assets(group.kinds);
    if (!rows.length) return '';
    return `<div class="asset-group"><div class="section-label" style="margin:10px 6px 3px;font-size:8px">${esc(group.label)}</div>${rows.map(assetRow).join('')}</div>`;
  }).join('');
  tree.innerHTML = groups || `<div class="empty-tree">Import Native, Timeseries, Face Mapper, video or C3D assets.</div>`;
  $$('.asset-row', tree).forEach(row => row.onclick = () => { appState.selectedAssetId = row.dataset.id; if (appState.page === 'project') navigate('data'); else render(); });

  const art = artifacts();
  artifactTree.innerHTML = art.length ? art.slice().reverse().slice(0, 12).map(a => `<div class="artifact-row" data-id="${esc(a.id)}">
    <span class="asset-kind-icon">${esc(a.kind === 'gaze_rays' ? '↗' : a.kind === 'face_mapper_filter' ? 'F✓' : a.kind === 'privacy_mask_preview' ? 'M✓' : a.kind === 'privacy_video' ? '◉' : a.kind === 'calibration_profile' ? '◎' : a.kind === 'semantic_targets' || a.kind === 'target_fusion' ? '⌾' : a.kind === 'face_aoi' ? 'F' : a.kind === 'body_aoi' ? 'B' : a.kind === 'aoi_review_video' ? '▶' : a.kind === 'aligned_clip_package' ? '▶' : a.kind === 'aligned_facial_video' ? 'FV' : a.kind === 'results_package' ? 'P' : a.kind === 'privacy_clip' ? '✂' : 'A')}</span>
    <span class="asset-copy"><span class="asset-label">${esc(a.label)}</span><span class="asset-sub">${esc(a.kind.replaceAll('_',' '))}</span></span><span class="asset-state"></span></div>`).join('') : `<div class="empty-tree">Outputs appear here with provenance.</div>`;
  $$('.artifact-row', artifactTree).forEach(row => row.onclick = () => { appState.selectedArtifactId = row.dataset.id; navigate('qc'); });

  const bytes = assets().reduce((sum, a) => sum + (Number(a.size_bytes) || 0), 0);
  $('#storage-label').textContent = fmtBytes(bytes);
  $('#storage-progress').style.width = `${Math.min(100, Math.max(4, Math.log10(Math.max(1,bytes))/12*100))}%`;
}

function assetRow(asset) {
  return `<div class="asset-row ${appState.selectedAssetId === asset.id ? 'selected' : ''}" data-id="${esc(asset.id)}" title="${esc(asset.source_path)}">
    <span class="asset-kind-icon">${esc(KIND_ICON[asset.kind] || '?')}</span>
    <span class="asset-copy"><span class="asset-label">${esc(asset.label)}</span><span class="asset-sub">${esc(asset.kind.replaceAll('_',' '))}${assetIdentity(asset).role_id?` · ${esc(assetIdentity(asset).role_id)}`:''}${assetIdentity(asset).participant_id?` · ${esc(assetIdentity(asset).participant_id)}`:''}${asset.recording_id ? ` · ${esc(asset.recording_id.slice(0,8))}` : ''}</span></span>
    <span class="asset-state ${asset.available ? '' : 'missing'}"></span></div>`;
}

function renderWorkspace() {
  const workspace = $('#workspace');
  if (!project()) {
    workspace.innerHTML = welcomeView(); bindProjectActions(); return;
  }
  const renderers = {
    project: projectView, data: dataView, clips: clipsView, privacy: privacyView,
    gaze3d: gaze3dView, targets: targetsView, dyadic: dyadicView, qc: qcView, advanced: advancedView,
    synchronize: synchronizeView, calibration: calibrationView,
  };
  workspace.innerHTML = (renderers[appState.page] || projectView)();
  bindPageActions();
  if (appState.page === 'data') requestAnimationFrame(() => {
    const scroller=$('#workspace .view-scroll');
    if(scroller)scroller.scrollLeft=0;
  });
}

function viewShell(eyebrow, title, content, actions = '') {
  return `<div class="view"><div class="view-toolbar"><div class="view-title"><div class="eyebrow">${esc(eyebrow)}</div><h1>${esc(title)}</h1></div><div class="view-actions">${actions}</div></div><div class="view-scroll">${content}</div></div>`;
}

function welcomeView() {
  return `<div class="welcome-empty"><div class="welcome-box"><div class="welcome-logo">◉</div>
    <h1>Start clean. Process one interaction at a time.</h1>
    <p>The normal workflow is Data → Calibration → Trim review → Privacy → Gaze 3D → Gaze Targets → Dyadic Attention → Results. Calibration resolves the wearer, verifies Pupil↔Vicon timing and derives both roles’ C3D-defined Pupil clips; manual start/end time entry is retained only as a recovery tool.</p>
    <div class="welcome-actions"><button class="primary-button" id="welcome-create">Start clean project</button><button class="soft-button" id="welcome-open">Open existing project</button></div>
  </div></div>`;
}

function projectView() {
  const p=project(),s=session(),ws=projectWorkspace();
  const stageRows=[['Data imported',assets().length>0],['Calibration & clock sync ready',calibrationWorkflowState().allRolesReady],['C3D-defined aligned clips created',calibrationWorkflowState().alignedClipsReady],['Privacy video ready',artifacts(['privacy_clip']).length>0],['Gaze 3D ready',artifacts(['gaze_rays']).length>0],['Target stream ready',artifacts(['target_fusion','semantic_targets']).length>0],['Dyadic output ready',artifacts(['dyadic_attention']).length>0]];
  const complete=stageRows.filter(row=>row[1]).length;const primary=PIPELINE.filter(item=>!item.hidden&&!item.utility&&item.id!=='project');
  const sessionCards=(ws.sessions||[]).map(row=>`<article class="project-session-card ${row.selected?'selected':''}"><header><div><span>${esc(row.session_code)}</span><strong>${esc(row.label)}</strong></div><span class="status-pill ${row.selected?'good':''}">${row.selected?'Current':'Session'}</span></header><div class="project-session-metrics"><div><span>Interactions</span><b>${row.interaction_count}</b></div><div><span>Assets</span><b>${row.asset_count}</b></div><div><span>Results</span><b>${row.artifact_count}</b></div></div><code>${esc(row.folder)}</code><div class="form-actions"><button class="soft-button compact project-session-open" data-session-id="${esc(row.session_id)}">Open</button><button class="soft-button compact project-session-folder" data-path="${esc(row.folder)}">Folder</button></div></article>`).join('');
  return viewShell('Project · Session · Interaction workspace',p.name,`
    <div class="hero-grid clean-start-hero"><section class="hero-copy"><span class="clean-room-note">● One Project · many Sessions · shared hardware Calibration</span><h2>Keep S004–S007 together without mixing their Interaction outputs.</h2><p>Each Session owns its Interaction imports, derived data, reports and annotations. Project-level Calibration Sets register AprilTag recordings and profile revisions once, then bind them to several Sessions without copying raw files; wearer assignment and clock synchronization remain Interaction-specific.</p><div class="hero-buttons"><button class="primary-button" id="project-manage-sessions">Manage Sessions & folders</button><button class="soft-button" data-go="data">Import to ${esc(s?.label||'current Session')}</button><button class="soft-button" id="project-start-fresh">Create another Project</button></div></section><section class="info-card"><h3>Project storage</h3><p class="card-sub">${esc(p.root)}</p><div class="stat-list"><div class="stat-row"><span>Sessions</span><strong>${ws.counts?.sessions??Object.keys(p.sessions||{}).length}</strong></div><div class="stat-row"><span>Interactions</span><strong>${ws.counts?.interactions??0}</strong></div><div class="stat-row"><span>Source assets</span><strong>${ws.counts?.assets??assets().length}</strong></div><div class="stat-row"><span>Derived artifacts</span><strong>${ws.counts?.artifacts??artifacts().length}</strong></div><div class="stat-row"><span>Calibration Sets</span><strong>${ws.counts?.calibration_sets??0}</strong></div><div class="stat-row"><span>Calibration profile records</span><strong>${ws.counts?.calibration_profiles??0}</strong></div></div></section></div>
    <section class="table-card project-session-map"><div class="iw-section-head compact"><div><div class="eyebrow">Project Session map</div><h2>Sessions and human-readable folders</h2></div><div class="wh-head-actions"><button class="soft-button" id="project-open-shared-calibration">Shared Calibration folder</button><button class="primary-button" id="project-manage-sessions-inline">＋ Sessions</button></div></div><div class="project-session-grid">${sessionCards||'<div class="empty-tree">No Sessions exist.</div>'}</div></section>
    <section class="table-card clean-start-checklist"><div class="iw-section-head compact"><div><div class="eyebrow">Current Session · ${esc(s?.label||'')}</div><h2>Workflow status</h2></div><span class="status-pill ${complete===stageRows.length?'good':''}">${complete}/${stageRows.length} stages</span></div><div class="clean-check-grid"><div><strong>Current Session folder</strong><span>${esc(sessionStoragePath('root')||p.root)}</span><span>Browser/drop imports go to this Session only.</span></div><div><strong>Interaction plan</strong><span>${esc((sessionWorkspace()?.planned_interactions||[]).join(', ')||'Add I01, I02… in Session Manager')}</span><span>Imported S/I/R identities populate the planned cards.</span></div><div><strong>Calibration rule</strong><span>Share camera-to-tracker hardware profiles</span><span>Recompute wearer resolution, offset and drift per Interaction</span></div></div></section>
    <div class="card-grid clean-primary-modules">${primary.map(item=>`<button class="module-tile ${moduleReady(item.id)?'ready':''}" data-go="${item.id}"><span class="tile-icon">${item.icon}</span><h3>${esc(item.label)}</h3><p>${esc(item.desc)}</p></button>`).join('')}</div>
  `,`<button class="primary-button" id="project-manage-sessions-toolbar">Manage Sessions</button><button class="soft-button" id="project-open">Open another</button>`);
}

function moduleReady(id) {
  if (id === 'data') return assets().length > 0;
  if (id === 'synchronize') return maps().length > 0;
  if (id === 'clips') return clips().length > 0 || artifacts(['native_reference_trim']).length > 0;
  if (id === 'privacy') return artifacts(['privacy_video','privacy_clip']).length > 0;
  if (id === 'calibration') return calibrationWorkflowState().allRolesReady;
  if (id === 'gaze3d') return artifacts(['gaze_rays']).length > 0;
  if (id === 'targets') return artifacts(['face_aoi','body_aoi','target_fusion','semantic_targets','aoi_review_video']).length > 0;
  if (id === 'qc') return artifacts().length > 0;
  if (id === 'advanced') return artifacts(['calibration_profile']).length > 0 || maps().length > 0;
  return false;
}

function dataView() {
  const allInteractionRows=assets();
  const rows=currentInteractionAssets();
  const hiddenInteractionCount=Math.max(0,allInteractionRows.length-rows.length);
  const unassignedInteractionCount=rows.filter(row=>!assetIdentity(row).interaction_id).length;
  const selected = rows.find(row=>row.id===appState.selectedAssetId) || rows[0] || null;
  const calSets = calibrationSetRows();
  const binding = sessionCalibrationBinding();
  const boundSet = calSets.find(row=>row.id===binding?.calibration_set_id) || null;
  const viewedSet = calibrationSetRecord(appState.calibrationSetId) || boundSet || calSets[0] || null;
  const viewedSetCode = String(viewedSet?.code || viewedSet?.id || '');
  if (viewedSetCode) appState.calibrationSetId = viewedSetCode;
  const calibrationRows = viewedSetCode ? calibrationAssets(null, viewedSetCode) : [];
  const selectedCalibration = calibrationRows.find(row=>row.id===appState.selectedCalibrationAssetId) || calibrationRows[0] || null;
  const identity = selected ? assetIdentity(selected) : null;
  const rowIds=new Set(rows.map(row=>row.id));
  const selectedIds = new Set((appState.selectedAssetIds || []).filter(id=>rowIds.has(id)));
  const c3dMeta = selected?.kind==='c3d' ? (selected.metadata||{}) : null;
  const staleC3d = rows.filter(row=>row.kind==='c3d' && row.metadata?.reader_backend!=='oculostudio_c3d_core_v2');
  const c3dRawSource = String(c3dMeta?.frame_count_source||'');
  const context = activeWorkflowContext();
  const sessionCode = sessionWorkspace()?.session_code || session()?.label || 'Session';
  const interactionCode = context.interactionId || 'No Interaction selected';
  const calibrationContext = viewedSet
    ? `${viewedSet.code} · revision ${viewedSet.active_revision_number??'—'} · ${(viewedSet.applicable_session_ids||[]).length} Sessions`
    : 'Create or assign a Calibration Set first';

  const c3dSummary = c3dMeta ? `<div class="divider"></div><div class="c3d-inspector-summary">
    <div class="section-label">C3D structure</div>
    <div class="stat-list">
      <div class="stat-row"><span>Frames</span><strong>${Number(c3dMeta.frame_count||0).toLocaleString()}</strong></div>
      <div class="stat-row"><span>Rate</span><strong>${c3dMeta.frame_rate?`${Number(c3dMeta.frame_rate).toFixed(3)} Hz`:'—'}</strong></div>
      <div class="stat-row"><span>Duration</span><strong>${c3dMeta.duration_sec!=null?`${Number(c3dMeta.duration_sec).toFixed(3)} s`:'—'}</strong></div>
      <div class="stat-row"><span>Markers</span><strong>${c3dMeta.marker_count??'—'}</strong></div>
      <div class="stat-row"><span>Processor</span><strong class="stat-value-wrap">${esc(c3dMeta.processor_format||'—')}</strong></div>
      <div class="stat-row"><span>Frame-count source</span><strong class="stat-value-wrap" title="${esc(c3dRawSource)}">${esc(compactC3DFrameSource(c3dRawSource))}</strong></div>
    </div>
    ${c3dMeta.long_frame_recovered?`<div class="success-card compact"><strong>Long C3D recovered</strong><p>${esc(c3dMeta.long_frame_notice||`Recovered ${Number(c3dMeta.frame_count||0).toLocaleString()} frames without 16-bit truncation.`)}</p></div>`:''}
    ${c3dMeta.reader_backend!=='oculostudio_c3d_core_v2'?`<div class="warning-card compact"><strong>Metadata uses an older C3D reader</strong><p>Click Re-read C3D metadata to recover long recordings and refresh the duration.</p></div>`:''}
  </div>` : '';

  const interactionTable = rows.length ? `<div class="asset-table-scroll" role="region" aria-label="Interaction assets table" tabindex="0"><table class="asset-table"><colgroup><col class="col-check"><col class="col-asset"><col class="col-identity"><col class="col-type"><col class="col-clock"><col class="col-status"><col class="col-actions"></colgroup><thead><tr><th class="asset-select-col"><input type="checkbox" id="asset-select-all" ${rows.length&&rows.every(a=>selectedIds.has(a.id))?'checked':''}></th><th>Asset</th><th>Identity</th><th>Type</th><th>Clock</th><th>Status</th><th></th></tr></thead><tbody>${rows.map(a=>{const ai=assetIdentity(a);return `<tr data-asset-table-id="${esc(a.id)}">
    <td class="asset-select-col"><input type="checkbox" class="asset-select" data-id="${esc(a.id)}" ${selectedIds.has(a.id)?'checked':''}></td>
    <td><span class="table-label">${esc(a.label)}</span><span class="table-path" title="${esc(a.source_path)}">${esc(a.source_path)}</span></td>
    <td><span class="identity-main">${esc(ai.participant_id||'Unassigned')}</span><span class="identity-sub">${esc([ai.interaction_id,ai.role_id].filter(Boolean).join(' · ')||'No I/R code')}</span></td>
    <td><span class="kind-pill">${esc(a.kind.replaceAll('_',' '))}</span></td>
    <td><span class="table-clock" title="${esc(session()?.time_domains?.[a.primary_time_domain_id]?.label || '—')}">${esc(session()?.time_domains?.[a.primary_time_domain_id]?.label || '—')}</span></td>
    <td><span class="status-pill ${a.available?'good':'bad'}">${a.available?'Available':'Missing'}</span></td>
    <td><div class="row-actions"><button class="row-button asset-inspect" data-id="${esc(a.id)}" title="Inspect">i</button><button class="row-button asset-edit-identity" data-id="${esc(a.id)}" title="Edit identity">ID</button><button class="row-button asset-refresh" data-id="${esc(a.id)}" title="Refresh">↻</button><button class="row-button asset-delete" data-id="${esc(a.id)}" title="Remove">×</button></div></td>
  </tr>`}).join('')}</tbody></table></div>` : `<div class="empty-tree">No Interaction assets are registered for ${esc(sessionCode)} / ${esc(interactionCode)}.</div>`;

  const calibrationTable = calibrationRows.length ? `<div class="asset-table-scroll calibration-source-table-scroll" role="region" aria-label="Calibration Set sources" tabindex="0"><table class="asset-table calibration-source-table"><thead><tr><th>Source</th><th>Purpose</th><th>Participant / device</th><th>Storage</th><th>Status</th><th></th></tr></thead><tbody>${calibrationRows.map(a=>{
    const meta=a.metadata||{};
    const alias=Boolean(meta.calibration_alias_source_asset_id);
    const identityText=a.participant_id||meta.hardware_hint||calibrationImportIdentityText({calibration_role:meta.calibration_role});
    return `<tr class="${selectedCalibration?.id===a.id?'selected':''}"><td><span class="table-label">${esc(a.label)}</span><span class="table-path" title="${esc(a.source_path)}">${esc(a.source_path)}</span></td><td><span class="kind-pill">${esc(calibrationRoleLabel(meta.calibration_role))}</span></td><td>${esc(identityText||'Shared source')}</td><td><span class="status-pill ${alias?'':'good'}">${alias?'Inherited reference':'Canonical source'}</span></td><td><span class="status-pill ${a.available?'good':'bad'}">${a.available?'Available':'Missing'}</span></td><td><button class="row-button calibration-source-inspect" data-id="${esc(a.id)}" title="Inspect">i</button></td></tr>`;
  }).join('')}</tbody></table></div>` : `<div class="empty-tree">No source files are registered in ${esc(viewedSet?.code||'this Calibration Set')}. Drop or append the actor AprilTag recordings and calibration C3Ds above.</div>`;

  const calibrationCards = calSets.length ? calSets.map(row=>{
    const inherited=row.id===binding?.calibration_set_id;
    const viewed=row.id===viewedSet?.id;
    const ready=row.hardware_ready||{};
    const board=row.board_geometry||{};
    const boardReady=Boolean(row.board_geometry_ready||String(board.status||'').toLowerCase()==='ready');
    const sessionNames=(row.sessions||[]).map(item=>item.session_code||item.session_label).join(', ');
    return `<article class="calibration-set-card ${inherited?'active':''} ${viewed?'viewed':''}" data-calibration-set="${esc(row.code)}"><div><span class="eyebrow">${inherited?'Inherited by '+esc(sessionCode):'Project Calibration Set'}</span><strong>${esc(row.label)}</strong><small>${esc(row.code)} · revision ${esc(row.active_revision_number??'—')} · ${esc(row.purpose||'combined')}</small></div><div class="calibration-set-card-stats"><span>${row.source_count||0}<small>sources</small></span><span>${row.profile_count||0}<small>profiles</small></span><span>${(row.applicable_session_ids||[]).length}<small>Sessions</small></span></div><div class="calibration-set-hardware-mini"><span class="${boardReady?'ready':'missing'}">Board ${boardReady?'mapped':'pending'}</span><span class="${ready.ETM_A?'ready':'missing'}">ETM-A ${ready.ETM_A?'ready':'missing'}</span><span class="${ready.ETM_B?'ready':'missing'}">ETM-B ${ready.ETM_B?'ready':'missing'}</span></div><p>${esc(sessionNames||'Not assigned to a Session')}</p><div class="row-actions"><button class="soft-button compact data-select-calibration-set" data-set-id="${esc(row.id)}">${viewed?'Viewing':'View data'}</button><button class="primary-button compact data-import-calibration" data-set-code="${esc(row.code)}">Append files</button><button class="soft-button compact data-manage-calibration-set" data-set-id="${esc(row.id)}">Manage set</button></div></article>`;
  }).join('') : `<div class="empty-tree">No Project-level Calibration Set exists. Create one in this Data workspace before importing shared calibration recordings.</div>`;

  const calibrationInspector = selectedCalibration ? `<div class="stat-list"><div class="stat-row"><span>Source</span><strong class="stat-value-wrap">${esc(selectedCalibration.label)}</strong></div><div class="stat-row"><span>Purpose</span><strong>${esc(calibrationRoleLabel(selectedCalibration.metadata?.calibration_role))}</strong></div><div class="stat-row"><span>Participant / device</span><strong>${esc(selectedCalibration.participant_id||selectedCalibration.metadata?.hardware_hint||'Shared')}</strong></div><div class="stat-row"><span>Calibration Set</span><strong>${esc(selectedCalibration.metadata?.calibration_set_id||viewedSetCode||'—')}</strong></div><div class="stat-row"><span>Storage</span><strong>${selectedCalibration.metadata?.calibration_alias_source_asset_id?'Inherited reference':'Canonical source'}</strong></div><div class="stat-row"><span>Size</span><strong>${fmtBytes(selectedCalibration.size_bytes)}</strong></div></div><details class="metadata-details"><summary>Technical metadata</summary><pre class="inspector-code">${esc(JSON.stringify({metadata:selectedCalibration.metadata,warnings:selectedCalibration.warnings,sha256:selectedCalibration.source_sha256},null,2))}</pre></details>` : `<div class="empty-tree">Select a Calibration source to inspect its role and lineage.</div>`;

  const interactionInspector = selected ? `<div class="stat-list"><div class="stat-row"><span>Kind</span><strong>${esc(selected.kind)}</strong></div><div class="stat-row"><span>Participant ID</span><strong>${esc(identity.participant_id||'Unassigned')}</strong></div><div class="stat-row"><span>Session</span><strong>${esc(identity.session_id||'—')}</strong></div><div class="stat-row"><span>Interaction</span><strong>${esc(identity.interaction_id||'—')}</strong></div><div class="stat-row"><span>Role</span><strong>${esc(identity.role_id||'—')}</strong></div><div class="stat-row"><span>Identity source</span><strong class="stat-value-wrap">${esc(identity.source||'—')}</strong></div><div class="stat-row"><span>Size</span><strong>${fmtBytes(selected.size_bytes)}</strong></div><div class="stat-row"><span>Recording ID</span><strong class="stat-value-wrap">${esc(selected.recording_id||'—')}</strong></div><div class="stat-row"><span>Probe score</span><strong>${Number(selected.probe_score||0).toFixed(2)}</strong></div></div>${c3dSummary}<div class="form-actions identity-actions"><button class="primary-button" id="edit-selected-identity" data-id="${esc(selected.id)}">Edit identity</button><button class="soft-button" id="clear-selected-participant" data-id="${esc(selected.id)}">Clear participant</button></div><details class="metadata-details"><summary>Technical metadata</summary><pre class="inspector-code">${esc(JSON.stringify({participant_id:selected.participant_id,identity,metadata:selected.metadata,warnings:selected.warnings,sha256:selected.source_sha256},null,2))}</pre></details>` : `<div class="empty-tree">Select an Interaction asset to inspect it.</div>`;

  const toolbarC3D = staleC3d.length ? `<button class="soft-button" id="data-refresh-c3d">↻ Re-read C3D metadata (${staleC3d.length})</button>` : rows.some(row=>row.kind==='c3d') ? `<span class="status-pill good data-c3d-current">C3D metadata current</span>` : '';

  return viewShell('Asset registry', 'Data management', `<div class="data-page-root data-management-workspaces">
    <section id="data-calibration-workspace" class="data-scope-workspace calibration-data-workspace">
      <header class="data-scope-header"><div><span class="eyebrow">Project-level shared data</span><h2>Calibration Data</h2><p>Manage actor AprilTag recordings, calibration-board/world C3D, ETM-A / ETM-B calibration C3D, board definitions and completed profiles. Files are registered once and inherited by assigned Sessions.</p></div><div class="data-scope-context"><span>Upload target</span><strong>${esc(viewedSet?.code||'No Calibration Set')}</strong><small>${esc(calibrationContext)}</small></div></header>
      <div class="calibration-set-card-grid">${calibrationCards}</div>
      <div id="calibration-data-drop-zone" class="import-drop-zone data-scope-drop calibration-drop-zone" role="button" tabindex="0" aria-label="Drop Calibration Set files">
        <strong>＋ Drop Calibration Set files here · 拖入共享 Calibration 数据</strong>
        <span>Calibration NRD/TSD, board C3D, ETM-A / ETM-B calibration C3D, world markers, JSON, .oculocal and .g3dcal.</span>
      </div>
      <div class="data-scope-actions"><button class="primary-button" id="data-calibration-choose-files" ${viewedSet?'':'disabled'}>Choose calibration files…</button><button class="soft-button" id="data-calibration-choose-folder" ${viewedSet?'':'disabled'}>Choose calibration folder…</button><button class="soft-button" id="data-manage-calibration-sets">Manage Sets, revisions & Sessions</button><button class="soft-button" id="data-open-calibration-folder" ${viewedSet?'':'disabled'}>Open Set folder</button></div>
      <div class="data-scope-content"><section class="table-card"><div class="iw-section-head compact"><div><span class="eyebrow">Registered Calibration sources</span><h3>${calibrationRows.length} source reference${calibrationRows.length===1?'':'s'}</h3></div><span class="status-pill ${viewedSet?.source_count?'good':''}">${esc(viewedSet?`${viewedSet.source_count||0} canonical sources`:'No Set')}</span></div>${calibrationTable}</section><section class="info-card data-scope-inspector"><h3>Calibration source inspector</h3>${calibrationInspector}</section></div>
    </section>

    <section id="data-interaction-workspace" class="data-scope-workspace interaction-data-workspace">
      <header class="data-scope-header"><div><span class="eyebrow">Session / Interaction-local data</span><h2>Interaction Data</h2><p>Manage NRD, TSD/TSDEV, FaceMapper, body and ETM C3D, facial video, props and other media for one Interaction. These files never enter the shared Calibration Set.</p></div><div class="data-scope-context"><span>Upload target</span><strong>${esc(sessionCode)} / ${esc(interactionCode)}</strong><small>${esc(context.roleId?`Active role view: ${context.roleId}`:'Paired Interaction context')}</small></div></header>
      <div id="interaction-data-drop-zone" class="import-drop-zone data-scope-drop interaction-drop-zone" role="button" tabindex="0" aria-label="Drop current Interaction files">
        <strong>＋ Drop current Interaction files here · 拖入当前 Interaction 数据</strong>
        <span>NRD / TSD / TSDEV / FM / body or ETM C3D / facial video / props. Filename identities are reviewed before registration.</span>
      </div>
      <div class="data-scope-actions"><button class="primary-button" id="data-interaction-choose-files">Choose Interaction files…</button><button class="soft-button" id="data-interaction-choose-folder">Choose Interaction folder…</button><button class="soft-button" id="data-edit-selected" ${selectedIds.size?'':'disabled'}>Edit selected identity (${selectedIds.size})</button>${toolbarC3D}</div>
      <div class="data-scope-content"><section class="table-card data-table-card"><div class="iw-section-head compact"><div><span class="eyebrow">Registered Interaction assets</span><h3>${rows.length} file${rows.length===1?'':'s'} in ${esc(sessionCode)} / ${esc(interactionCode)}</h3><p class="card-sub">${unassignedInteractionCount?`${unassignedInteractionCount} unassigned file${unassignedInteractionCount===1?'':'s'} are shown for routing. `:''}${hiddenInteractionCount?`${hiddenInteractionCount} file${hiddenInteractionCount===1?'':'s'} from other Interactions are hidden.`:'Only the current Interaction and unassigned files are shown.'}</p></div><span class="status-pill">${esc(interactionCode)}</span></div>${interactionTable}</section><section class="info-card data-scope-inspector"><h3>Interaction asset inspector</h3>${interactionInspector}</section></div>
    </section>
  </div>`);
}
function synchronizeView() {
  const ds = domains(); const ms = maps();
  const nativeAssets = assets(['pupil_native']); const cloudAssets = assets(['pupil_cloud']);
  return viewShell('Unified TimeGraph', 'Synchronize recordings and motion capture', `
    <div class="workflow-rail">${workflowRail('synchronize')}</div>
    <div class="timegraph-layout">
      <section class="graph-card"><h3>Clock graph</h3><p class="card-sub">Every edge is an explicit affine mapping. Processing services resolve paths rather than guessing time bases.</p>
        <div class="timegraph-canvas">${ds.map(d => `<div class="time-node ${d.id === session()?.global_time_domain_id ? 'global' : ''}"><b>${esc(d.label)}</b><span>${esc(d.clock_type)} · ${esc(d.id.slice(0,12))}</span></div>`).join('')}</div>
        <div class="map-list">${ms.length ? ms.map(m => `<div class="map-row"><div><strong>${esc(domainLabel(m.source_domain_id))} → ${esc(domainLabel(m.target_domain_id))}</strong><small>scale ${Number(m.scale).toPrecision(10)} · offset ${(Number(m.offset_ns)/1e6).toFixed(3)} ms · RMS ${(Number(m.quality?.rms_residual_ns||0)/1e6).toFixed(3)} ms</small></div><button class="row-button map-delete" data-id="${esc(m.id)}">×</button></div>`).join('') : `<div class="empty-tree">No explicit time maps. Add anchors in the properties panel.</div>`}</div>
      </section>
      <section class="form-card"><h3>Native–Cloud physical frame map</h3><p class="card-sub">This maps actual scene frames; equal visual timecodes are not assumed to represent the same physical frame.</p>
        <div class="field"><label>Native Recording</label><select id="frame-native">${optionRows(nativeAssets,'','Select Native')}</select></div>
        <div class="field" style="margin-top:9px"><label>Timeseries + Scene Video</label><select id="frame-cloud">${optionRows(cloudAssets,'','Select Cloud')}</select></div>
        <div class="field" style="margin-top:9px"><label>Maximum timestamp error (ms)</label><input id="frame-error" type="number" value="5" min="0.01" step="0.1"></div>
        <div class="form-actions"><button class="primary-button" id="run-frame-map" ${!nativeAssets.length||!cloudAssets.length?'disabled':''}>Build frame map</button></div>
        <div class="form-help">The artifact records frame indices, source timestamps, residuals and a validity flag for every Native scene frame.</div>
      </section>
    </div>`);
}

function clipsView() {
  nativeDraft().mode='native';
  return nativeTrimView();
}

function previewStage() {
  const selected = session()?.assets?.[appState.selectedAssetId];
  const previewable = selected && ['video','pupil_native','pupil_cloud'].includes(selected.kind);
  const src = previewable ? `/api/thumbnail/${encodeURIComponent(selectedSessionId())}/${encodeURIComponent(selected.id)}?at_sec=0` : '/static/assets/demo_scene.svg';
  return `<div class="preview-stage"><img src="${src}" onerror="this.src='/static/assets/demo_scene.svg'" alt="Scene-video preview"><div class="preview-overlay"><span class="preview-chip left"><i class="gaze-dot"></i>${previewable ? esc(selected.label) : 'Scene preview · demo overlay'}</span><span class="preview-chip right">Global Timebase</span></div></div>`;
}

function videoArtifacts() {
  return artifacts(['privacy_video','privacy_clip','face_aoi','aoi_review_video']).filter(a => (a.files||[]).some(f => /\.(mp4|mov|mkv|avi)$/i.test(f.path||'')));
}

function artifactVideoUrl(artifactId, variant='') {
  const base=`/api/media/artifact/${encodeURIComponent(selectedSessionId())}/${encodeURIComponent(artifactId)}`;
  return variant?`${base}?variant=${encodeURIComponent(variant)}`:base;
}

function videoChoiceKey(row){return row?._video_key||row?.id||'';}
function videoChoiceArtifactId(row){return row?._artifact_id||row?.id||'';}
function videoChoiceVariant(row){return row?._video_variant||'';}
function videoChoiceLabel(row){return row?._display_label||`${row?.label||'Video'} · ${String(row?.kind||'artifact').replaceAll('_',' ')}`;}

function mediaControlIcon(name){
  const paths={
    previous:'<path d="M8 5v14M19 6l-8 6 8 6V6z"/>',
    next:'<path d="M16 5v14M5 6l8 6-8 6V6z"/>',
    play:'<path d="M8 5.5v13l11-6.5L8 5.5z"/>',
    pause:'<path d="M7 5h4v14H7zM15 5h4v14h-4z"/>',
    volume:'<path d="M4 10v4h4l5 4V6L8 10H4zM16 9a4.5 4.5 0 010 6M18.5 6.5a8 8 0 010 11"/>',
    muted:'<path d="M4 10v4h4l5 4V6L8 10H4zM16.5 9l5 6M21.5 9l-5 6"/>',
    fullscreen:'<path d="M5 9V5h4M15 5h4v4M19 15v4h-4M9 19H5v-4"/>',
  };
  return `<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">${paths[name]||''}</svg>`;
}

function realVideoPlayer(rows, preferredId='', emptyMessage='Generate a video artifact to preview it here.') {
  const selected = rows.find(a=>videoChoiceKey(a)===preferredId||a.id===preferredId) || rows.slice().reverse()[0];
  if(!selected) return `<div class="real-player empty"><div class="player-empty-icon">▶</div><strong>No video artifact yet</strong><span>${esc(emptyMessage)}</span></div>`;
  const selectedKey=videoChoiceKey(selected), selectedArtifact=videoChoiceArtifactId(selected), selectedVariant=videoChoiceVariant(selected);
  const quickRows=rows.filter(a=>['raw_mapper','canonical_mapper','filter_qc'].includes(videoChoiceVariant(a)));
  const quickButtons=quickRows.length?`<div class="facemapper-preview-tabs" aria-label="FaceMapper preview layer">${quickRows.map(a=>`<button type="button" class="facemapper-preview-tab ${videoChoiceKey(a)===selectedKey?'active':''}" data-video-key="${esc(videoChoiceKey(a))}">${videoChoiceVariant(a)==='raw_mapper'?'Original source':videoChoiceVariant(a)==='canonical_mapper'?'Canonical anatomical':'Filtered QC canonical'}</button>`).join('')}</div>`:'';
  return `<div class="real-player">
    ${quickButtons}
    <video id="artifact-video-player" controls preload="metadata" playsinline src="${artifactVideoUrl(selectedArtifact,selectedVariant)}"></video>
    <div class="real-player-toolbar"><label>Preview</label><select id="artifact-video-select">${rows.slice().reverse().map(a=>`<option value="${esc(videoChoiceKey(a))}" data-artifact-id="${esc(videoChoiceArtifactId(a))}" data-variant="${esc(videoChoiceVariant(a))}" data-status="${esc(a._status_text||'Real artifact video · seek and play')}" ${videoChoiceKey(a)===selectedKey?'selected':''}>${esc(videoChoiceLabel(a))}</option>`).join('')}</select><span id="artifact-video-status">${esc(selected._status_text||'Real artifact video · seek and play')}</span></div>
  </div>`;
}

function bindArtifactVideoPlayer() {
  const select=$('#artifact-video-select'), video=$('#artifact-video-player'), status=$('#artifact-video-status');
  if(!select||!video)return;
  const tabs=$$('.facemapper-preview-tab');
  const selectedInfo=()=>{const option=select.selectedOptions?.[0];return {artifactId:option?.dataset?.artifactId||select.value,variant:option?.dataset?.variant||'',status:option?.dataset?.status||'Real artifact video · seek and play'};};
  const syncTabs=()=>tabs.forEach(tab=>tab.classList.toggle('active',tab.dataset.videoKey===select.value));
  select.onchange=()=>{const info=selectedInfo();syncTabs();video.pause();video.src=artifactVideoUrl(info.artifactId,info.variant);video.load();if(status)status.textContent=`Loading · ${info.status}`;};
  tabs.forEach(tab=>tab.addEventListener('click',()=>{if(!tab.dataset.videoKey)return;select.value=tab.dataset.videoKey;select.onchange();}));
  syncTabs();
  video.onloadedmetadata=()=>{const info=selectedInfo();if(status)status.textContent=`${info.status} · ${video.videoWidth}×${video.videoHeight} · ${video.duration.toFixed(3)} s`;};
  video.onerror=()=>{if(status){status.textContent='Could not stream this video. Check that the artifact file still exists.';status.classList.add('media-tools-error');}};
}


function facemapperLiveManifestUrl(artifactId){
  return `/api/facemapper-live/${encodeURIComponent(selectedSessionId())}/${encodeURIComponent(artifactId)}/manifest`;
}
function facemapperLiveVideoUrl(artifactId){
  return `/api/facemapper-live/${encodeURIComponent(selectedSessionId())}/${encodeURIComponent(artifactId)}/video`;
}
function faceMapperLivePlayer(artifact, initialMode='filter', options={}){
  if(!artifact)return `<div class="real-player empty"><div class="player-empty-icon">▶</div><strong>No FaceMapper filter yet</strong><span>Run FaceMapper QC & Filter to inspect source landmarks on the unchanged Native video.</span></div>`;
  const mode=['raw','canonical','filter'].includes(initialMode)?initialMode:'filter';
  const hasPrivacy=Boolean(options.hasPrivacy);
  const hasOtherGenerated=Boolean(options.hasOtherGenerated);
  return `<div class="real-player facemapper-live-player" id="facemapper-live-player" data-artifact-id="${esc(artifact.id)}" data-mode="${esc(mode)}">
    <div class="facemapper-preview-tabs" aria-label="Privacy and FaceMapper preview source">
      <button type="button" class="facemapper-live-tab ${mode==='raw'?'active':''}" data-live-layer="raw">Original source</button>
      <button type="button" class="facemapper-live-tab ${mode==='canonical'?'active':''}" data-live-layer="canonical">Canonical anatomical</button>
      <button type="button" class="facemapper-live-tab ${mode==='filter'?'active':''}" data-live-layer="filter">Filtered QC</button>
      <button type="button" class="facemapper-live-tab privacy-output-tab ${hasPrivacy?'privacy-ready':''}" data-privacy-preview-mode="privacy" ${hasPrivacy?'':'disabled'}>Blurred Privacy${hasPrivacy?'<span class="preview-ready-dot" aria-label="ready"></span>':''}</button>
      ${hasOtherGenerated?'<button type="button" class="facemapper-live-tab" data-privacy-preview-mode="generated">Other outputs</button>':''}
    </div>
    <div class="facemapper-live-stage" id="facemapper-live-stage">
      <video id="facemapper-live-video" preload="metadata" playsinline></video>
      <canvas id="facemapper-live-canvas" aria-label="Live FaceMapper overlay"></canvas>
      <div class="facemapper-live-loading" id="facemapper-live-loading">Loading original Native video and FaceMapper overlay…</div>
      <div class="facemapper-live-badge" id="facemapper-live-badge">Live source · no re-encode</div>
    </div>
    <div class="facemapper-live-controls" role="group" aria-label="FaceMapper preview playback controls">
      <div class="facemapper-transport-controls">
        <button type="button" class="facemapper-control-button" id="facemapper-live-prev" title="Previous retained Native frame" aria-label="Previous retained Native frame">${mediaControlIcon('previous')}</button>
        <button type="button" class="facemapper-control-button primary" id="facemapper-live-play" title="Play" aria-label="Play">${mediaControlIcon('play')}</button>
        <button type="button" class="facemapper-control-button" id="facemapper-live-next" title="Next retained Native frame" aria-label="Next retained Native frame">${mediaControlIcon('next')}</button>
      </div>
      <input class="facemapper-live-seek" id="facemapper-live-seek" type="range" min="0" max="1" value="0" step="0.001" aria-label="Trim-local timeline">
      <span id="facemapper-live-time">00:00.000 / 00:00.000</span>
      <select id="facemapper-live-speed" aria-label="Playback speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select>
      <div class="facemapper-utility-controls">
        <button type="button" class="facemapper-control-button" id="facemapper-live-mute" title="Mute" aria-label="Mute" aria-pressed="false">${mediaControlIcon('volume')}</button>
        <button type="button" class="facemapper-control-button" id="facemapper-live-fullscreen" title="Enter fullscreen" aria-label="Enter fullscreen">${mediaControlIcon('fullscreen')}</button>
      </div>
    </div>
    <div class="facemapper-live-options">
      <label><input type="checkbox" id="facemapper-live-boxes" checked> Face boxes</label>
      <label><input type="checkbox" id="facemapper-live-landmarks" checked> Landmarks</label>
      <label><input type="checkbox" id="facemapper-live-labels" checked> Labels</label>
      <span id="facemapper-live-status">Original Native pixels · overlay drawn in the browser</span>
    </div>
  </div>`;
}
function formatPreviewClock(seconds){
  const value=Math.max(0,Number(seconds)||0);const minutes=Math.floor(value/60);const secs=value-minutes*60;
  return `${String(minutes).padStart(2,'0')}:${secs.toFixed(3).padStart(6,'0')}`;
}
function bindFaceMapperLivePlayer(){
  const root=$('#facemapper-live-player');if(!root)return;
  const artifactId=root.dataset.artifactId, video=$('#facemapper-live-video'), canvas=$('#facemapper-live-canvas');
  const loading=$('#facemapper-live-loading'), badge=$('#facemapper-live-badge'), play=$('#facemapper-live-play');
  const seek=$('#facemapper-live-seek'), time=$('#facemapper-live-time'), status=$('#facemapper-live-status');
  const stage=$('#facemapper-live-stage');
  const state={manifest:null,mode:root.dataset.mode||'filter',playing:false,frameCallback:null,boxes:true,landmarks:true,labels:true,fallbackFullscreen:false};
  const nearestIndex=(values,target)=>{let lo=0,hi=values.length-1;if(hi<0)return -1;while(lo<hi){const mid=(lo+hi)>>1;if(Number(values[mid])<target)lo=mid+1;else hi=mid;}if(lo>0&&Math.abs(Number(values[lo-1])-target)<=Math.abs(Number(values[lo])-target))return lo-1;return lo;};
  const contentRect=()=>{const w=canvas.clientWidth||1,h=canvas.clientHeight||1;const vw=video.videoWidth||state.manifest?.image?.width||1,vh=video.videoHeight||state.manifest?.image?.height||1;const vr=vw/vh,cr=w/h;let x=0,y=0,dw=w,dh=h;if(vr>cr){dh=w/vr;y=(h-dh)/2;}else{dw=h*vr;x=(w-dw)/2;}return{x,y,w:dw,h:dh};};
  const prepareCanvas=()=>{const dpr=Math.min(2,window.devicePixelRatio||1),w=Math.max(1,canvas.clientWidth),h=Math.max(1,canvas.clientHeight);if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);return ctx;};
  const mapPoint=(point,rect)=>{const iw=Number(state.manifest.image.width)||1,ih=Number(state.manifest.image.height)||1;return [rect.x+Number(point[0])*rect.w/iw,rect.y+Number(point[1])*rect.h/ih];};
  const draw=()=>{const ctx=prepareCanvas();if(!state.manifest)return;const timeline=state.manifest.timeline||{},times=timeline.source_times_sec||[];if(!times.length)return;const idx=nearestIndex(times,video.currentTime);if(idx<0)return;const nativeIndex=(timeline.native_frame_indices||[])[idx];const rows=state.manifest.detections_by_native_frame?.[String(nativeIndex)]||[];const rect=contentRect();ctx.save();ctx.beginPath();ctx.rect(rect.x,rect.y,rect.w,rect.h);ctx.clip();const statusColors={accepted:'#55e76d',review:'#ffc12c',rejected:'#ef4c46'};for(const row of rows){const color=state.mode==='filter'?(statusColors[row.status]||'#eaff38'):'#eaff38';const b=row.box||[];if(state.boxes&&b.length===4){const p1=mapPoint([b[0],b[1]],rect),p2=mapPoint([b[2],b[3]],rect);ctx.strokeStyle=color;ctx.lineWidth=row.status==='rejected'?3:2;ctx.strokeRect(p1[0],p1[1],p2[0]-p1[0],p2[1]-p1[1]);if(state.labels){ctx.font='700 12px system-ui';ctx.fillStyle='rgba(16,18,13,.82)';const label=state.mode==='filter'?`${String(row.status||'').toUpperCase()} T${row.track_id}`:`T${row.track_id}`;const tw=ctx.measureText(label).width;ctx.fillRect(p1[0],Math.max(rect.y,p1[1]-19),tw+10,18);ctx.fillStyle=color;ctx.fillText(label,p1[0]+5,Math.max(rect.y+13,p1[1]-6));}}
      const points=state.mode==='raw'?row.source:row.canonical;if(state.landmarks&&points){const styles={left_eye:['#50f5ff',state.mode==='raw'?'LE[src]':'L-eye[subject]'],right_eye:['#ff78e6',state.mode==='raw'?'RE[src]':'R-eye[subject]'],nose:['#f2e75a','N'],mouth_left:['#ffb64b',state.mode==='raw'?'ML[src]':'L-mouth[subject]'],mouth_right:['#55bfff',state.mode==='raw'?'MR[src]':'R-mouth[subject]']};for(const pair of [['left_eye','right_eye'],['mouth_left','mouth_right']]){if(points[pair[0]]&&points[pair[1]]){const a=mapPoint(points[pair[0]],rect),b2=mapPoint(points[pair[1]],rect);ctx.strokeStyle='rgba(230,235,220,.55)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b2[0],b2[1]);ctx.stroke();}}for(const [name,point] of Object.entries(points)){if(!styles[name])continue;const mapped=mapPoint(point,rect),[pointColor,label]=styles[name];ctx.fillStyle=pointColor;ctx.beginPath();ctx.arc(mapped[0],mapped[1],5,0,Math.PI*2);ctx.fill();ctx.strokeStyle='rgba(10,12,9,.9)';ctx.lineWidth=1;ctx.stroke();if(state.labels){ctx.font='700 11px system-ui';ctx.fillStyle=pointColor;ctx.fillText(label,mapped[0]+7,mapped[1]-7);}}}}
    ctx.restore();ctx.font='800 14px system-ui';ctx.fillStyle='rgba(18,21,14,.82)';ctx.fillRect(14,14,440,42);ctx.fillStyle='#eaff38';const title=state.mode==='raw'?'RAW FACEMAPPER · source labels / source pixels':state.mode==='canonical'?'CANONICAL · subject anatomical names / same pixels':'FILTERED QC · live canvas / unchanged video';ctx.fillText(title,28,40);const local=Math.max(0,video.currentTime-Number(state.manifest.video.source_start_sec||0));if(status)status.textContent=`Native frame ${nativeIndex} · ${rows.length} detection${rows.length===1?'':'s'} · local ${formatPreviewClock(local)} · no preview re-encode`;};
  const updateControls=()=>{if(!state.manifest)return;const start=Number(state.manifest.video.source_start_sec||0),duration=Number(state.manifest.video.local_duration_sec||0);let local=Math.max(0,Math.min(duration,video.currentTime-start));seek.max=String(duration||1);seek.value=String(local);time.textContent=`${formatPreviewClock(local)} / ${formatPreviewClock(duration)}`;play.innerHTML=mediaControlIcon(video.paused?'play':'pause');play.title=video.paused?'Play':'Pause';play.setAttribute('aria-label',video.paused?'Play':'Pause');};
  const schedule=()=>{if(typeof video.requestVideoFrameCallback==='function'){state.frameCallback=video.requestVideoFrameCallback(()=>{draw();updateControls();if(!video.paused)schedule();});}};
  const ensureBounds=()=>{if(!state.manifest)return;const start=Number(state.manifest.video.source_start_sec||0),end=Number(state.manifest.video.source_end_sec||start);if(video.currentTime<start-0.02)video.currentTime=start;if(video.currentTime>=end){video.pause();video.currentTime=Math.max(start,end-0.001);}updateControls();draw();};
  const togglePlay=async()=>{if(!state.manifest)return;const start=Number(state.manifest.video.source_start_sec||0),end=Number(state.manifest.video.source_end_sec||start);if(video.currentTime>=end-0.01)video.currentTime=start;try{video.paused?await video.play():video.pause();}catch(error){if(status)status.textContent=`Playback failed: ${error.message||error}`;}};
  const stepFrame=direction=>{if(!state.manifest)return;video.pause();const times=state.manifest.timeline.source_times_sec||[];let idx=nearestIndex(times,video.currentTime);idx=Math.max(0,Math.min(times.length-1,idx+direction));if(idx>=0)video.currentTime=Number(times[idx]);};
  const exitFullscreen=async()=>{if(document.fullscreenElement){try{await document.exitFullscreen();}catch(_){}}root.classList.remove('is-fullscreen');state.fallbackFullscreen=false;};
  const toggleFullscreen=async()=>{if(document.fullscreenElement||state.fallbackFullscreen){await exitFullscreen();return;}try{await root.requestFullscreen();}catch(_){root.classList.add('is-fullscreen');state.fallbackFullscreen=true;}setTimeout(()=>{draw();},50);};
  $$('.facemapper-live-tab',root).forEach(tab=>tab.addEventListener('click',()=>{if(tab.disabled)return;if(tab.dataset.privacyPreviewMode){appState.privacyPreviewMode=tab.dataset.privacyPreviewMode;render();return;}state.mode=tab.dataset.liveLayer||'filter';appState.privacyPreviewMode=`live_${state.mode}`;root.dataset.mode=state.mode;$$('.facemapper-live-tab',root).forEach(item=>item.classList.toggle('active',item===tab));draw();}));
  $('#facemapper-live-boxes').onchange=e=>{state.boxes=e.target.checked;draw();};$('#facemapper-live-landmarks').onchange=e=>{state.landmarks=e.target.checked;draw();};$('#facemapper-live-labels').onchange=e=>{state.labels=e.target.checked;draw();};
  play.onclick=togglePlay;stage.addEventListener('click',event=>{if(event.target===canvas||event.target===video)togglePlay();});$('#facemapper-live-prev').onclick=()=>stepFrame(-1);$('#facemapper-live-next').onclick=()=>stepFrame(1);seek.oninput=()=>{if(!state.manifest)return;video.pause();video.currentTime=Number(state.manifest.video.source_start_sec||0)+Number(seek.value||0);};$('#facemapper-live-speed').onchange=e=>{video.playbackRate=Number(e.target.value)||1;};$('#facemapper-live-mute').onclick=e=>{video.muted=!video.muted;e.currentTarget.innerHTML=mediaControlIcon(video.muted?'muted':'volume');e.currentTarget.title=video.muted?'Unmute':'Mute';e.currentTarget.setAttribute('aria-label',video.muted?'Unmute':'Mute');e.currentTarget.setAttribute('aria-pressed',video.muted?'true':'false');};$('#facemapper-live-fullscreen').onclick=toggleFullscreen;
  video.addEventListener('play',()=>{updateControls();schedule();});video.addEventListener('pause',()=>{updateControls();draw();});video.addEventListener('timeupdate',()=>{if(typeof video.requestVideoFrameCallback!=='function'){ensureBounds();}});video.addEventListener('seeked',()=>{ensureBounds();});video.addEventListener('loadedmetadata',()=>{if(!state.manifest)return;const start=Number(state.manifest.video.source_start_sec||0);video.currentTime=start;video.playbackRate=Number($('#facemapper-live-speed').value)||1;loading.hidden=true;stage.style.aspectRatio=`${state.manifest.image.width}/${state.manifest.image.height}`;ensureBounds();});video.addEventListener('error',()=>{loading.hidden=false;loading.textContent='Could not stream the original Native video.';});
  document.addEventListener('fullscreenchange',()=>{if(!document.fullscreenElement){root.classList.remove('is-fullscreen');state.fallbackFullscreen=false;}draw();},{once:false});window.addEventListener('resize',draw,{passive:true});root.addEventListener('keydown',event=>{if(event.code==='Space'){event.preventDefault();togglePlay();}else if(event.key==='ArrowLeft'){event.preventDefault();stepFrame(-1);}else if(event.key==='ArrowRight'){event.preventDefault();stepFrame(1);}else if(event.key==='Escape'&&state.fallbackFullscreen){exitFullscreen();}});root.tabIndex=0;
  Promise.all([fetch(facemapperLiveManifestUrl(artifactId)).then(response=>{if(!response.ok)return response.json().catch(()=>({detail:response.statusText})).then(data=>{throw new Error(data.detail||response.statusText)});return response.json();})]).then(([manifest])=>{state.manifest=manifest;video.src=facemapperLiveVideoUrl(artifactId);video.load();badge.textContent=`Original Native video · ${manifest.image.width}×${manifest.image.height} · no re-encode`;}).catch(error=>{loading.hidden=false;loading.textContent=`Live FaceMapper preview unavailable: ${error.message||error}`;if(status)status.textContent='Re-run FaceMapper QC/filter if this artifact predates live overlay support.';});
}

function timelineDurationNs() {
  const cs = clips();
  let max = cs.reduce((v,c)=>Math.max(v,Number(c.end_global_ns)||0),0);
  for (const a of assets()) {
    const d = Number(a.metadata?.duration_sec);
    if (Number.isFinite(d)) max = Math.max(max, d*1e9);
  }
  return Math.max(max, 5*60*1e9);
}
function timelineClip(c,total) {
  const left = Math.max(0, Math.min(99, Number(c.start_global_ns)/total*100));
  const width = Math.max(1.5, Math.min(100-left, (Number(c.end_global_ns)-Number(c.start_global_ns))/total*100));
  return `<div class="timeline-clip" style="left:${left}%;width:${width}%" title="${esc(c.label)} · ${fmtNs(c.start_global_ns)}–${fmtNs(c.end_global_ns)}">${esc(c.label)}</div>`;
}

function privacyView() {
  const filters=rowsForActiveWorkflowContext(artifacts(['face_mapper_filter']));
  const maskPreviews=rowsForActiveWorkflowContext(artifacts(['privacy_mask_preview']));
  const full=rowsForActiveWorkflowContext(artifacts(['privacy_video']));
  const clipsOut=rowsForActiveWorkflowContext(artifacts(['privacy_clip']));
  const faceOverlays=rowsForActiveWorkflowContext(artifacts(['face_aoi'])).filter(a=>(a.files||[]).some(f=>String(f.path||'').toLowerCase().endsWith('.mp4')));
  const privacyOutputs=[...full,...clipsOut].slice().sort(byCreated);
  const auxiliaryOutputs=[...maskPreviews,...faceOverlays].slice().sort(byCreated);
  const filter=filters.slice().sort(byCreated).slice(-1)[0]||null;
  const counts=filter?.qc||{};
  let previewMode=String(appState.privacyPreviewMode||'auto');
  const knownModes=new Set(['auto','privacy','generated','live_raw','live_canonical','live_filter']);
  if(!knownModes.has(previewMode))previewMode='auto';
  if(previewMode==='auto')previewMode=privacyOutputs.length?'privacy':filter?'live_filter':auxiliaryOutputs.length?'generated':'live_filter';
  if(previewMode==='privacy'&&!privacyOutputs.length)previewMode=filter?'live_filter':auxiliaryOutputs.length?'generated':'live_filter';
  if(previewMode==='generated'&&!auxiliaryOutputs.length)previewMode=privacyOutputs.length?'privacy':filter?'live_filter':'generated';
  if(previewMode.startsWith('live_')&&!filter)previewMode=privacyOutputs.length?'privacy':'generated';
  appState.privacyPreviewMode=previewMode;
  const liveMode=previewMode.startsWith('live_')?previewMode.replace('live_',''):'filter';
  const latestPrivacy=privacyOutputs.slice(-1)[0]||null;
  const latestAuxiliary=auxiliaryOutputs.slice(-1)[0]||null;
  const privacyPreview=`<div class="generated-privacy-player privacy-final-player"><div class="facemapper-preview-tabs" aria-label="Privacy preview source"><button type="button" class="facemapper-live-tab" data-privacy-preview-mode="live_filter">FaceMapper QC</button><button type="button" class="facemapper-live-tab active">Blurred Privacy</button>${auxiliaryOutputs.length?'<button type="button" class="facemapper-live-tab" data-privacy-preview-mode="generated">Other outputs</button>':''}</div>${realVideoPlayer(privacyOutputs,videoChoiceKey(latestPrivacy),'Create the synchronized Privacy video to preview the final face blur here.')}</div>`;
  const generatedPreview=`<div class="generated-privacy-player"><div class="facemapper-preview-tabs" aria-label="Generated preview source"><button type="button" class="facemapper-live-tab" data-privacy-preview-mode="live_filter">FaceMapper QC</button>${privacyOutputs.length?'<button type="button" class="facemapper-live-tab privacy-ready" data-privacy-preview-mode="privacy">Blurred Privacy<span class="preview-ready-dot" aria-label="ready"></span></button>':''}<button type="button" class="facemapper-live-tab active">Other outputs</button></div>${realVideoPlayer(auxiliaryOutputs,videoChoiceKey(latestAuxiliary),'Generate an accepted-repair mask preview or Face AOI overlay to inspect it here.')}</div>`;
  const preview=previewMode==='privacy'?privacyPreview:previewMode==='generated'?generatedPreview:faceMapperLivePlayer(filter,liveMode,{hasPrivacy:privacyOutputs.length>0,hasOtherGenerated:auxiliaryOutputs.length>0});
  const featuredLabel=previewMode==='privacy'?'Blurred Privacy video':previewMode==='generated'?(latestAuxiliary?.kind==='privacy_mask_preview'?'Accepted + repair mask preview':'Generated analysis output'):'Live FaceMapper inspection';
  const featuredDetail=previewMode==='privacy'?'Final synchronized privacy-safe artifact · native video controls':previewMode==='generated'?'Auxiliary generated artifact playback':'Original Native video + browser Canvas overlay · no frame resampling or preview encode';
  const sourceLandmarkCount=Number(counts.raw_source_landmark_count ?? counts.filtered_qc_landmark_count ?? -1);
  const landmarkWarning=filter&&sourceLandmarkCount===0?`<div class="warning-card" style="margin-top:12px"><strong>No drawable FaceMapper landmarks were found</strong><p>The live preview can only draw face boxes because the source rows contain no finite LE/RE/N/ML/MR coordinates. Inspect source_face_detections_raw.csv before continuing to fine-grained Face AOI.</p></div>`:'';
  const previewInfo=previewMode==='privacy'&&latestPrivacy
    ?`<p class="card-sub">This is the final synchronized, blurred output for the active interaction and role. It is the same privacy-safe video used by downstream AOI review.</p><div class="stat-list"><div class="stat-row"><span>Label</span><strong class="stat-value-wrap">${esc(latestPrivacy.label)}</strong></div><div class="stat-row"><span>Artifact kind</span><strong>${esc(String(latestPrivacy.kind||'privacy_video').replaceAll('_',' '))}</strong></div><div class="stat-row"><span>Preview source</span><strong>Blurred artifact bytes</strong></div><div class="stat-row"><span>FaceMapper filter</span><strong>${filter?'Available':'Not selected'}</strong></div></div>`
    :filter?`<p class="card-sub">Filtering happens after Trim and before Privacy. The live viewer reads the immutable source and canonical tables directly; encoded QC MP4s are no longer required for normal review.</p><div class="stat-list"><div class="stat-row"><span>Accepted</span><strong>${esc(counts.accepted ?? '—')}</strong></div><div class="stat-row"><span>Review · kept for Privacy</span><strong>${esc(counts.review ?? '—')}</strong></div><div class="stat-row"><span>Rejected</span><strong>${esc(counts.rejected ?? '—')}</strong></div><div class="stat-row"><span>Source landmarks available</span><strong>${esc(counts.raw_source_landmark_count ?? counts.filtered_qc_landmark_count ?? '—')}</strong></div><div class="stat-row"><span>Semantic conversion</span><strong>${counts.bilateral_label_swap_applied===true?'Image-side → subject anatomy':counts.bilateral_label_swap_applied===false?'Already anatomical':'—'}</strong></div><div class="stat-row"><span>Geometry mirror</span><strong>Never</strong></div><div class="stat-row"><span>Preview video</span><strong>Original Native bytes</strong></div></div>`
    :(latestAuxiliary?`<div class="stat-list"><div class="stat-row"><span>Label</span><strong>${esc(latestAuxiliary.label)}</strong></div><div class="stat-row"><span>Kind</span><strong>${esc(latestAuxiliary.kind)}</strong></div></div>`:`<div class="empty-tree">No FaceMapper QC or Privacy output yet.</div>`);
  return viewShell('FaceMapper quality control and auditable anonymization', 'FaceMapper filter & Privacy', `
    <div class="workflow-rail">${workflowRail('privacy')}</div>
    <div class="split-view privacy-preview-grid"><section class="preview-card">${preview}<div class="player-bar"><span class="status-dot"></span><span>${esc(featuredLabel)}</span><span class="player-spacer"></span><span class="time-readout">${esc(featuredDetail)}</span></div></section>
      <section><div class="warning-card"><strong>Stable source-video inspection</strong><p>FaceMapper boxes and landmarks are drawn live on the unchanged Native scene video. Use <b>Blurred Privacy</b> to inspect the actual final anonymized artifact; the two players are intentionally separate so source geometry cannot be confused with encoded blur output.</p></div>
      ${landmarkWarning}<div class="info-card" style="margin-top:12px"><h3>${previewMode==='privacy'?'Final Privacy output':filter?'Latest FaceMapper QC':'Latest video artifact'}</h3>${previewInfo}</div></section>
    </div>`);
}

function calibrationView() {
  const profiles = calibrationProfilesForActiveSet().slice().reverse();
  const acceptances = artifacts(['calibration_acceptance']).slice().reverse();
  const activeSet = activeCalibrationSetId();
  const setRows = calibrationAssets(null, activeSet);
  const projectSets=calibrationSetRows();
  const activeSetRecord=calibrationSetRecord(activeSet);
  const setIds = projectSets.length?projectSets.map(row=>row.code):calibrationSetIds();
  const displayMode=calibrationDisplayMode();
  const detailRole=(displayMode==='R01'||displayMode==='R02')?displayMode:'';
  const stage=calibrationWorkflowState(detailRole||'R01');
  const profileRows=profiles.map(p=>{
    const source=p.parameters?.calibration_source||p.qc?.calibration_source||'profile';
    const hardware=p.parameters?.hardware_display_name||p.qc?.hardware_display_name||p.parameters?.hardware_id||p.qc?.hardware_id||'Unspecified hardware';
    const mount=p.parameters?.mount_version||p.qc?.mount_version||'';
    const registration=p.parameters?.hardware_registration_status||p.qc?.hardware_registration_status||'';
    const revision=p.parameters?.hardware_registration_revision||p.qc?.hardware_registration_revision||1;
    const transfer=p.qc?.transfer_confidence;
    const sessions=p.parameters?.valid_sessions||p.qc?.valid_sessions||[];
    const hasG3D=(p.files||[]).some(f=>String(f.path||'').toLowerCase().endsWith('.g3dcal'));
    return `<button type="button" class="map-row calibration-library-row calibration-profile-select" data-calibration-profile="${esc(p.id)}"><div><strong>${esc(p.label)}</strong><small>${esc(hardware)}${mount?' · '+esc(mount):''} · ${esc(source)}${sessions.length?' · '+esc(sessions.join(', ')):''}${revision>1?' · rev'+esc(revision):''}</small><small>${hasG3D?'Portable .g3dcal available':'OculoStudio profile'}${registration?' · '+esc(registration.replaceAll('_',' ')):''}${transfer?' · transfer '+esc(transfer):''}</small></div><span class="status-pill ${p.qc?.transfer_status==='failed'?'warning':'good'}">${esc(transfer||registration||'Ready')}</span></button>`;
  }).join('');
  const acceptanceRows=acceptances.map(a=>`<div class="map-row calibration-library-row"><div><strong>${esc(a.label)}</strong><small>${esc(a.parameters?.interaction_id||'Interaction')} · ${esc(a.parameters?.wearer_role_id||'Role')} · ${esc(a.qc?.allowed_target_granularity||'—')}</small><small>Transfer ${esc(a.qc?.transfer_status||'—')} · Time ${esc(a.qc?.time_status||'—')} · Head ${esc(a.qc?.head_projection_status||'—')} · Target ${esc(a.qc?.known_target_status||'unavailable')}</small></div><span class="status-pill ${a.qc?.production_ready?'good':a.qc?.acceptance_status==='rejected'?'warning':''}">${esc(a.qc?.acceptance_status||'review')}</span></div>`).join('');
  const sourceRows=setRows.length?setRows.map(asset=>{
    const role=asset.metadata?.calibration_role||'calibration_source';
    const hardware=asset.metadata?.hardware_hint||'';
    const participant=asset.participant_id||'';
    return `<article class="calibration-source-card"><span class="asset-kind-icon">${esc(KIND_ICON[asset.kind]||'?')}</span><div><strong>${esc(asset.label)}</strong><small>${esc(calibrationRoleLabel(role))}${hardware?' · '+esc(hardware):''}${participant?' · '+esc(participant):''}</small><small>${esc(asset.kind.replaceAll('_',' '))} · ${asset.available?'Available':'Missing'}</small></div><span class="status-pill ${asset.available?'good':'warning'}">${asset.available?'Ready':'Missing'}</span></article>`;
  }).join(''):`<div class="empty-tree calibration-empty-registry"><strong>No raw calibration-session sources registered.</strong><span>This is allowed when compatible .g3dcal/.oculocal profiles already exist. Import raw calibration files only when a dedicated calibration take was recorded or a new profile must be estimated.</span></div>`;
  const setButtons=setIds.length?setIds.map(id=>{const row=calibrationSetRecord(id);return `<button type="button" class="calibration-set-chip ${id===activeSet?'active':''}" data-calibration-set="${esc(id)}" title="${esc(row?.label||id)}">${esc(row?.code||id)}${row?.active_revision_number?` · r${esc(row.active_revision_number)}`:''}</button>`}).join(''):`<span class="status-pill">No sets</span>`;
  const preview=window.OculoCalibrationPreview?OculoCalibrationPreview.markup({sessionId:selectedSessionId()}):`<section class="ray-stage"><div class="ray-summary"><b>Calibration preview unavailable</b><span>Reload OculoStudio to load the real C3D preview module.</span></div></section>`;
  const workflowReady=stage.allRolesReady&&stage.alignedClipsReady;
  const detailRow=detailRole?stage.rows.find(row=>row.role===detailRole):null;
  const roleOption=row=>{const etm=row?.wearerMapping?.etm_id||row?.mapping?.etm_id||'';return etm?`${row.role} details · ${etm}`:`${row?.role||'Role'} details`;};
  const viewTabs=`<div class="calibration-view-tabs" role="tablist" aria-label="Calibration view"><button type="button" class="${displayMode==='paired'?'active':''}" data-calibration-view="paired"><strong>Paired summary</strong><span>Authoritative R01 + R02 workflow</span></button>${stage.rows.map(row=>`<button type="button" class="${displayMode===row.role?'active':''}" data-calibration-view="${esc(row.role)}"><strong>${esc(roleOption(row))}</strong><span>View role-specific profile and sync evidence</span></button>`).join('')}</div>`;
  const detailSummary=detailRow?`<section class="calibration-detail-summary"><div><span class="eyebrow">Read-only role detail view</span><h3>${esc(detailRow.role)} → ${esc(detailRow.wearerMapping?.etm_id||detailRow.mapping?.etm_id||'Unresolved ETM')}</h3><p>This view changes only the calibration evidence and role-specific profile form shown on screen. It does not change the paired wearer assignment, ETM hardware identity, TimeGraph, or the R01/R02 clip transaction.</p></div><div class="stat-list"><div class="stat-row"><span>Participant</span><strong>${esc(calibrationParticipantForRole(detailRow.role))}</strong></div><div class="stat-row"><span>Hardware profile</span><strong>${esc(detailRow.profile?.label||'Not bound')}</strong></div><div class="stat-row"><span>Clock sync</span><strong>${detailRow.timeSyncReady?'Verified':'Missing'}</strong></div><div class="stat-row"><span>Role status</span><strong>${detailRow.ready?'Ready':'Incomplete'}</strong></div></div></section>`:'';
  const stageStatus=workflowReady?'Calibration + aligned clips complete':stage.allRolesReady?'Calibration handoff ready · create aligned clips':'Calibration incomplete';
  return viewShell('Required calibration, synchronization and automatic interaction trim', displayMode==='paired'?'Calibration readiness':`${detailRole} calibration details`, `
    <div class="workflow-rail">${workflowRail('calibration')}</div>
    ${viewTabs}
    ${detailSummary}
    <section class="required-stage-banner ${workflowReady?'complete':''}"><div><span class="status-pill ${workflowReady?'good':'warning'}">Required before normal Trim / Privacy / Gaze 3D</span><strong>${stageStatus}</strong><p>The normal workflow now has five steps: 1 resolve the interaction wearer; 2 prepare ETM-A/ETM-B hardware profiles; 3 synchronize each Pupil IMU clock to its assigned ETM Vicon clock without ego video; 4 create the immutable Gaze 3D handoff; 5 derive R01/R02 Native-reference clips automatically from the common body-C3D interval.</p></div><button class="primary-button" data-go="clips" ${workflowReady?'':'disabled'}>Continue to Trim review</button></section>
    <section class="table-card calibration-readiness-overview"><div class="iw-section-head compact"><div><div class="eyebrow">Interaction calibration + interval handoff</div><h2>${stage.handoff?esc(stage.handoff.label):stage.wearerAssignment?esc(stage.wearerAssignment.label)+' · profiles not bound':'No resolved wearer assignment'}</h2></div><span class="status-pill ${workflowReady?'good':'warning'}">${workflowReady?'Aligned clips ready':stage.allRolesReady?'Create clips':stage.assignmentReady?'Profile binding required':'Resolve wearer first'}</span></div><div class="calibration-readiness-grid">${calibrationReadinessRows(stage)}</div><div class="form-help">Wearer resolution is immutable. Synchronization estimates offset and drift from Pupil IMU ↔ assigned ETM motion. The verified TimeGraph is then inverted to map the common body-C3D interval into each role's NRD clock; no manual start/end input is required.</div></section>
    <section class="table-card calibration-input-readiness"><div class="iw-section-head compact"><div><div class="eyebrow">Calibration input readiness · managed in Data</div><h2>${esc(activeSetRecord?.label||activeSet)}${activeSetRecord?` <small>· revision ${esc(activeSetRecord.active_revision_number??'—')}</small>`:''}</h2></div><span class="status-pill ${activeSetRecord?.source_count?'good':'warning'}">${activeSetRecord?.source_count||0} sources · ${activeSetRecord?.profile_count||0} profiles</span></div><p class="card-sub">This page performs wearer resolution, physical-profile building, clock synchronization and handoff validation. Source importing, appending, Set revisions and Session assignments are managed only in <b>Data → Calibration Data</b>.</p><div class="calibration-input-contract"><div><span>Calibration Set</span><strong>${esc(activeSetRecord?.code||activeSet||'Not assigned')}</strong></div><div><span>Assigned Sessions</span><strong>${esc((activeSetRecord?.sessions||[]).map(row=>row.session_code||row.session_label).join(', ')||'—')}</strong></div><div><span>Raw sources</span><strong>${esc(activeSetRecord?.source_count??0)}</strong></div><div><span>Physical profiles</span><strong>${esc(activeSetRecord?.profile_count??0)}</strong></div></div><div class="form-actions"><button class="soft-button" id="calibration-open-data-sources">Open Data → Calibration Data</button></div></section>
    <div class="split-view calibration-profile-grid calibration-live-grid">
      ${preview}
      <section class="info-card calibration-library-panel"><h3>Hardware profile library</h3><p class="card-sub">Profiles are required for Gaze 3D. Select a profile below to place its real camera transform into the tracker pose reconstructed from the active calibration set.</p>${profileRows||'<div class="empty-tree">No physical profile is committed yet. Build ETM-A and ETM-B from the shared Calibration Data, or use the advanced recovery tools for a completed legacy profile.</div>'}<h3 style="margin-top:18px">Transferred-profile acceptance</h3>${acceptanceRows||'<div class="empty-tree">Measured/imported profiles do not need transfer acceptance. A transferred profile must pass its interaction-specific gate before production Gaze 3D.</div>'}</section>
    </div>
    <div class="card-grid calibration-phase-grid calibration-phase-grid-simple" style="grid-template-columns:repeat(5,minmax(165px,1fr));margin-top:14px">
      <section class="info-card"><h3>1 · Resolve wearer</h3><p class="card-sub">Use interaction body and ETM motion to determine R01/R02 → ETM-A/ETM-B.</p></section>
      <section class="info-card"><h3>2 · Prepare ETM profiles</h3><p class="card-sub">Estimate or import one camera-to-tracker profile for each physical ETM.</p></section>
      <section class="info-card"><h3>3 · Synchronize clocks</h3><p class="card-sub">Match Pupil IMU motion to the assigned ETM C3D; no ego-video pixels are read.</p></section>
      <section class="info-card"><h3>4 · Create Gaze 3D handoff</h3><p class="card-sub">Bind both profiles and both verified affine TimeGraph mappings.</p></section>
      <section class="info-card"><h3>5 · Create aligned clips</h3><p class="card-sub">Map the common body-C3D interval into each NRD with no manual times.</p></section>
    </div>`, `<button class="primary-button" data-go="clips" ${workflowReady?'':'disabled'}>Continue to Trim review</button>`);
}

function advancedView() {
  const formalMaps=maps().filter(mapping=>!/provisional/i.test(String(mapping.method||'')));
  return viewShell('Recovery and diagnostics', 'Advanced tools', `
    <section class="advanced-intro info-card"><h3>Only for recovery, diagnostics and legacy compatibility</h3><p class="card-sub">Calibration is now a required main-flow stage before Gaze 3D. Advanced contains only timing recovery, diagnostics and legacy compatibility.</p></section>
    <div class="advanced-tool-grid">
      <article class="advanced-tool-card"><span class="advanced-tool-icon">⟳</span><div><h3>Timing recovery / TimeGraph</h3><p>Use only for genuinely raw, independently started, or mismatched recordings. It is bypassed by the shared-start trim workflow.</p><div class="stat-list"><div class="stat-row"><span>Formal mappings</span><strong>${formalMaps.length}</strong></div></div></div><button class="soft-button" data-go="synchronize">Open timing recovery</button></article>
      <article class="advanced-tool-card disabled"><span class="advanced-tool-icon">▦</span><div><h3>Legacy Global-Time clips</h3><p>The legacy clip editor has been removed from the primary interface. Existing clips remain readable, but new work should use Native-reference trim.</p></div><span class="status-pill">Compatibility only</span></article>
      <article class="advanced-tool-card"><span class="advanced-tool-icon">✓</span><div><h3>Project diagnostics</h3><p>Inspect artifacts, QC snapshots, file paths and immutable provenance without changing scientific results.</p></div><button class="soft-button" data-go="qc">Open Results & QC</button></article>
    </div>
    <section class="warning-card advanced-boundary"><strong>Clean-start rule</strong><p>Create a new project in an empty folder instead of deleting old artifacts or overwriting source C3D files. Keep previous projects only for audit.</p></section>
  `, `<button class="primary-button" id="advanced-start-fresh">＋ Start clean project</button>`);
}

function advancedProperties() {
  return `<section class="property-section"><div class="section-label">Normal workflow profile</div><div class="property-stack">
    <div class="stat-row"><span>Alignment</span><strong>Calibration affine TimeGraph</strong></div>
    <div class="stat-row"><span>TimeGraph required</span><strong>Yes · Pupil IMU ↔ assigned ETM</strong></div>
    <div class="form-help">Calibration is a required main-flow stage before Gaze 3D. Advanced contains only timing recovery and project diagnostics.</div>
  </div></section>
  <section class="property-section"><div class="section-label">Recovery tools</div><div class="property-stack"><button class="soft-button full" data-go="synchronize">Timing recovery / TimeGraph</button><button class="soft-button full" data-go="qc">Project diagnostics</button></div></section>
  <section class="property-section"><div class="section-label">Restart safely</div><div class="form-help">Start a clean project and import only corrected sources when a project history has become ambiguous.</div><button class="soft-button full" id="advanced-start-fresh-side">Start clean project</button></section>`;
}

function gaze3dView() {
  const previewSelection=gazePreviewSelection();
  const stage=calibrationWorkflowState();
  const assignment=stage.assignment;
  const mapping=assignment?.qc?.assignments_summary||{};
  const mappingRows=Object.entries(mapping).map(([role,row])=>{const profile=stage.profiles.find(item=>item.id===row.recommended_calibration_artifact_id);return `<div class="map-row"><div><strong>${esc(role)} → ${esc(row.etm_id||'ETM')}</strong><small>${esc(row.tracker_prefix||'tracker')} · ${Number(row.median_head_distance_mm||0).toFixed(1)} mm head distance</small><small>${profile?'Profile: '+esc(profile.label):'No compatible profile bound'}</small></div><span class="status-pill ${profile&&row.confidence==='HIGH'?'good':'warning'}">${profile?esc(row.confidence||'ready'):'blocked'}</span></div>`}).join('');
  const preview=window.OculoGazePreview?OculoGazePreview.markup({
    sessionId:selectedSessionId(),assignmentId:assignment?.id||'',gazeId:previewSelection.single?.id||'',
    peerGazeId:previewSelection.peer?.id||'',previewMode:previewSelection.mode,pairingToleranceMs:appState.gazePreviewToleranceMs||'',
  }):`<section class="ray-stage">${raySvg(false)}<div class="ray-summary"><b>Preview module unavailable</b><span>Reload OculoStudio to load the interactive preview.</span></div></section>`;
  const current=stage.current;
  const gate=current?.ready
    ?`<div class="success-card calibration-handoff-banner"><strong>${esc(current.role)} calibration handoff ready</strong><p>${esc(current.mapping?.etm_id||'ETM')} · ${esc(current.profile?.label||'profile')} · assignment ${esc(assignment?.label||'')}</p></div>`
    :`<div class="warning-card calibration-handoff-banner"><strong>Complete Calibration before generating rays</strong><p>${esc(current?.role||calibrationContextRole())} still needs a resolved ETM wearer and compatible hardware profile${current?.transferred&&!current?.acceptanceReady?' with accepted transfer':''}.</p><button class="primary-button" data-go="calibration">Open Calibration</button></div>`;
  return viewShell('World-space ocular geometry after calibration handoff', 'Gaze 3D rays', `
    <div class="workflow-rail">${workflowRail('gaze3d')}</div>
    ${gate}
    <div class="split-view gaze3d-assignment-grid">${preview}
      <section class="info-card"><h3>Calibration handoff</h3><p class="card-sub">Wearer assignment and physical ETM profile binding are completed in Calibration. Gaze 3D only consumes the immutable handoff.</p>${assignment?`${mappingRows||'<div class="empty-tree">Assignment artifact contains no role summary.</div>'}<div class="stat-list"><div class="stat-row"><span>Status</span><strong>${esc(assignment.qc?.status||'—')}</strong></div><div class="stat-row"><span>Global margin</span><strong>${assignment.qc?.global_assignment_margin_mm==null?'—':Number(assignment.qc.global_assignment_margin_mm).toFixed(1)+' mm'}</strong></div><div class="stat-row"><span>Current role</span><strong>${esc(calibrationContextRole())}</strong></div></div>`:'<div class="empty-tree">No calibration handoff exists. Complete Calibration first.</div>'}</section></div>
    <div class="card-grid" style="grid-template-columns:repeat(3,1fr)"><section class="info-card"><h3>Full transform</h3><p class="card-sub">Pixel unprojection → camera frame → calibrated tracker transform → interpolated Vicon world pose.</p></section><section class="info-card"><h3>Immutable calibration input</h3><p class="card-sub">The role-to-ETM assignment and recommended physical hardware profile are selected upstream in Calibration rather than recomputed here.</p></section><section class="info-card"><h3>Fail closed</h3><p class="card-sub">Missing calibration handoff, ambiguous identity, incompatible profile, missing time paths or long pose gaps block processing.</p></section></div>`);
}

function targetsView() {
  const contextTrims=targetRowsForActiveContext(artifacts(['native_reference_trim']));
  const contextTrimId=selectedTargetTrimId(contextTrims);
  const exactContextRows=rows=>targetRowsForActiveContext(rows,{trimId:contextTrimId,requireTrim:Boolean(contextTrimId)});
  const associationRows=exactContextRows(artifacts(['participant_association']));
  const faceRows=exactContextRows(artifacts(['face_aoi']));
  const bodyRows=exactContextRows(artifacts(['body_aoi']));
  const objectRows=exactContextRows(artifacts(['object_aoi']));
  const fusionRows=exactContextRows(artifacts(['target_fusion']));
  const privacyClips=exactContextRows(artifacts(['privacy_clip']));
  const reviewVideos=exactContextRows(artifacts(['aoi_review_video']));
  const association=associationRows.slice(-1)[0], face=faceRows.slice(-1)[0], body=bodyRows.slice(-1)[0], object=objectRows.slice(-1)[0], fusion=fusionRows.slice(-1)[0];
  const pct=value=>value==null?'—':`${(Number(value)*100).toFixed(1)}%`;
  const hz=value=>value==null||!Number.isFinite(Number(value))?'—':`${Number(value).toFixed(2)} Hz`;
  const hasClipFaceStats=face&&face.qc?.partner_face_sample_fraction!=null;
  const faceClipStats=face?`<section class="table-card face-clip-gaze-summary"><div class="iw-section-head compact"><div><div class="eyebrow">Complete clip · strict partner identity</div><h3>Partner-face gaze proportions</h3><p class="card-sub">Sample-based proportions use all valid gaze samples in the selected trimmed clip. The second group shows how partner-face hits are distributed within the face.</p></div><span class="status-pill ${hasClipFaceStats?'good':'warning'}">${hasClipFaceStats?'Calculated':'Regenerate Face AOI'}</span></div>${hasClipFaceStats?`<div class="card-grid" style="grid-template-columns:repeat(4,minmax(150px,1fr))"><div class="info-card compact"><h4>Partner face</h4><strong class="large-stat">${esc(pct(face.qc?.partner_face_sample_fraction))}</strong><small>${esc(face.qc?.partner_face_sample_count??'—')} / ${esc(face.qc?.valid_gaze_sample_count??'—')} valid gaze samples</small></div><div class="info-card compact"><h4>Eyes</h4><strong class="large-stat">${esc(pct(face.qc?.partner_eye_sample_fraction))}</strong><small>${esc(pct(face.qc?.partner_eye_share_within_partner_face))} of partner-face gaze</small></div><div class="info-card compact"><h4>Nose / bridge + alae</h4><strong class="large-stat">${esc(pct(face.qc?.partner_nose_sample_fraction))}</strong><small>${esc(pct(face.qc?.partner_nose_share_within_partner_face))} of partner-face gaze</small></div><div class="info-card compact"><h4>Mouth / lips</h4><strong class="large-stat">${esc(pct(face.qc?.partner_mouth_sample_fraction))}</strong><small>${esc(pct(face.qc?.partner_mouth_share_within_partner_face))} of partner-face gaze</small></div></div><div class="stat-list" style="margin-top:10px"><div class="stat-row"><span>Other / unspecified partner face within partner-face gaze</span><strong>${esc(pct(face.qc?.partner_other_face_share_within_partner_face))}</strong></div><div class="stat-row"><span>Recovered by green-box edge tolerance</span><strong>${esc(face.qc?.green_face_edge_tolerance_hit_sample_count??0)} samples</strong></div><div class="stat-row"><span>Face-edge policy</span><strong>${esc(face.qc?.face_edge_tolerance_policy||'exact')}</strong></div><div class="stat-row"><span>Clip frames containing at least one partner-face gaze sample</span><strong>${esc(pct(face.qc?.partner_face_frame_fraction))}</strong></div><div class="stat-row"><span>Geometry</span><strong>${face.qc?.nose_region_mode==='nasal_envelope'?'Expanded eyes + nasal bridge/alae envelope + balanced lips':face.qc?.nose_region_mode==='bridge_polygon'?'Legacy eye-line bridge + balanced lips':'Bounded ellipses'}</strong></div></div>`:'<div class="warning-card"><strong>This Face AOI predates full-clip statistics.</strong><p>Generate a new Face AOI with the expanded-eye / nasal bridge-and-alae geometry to calculate and export partner-face, eye, nose and mouth proportions.</p></div>'}</section>`:'';
  const faceTemporalCard=face&&face.qc?.face_geometry_time_policy?`<section class="table-card face-timebase-summary"><div class="iw-section-head compact"><div><div class="eyebrow">Face AOI temporal evidence</div><h3>Observed video frames → gaze-time geometry</h3><p class="card-sub">FaceMapper detections remain native video observations. Interpolation is permitted only along the same short, identity-stable face track; unsafe cases fall back to the nearest observed frame.</p></div><span class="status-pill good">${esc(face.qc?.face_geometry_time_policy||'nearest_frame')}</span></div><div class="card-grid" style="grid-template-columns:repeat(4,minmax(150px,1fr))"><div class="info-card compact"><h4>Gaze sample rate</h4><strong class="large-stat">${esc(hz(face.qc?.estimated_gaze_sample_rate_hz))}</strong><small>classification clock</small></div><div class="info-card compact"><h4>Video observation rate</h4><strong class="large-stat">${esc(hz(face.qc?.estimated_native_video_rate_hz))}</strong><small>FaceMapper evidence clock</small></div><div class="info-card compact"><h4>Safely interpolated</h4><strong class="large-stat">${esc(pct(face.qc?.track_interpolated_sample_fraction))}</strong><small>samples using same-track estimates</small></div><div class="info-card compact"><h4>Nearest-frame fallback</h4><strong class="large-stat">${esc(face.qc?.face_geometry_time_mode_counts?.nearest_frame??0)}</strong><small>samples retaining observed-frame geometry</small></div></div></section>`:'';
  const remainingReviewReasons=Object.entries(face?.qc?.remaining_review_reason_counts||{}).sort((a,b)=>Number(b[1])-Number(a[1])).slice(0,5);
  const remainingReviewReasonRows=remainingReviewReasons.map(([reason,count])=>`<div class="stat-row"><span>${esc(String(reason).replaceAll('_',' '))}</span><strong>${esc(count)}</strong></div>`).join('');
  const autoReviewStats=face&&face.qc?.auto_review_policy?`<section class="table-card face-auto-review-summary"><div class="iw-section-head compact"><div><div class="eyebrow">Automatic review reduction · auditable</div><h3>Adaptive evidence-fusion auto labelling</h3><p class="card-sub">Every gaze centre directly inside an identity-resolved green FaceMapper box is automatically accepted and does not require manual review. Projected participant boxes identify that green box; edge-band and genuinely ambiguous identities remain auditable. Outside-box samples remain no_face.</p></div><span class="status-pill ${Number(face.qc?.auto_resolved_sample_count||0)>0?'good':'warning'}">${esc(face.qc?.auto_review_policy||'off')}</span></div><div class="card-grid" style="grid-template-columns:repeat(4,minmax(150px,1fr))"><div class="info-card compact"><h4>Before auto resolution</h4><strong class="large-stat">${esc(face.qc?.review_samples_before_auto_resolution??'—')}</strong><small>gaze samples originally flagged</small></div><div class="info-card compact"><h4>Auto resolved</h4><strong class="large-stat">${esc(face.qc?.auto_resolved_sample_count??'—')}</strong><small>${esc(pct(face.qc?.auto_review_sample_reduction_fraction))} of flagged samples</small></div><div class="info-card compact"><h4>Still requires review</h4><strong class="large-stat">${esc(face.qc?.review_samples_remaining??'—')}</strong><small>kept for manual decisions</small></div><div class="info-card compact"><h4>Review frames reduced</h4><strong class="large-stat">${esc(pct(face.qc?.auto_review_frame_reduction_fraction))}</strong><small>${esc(face.qc?.review_frames_remaining??'—')} frames remain</small></div></div><div class="card-grid" style="grid-template-columns:minmax(260px,1fr) minmax(260px,1fr);margin-top:10px"><div class="info-card compact"><h4>Direct FaceMapper acceptance</h4><div class="stat-list"><div class="stat-row"><span>Identity-resolved exact green-box hits</span><strong>${esc(face.qc?.direct_green_box_review_eligible_sample_count??0)}</strong></div><div class="stat-row"><span>Previously flagged exact hits auto accepted</span><strong>${esc(face.qc?.direct_green_box_auto_accepted_sample_count??0)}</strong></div><div class="stat-row"><span>Repeated on-face rows with the same value</span><strong>${esc(face.qc?.facemapper_on_face_duplicate_consensus_sample_count??0)}</strong></div><div class="stat-row"><span>True on-face conflicts retained</span><strong>${esc(face.qc?.facemapper_on_face_conflict_sample_count??0)}</strong></div><div class="stat-row"><span>Outside green face boxes → no_face</span><strong>${esc(face.qc?.outside_green_face_box_sample_count??0)}</strong></div></div></div><div class="info-card compact"><h4>Top remaining review reasons</h4>${remainingReviewReasonRows?`<div class="stat-list">${remainingReviewReasonRows}</div>`:'<small>No unresolved review reason remains.</small>'}</div></div></section>`:'';
  const bodyDiagnostics=Array.isArray(body?.qc?.body_diagnostics)?body.qc.body_diagnostics:[];
  const torsoPrimitiveFrames=bodyDiagnostics.reduce((sum,row)=>sum+Number(row?.torso_primitive_frame_count||0),0);
  const bodyGeometryFrames=bodyDiagnostics.reduce((sum,row)=>sum+Number(row?.scene_frame_count||0),0);
  const torsoCoverage=bodyGeometryFrames>0?torsoPrimitiveFrames/bodyGeometryFrames:null;
  const bodyIntegrity=body?bodyGeometryIntegrityForArtifact(body):null;
  const bodyGeometryReady=Boolean(bodyIntegrity?.review_ready);
  const bodyIntegrityPct=value=>value==null?'—':`${(Number(value)*100).toFixed(1)}%`;
  const bodyIntegrityEntityRows=(bodyIntegrity?.entities||[]).filter(row=>row.required!==false).map(row=>{const protocol=row.marker_protocol?.name||'unknown marker profile';const prefix=row.marker_prefix_used?` · ${row.marker_prefix_used}`:'';return `<div class="body-geometry-entity ${esc(row.status||'')}"><div><strong>${esc(String(row.entity_relation||'person').toUpperCase())} · ${esc(row.entity_id||'unknown')}</strong><small>${esc(row.source_file_name||row.source_asset_label||'C3D source')} · ${esc(protocol)}${esc(prefix)}</small></div><div class="body-geometry-metrics"><span>Body <b>${esc(bodyIntegrityPct(row.body_frame_coverage_fraction))}</b></span><span>Torso <b>${esc(bodyIntegrityPct(row.torso_coverage_fraction))}</b></span><span>Pelvis <b>${esc(bodyIntegrityPct(row.pelvis_coverage_fraction))}</b></span><span>Fallback <b>${esc(bodyIntegrityPct(row.torso_fallback_fraction))}</b></span></div></div>`;}).join('');
  const bodyIntegrityIssues=(bodyIntegrity?.failures||[]).slice(0,5).map(row=>`<li>${esc(row.message||String(row.code||'Geometry check failed').replaceAll('_',' '))}</li>`).join('');
  const bodyIntegrityCard=body?`<section class="table-card body-geometry-integrity-card ${bodyGeometryReady?'passed':'blocked'}"><div class="iw-section-head compact"><div><div class="eyebrow">Body geometry acceptance · before ray intersection</div><h3>${bodyGeometryReady?'Torso geometry verified':'Body AOI is not review-ready'}</h3><p class="card-sub">Human review and target fusion are enabled only after every required self/partner C3D stream passes gaze-sample-timeline and torso-coverage checks.</p></div><span class="status-pill ${bodyGeometryReady?'good':'bad'}">${bodyGeometryReady?'PASSED':'BLOCKED'}</span></div>${bodyIntegrityEntityRows||'<div class="empty-tree">No verifiable body geometry diagnostics are available.</div>'}${bodyIntegrityIssues?`<div class="warning-card compact"><strong>Why this artifact is blocked</strong><ul>${bodyIntegrityIssues}</ul><p>Regenerate Body AOI with the current torso integrity gate. Do not manually review this artifact.</p></div>`:''}</section>`:'';
  const bodySourceRates=(body?.qc?.timebase_summary?.body_sources||[]).map(row=>`<div class="stat-row"><span>${esc(String(row.relation||'body').toUpperCase())} · ${esc(row.entity_id||row.label||'entity')}</span><strong>${esc(hz(row.source_frame_rate_hz))}</strong></div>`).join('');
  const bodyClockMode=body?.qc?.body_time_alignment_mode||body?.qc?.timebase_summary?.body_time_alignment_mode||'timegraph';
  const bodyClockLabel=bodyClockMode==='gaze_ray_c3d_time'?'Exact gaze-ray C3D time':bodyClockMode==='pretrimmed_shared_start'?'Shared-start local time':'Project Global TimeGraph';
  const bodyTimebaseCard=body&&body.qc?.body_geometry_clock?`<section class="table-card body-timebase-summary"><div class="iw-section-head compact"><div><div class="eyebrow">Body AOI temporal model</div><h3>One body geometry query per gaze timestamp</h3><p class="card-sub">Self and partner C3D are interpolated at every valid gaze sample. The world video clock is retained only for overlay, frame summaries and human-review navigation.</p></div><span class="status-pill good">${bodyClockMode==='gaze_ray_c3d_time'?'DIRECT C3D CLOCK':'SAMPLE CLOCK'}</span></div><div class="card-grid" style="grid-template-columns:repeat(4,minmax(150px,1fr))"><div class="info-card compact"><h4>Gaze sample rate</h4><strong class="large-stat">${esc(hz(body.qc?.estimated_gaze_sample_rate_hz))}</strong><small>3D intersection clock</small></div><div class="info-card compact"><h4>Video display rate</h4><strong class="large-stat">${esc(hz(body.qc?.estimated_scene_video_rate_hz))}</strong><small>review display only</small></div><div class="info-card compact"><h4>Preview availability</h4><strong class="large-stat">${esc(pct(body.qc?.scene_frame_available_fraction))}</strong><small>does not gate 3D hits</small></div><div class="info-card compact"><h4>Clock routing</h4><strong class="large-stat">${bodyClockMode==='gaze_ray_c3d_time'?'C3D':'Gaze'}</strong><small>${esc(bodyClockLabel)}</small></div></div><div class="stat-list" style="margin-top:10px"><div class="stat-row"><span>Project Global Time path required</span><strong>${body.qc?.global_timegraph_required===false?'No':'Yes'}</strong></div><div class="stat-row"><span>Ray-generation alignment</span><strong>${esc(body.qc?.alignment_mode||'—')}</strong></div></div>${bodySourceRates?`<div class="info-card compact" style="margin-top:10px"><h4>Body source clocks</h4><div class="stat-list">${bodySourceRates}</div></div>`:''}</section>`:'';
  const bodyReasonRows=Object.entries(body?.qc?.remaining_review_reason_counts||{}).sort((a,b)=>Number(b[1])-Number(a[1])).slice(0,4).map(([reason,count])=>`<div class="stat-row"><span>${esc(String(reason).replaceAll('_',' '))}</span><strong>${esc(count)}</strong></div>`).join('');
  const bodyAutoReviewStats=body&&body.qc?.auto_review_policy?`<section class="table-card body-auto-review-summary"><div class="iw-section-head compact"><div><div class="eyebrow">Body AOI · torso geometry + review reduction</div><h3>Fixation-aware, auditable near-body acceptance</h3><p class="card-sub">The automatic Body target is never changed. Clear-margin candidates, exact-hit anchors and homogeneous candidate-only fixations can leave the queue only when target, timing, angular miss and competing-surface checks agree. Self-occlusion, target conflicts and close competing surfaces remain for a person.</p></div><span class="status-pill ${Number(body.qc?.auto_resolved_sample_count||0)>0?'good':'warning'}">${esc(body.qc?.auto_review_policy||'off')}</span></div><div class="card-grid" style="grid-template-columns:repeat(5,minmax(140px,1fr))"><div class="info-card compact"><h4>Before automatic reduction</h4><strong class="large-stat">${esc(body.qc?.review_samples_before_auto_resolution??'—')}</strong><small>near-body samples originally flagged</small></div><div class="info-card compact"><h4>Automatically accepted</h4><strong class="large-stat">${esc(body.qc?.auto_resolved_sample_count??0)}</strong><small>${esc(pct(body.qc?.auto_review_sample_reduction_fraction))} of the original sample queue</small></div><div class="info-card compact"><h4>Review events remaining</h4><strong class="large-stat">${esc(body.qc?.review_units_remaining??body.qc?.review_fixations_remaining??'—')}</strong><small>${esc(pct(body.qc?.auto_review_unit_reduction_fraction))} event reduction</small></div><div class="info-card compact"><h4>Flagged samples remaining</h4><strong class="large-stat">${esc(body.qc?.review_samples_remaining??'—')}</strong><small>${esc(pct(body.qc?.review_required_fraction))} of all gaze samples</small></div><div class="info-card compact"><h4>Torso primitive coverage</h4><strong class="large-stat">${esc(pct(torsoCoverage))}</strong><small>${esc(torsoPrimitiveFrames)} / ${esc(bodyGeometryFrames||'—')} gaze-sample entity queries</small></div></div><div class="card-grid" style="grid-template-columns:minmax(260px,1fr) minmax(260px,1fr);margin-top:10px"><div class="info-card compact"><h4>Automatic evidence sources</h4><div class="stat-list">${Object.entries(body.qc?.auto_review_resolution_source_counts||{}).length?Object.entries(body.qc?.auto_review_resolution_source_counts||{}).map(([source,count])=>`<div class="stat-row"><span>${esc(String(source).replaceAll('_',' '))}</span><strong>${esc(count)}</strong></div>`).join(''):'<small>No review rows were automatically accepted.</small>'}</div></div><div class="info-card compact"><h4>What still needs a person</h4>${bodyReasonRows?`<div class="stat-list">${bodyReasonRows}</div>`:'<small>No unresolved Body AOI review reason remains.</small>'}</div></div></section>`:'';
  const reviewMarkup=window.OculoReview?OculoReview.markup({
    faceArtifacts:faceRows,bodyArtifacts:bodyRows,fusionArtifacts:fusionRows,privacyClips,reviewVideos,
    initialMode:appState.aoiReviewInitialMode,
    initialArtifactId:appState.aoiReviewInitialArtifactId,
    initialVideoArtifactId:appState.aoiReviewInitialVideoArtifactId,
  }):'<div class="warning-card"><strong>AOI Review Player unavailable</strong><p>The review JavaScript did not load.</p></div>';
  return viewShell('Independent evidence, explicit fusion', 'Gaze targets', `
    <div class="workflow-rail">${workflowRail('targets')}</div>
    <div class="success-card compact target-context-lock"><strong>Context-locked evidence</strong><p>${esc(activeWorkflowContext().interactionId||'Interaction')} · ${esc(activeWorkflowContext().roleId||'Role')} · ${contextTrimId?esc(contextTrimId):'no retained trim selected'}. Only artifacts descended from this exact role/trim are shown or opened for review.</p></div>
    <div class="target-pipeline participant-pipeline">
      <article class="target-step ${association?'complete':''}"><span class="target-step-number">0</span><div><strong>Participant association</strong><p>C3D head projection → Face Mapper identity</p></div><b>${association?esc(pct(association.qc?.matched_participant_fraction)):'Optional'}</b></article>
      <span class="target-pipeline-arrow">→</span>
      <article class="target-step ${face?'complete':''}"><span class="target-step-number">1</span><div><strong>Face AOI · 2D</strong><p>Partner-aware TSD + Face Mapper labels</p></div><b>${face?esc(pct(face.qc?.face_aoi_hit_fraction)):'Not run'}</b></article>
      <span class="target-pipeline-arrow">＋</span>
      <article class="target-step ${body?(bodyGeometryReady?'complete':'blocked'):''}"><span class="target-step-number">2</span><div><strong>Body AOI · self + partner</strong><p>Torso-gated first-hit C3D articulated primitives</p></div><b>${body?(bodyGeometryReady?esc(pct(body.qc?.body_hit_fraction)):'Geometry blocked'):'Not run'}</b></article>
      <span class="target-pipeline-arrow">＋</span>
      <article class="target-step ${object?'complete':''}"><span class="target-step-number">3</span><div><strong>Object AOI · 3D</strong><p>Gaze rays + marker-prop C3D</p></div><b>${object?esc(pct(object.qc?.object_hit_fraction)):'Optional'}</b></article>
      <span class="target-pipeline-arrow">→</span>
      <article class="target-step final ${fusion?'complete':''}"><span class="target-step-number">4</span><div><strong>Full target file</strong><p>Face, Body and Object evidence retained</p></div><b>${fusion?esc(pct(fusion.qc?.known_target_fraction)):'Not run'}</b></article>
    </div>
    ${faceTemporalCard}
    ${autoReviewStats}
    ${bodyIntegrityCard}
    ${bodyTimebaseCard}
    ${bodyAutoReviewStats}
    ${reviewMarkup}
    ${faceClipStats}
    <div class="card-grid target-evidence-cards" style="grid-template-columns:repeat(5,minmax(180px,1fr))">
      <div class="info-card"><h3>Participant identity</h3><p class="card-sub">Partner, self and other-person face identities.</p>${association?`<div class="stat-list"><div class="stat-row"><span>Matched</span><strong>${esc(pct(association.qc?.matched_participant_fraction))}</strong></div><div class="stat-row"><span>Other people</span><strong>${esc(association.qc?.other_person_count??'—')}</strong></div></div>`:'<div class="empty-tree">Recommended for partner-only Face AOI.</div>'}</div>
      <div class="info-card"><h3>Face AOI</h3>${face?`<div class="stat-list"><div class="stat-row"><span>Rows</span><strong>${esc(face.qc?.output_rows??'—')}</strong></div><div class="stat-row"><span>Coverage</span><strong>${esc(pct(face.qc?.face_aoi_hit_fraction))}</strong></div><div class="stat-row"><span>Partner face</span><strong>${esc(pct(face.qc?.partner_face_sample_fraction))}</strong></div><div class="stat-row"><span>Human review remaining</span><strong>${esc(face.qc?.review_samples_remaining??'—')}</strong></div></div>`:'<div class="empty-tree">Run step 1.</div>'}</div>
      <div class="info-card"><h3>Body AOI</h3>${body?`<div class="stat-list"><div class="stat-row"><span>Geometry acceptance</span><strong>${bodyGeometryReady?'Passed':'Blocked'}</strong></div><div class="stat-row"><span>Rows</span><strong>${esc(body.qc?.output_rows??'—')}</strong></div><div class="stat-row"><span>All body hits</span><strong>${esc(pct(body.qc?.body_hit_fraction))}</strong></div><div class="stat-row"><span>Self hits</span><strong>${esc(pct(body.qc?.self_hit_fraction))}</strong></div><div class="stat-row"><span>Partner hits</span><strong>${esc(pct(body.qc?.partner_hit_fraction))}</strong></div><div class="stat-row"><span>Auto accepted from review</span><strong>${esc(body.qc?.auto_resolved_sample_count??0)}</strong></div><div class="stat-row"><span>Human review remaining</span><strong>${esc(body.qc?.review_samples_remaining??'—')}</strong></div><div class="stat-row"><span>Torso geometry coverage</span><strong>${esc(pct(torsoCoverage))}</strong></div><div class="stat-row"><span>Self occludes partner</span><strong>${esc(pct(body.qc?.self_occlusion_fraction))}</strong></div></div>`:'<div class="empty-tree">Run step 2 after Gaze 3D.</div>'}</div>
      <div class="info-card"><h3>Object AOI</h3><p class="card-sub">Stable prop instance IDs from marker C3D.</p>${object?`<div class="stat-list"><div class="stat-row"><span>Rows</span><strong>${esc(object.qc?.output_rows??'—')}</strong></div><div class="stat-row"><span>Object hits</span><strong>${esc(pct(object.qc?.object_hit_fraction))}</strong></div></div>`:'<div class="empty-tree">Optional until Object.c3d is available.</div>'}</div>
      <div class="info-card"><h3>Fused target</h3>${fusion?`<div class="stat-list"><div class="stat-row"><span>Known target</span><strong>${esc(pct(fusion.qc?.known_target_fraction))}</strong></div><div class="stat-row"><span>Object target</span><strong>${esc(pct(fusion.qc?.object_target_fraction))}</strong></div></div>`:'<div class="empty-tree">Fuse Face + Body, optionally Object.</div>'}</div>
    </div>
    <div class="densepose-slot"><span class="densepose-badge">Object C3D ready</span><div><strong>Marker props improve joint attention.</strong><p>Each stable object ID can become a shared 3D target. A profile can add physical dimensions beyond the marker cluster.</p></div></div>
  `);
}

function dyadicView() {
  const rows=artifacts(['dyadic_attention']); const last=rows.slice(-1)[0];
  const pct=value=>value==null?'—':`${(Number(value)*100).toFixed(1)}%`;
  return viewShell('Two synchronized wearers, one interaction timeline', 'Dyadic attention', `
    <div class="workflow-rail">${workflowRail('dyadic')}</div>
    <div class="split-view">
      <section class="info-card"><h3>Mutual gaze</h3><p class="card-sub">Broad mutual face gaze and strict mutual eye contact are stored separately.</p>
        <div class="stat-list"><div class="stat-row"><span>Mutual face frames</span><strong>${last?esc(pct(last.qc?.mutual_face_gaze_fraction)):'Not run'}</strong></div><div class="stat-row"><span>Mutual eye contact</span><strong>${last?esc(pct(last.qc?.mutual_eye_contact_fraction)):'Not run'}</strong></div><div class="stat-row"><span>Episodes</span><strong>${last?esc(last.qc?.mutual_gaze_episode_count??'—'):'—'}</strong></div></div>
      </section>
      <section class="info-card"><h3>Joint attention</h3><p class="card-sub">Same object instance or shared third-person target on the common Global Time timeline.</p>
        <div class="stat-list"><div class="stat-row"><span>Joint frames</span><strong>${last?esc(pct(last.qc?.joint_attention_fraction)):'Not run'}</strong></div><div class="stat-row"><span>Episodes</span><strong>${last?esc(last.qc?.joint_attention_episode_count??'—'):'—'}</strong></div><div class="stat-row"><span>Overlap</span><strong>${last?`${Number(last.qc?.overlap_duration_seconds||0).toFixed(1)} s`:'—'}</strong></div></div>
      </section>
    </div>
    <div class="card-grid" style="grid-template-columns:repeat(3,1fr)"><section class="info-card"><h3>Common frame timeline</h3><p class="card-sub">R01 and R02 target streams are converted to Global Time and summarized at the selected FPS.</p></section><section class="info-card"><h3>Gaze following</h3><p class="card-sub">Joint-attention episodes store the first target onset, initiator and follow latency.</p></section><section class="info-card"><h3>Fail closed</h3><p class="card-sub">Missing TimeGraph paths or non-overlapping recordings stop the job instead of guessing alignment.</p></section></div>
  `);
}

function targetFusionSvg() {
  return `<svg viewBox="0 0 1000 500" aria-label="Semantic target fusion illustration">
    <defs><linearGradient id="target-floor" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#34372a"/><stop offset="1" stop-color="#1e2119"/></linearGradient><filter id="target-glow"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
    <rect width="1000" height="500" fill="url(#target-floor)"/>
    <g stroke="#767d60" fill="none" opacity=".38"><path d="M70 430H930M120 380H880M180 330H820M240 280H760M300 230H700"/><path d="M70 430L500 105M230 430L575 105M390 430L650 105M550 430L725 105M710 430L800 105"/></g>
    <g transform="translate(190 122)"><rect x="0" y="0" width="205" height="250" rx="15" fill="#e5e5d9" opacity=".95"/><ellipse cx="103" cy="65" rx="47" ry="60" fill="#bdc0ad"/><ellipse cx="82" cy="56" rx="12" ry="7" fill="#effd3f"/><ellipse cx="124" cy="56" rx="12" ry="7" fill="#7def59"/><ellipse cx="103" cy="78" rx="9" ry="13" fill="#abb09a"/><ellipse cx="103" cy="101" rx="23" ry="8" fill="#c6b06e"/><path d="M103 126v92m0-60l-59 48m59-48l58 48m-58 11l-43 70m43-70l44 70" stroke="#666b50" stroke-width="18" stroke-linecap="round" fill="none"/><text x="24" y="236" fill="#303322" font-size="16" font-family="system-ui">Face semantic AOIs</text></g>
    <g transform="translate(475 105)"><circle cx="100" cy="50" r="42" fill="none" stroke="#effd3f" stroke-width="4"/><path d="M100 92L100 238M100 130L35 196M100 130L170 188M100 238L48 342M100 238L164 342" stroke="#d0d3bf" stroke-width="28" stroke-linecap="round" fill="none"/><path d="M100 92L100 238M100 130L35 196M100 130L170 188M100 238L48 342M100 238L164 342" stroke="#7e8561" stroke-width="2" fill="none"/><text x="19" y="385" fill="#fff" font-size="16" font-family="system-ui">C3D primitive body</text></g>
    <g filter="url(#target-glow)"><path d="M110 250C320 224 525 210 741 175" stroke="#effd3f" stroke-width="5" fill="none"/><circle cx="110" cy="250" r="8" fill="#effd3f"/><circle cx="741" cy="175" r="10" fill="#fff"/></g>
    <g transform="translate(762 122)"><rect width="172" height="220" rx="16" fill="#effd3f" opacity=".95"/><text x="22" y="42" fill="#303322" font-size="13" font-weight="700" font-family="system-ui">FUSED TARGET</text><text x="22" y="91" fill="#303322" font-size="23" font-weight="800" font-family="system-ui">P002</text><text x="22" y="122" fill="#303322" font-size="20" font-family="system-ui">left_eye</text><text x="22" y="163" fill="#303322" font-size="13" font-family="system-ui">confidence 0.93</text><text x="22" y="190" fill="#303322" font-size="13" font-family="system-ui">review false</text></g>
  </svg>`;
}

function raySvg(calibration = false) {
  return `<svg viewBox="0 0 1000 500" aria-label="3D ${calibration?'calibration':'gaze ray'} illustration">
    <defs><linearGradient id="floor" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#34372a"/><stop offset="1" stop-color="#1e2119"/></linearGradient><filter id="rg"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
    <rect width="1000" height="500" fill="url(#floor)"/>
    <g class="ray-grid" stroke="#a7ad87" fill="none"><path d="M80 420L500 120L920 420M80 420H920M140 375H860M210 325H790M280 275H720M350 225H650"/><path d="M80 420L500 120M220 420L570 120M360 420L640 120M500 420L710 120M640 420L780 120M780 420L850 120"/></g>
    ${calibration ? `<g transform="translate(260 180)"><rect x="0" y="0" width="210" height="145" rx="6" fill="#e7e7dc" stroke="#effd3f" stroke-width="3"/><g fill="#25271d">${[0,1,2].map(r=>[0,1,2,3].map(c=>`<rect x="${19+c*46}" y="${16+r*42}" width="31" height="31"/>`).join('')).join('')}</g><g fill="#fff">${[0,1,2].map(r=>[0,1,2,3].map(c=>`<rect x="${25+c*46}" y="${22+r*42}" width="19" height="19"/>`).join('')).join('')}</g></g><g transform="translate(660 225)"><circle cx="0" cy="0" r="36" fill="#d6d7cc"/><circle cx="-28" cy="-22" r="7" fill="#effd3f"/><circle cx="31" cy="-18" r="7" fill="#effd3f"/><circle cx="22" cy="29" r="7" fill="#effd3f"/><circle cx="-25" cy="28" r="7" fill="#effd3f"/></g><path d="M470 250L660 225" stroke="#effd3f" stroke-width="4" stroke-dasharray="10 7" filter="url(#rg)"/>` : `<g transform="translate(285 175)"><circle cx="0" cy="0" r="37" fill="#d9dace"/><path d="M0 37v110m0-70l-62 66m62-66l70 52m-70 18l-55 90m55-90l64 90" stroke="#c3c6b6" stroke-width="13" stroke-linecap="round" fill="none"/><circle cx="-14" cy="-4" r="6" fill="#effd3f"/><circle cx="12" cy="-4" r="6" fill="#7def59"/></g><g filter="url(#rg)" fill="none" stroke-width="4"><path d="M286 171C450 160 635 186 830 143" stroke="#effd3f"/><path d="M286 173C455 220 662 220 860 250" stroke="#7def59"/><path d="M286 174C470 274 675 302 803 360" stroke="#effd3f"/></g><g fill="#fff"><circle cx="830" cy="143" r="7"/><circle cx="860" cy="250" r="7"/><circle cx="803" cy="360" r="7"/></g>`}
  </svg>`;
}

function qcView() {
  const qc=appState.data.qc||{metrics:[],warnings:[]};
  const pct=value=>value==null?'—':`${(Number(value)*100).toFixed(1)}%`;
  const arts=artifacts().slice().reverse();
  const selected=session()?.artifacts?.[appState.selectedArtifactId]||arts[0]||null;
  if(selected && appState.selectedArtifactId!==selected.id)appState.selectedArtifactId=selected.id;
  const root=selected?commonArtifactRoot(selected):'';
  const primary=selected?artifactPrimaryFile(selected):null;
  const files=selected?.files||[];
  const faceVideos=artifacts(['face_aoi']).flatMap(a=>(a.files||[]).filter(f=>/overlay.*\.mp4$/i.test(f.path||'')).map(f=>({artifact:a,file:f})));
  const reviewVideos=artifacts(['aoi_review_video']).flatMap(a=>(a.files||[]).filter(f=>/\.mp4$/i.test(f.path||'')).map(f=>({artifact:a,file:f})));
  const privacyVideos=artifacts(['privacy_clip']).flatMap(a=>(a.files||[]).filter(f=>/\.mp4$/i.test(f.path||'')).map(f=>({artifact:a,file:f})));
  const latestPackage=artifacts(['results_package']).slice(-1)[0]||null;
  const selectedFaceSummary=selected?.kind==='face_aoi'&&selected.qc?.partner_face_sample_fraction!=null?`<div class="result-face-clip-summary"><strong>Full-clip partner-face gaze</strong><div class="stat-list"><div class="stat-row"><span>Partner face / valid gaze</span><strong>${esc(pct(selected.qc.partner_face_sample_fraction))}</strong></div><div class="stat-row"><span>Eyes / valid gaze</span><strong>${esc(pct(selected.qc.partner_eye_sample_fraction))}</strong></div><div class="stat-row"><span>Nose / valid gaze</span><strong>${esc(pct(selected.qc.partner_nose_sample_fraction))}</strong></div><div class="stat-row"><span>Mouth / lips / valid gaze</span><strong>${esc(pct(selected.qc.partner_mouth_sample_fraction))}</strong></div><div class="stat-row"><span>Eyes within partner face</span><strong>${esc(pct(selected.qc.partner_eye_share_within_partner_face))}</strong></div><div class="stat-row"><span>Nose within partner face</span><strong>${esc(pct(selected.qc.partner_nose_share_within_partner_face))}</strong></div><div class="stat-row"><span>Mouth / lips within partner face</span><strong>${esc(pct(selected.qc.partner_mouth_share_within_partner_face))}</strong></div></div></div>`:'';
  const artifactList=resultArtifactBrowserMarkup(arts,selected);
  const fileTable=selected&&files.length?`<table class="result-file-table"><thead><tr><th>File</th><th>Role</th><th>Size</th><th>Actions</th></tr></thead><tbody>${files.map((f,index)=>`<tr><td><span class="table-label">${esc(String(f.path).split(/[\/]/).pop())}</span><span class="table-path" title="${esc(f.path)}">${esc(f.path)}</span></td><td>${esc(f.role||'support')}</td><td>${fmtBytes(f.size_bytes)}</td><td><div class="row-actions"><button class="row-button result-open-file" data-path="${esc(f.path)}" title="Open with default application">Open</button><button class="row-button result-reveal-file" data-path="${esc(f.path)}" title="Show in Explorer">Show</button><a class="row-button result-download-file" href="/api/artifact-file/${encodeURIComponent(selected.session_id)}/${encodeURIComponent(selected.id)}/${index}" target="_blank" title="Open/download through browser">↗</a></div></td></tr>`).join('')}</tbody></table>`:'<div class="empty-tree">This artifact has no registered files.</div>';
  const videoRows=[...privacyVideos.map(x=>({...x,type:'Privacy clip'})),...faceVideos.map(x=>({...x,type:'Face AOI overlay'})),...reviewVideos.map(x=>({...x,type:'AOI review video'}))];
  const videoHtml=videoRows.length?videoRows.slice().reverse().slice(0,8).map(x=>`<div class="result-video-row"><div><strong>${esc(x.type)}</strong><span>${esc(x.artifact.label)}</span><small title="${esc(x.file.path)}">${esc(x.file.path)}</small></div><button class="soft-button result-open-file" data-path="${esc(x.file.path)}">Play</button><button class="soft-button result-reveal-file" data-path="${esc(x.file.path)}">Show</button></div>`).join(''):'<div class="empty-tree">No generated AOI or Privacy videos yet.</div>';
  return viewShell('Processed data, quality gates and lineage','Results & QC',`
    <div class="workflow-rail">${workflowRail('qc')}</div>
    <div class="results-path-grid">
      <section class="result-path-card"><span>Derived artifacts</span><strong>${esc(sessionStoragePath('derived')||joinLocalPath(project().root,`derived/${selectedSessionId()}`))}</strong><button class="soft-button result-open-folder" data-path="${esc(sessionStoragePath('derived')||joinLocalPath(project().root,`derived/${selectedSessionId()}`))}">Open derived folder</button></section>
      <section class="result-path-card"><span>Human-readable exports</span><strong>${esc(project().root)}/exports</strong><button class="soft-button result-open-folder" data-path="${esc(project().root)}/exports">Open exports folder</button></section>
      <section class="result-path-card"><span>QC reports</span><strong>${esc(sessionStoragePath('reports')||joinLocalPath(project().root,`reports/${selectedSessionId()}`))}</strong><button class="soft-button result-open-folder" data-path="${esc(sessionStoragePath('reports')||joinLocalPath(project().root,`reports/${selectedSessionId()}`))}">Open reports folder</button></section>
    </div>
    <div class="results-layout">
      <section class="table-card results-browser"><div class="result-browser-head"><div><h3>Processed artifacts</h3><p class="card-sub">Select an artifact to see its exact folder and every registered output file.</p></div><button class="primary-button" id="run-qc-report">Export QC report</button></div><div class="result-artifact-list">${artifactList}</div></section>
      <section class="info-card results-detail"><h3>${selected?esc(selected.label):'Artifact detail'}</h3>${selected?`<div class="result-detail-meta"><span class="kind-pill">${esc(selected.kind.replaceAll('_',' '))}</span><code>${esc(selected.id)}</code></div><div class="field"><label>Artifact folder</label><textarea readonly>${esc(root)}</textarea></div><div class="form-actions"><button class="primary-button result-open-folder" data-path="${esc(root)}">Open folder</button><button class="soft-button result-reveal-file" data-path="${esc(primary?.path||root)}">Reveal primary</button>${primary?`<button class="soft-button result-open-file" data-path="${esc(primary.path)}">Open primary file</button>`:''}<button class="danger-button" id="cleanup-select-current" data-artifact-id="${esc(selected.id)}">Select for cleanup</button></div>${selectedFaceSummary}<div class="result-qc-summary"><strong>QC snapshot</strong><pre class="inspector-code">${esc(JSON.stringify(selected.qc||{},null,2))}</pre></div>`:'<div class="empty-tree">Select a processed artifact.</div>'}</section>
    </div>
    <section class="table-card artifact-cleanup-card" id="artifact-cleanup-card"><div class="result-browser-head"><div><h3>Artifact cleanup & Project Trash</h3><p class="card-sub">Remove obsolete processed artifacts, duplicate revisions, broken records and unreferenced project folders without touching raw recordings.</p></div><span class="status-pill warn">Dependency-aware</span></div><div id="artifact-cleanup-root"><div class="cleanup-loading"><span class="mini-spinner"></span><strong>Loading cleanup analysis…</strong></div></div></section>
    <section class="table-card"><h3>Files in selected artifact</h3><p class="card-sub">Open files in their default application, reveal them in Explorer, or use the browser link.</p>${fileTable}</section>
    <div class="hero-grid" style="grid-template-columns:1.1fr .9fr"><section class="table-card"><h3>AOI and Privacy videos</h3><p class="card-sub">Face AOI overlays are stored under <code>derived/&lt;session&gt;/face_aoi</code>. Pre-rendered review videos are stored under <code>derived/&lt;session&gt;/aoi_review_video</code>.</p>${videoHtml}</section><section class="info-card"><h3>Review gates</h3><p class="card-sub">Successful processing is not silently upgraded into scientific validity or privacy assurance.</p>${qc.warnings?.length?`<div class="stat-list">${qc.warnings.map(w=>`<div class="warning-card"><p>${esc(w)}</p></div>`).join('')}</div>`:`<div class="status-pill good">No project-level warnings</div>`}${latestPackage?`<div class="divider"></div><h4>Latest organized package</h4><p class="card-sub">${esc(latestPackage.label)}</p><button class="soft-button result-open-folder" data-path="${esc(commonArtifactRoot(latestPackage))}">Open package</button>`:''}</section></div>
  `);
}

function workflowRail(current) {
  const visible = ['data','calibration','clips','privacy','gaze3d','targets','dyadic','qc'];
  const currentIndex = visible.indexOf(current);
  return visible.map((id,index) => {
    const item = PIPELINE.find(x=>x.id===id); const cls = index<currentIndex?'done':index===currentIndex?'current':'';
    return `${index?'<span class="workflow-arrow">→</span>':''}<span class="workflow-step ${cls}"><i>${index<currentIndex?'✓':item.icon}</i>${esc(item.label)}</span>`;
  }).join('');
}

function domainLabel(id) { return session()?.time_domains?.[id]?.label || id; }

function artifactById(id) { return id ? session()?.artifacts?.[id] || null : null; }
function assetById(id) { return id ? session()?.assets?.[id] || null : null; }

function rayTrackerAssetId(assignmentId, role, manualTrackerId='') {
  if (assignmentId) {
    const summary=artifactById(assignmentId)?.qc?.assignments_summary?.[role];
    if (summary?.etm_asset_id) return summary.etm_asset_id;
  }
  return manualTrackerId || '';
}

function timeGraphPathStatus(sourceDomainId, targetDomainId) {
  if (!sourceDomainId || !targetDomainId) return {connected:false, formal:false, path:[]};
  if (sourceDomainId === targetDomainId) return {connected:true, formal:true, path:[]};
  const edges=[];
  maps().forEach(mapping=>{
    const formal=Number(mapping?.quality?.anchor_count||0)>0 && !/provisional/i.test(String(mapping?.method||''));
    edges.push({from:mapping.source_domain_id,to:mapping.target_domain_id,formal,mapping});
    edges.push({from:mapping.target_domain_id,to:mapping.source_domain_id,formal,mapping});
  });
  const search=(formalOnly)=>{
    const queue=[{id:sourceDomainId,path:[]}], visited=new Set([sourceDomainId]);
    while(queue.length){
      const current=queue.shift();
      for(const edge of edges){
        if(edge.from!==current.id || (formalOnly && !edge.formal) || visited.has(edge.to))continue;
        const path=[...current.path,edge];
        if(edge.to===targetDomainId)return path;
        visited.add(edge.to);queue.push({id:edge.to,path});
      }
    }
    return null;
  };
  const formalPath=search(true);
  if(formalPath)return {connected:true,formal:true,path:formalPath};
  const anyPath=search(false);
  return {connected:Boolean(anyPath),formal:false,path:anyPath||[]};
}

function gazeSyncContext(cloudId, assignmentId, role, manualTrackerId='') {
  const source=assetById(cloudId);
  const trackerId=rayTrackerAssetId(assignmentId,role,manualTrackerId);
  const target=assetById(trackerId);
  if(!source||!target)return null;
  return {
    session_id:selectedSessionId(),
    source_asset_id:source.id,
    source_asset_label:source.label,
    source_domain_id:source.primary_time_domain_id,
    source_domain_label:domainLabel(source.primary_time_domain_id),
    target_asset_id:target.id,
    target_asset_label:target.label,
    target_domain_id:target.primary_time_domain_id,
    target_domain_label:domainLabel(target.primary_time_domain_id),
    tracker_asset_id:target.id,
    assignment_artifact_id:assignmentId||null,
    wearer_role_id:assignmentId?role:null,
    return_page:'gaze3d',
  };
}

function nsTextToSeconds(value) {
  try {
    let ns=BigInt(String(value??'0').trim()||'0');
    const sign=ns<0n?'-':'';if(ns<0n)ns=-ns;
    const whole=ns/1000000000n, frac=String(ns%1000000000n).padStart(9,'0').replace(/0+$/,'');
    return sign+String(whole)+(frac?'.'+frac:'');
  } catch (_) { return '0'; }
}

function secondsTextToNs(value) {
  const text=String(value??'').trim();
  const match=text.match(/^([+-]?)(\d+)(?:\.(\d{0,9}))?$/);
  if(!match)throw new Error('Vicon/C3D time must be seconds with at most 9 decimal places');
  const sign=match[1]==='-'?-1n:1n;
  const whole=BigInt(match[2]);
  const fraction=BigInt((match[3]||'').padEnd(9,'0')||'0');
  return String(sign*(whole*1000000000n+fraction));
}

function openGazeSyncRequired(context, detail='') {
  if(!context){showError(new Error(detail||'The selected Timeseries and tracker clocks are not connected.'),'Synchronization required');return;}
  const status=timeGraphPathStatus(context.source_domain_id,context.target_domain_id);
  const reason=status.connected&&!status.formal
    ? 'A provisional zero-offset path exists, but it contains no verified shared-event anchor.'
    : 'No TimeGraph path currently connects these clocks.';
  modal('Synchronize Pupil and Vicon clocks',`
    <div class="warning-card"><strong>Gaze 3D is blocked before processing</strong><p>${esc(reason)}</p></div>
    <div class="stat-list">
      <div class="stat-row"><span>Timeseries</span><strong>${esc(context.source_asset_label)}</strong></div>
      <div class="stat-row"><span>Pupil clock</span><strong>${esc(context.source_domain_label)}</strong></div>
      <div class="stat-row"><span>Tracker</span><strong>${esc(context.target_asset_label)}</strong></div>
      <div class="stat-row"><span>Vicon clock</span><strong>${esc(context.target_domain_label)}</strong></div>
    </div>
    <p>Use the same physical event in both recordings. Enter the Pupil timestamp in nanoseconds and the corresponding Vicon/C3D time in seconds. One anchor fits offset only; two or more anchors are recommended to estimate drift.</p>
    ${detail?`<div class="form-help">${esc(detail)}</div>`:''}
    <div class="form-actions"><button class="soft-button" id="gaze-sync-cancel">Cancel</button><button class="primary-button" id="gaze-sync-open">Open Synchronize</button></div>`,(root,close)=>{
      $('#gaze-sync-cancel',root).onclick=close;
      $('#gaze-sync-open',root).onclick=()=>{appState.syncPrefill={...context};appState.timeAnchors=[{source_ns:'0',target_ns:'0',label:'Anchor 1',weight:1}];close();navigate('synchronize');};
    });
}



function autoSyncMetric(value,digits=3,suffix='') {
  const n=Number(value);
  if(!Number.isFinite(n))return '—';
  const threshold=0.5*Math.pow(10,-digits);
  if(n!==0&&Math.abs(n)<threshold)return `${n<0?'−':'<'}${Math.pow(10,-digits).toFixed(digits)}${suffix}`;
  return `${n.toFixed(digits)}${suffix}`;
}

function autoSyncDriftText(metrics) {
  const value=Number(metrics?.estimated_drift_ppm);
  const ci=Number(metrics?.drift_ci95_ppm);
  if(!Number.isFinite(value))return '—';
  if(Number.isFinite(ci))return `${value.toFixed(2)} ± ${ci.toFixed(2)} ppm (95%)`;
  return `${value.toFixed(2)} ppm · legacy precision`;
}

function autoSyncResidualText(metrics) {
  const p95=Number(metrics?.cross_validated_p95_residual_ms ?? metrics?.p95_residual_ms);
  const resolution=Number(metrics?.residual_resolution_ms);
  if(!Number.isFinite(p95))return '—';
  if(Number.isFinite(resolution)&&resolution>0&&p95<resolution){
    return `< ${resolution.toFixed(2)} ms estimator resolution`;
  }
  return `${p95.toFixed(2)} ms`;
}

function autoSyncPrecisionBadge(metrics) {
  const audited=String(metrics?.precision_audit_version||'').includes('subsample-lag');
  if(!audited)return '<span class="status-pill warning">Legacy precision</span>';
  if(metrics?.residual_below_resolution)return '<span class="status-pill good">Precision-limited · not exact zero</span>';
  return '<span class="status-pill good">Independent precision audit</span>';
}

function autoSyncSummary(candidate) {
  const m=candidate?.metrics||{};
  const cls=candidate?.confidence==='HIGH'?'good':candidate?.confidence==='REVIEW'?'warn':'bad';
  const coStart=m.co_start_prior_detected===true;
  return `<div class="auto-sync-summary">
    <div class="auto-sync-summary-head"><div class="status-pill ${cls}">${esc(candidate?.confidence||'UNKNOWN')} confidence</div>${autoSyncPrecisionBadge(m)}</div>
    <div class="stat-list">
      <div class="stat-row"><span>Relative start alignment</span><strong>${autoSyncMetric(m.estimated_offset_sec,6,' s')}</strong></div>
      <div class="stat-row"><span>Estimated drift</span><strong>${esc(autoSyncDriftText(m))}</strong></div>
      <div class="stat-row"><span>Aligned correlation</span><strong>${autoSyncMetric(m.aligned_correlation,4)}</strong></div>
      <div class="stat-row"><span>Unique-peak margin</span><strong>${autoSyncMetric(m.global_peak_margin,4)}</strong></div>
      <div class="stat-row"><span>Matched / holdout windows</span><strong>${esc(m.matched_window_count??0)} / ${esc(m.holdout_window_count??0)} (${esc(m.window_count??0)} total)</strong></div>
      <div class="stat-row"><span>Cross-validated P95</span><strong>${esc(autoSyncResidualText(m))}</strong></div>
      <div class="stat-row"><span>Local offset range</span><strong>${autoSyncMetric(m.window_local_offset_range_ms,3,' ms')}</strong></div>
      <div class="stat-row"><span>Window time span</span><strong>${autoSyncMetric(m.window_time_span_sec,2,' s')}</strong></div>
      <div class="stat-row"><span>Usable overlap</span><strong>${autoSyncMetric(m.usable_overlap_sec,1,' s')}</strong></div>
    </div>
    ${coStart?'<div class="form-help auto-sync-co-start"><strong>Co-start prior detected.</strong> Manual trimming can make the relative start offset nearly zero; drift is still estimated from independent windows across the recording.</div>':''}
  </div>`;
}

function autoSyncPreviewUrl({candidateId='',artifactId=''}) {
  const query=new URLSearchParams({maximum_samples:'2200'});
  if(candidateId)query.set('candidate_id',candidateId);
  if(artifactId)query.set('artifact_id',artifactId);
  return `/api/time-maps/auto-preview/${encodeURIComponent(selectedSessionId())}?${query.toString()}`;
}

function drawAutoSyncTrace(canvas,data) {
  if(!canvas||!data?.signals?.length)return;
  const rows=data.signals.filter(row=>Number.isFinite(Number(row.pupil_rel_sec)));
  if(rows.length<2)return;
  const dpr=Math.max(1,Math.min(2,window.devicePixelRatio||1));
  const cssWidth=Math.max(760,canvas.clientWidth||980),cssHeight=300;
  canvas.width=Math.round(cssWidth*dpr);canvas.height=Math.round(cssHeight*dpr);
  const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
  const pad={l:52,r:18,t:24,b:38},w=cssWidth-pad.l-pad.r,h=cssHeight-pad.t-pad.b;
  const times=rows.map(row=>Number(row.pupil_rel_sec));
  const t0=Math.min(...times),t1=Math.max(...times),span=Math.max(t1-t0,1e-9);
  const values=[];for(const row of rows){for(const key of ['pupil_signature','vicon_signature']){const v=Number(row[key]);if(Number.isFinite(v))values.push(Math.abs(v));}}
  values.sort((a,b)=>a-b);const scale=Math.max(1,values[Math.floor(values.length*.98)]||1);
  const x=t=>pad.l+(t-t0)/span*w;
  const y=v=>pad.t+h/2-(Math.max(-scale,Math.min(scale,v))/scale)*(h*.42);
  ctx.clearRect(0,0,cssWidth,cssHeight);ctx.fillStyle='#111410';ctx.fillRect(0,0,cssWidth,cssHeight);
  ctx.strokeStyle='rgba(255,255,255,.10)';ctx.lineWidth=1;
  for(let i=0;i<=4;i++){const yy=pad.t+i*h/4;ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(pad.l+w,yy);ctx.stroke();}
  const windows=(data.windows||[]).filter(row=>Number.isFinite(Number(row.pupil_rel_sec)));
  for(const row of windows){const xx=x(Number(row.pupil_rel_sec));ctx.strokeStyle=String(row.accepted).toLowerCase()==='true'?'rgba(209,255,64,.28)':'rgba(255,126,99,.35)';ctx.beginPath();ctx.moveTo(xx,pad.t);ctx.lineTo(xx,pad.t+h);ctx.stroke();}
  const line=(key,color)=>{ctx.strokeStyle=color;ctx.lineWidth=1.7;ctx.beginPath();let drawing=false;for(const row of rows){const v=Number(row[key]),xx=x(Number(row.pupil_rel_sec));if(!Number.isFinite(v)){drawing=false;continue;}const yy=y(v);if(!drawing){ctx.moveTo(xx,yy);drawing=true;}else ctx.lineTo(xx,yy);}ctx.stroke();};
  line('pupil_signature','#d8ff3f');line('vicon_signature','#73d7ce');
  ctx.fillStyle='#d8ff3f';ctx.font='12px system-ui';ctx.fillText('Pupil IMU',pad.l,pad.t-7);ctx.fillStyle='#73d7ce';ctx.fillText('Aligned ETM C3D',pad.l+92,pad.t-7);
  ctx.fillStyle='rgba(255,255,255,.72)';ctx.font='11px system-ui';ctx.fillText(`${t0.toFixed(1)} s`,pad.l,pad.t+h+23);ctx.fillText(`${t1.toFixed(1)} s`,pad.l+w-42,pad.t+h+23);
}

function drawAutoSyncOffsets(canvas,data) {
  if(!canvas||!data?.windows?.length)return;
  const rows=data.windows.filter(row=>String(row.accepted).toLowerCase()==='true'&&Number.isFinite(Number(row.pupil_rel_sec))&&Number.isFinite(Number(row.local_offset_ms)));
  if(rows.length<2)return;
  const dpr=Math.max(1,Math.min(2,window.devicePixelRatio||1));
  const cssWidth=Math.max(760,canvas.clientWidth||980),cssHeight=230;
  canvas.width=Math.round(cssWidth*dpr);canvas.height=Math.round(cssHeight*dpr);
  const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
  const pad={l:62,r:20,t:22,b:40},w=cssWidth-pad.l-pad.r,h=cssHeight-pad.t-pad.b;
  const xs=rows.map(r=>Number(r.pupil_rel_sec)),ys=rows.map(r=>Number(r.local_offset_ms));
  const x0=Math.min(...xs),x1=Math.max(...xs),xspan=Math.max(x1-x0,1e-9);
  let y0=Math.min(...ys),y1=Math.max(...ys);const extra=Math.max(1,(y1-y0)*.2);y0-=extra;y1+=extra;const yspan=Math.max(y1-y0,1e-9);
  const x=v=>pad.l+(v-x0)/xspan*w,y=v=>pad.t+h-(v-y0)/yspan*h;
  ctx.fillStyle='#f6f6ef';ctx.fillRect(0,0,cssWidth,cssHeight);ctx.strokeStyle='#d9d9cf';ctx.strokeRect(pad.l,pad.t,w,h);
  const drift=Number(data.metrics?.estimated_drift_ppm),start=Number(data.metrics?.estimated_offset_sec)*1000;
  if(Number.isFinite(drift)&&Number.isFinite(start)){ctx.strokeStyle='#77834a';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x(x0),y(start+drift*x0/1000));ctx.lineTo(x(x1),y(start+drift*x1/1000));ctx.stroke();}
  for(const row of rows){const xx=x(Number(row.pupil_rel_sec)),yy=y(Number(row.local_offset_ms));ctx.fillStyle='#315f59';ctx.beginPath();ctx.arc(xx,yy,4,0,Math.PI*2);ctx.fill();const cv=Number(row.cross_validated_residual_ms);if(Number.isFinite(cv)){ctx.strokeStyle='rgba(49,95,89,.35)';ctx.beginPath();ctx.moveTo(xx,yy-5);ctx.lineTo(xx,yy+5);ctx.stroke();}}
  ctx.fillStyle='#555b50';ctx.font='11px system-ui';ctx.fillText('Independent local offset (ms)',pad.l,pad.t-7);ctx.fillText(`${x0.toFixed(1)} s`,pad.l,pad.t+h+24);ctx.fillText(`${x1.toFixed(1)} s`,pad.l+w-42,pad.t+h+24);ctx.fillText(`${y1.toFixed(2)}`,8,pad.t+5);ctx.fillText(`${y0.toFixed(2)}`,8,pad.t+h);
}

async function openCalibrationSyncPreview({role='',artifactId='',candidateId=''}) {
  try{
    const data=await api(autoSyncPreviewUrl({artifactId,candidateId}));
    const m=data?.metrics||{};
    const rows=(data?.windows||[]).map(row=>`<tr><td>${esc(row.window_index??'—')}</td><td>${autoSyncMetric(row.pupil_rel_sec,2,' s')}</td><td>${autoSyncMetric(row.local_offset_ms,3,' ms')}</td><td>${autoSyncMetric(row.correlation,4)}</td><td>${autoSyncMetric(row.cross_validated_residual_ms,3,' ms')}</td><td>${autoSyncMetric(row.lag_uncertainty_ms,3,' ms')}</td><td>${String(row.accepted).toLowerCase()==='true'?'Accepted':'Rejected'}</td></tr>`).join('');
    modal(`${role||data?.wearer_role_id||''} synchronization evidence`,`
      <div class="sync-audit-modal">
        <div class="sync-audit-header"><div><span class="eyebrow">No ego video used</span><h3>${esc(data?.source_asset_label||'Pupil IMU')} ↔ ${esc(data?.target_asset_label||'ETM C3D')}</h3></div>${autoSyncPrecisionBadge(m)}</div>
        ${autoSyncSummary({confidence:data?.confidence,metrics:m})}
        <div class="sync-audit-chart-card"><h4>Aligned angular-motion evidence</h4><canvas id="sync-audit-trace"></canvas></div>
        <div class="sync-audit-chart-card"><h4>Independent window offsets and fitted drift</h4><canvas id="sync-audit-offsets"></canvas></div>
        <div class="sync-audit-table-wrap"><table class="sync-audit-table"><thead><tr><th>Window</th><th>Time</th><th>Local offset</th><th>Correlation</th><th>Holdout residual</th><th>Lag precision</th><th>Status</th></tr></thead><tbody>${rows||'<tr><td colspan="7">No distributed-window rows are available.</td></tr>'}</tbody></table></div>
        <div class="form-help"><strong>Interpretation:</strong> a co-start trim can make the starting offset nearly zero. It cannot force the independent window slope to zero. Drift is therefore reported with a 95% interval, and a residual below estimator resolution is shown as precision-limited rather than exact.</div>
      </div>`,(root)=>{requestAnimationFrame(()=>{drawAutoSyncTrace($('#sync-audit-trace',root),data);drawAutoSyncOffsets($('#sync-audit-offsets',root),data);});});
  }catch(error){showError(error,'Could not open synchronization evidence');}
}

function automaticTimeMapBody(body,maximumOffsetSec=120) {
  return {
    session_id:body.session_id,
    timeseries_asset_id:body.timeseries_asset_id,
    tracker_asset_id:body.tracker_asset_id||null,
    etm_assignment_artifact_id:body.etm_assignment_artifact_id||null,
    wearer_role_id:body.wearer_role_id||null,
    maximum_offset_sec:Number(maximumOffsetSec)||120,
    sample_rate_hz:120,
    save_if_high:true,
  };
}

async function queueGaze3D(body) {
  const preflight=await api('/api/gaze3d/time-preflight',{method:'POST',body});
  const response=await api('/api/jobs/gaze3d',{method:'POST',body});
  appState.pendingGaze3DJobId=response?.job?.id||null;
  const syncText=preflight?.method==='pretrimmed_shared_start'
    ? 'Native trim local zero ↔ first C3D frame; only retained interaction rows will be processed.'
    : Number(preflight?.anchor_count||0)>1
      ? `${preflight.anchor_count} synchronization anchors · RMS ${Number(preflight.rms_residual_ms||0).toFixed(3)} ms.`
      : 'Verified one-anchor synchronization; clock drift is not estimated.';
  toast('Gaze-ray job queued',syncText);
  await refreshState({quiet:false});
}

function openAutomaticSyncReview(result,body,context,maximumOffsetSec=120) {
  const candidate=result?.candidate;appState.autoSyncCandidate=candidate||null;
  modal('Review automatic Pupil ↔ Vicon synchronization',`
    <div class="warning-card"><strong>A plausible motion alignment was found</strong><p>OculoStudio did not save it automatically because one or more HIGH-confidence gates were not met. Review the metrics before accepting.</p></div>
    <div class="stat-list"><div class="stat-row"><span>Pupil source</span><strong>${esc(context?.source_asset_label||candidate?.source_asset_label||'—')}</strong></div><div class="stat-row"><span>Vicon tracker</span><strong>${esc(context?.target_asset_label||candidate?.target_asset_label||'—')}</strong></div></div>
    ${autoSyncSummary(candidate)}
    ${(candidate?.reasons||[]).length?`<div class="form-help">${candidate.reasons.map(esc).join('<br>')}</div>`:''}
    <div class="form-actions auto-sync-actions"><button class="soft-button" id="auto-sync-manual">Manual anchors</button><button class="soft-button" id="auto-sync-inspect">Inspect in Synchronize</button><button class="soft-button" id="auto-sync-wide">Retry wider search</button><button class="primary-button" id="auto-sync-accept">Accept & generate rays</button></div>`,(root,close)=>{
      $('#auto-sync-manual',root).onclick=()=>{appState.syncPrefill={...context};appState.timeAnchors=[{source_ns:'0',target_ns:'0',label:'Anchor 1',weight:1}];close();navigate('synchronize');};
      $('#auto-sync-inspect',root).onclick=()=>{appState.syncPrefill={...context};close();navigate('synchronize');};
      $('#auto-sync-wide',root).onclick=async()=>{close();await runAutomaticGazeSync(body,context,Math.max(600,Number(maximumOffsetSec)||120));};
      $('#auto-sync-accept',root).onclick=async()=>{const button=$('#auto-sync-accept',root);try{button.disabled=true;button.textContent='Saving mapping…';await api('/api/time-maps/auto-accept',{method:'POST',body:{session_id:selectedSessionId(),candidate_id:candidate.candidate_id}});appState.autoSyncCandidate=null;close();await refreshState({quiet:true});toast('Automatic synchronization accepted','The reviewed IMU↔ETM mapping is now part of the TimeGraph.');await queueGaze3D(body);}catch(error){button.disabled=false;button.textContent='Accept & generate rays';showError(error,'Could not accept automatic synchronization')}};
    });
}

function openAutomaticSyncFailure(result,body,context,maximumOffsetSec=120) {
  const candidate=result?.candidate||null;appState.autoSyncCandidate=candidate;
  const detail=result?.message||result?.error||'Automatic alignment did not produce an identifiable mapping.';
  modal('Automatic synchronization needs a fallback',`
    <div class="warning-card"><strong>No reliable automatic mapping was saved</strong><p>${esc(detail)}</p></div>
    ${candidate?autoSyncSummary(candidate):''}
    ${(candidate?.reasons||[]).length?`<div class="form-help">${candidate.reasons.map(esc).join('<br>')}</div>`:''}
    <p>Nothing has been aligned by first frame or zero offset. You can retry a wider search or enter shared-event anchors manually.</p>
    <div class="form-actions"><button class="soft-button" id="auto-sync-fallback-cancel">Cancel</button><button class="soft-button" id="auto-sync-fallback-wide">Retry ±600 s</button><button class="primary-button" id="auto-sync-fallback-manual">Open manual anchors</button></div>`,(root,close)=>{
      $('#auto-sync-fallback-cancel',root).onclick=close;
      $('#auto-sync-fallback-wide',root).onclick=async()=>{close();await runAutomaticGazeSync(body,context,Math.max(600,Number(maximumOffsetSec)||120));};
      $('#auto-sync-fallback-manual',root).onclick=()=>{appState.syncPrefill={...context};appState.timeAnchors=[{source_ns:'0',target_ns:'0',label:'Anchor 1',weight:1}];close();navigate('synchronize');};
    });
}

async function runAutomaticGazeSync(body,context,maximumOffsetSec=120) {
  const button=$('#run-gaze3d');const original=button?.textContent;
  try{
    if(button){button.disabled=true;button.textContent='Auto-synchronizing IMU ↔ ETM…';}
    const result=await api('/api/time-maps/auto-fit',{method:'POST',body:automaticTimeMapBody(body,maximumOffsetSec)});
    if(result?.status==='saved'){
      appState.autoSyncCandidate=null;
      await refreshState({quiet:true});
      toast('Automatic synchronization saved',`${result?.candidate?.metrics?.matched_window_count||0} motion windows · ${autoSyncMetric(result?.candidate?.metrics?.aligned_correlation,3)} aligned correlation.`);
      await queueGaze3D(body);return true;
    }
    if(result?.status==='review_required'){openAutomaticSyncReview(result,body,context,maximumOffsetSec);return false;}
    openAutomaticSyncFailure(result,body,context,maximumOffsetSec);return false;
  }catch(error){openAutomaticSyncFailure({message:String(error?.message||error)},body,context,maximumOffsetSec);return false;}
  finally{if(button?.isConnected){button.disabled=false;button.textContent=original||'Generate calibrated rays';}}
}

function captureCalibrationSyncSettings() {
  const maximum=Number($('#cal-sync-max-offset')?.value??appState.calibrationSyncSettings?.maximumOffsetSec??120);
  const sample=Number($('#cal-sync-sample-rate')?.value??appState.calibrationSyncSettings?.sampleRateHz??120);
  appState.calibrationSyncSettings={
    maximumOffsetSec:Number.isFinite(maximum)?maximum:120,
    sampleRateHz:Number.isFinite(sample)?sample:120,
    saveIfHigh:$('#cal-sync-save-high')?$('#cal-sync-save-high').dataset.on==='true':appState.calibrationSyncSettings?.saveIfHigh!==false,
  };
  $$('.cal-sync-timeseries').forEach(select=>{appState.calibrationSyncSelections[String(select.dataset.role||'').toUpperCase()]=select.value||'';});
  return appState.calibrationSyncSettings;
}

function calibrationAutoSyncRequest(roleId) {
  const role=String(roleId||'').toUpperCase();
  const stage=calibrationWorkflowState(role);
  const assignment=stage.wearerAssignment;
  if(!assignment)throw new Error('Resolve the interaction wearer before synchronizing clocks');
  const rows=calibrationTimeseriesForRole(role);
  const selected=$(`#cal-sync-timeseries-${role.toLowerCase()}`)?.value||calibrationSyncSelection(role,rows);
  if(!selected)throw new Error(`Select the ${role} Pupil Timeseries containing imu.csv`);
  const settings=captureCalibrationSyncSettings();
  return {
    session_id:selectedSessionId(),
    timeseries_asset_id:selected,
    tracker_asset_id:null,
    etm_assignment_artifact_id:assignment.id,
    wearer_role_id:role,
    maximum_offset_sec:Number(settings.maximumOffsetSec)||120,
    sample_rate_hz:Number(settings.sampleRateHz)||120,
    save_if_high:settings.saveIfHigh!==false,
  };
}

async function performCalibrationAutoSync(roleId) {
  const role=String(roleId||'').toUpperCase();
  const request=calibrationAutoSyncRequest(role);
  const result=await api('/api/time-maps/auto-fit',{method:'POST',body:request});
  if(result?.status==='saved'){
    appState.calibrationAutoSyncCandidates[role]=null;
    toast(`${role} clock synchronization saved`,`${result?.candidate?.metrics?.matched_window_count||0} distributed motion windows · drift ${autoSyncMetric(result?.candidate?.metrics?.estimated_drift_ppm,1,' ppm')}.`);
  }else{
    appState.calibrationAutoSyncCandidates[role]=result||{status:'failed',message:'No synchronization result was returned'};
    if(result?.status==='review_required')toast(`${role} synchronization needs review`,'Inspect the offset, drift and distributed-window evidence, then accept only if it is plausible.');
    else toast(`${role} automatic synchronization failed`,result?.message||'No reliable IMU↔ETM alignment was found.',true);
  }
  return result;
}

async function runCalibrationAutoSync(roleId) {
  const role=String(roleId||'').toUpperCase();
  if(appState.calibrationAutoSyncRunningRole)return;
  appState.calibrationAutoSyncRunningRole=role;
  renderProperties();
  try{
    await performCalibrationAutoSync(role);
    await refreshState({quiet:true});
  }catch(error){showError(error,`Could not synchronize ${role} Pupil IMU to the assigned ETM`);}
  finally{appState.calibrationAutoSyncRunningRole='';renderProperties();}
}

async function runCalibrationAutoSyncBoth() {
  if(appState.calibrationAutoSyncRunningRole)return;
  appState.calibrationAutoSyncRunningRole='BOTH';
  renderProperties();
  try{
    for(const role of ['R01','R02'])await performCalibrationAutoSync(role);
    await refreshState({quiet:true});
  }catch(error){showError(error,'Could not complete both role-specific clock synchronizations');}
  finally{appState.calibrationAutoSyncRunningRole='';renderProperties();}
}

async function acceptCalibrationAutoSync(roleId) {
  const role=String(roleId||'').toUpperCase();
  const result=appState.calibrationAutoSyncCandidates?.[role];
  const candidate=result?.candidate;
  if(!candidate?.candidate_id){showError(new Error('The synchronization candidate is unavailable; rerun automatic synchronization.'),'Could not accept synchronization');return;}
  try{
    await api('/api/time-maps/auto-accept',{method:'POST',body:{session_id:selectedSessionId(),candidate_id:candidate.candidate_id}});
    appState.calibrationAutoSyncCandidates[role]=null;
    await refreshState({quiet:true});
    toast(`${role} synchronization accepted`,'The reviewed affine TimeGraph mapping is now linked to Calibration.');
  }catch(error){showError(error,`Could not accept ${role} synchronization`);}
}


function automaticBodyFromSyncPrefill(pre,maximumOffsetSec=120) {
  return {
    session_id:selectedSessionId(),
    timeseries_asset_id:pre?.source_asset_id||'',
    tracker_asset_id:pre?.tracker_asset_id||pre?.target_asset_id||null,
    etm_assignment_artifact_id:pre?.assignment_artifact_id||null,
    wearer_role_id:pre?.assignment_artifact_id?(pre?.wearer_role_id||null):null,
    maximum_offset_sec:Number(maximumOffsetSec)||120,
    sample_rate_hz:120,
    save_if_high:true,
  };
}

async function runAutomaticSyncFromPage() {
  const pre=appState.syncPrefill?.session_id===selectedSessionId()?appState.syncPrefill:null;
  if(!pre){showError(new Error('Open Synchronize from a blocked Gaze 3D request so the Pupil and ETM assets are known.'),'Automatic synchronization needs a Gaze 3D context');return;}
  const button=$('#run-auto-sync-from-sync');const original=button?.textContent;
  try{
    if(button){button.disabled=true;button.textContent='Comparing IMU and ETM motion…';}
    const maximum=Number($('#sync-auto-max-offset')?.value||120);
    const result=await api('/api/time-maps/auto-fit',{method:'POST',body:automaticBodyFromSyncPrefill(pre,maximum)});
    appState.autoSyncCandidate=result?.candidate||null;
    if(result?.status==='saved'){
      appState.autoSyncCandidate=null;
      await refreshState({quiet:true});
      toast('Automatic synchronization saved','A HIGH-confidence IMU↔ETM mapping was added to the TimeGraph.');
      appState.syncPrefill=null;navigate(pre.return_page||'gaze3d');return;
    }
    renderProperties();
    const panel=$('#properties-content');if(panel)panel.scrollTop=0;
    if(result?.status==='review_required')toast('Review automatic synchronization','The candidate has not been saved. Inspect the metrics and accept only if they are plausible.');
    else toast('Automatic synchronization not saved',result?.message||'Use manual shared-event anchors as a fallback.',true);
  }catch(error){appState.autoSyncCandidate=null;renderProperties();showError(error,'Automatic synchronization could not run');}
  finally{if(button?.isConnected){button.disabled=false;button.textContent=original||'Auto-synchronize IMU ↔ ETM motion';}}
}

async function acceptAutomaticSyncFromPage() {
  const candidate=appState.autoSyncCandidate;
  const pre=appState.syncPrefill;
  if(!candidate?.candidate_id){showError(new Error('No reviewable automatic synchronization candidate is available.'),'Nothing to accept');return;}
  const button=$('#accept-auto-sync-from-sync');
  try{
    if(button){button.disabled=true;button.textContent='Saving reviewed mapping…';}
    await api('/api/time-maps/auto-accept',{method:'POST',body:{session_id:selectedSessionId(),candidate_id:candidate.candidate_id}});
    appState.autoSyncCandidate=null;await refreshState({quiet:true});toast('Reviewed synchronization saved');appState.syncPrefill=null;navigate(pre?.return_page||'gaze3d');
  }catch(error){if(button){button.disabled=false;button.textContent='Accept reviewed mapping';}showError(error,'Could not accept automatic synchronization');}
}

function sharedAudioSyncArtifacts(roleId='') {
  const role=String(roleId||'').toUpperCase();
  return artifacts(['time_sync']).filter(row=>{
    if(String(row.parameters?.sync_type||row.qc?.sync_type||'')!=='shared_audio_video')return false;
    const artifactRole=String(row.parameters?.wearer_role_id||row.qc?.wearer_role_id||'').toUpperCase();
    return !role||!artifactRole||artifactRole===role;
  }).sort(byCreated);
}

function audioVideoSyncAssetPools(roleId='R01') {
  const role=String(roleId||'R01').toUpperCase();
  const ctx=activeWorkflowContext();
  const rows=assets(['video','pupil_native','pupil_cloud']).filter(asset=>{
    const identity=assetIdentity(asset);
    if(ctx.interactionId&&identity.interaction_id&&identity.interaction_id!==ctx.interactionId)return false;
    if(role&&identity.role_id&&identity.role_id!==role)return false;
    return true;
  });
  const text=asset=>[asset.label,asset.source_path].filter(Boolean).join(' ').toLowerCase();
  const sourceScore=asset=>(asset.kind==='video'?50:0)+(/facial|expression|face|front|portrait/.test(text(asset))?40:0)-(/ego|native|scene/.test(text(asset))?15:0);
  const referenceScore=asset=>(asset.kind==='pupil_native'?70:asset.kind==='pupil_cloud'?45:0)+(/ego|native|scene|neon/.test(text(asset))?30:0)-(/facial|expression|portrait/.test(text(asset))?20:0);
  return {
    source:rows.slice().sort((a,b)=>sourceScore(b)-sourceScore(a)||String(a.label).localeCompare(String(b.label))),
    reference:rows.slice().sort((a,b)=>referenceScore(b)-referenceScore(a)||String(a.label).localeCompare(String(b.label))),
  };
}

function audioVideoSyncMetric(value,digits=2,suffix='') {
  const number=Number(value);return Number.isFinite(number)?`${number.toFixed(digits)}${suffix}`:'—';
}

function audioVideoSyncPanel({roleId='',compact=false}={}) {
  const role=String(roleId||appState.audioVideoSyncRole||calibrationContextRole()||'R01').toUpperCase();
  appState.audioVideoSyncRole=role;
  const pools=audioVideoSyncAssetPools(role);
  const sourceIds=new Set(pools.source.map(row=>row.id)),referenceIds=new Set(pools.reference.map(row=>row.id));
  const sourceDefault=sourceIds.has(appState.audioVideoSyncSourceId)?appState.audioVideoSyncSourceId:(pools.source.find(row=>row.kind==='video')||pools.source[0])?.id||'';
  const referenceDefault=referenceIds.has(appState.audioVideoSyncReferenceId)?appState.audioVideoSyncReferenceId:(pools.reference.find(row=>row.kind==='pupil_native')||pools.reference[0])?.id||'';
  appState.audioVideoSyncSourceId=sourceDefault;appState.audioVideoSyncReferenceId=referenceDefault;
  const accepted=sharedAudioSyncArtifacts(role).slice().reverse();
  const running=Boolean(appState.pendingAudioVideoSyncJobId);
  const acceptedRows=accepted.length?accepted.slice(0,4).map(row=>{
    const metrics=row.qc||{};const candidateId=row.parameters?.candidate_id||'';
    return `<article class="avsync-result-card"><div><strong>${esc(row.label)}</strong><small>${audioVideoSyncMetric(metrics.estimated_offset_sec,3,' s')} offset · ${audioVideoSyncMetric(metrics.estimated_drift_ppm,1,' ppm')} drift · P95 ${audioVideoSyncMetric(metrics.p95_residual_ms,1,' ms')}</small></div><span class="status-pill ${metrics.visual_review_confirmed?'good':'warning'}">${metrics.visual_review_confirmed?'Visual verified':'Review required'}</span>${candidateId?`<button type="button" class="soft-button compact avsync-review" data-candidate-id="${esc(candidateId)}" data-artifact-id="${esc(row.id)}">Review</button>`:''}</article>`;
  }).join(''):'<div class="empty-tree">No accepted shared-audio video synchronization for this role.</div>';
  return `<section class="property-section shared-audio-sync-panel ${compact?'compact':''}"><div class="section-label">${compact?'Shared-audio video synchronization':'3A · Shared-audio facial ↔ ego video sync · optional'}</div><div class="property-stack">
    <div class="form-help"><strong>For two devices recording the same physical interaction.</strong> Common speech and acoustic events are matched across distributed windows. Low-agreement or ambiguous windows can be rejected; short one-sided sounds are tolerated only when the remaining pattern still supports the same lag. The result estimates both start offset and clock drift.</div>
    <div class="field"><label>Wearer / interaction role</label><select id="avsync-role"><option value="R01" ${role==='R01'?'selected':''}>R01</option><option value="R02" ${role==='R02'?'selected':''}>R02</option></select></div>
    <div class="field"><label>Facial-expression video · source clock</label><select id="avsync-source">${optionRows(pools.source,sourceDefault,'Select facial-expression video',row=>`${row.label} · ${String(row.kind).replaceAll('_',' ')}`)}</select></div>
    <div class="field"><label>Ego video · reference clock</label><select id="avsync-reference">${optionRows(pools.reference,referenceDefault,'Select ego / Native video',row=>`${row.label} · ${String(row.kind).replaceAll('_',' ')}`)}</select></div>
    <div class="field"><label>Maximum start-offset search (s)</label><input id="avsync-max-offset" type="number" min="1" max="7200" step="1" value="${esc(appState.audioVideoSyncMaximumOffsetSec||300)}"></div>
    <details><summary>Scientific acceptance contract</summary><div class="form-help" style="margin-top:8px">The automatic candidate uses a single affine map, distributed local audio matches, low-agreement-window rejection, short-mismatch tolerance, sub-sample lag refinement and leave-one-window-out validation. It never uses arbitrary dynamic time warping. A split-video review at start, middle and end is required before the mapping is published to the TimeGraph.</div></details>
    <button class="primary-button full" id="run-audio-video-sync" ${running||!sourceDefault||!referenceDefault||sourceDefault===referenceDefault?'disabled':''}>${running?'Matching shared audio…':'Auto-align videos from shared audio'}</button>
    <div class="avsync-accepted-list">${acceptedRows}</div>
  </div></section>`;
}

function applyAudioVideoTimeMap(mapping,sourceSec) {
  const sourceNs=Number(sourceSec)*1e9;
  if(mapping?.source_origin_ns!=null&&mapping?.target_origin_ns!=null){
    return (Number(mapping.target_origin_ns)+Number(mapping.scale||1)*(sourceNs-Number(mapping.source_origin_ns)))/1e9;
  }
  return (Number(mapping?.scale||1)*sourceNs+Number(mapping?.offset_ns||0))/1e9;
}

function drawAudioVideoSyncChart(canvas,preview) {
  if(!canvas)return;
  const ctx=canvas.getContext('2d');const ratio=window.devicePixelRatio||1;
  const width=Math.max(720,canvas.clientWidth||900),height=260;
  canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);ctx.setTransform(ratio,0,0,ratio,0,0);
  ctx.clearRect(0,0,width,height);ctx.fillStyle='#f7f7ef';ctx.fillRect(0,0,width,height);
  const rows=(preview.signals||[]).filter(row=>Number.isFinite(Number(row.source_video_sec)));
  if(rows.length<2){ctx.fillStyle='#666';ctx.fillText('No aligned audio trace is available.',20,30);return;}
  const xs=rows.map(row=>Number(row.source_video_sec)),minX=Math.min(...xs),maxX=Math.max(...xs);const span=Math.max(maxX-minX,1e-6);
  const x=value=>42+(Number(value)-minX)/span*(width-62);
  const y=(value,base)=>base-Math.max(-3,Math.min(3,Number(value)||0))*26;
  ctx.strokeStyle='#d5d7c8';ctx.lineWidth=1;[70,130,190].forEach(v=>{ctx.beginPath();ctx.moveTo(42,v);ctx.lineTo(width-20,v);ctx.stroke();});
  (preview.windows||[]).forEach(row=>{const px=x(row.source_video_sec);ctx.strokeStyle=row.accepted===true||String(row.accepted).toLowerCase()==='true'?'rgba(143,170,20,.42)':'rgba(190,120,55,.22)';ctx.beginPath();ctx.moveTo(px,18);ctx.lineTo(px,height-24);ctx.stroke();});
  const trace=(key,base,stroke)=>{ctx.beginPath();let started=false;rows.forEach(row=>{const value=Number(row[key]);if(!Number.isFinite(value)){started=false;return;}const px=x(row.source_video_sec),py=y(value,base);if(!started){ctx.moveTo(px,py);started=true}else ctx.lineTo(px,py);});ctx.strokeStyle=stroke;ctx.lineWidth=1.4;ctx.stroke();};
  trace('source_signature',82,'#697512');trace('reference_signature',177,'#4b6282');
  ctx.fillStyle='#30351e';ctx.font='12px system-ui';ctx.fillText('Facial-video audio pattern',48,24);ctx.fillStyle='#697512';ctx.fillRect(24,16,16,3);ctx.fillStyle='#30351e';ctx.fillText('Mapped ego-video audio pattern',48,120);ctx.fillStyle='#4b6282';ctx.fillRect(24,112,16,3);
  ctx.fillStyle='#666';ctx.fillText(`${minX.toFixed(1)} s`,42,height-8);ctx.textAlign='right';ctx.fillText(`${maxX.toFixed(1)} s`,width-20,height-8);ctx.textAlign='left';
}

async function openAudioVideoSyncReview({candidateId='',artifactId=''}={}) {
  if(!candidateId&&!artifactId)return;
  try{
    const query=candidateId?`candidate_id=${encodeURIComponent(candidateId)}`:`artifact_id=${encodeURIComponent(artifactId)}`;
    const preview=await api(`/api/time-maps/audio-video/preview/${encodeURIComponent(selectedSessionId())}?${query}&maximum_samples=2600`);
    const resolvedCandidate=preview.candidate_id||candidateId;const metrics=preview.metrics||{};const accepted=preview.accepted||null;
    const mapping=accepted?.time_map||preview.preview_time_map;const failed=preview.confidence==='FAILED'||!mapping;
    const acceptedWindows=(preview.windows||[]).filter(row=>row.accepted===true||String(row.accepted).toLowerCase()==='true');
    const sourceTimes=acceptedWindows.map(row=>Number(row.source_video_sec)).filter(Number.isFinite).sort((a,b)=>a-b);
    const first=sourceTimes[0]??0,last=sourceTimes[sourceTimes.length-1]??Math.max(1,Number(metrics.source_audio?.video_duration_sec||30)-1),middle=sourceTimes.length?sourceTimes[Math.floor(sourceTimes.length/2)]:(first+last)/2;
    const alreadyAccepted=Boolean(accepted||artifactId);
    const initialCorrection=alreadyAccepted?0:Number(metrics.visual_offset_adjustment_ms||0);
    const body=`<div class="avsync-review-root">
      <div class="avsync-review-summary"><article><span>Confidence</span><strong>${esc(preview.confidence||'—')}</strong></article><article><span>Offset</span><strong>${audioVideoSyncMetric(metrics.estimated_offset_sec,4,' s')}</strong></article><article><span>Drift</span><strong>${audioVideoSyncMetric(metrics.estimated_drift_ppm,2,' ppm')}</strong></article><article><span>Accepted windows</span><strong>${esc(metrics.matched_window_count??acceptedWindows.length)} / ${esc(metrics.window_count??'—')}</strong></article><article><span>LOO P95 residual</span><strong>${audioVideoSyncMetric(metrics.p95_residual_ms,2,' ms')}</strong></article><article><span>Coverage</span><strong>${audioVideoSyncMetric(Number(metrics.matched_time_coverage_fraction||0)*100,1,'%')}</strong></article></div>
      <div class="${failed?'warning-card':'success-card'}"><strong>${failed?'No publishable mapping':'Shared physical audio pattern found'}</strong><p>${(preview.reasons||[]).map(esc).join('<br>')||'Review the evidence below.'}</p></div>
      <canvas id="avsync-chart" class="avsync-chart" aria-label="Aligned audio-pattern traces"></canvas>
      <div class="avsync-position-buttons"><button class="soft-button compact" data-avsync-position="${first}">Review start</button><button class="soft-button compact" data-avsync-position="${middle}">Review middle</button><button class="soft-button compact" data-avsync-position="${last}">Review end</button></div>
      <div class="avsync-video-grid"><section><header><strong>Facial-expression video</strong><small>Master playback · source clock</small></header><video id="avsync-source-video" controls preload="metadata" playsinline src="/api/media/audio-video-sync/${encodeURIComponent(selectedSessionId())}/${encodeURIComponent(resolvedCandidate)}/source"></video></section><section><header><strong>Ego video</strong><small>Follows the candidate TimeGraph</small></header><video id="avsync-reference-video" controls preload="metadata" playsinline muted src="/api/media/audio-video-sync/${encodeURIComponent(selectedSessionId())}/${encodeURIComponent(resolvedCandidate)}/reference"></video></section></div>
      <div class="avsync-review-controls"><label>Audio monitor<select id="avsync-audio-monitor"><option value="source">Facial video only</option><option value="reference">Ego video only</option><option value="mute">Mute both</option></select></label><label>Visual A/V correction (ms)<input id="avsync-visual-correction" type="number" min="-2000" max="2000" step="1" value="${esc(initialCorrection)}" ${alreadyAccepted?'disabled':''}></label><div class="form-help">Positive correction advances to a later ego-video time for the same facial-video frame. Use only for a stable device-internal A/V latency visible at start, middle and end.</div></div>
      <div id="avsync-playback-status" class="form-help">The facial video is the playback master. The ego video is muted by default to prevent echo.</div>
      ${alreadyAccepted?`<div class="success-card compact"><strong>Accepted TimeGraph</strong><p>Visual review: ${accepted?.visual_review_confirmed?'confirmed':'not confirmed'} · applied correction ${audioVideoSyncMetric(accepted?.visual_offset_adjustment_ms,1,' ms')}</p></div>`:failed?'':`<label class="check-row avsync-confirm"><input id="avsync-visual-confirm" type="checkbox"><span>I reviewed start, middle and end, and the visible mouth/impact timing is consistent within approximately one video frame.</span></label><button class="primary-button full" id="avsync-accept">Accept reviewed mapping into TimeGraph</button>`}
    </div>`;
    modal('Shared-audio facial ↔ ego video synchronization',body,(root,close)=>{
      $('.modal',root)?.classList.add('avsync-modal');
      requestAnimationFrame(()=>drawAudioVideoSyncChart($('#avsync-chart',root),preview));
      const source=$('#avsync-source-video',root),reference=$('#avsync-reference-video',root),correction=$('#avsync-visual-correction',root),status=$('#avsync-playback-status',root),monitor=$('#avsync-audio-monitor',root);
      let timer=null;
      const correctionSec=()=>Number(correction?.value||0)/1000;
      const mappedTime=()=>Math.max(0,applyAudioVideoTimeMap(mapping,Number(source?.currentTime||0))+correctionSec());
      const syncReference=({force=false}={})=>{
        if(!source||!reference||!mapping)return;
        const target=mappedTime();if(!Number.isFinite(target))return;
        const difference=target-Number(reference.currentTime||0);
        if(force||Math.abs(difference)>.075){try{reference.currentTime=Math.min(Math.max(0,target),Number.isFinite(reference.duration)?Math.max(0,reference.duration-.001):target);}catch(_){}}
        reference.playbackRate=Math.max(.5,Math.min(2,Number(source.playbackRate||1)*Number(mapping.scale||1)));
        if(status)status.textContent=`Facial ${source.currentTime.toFixed(3)} s → ego ${target.toFixed(3)} s · residual ${(difference*1000).toFixed(1)} ms`;
      };
      const stopTimer=()=>{if(timer){clearInterval(timer);timer=null;}};
      source?.addEventListener('play',async()=>{syncReference({force:true});try{await reference.play()}catch(_){};timer=setInterval(()=>syncReference(),180);});
      source?.addEventListener('pause',()=>{reference.pause();stopTimer();syncReference({force:true});});
      source?.addEventListener('ended',()=>{reference.pause();stopTimer();});
      source?.addEventListener('seeking',()=>syncReference({force:true}));source?.addEventListener('ratechange',()=>syncReference({force:true}));
      correction?.addEventListener('input',()=>syncReference({force:true}));
      monitor?.addEventListener('change',()=>{const value=monitor.value;source.muted=value!=='source';reference.muted=value!=='reference';});
      $$('[data-avsync-position]',root).forEach(button=>button.onclick=()=>{source.pause();source.currentTime=Math.max(0,Number(button.dataset.avsyncPosition||0)-1.5);syncReference({force:true});});
      source?.addEventListener('loadedmetadata',()=>{source.currentTime=Math.max(0,first-1.5);syncReference({force:true});});
      $('#avsync-accept',root)?.addEventListener('click',async()=>{
        const button=$('#avsync-accept',root);try{
          if(!$('#avsync-visual-confirm',root)?.checked)throw new Error('Confirm the start/middle/end visual review before accepting the mapping.');
          button.disabled=true;button.textContent='Publishing reviewed TimeGraph…';
          await api('/api/time-maps/audio-video/accept',{method:'POST',body:{session_id:selectedSessionId(),candidate_id:resolvedCandidate,visual_offset_adjustment_ms:Number(correction?.value||0),visual_review_confirmed:true}});
          appState.audioVideoSyncCandidate=null;close();toast('Shared-audio video TimeGraph saved','The facial-expression video now resolves to the ego/Pupil clock.');await refreshState({quiet:false,fullRender:true});
        }catch(error){button.disabled=false;button.textContent='Accept reviewed mapping into TimeGraph';showError(error,'Could not accept shared-audio synchronization');}
      });
      $('.modal-close',root)?.addEventListener('click',stopTimer,{once:true});
    });
  }catch(error){showError(error,'Could not open shared-audio synchronization review');}
}

async function submitAudioVideoSync() {
  const role=String($('#avsync-role')?.value||appState.audioVideoSyncRole||'R01').toUpperCase();
  const source=$('#avsync-source')?.value||'';const reference=$('#avsync-reference')?.value||'';const maxOffset=Number($('#avsync-max-offset')?.value||300);
  appState.audioVideoSyncRole=role;appState.audioVideoSyncSourceId=source;appState.audioVideoSyncReferenceId=reference;appState.audioVideoSyncMaximumOffsetSec=maxOffset;
  if(!source||!reference){showError(new Error('Select both the facial-expression video and the ego video.'),'Shared-audio inputs incomplete');return;}
  if(source===reference){showError(new Error('The source and reference must be different video assets.'),'Shared-audio inputs invalid');return;}
  const button=$('#run-audio-video-sync');
  try{
    if(button){button.disabled=true;button.textContent='Queueing shared-audio match…';}
    const response=await api('/api/jobs/audio-video-sync',{method:'POST',body:{session_id:selectedSessionId(),source_video_asset_id:source,reference_video_asset_id:reference,wearer_role_id:role,maximum_offset_sec:maxOffset,decode_rate_hz:8000,feature_rate_hz:100}});
    appState.pendingAudioVideoSyncJobId=response?.job?.id||null;
    toast('Shared-audio synchronization queued','OculoStudio will tolerate or reject one-sided instructions according to local agreement, then fit one offset+drift mapping from distributed common sound.');
    await refreshState({quiet:true});renderProperties();
  }catch(error){showError(error,'Could not start shared-audio video synchronization');}
  finally{if(button?.isConnected){button.disabled=Boolean(appState.pendingAudioVideoSyncJobId);button.textContent=appState.pendingAudioVideoSyncJobId?'Matching shared audio…':'Auto-align videos from shared audio';}}
}


function renderProperties() {
  const item = PIPELINE.find(x => x.id === appState.page) || PIPELINE[0];
  $('#module-card').innerHTML = `<div class="module-wrap"><div class="section-label">Module</div><div class="module-card"><span class="module-icon">${item.icon}</span><span><b>${esc(item.label)}</b><small>${esc(item.desc)}</small></span></div></div>`;
  if (!project()) {
    $('#properties-content').innerHTML = `<div class="empty-tree">Create or open a project to configure this module.</div>`; return;
  }
  const renderers = { project: projectProperties, data: dataProperties, clips: clipsProperties, privacy: privacyProperties, gaze3d: gaze3dProperties, targets: targetsProperties, dyadic: dyadicProperties, qc: qcProperties, advanced: advancedProperties, synchronize: syncProperties, calibration: calibrationProperties };
  $('#properties-content').innerHTML = (renderers[appState.page] || projectProperties)();
  bindPropertyActions();
}

function projectProperties() {
  const p=project(),s=session(),ws=projectWorkspace();
  const sessionOptions=(ws.sessions||[]).map(row=>`<option value="${esc(row.session_id)}" ${row.selected?'selected':''}>${esc(row.session_code)} · ${esc(row.label)}</option>`).join('');
  const binding=sessionCalibrationBinding();const bound=(ws.calibration_sets||[]).find(row=>row.id===binding?.calibration_set_id)||null;
  return `<section class="property-section"><div class="section-label">Project storage</div><div class="property-stack"><div class="field"><label>Project name</label><input value="${esc(p.name)}" disabled></div><div class="field"><label>Project root</label><textarea disabled>${esc(p.root)}</textarea></div><div class="field"><label>Folder layout</label><input value="${ws.folder_layout_version>=3?'Project / Calibration Sets / Sessions':ws.folder_layout_version>=2?'Project / Sessions / legacy shared Calibration':'Legacy project layout'}" disabled></div></div></section><section class="property-section"><div class="section-label">Selected Session</div><div class="property-stack"><div class="field"><label>Session</label><select id="project-session-select">${sessionOptions}</select></div><div class="field"><label>Session folder</label><textarea disabled>${esc(sessionStoragePath('root')||p.root)}</textarea></div><div class="field"><label>Inherited Calibration</label><input value="${esc(bound?`${bound.code} · revision ${bound.active_revision_number??'—'}`:'Not assigned')}" disabled></div><button class="primary-button full" id="project-manage-sessions-side">Manage Sessions, Interactions & Calibration Sets</button><button class="soft-button full" id="open-session-folder">Open selected Session folder</button></div></section><section class="property-section"><div class="section-label">Project Calibration Sets</div><div class="property-stack"><div class="form-help">Calibration recordings and profile revisions are registered once at Project level. Sessions inherit a selected revision; Interaction wearer assignment and clock synchronization are never copied.</div><button class="soft-button full" id="open-calibration-sets-folder">Open Calibration Sets folder</button><button class="soft-button full" id="open-project-folder">Open Project folder</button></div></section><section class="property-section"><div class="section-label">Project lifecycle</div><div class="property-stack"><button class="soft-button full" id="project-start-fresh-side">Create another Project</button><button class="soft-button full" data-go="advanced">Advanced / recovery tools</button></div></section>`;
}

function dataProperties() {
  const binding=sessionCalibrationBinding();
  const bound=calibrationSetRows().find(row=>row.id===binding?.calibration_set_id)||null;
  const ctx=activeWorkflowContext();
  const sessionCode=sessionWorkspace()?.session_code||session()?.label||'Session';
  const selected=session()?.assets?.[appState.selectedAssetId]||null;
  return `<section class="property-section"><div class="section-label">Data management scopes</div><div class="property-stack">
    <div class="data-property-scope calibration"><span>Project-level</span><strong>Calibration Data</strong><small>${esc(bound?`${bound.code} · revision ${bound.active_revision_number??'—'} · inherited by ${(bound.applicable_session_ids||[]).length} Sessions`:'No Calibration Set assigned')}</small><button class="soft-button full" id="data-jump-calibration">Open Calibration Data block</button></div>
    <div class="data-property-scope interaction"><span>Session-local</span><strong>Interaction Data</strong><small>${esc(`${sessionCode} / ${ctx.interactionId||'No Interaction selected'}${ctx.roleId?' · '+ctx.roleId:''}`)}</small><button class="soft-button full" id="data-jump-interaction">Open Interaction Data block</button></div>
    <div class="form-help">All source-file importing, appending, Set assignment and identity editing now lives in the two Data workspaces. The Calibration page performs calibration calculations only.</div>
  </div></section>
  <section class="property-section"><div class="section-label">Selected Interaction asset</div><div class="property-stack">${selected?`<div class="stat-list"><div class="stat-row"><span>Asset</span><strong class="stat-value-wrap">${esc(selected.label)}</strong></div><div class="stat-row"><span>Kind</span><strong>${esc(selected.kind)}</strong></div><div class="stat-row"><span>Size</span><strong>${fmtBytes(selected.size_bytes)}</strong></div><div class="stat-row"><span>Recording ID</span><strong class="stat-value-wrap">${esc(selected.recording_id||'—')}</strong></div></div><button class="primary-button full" id="data-edit-current-asset" data-id="${esc(selected.id)}">Edit identity</button>`:'<div class="empty-tree">Select an Interaction asset in the main Data workspace.</div>'}</div></section>
  <section class="property-section"><div class="section-label">Folder contract</div><div class="form-help mono">calibration_sets/CAL.../sources/<br>sessions/S###/imports/<br>sessions/S###/derived/</div><div class="form-help">Calibration sources are registered once at Project level. Interaction files remain Session-local and never enter a Calibration Set.</div></section>`;
}
function syncProperties() {
  const ds=domains();
  const pre=appState.syncPrefill?.session_id===selectedSessionId()?appState.syncPrefill:null;
  const sourceSelected=pre?.source_domain_id||'';
  const targetSelected=pre?.target_domain_id||session()?.global_time_domain_id||'';
  const targetSeconds=Boolean(pre?.target_domain_id);
  const candidate=(appState.autoSyncCandidate && (!pre || (appState.autoSyncCandidate.source_domain_id===pre.source_domain_id && appState.autoSyncCandidate.target_domain_id===pre.target_domain_id)))?appState.autoSyncCandidate:null;
  const helper=pre?`<section class="property-section auto-sync-primary"><div class="section-label">Automatic Pupil ↔ Vicon synchronization</div><div class="property-stack">
    <div class="form-help"><strong>${esc(pre.source_asset_label)}</strong> must be synchronized to <strong>${esc(pre.target_asset_label)}</strong> before world-space rays can be generated. OculoStudio will compare Pupil gyroscope magnitude with the assigned ETM rigid-body angular velocity.</div>
    <div class="stat-list"><div class="stat-row"><span>Pupil source</span><strong>${esc(pre.source_domain_label)}</strong></div><div class="stat-row"><span>Vicon target</span><strong>${esc(pre.target_domain_label)}</strong></div><div class="stat-row"><span>Wearer</span><strong>${esc(pre.wearer_role_id||'Manual tracker')}</strong></div></div>
    <div class="field"><label>Maximum offset search (s)</label><input id="sync-auto-max-offset" type="number" value="120" min="1" max="3600" step="1"></div>
    ${candidate?autoSyncSummary(candidate):'<div class="form-help">HIGH-confidence mappings are saved automatically. REVIEW candidates remain unsaved until accepted. FAILED candidates do not modify the TimeGraph.</div>'}
    ${(candidate?.reasons||[]).length?`<div class="form-help">${candidate.reasons.map(esc).join('<br>')}</div>`:''}
    <button class="primary-button full" id="run-auto-sync-from-sync">${candidate?'Run automatic synchronization again':'Auto-synchronize IMU ↔ ETM motion'}</button>
    ${candidate?.confidence==='REVIEW'?'<button class="soft-button full" id="accept-auto-sync-from-sync">Accept reviewed mapping</button>':''}
    <button class="soft-button full" id="cancel-gaze-sync-prefill">Return to Gaze 3D without saving</button>
  </div></section>`:'';
  const manualOpen=!pre;
  return `${helper}${audioVideoSyncPanel({compact:true})}<section class="property-section"><details id="manual-sync-details" ${manualOpen?'open':''}><summary>${pre?'Advanced fallback · manual shared-event anchors':'Fit affine time map'}</summary><div class="property-stack" style="margin-top:10px">
    ${pre?'<div class="form-help">Use this only when automatic motion alignment is unavailable or ambiguous. Choose the same physical event in both recordings. One anchor estimates offset only; at least two separated anchors estimate drift.</div>':''}
    <div class="field"><label>Source clock</label><select id="sync-source">${optionRows(ds,sourceSelected,'Select source',d=>`${d.label} · ${d.clock_type}`)}</select></div>
    <div class="field"><label>Target clock</label><select id="sync-target">${optionRows(ds,targetSelected,'Select target',d=>`${d.label} · ${d.clock_type}`)}</select></div>
    <div class="field"><label>Method</label><select id="sync-method"><option value="robust_affine">Robust affine</option><option value="manual_affine">Direct / one anchor</option></select></div>
    <div class="field"><label>Shared-event anchors</label><div id="anchor-list" class="anchor-list">${anchorRows(targetSeconds)}</div><button class="soft-button" id="add-anchor">＋ Anchor</button></div>
    <button class="primary-button full" id="fit-time-map">Fit & save manual mapping</button>
  </div></details></section>
  <section class="property-section"><div class="section-label">Model</div><div class="form-help mono">t_target = target_origin + scale × (t_source − source_origin)</div>${pre?'<div class="form-help">After a verified mapping is saved, return to Gaze 3D and click Generate calibrated rays. Existing failed jobs remain only as audit history.</div>':''}</section>`;
}
function anchorRows(targetSeconds=false) {
  return appState.timeAnchors.map((a,i)=>`<div class="anchor-row" data-anchor="${i}"><div class="field"><label>${targetSeconds?'Pupil timestamp · ns':'Source · ns'}</label><input class="anchor-source" inputmode="numeric" value="${esc(String(a.source_ns??'0'))}"></div><div class="field"><label>${targetSeconds?'Vicon/C3D time · seconds':'Target · ns'}</label><input class="${targetSeconds?'anchor-target-seconds':'anchor-target'}" inputmode="decimal" value="${esc(targetSeconds?nsTextToSeconds(a.target_ns??'0'):String(a.target_ns??'0'))}"></div><button class="row-button anchor-remove" data-index="${i}">×</button></div>`).join('');
}

function clipsProperties() {
  nativeDraft().mode='native';
  return nativeTrimProperties();
}


function activeWorkflowContext() {
  const ws=appState.data?.interaction_workspace||{};
  return {
    interactionId:String(ws.selected_interaction_id||'').trim().toUpperCase(),
    roleId:String(ws.selected_role_id||'').trim().toUpperCase(),
  };
}

function gazeArtifactRole(row) {
  return String(row?.qc?.wearer_role_id||row?.parameters?.wearer_role_id||codeFromText(contextRecordText(row),'R')||'').trim().toUpperCase();
}

function gazePreviewSelection() {
  const ctx=activeWorkflowContext();
  const all=artifacts(['gaze_rays']).slice().sort(byCreated);
  let rows=all.filter(row=>!ctx.interactionId||contextRecordText(row).includes(ctx.interactionId.toLowerCase()));
  if(!rows.length)rows=all;
  const byRole={R01:rows.filter(row=>gazeArtifactRole(row)==='R01'),R02:rows.filter(row=>gazeArtifactRole(row)==='R02')};
  const pick=role=>{
    const selected=appState.gazePreviewArtifactIds?.[role]||'';
    const found=byRole[role].find(row=>row.id===selected);
    return found||byRole[role].slice(-1)[0]||null;
  };
  const r01=pick('R01'),r02=pick('R02');
  const both=Boolean(r01&&r02);
  let mode=appState.gazePreviewMode||(both?'dyadic':'single');
  if(mode==='dyadic'&&!both)mode='single';
  const currentRole=calibrationContextRole();
  const single=(currentRole==='R02'?r02:r01)||rows.slice(-1)[0]||null;
  const peer=single?.id===r01?.id?r02:r01;
  return {rows,byRole,r01,r02,both,mode,currentRole,single,peer};
}

function contextRecordText(record) {
  const parts=[record?.label,record?.kind,record?.parameters?.interaction_id,record?.parameters?.role_id,record?.parameters?.wearer_role_id];
  for(const id of record?.source_asset_ids||[]){
    const source=session()?.assets?.[id];
    if(source){
      const ai=assetIdentity(source);
      parts.push(source.label,source.source_path,ai.interaction_id,ai.role_id,ai.participant_id);
    }
  }
  if(record?.id && session()?.assets?.[record.id]){
    const source=session().assets[record.id], ai=assetIdentity(source);
    parts.push(source.label,source.source_path,ai.interaction_id,ai.role_id,ai.participant_id);
  }
  return parts.filter(Boolean).join(' ').toLowerCase();
}

function nativeTrimIdForRecord(record,visited=new Set()) {
  if(!record)return '';
  const recordId=String(record.id||'');
  if(recordId){if(visited.has(recordId))return '';visited.add(recordId);}
  if(record.kind==='native_reference_trim')return recordId;
  const direct=record?.provenance?.native_trim_artifact_id||record?.parameters?.native_trim_artifact_id||record?.parameters?.trim_artifact_id||record?.qc?.native_trim_artifact_id||'';
  if(direct)return String(direct);
  const s=session();
  for(const id of record?.parent_artifact_ids||[]){
    const parent=s?.artifacts?.[id];
    if(parent?.kind==='native_reference_trim')return String(id);
    const nested=nativeTrimIdForRecord(parent,visited);
    if(nested)return nested;
  }
  return '';
}

function wearerRoleForRecord(record) {
  return String(record?.parameters?.wearer_role_id||record?.provenance?.wearer_role_id||record?.qc?.wearer_role_id||codeFromText(contextRecordText(record),'R')||'').trim().toUpperCase();
}

function recordUsesNativeTrim(record, trimId) {
  return Boolean(record&&trimId&&nativeTrimIdForRecord(record)===String(trimId));
}

function targetTrimSelectionKey() {
  const ctx=activeWorkflowContext();
  return [selectedSessionId()||'session',ctx.interactionId||'interaction',ctx.roleId||'role'].join('::');
}

function selectedTargetTrimId(rows=[]) {
  const remembered=appState.targetTrimSelections?.[targetTrimSelectionKey()]||'';
  return rows.some(row=>row.id===remembered)?remembered:(rows.slice(-1)[0]?.id||'');
}

function rememberTargetTrimSelection(trimId) {
  appState.targetTrimSelections=appState.targetTrimSelections||{};
  appState.targetTrimSelections[targetTrimSelectionKey()]=String(trimId||'');
  renderProperties();
}

function rowsForActiveWorkflowContext(rows,{role=true,allowUnscoped=false,fallback=true}={}) {
  const ctx=activeWorkflowContext();
  if(!ctx.interactionId&&!ctx.roleId)return rows||[];
  const filtered=(rows||[]).filter(row=>{
    const text=contextRecordText(row);
    const hasInteraction=/(?:^|[^a-z0-9])i\d+(?![a-z0-9])/i.test(text);
    const hasRole=/(?:^|[^a-z0-9])r\d+(?![a-z0-9])/i.test(text);
    if(ctx.interactionId&&hasInteraction&&!text.includes(ctx.interactionId.toLowerCase()))return false;
    if(role&&ctx.roleId&&hasRole&&!text.includes(ctx.roleId.toLowerCase()))return false;
    if(!allowUnscoped&&ctx.interactionId&&!hasInteraction)return false;
    return true;
  });
  return filtered.length||!fallback?filtered:(rows||[]);
}

function targetRowsForActiveContext(rows,{trimId='',requireTrim=false}={}) {
  const ctx=activeWorkflowContext();
  return rowsForActiveWorkflowContext(rows,{fallback:false}).filter(row=>{
    const resolvedRole=wearerRoleForRecord(row);
    if(ctx.roleId&&resolvedRole!==ctx.roleId)return false;
    if(requireTrim&&trimId&&nativeTrimIdForRecord(row)!==String(trimId))return false;
    return true;
  });
}

function assignmentSummaryHasProfiles(artifact) {
  const summary=artifact?.qc?.assignments_summary||{};
  const rows=Object.values(summary);
  return Boolean(rows.length && rows.every(row=>Boolean(row?.recommended_calibration_artifact_id)));
}

function isCalibrationHandoffArtifact(artifact) {
  return Boolean(
    artifact?.kind==='etm_role_assignment' &&
    (artifact?.parameters?.calibration_phase==='handoff' ||
     artifact?.parameters?.binding_source_assignment_artifact_id ||
     artifact?.qc?.handoff_complete ||
     assignmentSummaryHasProfiles(artifact))
  );
}

function calibrationHandoffArtifacts() {
  return artifacts(['etm_role_assignment']).filter(row=>row.qc?.status==='resolved'&&isCalibrationHandoffArtifact(row)).sort(byCreated);
}

function calibrationTimeseriesForRole(roleId) {
  const role=String(roleId||'').toUpperCase();
  const ctx=activeWorkflowContext();
  const rows=assets(['pupil_cloud']).slice().sort(byCreated);
  const interactionRows=rows.filter(asset=>{
    const identity=assetIdentity(asset);
    return !ctx.interactionId || !identity.interaction_id || identity.interaction_id===ctx.interactionId;
  });
  const exact=interactionRows.filter(asset=>{
    const identity=assetIdentity(asset);
    const inferred=identity.role_id||codeFromText(`${asset.label||''} ${asset.source_path||''}`,'R');
    return inferred===role;
  });
  return exact.length?exact:interactionRows.length===1?interactionRows:rows.filter(asset=>{
    const identity=assetIdentity(asset);
    return identity.role_id===role||codeFromText(`${asset.label||''} ${asset.source_path||''}`,'R')===role;
  });
}

function calibrationTimeSyncArtifactRole(artifact,wearerAssignment=null) {
  let role=String(artifact?.parameters?.wearer_role_id||artifact?.qc?.wearer_role_id||'').toUpperCase();
  if(role)return role;
  const target=String(artifact?.parameters?.target_asset_id||'');
  const summary=wearerAssignment?.qc?.assignments_summary||{};
  for(const [candidate,row] of Object.entries(summary))if(String(row?.etm_asset_id||'')===target)return String(candidate).toUpperCase();
  return '';
}

function calibrationTimeSyncProductionReady(artifact) {
  if(!artifact)return false;
  const qc=artifact.qc||{},parameters=artifact.parameters||{};
  const noVideo=parameters.ego_video_used!==true&&qc.ego_video_used!==true;
  const verified=qc.sync_verified===true || (String(qc.confidence||'').toUpperCase()==='HIGH') || (String(qc.confidence||'').toUpperCase()==='REVIEW'&&Boolean(qc.accepted_by_user||parameters.accepted_by_user));
  const distributed=qc.distributed_drift_validated===true || (Number(qc.matched_window_count||0)>=2&&Number(qc.anchor_count||0)>=2);
  const precision=qc.precision_validated===true && Number(qc.holdout_window_count||0)>=3 && Number.isFinite(Number(qc.drift_ci95_ppm)) && qc.residual_validation_mode==='leave_one_window_out';
  const mapId=String(parameters.time_map_id||qc.time_map_id||'');
  return Boolean(noVideo&&verified&&distributed&&precision&&mapId&&session()?.time_maps?.[mapId]);
}

function calibrationAcceptedTimeSyncByRole(wearerAssignment=null) {
  const result={R01:null,R02:null};
  const summary=wearerAssignment?.qc?.assignments_summary||{};
  for(const artifact of artifacts(['time_sync']).slice().sort(byCreated)){
    const role=calibrationTimeSyncArtifactRole(artifact,wearerAssignment);
    if(!Object.prototype.hasOwnProperty.call(result,role))continue;
    const declared=String(artifact?.parameters?.assignment_artifact_id||'');
    if(wearerAssignment&&declared&&declared!==wearerAssignment.id)continue;
    const expectedTarget=String(summary?.[role]?.etm_asset_id||'');
    const actualTarget=String(artifact?.parameters?.target_asset_id||'');
    if(expectedTarget&&actualTarget&&expectedTarget!==actualTarget)continue;
    const current=result[role];
    const candidateReady=calibrationTimeSyncProductionReady(artifact);
    const currentReady=calibrationTimeSyncProductionReady(current);
    if(candidateReady||!current||!currentReady)result[role]=artifact;
  }
  return result;
}

function calibrationTimeSyncMetricsMarkup(artifact) {
  const qc=artifact?.qc||{};
  return `<div class="calibration-sync-metrics">
    <div><span>Relative start</span><strong>${autoSyncMetric(qc.estimated_offset_sec,6,' s')}</strong></div>
    <div><span>Drift · 95%</span><strong>${esc(autoSyncDriftText(qc))}</strong></div>
    <div><span>Correlation</span><strong>${autoSyncMetric(qc.aligned_correlation,4)}</strong></div>
    <div><span>Windows / holdout</span><strong>${esc(qc.matched_window_count??0)} / ${esc(qc.holdout_window_count??0)}</strong></div>
    <div><span>CV P95</span><strong>${esc(autoSyncResidualText(qc))}</strong></div>
    <div><span>Video input</span><strong>${qc.ego_video_used===true?'Unexpected':'None'}</strong></div>
  </div>`;
}

function calibrationSyncSelection(roleId,rows) {
  const role=String(roleId||'').toUpperCase();
  const selected=String(appState.calibrationSyncSelections?.[role]||'');
  if(selected&&rows.some(row=>row.id===selected))return selected;
  const fallback=rows.slice(-1)[0]?.id||'';
  appState.calibrationSyncSelections[role]=fallback;
  return fallback;
}

function calibrationSyncPhaseMarkup(stage) {
  const assignment=stage.wearerAssignment;
  const running=String(appState.calibrationAutoSyncRunningRole||'');
  const cards=['R01','R02'].map(role=>{
    const row=stage.rows.find(item=>item.role===role)||{};
    const timeseries=calibrationTimeseriesForRole(role);
    const selected=calibrationSyncSelection(role,timeseries);
    const sync=row.timeSyncArtifact||null;
    const candidateResult=appState.calibrationAutoSyncCandidates?.[role]||null;
    const candidate=candidateResult?.candidate||null;
    const assignmentText=row.wearerMapping?`${row.wearerMapping.etm_id||'ETM'} · ${row.wearerMapping.etm_asset_label||row.wearerMapping.etm_asset_id||''}`:'Resolve wearer first';
    const status=row.timeSyncReady?'Verified':candidateResult?.status==='review_required'?'Review candidate':candidateResult?.status==='failed'?'Failed':sync?'Not production-ready':'Not run';
    const statusClass=row.timeSyncReady?'good':candidateResult?.status==='failed'?'bad':'warning';
    const candidateBlock=candidate?`<div class="calibration-sync-candidate ${candidateResult.status==='failed'?'failed':''}">${autoSyncSummary(candidate)}${(candidate.reasons||[]).length?`<div class="form-help">${candidate.reasons.map(esc).join('<br>')}</div>`:''}<button class="soft-button full cal-sync-preview" data-role="${role}" data-candidate-id="${esc(candidate.candidate_id||'')}">Preview independent evidence</button>${candidateResult.status==='review_required'?`<button class="soft-button full cal-sync-accept" data-role="${role}">Accept reviewed mapping</button>`:''}</div>`:'';
    return `<article class="calibration-sync-role-card ${row.timeSyncReady?'ready':'pending'}">
      <div class="calibration-sync-role-head"><div><span class="eyebrow">${role} Pupil clock → assigned ETM clock</span><strong>${esc(assignmentText)}</strong></div><span class="status-pill ${statusClass}">${esc(status)}</span></div>
      <div class="field"><label>Pupil Timeseries with imu.csv</label><select class="cal-sync-timeseries" data-role="${role}" id="cal-sync-timeseries-${role.toLowerCase()}">${optionRows(timeseries,selected,`Select ${role} Timeseries`)}</select></div>
      ${sync?`<div class="${row.timeSyncReady?'success-card':'warning-card'} compact"><strong>${esc(sync.label)}</strong>${calibrationTimeSyncMetricsMarkup(sync)}<button class="soft-button full cal-sync-preview" data-role="${role}" data-artifact-id="${esc(sync.id||'')}">Preview synchronization evidence</button>${row.timeSyncReady?'':'<p>This accepted artifact lacks the independent precision audit required by the current synchronization contract. Re-run synchronization.</p>'}</div>`:'<div class="form-help">No accepted affine TimeGraph mapping for this role yet.</div>'}
      ${candidateBlock}
      <button class="soft-button full cal-sync-run" data-role="${role}" ${!assignment||!selected||running?'disabled':''}>${running===role?'Comparing IMU ↔ ETM motion…':row.timeSyncReady?'Re-run automatic synchronization':'Run automatic synchronization'}</button>
    </article>`;
  }).join('');
  const settings=appState.calibrationSyncSettings||{};
  return `<div class="calibration-sync-intro"><strong>No ego video is used.</strong><p>Pupil <code>imu.csv</code> angular motion is compared with the assigned ETM rigid-body motion from Vicon. Distributed windows estimate both offset and clock drift. The precision audit uses sub-sample local lags, leave-one-window-out validation and a 95% drift interval.</p></div>
    <div class="calibration-sync-role-grid">${cards}</div>
    <details class="calibration-sync-settings"><summary>Automatic synchronization settings</summary><div class="property-stack calibration-route-body">
      <div class="form-row"><div class="field"><label>Maximum offset search (s)</label><input id="cal-sync-max-offset" type="number" min="1" max="3600" step="1" value="${esc(settings.maximumOffsetSec??120)}"></div><div class="field"><label>Motion comparison rate (Hz)</label><input id="cal-sync-sample-rate" type="number" min="20" max="240" step="1" value="${esc(settings.sampleRateHz??120)}"></div></div>
      <div class="toggle-row"><span>Automatically save HIGH-confidence mappings</span><button class="toggle ${settings.saveIfHigh===false?'':'on'}" id="cal-sync-save-high" data-on="${String(settings.saveIfHigh!==false)}"></button></div>
      <div class="form-help">REVIEW candidates are never silently promoted. A displayed zero is never treated as exact: the card reports estimator resolution and drift confidence. FAILED candidates do not alter the TimeGraph.</div>
    </div></details>
    <button class="primary-button full" id="run-calibration-sync-both" ${!assignment||running?'disabled':''}>${running==='BOTH'?'Synchronizing both roles…':'Run R01 + R02 synchronization'}</button>`;
}

function calibrationClipAssetRows(kind,role){
  const wanted=String(role||'').toUpperCase();
  const ctx=activeWorkflowContext();
  const all=assets([kind]).slice().sort(byCreated);
  let scoped=all.filter(asset=>{
    const identity=assetIdentity(asset);
    if(ctx.interactionId&&identity.interaction_id&&identity.interaction_id!==ctx.interactionId)return false;
    return true;
  });
  const exact=scoped.filter(asset=>assetIdentity(asset).role_id===wanted);
  if(exact.length)return exact;
  const compatible=scoped.filter(asset=>!assetIdentity(asset).role_id||contextRecordText(asset).includes(wanted.toLowerCase()));
  return compatible.length?compatible:scoped;
}

function calibrationClipInputId(role,field,rows,preferred=''){
  const wanted=String(role||'').toUpperCase();
  const state=appState.calibrationClipSelections?.[wanted]||{};
  const stored=String(state[field]||'');
  if(rows.some(row=>row.id===stored))return stored;
  if(preferred&&rows.some(row=>row.id===preferred))return preferred;
  return rows.slice(-1)[0]?.id||'';
}

function calibrationAlignedClipSets(handoff){
  if(!handoff)return [];
  return rowsForActiveWorkflowContext(artifacts(['aligned_clip_set']),{role:false,allowUnscoped:true})
    .filter(row=>String(row.parameters?.calibration_handoff_artifact_id||'')===String(handoff.id||''))
    .slice().sort(byCreated);
}

function calibrationWorkflowState(roleId=calibrationContextRole()) {
  const profiles=calibrationProfilesForActiveSet();
  const allAssignments=artifacts(['etm_role_assignment']).filter(row=>row.qc?.status==='resolved'||isCalibrationHandoffArtifact(row)).slice().sort(byCreated);
  const handoffs=allAssignments.filter(isCalibrationHandoffArtifact);
  const latest=allAssignments.slice(-1)[0]||null;
  let wearerAssignment=latest;
  if(latest&&isCalibrationHandoffArtifact(latest)){
    const sourceId=latest.parameters?.binding_source_assignment_artifact_id||latest.qc?.binding_source_assignment_artifact_id||'';
    wearerAssignment=allAssignments.find(row=>row.id===sourceId)||latest;
  }
  const compatibleHandoffs=wearerAssignment?handoffs.filter(row=>{
    const sourceId=row.parameters?.binding_source_assignment_artifact_id||row.qc?.binding_source_assignment_artifact_id||'';
    return row.id===wearerAssignment.id || sourceId===wearerAssignment.id;
  }):[];
  const handoff=compatibleHandoffs.slice(-1)[0]||(wearerAssignment&&isCalibrationHandoffArtifact(wearerAssignment)?wearerAssignment:null);
  const assignment=handoff;
  const acceptances=artifacts(['calibration_acceptance']).slice().sort(byCreated);
  const wearerSummary=wearerAssignment?.qc?.assignments_summary||{};
  const handoffSummary=handoff?.qc?.assignments_summary||{};
  const acceptedSyncByRole=calibrationAcceptedTimeSyncByRole(wearerAssignment);
  const roles=['R01','R02'];
  const rows=roles.map(role=>{
    const wearerMapping=wearerSummary[role]||null;
    const boundMapping=handoffSummary[role]||null;
    const mapping=boundMapping||wearerMapping;
    const profileId=boundMapping?.recommended_calibration_artifact_id||'';
    const profile=profiles.find(item=>item.id===profileId)||null;
    const source=String(profile?.parameters?.calibration_source||profile?.qc?.calibration_source||'');
    const transferred=source==='transferred_hardware_profile';
    const acceptance=acceptances.filter(item=>{
      const acceptanceRole=String(item.parameters?.wearer_role_id||item.qc?.wearer_role_id||'').toUpperCase();
      const acceptanceProfile=String(item.provenance?.calibration_profile_artifact_id||item.qc?.transferred_profile_artifact_id||'');
      return acceptanceRole===role && (!profileId||!acceptanceProfile||acceptanceProfile===profileId);
    }).slice(-1)[0]||null;
    const assignmentReady=Boolean(wearerMapping && !['LOW','AMBIGUOUS'].includes(String(wearerMapping.confidence||'').toUpperCase()));
    const profileReady=Boolean(profile&&boundMapping);
    const acceptanceReady=!transferred || Boolean(acceptance?.qc?.production_ready);
    const boundSyncId=String(boundMapping?.recommended_time_sync_artifact_id||'');
    const timeSyncArtifact=(boundSyncId?artifacts(['time_sync']).find(item=>item.id===boundSyncId):null)||acceptedSyncByRole[role]||null;
    const timeSyncReady=calibrationTimeSyncProductionReady(timeSyncArtifact);
    const handoffSyncReady=Boolean(boundMapping&&boundSyncId&&timeSyncReady&&boundMapping?.time_sync_production_ready!==false);
    return {role,mapping,wearerMapping,boundMapping,profile,profileId,source,transferred,acceptance,assignmentReady,profileReady,acceptanceReady,timeSyncArtifact,timeSyncReady,handoffSyncReady,ready:assignmentReady&&profileReady&&acceptanceReady&&handoffSyncReady};
  });
  const alignedClipSets=calibrationAlignedClipSets(handoff);
  const alignedClipSet=alignedClipSets.slice(-1)[0]||null;
  const roleTrimIds=alignedClipSet?.parameters?.role_trim_artifact_ids||{};
  const alignedRoleTrims=Object.fromEntries(['R01','R02'].map(role=>[
    role,
    roleTrimIds[role]?session()?.artifacts?.[roleTrimIds[role]]||null:null,
  ]));
  const alignedClipsReady=Boolean(
    alignedClipSet?.qc?.all_roles_created && alignedRoleTrims.R01 && alignedRoleTrims.R02
  );
  const wanted=String(roleId||'R01').toUpperCase();
  const current=rows.find(row=>row.role===wanted)||rows[0];
  return {
    profiles,allAssignments,wearerAssignments:allAssignments.filter(row=>!isCalibrationHandoffArtifact(row)),handoffs,
    wearerAssignment,handoff,assignment,assignments:handoff?[handoff]:[],acceptances,rows,current,
    timeSyncByRole:Object.fromEntries(rows.map(row=>[row.role,row.timeSyncArtifact])),
    alignedClipSets,alignedClipSet,alignedRoleTrims,alignedClipsReady,
    assignmentReady:Boolean(wearerAssignment)&&rows.every(row=>row.assignmentReady),
    synchronizationReady:Boolean(wearerAssignment)&&rows.every(row=>row.timeSyncReady),
    handoffReady:Boolean(handoff)&&Boolean(handoff?.qc?.handoff_complete)&&rows.every(row=>row.handoffSyncReady),
    allRolesReady:Boolean(handoff)&&Boolean(handoff?.qc?.handoff_complete)&&rows.every(row=>row.ready),
    currentReady:Boolean(current?.ready),
  };
}

function calibrationReadinessRows(state) {
  return state.rows.map(row=>{
    const etm=row.wearerMapping?.etm_id||'Unresolved';
    const profile=row.profile?.label||'Profile not bound';
    const confidence=row.wearerMapping?.confidence||'—';
    const sync=row.timeSyncArtifact;
    const syncText=sync?`${autoSyncMetric(sync.qc?.estimated_offset_sec,4,' s')} · ${autoSyncMetric(sync.qc?.estimated_drift_ppm,1,' ppm')}`:'Time sync missing';
    const status=row.ready?'Ready':row.transferred&&!row.acceptanceReady?'Acceptance required':!row.assignmentReady?'Resolve wearer':!row.timeSyncReady?'Synchronize clocks':!row.profileReady?'Bind profile':!row.handoffSyncReady?'Create new handoff':'Profile required';
    return `<div class="calibration-readiness-role ${row.ready?'ready':'blocked'}"><div><strong>${esc(row.role)} → ${esc(etm)}</strong><small>${esc(profile)}</small><small>Wearer ${esc(confidence)} · Sync ${esc(syncText)}${row.transferred?' · transferred profile':''}</small></div><span class="status-pill ${row.ready?'good':'warning'}">${esc(status)}</span></div>`;
  }).join('');
}

function isLegacyPrivacyArtifact(artifact) {
  return Boolean(artifact?.parameters?.native_trim_artifact_id && !artifact?.parameters?.frame_coverage_engine);
}

function privacyArtifactCoversTrim(artifact, trimId) {
  if (!artifact || isLegacyPrivacyArtifact(artifact)) return false;
  const scopedTrim=artifact.parameters?.native_trim_artifact_id||'';
  return !trimId || !scopedTrim || scopedTrim===trimId;
}

function privacyArtifactLabel(artifact) {
  if (isLegacyPrivacyArtifact(artifact)) return `${artifact.label} · OLD boundary coverage — recreate`;
  const engine=artifact.parameters?.frame_coverage_engine;
  return `${artifact.label}${engine?' · compatible':''}`;
}

function privacyArtifactOptionRows(rows, selected='') {
  const current=rows.filter(row=>!isLegacyPrivacyArtifact(row));
  const legacy=rows.filter(isLegacyPrivacyArtifact);
  let html=`<option value="">Select reusable full-recording Privacy source</option>`;
  html+=current.map(row=>`<option value="${esc(row.id)}" ${row.id===selected?'selected':''}>${esc(privacyArtifactLabel(row))}</option>`).join('');
  if(legacy.length){
    html+=`<optgroup label="Older artifacts — recreate before export">${legacy.map(row=>`<option value="${esc(row.id)}" disabled>${esc(privacyArtifactLabel(row))}</option>`).join('')}</optgroup>`;
  }
  return html;
}

function isReusablePrivacySource(artifact) {
  if (!artifact || isLegacyPrivacyArtifact(artifact)) return false;
  const trimId=artifact.parameters?.native_trim_artifact_id||'';
  const intervalSource=artifact.parameters?.interval_source||'';
  return !trimId && (!intervalSource || intervalSource==='full_recording');
}

function privacyProperties() {
  const native=rowsForActiveWorkflowContext(assets(['pupil_native']));
  const face=rowsForActiveWorkflowContext(assets(['face_mapper']));
  const trims=rowsForActiveWorkflowContext(artifacts(['native_reference_trim'])).slice().sort((a,b)=>String(b.created_at||'').localeCompare(String(a.created_at||'')));
  const filters=rowsForActiveWorkflowContext(artifacts(['face_mapper_filter'])).slice().sort((a,b)=>String(b.created_at||'').localeCompare(String(a.created_at||'')));
  const finalClips=rowsForActiveWorkflowContext(artifacts(['privacy_clip'])).slice().sort((a,b)=>String(b.created_at||'').localeCompare(String(a.created_at||'')));
  const coverageArtifacts=rowsForActiveWorkflowContext(artifacts(['privacy_video'])).slice().sort((a,b)=>String(b.created_at||'').localeCompare(String(a.created_at||'')));
  const directClip=finalClips.find(row=>row.parameters?.synchronized_privacy_engine)||finalClips[0]||null;
  const newestFilter=filters[0]||null;
  const basis=directClip||coverageArtifacts[0]||null;
  const basisParams=basis?.parameters||{};
  const basisSources=new Set(basis?.source_asset_ids||[]);
  const defaultFilter=basisParams.facemapper_filter_artifact_id||newestFilter?.id||'';
  const selectedFilter=filters.find(row=>row.id===defaultFilter)||null;
  const defaultTrim=basisParams.native_trim_artifact_id||selectedFilter?.parameters?.native_trim_artifact_id||trims[0]?.id||'';
  const defaultNative=native.find(row=>basisSources.has(row.id))?.id||native.find(row=>selectedFilter?.source_asset_ids?.includes(row.id))?.id||native[0]?.id||'';
  const defaultFace=face.find(row=>basisSources.has(row.id))?.id||face[0]?.id||'';
  const defaultMode=basisParams.mode||'gaussian';
  const defaultDetector=basisParams.detector_mode||'facemapper';
  const defaultCandidatePolicy=basisParams.exact_filter_match?'accepted_only':(basisParams.candidate_policy||'accepted_repair');
  const defaultPolicy=basisParams.coverage_policy||'strict';
  const defaultAudio=basisParams.preserve_audio!==false;
  const defaultFps=Number(basisParams.output_fps||basis?.qc?.output_fps||30);
  const trimForLabel=trims.find(row=>row.id===defaultTrim)||null;
  const defaultLabel=`${trimForLabel?.label||'privacy'}_privacy_${defaultFps}fps`;
  const reusableSources=coverageArtifacts.filter(isReusablePrivacySource);
  const preferredSource=reusableSources[0]||null;
  const advancedLabel=`${trimForLabel?.label||'privacy'}_privacy_derived_${defaultFps}fps`;
  const oldTrimScoped=coverageArtifacts.filter(row=>row.parameters?.native_trim_artifact_id);
  const filterReady=Boolean(selectedFilter);
  const detectorNames={facemapper:'Filtered FaceMapper only',hybrid:'Filtered FaceMapper + Haar supplement',haar:'Haar detector only'};
  const candidateNames={accepted_repair:'Accepted + short-gap repair',accepted_only:'Accepted only',accepted_review:'Accepted + review'};
  const maskNames={gaussian:'Gaussian blur',pixelate:'Pixelation',solid:'Solid mask'};
  const policyNames={strict:'Strict coverage',allow:'Allow + report gaps'};

  return `<section class="property-section privacy-step-section"><details class="privacy-step privacy-filter-step ${filterReady?'is-complete':''}" ${filterReady?'':'open'}>
    <summary><span>1 · FaceMapper QC & Filter</span><small>${filterReady?`Ready · ${esc(selectedFilter.label)}`:'Required before Privacy'}</small></summary>
    <div class="property-stack privacy-step-body">
      <div class="form-help"><strong>Run this after Trim and before Privacy.</strong> The original FaceMapper package remains unchanged. OculoStudio creates a Privacy-safe package (accepted + review candidates) and an Analysis-strict package (accepted only).</div>
      <div class="field"><label>Native-reference trim</label><select id="fm-filter-trim">${optionRows(trims,defaultTrim,'Select retained clip')}</select></div>
      <div class="field"><label>Output label</label><input id="fm-filter-label" value="${esc(trimForLabel?.label?trimForLabel.label+'_facemapper_filter':'Filtered Face Mapper')}"></div>
      <div class="field"><label>FaceMapper source bilateral convention</label><select id="fm-filter-bilateral"><option value="pupil_image_side" selected>Pupil FaceMapper image-side labels → subject anatomy · recommended</option><option value="subject_anatomical">Source labels are already subject-anatomical</option><option value="auto_audit">Auto-audit x-order, then normalise names</option></select><small>Coordinates are always preserved exactly. This setting only decides whether eye/mouth left-right names are exchanged.</small></div>
      <div class="form-help"><strong>Live QC is the default.</strong> OculoStudio plays the unchanged Native scene video and draws source, canonical or filtered overlays in a browser Canvas using the Native PTS clock. No preview video encode is required.</div>
      <details><summary>Optional · export legacy encoded QC MP4s</summary><div class="property-stack" style="margin-top:8px">
        <div class="toggle-row"><span>Render encoded accepted / review / rejected QC MP4</span><button class="toggle" id="fm-filter-preview" data-on="false"></button></div>
        <div class="toggle-row"><span>Also encode original and canonical landmark MP4s</span><button class="toggle" id="fm-filter-source-preview" data-on="false"></button></div>
        <div class="form-help">Use this only when a shareable standalone QC movie is required. Normal review uses the original video plus live Canvas overlays and avoids frame resampling, colour conversion and aspect-ratio changes.</div>
      </div></details>
      <div class="field"><label>Encoded QC MP4 scope · optional</label><select id="fm-filter-preview-scope"><option value="sampled">Sampled checkpoints · faster</option><option value="full">Full retained interval · exact full-length review</option></select></div>
      <div class="form-row" id="fm-filter-sampled-settings"><div class="field"><label>Preview samples</label><input id="fm-filter-preview-samples" type="number" value="9" min="3" max="15"></div><div class="field"><label>Seconds per sample</label><input id="fm-filter-preview-seconds" type="number" value="6" min="2" max="30" step="1"></div></div>
      <div class="form-help">Full preview renders the entire retained interval with green accepted, amber review, and red rejected boxes. It may take several minutes for a 9-minute recording.</div>
      <details><summary>Advanced false-positive gates</summary><div class="property-stack" style="margin-top:8px">
        <div class="form-row"><div class="field"><label>Large-box area ratio</label><input id="fm-filter-large-area" type="number" value="0.075" min="0.01" max="0.5" step="0.005"></div><div class="field"><label>Maximum face area ratio</label><input id="fm-filter-max-area" type="number" value="0.45" min="0.05" max="0.95" step="0.01"></div></div>
        <div class="form-row"><div class="field"><label>Minimum landmark span X / box</label><input id="fm-filter-span-x" type="number" value="0.16" min="0" max="1" step="0.01"></div><div class="field"><label>Minimum landmark span Y / box</label><input id="fm-filter-span-y" type="number" value="0.22" min="0" max="1" step="0.01"></div></div>
        <div class="form-row"><div class="field"><label>Minimum interocular / box width</label><input id="fm-filter-eye-span" type="number" value="0.10" min="0" max="1" step="0.01"></div><div class="field"><label>Minimum landmarks inside box</label><input id="fm-filter-inside" type="number" value="0.80" min="0" max="1" step="0.05"></div></div>
        <div class="form-row"><div class="field"><label>Maximum landmark-centre offset</label><input id="fm-filter-centre" type="number" value="0.36" min="0" max="1" step="0.01"></div><div class="field"><label>Maximum temporal scale jump</label><input id="fm-filter-scale-jump" type="number" value="3.5" min="1.1" max="20" step="0.1"></div></div>
        <div class="field"><label>Minimum track frames</label><input id="fm-filter-track-frames" type="number" value="3" min="1" max="60"></div>
      </div></details>
      <button class="primary-button full" id="run-facemapper-filter" ${!trims.length?'disabled':''}>Filter FaceMapper candidates before Privacy</button>
    </div>
  </details></section>

  <section class="property-section privacy-primary-section"><div class="section-label">2 · Create final synchronized Privacy video</div><div class="property-stack">
    <div class="form-help privacy-final-help"><strong>One trim, one processing job, one final video.</strong> Fixed FPS, face masks, audio and the Native frame map are created together without an intermediate video.</div>
    ${directClip?`<div class="warning-card compact privacy-current-video"><strong>Current generated Privacy video</strong><p>${esc(directClip.label)}</p><button class="danger-button full" id="remove-current-privacy" data-artifact-id="${esc(directClip.id)}">Move current Privacy video to Project Trash</button></div>`:''}
    <div class="field"><label>Native recording</label><select id="privacy-native">${optionRows(native,defaultNative,'Select Native')}</select></div>
    <div class="field"><label>Filtered FaceMapper</label><select id="privacy-filter">${optionRows(filters,defaultFilter,'No filtered package')}</select></div>
    <div class="field"><label>Mask source</label><select id="privacy-candidate-policy"><option value="accepted_repair" ${defaultCandidatePolicy==='accepted_repair'?'selected':''}>Accepted + short-gap repair · recommended</option><option value="accepted_only" ${defaultCandidatePolicy==='accepted_only'?'selected':''}>Accepted only · exact green boxes</option><option value="accepted_review" ${defaultCandidatePolicy==='accepted_review'?'selected':''}>Accepted + review · conservative privacy</option></select></div>
    <div class="form-help" id="privacy-mask-policy-help"><strong>Accepted + short-gap repair</strong> never uses amber review candidates. It follows green accepted tracks and repairs only brief, motion-plausible drop-outs.</div>
    <details class="privacy-compact-options accepted-repair-options" id="privacy-repair-options" open><summary><span>Accepted-only temporal repair</span><small>Green seeds only · no yellow boxes</small></summary><div class="property-stack privacy-options-body">
      <div class="form-row"><div class="field"><label>Internal gap limit (frames)</label><input id="privacy-gap" type="number" value="${Number(basisParams.gap_fill_frames??3)}" min="0" max="10"></div><div class="field"><label>Track-edge repair (frames)</label><input id="privacy-edge-gap" type="number" value="${Number(basisParams.edge_fill_frames??1)}" min="0" max="3"></div></div>
      <div class="form-row"><div class="field"><label>Maximum normalized motion / frame</label><input id="privacy-repair-motion" type="number" value="${Number(basisParams.repair_maximum_center_shift_per_frame??0.85)}" min="0.05" max="3" step="0.05"></div><div class="field"><label>Maximum scale ratio</label><input id="privacy-repair-scale" type="number" value="${Number(basisParams.repair_maximum_scale_ratio??1.75)}" min="1" max="5" step="0.05"></div></div>
      <div class="field"><label>Minimum accepted observations for edge repair</label><input id="privacy-repair-min-track" type="number" value="${Number(basisParams.repair_minimum_track_observations??3)}" min="2" max="20"></div>
      <div class="form-row"><div class="field"><label>Mask preview scope</label><select id="privacy-repair-preview-scope"><option value="full">Full retained interval · required for final export</option><option value="sampled">Representative checkpoints · tuning only</option></select></div><div class="field"><label>Preview FPS</label><select id="privacy-repair-preview-fps"><option value="30" selected>30 · exact frame review</option><option value="15">15 · tuning only</option></select></div></div>
      <button class="soft-button full" id="run-privacy-mask-preview" ${!native.length||!trims.length||!filters.length?'disabled':''}>Render accepted + repair mask preview</button>
      <div class="form-help">Preview and final Privacy use the same accepted-only repair engine. Green boxes are accepted detections; cyan boxes fill internal gaps; orange boxes cover only the immediately adjacent track edge. Final export requires a full preview at the same FPS as the final video.</div>
    </div></details>
    <div class="field"><label>Native-reference trim · required</label><select id="privacy-native-trim-direct">${optionRows(trims,defaultTrim,'Select the retained R01/R02 interval')}</select></div>
    <div class="form-row"><div class="field"><label>Final output FPS</label><select id="privacy-output-fps">${[24,25,30,50,60].map(value=>`<option value="${value}" ${value===defaultFps?'selected':''}>${value}</option>`).join('')}</select></div><div class="field"><label>Final output label</label><input id="privacy-output-label" value="${esc(defaultLabel)}"></div></div>
    <button class="primary-button full privacy-final-action" id="run-privacy-final" ${!native.length||!trims.length?'disabled':''}>Create synchronized Privacy video</button>

    <details class="privacy-compact-options"><summary><span>Mask & verification settings</span><small>${esc(maskNames[defaultMode]||defaultMode)} · ${esc(candidateNames[defaultCandidatePolicy]||defaultCandidatePolicy)} · ${esc(detectorNames[defaultDetector]||defaultDetector)}</small></summary><div class="property-stack privacy-options-body">
      <div class="field"><label>Raw FaceMapper fallback · only used without a filtered artifact</label><select id="privacy-face">${optionRows(face,defaultFace,'Use detector only')}</select></div>
      <div class="field"><label>Detection source</label><select id="privacy-detector"><option value="facemapper" ${defaultDetector==='facemapper'?'selected':''}>Filtered FaceMapper only · matches QC</option><option value="hybrid" ${defaultDetector==='hybrid'?'selected':''}>Filtered FaceMapper + Haar supplement · may add boxes</option><option value="haar" ${defaultDetector==='haar'?'selected':''}>Haar detector only</option></select></div>
      <div class="field"><label>Mask</label><select id="privacy-mode"><option value="gaussian" ${defaultMode==='gaussian'?'selected':''}>Gaussian blur</option><option value="pixelate" ${defaultMode==='pixelate'?'selected':''}>Pixelation</option><option value="solid" ${defaultMode==='solid'?'selected':''}>Solid mask</option></select></div>
      <div class="form-row"><div class="field"><label>Strength</label><input id="privacy-strength" type="number" value="${Number(basisParams.strength??30)}" min="2" max="100"></div><div class="field"><label>Bounding-box padding (%)</label><input id="privacy-padding" type="number" value="${Number(basisParams.padding_percent??0)}" min="0" max="100" step="1"></div></div>
      <div class="field"><label>Timestamp tolerance (ms)</label><input id="privacy-tolerance" type="number" value="${Number(basisParams.timestamp_tolerance_ms??5)}" min="0.1" step="0.1"></div>
      <div class="field"><label>Coverage policy</label><select id="privacy-policy"><option value="strict" ${defaultPolicy==='strict'?'selected':''}>Strict — block uncovered timestamps</option><option value="allow" ${defaultPolicy==='allow'?'selected':''}>Allow — report uncovered timestamps</option></select></div>
      <div class="toggle-row"><span>Preserve source audio</span><button class="toggle ${defaultAudio?'on':''}" id="privacy-audio" data-on="${String(defaultAudio)}"></button></div>
    </div></details>

    <details class="privacy-compact-options privacy-media-options"><summary><span>Media tools</span><small>${appState.server.media_tools?.ready?'FFmpeg + ffprobe ready':'Check FFmpeg + ffprobe'}</small></summary><div class="privacy-options-body">${mediaToolsPanel()}</div></details>
    <div class="form-help">The resulting <b>Privacy clip</b> is ready for Face AOI, gaze-to-frame mapping, Vicon alignment, mutual attention and joint-attention analysis. Partial boundary frames are retained automatically.</div>
  </div></section>

  <section class="property-section privacy-advanced-section"><details class="privacy-advanced"><summary>Advanced · reusable full-recording source and derived clips</summary><div class="property-stack privacy-advanced-body">
    <div class="form-help"><strong>Not needed for the normal R01/R02 workflow.</strong> Use this only when you intentionally want to blur an entire recording once and later derive several different intervals from it. The normal button above is faster and avoids a second video encode.</div>
    <div class="advanced-block"><strong>A · Build a reusable full-recording source</strong><p>Uses the Native recording, Raw Face Mapper fallback, and mask settings selected above. A trim-scoped filtered package is intentionally not used because it cannot cover the full recording.</p><button class="soft-button full" id="run-privacy-full" ${!native.length?'disabled':''}>Create full-recording Privacy source</button></div>
    <div class="advanced-divider"></div>
    <div class="advanced-block"><strong>B · Derive a clip from an existing reusable source</strong>
      <div class="field"><label>Full-recording Privacy source</label><select id="privacy-clip-source">${privacyArtifactOptionRows(reusableSources,preferredSource?.id||'')}</select></div>
      <div class="field"><label>Use existing Native-reference trim</label><select id="privacy-native-trim">${optionRows(trims,defaultTrim,'Manual start + duration')}</select></div>
      <div class="field"><label>Native source · manual mode</label><select id="privacy-clip-native">${optionRows(native,defaultNative,'Infer from Privacy source')}</select></div>
      <div class="form-row"><div class="field"><label>Start in NRD video</label><input id="privacy-clip-start" value="00:00:00:00"></div><div class="field"><label>Retained duration</label><input id="privacy-clip-duration" value="00:00:10:00"></div></div>
      <div class="form-row"><div class="field"><label>Timecode FPS</label><select id="privacy-clip-timecode-fps"><option>24</option><option>25</option><option selected>30</option><option>50</option><option>60</option></select></div><div class="field"><label>Output video FPS</label><select id="privacy-clip-output-fps">${[24,25,30,50,60].map(value=>`<option value="${value}" ${value===defaultFps?'selected':''}>${value}</option>`).join('')}</select></div></div>
      <div class="field"><label>Output label</label><input id="privacy-clip-label" value="${esc(advancedLabel)}"></div>
      <button class="soft-button full" id="run-privacy-clip" ${!reusableSources.length?'disabled':''}>Derive synchronized clip from source</button>
      ${!reusableSources.length?'<div class="form-help">No reusable full-recording Privacy source exists yet. Create one with A, or use the recommended one-step workflow above.</div>':''}
    </div>
    ${oldTrimScoped.length?`<div class="form-help">${oldTrimScoped.length} older trim-scoped Privacy coverage artifact(s) remain available in Derived Artifacts for provenance, but are no longer part of the normal workflow.</div>`:''}
  </div></details></section>`;
}

function calibrationC3DClipPhaseMarkup(stage){
  const handoff=stage.handoff;
  const running=Boolean(appState.pendingC3DDefinedClipsJobId);
  let allInputsReady=Boolean(handoff&&stage.allRolesReady);
  const cards=['R01','R02'].map(role=>{
    const row=stage.rows.find(item=>item.role===role)||{};
    const nativeRows=calibrationClipAssetRows('pupil_native',role);
    const timeseriesRows=calibrationClipAssetRows('pupil_cloud',role);
    const facemapperRows=calibrationClipAssetRows('face_mapper',role);
    const preferredTimeseries=String(row.timeSyncArtifact?.parameters?.source_asset_id||'');
    const nativeId=calibrationClipInputId(role,'native_asset_id',nativeRows);
    const timeseriesId=calibrationClipInputId(role,'timeseries_asset_id',timeseriesRows,preferredTimeseries);
    const facemapperId=calibrationClipInputId(role,'facemapper_asset_id',facemapperRows);
    appState.calibrationClipSelections[role]={native_asset_id:nativeId,timeseries_asset_id:timeseriesId,facemapper_asset_id:facemapperId};
    const ready=Boolean(nativeId&&timeseriesId&&facemapperId&&row.timeSyncReady);
    allInputsReady=allInputsReady&&ready;
    return `<article class="calibration-sync-role-card ${ready?'ready':'pending'}">
      <div class="calibration-sync-role-head"><div><span class="eyebrow">${role} · inverse TimeGraph → NRD</span><strong>${esc(row.wearerMapping?.etm_id||'Assigned ETM unavailable')}</strong></div><span class="status-pill ${ready?'good':'warning'}">${ready?'Inputs ready':'Select inputs'}</span></div>
      <div class="field"><label>Original Native Recording · NRD</label><select class="cal-c3d-clip-input" data-role="${role}" data-field="native_asset_id" id="cal-c3d-clip-native-${role.toLowerCase()}">${optionRows(nativeRows,nativeId,`Select ${role} NRD`)}</select></div>
      <div class="field"><label>Pupil Timeseries · same source used for sync</label><select class="cal-c3d-clip-input" data-role="${role}" data-field="timeseries_asset_id" id="cal-c3d-clip-timeseries-${role.toLowerCase()}">${optionRows(timeseriesRows,timeseriesId,`Select ${role} TSD`)}</select></div>
      <div class="field"><label>FaceMapper export</label><select class="cal-c3d-clip-input" data-role="${role}" data-field="facemapper_asset_id" id="cal-c3d-clip-facemapper-${role.toLowerCase()}">${optionRows(facemapperRows,facemapperId,`Select ${role} FaceMapper`)}</select></div>
      <div class="form-help">The body C3D interval is converted to this role's Pupil clock with the accepted affine mapping. Exact NRD frames come from the companion <code>.time</code> index and MP4 PTS.</div>
    </article>`;
  }).join('');
  const pair=stage.alignedClipSet;
  const pairStatus=pair?`<div class="success-card compact"><strong>Aligned R01 + R02 clips ready</strong><p>${esc(pair.label)} · ${Number(pair.qc?.common_duration_sec||0).toFixed(3)} s · maximum frame-boundary error ${Math.max(Number(pair.qc?.maximum_start_boundary_error_ms||0),Number(pair.qc?.maximum_end_boundary_error_ms||0)).toFixed(3)} ms</p><div class="stat-list"><div class="stat-row"><span>R01 trim</span><strong>${esc(stage.alignedRoleTrims?.R01?.label||'Missing')}</strong></div><div class="stat-row"><span>R02 trim</span><strong>${esc(stage.alignedRoleTrims?.R02?.label||'Missing')}</strong></div><div class="stat-row"><span>Source video re-encoded</span><strong>No</strong></div></div></div>`
    :`<div class="form-help">After the synchronized handoff is ready, OculoStudio will derive one shared physical Vicon interval from the overlap of the R01 and R02 body C3Ds. No start or end time is typed manually.</div>`;
  const ctx=activeWorkflowContext();
  const defaultLabel=`${session()?.label||'S001'}_${ctx.interactionId||'I01'}_C3D_aligned_clip`;
  return `<div class="calibration-sync-intro"><strong>C3D defines the interaction interval.</strong><p>OculoStudio uses the common R01/R02 body-C3D interval, inverts each verified Pupil↔Vicon TimeGraph, and creates role-specific virtual NRD clips. The original ego videos are unchanged.</p></div>
    <div class="calibration-sync-role-grid">${cards}</div>
    <details class="calibration-sync-settings"><summary>Automatic clip settings</summary><div class="property-stack calibration-route-body">
      <div class="field"><label>Output label prefix</label><input id="cal-c3d-clip-label" value="${esc(defaultLabel)}"></div>
      <div class="form-row"><div class="field"><label>Boundary coverage tolerance (ms)</label><input id="cal-c3d-clip-boundary-tolerance" type="number" min="0" max="100" step="0.1" value="20"></div><div class="field"><label>Event interval policy</label><select id="cal-c3d-clip-policy"><option value="clip" selected>Clip overlapping events · recommended</option><option value="inside">Keep only fully contained events</option></select></div></div>
      <div class="form-help">The normal output is a virtual Native-reference interval plus trimmed TSD/FaceMapper tables and an exact frame map. Privacy can encode a physical video later without changing scientific timestamps.</div>
    </div></details>
    <button class="primary-button full" id="run-c3d-defined-clips" ${!allInputsReady||running?'disabled':''}>${running?'Creating aligned R01 + R02 clips…':pair?'Re-create clips from current C3D + TimeGraph':'Create aligned R01 + R02 clips'}</button>
    ${pairStatus}`;
}


function calibrationPairedProperties() {
  const mocap=assets(['c3d','mocap_npz']);
  const profiles=calibrationProfilesForActiveSet();
  const stage=calibrationWorkflowState('R01');
  const ctx=activeWorkflowContext();
  const bodyGuess=mocap.filter(a=>!/ETM|EYE[_\- ]?TRACKER/i.test(a.label||'') && Number(a.metadata?.marker_count||999)>12);
  const etmGuess=mocap.filter(a=>/ETM|EYE[_\- ]?TRACKER/i.test(a.label||'') || Number(a.metadata?.marker_count||999)<=12);
  const bodyOptions=bodyGuess.length?bodyGuess:mocap,etmOptions=etmGuess.length?etmGuess:mocap;
  const suggestedR01=bodyOptions.find(a=>/(?:^|[_\-])R0?1(?:[_\-.]|$)/i.test(a.label||''))?.id||'';
  const suggestedR02=bodyOptions.find(a=>/(?:^|[_\-])R0?2(?:[_\-.]|$)/i.test(a.label||''))?.id||'';
  const suggestedA=etmOptions.find(a=>/(?:^|[_\-])ETM[_\-]?A(?:[_\-.]|$)/i.test(a.label||''))?.id||'';
  const suggestedB=etmOptions.find(a=>/(?:^|[_\-])ETM[_\-]?B(?:[_\-.]|$)/i.test(a.label||''))?.id||'';
  const assignmentDefaultLabel=`${session()?.label||'S001'}_${ctx.interactionId||'I01'}_ETM_wearer_assignment`;
  const handoffDefaultLabel=`${session()?.label||'S001'}_${ctx.interactionId||'I01'}_calibration_handoff`;
  const normalizeHardware=value=>{
    const token=String(value||'').toUpperCase().replace(/[^A-Z0-9]+/g,'');
    if(token.includes('ETMA')||token.includes('EYETRACKER1'))return 'ETM-A';
    if(token.includes('ETMB')||token.includes('EYETRACKER2'))return 'ETM-B';
    return '';
  };
  const profileHardware=profile=>calibrationProfileHardware(profile).replace('_','-');
  const profileA=calibrationSetProfileForHardware('ETM_A'),profileB=calibrationSetProfileForHardware('ETM_B');
  let automaticBindingProfiles=[profileA,profileB].filter(Boolean);
  if(automaticBindingProfiles.length<2)automaticBindingProfiles=profiles.slice(-2);
  const automaticBindingIds=new Set(automaticBindingProfiles.map(profile=>profile.id));
  const bindingReady=Boolean(stage.wearerAssignment&&automaticBindingProfiles.length>=2&&stage.synchronizationReady&&stage.rows.filter(row=>row.timeSyncArtifact).length>=2);
  const bindingProfileRows=profiles.length?profiles.map(profile=>`<label class="check-row"><input type="checkbox" class="cal-binding-profile-check" value="${esc(profile.id)}" ${automaticBindingIds.has(profile.id)?'checked':''}><span>${esc(profile.label)}${profileHardware(profile)?' · '+esc(profileHardware(profile)):''}</span></label>`).join(''):'<span class="muted small">Create or import both physical ETM profiles in a role detail view first.</span>';
  const setRow=activeCalibrationSetRecord();
  const hardwareReady=setRow?.hardware_ready||{};
  const profileBuilderSummary=`<div class="calibration-set-profile-route ${setRow?.physical_profiles_ready?'ready':'blocked'}"><div><span class="eyebrow">Project-level physical hardware profiles</span><strong>${esc(setRow?.code||activeCalibrationSetId())} · ${setRow?.physical_profiles_ready?'ETM-A and ETM-B ready':'Resolve board geometry and build shared profiles'}</strong><p>Head-marker evidence identifies the physical ETM. Scene AprilTags and the 16-marker board independently resolve board orientation. Calibration actors may differ from the current Interaction wearers.</p></div><div class="calibration-set-profile-slots"><span class="${hardwareReady.ETM_A?'ready':'missing'}">ETM-A · ${hardwareReady.ETM_A?'Ready':'Missing'}</span><span class="${hardwareReady.ETM_B?'ready':'missing'}">ETM-B · ${hardwareReady.ETM_B?'Ready':'Missing'}</span></div><button type="button" class="primary-button calibration-build-set-profiles" data-set-id="${esc(setRow?.id||'')}">${setRow?.physical_profiles_ready?'Review / rebuild physical profiles':'Resolve board geometry & build both profiles'}</button></div>`;
  const roleCards=stage.rows.map(row=>{
    const etm=normalizeHardware(row.wearerMapping?.etm_id||row.mapping?.etm_id||'')||row.wearerMapping?.etm_id||'Unresolved ETM';
    const availableProfile=etm==='ETM-A'?profileA:etm==='ETM-B'?profileB:null;
    const ready=Boolean(availableProfile&&row.assignmentReady);
    const status=row.ready?'Ready':!row.assignmentReady?'Wearer unresolved':!availableProfile?'Profile missing':!row.timeSyncReady?'Sync missing':!row.handoffSyncReady?'Handoff pending':'Review';
    return `<article class="calibration-role-detail-card ${ready?'ready':''}"><header><div><span class="eyebrow">${esc(row.role)} · ${esc(calibrationParticipantForRole(row.role))}</span><strong>${esc(row.role)} → ${esc(etm)}</strong></div><span class="status-pill ${row.ready?'good':'warning'}">${esc(status)}</span></header><small>${availableProfile?`Profile: ${esc(availableProfile.label)}`:'No compatible physical-hardware profile'}</small><small>${row.timeSyncArtifact?`TimeGraph: ${esc(row.timeSyncArtifact.label)}`:'No accepted Pupil IMU ↔ ETM TimeGraph mapping'}</small><button type="button" class="soft-button full" data-calibration-view="${esc(row.role)}">Open ${esc(row.role)} details</button></article>`;
  }).join('');
  return `<section class="property-section calibration-stage-readiness"><div class="section-label">Paired calibration · R01 + R02</div><div class="property-stack"><div class="calibration-paired-intro"><strong>One paired workflow, two role-specific evidence records.</strong><p>Actions on this page resolve both wearers, synchronize both clocks, bind both hardware profiles and create both aligned clips transactionally. R01/R02 detail views change only what evidence and profile form is displayed.</p></div><div class="calibration-readiness-grid compact">${calibrationReadinessRows(stage)}</div><button class="primary-button full" data-go="clips" ${stage.allRolesReady&&stage.alignedClipsReady?'':'disabled'}>Continue to Trim review</button></div></section>
  <section class="property-section calibration-phase calibration-phase-a"><div class="section-label">1 · Resolve interaction wearer · paired</div><div class="property-stack"><div class="form-help">Determine R01/R02 → physical ETM-A/ETM-B from the four interaction C3D files. This assignment is immutable after the synchronized handoff is published.</div>${stage.wearerAssignment?`<div class="success-card compact"><strong>Wearer assignment ready</strong><p>${esc(stage.wearerAssignment.label)}</p>${Object.entries(stage.wearerAssignment.qc?.assignments_summary||{}).map(([role,row])=>`<div class="stat-row"><span>${esc(role)}</span><strong>${esc(row.etm_id||'—')} · ${esc(row.confidence||'—')}</strong></div>`).join('')}</div>`:'<div class="warning-card compact"><strong>Wearer assignment missing</strong><p>Select both body C3Ds and both full-length ETM C3Ds.</p></div>'}<details ${stage.wearerAssignment?'':'open'}><summary>${stage.wearerAssignment?'Re-run paired wearer resolution':'Paired wearer-resolution inputs'}</summary><div class="property-stack calibration-route-body"><div class="field"><label>R01 body C3D</label><select id="etm-body-r01">${optionRows(bodyOptions,suggestedR01,'Select R01 body')}</select></div><div class="field"><label>R02 body C3D</label><select id="etm-body-r02">${optionRows(bodyOptions,suggestedR02,'Select R02 body')}</select></div><div class="field"><label>Interaction ETM candidate A</label><select id="etm-candidate-a">${optionRows(etmOptions,suggestedA,'Select ETM A')}</select></div><div class="field"><label>Interaction ETM candidate B</label><select id="etm-candidate-b">${optionRows(etmOptions,suggestedB,'Select ETM B')}</select></div><details><summary>Advanced wearer-resolution gates</summary><div class="property-stack" style="margin-top:8px"><div class="field"><label>Maximum head distance (mm)</label><input id="etm-max-distance" type="number" value="350" min="1"></div><div class="field"><label>Minimum role/global margin (mm)</label><input id="etm-role-margin" type="number" value="100" min="0"></div></div></details><div class="field"><label>Wearer-assignment label</label><input id="etm-assignment-label" value="${esc(assignmentDefaultLabel)}"></div><button class="primary-button full" id="run-etm-assignment" ${bodyOptions.length<2||etmOptions.length<2?'disabled':''}>Resolve R01 + R02 wearer mapping</button></div></details></div></section>
  <section class="property-section calibration-phase calibration-phase-b"><div class="section-label">2 · Build physical ETM profiles from the shared Calibration Set</div><div class="property-stack">${profileBuilderSummary}<div class="calibration-profile-detail-grid">${roleCards}</div><div class="form-help">The shared Calibration Set is the source of truth. R01/R02 detail views are evidence inspection only; they do not decide which calibration actor owns ETM-A or ETM-B.</div></div></section>
  <section class="property-section calibration-phase calibration-phase-c"><div class="section-label">3 · Synchronize both Pupil clocks</div><div class="property-stack">${calibrationSyncPhaseMarkup(stage)}</div></section>
  ${audioVideoSyncPanel()}
  <section class="property-section calibration-phase calibration-phase-d"><div class="section-label">4 · Validate both profiles and create the paired handoff</div><div class="property-stack"><div class="form-help">This publishes one immutable handoff containing the fixed wearer mapping, two physical-hardware profiles and two production-ready affine TimeGraph mappings.</div><div class="stat-list"><div class="stat-row"><span>Wearer assignment</span><strong>${esc(stage.wearerAssignment?.label||'Not ready')}</strong></div><div class="stat-row"><span>ETM-A profile</span><strong>${esc(profileA?.label||'Missing')}</strong></div><div class="stat-row"><span>ETM-B profile</span><strong>${esc(profileB?.label||'Missing')}</strong></div><div class="stat-row"><span>R01 clock sync</span><strong>${stage.rows.find(row=>row.role==='R01')?.timeSyncReady?'Verified':'Missing'}</strong></div><div class="stat-row"><span>R02 clock sync</span><strong>${stage.rows.find(row=>row.role==='R02')?.timeSyncReady?'Verified':'Missing'}</strong></div></div><input type="hidden" id="cal-binding-assignment" value="${esc(stage.wearerAssignment?.id||'')}"><details><summary>Manual profile selection and binding gates</summary><div class="property-stack calibration-route-body"><div class="check-list">${bindingProfileRows}</div><div class="field"><label>Maximum tracker-shape RMS (mm)</label><input id="cal-binding-max-rms" type="number" value="8" min="0.1" step="0.1"></div><div class="field"><label>Minimum shape-separation margin (mm)</label><input id="cal-binding-margin" type="number" value="4" min="0" step="0.1"></div></div></details><div class="field"><label>Calibration handoff label</label><input id="cal-binding-label" value="${esc(handoffDefaultLabel)}"></div><button class="primary-button full" id="run-calibration-binding" ${bindingReady?'':'disabled'}>Create paired Gaze 3D handoff</button>${stage.handoff?`<div class="success-card compact"><strong>Current paired handoff</strong><p>${esc(stage.handoff.label)} · ${stage.handoff.qc?.handoff_complete?'complete':'review required'} · both-clock sync ${stage.handoff.qc?.time_sync_complete?'verified':'missing'}</p></div>`:`<div class="form-help">${bindingReady?'Both hardware profiles and both clock mappings are ready.':!stage.synchronizationReady?'Complete both automatic synchronizations first.':'Resolve both wearers and create both ETM profiles first.'}</div>`}</div></section>
  <section class="property-section calibration-phase calibration-phase-e"><div class="section-label">5 · Create paired C3D-defined Pupil clips</div><div class="property-stack">${calibrationC3DClipPhaseMarkup(stage)}</div></section>`;
}

function calibrationProperties() {
  const mode=calibrationDisplayMode();
  if(mode==='paired')return calibrationPairedProperties();
  const stage=calibrationWorkflowState(mode);
  const row=stage.rows.find(item=>item.role===mode);
  const etm=row?.wearerMapping?.etm_id||row?.mapping?.etm_id||'Unresolved ETM';
  const setRow=activeCalibrationSetRecord();
  return `<section class="property-section"><div class="section-label">${esc(mode)} evidence detail</div><div class="property-stack"><div class="calibration-paired-intro"><strong>${esc(mode)} → ${esc(etm)}</strong><p>This view inspects Interaction evidence only. Physical ETM profiles are built once from ${esc(setRow?.code||activeCalibrationSetId())}; the calibration actor does not have to equal the current wearer.</p></div><button type="button" class="primary-button full calibration-build-set-profiles" data-set-id="${esc(setRow?.id||'')}">${setRow?.physical_profiles_ready?'Review shared ETM profiles':'Build missing shared ETM profiles'}</button><button type="button" class="soft-button full" data-calibration-view="paired">Return to paired summary</button><button type="button" class="soft-button full" data-calibration-view="${mode==='R01'?'R02':'R01'}">View ${mode==='R01'?'R02':'R01'} evidence</button></div></section>${audioVideoSyncPanel({roleId:mode,compact:true})}<details class="property-section calibration-manual-recovery"><summary class="section-label">Advanced manual profile estimation / recovery only</summary><div class="calibration-manual-recovery-body">${calibrationRoleProperties()}</div></details>`;
}

function calibrationRoleProperties() {
  const mocap=assets(['c3d','mocap_npz']), profiles=calibrationProfilesForActiveSet();
  const activeSet=activeCalibrationSetId();
  const activeSetRecord=calibrationSetRecord(activeSet);
  const projectSets=calibrationSetRows();
  const calibrationRows=calibrationAssets(null,activeSet);
  const calibrationNative=calibrationAssets(['pupil_native'],activeSet);
  const calibrationMocap=calibrationAssets(['c3d','mocap_npz'],activeSet);
  const unregisteredMeasured=profiles.filter(p=>{
    const hardware=p.parameters?.hardware_id||p.qc?.hardware_id;
    const source=p.parameters?.calibration_source||p.qc?.calibration_source||'';
    return !hardware && /measured|apriltag/i.test(source);
  });
  const transferred=profiles.filter(p=>(p.parameters?.calibration_source||p.qc?.calibration_source)==='transferred_hardware_profile');
  const trims=artifacts(['native_reference_trim']), associations=artifacts(['participant_association']);
  const trackers=mocap.filter(a=>/ETM|EYE[_\- ]?TRACKER/i.test(a.label||'')||Number(a.metadata?.marker_count||999)<=12);
  const trackerCandidates=calibrationMocap.filter(a=>a.metadata?.calibration_role==='tracker_candidate');
  const boardCandidates=calibrationMocap.filter(a=>a.metadata?.calibration_role==='calibration_board');
  const worldCandidates=calibrationMocap.filter(a=>a.metadata?.calibration_role==='world_reference');
  const focusedRole=calibrationContextRole();
  const focusedParticipant=calibrationParticipantForRole(focusedRole);
  const draft=calibrationDraft(activeSet,focusedRole);
  const matchingNative=calibrationNativeForParticipant(calibrationNative,focusedParticipant);
  const selectedNative=calibrationNative.some(row=>row.id===draft.nativeId)?draft.nativeId:(matchingNative?.id||calibrationNative[0]?.id||'');
  const selectedBoard=boardCandidates.some(row=>row.id===draft.boardId)?draft.boardId:(boardCandidates[0]?.id||'');
  const selectedWorld=worldCandidates.some(row=>row.id===draft.worldId)?draft.worldId:(worldCandidates[0]?.id||'');
  const selectedTrackerIds=new Set((draft.trackerIds?.length?draft.trackerIds:trackerCandidates.map(row=>row.id)).filter(id=>trackerCandidates.some(row=>row.id===id)));
  const matchingProfiles=profiles.filter(profile=>calibrationProfileParticipant(profile)===focusedParticipant).sort(byCreated);
  const previewProfile=matchingProfiles.slice(-1)[0]?.id||profiles.slice().sort(byCreated).slice(-1)[0]?.id||'';
  const sessionLabel=session()?.label||'S001';
  const defaultValidSessions=draft.validSessions||calibrationDefaultValidSessions(activeSet,calibrationRows);
  const participantValue=draft.participantId||focusedParticipant;
  const mountValue=draft.mountVersion||'mount_v1';
  const contextNative=calibrationNative.find(row=>row.id===selectedNative);
  const ctx=activeWorkflowContext();
  const bodyGuess=mocap.filter(a=>!/ETM|EYE[_\- ]?TRACKER/i.test(a.label||'') && Number(a.metadata?.marker_count||999)>12);
  const etmGuess=mocap.filter(a=>/ETM|EYE[_\- ]?TRACKER/i.test(a.label||'') || Number(a.metadata?.marker_count||999)<=12);
  const bodyOptions=bodyGuess.length?bodyGuess:mocap, etmOptions=etmGuess.length?etmGuess:mocap;
  const suggestedR01=bodyOptions.find(a=>/(?:^|[_\-])R0?1(?:[_\-.]|$)/i.test(a.label||''))?.id||'';
  const suggestedR02=bodyOptions.find(a=>/(?:^|[_\-])R0?2(?:[_\-.]|$)/i.test(a.label||''))?.id||'';
  const suggestedA=etmOptions.find(a=>/(?:^|[_\-])ETM[_\-]?A(?:[_\-.]|$)/i.test(a.label||''))?.id||'';
  const suggestedB=etmOptions.find(a=>/(?:^|[_\-])ETM[_\-]?B(?:[_\-.]|$)/i.test(a.label||''))?.id||'';
  const stage=calibrationWorkflowState(focusedRole);
  const assignments=stage.handoffs;
  const scopedAssignments=rowsForActiveWorkflowContext(assignments,{role:false,allowUnscoped:true}).slice().sort(byCreated);
  const bodyAssignment=scopedAssignments.slice(-1)[0]||assignments.slice(-1)[0]||null;
  const wearerAssignmentRows=stage.wearerAssignments.length?stage.wearerAssignments:(stage.wearerAssignment?[stage.wearerAssignment]:[]);
  const assignmentDefaultLabel=`${session()?.label||'S001'}_${ctx.interactionId||'I01'}_ETM_wearer_assignment`;
  const handoffDefaultLabel=`${session()?.label||'S001'}_${ctx.interactionId||'I01'}_calibration_handoff`;

  const normalizeHardware=value=>{
    const token=String(value||'').toUpperCase().replace(/[^A-Z0-9]+/g,'');
    if(token.includes('ETMA')||token.includes('EYETRACKER1'))return 'ETM-A';
    if(token.includes('ETMB')||token.includes('EYETRACKER2'))return 'ETM-B';
    return '';
  };
  const profileHardware=profile=>normalizeHardware([
    profile?.parameters?.hardware_display_name,profile?.qc?.hardware_display_name,
    profile?.parameters?.hardware_id,profile?.qc?.hardware_id,profile?.label
  ].filter(Boolean).join(' '));
  const sortedProfiles=profiles.slice().sort(byCreated);
  const latestFor=token=>sortedProfiles.filter(profile=>profileHardware(profile)===token).slice(-1)[0]||null;
  const profileA=latestFor('ETM-A'), profileB=latestFor('ETM-B');
  const targetHardware=normalizeHardware(stage.current?.wearerMapping?.etm_id||'');
  const targetProfile=targetHardware==='ETM-A'?profileA:targetHardware==='ETM-B'?profileB:null;
  const profileSlots=['ETM-A','ETM-B'].map(token=>{
    const profile=token==='ETM-A'?profileA:profileB;
    const source=profile?.parameters?.calibration_source||profile?.qc?.calibration_source||'';
    return `<article class="calibration-hardware-slot ${profile?'ready':'missing'}"><div><span class="eyebrow">Physical hardware</span><strong>${token}</strong><small>${profile?esc(profile.label):'No compatible profile yet'}</small>${profile?`<small>${esc(source||'registered profile')}</small>`:''}</div><span class="status-pill ${profile?'good':'warning'}">${profile?'Ready':'Missing'}</span></article>`;
  }).join('');
  const targetLabel=targetHardware||'Unresolved ETM';
  const nextTask=targetProfile
    ? `${targetLabel} already has a compatible profile. Complete the other missing hardware profile or continue to handoff validation.`
    : `Create or import the missing ${targetLabel} profile for ${focusedRole} / ${focusedParticipant}.`;

  const registryRows=calibrationRows.length?calibrationRows.map(a=>`<div class="calibration-registry-row"><div><strong>${esc(a.label)}</strong><small>${esc(calibrationRoleLabel(a.metadata?.calibration_role))}${a.metadata?.hardware_hint?' · '+esc(a.metadata.hardware_hint):''}${a.participant_id?' · '+esc(a.participant_id):''}</small></div><span class="status-pill ${a.available?'good':'warning'}">${a.available?'Ready':'Missing'}</span></div>`).join(''):'<div class="empty-tree">No sources in this calibration set.</div>';
  const scan=(appState.calibrationScan?.calibration_set_id===activeSet)?appState.calibrationScan:null;
  const scanRows=scan?.files?.length?scan.files.map(row=>`<div class="calibration-scan-row"><span class="calibration-scan-kind">${esc(calibrationRoleLabel(row.calibration_role))}</span><div><strong title="${esc(row.path)}">${esc(row.name)}</strong><small>${fmtBytes(row.size_bytes)}${row.hardware_hint?' · '+esc(row.hardware_hint):''}${row.participant_id?' · '+esc(row.participant_id):''}</small></div><span class="status-pill ${row.already_registered?'good':''}">${row.already_registered?'Registered':'New'}</span></div>`).join(''):'';
  const scanSummary=scan?`<div class="calibration-scan-summary ${scan.file_count===1?'warning':''}"><div><strong>${scan.file_count} supported source${scan.file_count===1?'':'s'} discovered</strong><span>${scan.new_count} new · ${scan.existing_count} already registered</span></div>${scan.file_count===1?'<div class="warning-card compact"><strong>Only one file was discovered.</strong><p>If this is a folder containing several files, choose <b>Scan folder</b> again and verify the path before importing.</p></div>':''}<div class="calibration-scan-list">${scanRows}</div></div>`:'<div class="form-help">Choose a folder or several files, then scan. OculoStudio shows the exact import list before changing the project.</div>';
  const importLabel=scan?`Import all ${scan.file_count} discovered source${scan.file_count===1?'':'s'}`:'Import scanned sources';
  const registryOpen=calibrationRows.length===0?'open':'';
  const estimateOpen=!targetProfile&&calibrationRows.length>0?'open':'';

  let automaticBindingProfiles=[profileA,profileB].filter(Boolean);
  if(automaticBindingProfiles.length<2)automaticBindingProfiles=sortedProfiles.slice(-2);
  const automaticBindingIds=new Set(automaticBindingProfiles.map(profile=>profile.id));
  const bindingSyncIds=stage.rows.map(row=>row.timeSyncArtifact?.id).filter(Boolean);
  const bindingReady=Boolean(wearerAssignmentRows.length&&automaticBindingProfiles.length>=2&&stage.synchronizationReady&&bindingSyncIds.length>=2);
  const bindingProfileRows=profiles.length?profiles.map(profile=>`<label class="check-row"><input type="checkbox" class="cal-binding-profile-check" value="${esc(profile.id)}" ${automaticBindingIds.has(profile.id)?'checked':''}><span>${esc(profile.label)}${profileHardware(profile)?' · '+esc(profileHardware(profile)):''}</span></label>`).join(''):'<span class="muted small">Create or import both physical ETM profiles first.</span>';

  const transferSection=transferred.length?`<details class="calibration-secondary-route"><summary>Production acceptance for transferred profiles</summary><div class="property-stack calibration-route-body">
    <div class="form-help">Only transferred profiles require this interaction-specific production gate. Measured or directly imported profiles do not.</div>
    <div class="field"><label>Transferred calibration profile</label><select id="cal-accept-profile">${optionRows(transferred,'','Select transferred profile')}</select></div>
    <div class="field"><label>Native-reference trim</label><select id="cal-accept-trim">${optionRows(trims,'','Select trim with TSD + Face Mapper')}</select></div>
    <div class="field"><label>ETM-role assignment</label><select id="cal-accept-etm">${optionRows(scopedAssignments.length?scopedAssignments:assignments,bodyAssignment?.id||'','Select assignment')}</select></div>
    <div class="field"><label>Wearer role</label><select id="cal-accept-role"><option value="R01" ${focusedRole==='R01'?'selected':''}>R01</option><option value="R02" ${focusedRole==='R02'?'selected':''}>R02</option></select></div>
    <div class="field"><label>Existing participant association (optional)</label><select id="cal-accept-association">${optionRows(associations,'','Run head projection automatically')}</select></div>
    <div class="field"><label>Known-target observations CSV (optional)</label><div class="inline-input"><input id="cal-accept-known" placeholder="timestamp_ns,target_x_mm,target_y_mm,target_z_mm"><button class="input-action" data-browse="cal-accept-known">…</button></div></div>
    <details><summary>Acceptance thresholds</summary><div class="property-stack" style="margin-top:8px">
      <div class="field"><label>Maximum synchronization RMS (ms)</label><input id="cal-accept-sync-rms" type="number" value="25" min="0.1" step="0.1"></div>
      <div class="field"><label>Minimum projected-head valid fraction</label><input id="cal-accept-proj-valid" type="number" value="0.60" min="0" max="1" step="0.05"></div>
      <div class="field"><label>Minimum partner match frame fraction</label><input id="cal-accept-match" type="number" value="0.55" min="0" max="1" step="0.05"></div>
      <div class="field"><label>Maximum median normalized centre error</label><input id="cal-accept-centre" type="number" value="0.75" min="0.05" step="0.05"></div>
      <div class="field"><label>Minimum median projected IoU</label><input id="cal-accept-iou" type="number" value="0.03" min="0" max="1" step="0.01"></div>
      <div class="field"><label>Maximum systematic offset (face-size units)</label><input id="cal-accept-offset" type="number" value="0.40" min="0" step="0.05"></div>
      <div class="field"><label>Maximum review-required fraction</label><input id="cal-accept-review" type="number" value="0.40" min="0" max="1" step="0.05"></div>
      <div class="field"><label>Known-target median angular error (deg)</label><input id="cal-accept-angle" type="number" value="3" min="0.1" step="0.1"></div>
      <div class="field"><label>Known-target maximum point-to-ray distance (mm)</label><input id="cal-accept-distance" type="number" value="80" min="1" step="1"></div>
    </div></details>
    <div class="field"><label>Acceptance label</label><input id="cal-accept-label" placeholder="S001_I01_R01_calibration_acceptance"></div>
    <div class="toggle-row"><span>Write head-projection Parquet</span><button class="toggle" id="cal-accept-parquet" data-on="false"></button></div>
    <button class="primary-button full" id="run-calibration-acceptance">Run automatic acceptance gate</button>
  </div></details>`:`<div class="success-card compact calibration-no-transfer"><strong>No transfer gate required</strong><p>Measured or directly imported profiles can proceed directly to handoff validation.</p></div>`;

  return `<section class="property-section calibration-stage-readiness"><div class="section-label">Calibration stage · required before Gaze 3D</div><div class="property-stack">
    <div class="calibration-readiness-grid compact">${calibrationReadinessRows(stage)}</div>
    <button class="primary-button full" data-go="gaze3d" ${stage.allRolesReady?'':'disabled'}>Continue to Gaze 3D</button>
  </div></section>
  <section class="property-section calibration-phase calibration-phase-a" id="calibration-phase-a"><div class="section-label">1 · Resolve interaction wearer</div><div class="property-stack">
    <div class="form-help">Determine only who wore interaction ETM-A and ETM-B. Hardware profiles are not used and cannot change this result.</div>
    ${stage.wearerAssignment?`<div class="success-card compact"><strong>Wearer assignment ready</strong><p>${esc(stage.wearerAssignment.label)}</p>${Object.entries(stage.wearerAssignment.qc?.assignments_summary||{}).map(([role,row])=>`<div class="stat-row"><span>${esc(role)}</span><strong>${esc(row.etm_id||'—')} · ${esc(row.confidence||'—')}</strong></div>`).join('')}</div>`:'<div class="warning-card compact"><strong>Wearer assignment missing</strong><p>Select the four interaction C3Ds and resolve the wearer before preparing profiles.</p></div>'}
    <details ${stage.wearerAssignment?'':'open'}><summary>${stage.wearerAssignment?'Re-run wearer resolution':'Wearer-resolution inputs'}</summary><div class="property-stack calibration-route-body">
      <div class="field"><label>R01 body C3D</label><select id="etm-body-r01">${optionRows(bodyOptions,suggestedR01,'Select R01 body')}</select></div>
      <div class="field"><label>R02 body C3D</label><select id="etm-body-r02">${optionRows(bodyOptions,suggestedR02,'Select R02 body')}</select></div>
      <div class="field"><label>Interaction ETM candidate A</label><select id="etm-candidate-a">${optionRows(etmOptions,suggestedA,'Select ETM A')}</select></div>
      <div class="field"><label>Interaction ETM candidate B</label><select id="etm-candidate-b">${optionRows(etmOptions,suggestedB,'Select ETM B')}</select></div>
      <details><summary>Advanced wearer-resolution gates</summary><div class="property-stack" style="margin-top:8px"><div class="field"><label>Maximum head distance (mm)</label><input id="etm-max-distance" type="number" value="350" min="1"></div><div class="field"><label>Minimum role/global margin (mm)</label><input id="etm-role-margin" type="number" value="100" min="0"></div></div></details>
      <div class="field"><label>Wearer-assignment label</label><input id="etm-assignment-label" value="${esc(assignmentDefaultLabel)}"></div>
      <button class="primary-button full" id="run-etm-assignment" ${bodyOptions.length<2||etmOptions.length<2?'disabled':''}>Resolve wearer only</button>
    </div></details>
  </div></section>
  <section class="property-section calibration-phase calibration-phase-b" id="calibration-phase-b"><div class="section-label">2 · Prepare physical ETM profiles</div><div class="property-stack">
    <div class="calibration-profile-slot-grid">${profileSlots}</div>
    <div class="calibration-target-card ${targetProfile?'ready':'missing'}"><div><span class="eyebrow">Current profile task</span><strong>${esc(focusedRole)} / ${esc(focusedParticipant)} → ${esc(targetLabel)}</strong><small>${esc(contextNative?.label||'No matching calibration recording')}</small></div><span class="status-pill ${targetProfile?'good':'warning'}">${targetProfile?'Profile ready':'Profile missing'}</span></div>
    <div class="form-help">${esc(nextTask)} Selecting <b>R01 details / R02 details</b> changes only the role-specific calibration evidence shown here; the paired wearer assignment and physical ETM-A/ETM-B identities remain locked.</div>

    <div class="calibration-data-route-notice"><div><span class="eyebrow">Source data location</span><strong>${esc(activeSetRecord?.code||activeSet)} · ${calibrationRows.length} registered source reference${calibrationRows.length===1?'':'s'}</strong><p>Calibration source importing, appending, revision management and Session assignment are available only in Data → Calibration Data.</p></div><button type="button" class="soft-button calibration-open-data-sources">Open Data → Calibration Data</button></div>

    <details class="calibration-primary-route" ${estimateOpen}><summary>Estimate ${esc(targetHardware||'a missing ETM')} profile from the registered calibration session</summary><div class="property-stack calibration-route-body">
      <div class="field"><label>Calibration recording · ${esc(focusedParticipant)}</label><select id="cal-native">${optionRows(calibrationNative,selectedNative,'Select Native calibration recording')}</select></div>
      <div class="field"><label>Calibration board C3D</label><select id="cal-board">${optionRows(boardCandidates,selectedBoard,'Select board C3D/NPZ')}</select></div>
      <div class="field"><label>Tracker candidates</label><div class="check-list">${trackerCandidates.length?trackerCandidates.map(a=>`<label class="check-row"><input type="checkbox" class="cal-tracker-check" value="${esc(a.id)}" ${selectedTrackerIds.has(a.id)?'checked':''}><span>${esc(a.label)}</span></label>`).join(''):'<span class="muted small">No calibration tracker sources in this set</span>'}</div></div>
      <div class="field"><label>World/reference C3D</label><select id="cal-world">${optionRows(worldCandidates,selectedWorld,'No world C3D')}</select></div>
      <div class="form-row"><div class="field"><label>Calibration wearer ID</label><input id="cal-participant" value="${esc(participantValue)}"></div><div class="field"><label>Mount version</label><input id="cal-mount" value="${esc(mountValue)}"></div></div>
      <div class="field"><label>Valid sessions</label><input id="cal-valid-sessions" value="${esc(defaultValidSessions)}" placeholder="S001,S002,S003"></div>
      <details><summary>Advanced estimation settings</summary><div class="property-stack calibration-route-body">
        <div class="field"><label>Board geometry mode</label><select id="cal-board-mode"><option value="auto_legacy_16_marker" ${(draft.boardMode||'auto_legacy_16_marker')==='auto_legacy_16_marker'?'selected':''}>Auto-infer legacy 16-marker / 4-tag board</option><option value="explicit_definition" ${draft.boardMode==='explicit_definition'?'selected':''}>Explicit Board Definition JSON</option></select></div>
        <div class="field"><label>Board definition JSON</label><div class="inline-input"><input id="cal-board-def" value="${esc(draft.boardDefinition||'')}" placeholder="Only for explicit board mode"><button class="input-action" data-browse="cal-board-def">…</button></div></div>
        <div class="field"><label>Camera intrinsics JSON</label><div class="inline-input"><input id="cal-intrinsics" value="${esc(draft.intrinsics||'')}" placeholder="Legacy mode reads calibration.bin"><button class="input-action" data-browse="cal-intrinsics">…</button></div></div>
        <div class="form-row"><div class="field"><label>Sample every N video frames</label><input id="cal-sample-every" type="number" value="${esc(draft.sampleEvery||'2')}" min="1" max="60"></div><div class="field"><label>Maximum detected tag frames</label><input id="cal-max-detections" type="number" value="${esc(draft.maxDetections||'400')}" min="20" max="2000"></div></div>
        <div class="field"><label>Minimum tracker identity confidence</label><select id="cal-identity-confidence"><option value="MEDIUM" ${(draft.identityConfidence||'MEDIUM')==='MEDIUM'?'selected':''}>MEDIUM (recommended)</option><option value="HIGH" ${draft.identityConfidence==='HIGH'?'selected':''}>HIGH</option><option value="SINGLE_CANDIDATE" ${draft.identityConfidence==='SINGLE_CANDIDATE'?'selected':''}>Allow one candidate</option><option value="LOW" ${draft.identityConfidence==='LOW'?'selected':''}>Allow LOW for pilot only</option></select></div>
        <div class="toggle-row"><span>Include raw wearer/subject identity metadata</span><button class="toggle ${draft.rawIdentity?'on':''}" id="cal-raw-identity" data-on="${String(Boolean(draft.rawIdentity))}"></button></div>
        <div class="toggle-row"><span>Auto-register selected tracker as ETM-A / ETM-B</span><button class="toggle ${draft.autoRegister===false?'':'on'}" id="cal-auto-register" data-on="${String(draft.autoRegister!==false)}"></button></div>
        <div class="toggle-row"><span>Auto-name profile from selected hardware</span><button class="toggle ${draft.autoName===false?'':'on'}" id="cal-auto-name" data-on="${String(draft.autoName!==false)}"></button></div>
        <div class="field"><label>Manual Hardware ID override</label><input id="cal-hardware" value="${esc(draft.hardwareId||'')}" placeholder="Leave blank for automatic ETM-A / ETM-B"></div>
        <div class="toggle-row"><span>Allow manual Hardware ID override</span><button class="toggle ${draft.hardwareOverride?'on':''}" id="cal-hardware-override" data-on="${String(Boolean(draft.hardwareOverride))}"></button></div>
        <div class="field"><label>Profile name (optional)</label><input id="cal-name" value="${esc(draft.profileName||'')}" placeholder="Automatic from hardware and sessions"></div>
      </div></details>
      <div class="toggle-row"><span>Also write portable .g3dcal</span><button class="toggle ${draft.outputG3dcal===false?'':'on'}" id="cal-output-g3dcal" data-on="${String(draft.outputG3dcal!==false)}"></button></div>
      <button class="primary-button full" id="run-calibration" ${!calibrationNative.length||!boardCandidates.length||!trackerCandidates.length?'disabled':''}>Estimate ${esc(targetHardware||'hardware')} profile</button>
    </div></details>

    <div class="field"><label>Profile shown in live preview</label><select id="cal-preview-profile">${optionRows(profiles,previewProfile,'Raw C3D only · no camera transform')}</select></div>

    <details class="calibration-secondary-route"><summary>Other ways to obtain a profile</summary><div class="property-stack calibration-route-body">
      <div class="calibration-route-subsection"><strong>Import a completed profile</strong><p class="form-help">Use an existing <code>.g3dcal</code> or <code>.oculocal</code>.</p>
        <div class="field"><label>Profile file</label><div class="inline-input"><input id="cal-import-path" placeholder=".g3dcal / .oculocal"><button class="input-action" data-browse="cal-import-path">…</button></div></div>
        <div class="field"><label>Display label</label><input id="cal-import-label" placeholder="ETM-A_mount_v1"></div>
        <div class="form-row"><div class="field"><label>Hardware ID</label><input id="cal-import-hardware" placeholder="ETM-A"></div><div class="field"><label>Mount version</label><input id="cal-import-mount" value="mount_v1"></div></div>
        <div class="field"><label>Valid sessions</label><input id="cal-import-sessions" value="${esc(defaultValidSessions||sessionLabel)}" placeholder="S001,S002,S003"></div>
        <div class="field"><label>Source note</label><textarea id="cal-import-note" placeholder="Original calibration recording or hardware history"></textarea></div>
        <button class="soft-button full" id="run-calibration-import">Import & register profile</button>
      </div>
      <div class="calibration-route-subsection"><strong>Register an older measured result</strong>
        <div class="field"><label>Unregistered measured profile</label><select id="cal-register-measured-profile">${optionRows(unregisteredMeasured,'','Select older measured result')}</select></div>
        <div class="form-row"><div class="field"><label>Mount version</label><input id="cal-register-measured-mount" value="mount_v1"></div><div class="field"><label>Valid sessions</label><input id="cal-register-measured-sessions" value="${esc(defaultValidSessions||sessionLabel)}"></div></div>
        <button class="soft-button full" id="run-calibration-register-measured" ${!unregisteredMeasured.length?'disabled':''}>Register selected measured result</button>
      </div>
      <div class="calibration-route-subsection"><strong>Transfer a compatible historical profile</strong><p class="form-help">Use only when no direct calibration recording/profile exists for this hardware session.</p>
        <div class="field"><label>Source hardware profile</label><select id="cal-transfer-profile">${optionRows(profiles,'','Select profile')}</select></div>
        <div class="field"><label>Current ETM motion</label><select id="cal-transfer-tracker">${optionRows(trackers.length?trackers:mocap,'','Select ETM C3D/NPZ')}</select></div>
        <div class="form-row"><div class="field"><label>Interaction</label><input id="cal-transfer-interaction" value="${esc(ctx.interactionId||'I01')}"></div><div class="field"><label>Wearer role</label><input id="cal-transfer-role" value="${esc(focusedRole)}"></div></div>
        <details><summary>Transfer gates</summary><div class="property-stack calibration-route-body"><div class="field"><label>Maximum tracker-shape RMS (mm)</label><input id="cal-transfer-max-rms" type="number" value="5" min="0.1" step="0.1"></div><div class="field"><label>High-confidence RMS (mm)</label><input id="cal-transfer-high-rms" type="number" value="2" min="0.1" step="0.1"></div><div class="field"><label>Minimum marker valid fraction</label><input id="cal-transfer-valid" type="number" value="0.5" min="0" max="1" step="0.05"></div></div></details>
        <div class="field"><label>Transferred profile label</label><input id="cal-transfer-label" placeholder="ETM-A · validated transfer"></div>
        <button class="soft-button full" id="run-calibration-transfer" ${!profiles.length||!(trackers.length||mocap.length)?'disabled':''}>Validate & create transferred profile</button>
      </div>
    </div></details>
  </div></section>
  <section class="property-section calibration-phase calibration-phase-c" id="calibration-phase-c"><div class="section-label">3 · Synchronize Pupil IMU ↔ Vicon clocks</div><div class="property-stack">
    ${calibrationSyncPhaseMarkup(stage)}
  </div></section>
  <section class="property-section calibration-phase calibration-phase-d" id="calibration-phase-d"><div class="section-label">4 · Validate profiles and create Gaze 3D handoff</div><div class="property-stack">
    <div class="form-help">Wearer resolution is fixed. This step binds one profile and one production-ready affine time map to each assigned ETM, then creates the immutable handoff used by Gaze 3D.</div>
    <div class="stat-list"><div class="stat-row"><span>Wearer assignment</span><strong>${esc(stage.wearerAssignment?.label||'Not ready')}</strong></div><div class="stat-row"><span>ETM-A profile</span><strong>${esc(profileA?.label||'Missing')}</strong></div><div class="stat-row"><span>ETM-B profile</span><strong>${esc(profileB?.label||'Missing')}</strong></div><div class="stat-row"><span>R01 time sync</span><strong>${stage.rows.find(row=>row.role==='R01')?.timeSyncReady?'Verified':'Missing'}</strong></div><div class="stat-row"><span>R02 time sync</span><strong>${stage.rows.find(row=>row.role==='R02')?.timeSyncReady?'Verified':'Missing'}</strong></div></div>
    <input type="hidden" id="cal-binding-assignment" value="${esc(stage.wearerAssignment?.id||'')}">
    <details><summary>Manual profile selection and binding gates</summary><div class="property-stack calibration-route-body"><div class="check-list">${bindingProfileRows}</div><div class="field"><label>Maximum tracker-shape RMS (mm)</label><input id="cal-binding-max-rms" type="number" value="8" min="0.1" step="0.1"></div><div class="field"><label>Minimum shape-separation margin (mm)</label><input id="cal-binding-margin" type="number" value="4" min="0" step="0.1"></div></div></details>
    <div class="field"><label>Calibration handoff label</label><input id="cal-binding-label" value="${esc(handoffDefaultLabel)}"></div>
    <button class="primary-button full" id="run-calibration-binding" ${bindingReady?'':'disabled'}>Validate profiles + time maps & create Gaze 3D handoff</button>
    ${stage.handoff?`<div class="success-card compact"><strong>Current handoff</strong><p>${esc(stage.handoff.label)} · ${stage.handoff.qc?.handoff_complete?'complete':'review required'} · IMU↔ETM sync ${stage.handoff.qc?.time_sync_complete?'verified':'missing'}</p></div>`:`<div class="form-help">${bindingReady?'Both hardware profiles and both role-specific time maps are ready. Create the handoff when ready.':!stage.synchronizationReady?'Complete automatic synchronization for R01 and R02 first.':'A resolved wearer assignment plus ETM-A and ETM-B profiles are required.'}</div>`}
    ${transferSection}
  </div></section>
  <section class="property-section calibration-phase calibration-phase-e" id="calibration-phase-e"><div class="section-label">5 · Create C3D-defined Pupil clips</div><div class="property-stack">
    ${calibrationC3DClipPhaseMarkup(stage)}
  </div></section>`;
}
function gaze3dProperties() {
  const role=calibrationContextRole();
  const stage=calibrationWorkflowState(role);
  const syncRequired=Boolean(stage.handoff?.parameters?.require_time_sync||stage.handoff?.qc?.time_sync_required);
  const cloud=rowsForActiveWorkflowContext(assets(['pupil_cloud']));
  const mocap=assets(['c3d','mocap_npz']);
  const profiles=artifacts(['calibration_profile']);
  const acceptances=artifacts(['calibration_acceptance']).filter(item=>String(item.parameters?.wearer_role_id||'').toUpperCase()===role);
  const assignments=stage.assignments;
  const cs=clips();
  const trims=rowsForActiveWorkflowContext(artifacts(['native_reference_trim']).slice().sort(byCreated));
  const latestAssignment=stage.assignment?.id||'';
  const automaticTrims=trims.filter(row=>row.parameters?.window_source==='common_body_c3d_overlap');
  const latestTrim=(automaticTrims.slice(-1)[0]||trims.slice(-1)[0])?.id||'';
  const defaultScope=trims.length?'native_trim':'full';
  const current=stage.current;
  const currentProfile=current?.profile;
  const currentAcceptance=current?.acceptance;
  const handoffText=current?.ready
    ?`${role} → ${current.mapping?.etm_id||'ETM'} · ${currentProfile?.label||'profile'}`
    :`${role} calibration handoff is incomplete`;
  const preview=gazePreviewSelection();
  const r01Rows=preview.byRole.R01||[],r02Rows=preview.byRole.R02||[];
  const previewHelp=preview.both
    ?'Dyadic mode aligns the latest compatible R01/R02 ray artifacts on shared Native-trim local time. It is visual QC only and never modifies either artifact.'
    :'Generate both R01 and R02 calibrated rays to enable the shared-time dual-ray preview.';
  return `<section class="property-section calibration-gaze-handoff"><div class="section-label">Calibration handoff · required</div><div class="property-stack">
    <div class="stat-list"><div class="stat-row"><span>Current role</span><strong>${esc(role)}</strong></div><div class="stat-row"><span>Wearer assignment</span><strong>${esc(current?.mapping?.etm_id||'Not resolved')}</strong></div><div class="stat-row"><span>Hardware profile</span><strong>${esc(currentProfile?.label||'Not bound')}</strong></div><div class="stat-row"><span>Status</span><strong>${current?.ready?'Ready':'Blocked'}</strong></div></div>
    <div class="form-help">${esc(handoffText)}. Wearer resolution and profile binding are managed in Calibration, not recomputed in Gaze 3D.</div>
    <button class="soft-button full" data-go="calibration">Open Calibration</button>
  </div></section>
  <section class="property-section gaze-preview-settings"><div class="section-label">0 · Preview processed rays</div><div class="property-stack">
    <div class="field"><label>Preview mode</label><select id="gaze-preview-mode"><option value="single" ${preview.mode==='single'?'selected':''}>Single role</option><option value="dyadic" ${preview.mode==='dyadic'?'selected':''} ${!preview.both?'disabled':''}>Dyadic dual-ray · shared local time</option></select></div>
    <div class="field"><label>R01 ray artifact</label><select id="gaze-preview-r01">${optionRows(r01Rows,preview.r01?.id||'','No R01 gaze rays')}</select></div>
    <div class="field"><label>R02 ray artifact</label><select id="gaze-preview-r02">${optionRows(r02Rows,preview.r02?.id||'','No R02 gaze rays')}</select></div>
    <div class="field"><label>Pairing tolerance (ms)</label><input id="gaze-preview-tolerance" type="number" min="1" max="500" step="1" value="${esc(appState.gazePreviewToleranceMs||'')}" placeholder="Adaptive · recommended"></div>
    <div class="form-help">${esc(previewHelp)}</div>
  </div></section>
  <section class="property-section"><div class="section-label">1 · Generate world-space gaze rays</div><div class="property-stack">
    <div class="field"><label>Timeseries data · ${esc(role)}</label><select id="ray-cloud">${optionRows(cloud,'','Select Timeseries')}</select></div>
    <div class="field"><label>Calibration handoff artifact</label><select id="ray-assignment">${optionRows(assignments,latestAssignment,'Complete Calibration first')}</select></div>
    <input id="ray-role" type="hidden" value="${esc(role)}">
    <details><summary>Emergency manual override · legacy/recovery only</summary><div class="property-stack" style="margin-top:8px">
      <div class="field"><label>Interaction tracker motion</label><select id="ray-tracker">${optionRows(mocap,'','Resolved from Calibration assignment')}</select></div>
      <div class="field"><label>Calibration artifact</label><select id="ray-profile-artifact">${optionRows(profiles,currentProfile?.id||'','Resolved from Calibration assignment')}</select></div>
      <div class="field"><label>External .oculocal / .g3dcal</label><div class="inline-input"><input id="ray-profile-path" placeholder="Recovery only"><button class="input-action" data-browse="ray-profile-path">…</button></div></div>
    </div></details>
    <div class="field"><label>Scope</label><select id="ray-scope"><option value="native_trim" ${defaultScope==='native_trim'?'selected':''}>Native-reference trim · pre-trimmed interaction</option><option value="full" ${defaultScope==='full'?'selected':''}>Full recording · TimeGraph</option><option value="clip">Legacy Global-Time clip</option></select></div>
    <div class="field" id="ray-native-trim-row"><label>Native-reference trim</label><select id="ray-native-trim">${optionRows(trims,latestTrim,`Select ${role} retained interval`)}</select></div>
    <div class="field" id="ray-alignment-row"><label>Time alignment</label><select id="ray-alignment-mode"><option value="timegraph" ${syncRequired?'selected':''}>Calibration handoff · verified affine TimeGraph</option><option value="pretrimmed_shared_start" ${syncRequired?'disabled':'selected'}>Pre-trimmed shared start · diagnostic legacy mode</option></select></div>
    <details id="ray-shared-start-settings" class="shared-start-settings" ${syncRequired?'hidden':'open'}><summary>Pre-trimmed shared-start settings</summary><div class="property-stack" style="margin-top:8px">
      <div class="form-help"><strong>Use this when the selected Native trim, ETM C3D and body C3D were already cut to the same physical interval.</strong> OculoStudio maps <code>oculo_clip_video_ns = 0</code> to the first C3D frame and preserves original Pupil row IDs.</div>
      <div class="field"><label>Known residual start offset (s)</label><input id="ray-shared-start-offset" type="number" value="0" min="-5" max="5" step="0.001"></div>
    </div></details>
    <div id="ray-timegraph-settings">
      ${syncRequired?'<div class="success-card compact"><strong>Calibration time map locked</strong><p>The selected handoff supplies the accepted Pupil IMU ↔ assigned ETM affine mapping. Ego video was not used for synchronization.</p></div>':''}
      <div class="field"><label>Missing clock path</label><select id="ray-sync-policy" ${syncRequired?'disabled':''}><option value="auto" ${syncRequired?'':'selected'}>Auto-sync Pupil IMU ↔ ETM motion</option><option value="manual" ${syncRequired?'selected':''}>Require Calibration handoff map</option></select></div>
      <details class="auto-sync-settings"><summary>Automatic synchronization settings</summary><div class="property-stack" style="margin-top:8px"><div class="field"><label>Maximum offset search (s)</label><input id="ray-sync-max-offset" type="number" value="120" min="1" max="3600" step="1"></div></div></details>
    </div>
    <div class="field" id="ray-legacy-clip-row"><label>Legacy Global-Time clip</label><select id="ray-clip">${optionRows(cs,appState.selectedClipId,'Select clip when scope=clip')}</select></div>
    <div class="field"><label>Transferred-profile acceptance</label><select id="ray-calibration-acceptance">${optionRows(acceptances,currentAcceptance?.id||'',current?.transferred?'Accepted transfer required':'Not required for measured/imported profile')}</select></div>
    <button class="toggle" id="ray-require-acceptance" data-on="${String(Boolean(current?.transferred))}" hidden></button>
    <div class="field"><label>Maximum pose interpolation gap (s)</label><input id="ray-gap" type="number" value="0.05" min="0" step="0.01"></div>
    <div class="toggle-row"><span>Also write Parquet</span><button class="toggle" id="ray-parquet" data-on="false"></button></div>
    <div class="form-help" id="ray-scope-help">Native-reference trim mode generates rays only for retained interaction rows and reports trim validity as the primary QC metric.</div>
    <button class="primary-button full" id="run-gaze3d" ${!current?.ready||!cloud.length||!trims.length?'disabled':''}>Generate calibrated rays</button>
    ${!current?.ready?'<div class="warning-card compact"><strong>Blocked by Calibration</strong><p>Complete wearer assignment, bind the hardware profile, and verify Pupil IMU ↔ ETM synchronization for this role before Gaze 3D.</p></div>':''}
  </div></section>`;
}

function targetsProperties() {
  const rayArtifacts=artifacts(['gaze_rays']), cloud=assets(['pupil_cloud']);
  const mocap=assets(['c3d','mocap_npz']), densepose=assets(['densepose_export']);
  const trims=artifacts(['native_reference_trim']), privacyClips=artifacts(['privacy_clip']), fmFilters=artifacts(['face_mapper_filter']);
  const assignments=calibrationHandoffArtifacts(), profiles=artifacts(['calibration_profile']);
  const associations=artifacts(['participant_association']);
  const roleSet=new Set();assignments.forEach(a=>Object.keys(a.qc?.assignments_summary||{}).forEach(role=>roleSet.add(role)));
  if(!roleSet.size){roleSet.add('R01');roleSet.add('R02');}
  let roleOptions=[...roleSet].sort().map(role=>`<option value="${esc(role)}">${esc(role)}</option>`).join('');
  const faceAOI=artifacts(['face_aoi']), bodyAOI=artifacts(['body_aoi']), objectAOI=artifacts(['object_aoi']);
  const targetContext=activeWorkflowContext();
  const bodyRole=String(targetContext.roleId||calibrationContextRole()||'R01').toUpperCase();
  const scopedAssignments=rowsForActiveWorkflowContext(assignments,{role:false,allowUnscoped:true}).slice().sort(byCreated);
  const bodyAssignment=scopedAssignments.slice(-1)[0]||assignments.slice(-1)[0]||null;
  const bodySummary=bodyAssignment?.qc?.assignments_summary||{};
  const wearerSummary=bodySummary[bodyRole]||null;
  const partnerRole=Object.keys(bodySummary).find(role=>String(role).toUpperCase()!==bodyRole)||'';
  const partnerSummary=partnerRole?bodySummary[partnerRole]:null;
  const wearerBodyId=wearerSummary?.body_asset_id||'';
  const partnerBodyId=partnerSummary?.body_asset_id||'';
  const roleRays=rayArtifacts.filter(row=>gazeArtifactRole(row)===bodyRole).slice().sort(byCreated);
  const bodyRayDefault=roleRays.slice(-1)[0]?.id||'';
  const scopedTrims=rowsForActiveWorkflowContext(trims,{fallback:false}).slice().sort(byCreated);
  const bodyTrimDefault=selectedTargetTrimId(scopedTrims);
  const roleCloud=cloud.filter(row=>assetIdentity(row).role_id===bodyRole||codeFromText(`${row.label} ${row.source_path}`,'R')===bodyRole);
  const bodyCloudDefault=roleCloud.slice(-1)[0]?.id||'';
  const bodyLabel=[targetContext.interactionId||'interaction',bodyRole,'body_aoi'].filter(Boolean).join('_');
  roleOptions=[...roleSet].sort().map(role=>`<option value="${esc(role)}" ${String(role).toUpperCase()===bodyRole?'selected':''}>${esc(role)}</option>`).join('');
  const scopedFilters=rowsForActiveWorkflowContext(fmFilters,{fallback:false});
  const scopedPrivacy=rowsForActiveWorkflowContext(privacyClips,{fallback:false});
  const scopedAssociations=rowsForActiveWorkflowContext(associations,{fallback:false});
  const compatibleFilters=scopedFilters.filter(row=>recordUsesNativeTrim(row,bodyTrimDefault));
  const compatiblePrivacy=scopedPrivacy.filter(row=>recordUsesNativeTrim(row,bodyTrimDefault));
  const compatibleAssociations=scopedAssociations.filter(row=>recordUsesNativeTrim(row,bodyTrimDefault)&&(!wearerRoleForRecord(row)||wearerRoleForRecord(row)===bodyRole));
  const recommendedProfileId=wearerSummary?.recommended_calibration_artifact_id||'';
  const recommendedProfile=profiles.find(row=>row.id===recommendedProfileId)||null;
  const resolvedEtmId=wearerSummary?.etm_id||wearerSummary?.etm_asset_label||'Not resolved';
  const scopedFaceAOI=targetRowsForActiveContext(faceAOI,{trimId:bodyTrimDefault,requireTrim:Boolean(bodyTrimDefault)}).slice().sort(byCreated);
  const scopedBodyAOI=targetRowsForActiveContext(bodyAOI,{trimId:bodyTrimDefault,requireTrim:Boolean(bodyTrimDefault)}).slice().sort(byCreated);
  const scopedObjectAOI=targetRowsForActiveContext(objectAOI,{trimId:bodyTrimDefault,requireTrim:Boolean(bodyTrimDefault)}).slice().sort(byCreated);
  const reviewFaceCandidates=scopedFaceAOI.slice();
  const reviewFace=[...reviewFaceCandidates].reverse().find(row=>Number(row.qc?.review_samples_remaining??row.qc?.manual_review_pending_sample_count??0)>0||Number(row.qc?.review_required_fraction||0)>0)||reviewFaceCandidates.slice(-1)[0]||null;
  const reviewPending=Number(reviewFace?.qc?.review_samples_remaining??reviewFace?.qc?.manual_review_pending_sample_count??0);
  const reviewBodyCandidates=scopedBodyAOI.slice();
  const reviewReadyBodyCandidates=reviewBodyCandidates.filter(bodyArtifactReviewReady);
  const newestBlockedBody=[...reviewBodyCandidates].reverse().find(row=>!bodyArtifactReviewReady(row))||null;
  const fusionBodyCandidates=scopedBodyAOI.filter(bodyArtifactFusionReady).slice().sort(byCreated);
  const bodyPendingFor=row=>{
    const qc=row?.qc||{};
    for(const value of [qc.manual_review_pending_sample_count,qc.review_required_count,qc.review_samples_remaining]){
      if(value===null||value===undefined||value==='')continue;
      const parsed=Number(value);if(Number.isFinite(parsed)&&parsed>=0)return parsed;
    }
    const rows=Number(qc.output_rows||qc.input_gaze_rows||0),fraction=Number(qc.review_required_fraction);
    return Number.isFinite(rows)&&rows>=0&&Number.isFinite(fraction)&&fraction>=0?Math.round(rows*fraction):0;
  };
  const bodyPendingUnitsFor=row=>{
    const qc=row?.qc||{};
    for(const value of [qc.manual_review_pending_review_unit_count,qc.review_units_remaining,qc.review_fixations_remaining]){
      if(value===null||value===undefined||value==='')continue;
      const parsed=Number(value);if(Number.isFinite(parsed)&&parsed>=0)return parsed;
    }
    return bodyPendingFor(row);
  };
  const reviewBody=[...reviewReadyBodyCandidates].reverse().find(row=>bodyPendingFor(row)>0)||reviewReadyBodyCandidates.slice(-1)[0]||null;
  const reviewBodyPending=bodyPendingFor(reviewBody);
  const reviewBodyPendingUnits=bodyPendingUnitsFor(reviewBody);
  const humanReviewSection=`<section class="property-section human-review-property-entry"><div class="section-label">Human review</div><div class="human-review-property-grid">
    <div class="human-review-property-card face">${reviewFace?`<div><strong>Face AOI · ${esc(bodyRole)}</strong><small>${reviewPending>0?`${reviewPending} flagged samples pending`:'Queue ready'}</small></div><button class="primary-button full" id="open-face-human-review-side" data-artifact-id="${esc(reviewFace.id)}">${reviewPending>0?'Review flagged Face AOI':'Open Face AOI review'}</button>`:`<div><strong>Face AOI · ${esc(bodyRole)}</strong><small>No artifact for the active role and retained trim</small></div><button class="primary-button full" disabled>Generate ${esc(bodyRole)} Face AOI first</button>`}<p>Only the active interaction, role and exact Native-reference trim can open this queue. Decisions are saved as a draft, then finalized into a new immutable Face AOI.</p></div>
    <div class="human-review-property-card body">${reviewBody?`<div><strong>Body AOI · ${esc(bodyRole)}</strong><small id="body-human-review-side-count">${reviewBodyPending>0?`${reviewBodyPendingUnits} review events · ${reviewBodyPending} samples`:'Queue ready'}</small></div><button class="primary-button full" id="open-body-human-review-side" data-artifact-id="${esc(reviewBody.id)}">${reviewBodyPending>0?'Review flagged Body AOI':'Open Body AOI review'}</button>`:newestBlockedBody?`<div><strong>Body AOI · ${esc(bodyRole)} blocked</strong><small>Torso/body coverage did not pass acceptance</small></div><button class="primary-button full" disabled>Regenerate ${esc(bodyRole)} Body AOI</button>`:`<div><strong>Body AOI · ${esc(bodyRole)}</strong><small>No artifact for the active role and retained trim</small></div><button class="primary-button full" disabled>Generate ${esc(bodyRole)} Body AOI first</button>`}<p>${reviewBody?'Only the current role/trim queue is available. Verify self/partner/other and body region.':newestBlockedBody?'The invalid current-role artifact remains diagnostic only; manual review and fusion are disabled.':'Generate a torso-verified Body AOI for this exact role/trim first.'}</p>${reviewBody&&reviewBodyPending>0?`<div class="body-review-retune"><div class="field"><label>Reduce this existing queue</label><select id="body-review-retune-policy"><option value="semantic_guided" selected>Semantic coarse-target consensus · recommended</option><option value="evidence_guided">Evidence-guided fixation consensus</option><option value="conservative">Conservative</option><option value="adaptive">Adaptive v1</option><option value="balanced">Balanced · largest reduction</option></select></div><button class="soft-button full" id="reduce-existing-body-review" data-artifact-id="${esc(reviewBody.id)}">Re-evaluate saved evidence</button><small>Creates a new immutable Body AOI without rerunning C3D or ray intersection. It reuses the existing gaze-sample intersections and never changes the automatic person or body-region label. Run it before saving manual decisions.</small></div>`:''}</div>
  </div></section>`;
  return `${humanReviewSection}<section class="property-section"><div class="section-label">0 · Participant association</div><div class="property-stack">
    <div class="success-card compact role-locked-projection-card"><strong>Role-locked partner-head projection</strong><p>${esc(bodyRole)} → ${esc(resolvedEtmId)} → ${esc(recommendedProfile?.label||'Calibration profile missing')}. The paired Calibration handoff is authoritative; switching the active role cannot reuse the other wearer’s profile.</p></div>
    <div class="field"><label>Native-reference trim</label><select id="association-trim">${optionRows(scopedTrims.length?scopedTrims:trims,bodyTrimDefault,'Select trim')}</select></div>
    <div class="field"><label>Time alignment</label><select id="association-alignment"><option value="pretrimmed_shared_start" selected>Pre-trimmed shared start · local time zero</option><option value="timegraph">Verified TimeGraph mapping</option></select></div>
    <div class="field"><label>Filtered FaceMapper · exact same trim</label><select id="association-filter">${optionRows(compatibleFilters,'','Use trim raw FaceMapper')}</select><small>Only filters descended from the selected Native-reference trim are listed.</small></div>
    <div class="field"><label>Paired Calibration handoff</label><select id="association-etm">${optionRows(scopedAssignments.length?scopedAssignments:assignments,bodyAssignment?.id||'','Select assignment')}</select></div>
    <input id="association-wearer" type="hidden" value="${esc(bodyRole)}">
    <input id="association-profile" type="hidden" value="">
    <details class="advanced-section"><summary>Emergency calibration-profile override · recovery only</summary><div class="property-stack" style="margin-top:8px"><div class="form-help">Normally leave this disabled. A role-mismatched override is rejected by the backend even when selected manually.</div><div class="toggle-row"><span>Enable manual profile override</span><button class="toggle" id="association-profile-override-enabled" data-on="false"></button></div><div class="field"><label>Override profile</label><select id="association-profile-override">${optionRows(profiles,'','No override')}</select></div></div></details>
    <div class="field"><label>Target policy</label><select id="association-policy"><option value="partner_only" selected>Partner only · recommended</option><option value="tracked_participants">Recorded participants</option><option value="all_detected">All faces</option></select></div>
    <div class="field"><label>Face tolerance (ms)</label><input id="association-face-tolerance" type="number" value="8"></div><input id="association-centre-error" type="hidden" value="1.25"><input id="association-margin" type="hidden" value="0.12"><input id="association-confidence" type="hidden" value="0.42">
    <div class="field"><label>Output label</label><input id="association-label" value="${esc([targetContext.interactionId||'interaction',bodyRole,'participant_association'].join('_'))}"></div><button class="toggle" id="association-parquet" data-on="false" hidden></button>
    ${recommendedProfile?'<button class="primary-button full" id="run-participant-association">Project heads & assign faces</button>':'<button class="primary-button full" disabled>Complete paired Calibration profile binding</button>'}
    <small>The light-blue projected partner-head box is an identity gate only. A partner-only job now fails instead of publishing when it produces zero valid in-view partner projections.</small>
  </div></section>
  <section class="property-section"><div class="section-label">1 · Face AOI</div><div class="property-stack">
    <div class="field"><label>Trimmed TSD + FM</label><select id="face-aoi-trim">${optionRows(scopedTrims.length?scopedTrims:trims,bodyTrimDefault,'Select trim')}</select></div>
    <div class="field"><label>Filtered FaceMapper · exact same trim</label><select id="face-aoi-filter">${optionRows(compatibleFilters,compatibleFilters.slice(-1)[0]?.id||'','Use trim raw FaceMapper')}</select></div>
    <div class="field"><label>Participant association · exact same trim</label><select id="face-aoi-association">${optionRows(compatibleAssociations,compatibleAssociations.slice(-1)[0]?.id||'','Run participant association for this trim')}</select><small>${compatibleAssociations.length?'Only role- and trim-compatible associations are listed.':'No compatible participant association exists for this role and trim. Run step 0 again.'}</small></div>
    <div class="field"><label>Person policy</label><select id="face-aoi-person-policy"><option value="partner_only" selected>Partner only · recommended</option><option value="tracked_participants">Recorded participants</option><option value="all_detected">All faces</option></select></div>
    <div class="field"><label>Trimmed blurred video · exact same trim</label><select id="face-aoi-privacy">${optionRows(compatiblePrivacy,compatiblePrivacy.slice(-1)[0]?.id||'','Tables only')}</select></div>
    <div class="field"><label>Output label</label><input id="face-aoi-label" value="${esc([targetContext.interactionId||'interaction',bodyRole,'face_aoi'].join('_'))}"></div>
    <div class="field"><label>Face tolerance (ms)</label><input id="face-aoi-tolerance" type="number" value="8"></div>
    <div class="field"><label>Face geometry time model</label><select id="face-aoi-time-policy"><option value="track_interpolated" selected>Track-aware gaze-time interpolation · recommended</option><option value="nearest_frame">Nearest observed video frame · legacy</option></select><small>FaceMapper remains observed at the native video FPS. Recommended mode safely interpolates the same accepted face track to each gaze timestamp; it never invents a new face identity or promotes gaze outside the green face box.</small></div>
    <div class="form-row"><div class="field"><label>Maximum interpolation gap (ms)</label><input id="face-aoi-interpolation-gap" type="number" value="75" min="10" max="500" step="1"></div><div class="field"><label>Maximum normalized face motion</label><input id="face-aoi-interpolation-motion" type="number" value="1.25" min="0.25" max="5" step="0.05"></div></div>
    <div class="field"><label>Fine facial AOI size</label><select id="face-aoi-feature-scale"><option value="1.35">Balanced · 1.35×</option><option value="1.55" selected>Large · recommended</option><option value="1.75">Maximum bounded · 1.75×</option></select><small>Controls landmark-derived eye and nose expansion. Mouth size is governed by its own model, so enlarging the eyes cannot silently enlarge the mouth.</small></div>
    <div class="field"><label>Eye-region expansion</label><select id="face-aoi-eye-scale"><option value="1.15">Moderate eye expansion</option><option value="1.35" selected>Large eye regions · recommended</option><option value="1.50">Extra-large eye regions</option></select><small>This multiplier is applied on top of the facial AOI size, making both eye AOIs substantially larger while preserving left/right labels and face-box containment.</small></div>
    <div class="field"><label>Nose geometry</label><select id="face-aoi-nose-mode"><option value="nasal_envelope" selected>Nasal bridge + both alae · recommended</option><option value="bridge_polygon">Legacy eye-line bridge polygon</option><option value="ellipse">Legacy nose-landmark ellipse</option></select><small>The recommended single polygon is narrow below the inner-eye line, widens through the nose point, and is widest across the nostril/ala region. It avoids the legacy hourglass/8-shape and remains inside the green FaceMapper box.</small></div>
    <div class="field"><label>Mouth AOI model</label><select id="face-aoi-mouth-center"><option value="adaptive_lip_envelope" selected>Balanced lips · recommended</option><option value="anatomical_strict">Anatomical strict · compact</option><option value="speech_sensitive_mouth_jaw">Speech-sensitive mouth/jaw · broad</option><option value="lowered_landmark_midpoint">Legacy lowered ellipse</option><option value="landmark_midpoint">Legacy mouth-corner ellipse</option></select><small>Balanced mode expands only 15% beyond the mouth corners, stays at or below 18% of face height (hard cap 22%), and stops above the chin. The remaining lower face is other_face. The broad mouth/jaw mode is retained only for speech-reading studies.</small></div>
    <div class="field"><label>Green face-box edge tolerance</label><select id="face-aoi-edge-tolerance"><option value="edge_inclusive" selected>Edge-inclusive · recommended</option><option value="exact">Exact green box only</option><option value="forgiving">Wider uncertainty band</option></select><small>The drawn green FaceMapper box remains the face reference. Edge-inclusive mode counts gaze centres just outside its border as coarse partner face (other_face): max(6 px, 4.5% of the smaller face-box dimension), capped at 18 px. Eye, nose and mouth AOIs are not enlarged. Exact no-face samples farther outside remain no_face.</small></div>
    <div class="field"><label>Automatic review reduction</label><select id="face-aoi-auto-review"><option value="adaptive" selected>Adaptive evidence fusion · recommended</option><option value="conservative">Conservative · highest precision</option><option value="balanced">Balanced · maximum review reduction</option><option value="off">Off · flag every uncertain sample</option></select><small>Any gaze centre directly inside an identity-resolved green FaceMapper box is accepted automatically. The light-blue participant projection assigns identity only. Edge-band hits and genuinely ambiguous identities remain reviewable.</small></div>
    <div class="form-help">A new Face AOI artifact keeps Native video frames as observed evidence and exports full-clip partner-face, eyes, nose and mouth proportions. Existing artifacts remain unchanged.</div><div class="success-card compact"><strong>Canonical FaceMapper geometry</strong><p>Uses exact exported pixel coordinates. Only bilateral semantic names are converted; no landmark coordinate is mirrored.</p></div><input id="face-aoi-geometry" type="hidden" value="as_exported">
    <div class="field"><label>Face bilateral semantics</label><select id="face-aoi-lr"><option value="canonical_filter" selected>Canonical anatomical labels from FaceMapper filter · recommended</option><option value="as_exported">Legacy · keep raw source labels</option><option value="swap_eyes">Legacy raw fallback · swap bilateral names once</option></select></div>
    <div class="toggle-row"><span>Render overlay</span><button class="toggle on" id="face-aoi-overlay" data-on="true"></button></div><div class="toggle-row"><span>Write Parquet</span><button class="toggle" id="face-aoi-parquet" data-on="false"></button></div><button class="primary-button full" id="run-face-aoi" ${!trims.length?'disabled':''}>Generate Face AOI</button>
  </div></section>
  <section class="property-section body-aoi-social-workflow"><div class="section-label">2 · Body AOI · self + partner</div><div class="property-stack">
    <div class="success-card compact"><strong>C3D articulated primitives</strong><p>Head, torso, arms, hands and legs are reconstructed from the aligned Vicon markers. Self and partner are intersected together using strict first-hit visibility.</p></div>
    <div class="field"><label>Gaze rays · ${esc(bodyRole)}</label><select id="body-aoi-rays">${optionRows(roleRays.length?roleRays:rayArtifacts,bodyRayDefault,'Select current-role Gaze 3D')}</select></div>
    <div class="field"><label>Native-reference trim</label><select id="body-aoi-trim">${optionRows(scopedTrims.length?scopedTrims:trims,bodyTrimDefault,'Select retained interval')}</select></div>
    <div class="field"><label>Calibration handoff / wearer assignment</label><select id="body-aoi-assignment">${optionRows(scopedAssignments.length?scopedAssignments:assignments,bodyAssignment?.id||'','Select Calibration handoff')}</select></div>
    <div class="success-card compact"><strong>Exact Body AOI clock</strong><p>For a matching C3D-defined trim, Body AOI reuses the per-sample ETM/Vicon C3D time already stored in Gaze 3D. A separate Timeseries → Global Time map is not required. The job still fails closed when the selected body C3D does not overlap that clock.</p></div>
    <div class="stat-list social-body-map"><div class="stat-row"><span>Wearer role</span><strong>${esc(bodyRole)}</strong></div><div class="stat-row"><span>Partner role</span><strong>${esc(partnerRole||'Not resolved')}</strong></div></div>
    <input id="body-aoi-wearer-role" type="hidden" value="${esc(bodyRole)}">
    <div class="field"><label>Wearer body · self</label><select id="body-aoi-wearer-body">${optionRows(mocap,wearerBodyId,'Select wearer Body C3D')}</select></div>
    <div class="field"><label>Partner body · target</label><select id="body-aoi-partner-body">${optionRows(mocap,partnerBodyId,'Select partner Body C3D')}</select></div>
    <div class="toggle-row"><span>Include visible self body</span><button class="toggle on" id="body-aoi-include-self" data-on="true"></button></div>
    <div class="toggle-row"><span>Exclude self head / neck near the ETM origin</span><button class="toggle on" id="body-aoi-exclude-self-head" data-on="true"></button></div>
    <div class="field"><label>Self near-origin exclusion (m)</label><input id="body-aoi-self-origin" type="number" value="0.20" min="0" max="2" step="0.01"></div>
    <div class="toggle-row"><span>Strict first-hit occlusion + second-hit audit</span><button class="toggle on" id="body-aoi-first-hit" data-on="true"></button></div>
    <div class="field"><label>Output granularity</label><select id="body-aoi-granularity"><option value="coarse_social_regions" selected>Coarse social-body regions · recommended</option><option value="detailed_regions">Detailed primitive regions</option></select></div>
    <div class="field"><label>Automatic review reduction</label><select id="body-aoi-auto-review"><option value="conservative">Conservative · smallest reduction</option><option value="adaptive">Adaptive v1 · exact-anchor rules</option><option value="evidence_guided">Evidence-guided fixation consensus</option><option value="semantic_guided" selected>Semantic coarse-target consensus · recommended</option><option value="balanced">Balanced · largest reduction</option><option value="off">Off · review every uncertainty candidate</option></select><small>The recommended policy keeps the automatic target unchanged, recognises raw surfaces that map to the same coarse person/body label, uses nearby exact hits across split fixations, and accepts only target-stable continuous evidence. Self-occlusion, different people, different coarse regions and unresolved competing surfaces stay pending.</small></div>
    <div class="body-geometry-gate-settings">
      <div class="field"><label>Body geometry acceptance</label><select id="body-aoi-geometry-policy"><option value="required" selected>Required · block invalid torso geometry</option><option value="warn">Diagnostic only · continue but disable review/fusion</option><option value="off">Off · developer diagnosis only</option></select><small>Required is recommended. A completed job is not allowed to hide a missing torso.</small></div>
      <div class="form-row"><div class="field"><label>Minimum torso coverage (%)</label><input id="body-aoi-min-torso" type="number" value="80" min="0" max="100" step="1"></div><div class="field"><label>Minimum body timeline coverage (%)</label><input id="body-aoi-min-body" type="number" value="90" min="0" max="100" step="1"></div></div>
    </div>
    <div class="form-help">Main labels are <code>self/partner × head, torso, left/right arm, left/right hand, left/right leg</code>. Detailed labels remain in audit columns. Body geometry is interpolated separately at every gaze-sample timestamp; the approximately 30 FPS scene video is used only for overlay and human-review navigation. Missing central torso markers can be reconstructed from observed shoulders/hips or the body axis with reduced confidence and explicit provenance.</div>
    <div class="field"><label>Timeseries</label><select id="body-aoi-cloud">${optionRows(roleCloud.length?roleCloud:cloud,bodyCloudDefault,'Infer from gaze rays')}</select></div>
    <details class="advanced-section"><summary>Advanced geometry and marker settings</summary><div class="property-stack" style="margin-top:8px">
      <div class="field"><label>Body profile JSON</label><div class="inline-input"><input id="body-aoi-profile"><button class="input-action" data-browse="body-aoi-profile">…</button></div></div>
      <textarea id="body-aoi-prefixes" hidden></textarea>
      <div class="field"><label>Maximum marker interpolation gap (s)</label><input id="body-aoi-gap" type="number" value="0.1" min="0" max="2" step="0.01"></div>
      <div class="field"><label>Uncertainty cone (deg)</label><input id="body-aoi-cone" type="number" value="1.5" min="0" max="20" step="0.1"></div>
      <div class="field"><label>Maximum target distance (m)</label><input id="body-aoi-distance" type="number" value="10" min="0.1" max="100" step="0.1"></div>
      <div class="field"><label>Scene-frame tolerance (ms)</label><input id="body-aoi-frame-tolerance" type="number" value="20" min="1" max="500" step="1"></div>
    </div></details>
    <div class="field"><label>Output label</label><input id="body-aoi-label" value="${esc(bodyLabel)}"></div>
    <button class="toggle" id="body-aoi-parquet" data-on="false" hidden></button>
    <button class="primary-button full" id="run-body-aoi" ${!rayArtifacts.length||!mocap.length||!assignments.length?'disabled':''}>Generate self + partner Body AOI</button>
  </div></section>
  <section class="property-section"><div class="section-label">3 · Object AOI · marker props</div><div class="property-stack"><div class="field"><label>Gaze rays</label><select id="object-aoi-rays">${optionRows(rayArtifacts,'','Select Gaze 3D')}</select></div><div class="field"><label>Native trim</label><select id="object-aoi-trim">${optionRows(trims,'','Full scope')}</select></div><div class="field"><label>Timeseries</label><select id="object-aoi-cloud">${optionRows(cloud,'','Infer')}</select></div><div class="field"><label>Object C3D / mocap</label><div class="check-list">${mocap.length?mocap.map(a=>`<label class="check-row"><input type="checkbox" class="object-aoi-check" value="${esc(a.id)}"><span>${esc(a.label)}</span></label>`).join(''):'<span class="muted small">Import a clipped Object.c3d with labelled prop marker groups.</span>'}</div></div><div class="field"><label>Object profile JSON (optional)</label><div class="inline-input"><input id="object-aoi-profile" placeholder="Auto-discover marker prefixes"><button class="input-action" data-browse="object-aoi-profile">…</button></div></div><div class="field"><label>Object ID overrides JSON</label><textarea id="object-aoi-overrides" placeholder='{"Cup":"cup_01"}'></textarea></div><div class="form-help">Best results use stable instance IDs and physical dimensions in <code>examples/object_profile.example.json</code>.</div><div class="field"><label>Output label</label><input id="object-aoi-label" value="object_aoi"></div><div class="toggle-row"><span>Write Parquet</span><button class="toggle" id="object-aoi-parquet" data-on="false"></button></div><button class="primary-button full" id="run-object-aoi" ${!rayArtifacts.length||!mocap.length?'disabled':''}>Generate Object AOI</button></div></section>
  <section class="property-section"><div class="section-label">4 · Fuse evidence</div><div class="property-stack"><div class="field"><label>Face AOI · exact role/trim</label><select id="fusion-face">${optionRows(scopedFaceAOI,scopedFaceAOI.slice(-1)[0]?.id||'','Generate current-role Face AOI first')}</select></div><div class="field"><label>Body AOI · exact role/trim + geometry accepted</label><select id="fusion-body">${optionRows(fusionBodyCandidates,fusionBodyCandidates.slice(-1)[0]?.id||'','Generate torso-verified current-role Body AOI')}</select><small>${fusionBodyCandidates.length?'Only current-role, current-trim, review/fusion-ready artifacts are listed.':'No current-role Body AOI has passed the torso integrity gate.'}</small></div><div class="field"><label>Object AOI (optional · exact role/trim)</label><select id="fusion-object">${optionRows(scopedObjectAOI,scopedObjectAOI.slice(-1)[0]?.id||'','No current-role object evidence')}</select></div><div class="field"><label>Output label</label><input id="fusion-label" value="${esc([targetContext.interactionId||'interaction',bodyRole,'gaze_targets_full'].join('_'))}"></div><div class="toggle-row"><span>Write Parquet</span><button class="toggle" id="fusion-parquet" data-on="false"></button></div><div class="form-help">Fusion rule: a valid sample labelled no_face by the strict green-box gate and no_body by Body AOI becomes environment. no_face_detection, invalid gaze, or missing body geometry remain unknown. A valid Object AOI hit can still replace environment.</div><button class="primary-button full" id="run-target-fusion" ${!scopedFaceAOI.length||!fusionBodyCandidates.length?'disabled':''}>Fuse full target file</button></div></section>
  <details class="property-section"><summary class="section-label">Legacy combined job</summary><div class="property-stack"><div class="field"><label>Gaze rays</label><select id="target-rays">${optionRows(rayArtifacts,'','Select')}</select></div><div class="field"><label>DensePose</label><select id="target-densepose">${optionRows(densepose,'','None')}</select></div><input id="target-cloud" type="hidden"><input id="target-face" type="hidden"><input id="target-body-profile" type="hidden"><textarea id="target-prefixes" hidden></textarea><input id="target-densepose-profile" type="hidden"><input id="target-frame-tolerance" type="hidden" value="10"><input id="target-face-tolerance" type="hidden" value="8"><input id="target-marker-gap" type="hidden" value="0.1"><input id="target-cone" type="hidden" value="1.5"><input id="target-distance" type="hidden" value="10"><input id="target-frame-policy" type="hidden" value="all_scene_frames"><input id="target-face-lr" type="hidden" value="anatomical_auto"><button class="toggle" id="target-parquet" data-on="false" hidden></button><button class="soft-button full" id="run-semantic-targets" ${!rayArtifacts.length?'disabled':''}>Run previous combined target job</button></div></details>`;
}

function dyadicProperties() {
  const targetRows=artifacts(['target_fusion','face_aoi','body_aoi','object_aoi','semantic_targets']);
  const domainOptions=domains().map(d=>`<option value="${esc(d.id)}">${esc(d.label)} · ${esc(d.clock_type)}</option>`).join('');
  return `<section class="property-section"><div class="section-label">Two-role target streams</div><div class="property-stack">
    <div class="form-row"><div class="field"><label>Role A ID</label><input id="dyadic-role-a" value="R01"></div><div class="field"><label>Role B ID</label><input id="dyadic-role-b" value="R02"></div></div>
    <div class="field"><label>Role A full target artifact</label><select id="dyadic-artifact-a">${optionRows(targetRows,'','Select R01 target file')}</select></div>
    <div class="field"><label>Role B full target artifact</label><select id="dyadic-artifact-b">${optionRows(targetRows,'','Select R02 target file')}</select></div>
  </div></section>
  <section class="property-section"><div class="section-label">Timeline alignment</div><div class="property-stack">
    <div class="field"><label>Alignment mode</label><select id="dyadic-alignment"><option value="pretrimmed_shared_start" selected>Shared Native-trim local time · recommended</option><option value="timegraph">Verified TimeGraph mapping · advanced</option></select></div>
    <div class="form-help">Use the recommended mode when R01/R02 eye data, ETM C3D and body/prop C3D were jointly trimmed to the same physical start. Synchronize is not required.</div>
    <details class="advanced-section"><summary>Advanced · source clocks for raw or mismatched recordings</summary><div class="field"><label>Role A source clock</label><select id="dyadic-domain-a"><option value="">Select clock</option>${domainOptions}</select></div><div class="field"><label>Role B source clock</label><select id="dyadic-domain-b"><option value="">Select clock</option>${domainOptions}</select></div><div class="form-help">Only used when Alignment mode is Verified TimeGraph mapping.</div></details>
  </div></section>
  <section class="property-section"><div class="section-label">Episodes</div><div class="property-stack">
    <div class="field"><label>Interaction FPS</label><input id="dyadic-fps" type="number" value="30" min="1" max="120"></div>
    <div class="field"><label>Cross-wearer tolerance (ms)</label><input id="dyadic-tolerance" type="number" value="25" min="1" step="1"></div>
    <div class="field"><label>Minimum episode (ms)</label><input id="dyadic-min-episode" type="number" value="100" min="0" step="10"></div>
    <div class="field"><label>Bridge short gap (ms)</label><input id="dyadic-gap" type="number" value="50" min="0" step="10"></div>
    <div class="field"><label>Output label</label><input id="dyadic-label" value="dyadic_attention"></div>
    <div class="toggle-row"><span>Write Parquet</span><button class="toggle" id="dyadic-parquet" data-on="false"></button></div>
    <div class="form-help">Mutual gaze uses partner face/head labels. Joint object attention requires the same stable object instance ID in both target streams.</div>
    <button class="primary-button full" id="run-dyadic-attention" ${targetRows.length<2?'disabled':''}>Annotate mutual gaze + joint attention</button>
  </div></section>`;
}

function qcProperties() {
  const runtime=appState.server;
  const rows=resultArtifactRows().slice().reverse();
  const selected=rows.find(a=>a.id===appState.selectedArtifactId)||rows[0]||null;
  const defaults=inferredResultCodes(selected?[selected]:rows.slice(0,3));
  const checklist=resultPackageChecklistMarkup(rows,selected);
  return `<section class="property-section"><div class="section-label">Runtime</div><div class="stat-list">
    <div class="stat-row"><span>Version</span><strong>${esc(runtime.version||'—')}</strong></div><div class="stat-row"><span>Python</span><strong>${esc(runtime.python||'—')}</strong></div><div class="stat-row"><span>FFmpeg</span><strong>${runtime.ffmpeg?'Available':'Missing'}</strong></div><div class="stat-row"><span>Built-in C3D core</span><strong>${runtime.builtin_c3d_reader?'Available':'Missing'}</strong></div><div class="stat-row"><span>ezc3d fallback</span><strong>${runtime.optional?.ezc3d?'Available':'Optional'}</strong></div><div class="stat-row"><span>Legacy code</span><strong>${runtime.legacy_engines_loaded?'Loaded':'Not loaded'}</strong></div></div></section>
    <section class="property-section"><div class="section-label">Quick folders</div><div class="property-stack"><button class="soft-button full" id="open-derived">Open derived outputs</button><button class="soft-button full" id="open-exports">Open organized exports</button><button class="soft-button full" id="open-reports">Open QC reports</button></div></section>
    <section class="property-section"><div class="section-label">Processed artifact cleanup</div><div class="property-stack"><div class="form-help">Move obsolete generated artifacts and unreferenced project folders to Project Trash. Raw recordings are never touched.</div><button class="soft-button full" id="jump-artifact-cleanup">Review cleanup candidates</button></div></section>
    <section class="property-section"><div class="section-label">Organized export package</div><div class="property-stack">
      <div class="form-help">Creates a human-readable <code>exports/S/I/R</code> package without changing immutable artifacts.</div>
      <div class="form-row"><div class="field"><label>Session code</label><input id="result-session-code" value="${esc(defaults.s||'') }" placeholder="S001"></div><div class="field"><label>Interaction</label><input id="result-interaction-code" value="${esc(defaults.i||'')}" placeholder="I04"></div></div>
      <div class="field"><label>Role</label><input id="result-role-code" value="${esc(defaults.r||'')}" placeholder="R01"></div>
      <div class="field"><label>Package label</label><input id="result-package-label" value="${esc([defaults.s,defaults.i,defaults.r,'results'].filter(Boolean).join('_')||'OculoStudio_results')}"></div>
      <div class="field"><label>File mode</label><select id="result-copy-mode"><option value="copy">Copy selected outputs into package</option><option value="index_only">Index only · no duplicate large files</option></select></div>
      <div class="result-select-actions"><button class="row-button" id="result-select-all">Select all</button><button class="row-button" id="result-select-none">Clear</button></div>
      <div class="check-list result-package-list">${checklist}</div>
      <button class="primary-button full" id="run-results-package" ${rows.length?'':'disabled'}>Build organized package</button>
    </div></section>
    <section class="property-section"><div class="section-label">QC report</div><button class="primary-button full" id="run-qc-report-side">Export HTML + JSON QC report</button></section>`;
}

function renderMetrics() {
  const metrics = appState.data.qc?.metrics || [];
  const placeholders = PIPELINE.slice(1,7).map((x,i)=>({id:x.id,label:x.label,display:'Not run',detail:x.desc,value:null,status:'neutral'}));
  const rows = (metrics.length ? metrics : placeholders).slice(0,6);
  $('#qc-strip').innerHTML = rows.map((m,i) => {
    const value = typeof m.value === 'number' && Number.isFinite(m.value) ? Math.max(0,Math.min(100,m.value*100)) : 0;
    const ringText = typeof m.value === 'number' && Number.isFinite(m.value) ? `${Math.round(value)}%` : '—';
    const points = Array.from({length:15},(_,j)=>`${j*8},${18-(5+Math.sin(j*.8+i)*4+(j%4)*1.2)}`).join(' ');
    return `<article class="metric-card"><div class="metric-label">${esc(m.label)}</div><div class="metric-main"><div class="ring" style="--value:${value}"><span>${ringText}</span></div><div class="metric-copy"><div class="metric-value">${esc(m.display)}</div><div class="metric-detail">${esc(m.detail)}</div></div></div><div class="spark"><svg viewBox="0 0 112 24" preserveAspectRatio="none"><polyline points="${points}" fill="none" stroke="#aebe21" stroke-width="1.4"/><line x1="0" y1="22" x2="112" y2="22" stroke="#d7d7ce"/></svg></div></article>`;
  }).join('');
}

function renderJobs() {
  const rows = jobs(); const list=$('#job-list'); const area=list?.closest('.job-area');
  area?.classList.toggle('is-empty',!rows.length);
  if (!rows.length) { list.innerHTML = `<div class="empty-jobs">Processing jobs and progress appear here.</div>`; return; }
  const statusLabel=status=>({completed:'ready',failed:'failed',cancelled:'cancelled',running:'running',queued:'queued'}[String(status)]||String(status||''));
  list.innerHTML = rows.slice(0,15).map(job => {
    const artifactId=jobResultArtifactId(job);
    const isBodyAOIJob=['body_aoi','body_aoi_review_reduction'].includes(job.operation);
    const resultBody=isBodyAOIJob&&artifactId?bodyArtifactById(artifactId):null;
    const bodyGeometryReady=Boolean(resultBody&&bodyArtifactReviewReady(resultBody));
    const bodyReview=isBodyAOIJob&&job.status==='completed'&&artifactId&&bodyGeometryReady;
    const bodyBlocked=isBodyAOIJob&&job.status==='completed'&&artifactId&&!bodyGeometryReady;
    const faceReview=['face_aoi','face_aoi_manual_review'].includes(job.operation)&&job.status==='completed'&&artifactId;
    const stage=bodyBlocked?'Geometry blocked · regenerate':job.status==='completed'?'Result ready':(job.stage||'');
    return `<article class="job-card status-${esc(job.status)}"><div class="job-top"><b>${esc(job.label)}</b><span>${esc(statusLabel(job.status))}</span></div>
      <div class="job-progress"><i style="width:${Math.round((job.progress||0)*100)}%"></i></div><div class="job-bottom"><span>${esc(stage)}</span>${['queued','running'].includes(job.status)?`<button class="job-cancel" data-id="${esc(job.id)}">Cancel</button>`:`<span>${Math.round((job.progress||0)*100)}%</span>`}</div>
      ${faceReview?`<button class="job-result" data-mode="face" data-open-manual="true" data-artifact-id="${esc(artifactId)}">Open Face Human Review</button>`:''}${bodyReview?`<button class="job-result" data-mode="body" data-open-manual="true" data-artifact-id="${esc(artifactId)}">Open Body Human Review</button>`:''}${bodyBlocked?`<button class="job-result" data-mode="body" data-open-manual="false" data-artifact-id="${esc(artifactId)}">Inspect geometry failure</button>`:''}${job.error?`<div class="job-error" title="${esc(job.error)}">${esc(job.error)}</div>`:''}</article>`;
  }).join('');
  $$('.job-cancel',list).forEach(b=>b.onclick=async()=>{try{await api(`/api/jobs/${encodeURIComponent(b.dataset.id)}/cancel`,{method:'POST'});await refreshState({quiet:false});}catch(e){showError(e,'Could not cancel job')}});
  $$('.job-result',list).forEach(button=>button.onclick=()=>openAOIReviewResult(button.dataset.mode,button.dataset.artifactId,{scroll:true,openManual:button.dataset.openManual==='true'}));
}

function bindGlobalActions() {
  $('#brand-home').onclick=()=>navigate('project');
  $('#refresh-state').onclick=()=>refreshState({quiet:false});
  $('#import-asset-sidebar').onclick=()=>scrollDataWorkspace('interaction');
  $('#asset-refresh-all').onclick=async()=>{
    const s=session(); if(!s)return;
    try { for(const a of assets()) await api(`/api/assets/${encodeURIComponent(s.id)}/${encodeURIComponent(a.id)}/refresh?checksum=false`,{method:'POST'}); toast('Assets refreshed'); await refreshState({quiet:false}); }
    catch(e){showError(e,'Could not refresh assets')}
  };
  $('#clear-jobs').onclick=async()=>{try{await api('/api/jobs/finished',{method:'DELETE'});await refreshState({quiet:false});}catch(e){showError(e,'Could not clear jobs')}};
  $$('.side-link').forEach(b=>b.onclick=()=>navigate(b.dataset.sidePage));
}

function bindProjectActions() {
  $('#welcome-create')?.addEventListener('click', openCreateProject);
  $('#welcome-open')?.addEventListener('click', openOpenProject);
}

function bindResultsWorkspaceActions() {
  bindResultGroupControls(document);
  $$('[data-result-artifact]').forEach(button=>button.onclick=()=>{appState.selectedArtifactId=button.dataset.resultArtifact;render()});
  $$('.result-open-file').forEach(button=>button.onclick=()=>openLocalPath(button.dataset.path));
  $$('.result-open-folder').forEach(button=>button.onclick=()=>openLocalPath(button.dataset.path));
  $$('.result-reveal-file').forEach(button=>button.onclick=()=>revealLocalPath(button.dataset.path));
  $('#jump-artifact-cleanup')?.addEventListener('click',()=>$('#artifact-cleanup-card')?.scrollIntoView({behavior:'smooth',block:'start'}));
  $('#cleanup-select-current')?.addEventListener('click',event=>{const id=event.currentTarget.dataset.artifactId;if(!id)return;const selected=cleanupSelectionSet();selected.add(cleanupEntryKey('artifact',id));appState.artifactCleanupSelection=[...selected];$('#artifact-cleanup-card')?.scrollIntoView({behavior:'smooth',block:'start'});renderArtifactCleanupPanel();});
  renderArtifactCleanupPanel();
  if(appState.artifactCleanupPlanSessionId!==selectedSessionId() || !appState.artifactCleanupPlan){
    loadArtifactCleanupPlan(false);
  }
}

function bindPageActions() {
  nativeBindActions();
  bindArtifactVideoPlayer();
  bindFaceMapperLivePlayer();
  $$(`[data-privacy-preview-mode]`).forEach(button=>button.addEventListener('click',()=>{if(button.disabled)return;appState.privacyPreviewMode=button.dataset.privacyPreviewMode||'live_filter';render();}));
  if(appState.page==='targets' && window.OculoReview){
    const initialMode=appState.aoiReviewInitialMode;
    const initialArtifactId=appState.aoiReviewInitialArtifactId;
    const initialVideoArtifactId=appState.aoiReviewInitialVideoArtifactId;
    const shouldScroll=appState.aoiReviewScrollRequested;
    const openManualMode=appState.aoiReviewOpenManualMode;
    appState.aoiReviewInitialMode='';
    appState.aoiReviewInitialArtifactId='';
    appState.aoiReviewInitialVideoArtifactId='';
    appState.aoiReviewScrollRequested=false;
    appState.aoiReviewOpenManualMode='';
    const reviewContext=activeWorkflowContext();
    const reviewTrims=targetRowsForActiveContext(artifacts(['native_reference_trim']));
    const reviewTrimId=selectedTargetTrimId(reviewTrims);
    const binding=OculoReview.bind({sessionId:selectedSessionId(),interactionId:reviewContext.interactionId,roleId:reviewContext.roleId,nativeTrimArtifactId:reviewTrimId,initialMode,initialArtifactId,initialVideoArtifactId});
    if(shouldScroll)scheduleAOIReviewScroll();
    Promise.resolve(binding).then(async()=>{
      if(openManualMode==='body')await OculoReview.openBodyHumanReview?.();
      else if(openManualMode==='face')await OculoReview.openHumanReview?.();
      if(shouldScroll)scheduleAOIReviewScroll();
    }).catch(()=>{});
  }
  if(appState.page==='gaze3d' && window.OculoGazePreview) OculoGazePreview.bind();
  if(appState.page==='qc') bindResultsWorkspaceActions();
  $$('[data-go]').forEach(b=>b.onclick=()=>navigate(b.dataset.go));
  $('#project-add-session')?.addEventListener('click', openAddSession);
  $('#project-manage-sessions')?.addEventListener('click',()=>openProjectSessionManager({focus:'sessions'}));
  $('#project-manage-sessions-inline')?.addEventListener('click',()=>openProjectSessionManager({focus:'sessions'}));
  $('#project-manage-sessions-toolbar')?.addEventListener('click',()=>openProjectSessionManager({focus:'sessions'}));
  $('#project-open-shared-calibration')?.addEventListener('click',()=>openLocalPath(projectWorkspace().shared_calibration_folder||joinLocalPath(project().root,'shared/calibration')));
  $$('.project-session-open').forEach(button=>button.onclick=async()=>{try{await api(`/api/session/${encodeURIComponent(button.dataset.sessionId)}/select`,{method:'POST'});await refreshState({quiet:false,fullRender:true});}catch(error){showError(error,'Could not open Session')}});
  $$('.project-session-folder').forEach(button=>button.onclick=()=>openLocalPath(button.dataset.path));
  $('#project-open')?.addEventListener('click', openOpenProject);
  $('#project-start-fresh')?.addEventListener('click', openCreateProject);
  $('#project-start-fresh-toolbar')?.addEventListener('click', openCreateProject);
  $('#advanced-start-fresh')?.addEventListener('click', openCreateProject);
  $('#data-interaction-choose-files')?.addEventListener('click',()=>OculoImport.requestOpen());
  $('#data-interaction-choose-folder')?.addEventListener('click',()=>OculoImport.chooseFolder());
  $('#interaction-data-drop-zone')?.addEventListener('click',()=>OculoImport.requestOpen());
  $('#interaction-data-drop-zone')?.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();OculoImport.requestOpen();}});
  $('#data-calibration-choose-files')?.addEventListener('click',()=>openCalibrationSetImportDialog({setCode:(calibrationSetRecord(appState.calibrationSetId)||activeCalibrationSetRecord()||calibrationSetRows()[0])?.code||'',autoPicker:'file'}));
  $('#data-calibration-choose-folder')?.addEventListener('click',()=>openCalibrationSetImportDialog({setCode:(calibrationSetRecord(appState.calibrationSetId)||activeCalibrationSetRecord()||calibrationSetRows()[0])?.code||'',autoPicker:'directory'}));
  $('#calibration-data-drop-zone')?.addEventListener('click',()=>openCalibrationSetImportDialog({setCode:(calibrationSetRecord(appState.calibrationSetId)||activeCalibrationSetRecord()||calibrationSetRows()[0])?.code||''}));
  $('#calibration-data-drop-zone')?.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();openCalibrationSetImportDialog({setCode:(calibrationSetRecord(appState.calibrationSetId)||activeCalibrationSetRecord()||calibrationSetRows()[0])?.code||''});}});
  $('#data-manage-calibration-sets')?.addEventListener('click',()=>openProjectSessionManager({focus:'calibration'}));
  $$('.data-manage-calibration-set').forEach(button=>button.onclick=()=>openProjectSessionManager({focus:'calibration',setId:button.dataset.setId||''}));
  $$('.data-select-calibration-set').forEach(button=>button.onclick=()=>{const row=calibrationSetRecord(button.dataset.setId);if(row){appState.calibrationSetId=row.code||row.id;appState.selectedCalibrationAssetId=null;render();}});
  $$('.data-import-calibration').forEach(button=>button.onclick=()=>openCalibrationSetImportDialog({setCode:button.dataset.setCode||activeCalibrationSetId()}));
  $$('.calibration-source-inspect').forEach(button=>button.onclick=()=>{appState.selectedCalibrationAssetId=button.dataset.id;render();scrollDataWorkspace('calibration');});
  $('#data-open-calibration-folder')?.addEventListener('click',()=>{const row=calibrationSetRecord(appState.calibrationSetId)||activeCalibrationSetRecord();if(row?.folder)openLocalPath(row.folder);});
  $('#data-refresh-c3d')?.addEventListener('click',async()=>{
    const rows=assets(['c3d']);
    if(!rows.length)return;
    const button=$('#data-refresh-c3d');
    try{
      button.disabled=true;button.textContent='Re-reading C3D files…';
      for(const row of rows)await api(`/api/assets/${encodeURIComponent(selectedSessionId())}/${encodeURIComponent(row.id)}/refresh?checksum=false`,{method:'POST'});
      toast('C3D metadata refreshed',`${rows.length} file${rows.length===1?'':'s'} re-read with long-frame recovery.`);
      await refreshState({quiet:false});
      requestAnimationFrame(()=>{const scroller=$('#workspace .view-scroll');if(scroller)scroller.scrollLeft=0;});
    }catch(error){showError(error,'Could not refresh C3D metadata')}
    finally{if(button){button.disabled=false;}}
  });
  OculoImport.bindPage();
  $$('.asset-inspect').forEach(b=>b.onclick=()=>{appState.selectedAssetId=b.dataset.id;render()});
  $$('.asset-select').forEach(box=>box.onchange=()=>{const set=new Set(appState.selectedAssetIds||[]);box.checked?set.add(box.dataset.id):set.delete(box.dataset.id);appState.selectedAssetIds=[...set];render()});
  $('#asset-select-all')?.addEventListener('change',e=>{appState.selectedAssetIds=e.target.checked?assets().map(a=>a.id):[];render()});
  $$('.asset-edit-identity').forEach(b=>b.onclick=()=>openAssetIdentityEditor([b.dataset.id]));
  $('#edit-selected-identity')?.addEventListener('click',e=>openAssetIdentityEditor([e.currentTarget.dataset.id]));
  $('#data-edit-selected')?.addEventListener('click',()=>openAssetIdentityEditor(appState.selectedAssetIds||[]));
  $('#clear-selected-participant')?.addEventListener('click',async e=>{try{const id=e.currentTarget.dataset.id;const a=session()?.assets?.[id];const ai=assetIdentity(a);await api(`/api/assets/${selectedSessionId()}/${id}/identity`,{method:'PATCH',body:{participant_id:null,interaction_id:ai.interaction_id||null,role_id:ai.role_id||null}});toast('Participant ID cleared');await refreshState({quiet:false})}catch(error){showError(error,'Could not clear participant ID')}});
  $$('.asset-refresh').forEach(b=>b.onclick=async()=>{try{await api(`/api/assets/${selectedSessionId()}/${b.dataset.id}/refresh?checksum=false`,{method:'POST'});toast('Asset refreshed');await refreshState({quiet:false})}catch(e){showError(e,'Refresh failed')}});
  $$('.asset-delete').forEach(b=>b.onclick=()=>confirmRemoveAsset(b.dataset.id));
  $$('.map-delete').forEach(b=>b.onclick=async()=>{try{await api(`/api/time-maps/${selectedSessionId()}/${b.dataset.id}`,{method:'DELETE'});toast('Time map removed');await refreshState({quiet:false})}catch(e){showError(e,'Could not remove map')}});
  $('#run-frame-map')?.addEventListener('click', submitFrameMap);
  $('#open-new-clip')?.addEventListener('click',()=>$('#clip-label')?.focus());
  $$('.clip-select').forEach(b=>b.onclick=()=>{appState.selectedClipId=b.dataset.id;render()});
  $$('.clip-delete').forEach(b=>b.onclick=async()=>{try{await api(`/api/clips/${selectedSessionId()}/${b.dataset.id}`,{method:'DELETE'});toast('Clip removed');await refreshState({quiet:false})}catch(e){showError(e,'Could not remove clip')}});
  $('#calibration-open-data-sources')?.addEventListener('click',()=>scrollDataWorkspace('calibration'));
  $$('.calibration-open-data-sources').forEach(button=>button.addEventListener('click',()=>scrollDataWorkspace('calibration')));
  $('#run-qc-report')?.addEventListener('click', submitQC);
}

function bindGaze3DWorkflowDefaults() {
  const scope=$('#ray-scope'), alignment=$('#ray-alignment-mode'), cloud=$('#ray-cloud'), role=$('#ray-role'), trim=$('#ray-native-trim');
  if(!scope||!alignment)return;
  const roleCode=row=>{
    const text=[row?.label,row?.parameters?.role_id,row?.parameters?.wearer_role_id,...(row?.source_asset_ids||[])].filter(Boolean).join(' ');
    return codeFromText(text,'R');
  };
  const selectMatchingInputs=({preferCloud=false}={})=>{
    if(scope.value!=='native_trim'||!cloud||!trim)return;
    const wantedRole=role?.value||'';
    if(preferCloud&&wantedRole){
      const candidate=assets(['pupil_cloud']).find(row=>assetIdentity(row).role_id===wantedRole||codeFromText(`${row.label} ${row.source_path}`,'R')===wantedRole);
      if(candidate)cloud.value=candidate.id;
    }
    const cloudId=cloud.value;
    const rows=artifacts(['native_reference_trim']).slice().sort(byCreated);
    const compatible=rows.filter(row=>{
      const sourceMatch=!cloudId||row.parameters?.timeseries_asset_id===cloudId||(row.source_asset_ids||[]).includes(cloudId);
      const rowRole=roleCode(row);
      const roleMatch=!wantedRole||!rowRole||rowRole===wantedRole;
      return sourceMatch&&roleMatch;
    });
    const current=rows.find(row=>row.id===trim.value);
    const currentCompatible=current&&compatible.some(row=>row.id===current.id);
    if(!currentCompatible&&compatible.length)trim.value=compatible[compatible.length-1].id;
  };
  const update=(scopeChanged=false)=>{
    const nativeTrim=scope.value==='native_trim';
    const legacyClip=scope.value==='clip';
    const stage=calibrationWorkflowState(role?.value||calibrationContextRole());
    const calibrationLocked=Boolean(stage.handoff?.parameters?.require_time_sync||stage.handoff?.qc?.time_sync_required);
    if(scopeChanged)alignment.value=nativeTrim?(calibrationLocked?'timegraph':'pretrimmed_shared_start'):'timegraph';
    if(calibrationLocked)alignment.value='timegraph';
    alignment.disabled=calibrationLocked;
    const shared=nativeTrim&&alignment.value==='pretrimmed_shared_start'&&!calibrationLocked;
    const trimRow=$('#ray-native-trim-row'), alignmentRow=$('#ray-alignment-row'), sharedSettings=$('#ray-shared-start-settings'), timegraphSettings=$('#ray-timegraph-settings'), clipRow=$('#ray-legacy-clip-row'), help=$('#ray-scope-help');
    if(trimRow)trimRow.hidden=!nativeTrim;
    if(alignmentRow)alignmentRow.hidden=!nativeTrim;
    if(sharedSettings)sharedSettings.hidden=!shared;
    if(timegraphSettings)timegraphSettings.hidden=shared;
    if(clipRow)clipRow.hidden=!legacyClip;
    if(help)help.textContent=calibrationLocked
      ?'The selected Calibration handoff supplies the verified role-specific Pupil IMU ↔ assigned ETM affine TimeGraph. Shared-start bypass is disabled.'
      :shared
        ?'Only rows retained by the Native-reference trim are processed. The trim local clock starts at the same physical instant as the first ETM/body C3D frame.'
        :nativeTrim
          ?'The selected trim limits rows, while a verified TimeGraph mapping supplies clock alignment.'
        :legacyClip
          ?'Legacy Global-Time clips require formal TimeGraph paths.'
          :'Full-recording mode preserves every source row and requires a formal TimeGraph path.';
  };
  scope.addEventListener('change',()=>{update(true);selectMatchingInputs();});
  alignment.addEventListener('change',()=>update(false));
  cloud?.addEventListener('change',()=>selectMatchingInputs());
  role?.addEventListener('change',()=>selectMatchingInputs({preferCloud:true}));
  selectMatchingInputs({preferCloud:!cloud?.value});
  update(false);
}

function bindPropertyActions() {
  nativeBindActions();
  $$('.toggle').forEach(button => button.onclick = () => { const on = button.dataset.on !== 'true'; button.dataset.on=String(on); button.classList.toggle('on',on); });
  $('#open-project-folder')?.addEventListener('click',()=>openLocalPath(project().root));
  $('#data-jump-calibration')?.addEventListener('click',()=>scrollDataWorkspace('calibration'));
  $('#data-jump-interaction')?.addEventListener('click',()=>scrollDataWorkspace('interaction'));
  $('#data-edit-current-asset')?.addEventListener('click',event=>openAssetIdentityEditor([event.currentTarget.dataset.id]));
  $('#calibration-open-data-sources')?.addEventListener('click',()=>scrollDataWorkspace('calibration'));
  $$('.calibration-open-data-sources').forEach(button=>button.addEventListener('click',()=>scrollDataWorkspace('calibration')));
  $('#open-session-folder')?.addEventListener('click',()=>openLocalPath(sessionStoragePath('root')||project().root));
  $('#open-shared-calibration-folder')?.addEventListener('click',()=>openLocalPath(projectWorkspace().shared_calibration_folder||joinLocalPath(project().root,'shared/calibration')));
  $('#open-calibration-sets-folder')?.addEventListener('click',()=>openLocalPath(projectWorkspace().calibration_sets_folder||joinLocalPath(project().root,'calibration_sets')));
  $('#project-manage-sessions-side')?.addEventListener('click',()=>openProjectSessionManager({focus:'sessions'}));
  $('#data-manage-calibration-side')?.addEventListener('click',()=>openProjectSessionManager({focus:'calibration'}));
  $('#data-import-calibration-side')?.addEventListener('click',()=>openCalibrationSetImportDialog({setCode:(activeCalibrationSetRecord()||calibrationSetRows()[0])?.code||''}));
  $('#project-session-select')?.addEventListener('change',async event=>{try{await api(`/api/session/${encodeURIComponent(event.target.value)}/select`,{method:'POST'});await refreshState({quiet:false,fullRender:true});}catch(error){showError(error,'Could not switch Session')}});
  $('#project-start-fresh-side')?.addEventListener('click', openCreateProject);
  $('#advanced-start-fresh-side')?.addEventListener('click', openCreateProject);
  $$('[data-go]', $('#properties-content')).forEach(button=>button.onclick=()=>navigate(button.dataset.go));
  $('#browse-asset')?.addEventListener('click',()=>OculoImport.requestOpen());
  $('#browse-asset-dir')?.addEventListener('click',async()=>{const p=await choosePath('directory','Select an extracted recording or data directory');if(p)$('#asset-path').value=p;});
  $('#submit-asset')?.addEventListener('click', submitAsset);
  $('#add-anchor')?.addEventListener('click',()=>{captureAnchorRows();appState.timeAnchors.push({source_ns:'0',target_ns:'0',label:`Anchor ${appState.timeAnchors.length+1}`,weight:1});renderProperties()});
  $$('.anchor-remove').forEach(b=>b.onclick=()=>{captureAnchorRows();appState.timeAnchors.splice(Number(b.dataset.index),1);if(!appState.timeAnchors.length)appState.timeAnchors.push({source_ns:'0',target_ns:'0',label:'Anchor 1',weight:1});renderProperties()});
  $('#fit-time-map')?.addEventListener('click', submitTimeMap);
  $('#run-auto-sync-from-sync')?.addEventListener('click', runAutomaticSyncFromPage);
  $('#accept-auto-sync-from-sync')?.addEventListener('click', acceptAutomaticSyncFromPage);
  $('#avsync-role')?.addEventListener('change',event=>{appState.audioVideoSyncRole=String(event.target.value||'R01').toUpperCase();appState.audioVideoSyncSourceId='';appState.audioVideoSyncReferenceId='';renderProperties();});
  $('#avsync-source')?.addEventListener('change',event=>{appState.audioVideoSyncSourceId=event.target.value||'';});
  $('#avsync-reference')?.addEventListener('change',event=>{appState.audioVideoSyncReferenceId=event.target.value||'';});
  $('#avsync-max-offset')?.addEventListener('change',event=>{appState.audioVideoSyncMaximumOffsetSec=Number(event.target.value||300);});
  $('#run-audio-video-sync')?.addEventListener('click',submitAudioVideoSync);
  $$('.avsync-review').forEach(button=>button.addEventListener('click',()=>openAudioVideoSyncReview({candidateId:button.dataset.candidateId||'',artifactId:button.dataset.artifactId||''})));
  $('#cancel-gaze-sync-prefill')?.addEventListener('click',()=>{appState.syncPrefill=null;appState.autoSyncCandidate=null;appState.timeAnchors=[{source_ns:'0',target_ns:'0',label:'Anchor 1',weight:1}];navigate('gaze3d');});
  $('#save-clip')?.addEventListener('click', submitClip);
  $('#run-clip')?.addEventListener('click', submitClipJob);
  $('#run-facemapper-filter')?.addEventListener('click', submitFaceMapperFilter);
  $('#run-privacy-mask-preview')?.addEventListener('click', submitPrivacyMaskPreview);
  $('#run-privacy-final')?.addEventListener('click', submitPrivacyFinal);
  $('#run-privacy-full')?.addEventListener('click', submitPrivacyFull);
  $('#run-privacy-clip')?.addEventListener('click', submitPrivacyClip);
  $('#remove-current-privacy')?.addEventListener('click',e=>confirmRemoveArtifact(e.currentTarget.dataset.artifactId));
  bindPrivacyWorkflowDefaults();
  bindPrivacyMaskPolicy();
  bindFaceMapperPreviewScope();
  bindGaze3DWorkflowDefaults();
  if(appState.page==='gaze3d'){
    $('#gaze-preview-mode')?.addEventListener('change',event=>{appState.gazePreviewMode=event.target.value;render();});
    $('#gaze-preview-r01')?.addEventListener('change',event=>{appState.gazePreviewArtifactIds.R01=event.target.value;render();});
    $('#gaze-preview-r02')?.addEventListener('change',event=>{appState.gazePreviewArtifactIds.R02=event.target.value;render();});
    $('#gaze-preview-tolerance')?.addEventListener('change',event=>{appState.gazePreviewToleranceMs=event.target.value.trim();render();});
  }
  bindMediaToolButtons();
  $$('[data-browse]').forEach(b=>b.onclick=async()=>{const p=await choosePath('file','Select a JSON or calibration profile');if(p)$(`#${b.dataset.browse}`).value=p;});
  if(appState.page==='calibration'){
    const updateCalibrationPreview=()=>window.OculoCalibrationPreview?.update();
    $('#cal-choose-files')?.addEventListener('click',()=>chooseCalibrationSources('file'));
    $('#cal-choose-folder')?.addEventListener('click',()=>chooseCalibrationSources('directory'));
    $('#cal-open-source-importer')?.addEventListener('click',()=>openCalibrationSetImportDialog({setCode:activeCalibrationSetId()}));
    $('#cal-scan-sources')?.addEventListener('click',()=>scanCalibrationSourceSelection({render:true,quiet:false}).catch(error=>showError(error,'Could not scan calibration sources')));
    $('#cal-import-sources')?.addEventListener('click',submitCalibrationSources);
    const updateCalibrationSetSelection=event=>{appState.calibrationSetId=event.target.value.trim();appState.calibrationScan=null;if(event.type==='change'){renderWorkspace();renderProperties();}};
    $('#cal-set-id')?.addEventListener('input',updateCalibrationSetSelection);
    $('#cal-set-id')?.addEventListener('change',updateCalibrationSetSelection);
    $('#cal-source-path')?.addEventListener('input',event=>{appState.calibrationSourcePath=event.target.value;appState.calibrationScan=null;});
    $('#cal-import-recursive')?.addEventListener('change',event=>{appState.calibrationRecursive=Boolean(event.target.checked);appState.calibrationScan=null;});
    $$('.calibration-set-chip').forEach(button=>button.addEventListener('click',()=>{
      appState.calibrationSetId=button.dataset.calibrationSet||'';
      renderWorkspace();renderProperties();
    }));
    ['cal-preview-profile','cal-board','cal-world'].forEach(id=>$('#'+id)?.addEventListener('change',updateCalibrationPreview));
    $$('.cal-tracker-check').forEach(input=>input.addEventListener('change',()=>{captureCalibrationDraft();updateCalibrationPreview();}));
    const calibrationDraftInputIds=['cal-native','cal-board','cal-world','cal-board-mode','cal-board-def','cal-intrinsics','cal-sample-every','cal-max-detections','cal-identity-confidence','cal-participant','cal-mount','cal-valid-sessions','cal-hardware','cal-name'];
    calibrationDraftInputIds.forEach(id=>{
      const element=$('#'+id);if(!element)return;
      element.addEventListener('input',captureCalibrationDraft);
      element.addEventListener('change',captureCalibrationDraft);
    });
    ['cal-raw-identity','cal-auto-register','cal-auto-name','cal-hardware-override','cal-output-g3dcal'].forEach(id=>$('#'+id)?.addEventListener('click',()=>setTimeout(captureCalibrationDraft,0)));
    $$('.calibration-profile-select').forEach(button=>button.addEventListener('click',()=>{const select=$('#cal-preview-profile');if(select){select.value=button.dataset.calibrationProfile||'';updateCalibrationPreview();}}));
    requestAnimationFrame(()=>window.OculoCalibrationPreview?.mount());
  }
  $$('.calibration-build-set-profiles').forEach(button=>button.addEventListener('click',()=>openCalibrationSetProfileBuilder({setId:button.dataset.setId||activeCalibrationSetRecord()?.id||''})));
  $('#run-calibration-import')?.addEventListener('click', submitCalibrationImport);
  $('#run-calibration-register-measured')?.addEventListener('click', submitCalibrationRegisterMeasured);
  $('#run-calibration-transfer')?.addEventListener('click', submitCalibrationTransfer);
  $('#run-calibration-acceptance')?.addEventListener('click', submitCalibrationAcceptance);
  $('#run-calibration')?.addEventListener('click', submitCalibration);
  $('#run-etm-assignment')?.addEventListener('click', submitETMAssignment);
  $$('.cal-sync-timeseries').forEach(select=>select.addEventListener('change',()=>{appState.calibrationSyncSelections[String(select.dataset.role||'').toUpperCase()]=select.value||'';}));
  $$('.cal-sync-run').forEach(button=>button.addEventListener('click',()=>runCalibrationAutoSync(button.dataset.role)));
  $$('.cal-sync-preview').forEach(button=>button.addEventListener('click',()=>openCalibrationSyncPreview({role:button.dataset.role||'',artifactId:button.dataset.artifactId||'',candidateId:button.dataset.candidateId||''})));
  $$('.cal-sync-accept').forEach(button=>button.addEventListener('click',()=>acceptCalibrationAutoSync(button.dataset.role)));
  $('#run-calibration-sync-both')?.addEventListener('click', runCalibrationAutoSyncBoth);
  $('#run-calibration-binding')?.addEventListener('click', submitCalibrationBinding);
  $$('.cal-c3d-clip-input').forEach(select=>select.addEventListener('change',()=>{
    const role=String(select.dataset.role||'').toUpperCase();
    const field=String(select.dataset.field||'');
    if(!appState.calibrationClipSelections[role])appState.calibrationClipSelections[role]={};
    appState.calibrationClipSelections[role][field]=select.value||'';
  }));
  $('#run-c3d-defined-clips')?.addEventListener('click', submitC3DDefinedAlignedClips);
  $('#run-gaze3d')?.addEventListener('click', submitGaze3D);
  $('#association-trim')?.addEventListener('change',e=>rememberTargetTrimSelection(e.target.value));
  $('#face-aoi-trim')?.addEventListener('change',e=>rememberTargetTrimSelection(e.target.value));
  $('#run-participant-association')?.addEventListener('click', submitParticipantAssociation);
  $('#run-face-aoi')?.addEventListener('click', submitFaceAOI);
  $('#open-face-human-review-side')?.addEventListener('click',event=>openAOIReviewResult('face',event.currentTarget.dataset.artifactId,{scroll:true,openManual:true}));
  $('#open-body-human-review-side')?.addEventListener('click',event=>openAOIReviewResult('body',event.currentTarget.dataset.artifactId,{scroll:true,openManual:true}));
  $('#reduce-existing-body-review')?.addEventListener('click', submitBodyReviewReduction);
  $('#run-body-aoi')?.addEventListener('click', submitBodyAOI);
  $('#run-object-aoi')?.addEventListener('click', submitObjectAOI);
  $('#run-target-fusion')?.addEventListener('click', submitTargetFusion);
  $('#run-dyadic-attention')?.addEventListener('click', submitDyadicAttention);
  $('#run-semantic-targets')?.addEventListener('click', submitSemanticTargets);
  $('#run-qc-report-side')?.addEventListener('click', submitQC);
  $('#open-derived')?.addEventListener('click',()=>openLocalPath(sessionStoragePath('derived')||`${project().root}/derived/${selectedSessionId()}`));
  $('#open-exports')?.addEventListener('click',()=>openLocalPath(`${project().root}/exports`));
  $('#open-reports')?.addEventListener('click',()=>openLocalPath(sessionStoragePath('reports')||`${project().root}/reports/${selectedSessionId()}`));
  $('#result-select-all')?.addEventListener('click',()=>$$('.result-package-check').forEach(x=>x.checked=true));
  $('#result-select-none')?.addEventListener('click',()=>$$('.result-package-check').forEach(x=>x.checked=false));
  $('#run-results-package')?.addEventListener('click', submitResultsPackage);
  if(appState.page==='qc') bindResultsWorkspaceActions();
}

function captureAnchorRows() {
  appState.timeAnchors = $$('.anchor-row').map((row,i)=>{
    const source=String($('.anchor-source',row).value||'').trim();
    const seconds=$('.anchor-target-seconds',row);
    const target=seconds?secondsTextToNs(seconds.value):String($('.anchor-target',row).value||'').trim();
    if(!/^[+-]?\d+$/.test(source))throw new Error(`Anchor ${i+1}: Pupil/source timestamp must be an integer number of nanoseconds`);
    if(!/^[+-]?\d+$/.test(target))throw new Error(`Anchor ${i+1}: target timestamp is invalid`);
    return {source_ns:source,target_ns:target,label:`Anchor ${i+1}`,weight:1};
  });
}

function openCreateProject() {
  const body=`<div class="project-wizard">
    <nav class="project-wizard-steps"><button class="active" data-project-step="1"><span>1</span><b>Location</b><small>Where the Project lives</small></button><button data-project-step="2"><span>2</span><b>Sessions</b><small>Create S004–S007 together</small></button><button data-project-step="3"><span>3</span><b>Shared resources</b><small>Calibration and folder plan</small></button></nav>
    <section class="project-wizard-panel" data-project-panel="1">
      <div class="project-wizard-intro"><div><span>Project storage</span><h3>Choose a parent directory and a new Project folder.</h3><p>The parent is only a container. OculoStudio creates one dedicated subfolder and never mixes it with the currently open Project.</p></div></div>
      <div class="form-grid"><div class="field full-span"><label>Parent directory</label><div class="inline-input"><input id="create-parent" value="" placeholder="Select or type a parent folder"><button class="input-action" id="browse-project-root" title="Choose parent folder">…</button></div><small>Example: D:\\DuoFlow\\OculoStudio_Projects. This example is not a selected value.</small></div><div class="field"><label>New Project folder</label><input id="create-folder" value="OCUMO_S004_S007"></div><div class="field"><label>Project display name</label><input id="create-name" value="OCUMO S004–S007"></div><div class="field full-span"><label>Full Project path</label><output id="create-path-preview" class="path-preview">No parent directory selected</output></div></div>
      <div id="create-location-status" class="project-location-status neutral"><strong>Location not checked</strong><span>Select or type a real parent directory. Manual input and the folder picker are both supported.</span></div>
    </section>
    <section class="project-wizard-panel" data-project-panel="2" hidden>
      <div class="project-wizard-intro"><div><span>Session plan</span><h3>Create all Sessions in one transaction.</h3><p>One Project can contain many Sessions; each Session can later contain multiple Interactions and two local roles.</p></div></div>
      <div class="form-grid"><div class="field full-span"><label>Sessions</label><input id="create-sessions-spec" value="S004-S007" placeholder="S004-S007 or S004, S005, S006, S007"><small>Ranges, commas, spaces and new lines are accepted.</small></div><div class="field"><label>Open after creation</label><select id="create-selected-session"></select></div><div class="field"><label>Optional Interaction template</label><input id="create-interactions-spec" value="" placeholder="I01-I06"><small>Leave blank when Sessions contain different Interaction counts; add them later in Session Manager.</small></div><div class="field full-span"><label>Sessions to create</label><div id="create-session-chips" class="project-code-chips"></div></div></div>
      <div class="project-hierarchy-preview"><strong>Project hierarchy</strong><code>Project → Session → Interaction → Role</code><span>Example: OCUMO S004–S007 → S004 → I01 → R01/R02</span></div>
    </section>
    <section class="project-wizard-panel" data-project-panel="3" hidden>
      <div class="project-wizard-intro"><div><span>Calibration Set</span><h3>Register shared Calibration recordings once, then bind them to several Sessions.</h3><p>A Calibration Set owns its raw AprilTag recordings, ETM profiles and revisions at Project level. Sessions inherit the selected revision; every Interaction still computes its own wearer assignment, offset, drift and Trim.</p></div></div>
      <label class="project-choice-card"><input id="create-calibration-set" type="checkbox" checked><span><strong>Create an initial Project-level Calibration Set</strong><small>Recommended for S004–S007 when the same calibration recordings or hardware profiles apply across those Sessions.</small></span></label>
      <div id="create-calibration-set-fields" class="form-grid calibration-set-create-fields">
        <div class="field"><label>Calibration Set ID</label><input id="create-calibration-code" value="CAL_S004_S007"><small>Stable code used in folders and provenance.</small></div>
        <div class="field"><label>Display name</label><input id="create-calibration-label" value="Calibration · S004–S007"></div>
        <div class="field full-span"><label>Purpose</label><select id="create-calibration-purpose"><option value="combined" selected>Hardware + known-target gaze calibration</option><option value="hardware">Hardware camera-to-tracker calibration only</option><option value="known_target_gaze">Participant known-target gaze calibration only</option></select></div>
        <div class="field full-span"><label>Applies to</label><div id="create-calibration-session-chips" class="project-code-chips"></div><small>All Sessions created in this wizard will inherit revision 1. You can change assignments later without copying source files.</small></div>
      </div>
      <div class="project-folder-tree" id="create-folder-tree"></div>
      <div id="create-summary" class="project-create-summary"></div>
    </section>
    <footer class="project-wizard-actions"><button class="soft-button" id="create-cancel">Cancel</button><span class="player-spacer"></span><button class="soft-button" id="create-back" hidden>Back</button><button class="primary-button" id="create-next">Next: Sessions</button><button class="primary-button" id="create-submit" hidden>Create Project + Sessions</button></footer>
  </div>`;
  modal('Create Project & Session workspace',body,(root,close)=>{
    let step=1;let locationCheck=null;let locationSerial=0;let folderTouched=false;let nameTouched=false;let calibrationCodeTouched=false;let calibrationLabelTouched=false;
    const panels=()=>$$('[data-project-panel]',root);const stepButtons=()=>$$('[data-project-step]',root);
    const sessions=()=>parseSessionSpecification($('#create-sessions-spec',root).value);
    const interactions=()=>{const raw=$('#create-interactions-spec',root).value.trim();return raw?parseInteractionSpecification(raw):[];};
    const updateLocationPreview=()=>{
      const parent=$('#create-parent',root).value.trim();
      const folder=safeProjectFolderName($('#create-folder',root).value);
      $('#create-folder',root).value=folder;
      const full=parent?joinLocalPath(parent,folder):'';
      const output=$('#create-path-preview',root);output.textContent=full||'No parent directory selected';output.title=full;
      locationCheck=null;
      const status=$('#create-location-status',root);status.className='project-location-status neutral';status.innerHTML='<strong>Location not checked</strong><span>Press Next to validate the parent and target folder.</span>';
      updateSummary();
    };
    const updateSessions=()=>{
      const chipRoot=$('#create-session-chips',root),select=$('#create-selected-session',root);
      try{
        const rows=sessions();
        chipRoot.innerHTML=rows.map(value=>`<span>${esc(value)}</span>`).join('')||'<em>No Sessions</em>';
        const current=select.value;select.innerHTML=rows.map(value=>`<option ${value===current?'selected':''}>${esc(value)}</option>`).join('');
        if(!folderTouched&&rows.length){$('#create-folder',root).value=`OCUMO_${rows[0]}_${rows.at(-1)}`;}
        if(!nameTouched&&rows.length){$('#create-name',root).value=rows.length===1?`OCUMO ${rows[0]}`:`OCUMO ${rows[0]}–${rows.at(-1)}`;}
        if(!calibrationCodeTouched&&rows.length){$('#create-calibration-code',root).value=`CAL_${rows[0]}_${rows.at(-1)}`;}
        if(!calibrationLabelTouched&&rows.length){$('#create-calibration-label',root).value=rows.length===1?`Calibration · ${rows[0]}`:`Calibration · ${rows[0]}–${rows.at(-1)}`;}
        const calChips=$('#create-calibration-session-chips',root);if(calChips)calChips.innerHTML=rows.map(value=>`<span>${esc(value)}</span>`).join('');
        updateLocationPreview();
      }catch(error){chipRoot.innerHTML=`<span class="invalid">${esc(error.message)}</span>`;select.innerHTML='';updateSummary();}
    };
    const updateSummary=()=>{
      let rows=[],plan=[];try{rows=sessions();plan=interactions();}catch(_){ }
      const parent=$('#create-parent',root)?.value.trim()||'';const folder=safeProjectFolderName($('#create-folder',root)?.value||'');const full=parent?joinLocalPath(parent,folder):'—';
      const createSet=$('#create-calibration-set',root)?.checked!==false;
      const setCode=safeProjectFolderName($('#create-calibration-code',root)?.value||'CALIBRATION');
      const tree=$('#create-folder-tree',root);if(tree)tree.innerHTML=`<strong>${esc(folder||'Project')}</strong><span>project.oculostudio.json</span>${createSet?`<span>calibration_sets/${esc(setCode)}/ <i>revision 1 · sources · profiles · reports</i></span>`:''}${rows.map(value=>`<span>sessions/${esc(value)}/ <i>imports · derived · reports · annotations</i></span>`).join('')}<span>exports/</span>`;
      const summary=$('#create-summary',root);if(summary)summary.innerHTML=`<div><span>Full path</span><strong>${esc(full)}</strong></div><div><span>Sessions</span><strong>${rows.length?esc(rows.join(', ')):'Invalid or empty'}</strong></div><div><span>Interaction template</span><strong>${plan.length?esc(plan.join(', ')):'Add per Session later'}</strong></div><div><span>Calibration Set</span><strong>${createSet?esc(setCode+' · revision 1 · '+rows.length+' Sessions'):'Create later'}</strong></div>`;
      const fields=$('#create-calibration-set-fields',root);if(fields)fields.classList.toggle('disabled-panel',!createSet);
    };
    const setStep=value=>{
      step=Math.max(1,Math.min(3,Number(value)||1));
      panels().forEach(panel=>panel.hidden=Number(panel.dataset.projectPanel)!==step);
      stepButtons().forEach(button=>button.classList.toggle('active',Number(button.dataset.projectStep)===step));
      $('#create-back',root).hidden=step===1;$('#create-next',root).hidden=step===3;$('#create-submit',root).hidden=step!==3;
      $('#create-next',root).textContent=step===1?'Next: Sessions':'Next: Shared resources';updateSummary();
    };
    const checkLocation=async()=>{
      const parent=$('#create-parent',root).value.trim();const folder=safeProjectFolderName($('#create-folder',root).value);
      const serial=++locationSerial;const status=$('#create-location-status',root);status.className='project-location-status checking';status.innerHTML='<strong>Checking location…</strong><span>Verifying the real parent folder and target contents.</span>';
      const result=await api('/api/project/location-check',{method:'POST',body:{parent,folder}});if(serial!==locationSerial)return false;locationCheck=result;
      status.className=`project-location-status ${result.valid?'valid':'invalid'}`;
      status.innerHTML=result.valid?`<strong>Ready</strong><span>${esc(result.target)}${result.warnings?.length?` · ${esc(result.warnings.join(' · '))}`:''}</span>`:`<strong>Cannot create here</strong><span>${esc((result.errors||[]).join(' · '))}</span>`;
      return Boolean(result.valid);
    };
    $('#browse-project-root',root).onclick=async()=>{const selected=await choosePath('directory','Choose the parent directory for the new OculoStudio Project');if(selected){$('#create-parent',root).value=selected;updateLocationPreview();await checkLocation();}};
    ['input','change','paste'].forEach(event=>$('#create-parent',root).addEventListener(event,()=>setTimeout(updateLocationPreview,0)));
    $('#create-folder',root).addEventListener('input',()=>{folderTouched=true;updateLocationPreview();});
    $('#create-name',root).addEventListener('input',()=>{nameTouched=true;updateSummary();});
    $('#create-sessions-spec',root).addEventListener('input',updateSessions);
    $('#create-interactions-spec',root).addEventListener('input',updateSummary);
    $('#create-selected-session',root).addEventListener('change',updateSummary);
    $('#create-calibration-set',root).addEventListener('change',updateSummary);
    $('#create-calibration-code',root).addEventListener('input',()=>{calibrationCodeTouched=true;updateSummary();});
    $('#create-calibration-label',root).addEventListener('input',()=>{calibrationLabelTouched=true;updateSummary();});
    $('#create-calibration-purpose',root).addEventListener('change',updateSummary);
    stepButtons().forEach(button=>button.onclick=async()=>{const target=Number(button.dataset.projectStep);if(target>step&&step===1&&!await checkLocation())return;try{sessions();if(target>1)interactions();}catch(error){showError(error,'Session plan is invalid');return;}setStep(target);});
    $('#create-cancel',root).onclick=close;
    $('#create-back',root).onclick=()=>setStep(step-1);
    $('#create-next',root).onclick=async()=>{try{if(step===1&&!await checkLocation())return;sessions();interactions();setStep(step+1);}catch(error){showError(error,'Project setup is incomplete');}};
    $('#create-submit',root).onclick=async()=>{try{
      if(!locationCheck?.valid&&!await checkLocation())return;
      const sessionLabels=sessions(),interactionPlan=interactions();if(!sessionLabels.length)throw new Error('Create at least one Session');
      const target=locationCheck.target;const name=$('#create-name',root).value.trim()||'OCUMO Study';
      $('#create-submit',root).disabled=true;$('#create-submit',root).textContent='Creating Project…';
      const createSet=$('#create-calibration-set',root).checked;
      const calibrationCode=safeProjectFolderName($('#create-calibration-code',root).value||`CAL_${sessionLabels[0]}_${sessionLabels.at(-1)}`);
      await api('/api/project/create',{method:'POST',body:{root:target,name,session_label:sessionLabels[0],session_labels:sessionLabels,selected_session_label:$('#create-selected-session',root).value||sessionLabels[0],planned_interactions:interactionPlan,settings:{folder_layout_version:3,session_folder_strategy:'session_code',shared_calibration:{enabled:createSet,scope:'project_calibration_sets'},initial_calibration_set:{enabled:createSet,code:calibrationCode,label:$('#create-calibration-label',root).value.trim()||calibrationCode,purpose:$('#create-calibration-purpose',root).value,all_sessions:true}}}});
      close();toast('Project and Sessions created',`${name} · ${sessionLabels.join(', ')}`);await refreshState({quiet:false,fullRender:true});OculoImport.projectReady();
      if(window.OculoInteractionWorkspace)window.OculoInteractionWorkspace.setMode('guided',{page:'project'});else navigate('project');
    }catch(error){showError(error,'Project creation failed');const button=$('#create-submit',root);if(button){button.disabled=false;button.textContent='Create Project + Sessions';}}};
    updateSessions();setStep(1);
  });
}

function openOpenProject() {
  modal('Open OculoStudio project', `<div class="field"><label>Project folder or project.oculostudio.json</label><div class="inline-input"><input id="open-path" placeholder="C:\\Projects\\OculoStudy"><button class="input-action" id="browse-open">…</button></div></div><div class="form-actions"><button class="primary-button" id="open-submit">Open project</button></div>`, (root,close)=>{
    $('#browse-open',root).onclick=async()=>{const p=await choosePath('file','Select project.oculostudio.json',[['OculoStudio project','*.json']]);if(p)$('#open-path',root).value=p};
    $('#open-submit',root).onclick=async()=>{try{const path=$('#open-path',root).value.trim();if(!path)throw new Error('Choose a project path');await api('/api/project/open',{method:'POST',body:{path}});close();toast('Project opened');await refreshState({quiet:false});OculoImport.projectReady()}catch(e){showError(e,'Could not open project')}};
  });
}

function openAddSession() { openProjectSessionManager({focus:'sessions'}); }

function openProjectSessionManager({focus='sessions'}={}) {
  const ws=projectWorkspace();const sessions=ws.sessions||[];const profiles=ws.calibration_profiles||[];const sets=ws.calibration_sets||[];const selected=ws.selected_session_id||selectedSessionId();
  const sessionRows=sessions.map(row=>{const binding=row.calibration_binding||{};const bound=sets.find(item=>item.id===binding.calibration_set_id);return `<article class="project-session-row ${row.selected?'selected':''}"><div><span>${esc(row.session_code)}</span><strong>${esc(row.label)}</strong><small>${row.interaction_count} Interactions · ${row.asset_count} assets · ${row.artifact_count} results</small><small class="session-calibration-binding">Calibration: ${bound?`${esc(bound.code)} · revision ${esc(bound.active_revision_number??'—')}`:'Not assigned'}</small><code>${esc(row.folder)}</code></div><div class="project-session-actions"><button class="soft-button compact project-select-session" data-session-id="${esc(row.session_id)}">${row.selected?'Current':'Open'}</button><button class="soft-button compact project-open-session-folder" data-path="${esc(row.folder)}">Folder</button></div></article>`}).join('');
  const sourceOptions=sessions.map(row=>`<option value="${esc(row.session_id)}" ${row.session_id===selected?'selected':''}>${esc(row.session_code)}</option>`).join('');
  const profileOptions=profiles.map(row=>`<option value="${esc(row.session_id+'::'+row.artifact_id)}">${esc(row.hardware_id||'Calibration')} · ${esc(row.label)} · ${esc(row.session_label)}</option>`).join('');
  const setCards=sets.map(row=>{
    const assigned=new Set(row.applicable_session_ids||[]);
    const sessionChecks=sessions.map(item=>`<label class="check-row"><input type="checkbox" class="calset-session-check" data-set-id="${esc(row.id)}" value="${esc(item.session_id)}" ${assigned.has(item.session_id)?'checked':''}><span>${esc(item.session_code)}</span></label>`).join('');
    const revisionOptions=(row.revisions||[]).map(rev=>`<option value="${esc(rev.id)}" ${rev.active?'selected':''}>Revision ${esc(rev.revision_number)} · ${esc(rev.label)} · ${esc(rev.status)}</option>`).join('');
    const assignedLabels=(row.sessions||[]).map(item=>item.session_code||item.session_label).join(', ')||'No Sessions';
    return `<article class="calibration-set-manager-card ${focus==='calibration'?'focus':''}" data-set-id="${esc(row.id)}"><header><div><span class="eyebrow">Project Calibration Set</span><strong>${esc(row.label)}</strong><small>${esc(row.code)} · ${esc(row.purpose||'combined')}</small></div><span class="status-pill ${row.status==='ready'?'good':'warning'}">${esc(row.status||'draft')}</span></header><div class="calibration-set-manager-metrics"><span><b>${row.source_count||0}</b><small>source files</small></span><span><b>${row.profile_count||0}</b><small>profiles</small></span><span><b>${(row.applicable_session_ids||[]).length}</b><small>Sessions</small></span></div><div class="field"><label>Active revision</label><select class="calset-revision-select" data-set-id="${esc(row.id)}">${revisionOptions}</select></div><div class="field"><label>Sessions inheriting this Set</label><div class="check-list compact-check-list">${sessionChecks}</div><small>${esc(assignedLabels)}</small></div><div class="form-actions wrap"><button class="primary-button calset-save-sessions" data-set-id="${esc(row.id)}">Save Session assignment</button><button class="soft-button calset-new-revision" data-set-id="${esc(row.id)}">New revision</button><button class="soft-button calset-activate-revision" data-set-id="${esc(row.id)}">Activate selected revision</button></div><div class="field"><label>Attach an existing profile to this revision</label><select class="calset-profile-select" data-set-id="${esc(row.id)}"><option value="">Select a generated/imported profile</option>${profileOptions}</select></div><div class="form-actions wrap"><button class="soft-button calset-attach-profile" data-set-id="${esc(row.id)}">Attach profile</button><button class="soft-button calset-materialize" data-set-id="${esc(row.id)}">Refresh inherited references</button><button class="primary-button calset-build-profiles" data-set-id="${esc(row.id)}">${row.physical_profiles_ready?'Review physical profiles':'Build physical profiles'}</button><button class="soft-button calset-import-sources" data-set-id="${esc(row.id)}" data-set-code="${esc(row.code)}">Import Calibration files</button><button class="soft-button calset-open-folder" data-path="${esc(row.folder)}">Open Set folder</button></div></article>`;
  }).join('');
  const createSessionChecks=sessions.map(row=>`<label class="check-row"><input type="checkbox" class="new-calset-session" value="${esc(row.session_id)}" checked><span>${esc(row.session_code)}</span></label>`).join('');
  modal('Project, Sessions & Calibration Sets',`<div class="project-manager" data-focus="${esc(focus)}">
    <header class="project-manager-head"><div><span>Project root</span><strong>${esc(project()?.root||'')}</strong><small>${ws.folder_layout_version>=3?'Project Calibration Sets + Session folders':ws.folder_layout_version>=2?'Session folders · legacy shared Calibration':'Legacy folder layout'}</small></div><button class="soft-button" id="project-manager-open-root">Open Project folder</button></header>
    <div class="project-manager-grid"><section><div class="iw-section-head compact"><div><div class="eyebrow">Sessions</div><h3>Project Session map</h3></div><span class="wh-count">${sessions.length}</span></div><div class="project-session-list">${sessionRows||'<div class="empty-tree">No Sessions</div>'}</div><div class="project-manager-add"><div class="field"><label>Add Session range or list</label><input id="manager-session-spec" placeholder="S008-S010 or S008, S009"></div><button class="primary-button" id="manager-add-sessions">Add Sessions</button></div></section>
      <aside><div class="project-manager-card"><span>Interaction planning</span><strong>Pre-create I-codes without importing fake data</strong><div class="field"><label>Session</label><select id="manager-plan-session">${sourceOptions}</select></div><div class="field"><label>Interactions to add</label><input id="manager-interaction-spec" placeholder="I01-I06"></div><button class="soft-button full" id="manager-plan-interactions">Add to Interaction plan</button></div><div class="project-manager-card"><span>Folder contract</span><code>calibration_sets/CAL.../revisions/</code><code>sessions/S###/imports</code><code>sessions/S###/derived</code><code>sessions/S###/reports</code><small>Calibration recordings are registered once at Project level. Sessions receive references and a revision binding; Interaction files remain Session-local.</small></div></aside></div>
    <section class="project-calibration-sets"><div class="iw-section-head compact"><div><div class="eyebrow">Project-level calibration</div><h3>Calibration Sets → multiple Sessions</h3></div><button class="soft-button" id="project-open-calibration-sets-folder">Open Calibration Sets folder</button></div><p>Each Set owns one canonical copy/reference of its AprilTag recordings and profile lineage. Assigning a Set to S004–S007 does not duplicate raw files and never copies Interaction clocks.</p><div class="calibration-set-manager-grid">${setCards||'<div class="empty-tree">No Calibration Set exists yet. Create one below.</div>'}</div>
      <details class="calibration-set-create" ${sets.length?'':'open'}><summary>Create another Calibration Set</summary><div class="form-grid"><div class="field"><label>Set ID</label><input id="manager-calset-code" placeholder="CAL_S004_S007"></div><div class="field"><label>Display name</label><input id="manager-calset-label" placeholder="Calibration · S004–S007"></div><div class="field full-span"><label>Purpose</label><select id="manager-calset-purpose"><option value="combined">Hardware + known-target gaze</option><option value="hardware">Hardware camera-to-tracker only</option><option value="known_target_gaze">Known-target gaze only</option></select></div><div class="field full-span"><label>Applicable Sessions</label><div class="check-list compact-check-list">${createSessionChecks}</div></div></div><div class="form-actions"><button class="primary-button" id="manager-create-calset">Create Calibration Set</button></div></details>
      ${profiles.length?`<details class="legacy-calibration-library"><summary>Legacy profile-copy workflow · recovery only</summary><p>Older projects may contain profiles copied between Sessions. Attach those profiles to a Calibration Set above to establish one revisioned source of truth.</p></details>`:''}
    </section>
  </div>`,(root,close)=>{
    const reopen=async()=>{close();await refreshState({quiet:false,fullRender:true});openProjectSessionManager({focus});};
    $('#project-manager-open-root',root).onclick=()=>openLocalPath(project().root);
    $('#project-open-calibration-sets-folder',root).onclick=()=>openLocalPath(ws.calibration_sets_folder||joinLocalPath(project().root,'calibration_sets'));
    $$('.project-open-session-folder',root).forEach(button=>button.onclick=()=>openLocalPath(button.dataset.path));
    $$('.project-select-session',root).forEach(button=>button.onclick=async()=>{try{await api(`/api/session/${encodeURIComponent(button.dataset.sessionId)}/select`,{method:'POST'});close();await refreshState({quiet:false,fullRender:true});}catch(error){showError(error,'Could not open Session')}});
    $('#manager-add-sessions',root).onclick=async()=>{try{const labels=parseSessionSpecification($('#manager-session-spec',root).value);if(!labels.length)throw new Error('Enter at least one Session');await api('/api/sessions/batch',{method:'POST',body:{labels,select_first:false}});toast('Sessions added',labels.join(', '));await reopen();}catch(error){showError(error,'Could not add Sessions')}};
    $('#manager-plan-interactions',root).onclick=async()=>{try{const ids=parseInteractionSpecification($('#manager-interaction-spec',root).value);if(!ids.length)throw new Error('Enter at least one Interaction');const sid=$('#manager-plan-session',root).value;await api(`/api/session/${encodeURIComponent(sid)}/interactions/plan`,{method:'POST',body:{interaction_ids:ids,replace:false}});toast('Interaction plan updated',ids.join(', '));await reopen();}catch(error){showError(error,'Could not update Interaction plan')}};
    $('#manager-create-calset',root)?.addEventListener('click',async()=>{try{const sessionIds=$$('.new-calset-session:checked',root).map(input=>input.value);if(!sessionIds.length)throw new Error('Assign the new Set to at least one Session');const code=$('#manager-calset-code',root).value.trim();if(!code)throw new Error('Enter a Calibration Set ID');await api('/api/project/calibration-sets',{method:'POST',body:{code,label:$('#manager-calset-label',root).value.trim()||code,purpose:$('#manager-calset-purpose',root).value,applicable_session_ids:sessionIds,owner_session_id:sessionIds[0],notes:''}});toast('Calibration Set created',code);await reopen();}catch(error){showError(error,'Could not create Calibration Set')}});
    $$('.calset-save-sessions',root).forEach(button=>button.onclick=async()=>{try{const setId=button.dataset.setId;const sessionIds=$$(`.calset-session-check[data-set-id="${CSS.escape(setId)}"]:checked`,root).map(input=>input.value);const revisionId=$(`.calset-revision-select[data-set-id="${CSS.escape(setId)}"]`,root)?.value||null;await api(`/api/project/calibration-sets/${encodeURIComponent(setId)}/sessions`,{method:'POST',body:{session_ids:sessionIds,revision_id:revisionId}});toast('Session assignment saved',`${sessionIds.length} Session(s)`);await reopen();}catch(error){showError(error,'Could not assign Sessions')}});
    $$('.calset-new-revision',root).forEach(button=>button.onclick=async()=>{try{const setId=button.dataset.setId;await api(`/api/project/calibration-sets/${encodeURIComponent(setId)}/revisions`,{method:'POST',body:{label:null,notes:'Created from Project manager',inherit_active:true,activate:true}});toast('Calibration revision created');await reopen();}catch(error){showError(error,'Could not create revision')}});
    $$('.calset-activate-revision',root).forEach(button=>button.onclick=async()=>{try{const setId=button.dataset.setId;const revisionId=$(`.calset-revision-select[data-set-id="${CSS.escape(setId)}"]`,root).value;await api(`/api/project/calibration-sets/${encodeURIComponent(setId)}/revisions/activate`,{method:'POST',body:{revision_id:revisionId}});toast('Calibration revision activated');await reopen();}catch(error){showError(error,'Could not activate revision')}});
    $$('.calset-attach-profile',root).forEach(button=>button.onclick=async()=>{try{const setId=button.dataset.setId;const value=$(`.calset-profile-select[data-set-id="${CSS.escape(setId)}"]`,root).value;if(!value)throw new Error('Select a profile');const [sourceSessionId,profileArtifactId]=value.split('::');const revisionId=$(`.calset-revision-select[data-set-id="${CSS.escape(setId)}"]`,root).value;await api(`/api/project/calibration-sets/${encodeURIComponent(setId)}/profiles`,{method:'POST',body:{source_session_id:sourceSessionId,profile_artifact_id:profileArtifactId,revision_id:revisionId}});toast('Profile attached to Calibration Set');await reopen();}catch(error){showError(error,'Could not attach profile')}});
    $$('.calset-materialize',root).forEach(button=>button.onclick=async()=>{try{const setId=button.dataset.setId;const revisionId=$(`.calset-revision-select[data-set-id="${CSS.escape(setId)}"]`,root).value;await api(`/api/project/calibration-sets/${encodeURIComponent(setId)}/materialize`,{method:'POST',body:{session_ids:null,revision_id:revisionId}});toast('Inherited Calibration references refreshed');await reopen();}catch(error){showError(error,'Could not refresh inherited Calibration')}});
    $$('.calset-open-folder',root).forEach(button=>button.onclick=()=>openLocalPath(button.dataset.path));
    $$('.calset-build-profiles',root).forEach(button=>button.onclick=()=>{const setId=button.dataset.setId||'';close();setTimeout(()=>openCalibrationSetProfileBuilder({setId}),0);});
    $$('.calset-import-sources',root).forEach(button=>button.onclick=()=>{const row=calibrationSetRecord(button.dataset.setId);const setCode=row?.code||button.dataset.setCode||'';close();setTimeout(()=>openCalibrationSetImportDialog({setCode}),0);});
  });
}

async function submitAsset() {
  OculoImport.queuePathFromForm();
}

function openAssetIdentityEditor(assetIds) {
  const ids=[...new Set((assetIds||[]).filter(Boolean))];
  if(!ids.length)return;
  const records=ids.map(id=>session()?.assets?.[id]).filter(Boolean);
  if(!records.length)return;
  const views=records.map(assetIdentity);
  const common=(key)=>views.every(v=>v[key]===views[0][key])?views[0][key]:'';
  const participant=common('participant_id'), interaction=common('interaction_id'), role=common('role_id');
  modal(ids.length===1?'Edit asset identity':`Edit identity for ${ids.length} assets`, `<div class="identity-editor-summary">${records.map(a=>`<div><strong>${esc(a.label)}</strong><span>${esc(assetIdentity(a).participant_id||'Unassigned')} · ${esc(assetIdentity(a).interaction_id||'—')} · ${esc(assetIdentity(a).role_id||'—')}</span></div>`).join('')}</div>
    <div class="form-grid"><div class="field"><label>Participant ID</label><input id="identity-participant" value="${esc(participant)}" placeholder="P01 · blank clears"></div><div class="field"><label>Interaction</label><input id="identity-interaction" value="${esc(interaction)}" placeholder="I01"></div><div class="field"><label>Role</label><input id="identity-role" value="${esc(role)}" placeholder="R01"></div></div>
    <div class="form-help">Participant is the anonymous person ID. Role is the interaction-local R01/R02 label. ETM_A/ETM_B should normally keep Participant ID blank until wearer resolution.</div>
    <div class="form-actions"><button class="primary-button" id="identity-save">Save identity</button></div>`,(root,close)=>{
      $('#identity-save',root).onclick=async()=>{try{
        const body={participant_id:$('#identity-participant',root).value.trim()||null,interaction_id:$('#identity-interaction',root).value.trim()||null,role_id:$('#identity-role',root).value.trim()||null};
        if(ids.length===1)await api(`/api/assets/${selectedSessionId()}/${ids[0]}/identity`,{method:'PATCH',body});
        else await api(`/api/assets/${selectedSessionId()}/identity/batch`,{method:'POST',body:{...body,asset_ids:ids}});
        close();appState.selectedAssetIds=[];toast('Asset identity saved');await refreshState({quiet:false});
      }catch(error){showError(error,'Could not save asset identity')}
    };
  });
}

function confirmRemoveAsset(id) {
  const a=session()?.assets?.[id];
  modal('Remove asset reference', `<p>Remove <strong>${esc(a?.label||id)}</strong> from this session? The raw source will not be deleted.</p><div class="form-actions"><button class="danger-button" id="remove-asset-submit">Remove reference</button></div>`,(root,close)=>{$('#remove-asset-submit',root).onclick=async()=>{try{await api(`/api/assets/${selectedSessionId()}/${id}`,{method:'DELETE'});close();toast('Asset reference removed');await refreshState({quiet:false})}catch(e){showError(e,'Could not remove asset')}}});
}

async function submitTimeMap() {
  try {
    captureAnchorRows();
    const source=$('#sync-source').value,target=$('#sync-target').value;
    if(!source||!target)throw new Error('Select both source and target clocks');
    if(source===target)throw new Error('Source and target clocks must differ');
    if(!appState.timeAnchors.length)throw new Error('Add at least one shared-event anchor');
    const returnPage=appState.syncPrefill?.return_page||null;
    const result=await api('/api/time-maps/fit',{method:'POST',body:{session_id:selectedSessionId(),source_domain_id:source,target_domain_id:target,anchors:appState.timeAnchors,method:$('#sync-method').value}});
    const mapping=result?.time_map;
    const count=Number(mapping?.quality?.anchor_count||appState.timeAnchors.length);
    const rms=Number(mapping?.quality?.rms_residual_ns||0)/1e6;
    toast('Verified time map saved',count===1?'Offset fitted from one anchor; drift is not estimated.':`${count} anchors · RMS ${rms.toFixed(3)} ms`);
    appState.syncPrefill=null;
    await refreshState({quiet:false});
    if(returnPage)navigate(returnPage);
  }catch(e){showError(e,'Time-map fitting failed')}
}

async function submitFrameMap() {
  try { const native=$('#frame-native').value,cloud=$('#frame-cloud').value;if(!native||!cloud)throw new Error('Select Native and Cloud assets');
    await api('/api/jobs/frame-map',{method:'POST',body:{session_id:selectedSessionId(),native_asset_id:native,cloud_asset_id:cloud,maximum_error_ms:Number($('#frame-error').value)}});toast('Frame-map job queued');await refreshState({quiet:false});
  }catch(e){showError(e,'Could not start frame-map job')}
}

async function submitClip() {
  try { await api('/api/clips/add',{method:'POST',body:{session_id:selectedSessionId(),label:$('#clip-label').value.trim(),start_timecode:$('#clip-start').value.trim(),end_timecode:$('#clip-end').value.trim(),notes:$('#clip-notes').value.trim(),frame_rate:30}});toast('Clip saved');await refreshState({quiet:false}); }
  catch(e){showError(e,'Could not save clip')}
}

async function submitClipJob() {
  try { const clip=$('#export-clip').value;const assetIds=$$('.clip-asset-check:checked').map(x=>x.value);if(!clip)throw new Error('Select a clip');if(!assetIds.length)throw new Error('Select at least one source asset');
    await api('/api/jobs/clip',{method:'POST',body:{session_id:selectedSessionId(),clip_id:clip,asset_ids:assetIds,interval_policy:$('#interval-policy').value}});toast('Multimodal export queued');await refreshState({quiet:false});
  }catch(e){showError(e,'Could not start clip export')}
}

async function submitFaceMapperFilter() {
  const button=$('#run-facemapper-filter');
  try {
    const trim=$('#fm-filter-trim').value;
    if(!trim)throw new Error('Select a Native-reference trim artifact');
    const body={
      session_id:selectedSessionId(),
      native_trim_artifact_id:trim,
      label:$('#fm-filter-label').value.trim()||'Filtered Face Mapper',
      large_box_area_ratio:Number($('#fm-filter-large-area').value),
      maximum_area_ratio:Number($('#fm-filter-max-area').value),
      minimum_landmark_span_x_ratio:Number($('#fm-filter-span-x').value),
      minimum_landmark_span_y_ratio:Number($('#fm-filter-span-y').value),
      minimum_interocular_ratio:Number($('#fm-filter-eye-span').value),
      minimum_landmark_in_box_fraction:Number($('#fm-filter-inside').value),
      maximum_landmark_center_offset:Number($('#fm-filter-centre').value),
      maximum_scale_jump_ratio:Number($('#fm-filter-scale-jump').value),
      minimum_track_frames:Number($('#fm-filter-track-frames').value),
      render_preview:$('#fm-filter-preview').dataset.on==='true',
      render_source_preview:$('#fm-filter-source-preview')?.dataset.on!=='false',
      source_bilateral_convention:$('#fm-filter-bilateral')?.value||'pupil_image_side',
      preview_sample_count:Number($('#fm-filter-preview-samples').value),
      preview_segment_seconds:Number($('#fm-filter-preview-seconds').value),
      preview_fps:15,
      preview_scope:$('#fm-filter-preview-scope')?.value||'sampled',
      ffprobe_path:(typeof nativeDraft==='function'?nativeDraft().ffprobe?.trim():'')||null,
    };
    if(button){button.disabled=true;button.textContent='Filtering FaceMapper candidates…';}
    if(body.render_preview){
      const check=await api('/api/media-tools/check',{method:'POST',body:{preserve_audio:false,ffprobe_path:body.ffprobe_path}});
      if(!check.ready)throw new Error(check.error||'FFmpeg is required for optional encoded QC MP4s');
      body.ffmpeg_path=check.preflight.ffmpeg.path;
    }else body.ffmpeg_path=null;
    await api('/api/jobs/facemapper-filter',{method:'POST',body});
    toast('FaceMapper QC/filter queued',body.render_preview?'Filter tables and optional encoded QC MP4s are being created.':'Filter tables are being created. Review will use the original Native video with a live Canvas overlay.');
    await refreshState({quiet:false});
  }catch(e){showError(e,'Could not filter FaceMapper candidates')}
  finally{if(button?.isConnected){button.disabled=false;button.textContent='Filter FaceMapper candidates before Privacy';}}
}

function bindPrivacyWorkflowDefaults() {
  const filter=$('#privacy-filter'), trim=$('#privacy-native-trim-direct'), fps=$('#privacy-output-fps'), label=$('#privacy-output-label');
  if(label){
    label.dataset.autoLabel='true';
    label.addEventListener('input',()=>{label.dataset.autoLabel='false';});
  }
  const updateLabel=()=>{
    if(!label||label.dataset.autoLabel==='false')return;
    const selectedTrim=artifacts(['native_reference_trim']).find(row=>row.id===trim?.value);
    const value=Number(fps?.value||30);
    label.value=`${selectedTrim?.label||'privacy'}_privacy_${value}fps`;
  };
  if(filter){
    filter.addEventListener('change',()=>{
      const selected=artifacts(['face_mapper_filter']).find(row=>row.id===filter.value);
      const parentTrim=selected?.parameters?.native_trim_artifact_id||'';
      if(parentTrim&&trim)trim.value=parentTrim;
      updateLabel();
    });
  }
  trim?.addEventListener('change',updateLabel);
  fps?.addEventListener('change',updateLabel);

  const fmTrim=$('#fm-filter-trim'), fmLabel=$('#fm-filter-label');
  if(fmLabel){
    fmLabel.dataset.autoLabel='true';
    fmLabel.addEventListener('input',()=>{fmLabel.dataset.autoLabel='false';});
  }
  fmTrim?.addEventListener('change',()=>{
    if(!fmLabel||fmLabel.dataset.autoLabel==='false')return;
    const selected=artifacts(['native_reference_trim']).find(row=>row.id===fmTrim.value);
    fmLabel.value=selected?.label?`${selected.label}_facemapper_filter`:'Filtered Face Mapper';
  });
}

function bindFaceMapperPreviewScope(){
  const scope=$('#fm-filter-preview-scope'), sampled=$('#fm-filter-sampled-settings');
  if(!scope)return;
  const update=()=>{if(sampled)sampled.hidden=scope.value==='full';};
  scope.addEventListener('change',update);update();
}

function bindPrivacyMaskPolicy(){
  const policy=$('#privacy-candidate-policy'), detector=$('#privacy-detector'), gap=$('#privacy-gap'), edge=$('#privacy-edge-gap'), padding=$('#privacy-padding'), repair=$('#privacy-repair-options'), help=$('#privacy-mask-policy-help');
  if(!policy)return;
  let previous=policy.value;
  const update=(changed=false)=>{
    const mode=policy.value;
    const repairMode=mode==='accepted_repair';
    const exactMode=mode==='accepted_only';
    if(repairMode||exactMode){
      if(detector)detector.value='facemapper';
      if(changed && previous!==mode && padding)padding.value='0';
    }
    if(exactMode){
      if(gap)gap.value='0';
      if(edge)edge.value='0';
    } else if(repairMode && changed && previous!==mode){
      if(gap && Number(gap.value)<=0)gap.value='3';
      if(edge && Number(edge.value)<=0)edge.value='1';
    }
    if(detector)detector.disabled=repairMode||exactMode;
    if(repair)repair.hidden=!repairMode;
    [gap,edge].forEach(control=>{if(control)control.disabled=!repairMode;});
    if(help){
      help.innerHTML=repairMode
        ?'<strong>Accepted + short-gap repair</strong> never reads amber review candidates. Only brief, motion-plausible holes in accepted tracks are added.'
        :exactMode
          ?'<strong>Accepted only</strong> uses the green QC boxes exactly. No review candidates, Haar detections, temporal repair, or automatic padding are added.'
          :'<strong>Accepted + review</strong> is conservative for publication but may blur amber false positives. Haar supplementation remains available in advanced settings.';
    }
    previous=mode;
  };
  policy.addEventListener('change',()=>update(true));
  update(false);
}

function confirmRemoveArtifact(artifactId){
  const row=artifacts().find(item=>item.id===artifactId);
  if(!row)return;
  modal('Remove generated Privacy video', `<p>Move <strong>${esc(row.label)}</strong> out of the active project?</p><div class="warning-card"><strong>Raw recordings and the FaceMapper filter are not deleted.</strong><p>The generated files are moved to <code>.trash/artifacts</code> inside the project. OculoStudio refuses removal if a downstream AOI or results artifact still depends on this video.</p></div><div class="form-actions"><button class="danger-button" id="remove-artifact-submit">Move to Project Trash</button></div>`,(root,close)=>{
    $('#remove-artifact-submit',root).onclick=async()=>{
      try{
        const result=await api(`/api/artifacts/${encodeURIComponent(selectedSessionId())}/${encodeURIComponent(artifactId)}`,{method:'DELETE'});
        close();toast('Privacy video removed',result?.removed?.trash_path||'Moved to project trash');await refreshState({quiet:false});
      }catch(error){showError(error,'Could not remove generated Privacy video')}
    };
  });
}

function privacyMaskSettings() {
  return {
    mode:$('#privacy-mode').value,
    strength:Number($('#privacy-strength').value),
    padding_percent:Number($('#privacy-padding').value),
    shape:'rectangle',
    gap_fill_frames:Number($('#privacy-gap').value),
    edge_fill_frames:Number($('#privacy-edge-gap')?.value||0),
    repair_maximum_center_shift_per_frame:Number($('#privacy-repair-motion')?.value||0.85),
    repair_maximum_scale_ratio:Number($('#privacy-repair-scale')?.value||1.75),
    repair_minimum_track_observations:Number($('#privacy-repair-min-track')?.value||3),
    timestamp_tolerance_ms:Number($('#privacy-tolerance').value),
    detector_mode:$('#privacy-detector').value,
    candidate_policy:$('#privacy-candidate-policy')?.value||'accepted_repair',
    exact_filter_match:false,
    coverage_policy:$('#privacy-policy').value,
    preserve_audio:$('#privacy-audio').dataset.on==='true',
    ffprobe_path:(typeof nativeDraft==='function' ? nativeDraft().ffprobe?.trim() : '')||null,
  };
}

function matchingPrivacyMaskPreview(nativeId,trimId,filterId,settings,outputFps){
  const numericKeys=['padding_percent','gap_fill_frames','edge_fill_frames','repair_maximum_center_shift_per_frame','repair_maximum_scale_ratio','repair_minimum_track_observations','timestamp_tolerance_ms'];
  return rowsForActiveWorkflowContext(artifacts(['privacy_mask_preview'])).slice().reverse().find(row=>{
    const p=row.parameters||{};
    if(p.native_asset_id!==nativeId||p.native_trim_artifact_id!==trimId||p.facemapper_filter_artifact_id!==filterId)return false;
    if(String(p.preview_scope||'')!=='full'||Number(p.preview_fps)!==Number(outputFps))return false;
    return numericKeys.every(key=>Number.isFinite(Number(p[key]))&&Math.abs(Number(p[key])-Number(settings[key]))<=1e-9);
  })||null;
}

async function preflightPrivacyPayload(payload) {
  const check=await api('/api/media-tools/check',{method:'POST',body:{preserve_audio:payload.preserve_audio,ffprobe_path:payload.ffprobe_path}});
  const status=$('#media-tools-message');
  if(!check.ready){
    if(status){status.textContent=check.error;status.classList.add('media-tools-error');}
    throw new Error(check.error);
  }
  payload.ffmpeg_path=check.preflight.ffmpeg.path;
  if(status){status.textContent='Encoder test passed · '+check.preflight.ffmpeg.source;status.classList.remove('media-tools-error');}
  return payload;
}

async function submitPrivacyMaskPreview() {
  const button=$('#run-privacy-mask-preview');
  try{
    const native=$('#privacy-native')?.value||'';
    const trim=$('#privacy-native-trim-direct')?.value||'';
    const filter=$('#privacy-filter')?.value||'';
    if(!native)throw new Error('Select a Native recording');
    if(!trim)throw new Error('Select the Native-reference trim');
    if(!filter)throw new Error('Select a FaceMapper QC/filter artifact');
    const settings=privacyMaskSettings();
    const payload={
      session_id:selectedSessionId(),
      native_asset_id:native,
      native_trim_artifact_id:trim,
      facemapper_filter_artifact_id:filter,
      label:`${$('#privacy-output-label')?.value.trim()||'privacy'}_accepted_repair_preview`,
      output_fps:Number($('#privacy-repair-preview-fps')?.value||30),
      preview_scope:$('#privacy-repair-preview-scope')?.value||'full',
      preview_sample_count:9,
      preview_segment_seconds:6,
      padding_percent:settings.padding_percent,
      gap_fill_frames:settings.gap_fill_frames,
      edge_fill_frames:settings.edge_fill_frames,
      repair_maximum_center_shift_per_frame:settings.repair_maximum_center_shift_per_frame,
      repair_maximum_scale_ratio:settings.repair_maximum_scale_ratio,
      repair_minimum_track_observations:settings.repair_minimum_track_observations,
      timestamp_tolerance_ms:settings.timestamp_tolerance_ms,
      ffprobe_path:settings.ffprobe_path,
    };
    if(button){button.disabled=true;button.textContent='Rendering accepted + repair preview…';}
    const queued=await api('/api/jobs/privacy-mask-preview',{method:'POST',body:payload});
    appState.pendingPrivacyJobId=queued?.job?.id||null;
    toast('Mask preview queued','Only green accepted detections seed the repair. Amber review candidates are not loaded.');
    await refreshState({quiet:false});
  }catch(error){showError(error,'Could not render accepted + repair mask preview');}
  finally{if(button?.isConnected){button.disabled=false;button.textContent='Render accepted + repair mask preview';}}
}

async function submitPrivacyFinal() {
  const button=$('#run-privacy-final');
  try {
    const native=$('#privacy-native').value;
    const trim=$('#privacy-native-trim-direct').value;
    if(!native)throw new Error('Select a Native recording');
    if(!trim)throw new Error('Select the Native-reference trim for this R01/R02 video');
    const settings=privacyMaskSettings();
    const face=$('#privacy-face').value||null, filter=$('#privacy-filter').value||null;
    if(['facemapper','hybrid'].includes(settings.detector_mode)&&!face&&!filter){
      throw new Error('Select a filtered FaceMapper artifact or a raw Face Mapper asset');
    }
    const outputFps=Number($('#privacy-output-fps').value);
    const maskPreview=settings.candidate_policy==='accepted_repair'
      ? matchingPrivacyMaskPreview(native,trim,filter,settings,outputFps)
      : null;
    if(settings.candidate_policy==='accepted_repair'&&!maskPreview){
      throw new Error('Render and review a full accepted + repair mask preview at the same FPS as the final video, using the current settings, before export');
    }
    const payload={
      session_id:selectedSessionId(),
      native_asset_id:native,
      native_trim_artifact_id:trim,
      facemapper_asset_id:filter?null:face,
      facemapper_filter_artifact_id:filter,
      privacy_mask_preview_artifact_id:maskPreview?.id||null,
      output_fps:outputFps,
      label:$('#privacy-output-label').value.trim()||'synchronized_privacy_video',
      ...settings,
    };
    if(button){button.disabled=true;button.textContent='Checking encoder and final frame map…';}
    await preflightPrivacyPayload(payload);
    if(button?.isConnected)button.textContent='Creating synchronized Privacy video…';
    const queued=await api('/api/jobs/privacy-final',{method:'POST',body:payload});
    appState.pendingPrivacyJobId=queued?.job?.id||null;
    toast('Synchronized Privacy job queued','The final fixed-FPS video, audio, Native frame map and mask audit will be created in one processing job.');
    await refreshState({quiet:false});
  }catch(e){showError(e,'Could not create the synchronized Privacy video')}
  finally{if(button?.isConnected){button.disabled=false;button.textContent='Create synchronized Privacy video';}}
}

async function submitPrivacyFull() {
  const button=$('#run-privacy-full');
  try {
    const native=$('#privacy-native').value;
    if(!native)throw new Error('Select a Native recording');
    const settings=privacyMaskSettings();
    const face=$('#privacy-face').value||null;
    if(['facemapper','hybrid'].includes(settings.detector_mode)&&!face){
      throw new Error('A full-recording source requires the Raw Face Mapper asset; a trim-scoped filtered package cannot cover the full recording.');
    }
    const payload={
      session_id:selectedSessionId(),
      native_asset_id:native,
      facemapper_asset_id:face,
      facemapper_filter_artifact_id:null,
      native_trim_artifact_id:null,
      clip_id:null,
      ...settings,
      // A raw full-recording FaceMapper source has no accepted/review QC
      // classification. Accepted-only repair is therefore intentionally
      // limited to the normal trim-scoped filtered workflow above.
      candidate_policy:'accepted_review',
      exact_filter_match:false,
      edge_fill_frames:0,
    };
    if(button){button.disabled=true;button.textContent='Checking full-recording source…';}
    await preflightPrivacyPayload(payload);
    if(button?.isConnected)button.textContent='Creating full-recording Privacy source…';
    const queued=await api('/api/jobs/privacy',{method:'POST',body:payload});
    appState.pendingPrivacyJobId=queued?.job?.id||null;
    toast('Full-recording Privacy source queued','This advanced source can later be reused for several different trims.');
    await refreshState({quiet:false});
  }catch(e){showError(e,'Could not create the full-recording Privacy source')}
  finally{if(button?.isConnected){button.disabled=false;button.textContent='Create full-recording Privacy source';}}
}

async function submitPrivacyClip() {
  const button=$('#run-privacy-clip');
  try {
    const reusable=rowsForActiveWorkflowContext(artifacts(['privacy_video'])).filter(isReusablePrivacySource).slice().sort((a,b)=>String(b.created_at||'').localeCompare(String(a.created_at||'')));
    let source=$('#privacy-clip-source')?.value||'';
    let sourceArtifact=reusable.find(row=>row.id===source)||reusable[0]||null;
    if(!sourceArtifact){
      const create=$('#run-privacy-full');
      create?.scrollIntoView({behavior:'smooth',block:'center'});
      create?.classList.add('attention-pulse');
      setTimeout(()=>create?.classList.remove('attention-pulse'),2200);
      throw new Error('No reusable full-recording Privacy source exists. Create one in Advanced A, or use the recommended one-step workflow above.');
    }
    source=sourceArtifact.id;
    const trim=$('#privacy-native-trim').value||null;
    const body={
      session_id:selectedSessionId(),
      privacy_artifact_id:source,
      native_asset_id:$('#privacy-clip-native').value||null,
      native_trim_artifact_id:trim,
      start_timecode:trim?null:$('#privacy-clip-start').value.trim(),
      duration_timecode:trim?null:$('#privacy-clip-duration').value.trim(),
      timecode_fps:Number($('#privacy-clip-timecode-fps').value),
      output_fps:Number($('#privacy-clip-output-fps').value),
      label:$('#privacy-clip-label').value.trim()||'privacy_derived_clip',
      preserve_audio:$('#privacy-audio')?.dataset.on!=='false',
      ffprobe_path:(typeof nativeDraft==='function'?nativeDraft().ffprobe?.trim():'')||null,
    };
    if(button){button.disabled=true;button.textContent='Deriving synchronized clip…';}
    const queued=await api('/api/jobs/privacy-clip',{method:'POST',body});
    appState.pendingPrivacyJobId=queued?.job?.id||null;
    toast('Derived Privacy clip queued','The advanced exporter will reuse the existing full-recording masked source.');
    await refreshState({quiet:false});
  }catch(e){showError(e,'Could not derive the Privacy clip')}
  finally{if(button?.isConnected){button.disabled=false;button.textContent='Derive synchronized clip from source';}}
}

async function submitParticipantAssociation() {
  try {
    const trim=$('#association-trim').value,assignment=$('#association-etm').value;if(!trim||!assignment)throw new Error('Select a Native-reference trim and paired Calibration handoff');
    const activeRole=String(activeWorkflowContext().roleId||'').toUpperCase();
    const wearer=String($('#association-wearer').value||'').toUpperCase();
    if(activeRole&&wearer!==activeRole)throw new Error(`Active role is ${activeRole}, but participant association is configured for ${wearer}. Refresh this page before continuing.`);
    const filter=$('#association-filter').value||null;
    if(filter&&!recordUsesNativeTrim(session()?.artifacts?.[filter],trim))throw new Error('The selected FaceMapper filter belongs to a different Native-reference trim.');
    const overrideEnabled=$('#association-profile-override-enabled')?.dataset.on==='true';
    const profileOverride=overrideEnabled?($('#association-profile-override')?.value||null):null;
    await api('/api/jobs/participant-association',{method:'POST',body:{session_id:selectedSessionId(),native_trim_artifact_id:trim,etm_assignment_artifact_id:assignment,facemapper_filter_artifact_id:filter,wearer_role_id:wearer,calibration_profile_artifact_id:profileOverride,target_policy:$('#association-policy').value,alignment_mode:$('#association-alignment')?.value||'pretrimmed_shared_start',shared_start_offset_sec:0,face_tolerance_ms:Number($('#association-face-tolerance').value),maximum_normalized_center_error:Number($('#association-centre-error').value),minimum_assignment_margin:Number($('#association-margin').value),minimum_identity_confidence:Number($('#association-confidence').value),output_parquet:$('#association-parquet').dataset.on==='true',label:$('#association-label').value.trim()||'participant_association'}});
    toast('Participant association queued',`${wearer} will use its role-locked ETM/profile. A partner-only result with zero in-view partner projections will be rejected.`);await refreshState({quiet:false});
  }catch(e){showError(e,'Could not project participants and assign faces')}
}

async function submitFaceAOI() {
  try {
    const trim=$('#face-aoi-trim').value;if(!trim)throw new Error('Select a Native-reference trim artifact');
    const privacy=$('#face-aoi-privacy').value||null;
    const filter=$('#face-aoi-filter').value||null;
    const association=$('#face-aoi-association').value||null;
    const personPolicy=$('#face-aoi-person-policy').value;
    if(personPolicy!=='all_detected'&&!association)throw new Error('No participant association exists for this exact role and Native-reference trim. Run step 0 again.');
    if(association&&!recordUsesNativeTrim(session()?.artifacts?.[association],trim))throw new Error('The selected participant association belongs to a different Native-reference trim.');
    if(filter&&!recordUsesNativeTrim(session()?.artifacts?.[filter],trim))throw new Error('The selected FaceMapper filter belongs to a different Native-reference trim.');
    if(privacy&&!recordUsesNativeTrim(session()?.artifacts?.[privacy],trim))throw new Error('The selected blurred video belongs to a different Native-reference trim.');
    await api('/api/jobs/face-aoi',{method:'POST',body:{session_id:selectedSessionId(),native_trim_artifact_id:trim,facemapper_filter_artifact_id:filter,participant_association_artifact_id:association,target_person_policy:personPolicy,privacy_clip_artifact_id:privacy,
      label:$('#face-aoi-label').value.trim()||'face_aoi',face_tolerance_ms:Number($('#face-aoi-tolerance').value),
      face_geometry_time_policy:$('#face-aoi-time-policy')?.value||'track_interpolated',face_interpolation_max_gap_ms:Number($('#face-aoi-interpolation-gap')?.value||75),face_interpolation_max_motion_ratio:Number($('#face-aoi-interpolation-motion')?.value||1.25),
      face_left_right_policy:$('#face-aoi-lr').value,face_landmark_geometry_policy:$('#face-aoi-geometry')?.value||'as_exported',fine_feature_aoi_scale:Number($('#face-aoi-feature-scale')?.value||1.55),eye_feature_aoi_scale:Number($('#face-aoi-eye-scale')?.value||1.35),nose_region_mode:$('#face-aoi-nose-mode')?.value||'nasal_envelope',mouth_center_policy:$('#face-aoi-mouth-center')?.value||'adaptive_lip_envelope',face_edge_tolerance_policy:$('#face-aoi-edge-tolerance')?.value||'edge_inclusive',auto_review_policy:$('#face-aoi-auto-review')?.value||'adaptive',output_parquet:$('#face-aoi-parquet').dataset.on==='true',
      render_overlay:$('#face-aoi-overlay').dataset.on==='true'&&!!privacy,preserve_audio:true,
      ffprobe_path:(typeof nativeDraft==='function'?nativeDraft().ffprobe?.trim():'')||null}});
    toast('Face AOI job queued','Observed FaceMapper frames will be used directly or safely interpolated along the same accepted track at each gaze timestamp. Temporal provenance remains auditable.');await refreshState({quiet:false});
  }catch(e){showError(e,'Could not generate Face AOIs')}
}

async function submitBodyAOI() {
  try {
    const rays=$('#body-aoi-rays').value;if(!rays)throw new Error('Select the current-role Gaze 3D artifact');
    const assignment=$('#body-aoi-assignment').value||null;if(!assignment)throw new Error('Select the Calibration handoff / wearer assignment');
    const wearerBody=$('#body-aoi-wearer-body').value||null;
    const partnerBody=$('#body-aoi-partner-body').value||null;
    const includeSelf=$('#body-aoi-include-self').dataset.on==='true';
    if(!partnerBody)throw new Error('Select the partner Body C3D');
    if(includeSelf&&!wearerBody)throw new Error('Include self body is enabled; select the wearer Body C3D');
    if(wearerBody&&partnerBody&&wearerBody===partnerBody)throw new Error('Wearer and partner Body C3D must be different assets');
    let prefixes={};const text=$('#body-aoi-prefixes').value.trim();if(text){try{prefixes=JSON.parse(text)}catch(_){throw new Error('Marker prefixes must be valid JSON')}}
    const bodyAssets=[...(includeSelf&&wearerBody?[wearerBody]:[]),partnerBody].filter(Boolean);
    const queued=await api('/api/jobs/body-aoi',{method:'POST',body:{
      session_id:selectedSessionId(),gaze_rays_artifact_id:rays,timeseries_asset_id:$('#body-aoi-cloud').value||null,
      body_asset_ids:bodyAssets,wearer_body_asset_id:wearerBody,partner_body_asset_id:partnerBody,wearer_role_id:$('#body-aoi-wearer-role').value||null,
      include_self_body:includeSelf,exclude_self_head_neck:$('#body-aoi-exclude-self-head').dataset.on==='true',
      self_near_origin_exclusion_m:Number($('#body-aoi-self-origin').value),first_hit_occlusion:$('#body-aoi-first-hit').dataset.on==='true',
      output_granularity:$('#body-aoi-granularity').value,auto_review_policy:$('#body-aoi-auto-review')?.value||'semantic_guided',
      geometry_acceptance_policy:$('#body-aoi-geometry-policy')?.value||'required',minimum_torso_coverage:Number($('#body-aoi-min-torso')?.value||80)/100,minimum_body_frame_coverage:Number($('#body-aoi-min-body')?.value||90)/100,
      body_profile_path:$('#body-aoi-profile').value.trim()||null,
      entity_marker_prefixes:prefixes,etm_assignment_artifact_id:assignment,native_trim_artifact_id:$('#body-aoi-trim').value||null,
      max_marker_gap_sec:Number($('#body-aoi-gap').value),uncertainty_cone_deg:Number($('#body-aoi-cone').value),
      maximum_target_distance_m:Number($('#body-aoi-distance').value),scene_frame_tolerance_ms:Number($('#body-aoi-frame-tolerance').value),sample_clock_chunk_size:4096,
      label:$('#body-aoi-label').value.trim()||'body_aoi',output_parquet:$('#body-aoi-parquet').dataset.on==='true'
    }});
    appState.pendingBodyAOIJobId=queued?.job?.id||null;
    const reusedArtifactId=queued?.deduplication?.artifact_id||jobResultArtifactId(queued?.job);
    if(reusedArtifactId){appState.aoiReviewInitialMode='body';appState.aoiReviewInitialArtifactId=reusedArtifactId;}
    toast('Self + partner Body AOI queued','Each gaze ray will be intersected with self and partner body geometry interpolated at that exact gaze timestamp; video frames are used only for review display.');
    await refreshState({quiet:false});
  }catch(e){showError(e,'Could not generate self + partner Body AOIs')}
}

async function submitBodyReviewReduction(event) {
  try {
    const artifactId=event?.currentTarget?.dataset?.artifactId||'';
    if(!artifactId)throw new Error('Select an existing gaze-sample-clock Body AOI artifact');
    const source=bodyArtifactById(artifactId);
    if(!source)throw new Error('The selected Body AOI artifact is no longer available');
    if(!bodyArtifactReviewReady(source))throw new Error('The selected Body AOI did not pass torso geometry acceptance');
    const policy=$('#body-review-retune-policy')?.value||'semantic_guided';
    const queued=await api('/api/jobs/body-aoi-review-reduction',{method:'POST',body:{
      session_id:selectedSessionId(),body_aoi_artifact_id:artifactId,auto_review_policy:policy,
      label:`${source.label||'Body AOI'} · reduced review`,output_parquet:false
    }});
    appState.pendingBodyAOIJobId=queued?.job?.id||null;
    const reusedArtifactId=queued?.deduplication?.artifact_id||jobResultArtifactId(queued?.job);
    if(reusedArtifactId){appState.aoiReviewInitialMode='body';appState.aoiReviewInitialArtifactId=reusedArtifactId;}
    toast('Body review reduction queued','The saved 3D intersections will be re-evaluated without rerunning C3D interpolation or changing any automatic Body target.');
    await refreshState({quiet:false});
  }catch(e){showError(e,'Could not reduce the existing Body AOI review queue')}
}


async function submitObjectAOI() {
  try {
    const rays=$('#object-aoi-rays').value;if(!rays)throw new Error('Select a Gaze 3D artifact');
    const objectAssets=$$('.object-aoi-check:checked').map(x=>x.value);if(!objectAssets.length)throw new Error('Select at least one Object C3D / mocap asset');
    let overrides={};const raw=$('#object-aoi-overrides').value.trim();if(raw){try{overrides=JSON.parse(raw)}catch(_){throw new Error('Object ID overrides must be valid JSON')}}
    await api('/api/jobs/object-aoi',{method:'POST',body:{session_id:selectedSessionId(),gaze_rays_artifact_id:rays,timeseries_asset_id:$('#object-aoi-cloud').value||null,object_asset_ids:objectAssets,object_profile_path:$('#object-aoi-profile').value.trim()||null,object_id_overrides:overrides,native_trim_artifact_id:$('#object-aoi-trim').value||null,max_marker_gap_sec:0.1,uncertainty_cone_deg:1.5,maximum_target_distance_m:10,scene_frame_tolerance_ms:20,label:$('#object-aoi-label').value.trim()||'object_aoi',output_parquet:$('#object-aoi-parquet').dataset.on==='true'}});
    toast('Object AOI job queued','Marker-prop instance IDs will be intersected with the calibrated gaze rays.');await refreshState({quiet:false});
  }catch(e){showError(e,'Could not generate Object AOIs')}
}

async function submitTargetFusion() {
  try {
    const face=$('#fusion-face').value,body=$('#fusion-body').value;if(!face||!body)throw new Error('Select both Face AOI and a torso-verified Body AOI artifact');
    const bodyArtifact=bodyArtifactById(body);if(!bodyArtifactFusionReady(bodyArtifact))throw new Error('Selected Body AOI did not pass torso geometry acceptance. Regenerate it before fusion.');
    await api('/api/jobs/target-fusion',{method:'POST',body:{session_id:selectedSessionId(),face_aoi_artifact_id:face,body_aoi_artifact_id:body,object_aoi_artifact_id:$('#fusion-object').value||null,
      label:$('#fusion-label').value.trim()||'gaze_targets_full',output_parquet:$('#fusion-parquet').dataset.on==='true'}});
    toast('Target fusion queued','Face, Body and optional Object evidence remain in separate columns.');await refreshState({quiet:false});
  }catch(e){showError(e,'Could not fuse Face and Body AOIs')}
}

async function submitDyadicAttention() {
  try {
    const artifactA=$('#dyadic-artifact-a').value,artifactB=$('#dyadic-artifact-b').value;
    const alignment=$('#dyadic-alignment')?.value||'pretrimmed_shared_start';
    const domainA=$('#dyadic-domain-a')?.value||null,domainB=$('#dyadic-domain-b')?.value||null;
    if(!artifactA||!artifactB)throw new Error('Select both target artifacts');
    if(alignment==='timegraph'&&(!domainA||!domainB))throw new Error('Select both source clocks for TimeGraph alignment');
    await api('/api/jobs/dyadic-attention',{method:'POST',body:{
      session_id:selectedSessionId(),role_a_target_artifact_id:artifactA,role_b_target_artifact_id:artifactB,
      role_a_id:$('#dyadic-role-a').value.trim()||'R01',role_b_id:$('#dyadic-role-b').value.trim()||'R02',
      alignment_mode:alignment,role_a_time_domain_id:domainA,role_b_time_domain_id:domainB,
      canonical_fps:Number($('#dyadic-fps').value),cross_wearer_tolerance_ms:Number($('#dyadic-tolerance').value),
      minimum_episode_ms:Number($('#dyadic-min-episode').value),short_gap_ms:Number($('#dyadic-gap').value),
      label:$('#dyadic-label').value.trim()||'dyadic_attention',output_parquet:$('#dyadic-parquet').dataset.on==='true'
    }});
    toast('Dyadic attention queued',alignment==='pretrimmed_shared_start'?'R01 and R02 will be paired on their shared Native-trim local timeline.':'Both role streams will be converted through verified TimeGraph mappings.');
    await refreshState({quiet:false});
  }catch(e){showError(e,'Could not annotate dyadic attention')}
}

function mediaToolsPanel() {
  const tools=appState.server.media_tools?.tools||{}, ff=tools.ffmpeg||{}, probe=tools.ffprobe||{};
  return `<section class="property-section" id="media-tools-panel"><div class="section-label">Video tools · 视频工具</div>
    <div class="property-stack">
      <div class="stat-row"><span>FFmpeg encoder</span><strong>${ff.available?'Available':'Missing / unchecked'}</strong></div>
      <div class="form-help media-tool-path" title="${esc(ff.path||'')}">${esc(ff.path||'Run REPAIR_MEDIA_TOOLS_WINDOWS.bat, or configure ffmpeg.exe below.')}</div>
      <div class="stat-row"><span>ffprobe indexer</span><strong>${probe.available?'Available':'Not found'}</strong></div>
      <button class="soft-button full" id="check-media-tools">Check both tools · 检查两项工具</button>
      <button class="soft-button full" id="configure-media-tools">Repair / paths · 修复与路径</button>
      <div class="form-help" id="media-tools-message" aria-live="polite">Both tools ship in tools/ffmpeg/bin. NRD checks ffprobe frame reading; Privacy checks FFmpeg encoding before large-file work.</div>
    </div></section>`;
}

function bindMediaToolButtons() {
  const config=$('#configure-media-tools'), checker=$('#check-media-tools');
  if(config)config.onclick=openMediaTools;
  if(checker)checker.onclick=async()=>{
    const button=$('#check-media-tools'), status=$('#media-tools-message');
    button.disabled=true;status.textContent='Checking H.264/AAC encoding AND actual ffprobe MP4 frame/JSON reading…';
    try{
      const result=await api('/api/media-tools/check',{method:'POST',body:{mode:'all',preserve_audio:$('#privacy-audio')?.dataset.on!=='false',
        ffprobe_path:(typeof nativeDraft==='function'?nativeDraft().ffprobe?.trim():'')||null}});
      if(!result.ready)throw new Error(result.error);
      status.textContent='PASS · FFmpeg + ffprobe · frame/JSON probe test passed';
      status.classList.remove('media-tools-error');
      await loadRuntime();
    }catch(error){status.textContent=error.message;status.classList.add('media-tools-error');}
    finally{if(button.isConnected)button.disabled=false;}
  };
}

async function openMediaTools() {
  try{
    const response=await api('/api/media-tools'), settings=response.settings||{};
    modal('Video tools · FFmpeg / ffprobe',`
      <p>Use automatic discovery, or select executable paths. Saving affects this computer only; project data and time mappings stay unchanged.</p>
      <div class="field"><label>FFmpeg executable or bin folder</label><div class="inline-input"><input id="media-ffmpeg" value="${esc(settings.ffmpeg||'')}" placeholder="Blank = automatic / imageio-ffmpeg"><button class="input-action" id="media-browse-ffmpeg">…</button></div></div>
      <div class="field"><label>ffprobe executable or bin folder · REQUIRED for NRD indexing</label><div class="inline-input"><input id="media-ffprobe" value="${esc(settings.ffprobe||'')}" placeholder="Blank = automatic"><button class="input-action" id="media-browse-ffprobe">…</button></div></div>
      <div class="form-help">FFmpeg encodes videos; ffprobe reads timestamps. Do not rename one to the other. Blank paths restore automatic discovery. imageio-ffmpeg supplies the encoder, not ffprobe.</div>
      <div class="form-help">This release includes BOTH Windows x64 executables in <strong>tools/ffmpeg/bin</strong>. Repair uses them first. Only if no working local pair is available will it download a pinned, SHA-256-checked full build from gyan.dev. Your recordings are never uploaded.</div>
      <div id="media-dialog-result" class="form-help" aria-live="polite"></div>
      <div class="form-actions"><button class="soft-button" id="media-dialog-check">Test both</button><button class="soft-button" id="media-dialog-repair">Use bundled pair / Repair</button><button class="primary-button" id="media-dialog-save">Save paths</button></div>`,(root,close)=>{
        for(const tool of ['ffmpeg','ffprobe']){
          $('#media-browse-'+tool,root).onclick=async()=>{
            const chosen=await choosePath('file','Select '+tool+' executable',[['Executable / all files','*.*']]);
            if(chosen)$('#media-'+tool,root).value=chosen;
          };
        }
        const paths=()=>({ffmpeg:$('#media-ffmpeg',root).value.trim(),ffprobe:$('#media-ffprobe',root).value.trim()});
        $('#media-dialog-check',root).onclick=async()=>{
          const target=$('#media-dialog-result',root), button=$('#media-dialog-check',root);button.disabled=true;
          target.textContent='Testing encoder AND ffprobe frame reading…';target.classList.remove('media-tools-error');
          try{
            const p=paths(), result=await api('/api/media-tools/check',{method:'POST',body:{mode:'all',ffmpeg_path:p.ffmpeg||null,ffprobe_path:p.ffprobe||null}});
            if(!result.ready)throw new Error(result.error);
            target.textContent='PASS · '+result.preflight.ffmpeg.path+' | ffprobe: '+result.preflight.ffprobe.path;
          }catch(e){target.textContent=e.message;target.classList.add('media-tools-error');}
          finally{button.disabled=false;}
        };
        $('#media-dialog-repair',root).onclick=async()=>{
          const button=$('#media-dialog-repair',root), target=$('#media-dialog-result',root);
          button.disabled=true;target.classList.remove('media-tools-error');
          try{
            if(!project()||!selectedSessionId())throw new Error('Create/open a project to use the job queue, or run REPAIR_FFPROBE_WINDOWS.bat.');
            const result=await api('/api/media-tools/install',{method:'POST',body:{session_id:selectedSessionId()}});
            target.textContent='Repair queued. Reuses local tools first; any download is shown in Job Queue. No recording is touched.';
            await refreshState({quiet:false});
            const poll=async()=>{
              try{
                const data=await api('/api/state'), job=data.project?.jobs?.[result.job.id];
                if(!job)throw new Error('Repair job no longer present; check the project Job Queue.');
                if(['failed','cancelled'].includes(job.status))throw new Error(job.error||job.status);
                if(job.status==='completed'){
                  const checked=await api('/api/media-tools/check',{method:'POST',body:{mode:'all'}});
                  if(!checked.ready)throw new Error(checked.error);
                  await loadRuntime();
                  if(typeof nativeDraft==='function')nativeDraft().ffprobe='';
                  if(root.isConnected){
                    $('#media-ffmpeg',root).value=checked.preflight.ffmpeg.path;
                    $('#media-ffprobe',root).value=checked.preflight.ffprobe.path;
                    target.textContent='PASS · Both tools ready. Keep original timecodes; close this dialog and run Preview again.';
                    button.disabled=false;
                  }
                  toast('Both media tools ready','Retry the failed preview; no re-import or timecode change needed.');
                  return;
                }
                if(root.isConnected){target.textContent=job.stage+' ('+Math.round(job.progress*100)+'%)';setTimeout(poll,700);}
              }catch(e){if(root.isConnected){target.textContent=e.message;target.classList.add('media-tools-error');button.disabled=false;}}
            };
            setTimeout(poll,500);
          }catch(e){target.textContent=e.message;target.classList.add('media-tools-error');button.disabled=false;}
        };
        $('#media-dialog-save',root).onclick=async()=>{
          const target=$('#media-dialog-result',root), button=$('#media-dialog-save',root);button.disabled=true;
          try{
            await api('/api/media-tools/settings',{method:'PUT',body:paths()});
            await loadRuntime();close();
            const old=$('#media-tools-panel');if(old){old.outerHTML=mediaToolsPanel();bindMediaToolButtons();}
            toast('Media paths saved','No system PATH change required.');
          }catch(e){target.textContent=e.message;target.classList.add('media-tools-error');}
          finally{if(button.isConnected)button.disabled=false;}
        };
      });
  }catch(e){showError(e,'Could not load video-tool settings')}
}


function calibrationImportIdentityText(row) {
  if(row.participant_id)return `Participant ${row.participant_id}`;
  if(row.hardware_hint)return row.hardware_hint;
  if(row.calibration_role==='calibration_board')return 'Shared AprilTag board';
  if(row.calibration_role==='world_reference')return 'Shared world/reference markers';
  if(row.calibration_role==='board_definition')return 'Board geometry definition';
  if(row.calibration_role==='completed_profile')return 'Completed hardware profile';
  return 'Project-level source';
}

function calibrationImportCompleteness(files=[]) {
  const count=role=>files.filter(row=>row.calibration_role===role).length;
  const actors=count('pupil_calibration');
  const trackers=count('tracker_candidate');
  const boards=count('calibration_board')+count('board_definition');
  const world=count('world_reference');
  return [
    {label:'Actor AprilTag recordings',value:actors,status:actors>0?'good':'warning',detail:actors>0?`${actors} detected`:'None detected'},
    {label:'ETM tracker calibration',value:trackers,status:trackers>=2?'good':trackers?'warning':'warning',detail:trackers>=2?`${trackers} candidates`:`${trackers} detected · normally ETM-A + ETM-B`},
    {label:'Calibration board',value:boards,status:boards>0?'good':'warning',detail:boards>0?`${boards} source${boards===1?'':'s'}`:'No board C3D / definition'},
    {label:'World reference',value:world,status:'neutral',detail:world?`${world} source${world===1?'':'s'}`:'Optional'},
  ];
}

function calibrationProfileBuilderSourceRows(rows, emptyText) {
  if(!rows?.length)return `<div class="profile-builder-empty">${esc(emptyText)}</div>`;
  return `<div class="profile-builder-source-list">${rows.map(row=>`<div class="profile-builder-source-row ${row.available?'':'missing'}"><span class="kind-pill">${esc(calibrationRoleLabel(row.calibration_role))}</span><div><strong>${esc(row.label)}</strong><small>${esc([row.participant_id,row.hardware_hint].filter(Boolean).join(' · ')||'Shared geometry source')}</small><small title="${esc(row.path)}">${esc(row.path)}</small></div><span class="status-pill ${row.available?'good':'bad'}">${row.available?'Available':'Missing'}</span></div>`).join('')}</div>`;
}

function openCalibrationSetProfileBuilder({setId='',setCode=''}={}) {
  const initial=calibrationSetRecord(setId)||calibrationSetRecord(setCode)||activeCalibrationSetRecord()||calibrationSetRows()[0];
  if(!initial){toast('No Calibration Set','Create and import a Project-level Calibration Set first.',true);return;}
  const state={setId:initial.id,status:null,job:null,loading:true,error:''};
  modal('Build physical ETM profiles', `<div class="calibration-profile-builder-root"><div class="profile-builder-loading"><strong>Reading Calibration Set…</strong><span>Checking shared AprilTag recordings, board C3D and ETM tracker candidates.</span></div></div>`, async(root,close)=>{
    const host=$('.calibration-profile-builder-root',root);
    const setMessage=(title,detail='',progress=null,error=false)=>{
      const bar=progress==null?'':`<progress max="100" value="${Math.max(0,Math.min(100,Number(progress)||0))}"></progress>`;
      const el=$('#profile-builder-job',root);if(el)el.innerHTML=`<strong class="${error?'error':''}">${esc(title)}</strong><span>${esc(detail)}</span>${bar}`;
    };
    const slotMarkup=(key,label)=>{
      const slot=state.status?.hardware_slots?.[key]||{};
      const newest=(slot.profiles||[]).slice(-1)[0];
      return `<article class="profile-builder-hardware-slot ${slot.ready?'ready':'missing'}"><div><span class="eyebrow">Physical hardware</span><strong>${esc(label)}</strong><small>${newest?esc(newest.label):'No committed profile in this revision'}</small></div><span class="status-pill ${slot.ready?'good':'warning'}">${slot.ready?'Ready':'Missing'}</span></article>`;
    };
    const render=()=>{
      const st=state.status,row=calibrationSetRecord(state.setId)||initial;
      if(state.loading){host.innerHTML='<div class="profile-builder-loading"><strong>Reading Calibration Set…</strong><span>Checking shared AprilTag recordings, board C3D and ETM tracker candidates.</span></div>';return;}
      if(state.error&&!st){host.innerHTML=`<div class="warning-card"><strong>Could not inspect this Calibration Set</strong><p>${esc(state.error)}</p></div><div class="form-actions"><button class="soft-button" id="profile-builder-close">Close</button></div>`;$('#profile-builder-close',root).onclick=close;return;}
      const problems=st.problems||[];
      const actors=st.pupil_recordings||[];
      const actorText=actors.map(row=>row.participant_id||row.label).join(', ')||'None';
      const board=st.board_geometry||{};
      const boardReady=Boolean(board.ready);
      const boardResolvable=Boolean(board.resolvable);
      const boardState=boardReady?'Verified':boardResolvable?'Ready to resolve':'Blocked';
      const boardDetail=boardReady
        ?`${Number(board.tag_corner_count||0)}/16 Tag corners · ${Number(board.outer_corner_count||0)}/4 outer corners · ${board.mapping_variant||board.mapping_source||'saved definition'}`
        :boardResolvable
          ?`${board.board_point_count||'16'}-marker planar board available; scene AprilTags + selected ETM will resolve orientation once and save it to this revision.`
          :(board.problems||[]).join('; ')||'Board geometry is unavailable.';
      const ready=Boolean(st.ready),running=Boolean(state.job&&['queued','running'].includes(state.job.status));
      const runLabel=running?'Resolving geometry / building profiles…':ready?'Rebuild / validate profiles':boardReady?'Build missing physical profiles':'Resolve board geometry & build profiles';
      const runDisabled=(!st.can_build&&!ready)||running;
      host.innerHTML=`<section class="profile-builder-hero"><div><span class="eyebrow">Project-level Calibration Set · interaction-independent</span><h3>${esc(st.calibration_set_label||row?.label||st.calibration_set_code)}</h3><p>${esc(st.calibration_set_code)} · revision ${esc(st.revision_number??'—')} · inherited by ${(st.applicable_session_ids||[]).length} Sessions</p></div><div class="profile-builder-hero-actions"><span class="status-pill ${ready?'good':'warning'}">${ready?'Both profiles ready':'Profiles incomplete'}</span><button class="primary-button" id="profile-builder-run-top" ${runDisabled?'disabled':''}>${runLabel}</button></div></section>
        <section class="profile-builder-contract"><strong>Why the old role-detail route failed</strong><p>Physical camera→tracker profiles belong to ETM-A and ETM-B, not to the participant currently wearing them in S004/I02. This builder uses the shared calibration actors (${esc(actorText)}) and automatically matches each AprilTag recording to the physical ETM tracker. It does not require the Calibration actor to equal the current R01/R02 participant.</p></section>
        <section class="profile-builder-contract ${boardReady?'ready':''}"><div class="iw-section-head compact"><div><span class="eyebrow">Shared AprilTag board geometry</span><h3>${esc(boardState)}</h3></div><span class="status-pill ${boardReady?'good':boardResolvable?'warning':'bad'}">${boardReady?'16 + 4 mapped':boardResolvable?'Inference available':'Missing'}</span></div><p>${esc(boardDetail)}</p>${board.definition_path?`<small title="${esc(board.definition_path)}">Stored with the active Calibration Set revision: ${esc(board.definition_path)}</small>`:''}</section>
        <section class="profile-builder-slots">${slotMarkup('ETM_A','ETM-A')}${slotMarkup('ETM_B','ETM-B')}</section>
        <section class="profile-builder-inventory"><div class="iw-section-head compact"><div><span class="eyebrow">Source inventory</span><h3>${(st.sources||[]).length} registered files</h3></div><button class="soft-button compact" id="profile-builder-import">Open Data → Calibration Data</button></div>${calibrationProfileBuilderSourceRows(st.sources,'No shared Calibration sources are registered.')}</section>
        ${problems.length?`<section class="warning-card profile-builder-problems"><strong>Build is blocked</strong><ul>${problems.map(item=>`<li>${esc(item)}</li>`).join('')}</ul></section>`:''}
        <section class="profile-builder-method"><strong>Independent evidence fusion</strong><ol><li>Use labelled head-marker proximity to identify which physical tracker is ETM-A or ETM-B for each calibration actor.</li><li>Independently test the eight rotations/reflections of the planar 16-marker grid against scene AprilTags and the already-selected tracker.</li><li>Require the calibration actors to agree, then store the accepted 16-corner + 4-outer-corner mapping in the active Calibration Set revision.</li><li>Estimate each camera→tracker transform with its own calibration-only offset search and full visual QC.</li><li>Commit ETM-A and ETM-B together and propagate traceable references to assigned Sessions.</li></ol><details><summary>Advanced build settings</summary><div class="form-grid profile-builder-settings"><div class="field"><label>Sample every N frames</label><input id="profile-builder-sample" type="number" min="1" max="60" value="4"></div><div class="field"><label>Maximum profile-detection frames</label><input id="profile-builder-maxframes" type="number" min="20" max="2000" value="240"></div><div class="field"><label>Minimum tracker identity confidence</label><select id="profile-builder-confidence"><option value="HIGH">High</option><option value="MEDIUM" selected>Medium · recommended</option><option value="SINGLE_CANDIDATE">Single candidate · recovery</option><option value="LOW">Low · diagnostic only</option></select></div><div class="field"><label>Mount version</label><input id="profile-builder-mount" value="mount_v1"></div><label class="check-row full-span"><input type="checkbox" id="profile-builder-force"><span>Force a new ETM-A + ETM-B revision even when both are already ready</span></label></div></details></section>
        <section id="profile-builder-job" class="profile-builder-job"><strong>${ready?'Ready':'Not started'}</strong><span>${ready?'Both physical hardware slots are committed to the active revision.':'No project data changes occur until the build completes successfully.'}</span></section>
        <footer class="profile-builder-footer"><div><button class="soft-button" id="profile-builder-folder">Open Calibration Set reports</button><button class="soft-button" id="profile-builder-refresh">Refresh inspection</button></div><div><button class="soft-button" id="profile-builder-close">Close</button><button class="primary-button" id="profile-builder-run" ${runDisabled?'disabled':''}>${runLabel}</button></div></footer>`;
      $('#profile-builder-close',root).onclick=close;
      $('#profile-builder-import',root).onclick=()=>{close();appState.calibrationSetId=st.calibration_set_code||appState.calibrationSetId;setTimeout(()=>scrollDataWorkspace('calibration'),0);};
      $('#profile-builder-folder',root).onclick=()=>openLocalPath(joinLocalPath(row?.folder||'', 'reports'));
      $('#profile-builder-refresh',root).onclick=()=>load();
      $('#profile-builder-run',root).onclick=run;
      $('#profile-builder-run-top',root).onclick=run;
      if(state.job)setMessage(state.job.status==='completed'?'Profiles built':state.job.status==='failed'?'Build failed':state.job.stage||'Building physical profiles…',state.job.error||state.job.stage||state.job.status,100*Number(state.job.progress||0),state.job.status==='failed');
    };
    const load=async()=>{
      state.loading=true;state.error='';render();
      try{const payload=await api(`/api/project/calibration-sets/${encodeURIComponent(state.setId)}/profile-build-status`);state.status=payload.build_status;state.loading=false;render();}
      catch(error){state.loading=false;state.error=error.message||String(error);render();}
    };
    const poll=async jobId=>{
      while(root.isConnected){
        const payload=await api('/api/state');
        const job=payload.project?.jobs?.[jobId]||null;
        if(job){state.job=job;render();if(['completed','failed','cancelled'].includes(job.status))return job;}
        await new Promise(resolve=>setTimeout(resolve,650));
      }
      return null;
    };
    const run=async()=>{
      try{
        const force=$('#profile-builder-force',root)?.checked===true;
        const response=await api(`/api/jobs/calibration-set-profiles/${encodeURIComponent(state.setId)}`,{method:'POST',body:{session_id:selectedSessionId(),revision_id:state.status?.revision_id||null,sample_every_n_frames:Number($('#profile-builder-sample',root)?.value||4),max_detection_frames:Number($('#profile-builder-maxframes',root)?.value||240),max_pose_gap_sec:0.1,minimum_tracker_identity_confidence:$('#profile-builder-confidence',root)?.value||'MEDIUM',mount_version:$('#profile-builder-mount',root)?.value.trim()||'mount_v1',force_rebuild:force,output_g3dcal:true}});
        const jobId=response?.job?.id;if(!jobId)throw new Error('The backend did not return a profile-build job ID');
        appState.pendingCalibrationSetProfileJobId=jobId;
        state.job={id:jobId,status:'queued',progress:0,stage:'Queued'};render();
        const job=await poll(jobId);
        if(job?.status==='failed')throw new Error(job.error||'Calibration Set profile build failed');
        if(job?.status==='cancelled')throw new Error('Calibration Set profile build was cancelled');
        await refreshState({quiet:false,fullRender:true});
        await load();
      }catch(error){state.job={...(state.job||{}),status:'failed',error:error.message||String(error),progress:state.job?.progress||0};render();}
    };
    await load();
  });
}

function inferCalibrationBrowserRow(file, setCode='') {
  const name=String(file?.name||'');
  const lower=name.toLowerCase();
  const extension=(name.match(/\.[^.]+$/)||[''])[0].toLowerCase();
  let calibration_role='calibration_source';
  if(['.g3dcal','.oculocal'].includes(extension))calibration_role='completed_profile';
  else if(extension==='.json' && /(board|apriltag)/i.test(name))calibration_role='board_definition';
  else if(/markers[_-]?world|world[_-]?marker|(?:^|[_-])world(?:[_-]|\.)/i.test(name))calibration_role='world_reference';
  else if(/calibration[_-]?board/i.test(name) || (/board/i.test(name)&&['.c3d','.npz'].includes(extension)))calibration_role='calibration_board';
  else if(/eye[_-]?tracker|(?:^|[_-])etm[_-]?[ab12]/i.test(name) && ['.c3d','.npz'].includes(extension))calibration_role='tracker_candidate';
  else if(extension==='.zip' && (/calibration/i.test(name)||/(?:^|[^A-Za-z0-9])P\d{1,6}(?![A-Za-z0-9])/i.test(name)))calibration_role='pupil_calibration';
  const participant=(name.match(/(?:^|[^A-Za-z0-9])(P\d{1,6})(?![A-Za-z0-9])/i)||[])[1]?.toUpperCase()||'';
  let hardware_hint='';
  if(/eye[_-]?tracker[_-]?a(?:[^a-z0-9]|$)|(?:^|[_-])etm[_-]?a(?:[^a-z0-9]|$)/i.test(name))hardware_hint='ETM-A';
  else if(/eye[_-]?tracker[_-]?b(?:[^a-z0-9]|$)|(?:^|[_-])etm[_-]?b(?:[^a-z0-9]|$)/i.test(name))hardware_hint='ETM-B';
  const existing=calibrationAssets(null,setCode).find(row=>String(row.label||'').toLowerCase()===lower && Number(row.size_bytes||0)===Number(file?.size||0));
  return {id:`calfile_${Date.now()}_${Math.random().toString(16).slice(2)}`,file,name,path:name,extension,size_bytes:Number(file?.size||0),calibration_role,participant_id:calibration_role==='pupil_calibration'?participant:'',hardware_hint,already_registered:Boolean(existing),existing_asset_id:existing?.id||'',status:existing?'reused':'ready',progress:existing?100:0,message:existing?'Already registered in this Calibration Set':'Ready to upload'};
}

function openCalibrationSetImportDialog({setCode='',initialFiles=[],autoPicker=''}={}) {
  const sets=calibrationSetRows();
  if(!project())return;
  if(!sets.length){
    toast('Create a Calibration Set first','Open Data → Calibration Data, create a Set, and assign the Sessions before importing source files.',true);
    openProjectSessionManager({focus:'calibration'});
    return;
  }
  const initial=calibrationSetRecord(setCode)||activeCalibrationSetRecord()||sets[0];
  const state={setCode:initial.code||initial.id,paths:[],recursive:false,scan:null,browserRows:[],busy:false,pickerBusy:false,job:null,result:null,error:''};
  const filters=[['Calibration sources','*.zip *.c3d *.npz *.json *.g3dcal *.oculocal'],['ZIP archives','*.zip'],['C3D motion','*.c3d'],['All files','*.*']];
  modal('Import / append Calibration Data · 导入或追加共享标定数据', `<div class="calibration-import-dialog">
    <section class="calibration-import-scope" id="calimport-scope"></section>
    <section class="calibration-import-source-panel">
      <div class="iw-section-head compact"><div><div class="eyebrow">1 · Add source files</div><h3>Drop files, choose local files, or select one folder</h3></div><span class="status-pill">Project-level Calibration Data</span></div>
      <p>This workspace appends files to the selected Calibration Set. Interaction files use the separate Interaction Data block.</p>
      <div id="calimport-drop-zone" class="import-drop-zone calibration-import-drop" tabindex="0" role="button"><strong>Drop Calibration ZIP / C3D / profile files here</strong><span>Browser drops are copied into <code>calibration_sets/CAL.../sources/</code>. Desktop local-path selection references originals without copying.</span></div>
      <input id="calimport-browser-input" type="file" multiple hidden accept=".zip,.c3d,.npz,.json,.g3dcal,.oculocal">
      <div class="calibration-import-picker-row"><button class="primary-button" id="calimport-browser-files">Choose files from this computer…</button><button class="soft-button" id="calimport-files">Reference local files…</button><button class="soft-button" id="calimport-folder">Reference local folder…</button></div>
      <details class="calibration-local-paths"><summary>Local path / folder selection · no copy</summary><label class="field"><span>Selected local paths</span><textarea id="calimport-paths" rows="4" placeholder="D:\\...\\EyeCalibration\n—or one file per line—"></textarea></label><label class="check-row compact"><input type="checkbox" id="calimport-recursive"><span>Include supported files in subfolders</span></label><div class="form-actions"><button class="soft-button" id="calimport-scan">Scan and review local paths</button><span id="calimport-selection-note" class="form-help"></span></div></details>
    </section>
    <section class="calibration-import-review-panel">
      <div class="iw-section-head compact"><div><div class="eyebrow">2 · Review detected purpose and identity</div><h3>Nothing is registered until you confirm</h3></div><span id="calimport-count" class="status-pill">No files yet</span></div>
      <div id="calimport-completeness" class="calibration-import-completeness"></div>
      <div id="calimport-scan-result" class="calibration-import-scan-empty">Drop files or choose a local folder above. OculoStudio will show every supported file and its detected participant / device role.</div>
    </section>
    <section class="calibration-import-footer">
      <div class="calibration-import-progress"><strong id="calimport-status">Ready</strong><span id="calimport-progress-text">The selected Calibration Set owns one canonical source reference or local project copy.</span><progress id="calimport-progress" value="0" max="100"></progress></div>
      <div class="form-actions"><button class="soft-button" id="calimport-close">Close</button><button class="primary-button" id="calimport-run" disabled>Import reviewed files</button></div>
    </section>
  </div>`,(root,close)=>{
    const dialog=$('.modal',root);dialog.classList.add('calibration-import-modal');
    const setRow=()=>calibrationSetRecord(state.setCode)||calibrationSetRows()[0];
    const pathRows=()=>String($('#calimport-paths',root)?.value||'').split(/\r?\n|\|/).map(value=>value.trim()).filter(Boolean);
    const reviewRows=()=>state.browserRows.length?state.browserRows:(state.scan?.files||[]);
    const pendingBrowserRows=()=>state.browserRows.filter(row=>!['done','reused'].includes(row.status));
    const setBusy=value=>{state.busy=value;for(const selector of ['#calimport-browser-files','#calimport-folder','#calimport-files','#calimport-scan','#calimport-run','#calimport-recursive','#calimport-set']){const element=$(selector,root);if(element)element.disabled=value;}const paths=$('#calimport-paths',root);if(paths)paths.disabled=value;};
    const setStatus=(title,detail='',progress=0,error=false)=>{const status=$('#calimport-status',root),note=$('#calimport-progress-text',root),bar=$('#calimport-progress',root);if(status){status.textContent=title;status.classList.toggle('error',error);}if(note)note.textContent=detail;if(bar)bar.value=Math.max(0,Math.min(100,Number(progress)||0));};
    const renderScope=()=>{
      const row=setRow();const sessions=(row?.sessions||[]).map(item=>item.session_code||item.session_label).filter(Boolean);
      $('#calimport-scope',root).innerHTML=`<div><span class="eyebrow">Project-level scope · never assigned to the current Interaction</span><h3>${esc(row?.label||state.setCode)}</h3><p>${esc(row?.code||state.setCode)} · revision ${esc(row?.active_revision_number??'—')} · ${esc(row?.purpose||'combined')}</p></div><div class="calibration-import-scope-fields"><label>Calibration Set<select id="calimport-set">${calibrationSetRows().map(item=>`<option value="${esc(item.code||item.id)}" ${(item.code||item.id)===state.setCode?'selected':''}>${esc(item.code)} · ${esc(item.label)}</option>`).join('')}</select></label><div><span>Inherited by Sessions</span><strong>${esc(sessions.join(', ')||'No Session assignment')}</strong></div><div><span>Canonical destination</span><strong>${esc(row?.folder||'Project calibration_sets folder')}</strong></div></div>`;
      $('#calimport-set',root)?.addEventListener('change',event=>{state.setCode=event.target.value;state.scan=null;state.browserRows=state.browserRows.map(row=>({...row,already_registered:false,status:row.status==='done'?'done':'ready'}));renderScope();renderReview();});
    };
    const renderCompleteness=()=>{const rows=reviewRows();$('#calimport-completeness',root).innerHTML=rows.length?calibrationImportCompleteness(rows).map(item=>`<div class="calibration-import-check ${item.status}"><span>${esc(item.label)}</span><strong>${esc(item.detail)}</strong></div>`).join(''):'';};
    const renderReview=()=>{
      const files=reviewRows(),result=$('#calimport-scan-result',root),count=$('#calimport-count',root),run=$('#calimport-run',root);renderCompleteness();
      if(!files.length){count.textContent='No files yet';count.className='status-pill';result.className='calibration-import-scan-empty';result.innerHTML='Drop files or choose a local folder above. OculoStudio will show every supported file and its detected participant / device role.';run.disabled=true;run.textContent='Import reviewed files';return;}
      const newCount=files.filter(row=>!row.already_registered&&!['done','reused'].includes(row.status)).length;
      count.textContent=`${files.length} files · ${newCount} ready`;count.className=`status-pill ${newCount?'good':''}`;
      const missing=files.filter(row=>row.calibration_role==='pupil_calibration'&&!row.participant_id);
      const warning=missing.length?`<div class="warning-card compact"><strong>Participant ID missing for ${missing.length} actor recording${missing.length===1?'':'s'}.</strong><p>Enter P01/P03 below before importing. Calibration C3D files should remain participant-independent.</p></div>`:'';
      result.className='calibration-import-scan-result';
      result.innerHTML=`${warning}<div class="calibration-import-table-wrap"><table class="calibration-import-table"><thead><tr><th>File</th><th>Detected purpose</th><th>Participant / device</th><th>Status</th><th>Size</th></tr></thead><tbody>${files.map(row=>`<tr><td><strong>${esc(row.name)}</strong><small title="${esc(row.path||row.name)}">${esc(row.path||'Browser file')}</small></td><td><span class="kind-pill">${esc(calibrationRoleLabel(row.calibration_role))}</span></td><td>${row.calibration_role==='pupil_calibration'?`<input class="calimport-participant-input" data-row-id="${esc(row.id||'')}" value="${esc(row.participant_id||'')}" placeholder="P01">`:esc(calibrationImportIdentityText(row))}</td><td><span class="status-pill ${['done','reused'].includes(row.status)||row.already_registered?'good':row.status==='failed'?'bad':''}">${esc(row.already_registered?'Already registered':row.status||'New')}</span>${row.message?`<small>${esc(row.message)}</small>`:''}${row.progress>0&&row.progress<100?`<progress max="100" value="${Number(row.progress)||0}"></progress>`:''}</td><td>${esc(fmtBytes(row.size_bytes))}</td></tr>`).join('')}</tbody></table></div>`;
      $$('.calimport-participant-input',result).forEach(input=>input.oninput=()=>{const row=state.browserRows.find(item=>item.id===input.dataset.rowId);if(row)row.participant_id=input.value.trim().toUpperCase();});
      run.disabled=state.busy||(!newCount&&files.every(row=>row.already_registered||['done','reused'].includes(row.status)));
      run.textContent=state.browserRows.length?`Import ${newCount} file${newCount===1?'':'s'} into ${state.setCode}`:(state.scan?.new_count?`Import ${state.scan.file_count} reviewed files into ${state.scan.calibration_set_id}`:`Verify ${state.scan?.file_count||files.length} existing files`);
    };
    const addBrowserFiles=fileList=>{
      const supported=new Set(['.zip','.c3d','.npz','.json','.g3dcal','.oculocal']);
      const existingKeys=new Set(state.browserRows.map(row=>`${row.name}|${row.size_bytes}|${row.file?.lastModified||0}`));
      for(const file of Array.from(fileList||[])){
        const extension=(String(file.name).match(/\.[^.]+$/)||[''])[0].toLowerCase();
        if(!supported.has(extension)){toast('Unsupported Calibration file',`${file.name} was skipped.`,true);continue;}
        const key=`${file.name}|${file.size}|${file.lastModified||0}`;if(existingKeys.has(key))continue;
        state.browserRows.push(inferCalibrationBrowserRow(file,state.setCode));existingKeys.add(key);
      }
      if(state.browserRows.length){state.scan=null;state.paths=[];const input=$('#calimport-paths',root);if(input)input.value='';setStatus('Files ready for review',`${state.browserRows.length} browser/drop file(s) selected. Nothing has been registered yet.`,0);renderReview();}
    };
    const scanPaths=async({quiet=false}={})=>{
      const paths=pathRows();if(!paths.length)throw new Error('Choose one calibration folder or several files first');
      state.browserRows=[];state.paths=paths;state.recursive=Boolean($('#calimport-recursive',root)?.checked);state.error='';setBusy(true);setStatus('Scanning selected paths…','No project files are changed during this step.',8);
      try{state.scan=await api('/api/calibration-assets/scan-paths',{method:'POST',body:{project_id:project().id,session_id:selectedSessionId(),paths,calibration_set_id:state.setCode,checksum:true,recursive:state.recursive}});appState.calibrationSetId=state.setCode;appState.calibrationSourcePath=paths.join('|');appState.calibrationRecursive=state.recursive;appState.calibrationScan=state.scan;setStatus('Scan complete',`${state.scan.file_count} supported files · ${state.scan.new_count} new · ${state.scan.existing_count} already registered`,100);if(!quiet)toast('Calibration file scan complete',`${state.scan.file_count} supported file(s) found.`);}finally{setBusy(false);renderScope();renderReview();}return state.scan;
    };
    const choose=async kind=>{
      if(state.busy||state.pickerBusy)return;state.pickerBusy=true;setStatus('Opening file picker…','Select shared Calibration source files.',0);
      try{const title=kind==='directory'?'Select the Calibration Set source folder':'Select Calibration Set ZIP / C3D / profile files';const response=window.pywebview?.api?.choose_paths?await window.pywebview.api.choose_paths(kind,title,filters,kind==='file'):await api('/api/dialog',{method:'POST',body:{kind,title,filetypes:filters,multiple:kind==='file'}});const paths=(response?.paths||[response?.path]).filter(Boolean);if(!paths.length){setStatus('Selection cancelled','No source was added.',0);return;}$('#calimport-paths',root).value=paths.join('\n');state.scan=null;state.browserRows=[];await scanPaths({quiet:false});}catch(error){state.error=error.message||String(error);setStatus('Could not select or scan files',state.error,0,true);showError(error,'Calibration source selection failed');}finally{state.pickerBusy=false;setBusy(false);renderReview();}
    };
    const pollJob=async(jobId,row=null)=>{while(root.isConnected){const payload=await api(`/api/imports/jobs/${encodeURIComponent(jobId)}`);const job=payload.job;state.job=job;if(row){row.message=job.error||job.stage||job.status;row.progress=80+20*Number(job.progress||0);renderReview();}else setStatus(job.status==='completed'?'Calibration files registered':job.status==='failed'?'Import failed':job.stage||'Importing Calibration Set files…',job.error||job.stage||job.status,100*Number(job.progress||0),job.status==='failed');if(['completed','failed','cancelled'].includes(job.status))return job;await new Promise(resolve=>setTimeout(resolve,450));}return null;};
    const uploadBrowserRow=async(row,index,total)=>{
      if(row.already_registered||['done','reused'].includes(row.status)){row.status='reused';row.progress=100;return;}
      const start=await api('/api/calibration-assets/uploads',{method:'POST',body:{project_id:project().id,session_id:selectedSessionId(),calibration_set_id:state.setCode,filename:row.file.name,size:row.file.size}});
      row.uploadId=start.upload_id;row.status='copying';row.message='Copying into the Calibration Set…';renderReview();
      let offset=0;const chunkBytes=Number(start.chunk_bytes||8*1024*1024);
      while(offset<row.file.size){const blob=row.file.slice(offset,Math.min(offset+chunkBytes,row.file.size));const response=await fetch(`/api/imports/uploads/${encodeURIComponent(row.uploadId)}?offset=${offset}`,{method:'PUT',body:blob,headers:{'Content-Type':'application/octet-stream'}});const payload=await response.json();if(!response.ok)throw new Error(payload.message||payload.detail||'Calibration file transfer failed');offset=payload.offset;row.progress=80*offset/row.file.size;row.message=`Copied ${fmtBytes(offset)} / ${fmtBytes(row.file.size)}`;setStatus(`Uploading ${index+1}/${total}`,row.name,((index+row.progress/100)/total)*100);renderReview();}
      const finished=await api(`/api/calibration-assets/uploads/${encodeURIComponent(row.uploadId)}/finish`,{method:'POST',body:{project_id:project().id,calibration_set_id:state.setCode,participant_id:row.participant_id||null}});
      row.jobId=finished.job.id;row.status='inspecting';row.message='Inspecting and registering Calibration evidence…';renderReview();const job=await pollJob(row.jobId,row);if(job.status==='failed')throw new Error(job.error||'Calibration import failed');if(job.status==='cancelled')throw new Error('Calibration import cancelled');row.status='done';row.progress=100;row.message='Registered in the Calibration Set';row.file=null;
    };
    const importBrowserFiles=async()=>{
      const pending=pendingBrowserRows().filter(row=>!row.already_registered);if(!pending.length){toast('Calibration sources already registered');return;}
      setBusy(true);let ok=0,failed=0;
      try{for(let index=0;index<pending.length;index++){const row=pending[index];try{await uploadBrowserRow(row,index,pending.length);ok++;}catch(error){row.status='failed';row.message=error.message||String(error);failed++;renderReview();}}await refreshState({quiet:false,fullRender:true});setStatus(failed?'Calibration import completed with issues':'Calibration Data updated',`${ok} imported · ${failed} failed`,100,failed>0);toast('Calibration Set updated',`${ok} source file${ok===1?'':'s'} registered to ${state.setCode}.`,failed>0);}finally{setBusy(false);renderScope();renderReview();}
    };
    const importLocalPaths=async()=>{
      try{let scan=state.scan;if(!scan||scan.calibration_set_id!==state.setCode)scan=await scanPaths({quiet:true});if(!scan?.files?.length)throw new Error('No supported calibration files are ready to import');setBusy(true);setStatus('Queueing one audited batch…',`${scan.file_count} files → ${scan.calibration_set_id}`,2);const response=await api('/api/calibration-assets/import-paths',{method:'POST',body:{project_id:project().id,session_id:selectedSessionId(),paths:scan.files.map(row=>row.path),calibration_set_id:scan.calibration_set_id,checksum:true,recursive:false}});appState.pendingCalibrationImportJobId=response?.job?.id||null;const job=await pollJob(response.job.id);if(job?.status==='failed')throw new Error(job.error||'Calibration import failed');state.result=job?.result||{};await refreshState({quiet:false,fullRender:true});const imported=Number(state.result.imported_count||0),reused=Number(state.result.reused_count||0),failed=Number(state.result.failed_count||0);setStatus(failed?'Calibration batch completed with issues':'Calibration Data updated',`${imported} imported · ${reused} reused · ${failed} failed`,100,failed>0);state.scan=await api('/api/calibration-assets/scan-paths',{method:'POST',body:{project_id:project().id,session_id:selectedSessionId(),paths:scan.files.map(row=>row.path),calibration_set_id:scan.calibration_set_id,checksum:true,recursive:false}});renderScope();renderReview();toast('Calibration Set updated',`${scan.file_count} source file(s) are now registered to ${scan.calibration_set_id}.`,failed>0);}catch(error){state.error=error.message||String(error);setStatus('Calibration import failed',state.error,0,true);showError(error,'Could not import Calibration Set files');}finally{setBusy(false);renderReview();}
    };
    const importFiles=()=>state.browserRows.length?importBrowserFiles():importLocalPaths();
    const drop=$('#calimport-drop-zone',root),browserInput=$('#calimport-browser-input',root);
    drop.onclick=()=>browserInput.click();drop.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();browserInput.click();}};
    for(const type of ['dragenter','dragover'])drop.addEventListener(type,event=>{event.preventDefault();event.stopPropagation();drop.classList.add('drag-active');});
    for(const type of ['dragleave','drop'])drop.addEventListener(type,event=>{event.preventDefault();event.stopPropagation();drop.classList.remove('drag-active');});
    drop.addEventListener('drop',event=>addBrowserFiles(event.dataTransfer?.files));
    browserInput.addEventListener('change',()=>{if(browserInput.files?.length)addBrowserFiles(browserInput.files);browserInput.value='';});
    $('#calimport-browser-files',root).onclick=()=>browserInput.click();$('#calimport-folder',root).onclick=()=>choose('directory');$('#calimport-files',root).onclick=()=>choose('file');$('#calimport-scan',root).onclick=()=>scanPaths({quiet:false}).catch(error=>{setBusy(false);setStatus('Scan failed',error.message||String(error),0,true);showError(error,'Could not scan Calibration Set files');});$('#calimport-run',root).onclick=importFiles;$('#calimport-close',root).onclick=close;$('#calimport-paths',root).addEventListener('input',()=>{state.scan=null;state.browserRows=[];renderReview();setStatus('Selection changed','Run Scan and review local paths before importing.',0);});$('#calimport-recursive',root).addEventListener('change',()=>{state.scan=null;renderReview();});
    renderScope();if(initialFiles?.length)addBrowserFiles(initialFiles);else renderReview();
    if(autoPicker==='file')setTimeout(()=>browserInput.click(),80);else if(autoPicker==='directory')setTimeout(()=>choose('directory'),80);
  });
}

window.OculoCalibrationImport={open:options=>openCalibrationSetImportDialog(options||{}),openWithFiles:(files,setCode='')=>openCalibrationSetImportDialog({setCode,initialFiles:Array.from(files||[])})};
function calibrationSelectionPaths(rawValue=null) {
  const raw=String(rawValue ?? $('#cal-source-path')?.value ?? appState.calibrationSourcePath ?? '');
  return raw.split(/\r?\n|\|/).map(value=>value.trim()).filter(Boolean);
}

async function scanCalibrationSourceSelection({render=true, quiet=false}={}) {
  if(!project()||!selectedSessionId())throw new Error('Open a project first');
  const setId=String($('#cal-set-id')?.value||appState.calibrationSetId||'').trim();
  const raw=String($('#cal-source-path')?.value??appState.calibrationSourcePath??'');
  const paths=calibrationSelectionPaths(raw);
  const recursive=Boolean($('#cal-import-recursive')?.checked ?? appState.calibrationRecursive);
  if(!setId)throw new Error('Enter a calibration set ID');
  if(!paths.length)throw new Error('Choose one or more calibration files or a folder');
  appState.calibrationSetId=setId;
  appState.calibrationSourcePath=raw;
  appState.calibrationRecursive=recursive;
  const result=await api('/api/calibration-assets/scan-paths',{method:'POST',body:{
    project_id:project().id,session_id:selectedSessionId(),paths,calibration_set_id:setId,
    checksum:true,recursive,
  }});
  appState.calibrationScan=result;
  if(render){
    const panel=$('#properties-content');const scrollTop=panel?.scrollTop||0;
    renderProperties();
    requestAnimationFrame(()=>{const next=$('#properties-content');if(next)next.scrollTop=scrollTop;});
  }
  if(!quiet)toast('Calibration source scan complete',`${result.file_count} supported source file(s) found; ${result.new_count} new.`);
  return result;
}

async function chooseCalibrationSources(kind='file') {
  try {
    const title=kind==='directory'?'Select calibration session folder':'Select calibration ZIP/C3D files';
    const filetypes=[['Calibration sources','*.zip *.c3d *.npz *.json *.g3dcal *.oculocal'],['All files','*.*']];
    const result=window.pywebview?.api?.choose_paths
      ? await window.pywebview.api.choose_paths(kind,title,filetypes,kind==='file')
      : await api('/api/dialog',{method:'POST',body:{kind,title,filetypes,multiple:kind==='file'}});
    const paths=(result?.paths||[]).filter(Boolean);
    if(!paths.length){toast('Selection cancelled','No calibration source was selected.');return;}
    appState.calibrationSourcePath=paths.join('|');
    appState.calibrationRecursive=kind==='directory'?false:appState.calibrationRecursive;
    appState.calibrationScan=null;
    const input=$('#cal-source-path');if(input)input.value=appState.calibrationSourcePath;
    const recursive=$('#cal-import-recursive');if(recursive)recursive.checked=appState.calibrationRecursive;
    await scanCalibrationSourceSelection({render:true,quiet:false});
  } catch(error) { showError(error,'Could not select or scan calibration sources'); }
}

async function submitCalibrationSources() {
  const button=$('#cal-import-sources');
  try {
    if(!project()||!selectedSessionId())throw new Error('Open a project first');
    let scan=appState.calibrationScan;
    const currentSet=String($('#cal-set-id')?.value||appState.calibrationSetId||'').trim();
    const currentPath=String($('#cal-source-path')?.value??appState.calibrationSourcePath??'');
    const currentRecursive=Boolean($('#cal-import-recursive')?.checked ?? appState.calibrationRecursive);
    const scanStale=!scan||scan.calibration_set_id!==currentSet||currentPath!==appState.calibrationSourcePath||currentRecursive!==appState.calibrationRecursive;
    if(scanStale)scan=await scanCalibrationSourceSelection({render:false,quiet:true});
    if(!scan?.files?.length)throw new Error('No supported calibration sources were discovered');
    if(button){button.disabled=true;button.textContent=`Importing ${scan.file_count} sources…`;}
    const explicitPaths=scan.files.map(row=>row.path);
    const result=await api('/api/calibration-assets/import-paths',{method:'POST',body:{
      project_id:project().id,session_id:selectedSessionId(),paths:explicitPaths,
      calibration_set_id:scan.calibration_set_id,checksum:true,recursive:false,
    }});
    appState.pendingCalibrationImportJobId=result?.job?.id||result?.jobs?.[0]?.id||null;
    toast('Calibration batch import queued',`${result.file_count} source file(s) → ${scan.calibration_set_id}. One job will report progress for the complete set.`);
    await refreshState({quiet:false});
  } catch(error) { showError(error,'Could not import calibration session sources'); }
  finally{if(button?.isConnected){button.disabled=false;button.textContent='Import scanned sources';}}
}

async function submitCalibrationImport() {
  try {
    const path=$('#cal-import-path').value.trim();if(!path)throw new Error('Select a .g3dcal or .oculocal profile');
    const sessions=$('#cal-import-sessions').value.split(',').map(x=>x.trim()).filter(Boolean);
    const setRecord=activeCalibrationSetRecord();
    await api('/api/jobs/calibration-profile-import',{method:'POST',body:{session_id:selectedSessionId(),profile_path:path,label:$('#cal-import-label').value.trim()||null,hardware_id:$('#cal-import-hardware').value.trim()||null,mount_version:$('#cal-import-mount').value.trim()||null,valid_sessions:sessions,source_note:$('#cal-import-note').value.trim()||null,calibration_set_id:setRecord?.id||activeCalibrationSetId(),calibration_revision_id:setRecord?.active_revision_id||null}});
    toast('Calibration profile import queued','The registered result will include normalized .oculocal and portable .g3dcal files.');await refreshState({quiet:false});
  }catch(e){showError(e,'Could not import calibration profile')}
}

async function submitCalibrationRegisterMeasured() {
  try {
    const profile=$('#cal-register-measured-profile').value;
    if(!profile)throw new Error('Select an older measured calibration result');
    const sessions=$('#cal-register-measured-sessions').value.split(',').map(x=>x.trim()).filter(Boolean);
    await api('/api/jobs/calibration-profile-register-measured',{method:'POST',body:{
      session_id:selectedSessionId(),profile_artifact_id:profile,
      mount_version:$('#cal-register-measured-mount').value.trim()||'mount_v1',
      valid_sessions:sessions,label:null
    }});
    toast('Measured profile registration queued','OculoStudio will infer ETM-A/ETM-B from the tracker selected by calibration; the calibration itself is not rerun.');
    await refreshState({quiet:false});
  }catch(e){showError(e,'Could not register measured calibration result')}
}

async function submitCalibrationTransfer() {
  try {
    const profile=$('#cal-transfer-profile').value,tracker=$('#cal-transfer-tracker').value;if(!profile||!tracker)throw new Error('Select a source hardware profile and current ETM motion');
    await api('/api/jobs/calibration-transfer',{method:'POST',body:{session_id:selectedSessionId(),tracker_asset_id:tracker,profile_artifact_id:profile,profile_path:null,interaction_id:$('#cal-transfer-interaction').value.trim()||null,role_id:$('#cal-transfer-role').value.trim()||null,label:$('#cal-transfer-label').value.trim()||null,maximum_shape_rms_mm:Number($('#cal-transfer-max-rms').value),high_confidence_rms_mm:Number($('#cal-transfer-high-rms').value),minimum_marker_valid_fraction:Number($('#cal-transfer-valid').value)}});
    toast('Calibration transfer validation queued','A passed marker-shape check creates a session-specific profile with review-required provenance.');await refreshState({quiet:false});
  }catch(e){showError(e,'Could not validate calibration transfer')}
}

async function submitCalibrationAcceptance() {
  try {
    const profile=$('#cal-accept-profile').value,trim=$('#cal-accept-trim').value,assignment=$('#cal-accept-etm').value,role=$('#cal-accept-role').value;
    if(!profile||!trim||!assignment||!role)throw new Error('Select transferred profile, trim, ETM assignment, and wearer role');
    await api('/api/jobs/calibration-acceptance',{method:'POST',body:{
      session_id:selectedSessionId(),transferred_profile_artifact_id:profile,native_trim_artifact_id:trim,
      etm_assignment_artifact_id:assignment,wearer_role_id:role,
      participant_association_artifact_id:$('#cal-accept-association').value||null,
      known_target_observations_path:$('#cal-accept-known').value.trim()||null,
      label:$('#cal-accept-label').value.trim()||`${session()?.label||'S'}_${role}_calibration_acceptance`,
      maximum_sync_rms_ms:Number($('#cal-accept-sync-rms').value),
      minimum_projection_valid_fraction:Number($('#cal-accept-proj-valid').value),
      minimum_partner_match_frame_fraction:Number($('#cal-accept-match').value),
      maximum_median_normalized_center_error:Number($('#cal-accept-centre').value),
      minimum_median_projected_iou:Number($('#cal-accept-iou').value),
      maximum_systematic_offset_normalized:Number($('#cal-accept-offset').value),
      maximum_review_fraction:Number($('#cal-accept-review').value),
      known_target_maximum_median_angular_error_deg:Number($('#cal-accept-angle').value),
      known_target_maximum_point_to_ray_distance_mm:Number($('#cal-accept-distance').value),
      output_parquet:$('#cal-accept-parquet').dataset.on==='true'
    }});
    toast('Calibration acceptance queued','The result will combine transfer, formal synchronization, head projection, and optional known-target evidence.');await refreshState({quiet:false});
  }catch(e){showError(e,'Could not run calibration acceptance')}
}

async function submitCalibration() {
  try {
    const trackers=$$('.cal-tracker-check:checked').map(x=>x.value),native=$('#cal-native').value,board=$('#cal-board').value;
    const mode=$('#cal-board-mode').value,definition=$('#cal-board-def').value.trim();
    if(!native||!board||!trackers.length)throw new Error('Select Native, board, and one or more tracker candidates');
    if(mode==='explicit_definition'&&!definition)throw new Error('Explicit board mode requires the real Board Definition JSON');
    const validSessions=$('#cal-valid-sessions').value.split(',').map(x=>x.trim()).filter(Boolean);
    await api('/api/jobs/calibration',{method:'POST',body:{
      session_id:selectedSessionId(),native_asset_id:native,tracker_asset_ids:trackers,board_asset_id:board,
      board_mode:mode,board_definition_path:definition||null,world_asset_id:$('#cal-world').value||null,
      intrinsics_path:$('#cal-intrinsics').value.trim()||null,
      participant_id:$('#cal-participant').value.trim()||'P01',profile_name:$('#cal-name').value.trim()||'Calibration',
      tracker_marker_labels:{},sample_every_n_frames:Number($('#cal-sample-every').value||2),
      max_detection_frames:Number($('#cal-max-detections').value||400),max_pose_gap_sec:.1,minimum_identity_margin_mm:10,
      minimum_tracker_identity_confidence:$('#cal-identity-confidence').value,
      include_raw_identity_metadata:$('#cal-raw-identity').dataset.on==='true',
      hardware_id:$('#cal-hardware').value.trim()||null,mount_version:$('#cal-mount').value.trim()||null,
      valid_sessions:validSessions,auto_register_hardware:$('#cal-auto-register').dataset.on==='true',
      auto_profile_name:$('#cal-auto-name').dataset.on==='true',allow_hardware_id_override:$('#cal-hardware-override').dataset.on==='true',
      output_g3dcal:$('#cal-output-g3dcal').dataset.on==='true',
      calibration_set_id:activeCalibrationSetRecord()?.id||activeCalibrationSetId(),
      calibration_revision_id:activeCalibrationSetRecord()?.active_revision_id||null
    }});
    toast('Calibration job queued',mode==='auto_legacy_16_marker'?'Tracker selection, ETM-A/ETM-B hardware registration, automatic offset search and Workflow-B QC will run as one job.':'Explicit calibration will select the tracker and auto-register its hardware identity when possible.');
    await refreshState({quiet:false});
  }catch(e){showError(e,'Could not start calibration')}
}


async function submitETMAssignment() {
  try {
    const r01=$('#etm-body-r01').value,r02=$('#etm-body-r02').value,a=$('#etm-candidate-a').value,b=$('#etm-candidate-b').value;
    if(!r01||!r02||!a||!b)throw new Error('Select R01 body, R02 body, ETM A and ETM B C3Ds');
    if(r01===r02)throw new Error('R01 and R02 body assets must be different');
    if(a===b)throw new Error('ETM A and ETM B assets must be different');
    const response=await api('/api/jobs/etm-role-assignment',{method:'POST',body:{
      session_id:selectedSessionId(),body_asset_ids:[r01,r02],etm_asset_ids:[a,b],role_labels:{[r01]:'R01',[r02]:'R02'},
      body_marker_prefixes:{},etm_marker_prefixes:{},calibration_profile_artifact_ids:[],
      maximum_head_distance_mm:Number($('#etm-max-distance').value),minimum_role_margin_mm:Number($('#etm-role-margin').value),
      minimum_role_ratio:1.5,minimum_global_margin_mm:Number($('#etm-role-margin').value),
      maximum_profile_shape_rms_mm:8,minimum_profile_shape_margin_mm:4,
      label:$('#etm-assignment-label').value.trim()||'ETM wearer assignment'
    }});
    appState.pendingETMAssignmentJobId=response?.job?.id||null;
    toast('Wearer resolution queued','This phase uses only interaction body/ETM motion; hardware profiles are intentionally excluded.');
    await refreshState({quiet:false});
  }catch(e){showError(e,'Could not resolve ETM wearer / role')}
}

async function submitCalibrationBinding() {
  try {
    const assignment=$('#cal-binding-assignment')?.value||'';
    const profiles=$$('.cal-binding-profile-check:checked').map(input=>input.value);
    const stage=calibrationWorkflowState();
    const timeSyncArtifacts=stage.rows.map(row=>row.timeSyncArtifact).filter(Boolean);
    const timeSyncIds=[...new Set(timeSyncArtifacts.map(item=>item.id))];
    if(!assignment)throw new Error('Complete Phase 1 and select the resolved wearer assignment');
    if(profiles.length<2)throw new Error('Select both physical ETM hardware profiles');
    if(timeSyncIds.length<2||!stage.rows.every(row=>row.timeSyncReady))throw new Error('Complete and verify Pupil IMU ↔ assigned ETM synchronization for both R01 and R02');
    const response=await api('/api/jobs/calibration-profile-binding',{method:'POST',body:{
      session_id:selectedSessionId(),assignment_artifact_id:assignment,calibration_profile_artifact_ids:profiles,
      time_sync_artifact_ids:timeSyncIds,require_time_sync:true,require_distributed_drift:true,
      maximum_profile_shape_rms_mm:Number($('#cal-binding-max-rms').value||8),
      minimum_profile_shape_margin_mm:Number($('#cal-binding-margin').value||4),
      label:$('#cal-binding-label').value.trim()||'Calibration handoff'
    }});
    appState.pendingCalibrationBindingJobId=response?.job?.id||null;
    toast('Calibration handoff queued','The immutable handoff will bind wearer assignment, hardware profiles and role-specific affine TimeGraph mappings.');
    await refreshState({quiet:false});
  }catch(e){showError(e,'Could not create the synchronized Calibration handoff')}
}

async function submitC3DDefinedAlignedClips(){
  try{
    const stage=calibrationWorkflowState();
    const handoff=stage.handoff;
    if(!handoff||!stage.allRolesReady)throw new Error('Complete the synchronized Calibration handoff before creating aligned clips');
    const roleInputs={};
    for(const role of ['R01','R02']){
      const suffix=role.toLowerCase();
      const nativeId=$(`#cal-c3d-clip-native-${suffix}`)?.value||'';
      const timeseriesId=$(`#cal-c3d-clip-timeseries-${suffix}`)?.value||'';
      const facemapperId=$(`#cal-c3d-clip-facemapper-${suffix}`)?.value||'';
      if(!nativeId||!timeseriesId||!facemapperId)throw new Error(`Select ${role} NRD, TSD and FaceMapper inputs`);
      roleInputs[role]={native_asset_id:nativeId,timeseries_asset_id:timeseriesId,facemapper_asset_id:facemapperId};
      appState.calibrationClipSelections[role]={...roleInputs[role]};
    }
    const boundaryTolerance=Number($('#cal-c3d-clip-boundary-tolerance')?.value||20);
    if(!Number.isFinite(boundaryTolerance)||boundaryTolerance<0||boundaryTolerance>100)throw new Error('Boundary tolerance must be between 0 and 100 ms');
    const response=await api('/api/jobs/c3d-defined-aligned-clips',{method:'POST',body:{
      session_id:selectedSessionId(),
      calibration_handoff_artifact_id:handoff.id,
      role_inputs:roleInputs,
      interval_policy:$('#cal-c3d-clip-policy')?.value||'clip',
      face_tolerance_ms:5,
      boundary_tolerance_ms:boundaryTolerance,
      label_prefix:$('#cal-c3d-clip-label')?.value.trim()||'C3D-defined aligned clip',
    }});
    appState.pendingC3DDefinedClipsJobId=response?.job?.id||null;
    toast('C3D-defined clip job queued','R01 and R02 will use one common Vicon body interval, role-specific inverse TimeGraphs and exact NRD frame timestamps.');
    await refreshState({quiet:false});
  }catch(error){showError(error,'Could not create C3D-defined aligned clips')}
}

async function submitGaze3D() {
  let context=null;
  try {
    const cloud=$('#ray-cloud').value,assignment=$('#ray-assignment').value||null,role=$('#ray-role').value||null,
      tracker=$('#ray-tracker').value||null,art=$('#ray-profile-artifact').value||null,path=$('#ray-profile-path').value.trim()||null,
      scope=$('#ray-scope').value,clip=$('#ray-clip')?.value||null,trim=$('#ray-native-trim')?.value||null,
      alignmentMode=scope==='native_trim'?($('#ray-alignment-mode')?.value||'timegraph'):'timegraph',
      sharedStartOffset=Number($('#ray-shared-start-offset')?.value||0),
      acceptance=$('#ray-calibration-acceptance').value||null,
      requireAcceptance=$('#ray-require-acceptance').dataset.on==='true',syncPolicy=$('#ray-sync-policy')?.value||'auto',
      maximumOffset=Number($('#ray-sync-max-offset')?.value||120);
    const gazeStage=calibrationWorkflowState(role||calibrationContextRole());
    const handoffSyncRequired=Boolean(gazeStage.handoff?.parameters?.require_time_sync||gazeStage.handoff?.qc?.time_sync_required);
    if(handoffSyncRequired&&alignmentMode!=='timegraph')throw new Error('The selected Calibration handoff requires its verified Pupil IMU ↔ ETM TimeGraph; shared-start alignment is blocked');
    if(handoffSyncRequired&&!gazeStage.current?.handoffSyncReady)throw new Error(`Calibration time synchronization for ${role||'this role'} is not production-ready`);
    if(!cloud)throw new Error('Select Timeseries data');
    if(!assignment&&!tracker)throw new Error('Complete Calibration and select its ETM-role assignment handoff');
    if(assignment&&!role)throw new Error('Select the wearer role');
    if(!assignment&&!art&&!path)throw new Error('Complete Calibration and bind a compatible hardware profile');
    if(scope==='clip'&&!clip)throw new Error('Select a legacy Global-Time clip');
    if(scope==='native_trim'&&!trim)throw new Error('Select a Native-reference trim');
    if(!Number.isFinite(sharedStartOffset)||Math.abs(sharedStartOffset)>5)throw new Error('Shared-start residual offset must be between -5 and +5 seconds');
    const body={session_id:selectedSessionId(),timeseries_asset_id:cloud,
      tracker_asset_id:tracker,profile_path:path,profile_artifact_id:art,etm_assignment_artifact_id:assignment,wearer_role_id:assignment?role:null,
      calibration_acceptance_artifact_id:acceptance,require_calibration_acceptance:requireAcceptance,
      clip_id:scope==='clip'?clip:null,native_trim_artifact_id:scope==='native_trim'?trim:null,
      scope,alignment_mode:alignmentMode,shared_start_offset_sec:sharedStartOffset,
      max_pose_gap_sec:Number($('#ray-gap').value),output_parquet:$('#ray-parquet').dataset.on==='true'};
    if(scope==='native_trim'&&alignmentMode==='pretrimmed_shared_start'){
      await queueGaze3D(body);return;
    }
    context=gazeSyncContext(cloud,assignment,role,tracker);
    if(context){
      const status=timeGraphPathStatus(context.source_domain_id,context.target_domain_id);
      if(!status.formal){
        if(syncPolicy==='manual'){openGazeSyncRequired(context);return;}
        await runAutomaticGazeSync(body,context,maximumOffset);return;
      }
    }
    await queueGaze3D(body);
  }catch(e){
    if(/synchronization required|verified synchronization|required before Gaze 3D|TimeGraph path/i.test(String(e?.message||e))){openGazeSyncRequired(context,String(e?.message||e));return;}
    showError(e,'Could not generate gaze rays');
  }
}

async function submitSemanticTargets() {
  try {
    const rayArtifact=$('#target-rays').value;
    if(!rayArtifact) throw new Error('Select a Gaze 3D artifact');
    let prefixes={};
    const prefixText=$('#target-prefixes').value.trim();
    if(prefixText){ try{prefixes=JSON.parse(prefixText)}catch(_){throw new Error('Marker prefixes must be valid JSON: {"asset_id":"subject_prefix"}')} }
    const bodyAssets=$$('.target-body-check:checked').map(x=>x.value);
    await api('/api/jobs/semantic-targets',{method:'POST',body:{
      session_id:selectedSessionId(),
      gaze_rays_artifact_id:rayArtifact,
      timeseries_asset_id:$('#target-cloud').value||null,
      facemapper_asset_id:$('#target-face').value||null,
      body_asset_ids:bodyAssets,
      body_profile_path:$('#target-body-profile').value.trim()||null,
      entity_marker_prefixes:prefixes,
      densepose_asset_id:$('#target-densepose').value||null,
      densepose_profile_path:$('#target-densepose-profile').value.trim()||null,
      scene_frame_tolerance_ms:Number($('#target-frame-tolerance').value),
      face_tolerance_ms:Number($('#target-face-tolerance').value),
      face_left_right_policy:$('#target-face-lr').value,
      max_marker_gap_sec:Number($('#target-marker-gap').value),
      uncertainty_cone_deg:Number($('#target-cone').value),
      maximum_target_distance_m:Number($('#target-distance').value),
      region_frame_policy:$('#target-frame-policy').value,
      output_parquet:$('#target-parquet').dataset.on==='true'
    }});
    toast('Semantic target job queued','Windows geometry-first mapping with optional DensePose-result import.');
    await refreshState({quiet:false});
  } catch(e){showError(e,'Could not annotate semantic gaze targets')}
}

async function submitResultsPackage() {
  try {
    const artifactIds=$$('.result-package-check:checked').map(x=>x.value);
    if(!artifactIds.length)throw new Error('Select at least one artifact to include.');
    const body={
      session_id:selectedSessionId(),
      artifact_ids:artifactIds,
      session_code:$('#result-session-code')?.value.trim()||null,
      interaction_code:$('#result-interaction-code')?.value.trim()||null,
      role_code:$('#result-role-code')?.value.trim()||null,
      label:$('#result-package-label')?.value.trim()||'OculoStudio results',
      copy_mode:$('#result-copy-mode')?.value||'copy'
    };
    await api('/api/jobs/results-package',{method:'POST',body});
    toast('Results package queued', body.copy_mode==='copy'?'Selected outputs will be copied into exports/S/I/R.':'An index-only package will reference the existing immutable artifacts.');
    await refreshState({quiet:false});
  } catch(e){showError(e,'Could not build results package')}
}

async function submitQC() {
  try { await api('/api/jobs/qc',{method:'POST',body:{session_id:selectedSessionId()}});toast('QC report queued');await refreshState({quiet:false}); }
  catch(e){showError(e,'Could not export QC report')}
}

async function openLocalPath(path) {
  try { await api('/api/open-path',{method:'POST',body:{path}}); }
  catch(e){showError(e,'Could not open path')}
}

async function revealLocalPath(path) {
  try { await api('/api/reveal-path',{method:'POST',body:{path}}); }
  catch(e){showError(e,'Could not reveal path')}
}


const PANEL_LAYOUT_STORAGE_KEY='oculostudio.panel-widths.v2';
const PANEL_LAYOUT_DEFAULTS={wide:{left:248,right:300},compact:{left:220,right:280}};

function installPanelResizerStyles(){
  if(document.getElementById('oculostudio-panel-resizer-styles'))return;
  const style=document.createElement('style');
  style.id='oculostudio-panel-resizer-styles';
  style.textContent=`
    .panel-resizer-overlay{position:absolute;top:0;bottom:0;width:18px;margin:0;padding:0;border:0;background:transparent;z-index:25;cursor:col-resize;touch-action:none;transform:translateX(-50%);outline:none}
    .panel-resizer-overlay .panel-resizer-grip{position:absolute;left:50%;top:50%;width:5px;height:54px;transform:translate(-50%,-50%);border-radius:999px;background:radial-gradient(circle,rgba(237,251,54,.92) 1.2px,transparent 1.6px) center 3px/5px 9px repeat-y;opacity:.48;filter:drop-shadow(0 1px 1px rgba(24,27,13,.55));transition:opacity .15s ease,background-color .15s ease,box-shadow .15s ease;pointer-events:none}
    .panel-resizer-overlay:hover .panel-resizer-grip,.panel-resizer-overlay:focus-visible .panel-resizer-grip,.panel-resizer-overlay.is-dragging .panel-resizer-grip{opacity:1;background-color:rgba(237,251,54,.10);box-shadow:0 0 0 4px rgba(237,251,54,.08)}
    .panel-resizer-overlay:focus-visible{box-shadow:inset 0 0 0 2px rgba(237,251,54,.55);border-radius:999px}
    body.oculo-panel-resizing,body.oculo-panel-resizing *{cursor:col-resize!important;user-select:none!important}
  `;
  document.head.appendChild(style);
}

function panelLayoutDefaults(){
  const compact=window.innerWidth<=1280;
  return {...(compact?PANEL_LAYOUT_DEFAULTS.compact:PANEL_LAYOUT_DEFAULTS.wide)};
}

function panelLayoutMetrics(grid){
  const computed=getComputedStyle(grid);
  const total=Math.max(1,Math.round(grid.getBoundingClientRect().width||window.innerWidth-56));
  const gap=Math.max(0,parseFloat(computed.columnGap||computed.gap)||9);
  const leftMin=180;
  const rightMin=220;
  const desiredCenter=total>=1400?600:total>=1080?520:360;
  const maxSideTotal=Math.max(leftMin+rightMin,total-(gap*2)-desiredCenter);
  return {
    total,gap,leftMin,rightMin,
    leftMax:Math.max(leftMin,Math.min(520,maxSideTotal-rightMin)),
    rightMax:Math.max(rightMin,Math.min(680,maxSideTotal-leftMin)),
    maxSideTotal,
  };
}

function clampPanelWidths(widths,grid,activeSide=null){
  const defaults=panelLayoutDefaults();
  const metrics=panelLayoutMetrics(grid);
  const finite=(value,fallback)=>Number.isFinite(Number(value))?Number(value):fallback;
  let left=Math.min(metrics.leftMax,Math.max(metrics.leftMin,finite(widths?.left,defaults.left)));
  let right=Math.min(metrics.rightMax,Math.max(metrics.rightMin,finite(widths?.right,defaults.right)));
  if(left+right>metrics.maxSideTotal){
    if(activeSide==='left'){
      left=Math.max(metrics.leftMin,metrics.maxSideTotal-right);
    }else if(activeSide==='right'){
      right=Math.max(metrics.rightMin,metrics.maxSideTotal-left);
    }else{
      let overflow=left+right-metrics.maxSideTotal;
      const leftRoom=Math.max(0,left-metrics.leftMin);
      const rightRoom=Math.max(0,right-metrics.rightMin);
      const room=leftRoom+rightRoom;
      if(room>0){
        const leftReduction=Math.min(leftRoom,overflow*(leftRoom/room));
        left-=leftReduction;
        overflow-=leftReduction;
        right-=Math.min(rightRoom,overflow);
      }
    }
  }
  return {left:Math.round(left),right:Math.round(right),metrics};
}

function currentPanelWidths(grid){
  const defaults=panelLayoutDefaults();
  return {
    left:Number(grid.dataset.leftPanelWidth)||defaults.left,
    right:Number(grid.dataset.rightPanelWidth)||defaults.right,
  };
}

function persistPanelWidths(widths){
  try{localStorage.setItem(PANEL_LAYOUT_STORAGE_KEY,JSON.stringify({left:widths.left,right:widths.right,layout:'overlay-v2'}))}catch(_){ }
}

function applyPanelWidths(grid,handles,widths,{persist=false,activeSide=null}={}){
  const normalized=clampPanelWidths(widths,grid,activeSide);
  const {left,right,metrics}=normalized;
  grid.dataset.leftPanelWidth=String(left);
  grid.dataset.rightPanelWidth=String(right);
  grid.style.gridTemplateColumns=`${left}px minmax(0,1fr) ${right}px`;
  const leftHandle=handles?.left;
  const rightHandle=handles?.right;
  if(leftHandle){
    leftHandle.style.left=`${left+(metrics.gap/2)}px`;
    leftHandle.setAttribute('aria-valuemin',String(metrics.leftMin));
    leftHandle.setAttribute('aria-valuemax',String(metrics.leftMax));
    leftHandle.setAttribute('aria-valuenow',String(left));
    leftHandle.title=`Drag to resize left panel · ${left}px · Double-click to reset`;
  }
  if(rightHandle){
    rightHandle.style.left=`${metrics.total-right-(metrics.gap/2)}px`;
    rightHandle.setAttribute('aria-valuemin',String(metrics.rightMin));
    rightHandle.setAttribute('aria-valuemax',String(metrics.rightMax));
    rightHandle.setAttribute('aria-valuenow',String(right));
    rightHandle.title=`Drag to resize right panel · ${right}px · Double-click to reset`;
  }
  if(persist)persistPanelWidths(normalized);
  return normalized;
}

function createPanelResizer(grid,side){
  const handle=document.createElement('div');
  handle.className='panel-resizer-overlay';
  handle.dataset.side=side;
  handle.tabIndex=0;
  handle.setAttribute('role','separator');
  handle.setAttribute('aria-orientation','vertical');
  handle.setAttribute('aria-label',side==='left'?'Resize project and assets panel':'Resize module properties panel');
  // These critical declarations keep the handle out of CSS Grid even if a stale
  // stylesheet is ever mixed with the current JavaScript.
  Object.assign(handle.style,{
    position:'absolute',top:'0',bottom:'0',width:'18px',margin:'0',padding:'0',border:'0',
    background:'transparent',zIndex:'25',cursor:'col-resize',touchAction:'none',transform:'translateX(-50%)'
  });
  const grip=document.createElement('span');
  grip.className='panel-resizer-grip';
  grip.setAttribute('aria-hidden','true');
  handle.appendChild(grip);
  grid.appendChild(handle);
  return handle;
}

function initPanelResizers(){
  const grid=$('.studio-grid');
  if(!grid||grid.dataset.panelResizeMode==='overlay-v2')return;

  // Remove v0.20.0's in-grid separators if an old HTML document survived in a
  // WebView cache. Their presence as normal grid children caused the scrambled
  // two-row layout reported on Windows.
  grid.querySelectorAll('#left-panel-resizer,#right-panel-resizer').forEach(node=>node.remove());
  document.body.classList.remove('panel-resizing');
  installPanelResizerStyles();
  grid.dataset.panelResizeMode='overlay-v2';
  grid.style.position='relative';

  const handles={left:createPanelResizer(grid,'left'),right:createPanelResizer(grid,'right')};
  let saved=panelLayoutDefaults();
  try{
    const parsed=JSON.parse(localStorage.getItem(PANEL_LAYOUT_STORAGE_KEY)||'null');
    if(parsed&&Number.isFinite(Number(parsed.left))&&Number.isFinite(Number(parsed.right)))saved={left:Number(parsed.left),right:Number(parsed.right)};
  }catch(_){ }
  applyPanelWidths(grid,handles,saved,{persist:false});

  let dragging=false;
  const reset=()=>applyPanelWidths(grid,handles,panelLayoutDefaults(),{persist:true});
  const bind=(handle,side)=>{
    handle.addEventListener('dblclick',event=>{event.preventDefault();reset()});
    handle.addEventListener('keydown',event=>{
      if(event.key==='Home'){event.preventDefault();reset();return}
      if(!['ArrowLeft','ArrowRight'].includes(event.key))return;
      event.preventDefault();
      const step=event.shiftKey?24:8;
      const movement=event.key==='ArrowRight'?step:-step;
      const next=currentPanelWidths(grid);
      if(side==='left')next.left+=movement;
      else next.right-=movement;
      applyPanelWidths(grid,handles,next,{persist:true,activeSide:side});
    });
    handle.addEventListener('pointerdown',event=>{
      if(event.button!==0)return;
      event.preventDefault();
      dragging=true;
      const startX=event.clientX;
      const start=currentPanelWidths(grid);
      handle.classList.add('is-dragging');
      document.body.classList.add('oculo-panel-resizing');
      try{handle.setPointerCapture(event.pointerId)}catch(_){ }
      const move=moveEvent=>{
        const delta=moveEvent.clientX-startX;
        const next={...start};
        if(side==='left')next.left=start.left+delta;
        else next.right=start.right-delta;
        applyPanelWidths(grid,handles,next,{persist:false,activeSide:side});
      };
      const finish=()=>{
        window.removeEventListener('pointermove',move);
        window.removeEventListener('pointerup',finish);
        window.removeEventListener('pointercancel',finish);
        dragging=false;
        handle.classList.remove('is-dragging');
        document.body.classList.remove('oculo-panel-resizing');
        const finalWidths=applyPanelWidths(grid,handles,currentPanelWidths(grid),{persist:true,activeSide:side});
        window.dispatchEvent(new CustomEvent('oculostudio:panel-layout-changed',{detail:{left:finalWidths.left,right:finalWidths.right}}));
      };
      window.addEventListener('pointermove',move);
      window.addEventListener('pointerup',finish);
      window.addEventListener('pointercancel',finish);
    });
  };
  bind(handles.left,'left');
  bind(handles.right,'right');

  let resizeFrame=0;
  const refit=()=>{
    if(dragging)return;
    cancelAnimationFrame(resizeFrame);
    resizeFrame=requestAnimationFrame(()=>applyPanelWidths(grid,handles,currentPanelWidths(grid),{persist:false}));
  };
  if('ResizeObserver' in window){
    const observer=new ResizeObserver(refit);
    observer.observe(grid);
  }else{
    window.addEventListener('resize',refit);
  }
  window.resetOculoStudioPanelWidths=reset;
}

async function init() {
  initPanelResizers();
  await loadRuntime();
  await refreshState({quiet:true, fullRender:true});
  appState.timer = setInterval(()=>refreshState({quiet:true, fullRender:false}), 1200);
}

document.addEventListener('DOMContentLoaded', init);
