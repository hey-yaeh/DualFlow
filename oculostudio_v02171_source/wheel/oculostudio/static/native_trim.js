/* Native-video-reference clipping is separate from the global/3D workflow.
 * Nanosecond UTC values are strings in the new API; never parse them as Number.
 * Drafts survive job polling and are isolated by project + session + interaction + role.
 */
const nativeTrimDrafts = new Map();

// Native-reference trim is role-specific. R01 and R02 often point to different
// NRD/TSD/FaceMapper sources and have different start offsets. Keeping one draft
// per project/session caused the Expert ribbon to switch roles while the Trim
// form silently retained the previous role's inputs. Scope every draft by the
// active interaction and role so the visual context and scientific inputs can
// never diverge.
function nativeWorkflowContext() {
  const ws=(typeof appState!=='undefined' && appState.data?.interaction_workspace)||{};
  return {
    interactionId:String(ws.selected_interaction_id||'').trim().toUpperCase(),
    roleId:String(ws.selected_role_id||'').trim().toUpperCase(),
  };
}
function nativeDraftKey() {
  const ctx=nativeWorkflowContext();
  return `${project()?.id||''}:${selectedSessionId()}:${ctx.interactionId||'*'}:${ctx.roleId||'*'}`;
}
function nativeAssetContextText(asset) {
  if(!asset)return '';
  const identity=(asset.metadata?.identity&&typeof asset.metadata.identity==='object')?asset.metadata.identity:{};
  return [identity.session_id,identity.interaction_id,identity.role_id,identity.participant_id,asset.participant_id,asset.label,asset.source_path]
    .filter(Boolean).join(' ').toLowerCase();
}
function nativeAssetMatchesContext(assetId, ctx=nativeWorkflowContext()) {
  if(!assetId)return false;
  if(!ctx.interactionId&&!ctx.roleId)return true;
  const asset=session()?.assets?.[assetId];if(!asset)return false;
  const text=nativeAssetContextText(asset);
  const identity=(asset.metadata?.identity&&typeof asset.metadata.identity==='object')?asset.metadata.identity:{};
  const interaction=String(identity.interaction_id||'').trim().toUpperCase();
  const role=String(identity.role_id||'').trim().toUpperCase();
  if(ctx.interactionId && interaction && interaction!==ctx.interactionId)return false;
  if(ctx.roleId && role && role!==ctx.roleId)return false;
  if(ctx.interactionId && !interaction && /(?:^|[^a-z0-9])i\d+(?![a-z0-9])/i.test(text) && !text.includes(ctx.interactionId.toLowerCase()))return false;
  if(ctx.roleId && !role && /(?:^|[^a-z0-9])r\d+(?![a-z0-9])/i.test(text) && !text.includes(ctx.roleId.toLowerCase()))return false;
  // A role-scoped trim must not inherit an explicitly different role. Roleless
  // legacy assets remain available only when the filename/identity is unscoped.
  if(ctx.roleId && !role && /(?:^|[^a-z0-9])r\d+(?![a-z0-9])/i.test(text))return text.includes(ctx.roleId.toLowerCase());
  return true;
}
function nativeRequestMatchesContext(request, ctx=nativeWorkflowContext()) {
  if(!request)return false;
  if(!ctx.interactionId&&!ctx.roleId)return true;
  return ['native_asset_id','timeseries_asset_id','facemapper_asset_id']
    .every(key=>nativeAssetMatchesContext(request[key],ctx));
}
function nativeLatestContextSeed(ctx=nativeWorkflowContext()) {
  const current=session();if(!current)return {last:null,request:null,artifact:null};
  const last=current.metadata?.native_reference_trim_last||null;
  if(nativeRequestMatchesContext(last?.request,ctx))return {last,request:last.request,artifact:null};
  const matches=Object.values(current.artifacts||{}).filter(artifact=>
    artifact.kind==='native_reference_trim' && nativeRequestMatchesContext(artifact.parameters,ctx));
  matches.sort((a,b)=>String(a.created_at||'').localeCompare(String(b.created_at||'')));
  const artifact=matches.at(-1)||null;
  return {last:null,request:artifact?.parameters||null,artifact};
}
function nativeContextAssets(kinds) {
  const rows=assets(kinds),ctx=nativeWorkflowContext();
  if(!ctx.interactionId&&!ctx.roleId)return rows;
  const matches=rows.filter(row=>nativeAssetMatchesContext(row.id,ctx));
  return matches.length?matches:rows;
}
function nativeDraft() {
  const key=nativeDraftKey();
  if (!nativeTrimDrafts.has(key)) {
    const ctx=nativeWorkflowContext();
    const seed=nativeLatestContextSeed(ctx), last=seed.last, r=seed.request;
    const metadata=session()?.metadata||{};
    const overlayLast=metadata.native_reference_overlay_last;
    const overlayByPreview=metadata.native_reference_overlay_by_preview||{};
    const seedPreviewId=String(r?.preview_id||last?.preview?.preview_id||'');
    const savedOverlay=(seedPreviewId&&overlayByPreview[seedPreviewId])||
      ((seedPreviewId&&overlayLast?.preview_id===seedPreviewId)?overlayLast:null);
    const seedAutomatic=!!r?.automatic_c3d_defined;
    const d={mode:'native', contextKey:key, native:r?.native_asset_id||'', tsd:r?.timeseries_asset_id||'', fm:r?.facemapper_asset_id||'',
      part:r?.native_video_member||'', label:r?.label||'Native data clip', start:r?.start_timecode||'', duration:r?.duration_timecode||'',
      fps:String(r?.timecode_fps??30), timeValidation:null, timeTimer:null, timeCache:new Map(),
      policy:r?.interval_policy||'clip', tolerance:String(r?.face_tolerance_ms??5), ffprobe:r?.ffprobe_path||'',
      options:r?.native_video_member?[{video_member:r.native_video_member}]:[], preview:last?.preview||null,
      previewJob:null, overlayJob:null, exportJob:null, packageJob:null, result:last?.result||null,
      overlayResult:savedOverlay?.result||null, overlaySubmittedSignature:null, packageResult:null,
      overlayScope:'representative', overlaySeconds:'8', overlaySamples:'7', overlayFps:'30', overlayFaceBoxes:true,
      overlayLandmarks:true, overlayAoi:true, overlayGaze:true, overlayLeftRight:'anatomical_auto',
      packageLabel:nativePackageDefaultLabel(ctx), packageFps:'30', packageVideo:true, packageTsd:true, packageFm:true, packageAudio:true,
      facialSync:'', facialLabel:nativeFacialDefaultLabel(ctx), facialFps:'30', facialAudio:true,
      facialJob:null, facialResult:null, facialSubmitting:false,
      overlaySubmitting:false, packageSubmitting:false, submittedSignature:null,lastHandledPreview:null,lastHandledOverlay:null,lastHandledExport:null,lastHandledPackage:null,lastHandledFacial:null};
    if(d.preview)d.submittedSignature=nativeSignature(d);
    if(d.overlayResult)d.overlaySubmittedSignature=nativeOverlaySignature(d,seedPreviewId,seedAutomatic);
    nativeTrimDrafts.set(key,d);
  }
  return nativeTrimDrafts.get(key);
}
function nativeSignature(d) {
  return JSON.stringify([d.native,d.tsd,d.fm,d.part,d.label,d.start,d.duration,d.fps,d.policy,d.tolerance,d.ffprobe]);
}
function nativePreviewCurrent(d) {return d.preview && d.submittedSignature === nativeSignature(d)}
function nativeOverlaySignature(d, previewId=null, automatic=false) {
  const resolved=String(previewId||d.overlayResult?.preview_id||d.preview?.preview_id||'');
  return JSON.stringify([resolved,automatic?'AUTOMATIC_C3D_DEFINED':d.submittedSignature,d.overlayScope,d.overlayScope==='full'?'FULL':d.overlaySeconds,d.overlayScope==='distributed'?d.overlaySamples:'',d.overlayFps,d.overlayFaceBoxes,d.overlayLandmarks,d.overlayAoi,d.overlayGaze,d.overlayLeftRight]);
}
function nativeOverlayIsAutomatic(overlay) {return !!(overlay?.automatic_c3d_defined||overlay?.preview_kind==='aligned_clip_qc')}
function nativeOverlayCurrent(d) {
  const overlay=d.overlayResult;if(!overlay)return false;
  const automatic=nativeOverlayIsAutomatic(overlay);
  if(!automatic&&!nativePreviewCurrent(d))return false;
  return d.overlaySubmittedSignature===nativeOverlaySignature(d,overlay.preview_id,automatic);
}
function nativeAutomaticPreviewId(auto=nativeAutomaticClipState()) {return String(auto?.parameters?.preview_id||'')}
function nativeAutomaticOverlay(d=nativeDraft(), auto=nativeAutomaticClipState()) {
  const previewId=nativeAutomaticPreviewId(auto);
  return previewId&&d.overlayResult?.preview_id===previewId?d.overlayResult:null;
}
function nativePackageDefaultLabel(ctx=nativeWorkflowContext()) {
  return [ctx.interactionId||'aligned',ctx.roleId||'role','aligned clip package'].join(' ');
}
function nativeTrimArtifactForPackage(auto=nativeAutomaticClipState(), d=nativeDraft()) {
  if(auto?.trim)return auto.trim;
  const id=String(d.result?.artifact_id||'');
  return id?session()?.artifacts?.[id]||null:null;
}
function nativeAlignedPackageArtifact(auto=nativeAutomaticClipState(), d=nativeDraft()) {
  const trim=nativeTrimArtifactForPackage(auto,d);if(!trim)return null;
  const rows=Object.values(session()?.artifacts||{}).filter(artifact=>
    artifact.kind==='aligned_clip_package' && (artifact.parent_artifact_ids||[]).includes(trim.id));
  rows.sort((a,b)=>String(a.created_at||a.provenance?.created_at||'').localeCompare(String(b.created_at||b.provenance?.created_at||'')));
  return rows.at(-1)||null;
}
function nativeParentDir(path) {
  const text=String(path||'');const index=Math.max(text.lastIndexOf('/'),text.lastIndexOf('\\'));
  return index>=0?text.slice(0,index):text;
}
function nativeAlignedPackageResult(auto=nativeAutomaticClipState(), d=nativeDraft()) {
  const trim=nativeTrimArtifactForPackage(auto,d);
  if(d.packageResult && (!trim || String(d.packageResult.native_trim_artifact_id||'')===trim.id))return d.packageResult;
  const artifact=nativeAlignedPackageArtifact(auto,d);if(!artifact)return null;
  const files=artifact.files||[];
  const find=pattern=>files.find(row=>pattern.test(String(row.path||'')))?.path||null;
  const report=find(/aligned_clip_report\.html$/i), manifest=find(/aligned_clip_manifest\.json$/i);
  const video=find(/\.(mp4|mov|mkv|avi)$/i), frameMap=find(/aligned_ego_frame_map\.csv$/i);
  const tsd=find(/TSD_aligned\.zip$/i), fm=find(/FaceMapper_aligned\.zip$/i);
  const anchor=report||manifest||video||files[0]?.path||'';
  return {artifact,artifact_id:artifact.id,native_trim_artifact_id:trim?.id||artifact.parameters?.native_trim_artifact_id||'',
    output_dir:nativeParentDir(anchor),video,frame_map:frameMap,tsd_zip:tsd,facemapper_zip:fm,report_html:report,manifest,wearer_role_id:artifact.parameters?.wearer_role_id||null};
}
function nativeAlignedPackageVideoUrl(pkg) {
  const id=String(pkg?.artifact_id||pkg?.artifact?.id||'');
  return id?`/api/media/artifact/${encodeURIComponent(selectedSessionId())}/${encodeURIComponent(id)}`:'';
}
function nativeFacialDefaultLabel(ctx=nativeWorkflowContext()) {
  return [ctx.interactionId||'aligned',ctx.roleId||'role','aligned facial video'].join(' ');
}
function nativeCompatibleFacialSyncArtifacts(auto=nativeAutomaticClipState(), d=nativeDraft()) {
  const trim=nativeTrimArtifactForPackage(auto,d);if(!trim)return [];
  const ctx=nativeWorkflowContext();
  const nativeId=String(trim.parameters?.native_asset_id||'');
  return Object.values(session()?.artifacts||{}).filter(artifact=>{
    if(artifact.kind!=='time_sync')return false;
    if(String(artifact.parameters?.sync_type||artifact.qc?.sync_type||'')!=='shared_audio_video')return false;
    if(!Boolean(artifact.parameters?.visual_review_confirmed||artifact.qc?.visual_review_confirmed))return false;
    if(artifact.qc?.sync_verified===false)return false;
    const role=String(artifact.parameters?.wearer_role_id||artifact.qc?.wearer_role_id||'').toUpperCase();
    if(ctx.roleId&&role&&role!==ctx.roleId)return false;
    const reference=String(artifact.parameters?.reference_asset_id||artifact.qc?.reference_asset_id||'');
    if(nativeId&&reference&&reference!==nativeId)return false;
    return true;
  }).sort((a,b)=>String(a.created_at||'').localeCompare(String(b.created_at||'')));
}
function nativeSelectedFacialSync(auto=nativeAutomaticClipState(), d=nativeDraft()) {
  const rows=nativeCompatibleFacialSyncArtifacts(auto,d);
  if(!rows.length){d.facialSync='';return null;}
  if(!rows.some(row=>row.id===d.facialSync))d.facialSync=rows.at(-1).id;
  return rows.find(row=>row.id===d.facialSync)||rows.at(-1);
}
function nativeAlignedFacialArtifact(auto=nativeAutomaticClipState(), d=nativeDraft()) {
  const trim=nativeTrimArtifactForPackage(auto,d), sync=nativeSelectedFacialSync(auto,d);if(!trim)return null;
  const rows=Object.values(session()?.artifacts||{}).filter(artifact=>
    artifact.kind==='aligned_facial_video' && (artifact.parent_artifact_ids||[]).includes(trim.id) &&
    (!sync || (artifact.parent_artifact_ids||[]).includes(sync.id)));
  rows.sort((a,b)=>String(a.created_at||a.provenance?.created_at||'').localeCompare(String(b.created_at||b.provenance?.created_at||'')));
  return rows.at(-1)||null;
}
function nativeAlignedFacialResult(auto=nativeAutomaticClipState(), d=nativeDraft()) {
  const trim=nativeTrimArtifactForPackage(auto,d), sync=nativeSelectedFacialSync(auto,d);
  if(d.facialResult && (!trim||String(d.facialResult.native_trim_artifact_id||'')===trim.id) && (!sync||String(d.facialResult.time_sync_artifact_id||'')===sync.id))return d.facialResult;
  const artifact=nativeAlignedFacialArtifact(auto,d);if(!artifact)return null;
  const files=artifact.files||[];const find=pattern=>files.find(row=>pattern.test(String(row.path||'')))?.path||null;
  const video=find(/\.(mp4|mov|mkv|avi)$/i), frameMap=find(/aligned_facial_frame_map\.csv$/i);
  const report=find(/aligned_facial_sync_report\.html$/i), manifest=find(/aligned_facial_manifest\.json$/i);
  const anchor=report||manifest||video||files[0]?.path||'';
  return {artifact,artifact_id:artifact.id,native_trim_artifact_id:trim?.id||artifact.parameters?.native_trim_artifact_id||'',
    time_sync_artifact_id:sync?.id||artifact.parameters?.time_sync_artifact_id||'',output_dir:nativeParentDir(anchor),video,frame_map:frameMap,
    report_html:report,manifest,wearer_role_id:artifact.parameters?.wearer_role_id||null,video_stats:artifact.qc||{}};
}
function nativeAlignedFacialVideoUrl(result) {
  const id=String(result?.artifact_id||result?.artifact?.id||'');
  return id?`/api/media/artifact/${encodeURIComponent(selectedSessionId())}/${encodeURIComponent(id)}`:'';
}
function nativeOverlayVideoUrl(d, overlay=d.overlayResult) {const previewId=overlay?.preview_id||d.preview?.preview_id;return `/api/media/native-trim-overlay/${encodeURIComponent(selectedSessionId())}/${encodeURIComponent(previewId)}/${encodeURIComponent(overlay.overlay_id)}`}
function nativeTimeSignature(d) {return JSON.stringify([d.start.trim(),d.duration.trim(),d.fps])}
function nativeTimeReady(d) {
  return d.timeValidation?.signature===nativeTimeSignature(d) && d.timeValidation?.status==='valid';
}
function nativeTimeFeedback(d) {
  const v=d.timeValidation;
  if(!d.start.trim() || !d.duration.trim())return '<span>可直接输入 HH:MM:SS:FF，例如 00:00:49:21；末段 FF 是帧号。也支持秒 / HH:MM:SS.mmm。</span>';
  if(!v || v.signature!==nativeTimeSignature(d) || v.status==='pending')return '<span>正在解释时间码… Checking input format…</span>';
  if(v.status==='error')return `<strong>时间格式需要修正：</strong> ${esc(v.error)}`;
  const r=v.parsed;
  return `<div class="native-time-converted"><span><small>START · 换算起点</small><strong>${esc(r.start.normalized_time)}</strong></span>
    <span><small>DURATION · 保留长度</small><strong>${esc(r.duration.normalized_time)}</strong></span>
    <span><small>END · 起点 + 长度</small><strong>${esc(r.end.normalized_time)}</strong></span></div>
    <span class="form-help">${r.uses_frame_timecode?`已按 ${r.timecode_fps} fps 非丢帧时间码解释；只转换输入中的帧号。`:'当前输入是秒/小数秒；不受时间码帧率影响。'} 这是输入换算，不代表已通过源视频和数据检查；仍需点击 Preview alignment &amp; counts。</span>`;
}
function nativeUpdateTimeFeedback(d) {
  if(nativeDraft()!==d || appState.page!=='clips' || d.mode!=='native')return;
  const el=$('#native-time-interpretation');
  if(el){el.innerHTML=nativeTimeFeedback(d);el.classList.toggle('is-error',d.timeValidation?.status==='error')}
  const preview=$('#native-preview-button');if(preview)preview.disabled=!nativeTimeReady(d) || !!d.timeSubmitting;
  const overlay=$('#native-overlay-button');if(overlay)overlay.disabled=!nativeTimeReady(d) || !nativePreviewCurrent(d) || !!d.overlaySubmitting;
  const output=$('#native-export-button');if(output)output.disabled=!nativeTimeReady(d) || !nativePreviewCurrent(d);
}
async function nativeValidateTimes(d) {
  const signature=nativeTimeSignature(d);
  if(!d.start.trim() || !d.duration.trim()){
    d.timeValidation=null;nativeUpdateTimeFeedback(d);return null;
  }
  const cached=d.timeCache.get(signature);
  if(cached){d.timeValidation=cached;nativeUpdateTimeFeedback(d);return cached.status==='valid'?cached.parsed:null}
  // Rendering/polling must not create multiple identical validation requests.
  if(d.timeValidation?.signature===signature && d.timeValidation?.promise)return d.timeValidation.promise;
  const validation={signature,status:'pending',parsed:null,error:''};d.timeValidation=validation;
  nativeUpdateTimeFeedback(d);
  validation.promise=(async()=>{
    try{
      const parsed=await api('/api/native-reference/parse-time',{method:'POST',body:{
        start_timecode:d.start.trim(),duration_timecode:d.duration.trim(),timecode_fps:Number(d.fps)}});
      validation.status='valid';validation.parsed=parsed;
    }catch(e){validation.status='error';validation.error=e.message||String(e)}
    delete validation.promise;
    d.timeCache.set(signature,validation);
    if(d.timeCache.size>32)d.timeCache.delete(d.timeCache.keys().next().value);
    // Never allow an old async response to validate a newer edit or FPS choice.
    if(nativeTimeSignature(d)===signature){d.timeValidation=validation;nativeUpdateTimeFeedback(d)}
    return validation.status==='valid'?validation.parsed:null;
  })();
  return validation.promise;
}
function nativeScheduleTimeCheck(d, delay=160) {
  clearTimeout(d.timeTimer);
  const signature=nativeTimeSignature(d), cached=d.timeCache.get(signature);
  if(cached){d.timeValidation=cached;nativeUpdateTimeFeedback(d);return}
  if(d.timeValidation?.signature!==signature)d.timeValidation={signature,status:'pending'};
  nativeUpdateTimeFeedback(d);
  d.timeTimer=setTimeout(()=>nativeValidateTimes(d),delay);
}
function nativeModeButtons() {
  return `<div class="native-mode-bar clean-trim-mode"><span class="status-pill good">Native-reference trim</span><span class="form-help">Calibration now creates the normal C3D-defined clip automatically. Manual timecode entry is recovery-only.</span></div>`;
}
function nativeAutomaticClipState(ctx=nativeWorkflowContext()) {
  const current=session();
  if(!current || !ctx.roleId)return null;
  const rows=Object.values(current.artifacts||{}).filter(artifact=>
    artifact.kind==='native_reference_trim' &&
    artifact.parameters?.window_source==='common_body_c3d_overlap' &&
    String(artifact.parameters?.wearer_role_id||'').toUpperCase()===ctx.roleId &&
    nativeRequestMatchesContext(artifact.parameters,ctx));
  rows.sort((a,b)=>String(a.created_at||'').localeCompare(String(b.created_at||'')));
  const trim=rows.at(-1)||null;
  if(!trim)return null;
  const pairId=trim.parameters?.paired_trim_set_id;
  const pair=Object.values(current.artifacts||{}).filter(artifact=>
    artifact.kind==='aligned_clip_set' && artifact.parameters?.paired_trim_set_id===pairId)
    .sort((a,b)=>String(a.created_at||'').localeCompare(String(b.created_at||''))).at(-1)||null;
  const files=[...(pair?.files||[]),...(trim.files||[])];
  const report=files.find(row=>/c3d_defined_clip_set_report\.html$/i.test(String(row.path||'')))?.path||
    files.find(row=>/trim_report\.html$/i.test(String(row.path||'')))?.path||'';
  return {trim,pair,report,parameters:trim.parameters||{},qc:pair?.qc||trim.qc||{}};
}
function nativeAutomaticClipCard(auto, compact=false, d=nativeDraft()) {
  if(!auto)return '';
  const p=auto.parameters||{}, native=p.native_window||{}, vicon=p.vicon_window||{}, pupil=p.pupil_window||{};
  const start=native.start_media_time||native.start_media_sec||(native.start_media_ns!=null?fmtNs(native.start_media_ns):null)||p.start_timecode||'—';
  const end=native.end_media_time||native.end_media_sec||(native.end_media_ns!=null?fmtNs(native.end_media_ns):null)||'—';
  const duration=Number(vicon.duration_sec??pupil.duration_pupil_sec??auto.qc?.common_duration_sec);
  const startErr=Number(native.start_boundary_error_ms??(native.start_boundary_error_ns!=null?Number(native.start_boundary_error_ns)/1e6:auto.qc?.maximum_start_boundary_error_ms));
  const endErr=Number(native.end_boundary_error_ms??(native.end_boundary_error_ns!=null?Number(native.end_boundary_error_ns)/1e6:auto.qc?.maximum_end_boundary_error_ms));
  const metrics=compact?'':`<div class="native-summary-grid"><div><small>PHYSICAL WINDOW</small><strong>${Number.isFinite(duration)?`${duration.toFixed(3)} s`:'C3D overlap'}</strong></div><div><small>NRD MEDIA</small><strong>${esc(String(start))} → ${esc(String(end))}</strong></div><div><small>START BOUNDARY</small><strong>${Number.isFinite(startErr)?`${startErr.toFixed(3)} ms`:'exact PTS'}</strong></div><div><small>END BOUNDARY</small><strong>${Number.isFinite(endErr)?`${endErr.toFixed(3)} ms`:'exact PTS'}</strong></div></div>`;
  const previewId=nativeAutomaticPreviewId(auto), overlay=nativeAutomaticOverlay(d,auto), pkg=nativeAlignedPackageResult(auto,d);
  const facialSync=nativeSelectedFacialSync(auto,d), facial=nativeAlignedFacialResult(auto,d);
  const reviewAction=compact?'':`<button class="primary-button native-review-auto-preview" ${previewId?'':'disabled'}>${overlay?'Open aligned clip preview':'Review aligned clip video'}</button>`;
  const exportAction=compact?'':`<button class="primary-button native-export-auto-package">${pkg?'Open aligned package':'Export aligned ego + TSD + FM'}</button>`;
  const facialAction=compact?'':`<button class="primary-button native-export-auto-facial" ${facialSync?'':'disabled'}>${facial?'Open aligned facial video':'Export aligned facial video'}</button>`;
  return `<section class="info-card native-auto-trim-ready"><div class="native-overlay-heading"><div><div class="section-label">AUTOMATIC C3D-DEFINED CLIP · ${esc(nativeWorkflowContext().roleId)}</div><h3>Aligned clip ready — no start or end time was typed</h3></div><span class="status-pill good">TIMEGRAPH VERIFIED</span></div>${metrics}<p class="card-sub">The common R01/R02 body-C3D interval was mapped through this role’s accepted affine TimeGraph, then snapped to the original NRD <code>.time</code> / MP4 PTS. The source video was not re-encoded.</p><div class="native-actions-row"><button class="soft-button" data-go="calibration">Review or regenerate in Calibration</button>${auto.report?'<button class="soft-button native-open-auto-report">Open automatic trim report</button>':''}${reviewAction}${exportAction}${facialAction}</div></section>`;
}
function nativeTrimView() {
  const d=nativeDraft(), p=nativePreviewCurrent(d)?d.preview:null, auto=nativeAutomaticClipState();
  const r=p?.report, result=d.result;
  const overlay=d.overlayResult||null, overlayIsCurrent=nativeOverlayCurrent(d), overlayAutomatic=nativeOverlayIsAutomatic(overlay);
  const packageResult=nativeAlignedPackageResult(auto,d), packageArtifact=packageResult?.artifact||session()?.artifacts?.[packageResult?.artifact_id];
  const facialResult=nativeAlignedFacialResult(auto,d), facialArtifact=facialResult?.artifact||session()?.artifacts?.[facialResult?.artifact_id];
  const overlayCard=overlay?`<section class="preview-card native-overlay-preview-card ${overlayIsCurrent?'':'is-stale'}" id="native-overlay-preview-card"><div class="native-overlay-heading"><div><div class="section-label">${overlayAutomatic?'Aligned Clip QC Preview':'Pre-trim Overlay Preview'}</div><h3>Face Mapper + preliminary AOI${overlay.request?.include_gaze?' + gaze':''}</h3></div><span class="status-pill ${overlayIsCurrent?'warning':'muted'}">${overlayIsCurrent?(overlayAutomatic?'TEMPORARY QC CACHE':'TEMPORARY CACHE'):'LAST COMPLETED · SETTINGS CHANGED'}</span></div>
    <div class="real-player"><video id="native-overlay-player" controls preload="metadata" playsinline src="${nativeOverlayVideoUrl(d,overlay)}"></video>
    <div class="real-player-toolbar"><span id="native-overlay-status">Loading temporary preview…</span><span class="player-spacer"></span><button class="soft-button compact" id="native-open-overlay-folder">Open preview folder</button></div></div>
    <div class="native-overlay-metrics"><span><small>SCOPE</small><strong>${esc(overlay.request.scope)}</strong></span><span><small>SEGMENTS</small><strong>${overlay.segments.length}</strong></span><span><small>OUTPUT</small><strong>${Number(overlay.output_duration_sec||0).toFixed(1)} s</strong></span><span><small>FM ROWS</small><strong>${Number(overlay.face_mapper_diagnostics.accepted_face_rows||0).toLocaleString()}</strong></span><span><small>GAZE MAPPED</small><strong>${Number(overlay.gaze_diagnostics.mapped||0).toLocaleString()}</strong></span></div>
    ${overlayIsCurrent?'':'<div class="native-warning">The controls have changed since this video was rendered. You can still play the last completed preview, but render again before relying on the current settings.</div>'}
    <div class="native-warning">${overlayAutomatic?'This temporary QC video uses the already verified C3D-defined interval. It does not modify the aligned clip or create a new scientific artifact.':'This is a temporary cache preview, not a processing artifact.'} Face AOIs are preliminary and include every detected face; participant identity has not been resolved.</div></section>`:'';
  const packageFiles=packageArtifact?.files||[];
  const packageHasVideo=!!(packageResult?.video||packageFiles.some(row=>/\.(mp4|mov|mkv|avi)$/i.test(String(row.path||''))));
  const packageCard=packageResult?`<section class="preview-card native-aligned-package-card" id="native-aligned-package-card"><div class="native-overlay-heading"><div><div class="section-label">FORMAL ALIGNED CLIP PACKAGE</div><h3>Clean ego video + aligned TSD + FaceMapper</h3></div><span class="status-pill good">IMMUTABLE ARTIFACT</span></div>
    ${packageHasVideo?`<div class="real-player"><video id="native-aligned-package-player" controls preload="metadata" playsinline src="${nativeAlignedPackageVideoUrl(packageResult)}"></video><div class="real-player-toolbar"><span id="native-aligned-package-status">Loading clean aligned ego video…</span><span class="player-spacer"></span></div></div>`:'<div class="empty-tree">This package contains aligned data only; no ego MP4 was requested.</div>'}
    <div class="native-overlay-metrics"><span><small>EGO VIDEO</small><strong>${packageHasVideo?'ready':'not included'}</strong></span><span><small>TSD</small><strong>${packageResult.tsd_zip?'ready':'not included'}</strong></span><span><small>FACEMAPPER</small><strong>${packageResult.facemapper_zip?'ready':'not included'}</strong></span><span><small>ROLE</small><strong>${esc(packageResult.wearer_role_id||nativeWorkflowContext().roleId||'—')}</strong></span></div>
    <p class="card-sub">The MP4 contains no QC slate or scientific overlay. Its explicit frame map links every output frame to the original Native frame and PTS. TSD and FaceMapper ZIPs are byte-identical copies of the verified trim outputs.</p>
    <div class="native-actions-row"><button class="primary-button" id="native-open-package-folder">Open aligned package folder</button>${packageResult.report_html?'<button class="soft-button" id="native-open-package-report">Open export report</button>':''}<button class="soft-button" id="native-regenerate-package">Export another revision</button></div></section>`:'';
  const facialFiles=facialArtifact?.files||[];
  const facialHasVideo=!!(facialResult?.video||facialFiles.some(row=>/\.(mp4|mov|mkv|avi)$/i.test(String(row.path||''))));
  const facialCard=facialResult?`<section class="preview-card native-aligned-package-card" id="native-aligned-facial-card"><div class="native-overlay-heading"><div><div class="section-label">FORMAL ALIGNED FACIAL VIDEO</div><h3>Drift-corrected facial-expression video</h3></div><span class="status-pill good">IMMUTABLE ARTIFACT</span></div>
    ${facialHasVideo?`<div class="real-player"><video id="native-aligned-facial-player" controls preload="metadata" playsinline src="${nativeAlignedFacialVideoUrl(facialResult)}"></video><div class="real-player-toolbar"><span id="native-aligned-facial-status">Loading aligned facial-expression video…</span><span class="player-spacer"></span></div></div>`:'<div class="empty-tree">The aligned facial video file is unavailable.</div>'}
    <div class="native-overlay-metrics"><span><small>VIDEO</small><strong>${facialHasVideo?'ready':'missing'}</strong></span><span><small>FRAME MAP</small><strong>${facialResult.frame_map?'ready':'missing'}</strong></span><span><small>DRIFT</small><strong>${Number.isFinite(Number(facialResult.video_stats?.clock_drift_ppm))?`${Number(facialResult.video_stats.clock_drift_ppm).toFixed(2)} ppm`:'corrected'}</strong></span><span><small>ROLE</small><strong>${esc(facialResult.wearer_role_id||nativeWorkflowContext().roleId||'—')}</strong></span></div>
    <p class="card-sub">Every output frame is sampled on the verified ego/C3D interaction clock through the accepted shared-audio affine TimeGraph. The clean MP4 contains no QC overlay; the frame map preserves facial-source and Native-ego lineage.</p>
    <div class="native-actions-row"><button class="primary-button" id="native-open-facial-folder">Open facial export folder</button>${facialResult.report_html?'<button class="soft-button" id="native-open-facial-report">Open synchronization report</button>':''}<button class="soft-button" id="native-regenerate-facial">Export another revision</button></div></section>`:'';
  const counts = p ? `<section class="table-card native-preview-result"><h3>Verified selection · ${esc(p.native_start)} → ${esc(p.native_end)}</h3>
    <div class="native-summary-grid"><div><small>RECORDING ID</small><code>${esc(p.recording_id)}</code></div><div><small>RETAINED LENGTH</small><strong>${esc(p.retained_duration)}</strong></div>
    <div><small>GAZE SAMPLES</small><strong>${r.gaze_rows.toLocaleString()}</strong></div><div><small>EXACT ON-FACE LINKS</small><strong>${r.fm_exact_gaze_matches.toLocaleString()} / ${r.gaze_rows.toLocaleString()}</strong></div></div>
    <p class="card-sub">Selected one of ${p.fm_recording_count} Face Mapper recordings automatically. Native source frames in the window: ${r.source_frames_in_window}. These are not verified frame numbers of your externally edited MP4.</p>
    <div class="native-utc"><span>Original UTC interval (ns, end excluded)</span><code>[${esc(p.utc_start_ns)}, ${esc(p.utc_end_ns)})</code></div>
    <div class="native-table-scroll"><table class="asset-table"><thead><tr><th>Table</th><th>Recording rows</th><th>Retained</th><th>Rule</th></tr></thead><tbody>${r.tables.map(t=>`<tr><td>${esc(t.group)} / ${esc(t.member.split('/').pop())}</td><td>${t.selected_recording_rows.toLocaleString()}</td><td><b>${t.output_rows.toLocaleString()}</b></td><td>${esc(t.kind)}</td></tr>`).join('')}</tbody></table></div>
    ${(r.warnings||[]).map(w=>`<div class="native-warning">${esc(w)}</div>`).join('')}
  </section>` : auto ? '' : `<section class="form-card native-empty"><div class="native-flow"><span>NRD video clock</span><b>→</b><span>Start + length</span><b>→</b><span>TSD + Face Mapper</span></div>
    <h3>You already cut your video. Now trim its data.</h3><p>Select your three original data packages, then enter the start read from the original NRD MP4 and the retained duration. Preview checks the actual MP4 frame timestamps, recording IDs and all table row counts.</p>
    <p>No Global Time anchors, calibration, C3D, gaze rays or DensePose are required. No original or edited video is re-encoded.</p>
    <div class="native-warning">Use the original scene-camera MP4 clock, not recording-start time, Cloud-video time, or an inferred frame-count ÷ source-average-FPS estimate. Editor HH:MM:SS:FF is supported using the explicit timecode FPS below. This mode assumes a simple cut without a speed change.</div>
    <button class="soft-button" data-go="data">Import NRD, TSD and Face Mapper in Data</button>
  </section>`;
  return viewShell('Aligned Pupil clip', 'Review automatic C3D-defined clips or recover with a manual cut', `
    ${nativeModeButtons()}
    ${nativeAutomaticClipCard(auto,false,d)}
    <details class="form-card native-manual-recovery" ${auto?'':'open'}><summary><strong>Manual recovery trim</strong> · only when Calibration cannot create the automatic clip</summary>
    <section class="native-cut-controls">
      <div class="native-timecode-toolbar"><div class="field"><label for="native-timecode-fps">Timecode FPS · 时间码帧率</label>
        <select id="native-timecode-fps" data-native-field="fps">${[24,25,30,50,60].map(f=>`<option value="${f}" ${String(f)===d.fps?'selected':''}>${f} fps${f===30?' · default 默认':''}</option>`).join('')}</select></div>
        <div class="form-help">HH:MM:SS:FF = 时:分:秒:帧。选择剪辑软件的时间线帧率（非丢帧）。此设置不改变 NRD 的真实 PTS / .time 索引，也不会转码视频。</div></div>
      <div class="native-input-grid">
        <div class="field"><label>Start in NRD video · 原视频起点</label><input id="native-trim-start" data-native-field="start" value="${esc(d.start)}" placeholder="00:00:49:21 或 00:00:49.700" spellcheck="false" aria-describedby="native-time-interpretation"></div>
        <div class="field"><label>Retained duration · 保留长度</label><input id="native-trim-duration" data-native-field="duration" value="${esc(d.duration)}" placeholder="00:09:24:15 或 00:09:24.500" spellcheck="false" aria-describedby="native-time-interpretation"></div>
      </div>
      <div id="native-time-interpretation" class="native-time-interpretation ${d.timeValidation?.status==='error'?'is-error':''}" role="status" aria-live="polite">${nativeTimeFeedback(d)}</div>
      <div class="native-actions-row"><div class="field"><label>Output label</label><input id="native-trim-label" data-native-field="label" value="${esc(d.label)}"></div>
        <button class="soft-button" id="native-preview-button" ${nativeTimeReady(d)?'':'disabled'}>Preview alignment & counts</button>
        <button class="soft-button" id="native-overlay-button" ${nativePreviewCurrent(d)&&nativeTimeReady(d)?'':'disabled'}>Render FM + AOI preview</button>
        <button class="primary-button" id="native-export-button" ${nativePreviewCurrent(d)&&nativeTimeReady(d)?'':'disabled'}>Export trimmed TSD + FM</button></div>
      <div class="form-help">Accepts HH:MM:SS:FF at the selected FPS, seconds or HH:MM:SS.mmm. Duration is a length, NOT an end time. Source files and your edited video stay unchanged.</div>
    </section></details>
    ${counts}
    ${overlayCard}
    ${packageCard}
    ${facialCard}
    ${d.preview && !p ? '<div class="native-warning">Inputs changed. Run Preview again before export.</div>' : ''}
    ${result ? `<section class="info-card native-export-result"><h3>Data export completed</h3><code>${esc(result.output_dir)}</code><p>TSD_trimmed.zip · FaceMapper_trimmed.zip · native_frame_map.csv · trim_report.html</p>
      <button class="primary-button" id="native-open-output">Open output folder</button><button class="soft-button" id="native-open-report">Open trim report</button></section>` : ''}
    <section class="info-card"><h3>What this workflow produces</h3><p>Gaze, eye states, IMU and on-face samples share one UTC cut. Partially clipped fixation/blink events retain their original endpoints and statistics in explicit columns. Face landmarks follow the retained Native frames, with boundary context flagged separately.</p>
    <p class="card-sub">These are cropped inputs for facial-AOI analysis, not newly computed eye/nose/mouth labels. The existing 3D/body annotation workflow is unchanged.</p></section>`);
}
function nativeTrimProperties() {
  const d=nativeDraft(), n=nativeContextAssets(['pupil_native']), t=nativeContextAssets(['pupil_cloud']), f=nativeContextAssets(['face_mapper']);
  const ctx=nativeWorkflowContext(), auto=nativeAutomaticClipState(ctx);
  const autoPreviewId=nativeAutomaticPreviewId(auto), autoOverlay=nativeAutomaticOverlay(d,auto);
  const overlayReady=auto?!!autoPreviewId:(nativePreviewCurrent(d)&&nativeTimeReady(d));
  const overlayButtonText=d.overlaySubmitting?'Rendering preview…':(auto?(autoOverlay?'Regenerate QC preview':'Generate QC preview'):'Render FM + AOI preview');
  const facialSyncRows=nativeCompatibleFacialSyncArtifacts(auto,d), facialSync=nativeSelectedFacialSync(auto,d), facialResult=nativeAlignedFacialResult(auto,d);
  const facialSyncLabel=row=>{const source=session()?.assets?.[row.parameters?.source_asset_id];const metrics=row.qc||{};const drift=Number(metrics.estimated_drift_ppm);return `${source?.label||row.label} · ${Number.isFinite(drift)?drift.toFixed(1)+' ppm':'verified'} · visual verified`;};
  return `${auto?`<section class="property-section"><div class="section-label">Automatic aligned clip</div>${nativeAutomaticClipCard(auto,true,d)}</section>`:''}<section class="property-section native-trim-form"><div class="section-label">${auto?'Manual recovery inputs':'1 · Original data packages'}</div><div class="property-stack">
    <div class="form-help native-context-lock"><strong>Active trim context:</strong> ${esc(ctx.interactionId||'Unscoped')} / ${esc(ctx.roleId||'Unscoped')}. Source selections and unsaved cut values are stored separately for each role.</div>
    <div class="field"><label>Native Recording (NRD)</label><select id="native-trim-nrd" data-native-field="native">${optionRows(n,d.native,'Select original NRD')}</select></div>
    <div class="field"><label>Scene MP4 part</label><select id="native-trim-part" data-native-field="part"><option value="">${d.native?'Auto: only when exactly one part':'Select NRD first'}</option>${d.options.map(o=>`<option value="${esc(o.video_member)}" ${d.part===o.video_member?'selected':''}>${esc(o.video_member.split('/').pop())}</option>`).join('')}</select></div>
    <div class="field"><label>Timeseries Data (TSD)</label><select id="native-trim-tsd" data-native-field="tsd">${optionRows(t,d.tsd,'Select TSD')}</select></div>
    <div class="field"><label>Face Mapper (FM)</label><select id="native-trim-fm" data-native-field="fm">${optionRows(f,d.fm,'Select Face Mapper')}</select></div>
  </div></section>
  <section class="property-section"><div class="section-label">2 · ${auto?'Aligned clip QC preview':'Pre-trim overlay preview'}</div><div class="property-stack">
    <div class="form-help">${auto?'Review the already verified C3D-defined interval with FaceMapper geometry, preliminary facial AOIs and gaze. The preview is temporary and does not modify the aligned clip.':'Render a temporary video before publishing the manual recovery trim. It reads the candidate interval directly from NRD/TSD/FM and never registers an immutable result.'}</div>
    <div class="field"><label>Preview scope</label><select data-native-overlay-field="overlayScope" id="native-overlay-scope"><option value="representative" ${d.overlayScope==='representative'?'selected':''}>Representative · start + middle + end</option><option value="distributed" ${d.overlayScope==='distributed'?'selected':''}>Distributed drift check · evenly spaced</option><option value="boundaries" ${d.overlayScope==='boundaries'?'selected':''}>Boundaries · start + end</option><option value="full" ${d.overlayScope==='full'?'selected':''}>Full candidate interval · slower</option></select></div>
    <div class="field"><label>Segment length (seconds)</label><input data-native-overlay-field="overlaySeconds" id="native-overlay-seconds" type="number" min="1" max="60" step="1" value="${esc(d.overlaySeconds)}" ${d.overlayScope==='full'?'disabled':''}><div class="form-help">${d.overlayScope==='full'?'Ignored in Full mode; the whole retained interval is rendered.':'Length of each sampled window; 1–60 seconds.'}</div></div>
    ${d.overlayScope==='distributed'?`<div class="field"><label>Evenly spaced samples</label><input data-native-overlay-field="overlaySamples" id="native-overlay-samples" type="number" min="3" max="12" step="1" value="${esc(d.overlaySamples)}"><div class="form-help">Use 7–9 samples to inspect gradual drift across a long interaction.</div></div>`:''}
    <div class="field"><label>Preview FPS</label><select data-native-overlay-field="overlayFps" id="native-overlay-fps">${[15,24,25,30,50,60].map(f=>`<option value="${f}" ${String(f)===d.overlayFps?'selected':''}>${f} fps</option>`).join('')}</select></div>
    <label class="toggle-row"><span>Face boxes</span><button class="toggle ${d.overlayFaceBoxes?'on':''}" data-native-overlay-toggle="overlayFaceBoxes" aria-pressed="${d.overlayFaceBoxes}"></button></label>
    <label class="toggle-row"><span>Face landmarks</span><button class="toggle ${d.overlayLandmarks?'on':''}" data-native-overlay-toggle="overlayLandmarks" aria-pressed="${d.overlayLandmarks}"></button></label>
    <label class="toggle-row"><span>Preliminary face AOIs</span><button class="toggle ${d.overlayAoi?'on':''}" data-native-overlay-toggle="overlayAoi" aria-pressed="${d.overlayAoi}"></button></label>
    <label class="toggle-row"><span>Gaze points</span><button class="toggle ${d.overlayGaze?'on':''}" data-native-overlay-toggle="overlayGaze" aria-pressed="${d.overlayGaze}"></button></label>
    <div class="field"><label>Left/right convention</label><select data-native-overlay-field="overlayLeftRight" id="native-overlay-left-right"><option value="anatomical_auto" ${(!d.overlayLeftRight||d.overlayLeftRight==='anatomical_auto')?'selected':''}>Anatomical · person's own left/right</option><option value="as_exported" ${d.overlayLeftRight==='as_exported'?'selected':''}>Use FaceMapper labels as exported · legacy</option><option value="swap_eyes" ${d.overlayLeftRight==='swap_eyes'?'selected':''}>Force swap exported bilateral labels</option></select></div>
    <div class="form-help">Representative is fastest. Distributed drift check samples 3–12 evenly spaced windows. Full renders the entire interval and ignores segment length.</div>
    <button class="primary-button full" id="native-properties-overlay-button" ${overlayReady&&!d.overlaySubmitting?'':'disabled'}>${overlayButtonText}</button>
    ${autoOverlay?'<button class="soft-button full" id="native-properties-open-overlay">Open preview</button>':''}
    ${auto&&!autoPreviewId?'<div class="native-warning">The automatic clip plan cache is unavailable. Return to Calibration and regenerate the aligned clips before rendering a QC preview.</div>':''}
  </div></section>
  <section class="property-section"><div class="section-label">3 · Export aligned ego + data</div><div class="property-stack">
    <div class="form-help">Create a formal, clean ego MP4 from the verified interval and package it with the already trimmed TSD and FaceMapper outputs. No overlay or QC slate is added.</div>
    <div class="field"><label>Output label</label><input data-native-package-field="packageLabel" id="native-package-label" value="${esc(d.packageLabel)}"></div>
    <div class="field"><label>Aligned ego-video FPS</label><select data-native-package-field="packageFps" id="native-package-fps">${[24,25,30,50,60].map(f=>`<option value="${f}" ${String(f)===d.packageFps?'selected':''}>${f} fps</option>`).join('')}</select><div class="form-help">Every CFR output frame is linked to its original Native frame/PTS in aligned_ego_frame_map.csv.</div></div>
    <label class="check-row"><input type="checkbox" data-native-package-check="packageVideo" ${d.packageVideo?'checked':''}><span>Export clean aligned ego video</span></label>
    <label class="check-row"><input type="checkbox" data-native-package-check="packageTsd" ${d.packageTsd?'checked':''}><span>Include aligned TSD ZIP</span></label>
    <label class="check-row"><input type="checkbox" data-native-package-check="packageFm" ${d.packageFm?'checked':''}><span>Include aligned FaceMapper ZIP</span></label>
    <label class="check-row"><input type="checkbox" data-native-package-check="packageAudio" ${d.packageAudio?'checked':''} ${d.packageVideo?'':'disabled'}><span>Preserve source ego-video audio</span></label>
    <button class="primary-button full" id="native-export-aligned-package" ${nativeTrimArtifactForPackage(auto,d)&&!d.packageSubmitting?'':'disabled'}>${d.packageSubmitting?'Exporting aligned package…':'Export aligned ego + TSD + FM'}</button>
    ${nativeAlignedPackageResult(auto,d)?'<button class="soft-button full" id="native-properties-open-package">Open latest aligned package</button>':''}
    <div class="form-help">The existing Native-reference trim remains unchanged. Every export creates a new immutable revision.</div>
  </div></section>
  <section class="property-section"><div class="section-label">4 · Export aligned facial video</div><div class="property-stack">
    <div class="form-help">Use the accepted shared-audio facial↔ego TimeGraph and the same verified C3D-defined interval. Video frames and facial audio are drift-corrected onto the ego/reference clock; no synchronization or boundary is re-estimated.</div>
    <div class="field"><label>Accepted shared-audio synchronization</label><select data-native-facial-field="facialSync" id="native-facial-sync">${optionRows(facialSyncRows,d.facialSync,'Select a visual-verified FV → ego mapping',facialSyncLabel)}</select></div>
    <div class="field"><label>Output label</label><input data-native-facial-field="facialLabel" id="native-facial-label" value="${esc(d.facialLabel)}"></div>
    <div class="field"><label>Aligned facial-video FPS</label><select data-native-facial-field="facialFps" id="native-facial-fps">${[24,25,30,50,60].map(f=>`<option value="${f}" ${String(f)===d.facialFps?'selected':''}>${f} fps</option>`).join('')}</select><div class="form-help">aligned_facial_frame_map.csv links every output frame to interaction time, Native ego frame/PTS and the original facial source frame/PTS.</div></div>
    <label class="check-row"><input type="checkbox" data-native-facial-check="facialAudio" ${d.facialAudio?'checked':''}><span>Preserve and drift-correct facial-video audio</span></label>
    <button class="primary-button full" id="native-export-aligned-facial" ${nativeTrimArtifactForPackage(auto,d)&&facialSync&&!d.facialSubmitting?'':'disabled'}>${d.facialSubmitting?'Exporting aligned facial video…':'Export aligned facial video'}</button>
    ${facialResult?'<button class="soft-button full" id="native-properties-open-facial">Open latest aligned facial video</button>':''}
    ${facialSyncRows.length?'':'<div class="native-warning">No compatible visual-verified shared-audio mapping exists for this role and Native ego recording. Complete Calibration → Shared-audio facial ↔ ego sync first.</div>'}
    <div class="form-help">The original FV and ego recordings remain unchanged. Each export creates a new immutable artifact linked to both the accepted Time Sync and Native-reference Trim.</div>
  </div></section>
  <section class="property-section"><div class="section-label">5 · Timing & export rules</div><div class="property-stack">
    <div class="form-help">Enter the two cut values in the central panel. Do not use recording-start time or Cloud-video time. No Global Time anchors are needed.</div>
    <details><summary>Advanced options</summary><div class="property-stack" style="margin-top:10px">
      <div class="field"><label>Events at the cut boundary</label><select data-native-field="policy" id="native-trim-policy"><option value="clip" ${d.policy==='clip'?'selected':''}>Clip overlaps; keep original statistics</option><option value="inside" ${d.policy==='inside'?'selected':''}>Keep only fully contained events</option></select></div>
      <div class="field"><label>Face-frame tolerance (ms)</label><input data-native-field="tolerance" id="native-trim-tolerance" type="number" min="0.1" max="15" step="0.1" value="${esc(d.tolerance)}"></div>
      <div class="field"><label>ffprobe.exe path (blank = bundled / saved / PATH)</label><div class="inline-input"><input data-native-field="ffprobe" id="native-trim-ffprobe" value="${esc(d.ffprobe)}"><button class="input-action" id="native-browse-ffprobe">…</button></div></div>
    </div></details>
    <div class="form-help">The source files stay unchanged. Export creates a new result folder every time.</div>
  </div></section>${mediaToolsPanel()}`;
}
function nativeBindActions() {
  nativeDraft().mode='native';
  const bind=(id, action)=>{const el=$(id);if(el)el.onclick=action};
  bind('#native-mode-button',()=>{nativeDraft().mode='native';render()});
  bind('#global-mode-button',()=>{nativeDraft().mode='global';render()});
  bind('#native-open-output',()=>openLocalPath(nativeDraft().result.output_dir));
  bind('#native-open-report',()=>openLocalPath(nativeDraft().result.report_html));
  bind('#native-open-overlay-folder',()=>openLocalPath(nativeDraft().overlayResult.output_dir));
  bind('#native-open-package-folder',()=>{const pkg=nativeAlignedPackageResult();if(pkg?.output_dir)openLocalPath(pkg.output_dir)});
  bind('#native-open-package-report',()=>{const pkg=nativeAlignedPackageResult();if(pkg?.report_html)openLocalPath(pkg.report_html)});
  bind('#native-properties-open-package',()=>nativeOpenAlignedPackage());
  bind('#native-regenerate-package',()=>submitNativeAlignedPackage());
  bind('#native-export-aligned-package',()=>submitNativeAlignedPackage());
  bind('#native-open-facial-folder',()=>{const result=nativeAlignedFacialResult();if(result?.output_dir)openLocalPath(result.output_dir)});
  bind('#native-open-facial-report',()=>{const result=nativeAlignedFacialResult();if(result?.report_html)openLocalPath(result.report_html)});
  bind('#native-properties-open-facial',()=>nativeOpenAlignedFacial());
  bind('#native-regenerate-facial',()=>submitNativeAlignedFacial());
  bind('#native-export-aligned-facial',()=>submitNativeAlignedFacial());
  bind('#native-properties-overlay-button',()=>nativeAutomaticClipState()?submitNativeAlignedOverlayPreview():submitNativeOverlayPreview());
  bind('#native-properties-open-overlay',()=>nativeOpenOverlayPreview(true));
  $$('.native-review-auto-preview').forEach(button=>button.onclick=()=>nativeOpenOverlayPreview(true));
  $$('.native-export-auto-package').forEach(button=>button.onclick=()=>nativeAlignedPackageResult()?nativeOpenAlignedPackage():submitNativeAlignedPackage());
  $$('.native-export-auto-facial').forEach(button=>button.onclick=()=>nativeAlignedFacialResult()?nativeOpenAlignedFacial():submitNativeAlignedFacial());
  $$('.native-open-auto-report').forEach(button=>button.onclick=()=>{const auto=nativeAutomaticClipState();if(auto?.report)openLocalPath(auto.report)});
  $$('[data-native-field]').forEach(el=>{
    const event = el.tagName==='SELECT'?'onchange':'oninput';
    el[event]=async()=>{
      const d=nativeDraft();d[el.dataset.nativeField]=el.value;
      if (el.dataset.nativeField==='native') {
        d.options=[];d.part='';
        if(d.native) {
          try {const sid=selectedSessionId(), id=d.native;
            const payload=await api(`/api/native-reference/options/${encodeURIComponent(sid)}/${encodeURIComponent(id)}`);
            if(d.native===id){d.options=payload.options; if(d.options.length===1)d.part=d.options[0].video_member;}
          }catch(e){showError(e,'Could not inspect Native video parts')}
        }
        renderProperties();
      }
      nativeScheduleTimeCheck(d);
      const overlayButton=$('#native-overlay-button');if(overlayButton)overlayButton.disabled=!nativePreviewCurrent(d)||!nativeTimeReady(d);
      const exportButton=$('#native-export-button');if(exportButton)exportButton.disabled=!nativePreviewCurrent(d)||!nativeTimeReady(d);
      // Preserve the active central time input and caret while typing.
      const previewCard=$('.native-preview-result');
      if(previewCard)previewCard.style.display=nativePreviewCurrent(d)?'':'none';
      if(el.tagName==='SELECT')renderWorkspace();
    };
  });
  bind('#native-browse-ffprobe',async()=>{
    const path=await choosePath('file','Select ffprobe.exe');if(path){nativeDraft().ffprobe=path;renderProperties();renderWorkspace()}
  });
  $$('[data-native-overlay-field]').forEach(el=>{el.onchange=el.oninput=()=>{const d=nativeDraft();d[el.dataset.nativeOverlayField]=el.value;d.overlayResult=d.overlayResult;renderWorkspace();renderProperties();};});
  $$('[data-native-overlay-toggle]').forEach(el=>{el.onclick=()=>{const d=nativeDraft();const key=el.dataset.nativeOverlayToggle;d[key]=!d[key];renderWorkspace();renderProperties();};});
  $$('[data-native-package-field]').forEach(el=>{el.onchange=el.oninput=()=>{const d=nativeDraft();d[el.dataset.nativePackageField]=el.value;};});
  $$('[data-native-package-check]').forEach(el=>{el.onchange=()=>{const d=nativeDraft();d[el.dataset.nativePackageCheck]=!!el.checked;if(el.dataset.nativePackageCheck==='packageVideo')renderProperties();};});
  $$('[data-native-facial-field]').forEach(el=>{el.onchange=el.oninput=()=>{const d=nativeDraft();d[el.dataset.nativeFacialField]=el.value;if(el.dataset.nativeFacialField==='facialSync'){d.facialResult=null;renderWorkspace();renderProperties();}};});
  $$('[data-native-facial-check]').forEach(el=>{el.onchange=()=>{nativeDraft()[el.dataset.nativeFacialCheck]=!!el.checked;};});
  const packageVideo=$('#native-aligned-package-player'), packageStatus=$('#native-aligned-package-status');
  if(packageVideo){packageVideo.onloadedmetadata=()=>{if(packageStatus)packageStatus.textContent=`${packageVideo.videoWidth}×${packageVideo.videoHeight} · ${packageVideo.duration.toFixed(3)} s · formal aligned artifact`;};packageVideo.onerror=()=>{if(packageStatus)packageStatus.textContent='Could not stream the aligned ego video. Open the package folder to inspect it.';};}
  const facialVideo=$('#native-aligned-facial-player'), facialStatus=$('#native-aligned-facial-status');
  if(facialVideo){facialVideo.onloadedmetadata=()=>{if(facialStatus)facialStatus.textContent=`${facialVideo.videoWidth}×${facialVideo.videoHeight} · ${facialVideo.duration.toFixed(3)} s · drift-corrected formal artifact`;};facialVideo.onerror=()=>{if(facialStatus)facialStatus.textContent='Could not stream the aligned facial video. Open the export folder to inspect it.';};}
  const overlayVideo=$('#native-overlay-player'), overlayStatus=$('#native-overlay-status');
  if(overlayVideo){overlayVideo.onloadedmetadata=()=>{if(overlayStatus)overlayStatus.textContent=`${overlayVideo.videoWidth}×${overlayVideo.videoHeight} · ${overlayVideo.duration.toFixed(2)} s · temporary cache`;};overlayVideo.onerror=()=>{if(overlayStatus)overlayStatus.textContent='Could not stream this temporary preview; render it again.';};}
  bindMediaToolButtons();
  bind('#native-preview-button',submitNativePreview);
  bind('#native-overlay-button',submitNativeOverlayPreview);
  bind('#native-export-button',submitNativeTrim);
  nativeScheduleTimeCheck(nativeDraft());
}
async function submitNativePreview() {
  const d=nativeDraft();
  try {
    if(!d.native||!d.tsd||!d.fm)throw new Error('Select NRD, TSD and Face Mapper first.');
    if(!d.start.trim()||!d.duration.trim())throw new Error('Enter the NRD video start and retained duration.');
    clearTimeout(d.timeTimer);
    const inputSignature=nativeTimeSignature(d);
    const interpreted=await nativeValidateTimes(d);
    if(inputSignature!==nativeTimeSignature(d))throw new Error('Time input changed during validation; check it and preview again.');
    if(!interpreted)throw new Error(d.timeValidation?.error||'Check the timecode input.');
    const signature=nativeSignature(d);
    d.timeSubmitting=true;nativeUpdateTimeFeedback(d);
    const tools=await api('/api/media-tools/check',{method:'POST',body:{mode:'probe',ffprobe_path:d.ffprobe.trim()||null}});
    if(!tools.ready)throw new Error(tools.error+' Open Video tools below or run REPAIR_FFPROBE_WINDOWS.bat.');
    if(signature!==nativeSignature(d))throw new Error('Inputs changed during tool check. Preview again.');
    const payload=await api('/api/jobs/native-trim-preview',{method:'POST',body:{
      session_id:selectedSessionId(), native_asset_id:d.native, timeseries_asset_id:d.tsd,
      facemapper_asset_id:d.fm, start_timecode:d.start.trim(), duration_timecode:d.duration.trim(), timecode_fps:Number(d.fps),
      label:d.label, native_video_member:d.part||null, interval_policy:d.policy,
      face_tolerance_ms:Number(d.tolerance), ffprobe_path:d.ffprobe.trim()||null}});
    d.previewJob=payload.job.id;d.submittedSignature=signature;d.preview=null;
    toast('Native alignment preview queued','This reads frame timestamps and table rows; it does not export a video.');
    await refreshState({quiet:false});
  }catch(e){showError(e,'Could not start Native-reference preview')}
  finally{d.timeSubmitting=false;nativeUpdateTimeFeedback(d)}
}
function nativeOpenOverlayPreview(automaticOnly=false) {
  const d=nativeDraft(), auto=nativeAutomaticClipState();
  const overlay=automaticOnly?nativeAutomaticOverlay(d,auto):d.overlayResult;
  if(!overlay){
    if(automaticOnly)return submitNativeAlignedOverlayPreview();
    return submitNativeOverlayPreview();
  }
  renderWorkspace();
  setTimeout(()=>{
    const card=$('#native-overlay-preview-card'), video=$('#native-overlay-player');
    if(card)card.scrollIntoView({behavior:'smooth',block:'start'});
    if(video){video.load();video.play().catch(()=>{});}
  },80);
}
async function submitNativeOverlayPreviewFor(previewId, automatic=false) {
  const d=nativeDraft();
  try {
    if(!previewId)throw new Error(automatic?'The automatic aligned-clip plan is unavailable. Regenerate it in Calibration.':'Run a fresh alignment preview after changing the cut.');
    if(!automatic&&(!nativePreviewCurrent(d)||!nativeTimeReady(d)))throw new Error('Run a fresh alignment preview after changing the cut.');
    const signature=nativeOverlaySignature(d,previewId,automatic);
    d.overlaySubmitting=true;renderProperties();nativeUpdateTimeFeedback(d);
    const tools=await api('/api/media-tools/check',{method:'POST',body:{mode:'encoder',preserve_audio:false}});
    if(!tools.ready)throw new Error(tools.error+' Configure FFmpeg in Video tools.');
    if(signature!==nativeOverlaySignature(d,previewId,automatic))throw new Error('Overlay settings changed during tool check. Render again.');
    const endpoint=automatic?'/api/jobs/aligned-clip-qc-preview':'/api/jobs/native-trim-overlay-preview';
    const payload=await api(endpoint,{method:'POST',body:{
      session_id:selectedSessionId(),preview_id:previewId,scope:d.overlayScope,
      segment_seconds:d.overlayScope==='full'?8:Number(d.overlaySeconds),sample_count:Number(d.overlaySamples||7),output_fps:Number(d.overlayFps),
      include_face_boxes:!!d.overlayFaceBoxes,include_landmarks:!!d.overlayLandmarks,
      include_aoi:!!d.overlayAoi,include_gaze:!!d.overlayGaze,
      left_right_policy:d.overlayLeftRight,face_tolerance_ms:8,gaze_frame_tolerance_ms:20,
      ffmpeg_path:null,ffprobe_path:null}});
    d.overlayJob=payload.job.id;d.overlaySubmittedSignature=signature;d.overlayResult=null;
    toast(automatic?'Aligned clip QC preview queued':'Pre-trim overlay queued',automatic?'The verified C3D-defined interval is being rendered; the aligned clip stays unchanged.':'This temporary cache preview does not export TSD or Face Mapper data.');
    await refreshState({quiet:false});
  }catch(e){showError(e,automatic?'Could not render aligned clip QC preview':'Could not render pre-trim overlay preview')}
  finally{d.overlaySubmitting=false;nativeUpdateTimeFeedback(d);if(appState.page==='clips')renderProperties();}
}
async function submitNativeOverlayPreview() {
  const d=nativeDraft();
  return submitNativeOverlayPreviewFor(d.preview?.preview_id||'',false);
}
async function submitNativeAlignedOverlayPreview() {
  const auto=nativeAutomaticClipState();
  return submitNativeOverlayPreviewFor(nativeAutomaticPreviewId(auto),true);
}

function nativeOpenAlignedPackage() {
  const pkg=nativeAlignedPackageResult();
  if(!pkg)return submitNativeAlignedPackage();
  renderWorkspace();
  setTimeout(()=>{const card=$('#native-aligned-package-card'),video=$('#native-aligned-package-player');if(card)card.scrollIntoView({behavior:'smooth',block:'start'});if(video){video.load();video.play().catch(()=>{});}},80);
}
async function submitNativeAlignedPackage() {
  const d=nativeDraft(),auto=nativeAutomaticClipState(),trim=nativeTrimArtifactForPackage(auto,d);
  try {
    if(!trim)throw new Error('Create or select a verified aligned trim before exporting the package.');
    if(!d.packageVideo&&!d.packageTsd&&!d.packageFm)throw new Error('Select at least one aligned output.');
    if(!String(d.packageLabel||'').trim())throw new Error('Enter an output label.');
    d.packageSubmitting=true;renderProperties();
    if(d.packageVideo){
      const tools=await api('/api/media-tools/check',{method:'POST',body:{mode:'encoder',preserve_audio:!!d.packageAudio,ffprobe_path:d.ffprobe.trim()||null}});
      if(!tools.ready)throw new Error(tools.error+' Configure FFmpeg in Video tools.');
    }
    const payload=await api('/api/jobs/aligned-clip-export',{method:'POST',body:{
      session_id:selectedSessionId(),native_trim_artifact_id:trim.id,label:String(d.packageLabel).trim(),
      include_ego_video:!!d.packageVideo,include_tsd:!!d.packageTsd,include_facemapper:!!d.packageFm,
      output_fps:Number(d.packageFps),preserve_audio:!!d.packageAudio,ffmpeg_path:null,ffprobe_path:d.ffprobe.trim()||null}});
    d.packageJob=payload.job.id;d.packageResult=null;
    toast('Aligned package export queued','The clean ego MP4 and verified TSD/FM subsets will be published as one immutable artifact.');
    await refreshState({quiet:false});
  }catch(e){showError(e,'Could not export aligned ego + TSD + FaceMapper')}
  finally{d.packageSubmitting=false;if(appState.page==='clips')renderProperties();}
}

function nativeOpenAlignedFacial() {
  const result=nativeAlignedFacialResult();
  if(!result)return submitNativeAlignedFacial();
  renderWorkspace();
  setTimeout(()=>{const card=$('#native-aligned-facial-card'),video=$('#native-aligned-facial-player');if(card)card.scrollIntoView({behavior:'smooth',block:'start'});if(video){video.load();video.play().catch(()=>{});}},80);
}
async function submitNativeAlignedFacial() {
  const d=nativeDraft(),auto=nativeAutomaticClipState(),trim=nativeTrimArtifactForPackage(auto,d),sync=nativeSelectedFacialSync(auto,d);
  try {
    if(!trim)throw new Error('Create or select a verified aligned trim before exporting the facial video.');
    if(!sync)throw new Error('Select a visual-verified shared-audio facial↔ego synchronization for this role.');
    if(!String(d.facialLabel||'').trim())throw new Error('Enter an output label.');
    d.facialSubmitting=true;renderProperties();
    const tools=await api('/api/media-tools/check',{method:'POST',body:{mode:'encoder',preserve_audio:!!d.facialAudio,ffprobe_path:d.ffprobe.trim()||null}});
    if(!tools.ready)throw new Error(tools.error+' Configure FFmpeg in Video tools.');
    const payload=await api('/api/jobs/aligned-facial-video-export',{method:'POST',body:{
      session_id:selectedSessionId(),native_trim_artifact_id:trim.id,time_sync_artifact_id:sync.id,
      label:String(d.facialLabel).trim(),output_fps:Number(d.facialFps),preserve_audio:!!d.facialAudio,
      crf:18,preset:'fast',ffmpeg_path:null,ffprobe_path:d.ffprobe.trim()||null}});
    d.facialJob=payload.job.id;d.facialResult=null;
    toast('Aligned facial-video export queued','The accepted audio TimeGraph and verified C3D interval will be applied without re-estimating either one.');
    await refreshState({quiet:false});
  }catch(e){showError(e,'Could not export aligned facial-expression video')}
  finally{d.facialSubmitting=false;if(appState.page==='clips')renderProperties();}
}

async function submitNativeTrim() {
  const d=nativeDraft();
  try {
    if(!nativePreviewCurrent(d)||!nativeTimeReady(d))throw new Error('Run a fresh preview after changing inputs.');
    const payload=await api('/api/jobs/native-trim',{method:'POST',body:{session_id:selectedSessionId(),preview_id:d.preview.preview_id}});
    d.exportJob=payload.job.id;
    toast('TSD + Face Mapper export queued','Native and edited videos will remain unchanged.');
    await refreshState({quiet:false});
  }catch(e){showError(e,'Could not export Native-reference data')}
}
function updateNativeTrimJobs() {
  if(!project()||!session())return;
  const d=nativeDraft(); let update=false;
  const jp=project().jobs?.[d.previewJob];
  if(jp && ['completed','failed','cancelled'].includes(jp.status) && d.lastHandledPreview!==jp.id){
    d.lastHandledPreview=jp.id;
    if(jp.status==='completed'){d.preview=jp.result;toast('Alignment preview ready','Review the interval and counts, then export.');}
    update=true;
  }
  const jo=project().jobs?.[d.overlayJob];
  if(jo && ['completed','failed','cancelled'].includes(jo.status) && d.lastHandledOverlay!==jo.id){
    d.lastHandledOverlay=jo.id;
    if(jo.status==='completed'){
      const metadata=session()?.metadata||{};
      const jobPreviewId=String(jo.result?.preview_id||'');
      d.overlayResult=jo.result || metadata.native_reference_overlay_by_preview?.[jobPreviewId]?.result || metadata.native_reference_overlay_last?.result || null;
      if(d.overlayResult){
        const aligned=nativeOverlayIsAutomatic(d.overlayResult);
        toast(aligned?'Aligned clip QC preview ready':'Pre-trim overlay ready',aligned?'The player is shown below the verified aligned clip.':'The player is shown below the verified selection.');
        setTimeout(()=>{const card=$('#native-overlay-preview-card'), video=$('#native-overlay-player');if(card)card.scrollIntoView({behavior:'smooth',block:'start'});if(video)video.load();},80);
      }else showError(new Error('The job completed but returned no preview metadata. Reload project status or render again.'),'Preview player could not be restored');
    }
    update=true;
  }
  const je=project().jobs?.[d.exportJob];
  if(je && ['completed','failed','cancelled'].includes(je.status) && d.lastHandledExport!==je.id){
    d.lastHandledExport=je.id;
    if(je.status==='completed'){d.result=je.result;toast('Trim complete','Both cropped packages and a report are ready.');}
    update=true;
  }
  const jk=project().jobs?.[d.packageJob];
  if(jk && ['completed','failed','cancelled'].includes(jk.status) && d.lastHandledPackage!==jk.id){
    d.lastHandledPackage=jk.id;
    if(jk.status==='completed'){d.packageResult=jk.result;toast('Aligned package ready','Clean ego video, TSD and FaceMapper outputs are ready.');setTimeout(()=>nativeOpenAlignedPackage(),100);}
    update=true;
  }
  const jf=project().jobs?.[d.facialJob];
  if(jf && ['completed','failed','cancelled'].includes(jf.status) && d.lastHandledFacial!==jf.id){
    d.lastHandledFacial=jf.id;
    if(jf.status==='completed'){d.facialResult=jf.result;toast('Aligned facial video ready','The clean drift-corrected FV and frame-level lineage are ready.');setTimeout(()=>nativeOpenAlignedFacial(),100);}
    update=true;
  }
  if(update && appState.page==='clips' && d.mode==='native'){renderWorkspace();renderProperties();}
}
