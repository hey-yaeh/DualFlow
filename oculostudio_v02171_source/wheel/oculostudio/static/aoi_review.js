/* Interactive AOI Review Player.
 * Loads canonical artifact tables through the local CleanCore API and draws
 * review-only overlays over the trimmed Privacy MP4 or, by explicit local
 * request, the frame-locked original Native video. Exports remain privacy-safe,
 * and the player never edits the scientific CSV/Parquet source artifacts.
 */

window.OculoReview = (() => {
  const state = {
    sessionId: '',
    options: null,
    manifest: null,
    mode: 'face',
    artifactId: '',
    videoArtifactId: '',
    videoSourceMode: 'privacy',
    sourceSwitchSerial: 0,
    sourceSwitchPending: false,
    currentFrame: -1,
    presentedFrame: -1,
    lastPresentedFrame: -1,
    lastPresentedMediaTime: null,
    pauseLockFrame: -1,
    pauseLockSerial: 0,
    pendingExplicitSeekFrame: -1,
    currentPayload: null,
    playbackRequestInFlight: false,
    playbackPromise: null,
    playbackQueuedFrame: null,
    playbackGeneration: 0,
    frameCache: new Map(),
    frameCacheOrder: [],
    playbackFrameCache: new Map(),
    playbackFrameCacheOrder: [],
    playbackBufferedStart: -1,
    playbackBufferedEnd: -1,
    lastOverlayFrame: -1,
    playbackStatus: 'Paused',
    playbackFramesRendered: 0,
    playbackLastFrameAt: 0,
    sampleIndex: 0,
    confidenceThreshold: 0.65,
    bodyView: 'split',
    bodyStableScale: true,
    bodyTimelineHover: -1,
    requestSerial: 0,
    videoFrameCallbackId: null,
    interactionId: '',
    roleId: '',
    nativeTrimArtifactId: '',
    initialMode: '',
    initialArtifactId: '',
    initialVideoArtifactId: '',
    keyboardBound: false,
    fullscreenBound: false,
    resizeBound: false,
    layers: {
      gaze: true,
      trail: true,
      face: true,
      participant: true,
      identity: true,
      other: true,
      body: true,
      final: true,
      confidence: true,
      review: true,
    },
  };

  const modes = {
    face: {label:'Face AOI', icon:'F'},
    body: {label:'Body AOI', icon:'B'},
    fusion: {label:'Fused targets', icon:'⌾'},
  };

  const optionHtml = (rows, selected='', empty='Select…') => `<option value="">${esc(empty)}</option>` + (rows||[]).map(row => {
    const id = row.artifact_id || row.id;
    return `<option value="${esc(id)}" ${id===selected?'selected':''}>${esc(row.label || id)}</option>`;
  }).join('');

  function markup({faceArtifacts=[], bodyArtifacts=[], fusionArtifacts=[], privacyClips=[], reviewVideos=[], initialMode='', initialArtifactId='', initialVideoArtifactId=''}={}) {
    const available = {face:faceArtifacts, body:bodyArtifacts, fusion:fusionArtifacts};
    const requestedMode=Object.keys(available).includes(initialMode)?initialMode:'';
    const artifactMode=Object.entries(available).find(([,rows])=>(rows||[]).some(row=>(row.id||row.artifact_id)===initialArtifactId))?.[0]||'';
    const preferredMode=artifactMode||requestedMode||(fusionArtifacts.length?'fusion':faceArtifacts.length?'face':bodyArtifacts.length?'body':'face');
    const preferred=(available[preferredMode]||[]).find(row=>(row.id||row.artifact_id)===initialArtifactId)||(available[preferredMode]||[]).slice(-1)[0];
    const preferredVideo=(privacyClips||[]).find(row=>(row.id||row.artifact_id)===initialVideoArtifactId)||(privacyClips||[]).slice(-1)[0];
    state.mode=preferredMode;
    state.artifactId=preferred?.id||preferred?.artifact_id||'';
    state.videoArtifactId=preferredVideo?.id||preferredVideo?.artifact_id||'';
    state.initialMode=preferredMode;
    state.initialArtifactId=state.artifactId;
    state.initialVideoArtifactId=state.videoArtifactId;
    return `<section class="aoi-review-shell" id="aoi-review-shell">
      <div class="aoi-review-head">
        <div><div class="eyebrow">Interactive quality control</div><h2>AOI Review Player</h2><p>Review Face, Body and fused targets on the real trimmed Privacy video. Use the local blur toggle only when facial detail is needed for annotation; exports remain privacy-safe. Face and Body AOI decisions are saved as separate mutable drafts; applying either draft creates a new immutable reviewed artifact.</p></div>
        <div class="aoi-review-selectors">
          <label>Mode<select id="aoi-review-mode">${Object.entries(modes).map(([id,row])=>`<option value="${id}" ${id===preferredMode?'selected':''}>${esc(row.label)}</option>`).join('')}</select></label>
          <label>AOI artifact<select id="aoi-review-artifact">${optionHtml((available[preferredMode]||[]).map(a=>({artifact_id:a.id||a.artifact_id,label:a.label})),state.artifactId,'No artifact')}</select></label>
          <label>Privacy clip<select id="aoi-review-video">${optionHtml((privacyClips||[]).map(a=>({artifact_id:a.id||a.artifact_id,label:a.label})),state.videoArtifactId,'No trimmed video')}</select></label>
          <label>Low-confidence threshold<input id="aoi-review-threshold" type="number" min="0" max="1" step="0.05" value="0.65"></label>
        </div>
      </div>
      <div class="aoi-human-review-entry" id="aoi-human-review-entry">
        <div class="aoi-human-review-copy">
          <div class="section-label">Human review · Face AOI</div>
          <strong>Verify face identity and facial region</strong>
          <span id="aoi-human-review-status">${faceArtifacts.length?'Open the Face AOI queue, verify flagged samples, and save corrections.':'Generate a Face AOI artifact before starting face review.'}</span>
          <small>Available from every player mode. Opening this queue switches the player to its Face AOI artifact.</small>
        </div>
        <div class="aoi-human-review-actions">
          <span class="status-pill ${faceArtifacts.length?'warning':'muted'}" id="aoi-human-review-count">${faceArtifacts.length?'Queue available':'No Face AOI'}</span>
          <button class="primary-button" id="aoi-review-open-human-review" type="button" ${faceArtifacts.length?'':'disabled'}>Open Face review</button>
        </div>
      </div>
      <div class="aoi-human-review-entry body-human-review-entry" id="aoi-body-human-review-entry">
        <div class="aoi-human-review-copy">
          <div class="section-label">Human review · Body AOI</div>
          <strong>Verify self/partner and body region</strong>
          <span id="aoi-body-human-review-status">${bodyArtifacts.length?'Open the Body AOI queue, verify red review samples, and save corrections.':'Generate a Body AOI artifact before starting body review.'}</span>
          <small>Supports accept, relation/region correction, no-body exclusion, defer, and immutable reviewed Body AOI output.</small>
        </div>
        <div class="aoi-human-review-actions">
          <span class="status-pill ${bodyArtifacts.length?'warning':'muted'}" id="aoi-body-human-review-count">${bodyArtifacts.length?'Queue available':'No Body AOI'}</span>
          <button class="primary-button" id="aoi-review-open-body-human-review" type="button" ${bodyArtifacts.length?'':'disabled'}>Open Body review</button>
        </div>
      </div>
      <div class="aoi-review-layout">
        <section class="aoi-review-player-card" id="aoi-review-player-card">
          <div class="aoi-review-video-wrap" id="aoi-review-video-wrap">
            <video id="aoi-review-video-element" preload="metadata" playsinline></video>
            <canvas id="aoi-review-overlay"></canvas>
            <div class="aoi-review-empty" id="aoi-review-empty"><span>▶</span><strong>Select a reviewable AOI artifact and compatible Privacy clip.</strong></div>
            <div class="aoi-review-badge" id="aoi-review-badge">AOI Review</div>
            <div class="aoi-review-unblurred-warning" id="aoi-review-unblurred-warning" hidden>UNBLURRED LOCAL REVIEW · EXPORT REMAINS BLURRED</div>
            <button class="aoi-review-fullscreen-exit" id="aoi-review-fullscreen-exit" type="button" aria-label="Exit fullscreen" title="Exit fullscreen">Exit fullscreen</button>
            <div class="aoi-review-world" id="aoi-review-world-wrap">
              <div class="aoi-review-world-head"><div><strong>3D Body evidence</strong><span id="aoi-review-world-subtitle">world-space orthographic views</span></div><div class="aoi-review-world-tabs"><button type="button" data-body-view="split" class="active">Split</button><button type="button" data-body-view="xz">X–Z</button><button type="button" data-body-view="yz">Y–Z</button></div></div>
              <canvas id="aoi-review-world"></canvas>
              <div class="aoi-review-world-legend"><span><i class="partner"></i>partner</span><span><i class="self"></i>self</span><span><i class="hit"></i>gaze hit</span></div>
            </div>
            <div class="body-aoi-video-key" id="body-aoi-video-key" hidden><span><i class="partner"></i>partner target</span><span><i class="self"></i>self target</span><span><i class="review"></i>near-body / review</span></div>
          </div>
          <div class="body-aoi-timeline-panel" id="body-aoi-timeline-panel" hidden>
            <div class="body-aoi-timeline-head"><div><strong>Whole-clip Body AOI timeline</strong><span>colour = self / partner · bar height = body-hit coverage</span></div><div class="body-aoi-timeline-key"><span><i class="partner"></i>partner</span><span><i class="self"></i>self</span><span><i class="review"></i>review</span></div></div>
            <canvas id="body-aoi-timeline" height="72"></canvas>
            <div class="body-aoi-timeline-tooltip" id="body-aoi-timeline-tooltip" hidden></div>
          </div>
          <div class="aoi-review-controls">
            <button class="round-control" id="aoi-review-prev-frame" title="Previous video frame">│◀</button>
            <button class="aoi-review-play-control" id="aoi-review-play" title="Play / pause" aria-label="Play review video">▶ Play</button>
            <button class="round-control" id="aoi-review-next-frame" title="Next video frame">▶│</button>
            <span class="aoi-review-time" id="aoi-review-time">00:00:00.000 / 00:00:00.000</span>
            <span class="aoi-review-playback-status" id="aoi-review-playback-status">Paused · full review frame</span>
            <input id="aoi-review-seek" type="range" min="0" max="1" step="0.001" value="0">
            <button class="aoi-review-blur-toggle" id="aoi-review-blur-toggle" type="button" disabled>Blur on · privacy</button>
            <button class="round-control" id="aoi-review-fullscreen" title="Enter fullscreen" aria-label="Enter fullscreen">⛶</button>
          </div>
          <div class="aoi-review-toolbar">
            <div class="aoi-review-sample-controls">
              <button class="soft-button compact" id="aoi-review-prev-sample">← sample</button><strong id="aoi-review-sample-position">0 / 0</strong><button class="soft-button compact" id="aoi-review-next-sample">sample →</button>
            </div>
            <div class="aoi-review-event-controls">
              <button class="soft-button compact manual-review-nav" data-review-nav="pending_review" data-direction="-1">← pending</button>
              <button class="soft-button compact manual-review-nav" data-review-nav="pending_review" data-direction="1">pending →</button>
              <button class="soft-button compact" data-review-nav="review" data-direction="1">next flagged</button>
              <button class="soft-button compact" data-review-nav="conflict" data-direction="1">next conflict</button>
              <button class="soft-button compact" data-review-nav="low_confidence" data-direction="1">next low confidence</button>
            </div>
          </div>
          <div class="aoi-review-layers">
            ${[['gaze','Gaze samples'],['trail','Gaze trail'],['face','Face AOIs'],['participant','Projected participant heads'],['identity','Face identity'],['other','Other-person faces'],['body','Body 3D evidence'],['final','Target label'],['confidence','Confidence'],['review','Review warnings']].map(([id,label])=>`<label><input type="checkbox" data-review-layer="${id}" checked><span>${label}</span></label>`).join('')}
          </div>
        </section>
        <aside class="aoi-review-inspector">
          <div class="body-aoi-overview" id="body-aoi-overview" hidden></div>
          <div class="aoi-review-summary" id="aoi-review-summary"><div class="empty-tree">Load a review artifact to inspect frame-level and sample-level evidence.</div></div>
          <div class="aoi-review-samples" id="aoi-review-samples"></div>
          <section class="face-manual-review" id="face-manual-review" hidden>
            <div class="face-review-head"><div><div class="section-label">Human review</div><strong id="face-review-progress">No queue loaded</strong></div><span class="status-pill" id="face-review-status">—</span></div>
            <div class="face-review-queue" id="face-review-queue"></div>
            <div class="face-review-auto" id="face-review-auto">Select a gaze sample in a flagged frame.</div>
            <div class="face-review-guidance" id="face-review-guidance"></div>
            <div class="face-review-form">
              <label>Apply decision to<select id="face-review-apply-to"><option value="sample">Selected gaze sample</option><option value="frame">All flagged samples in this frame</option><option value="fixation">All flagged samples in this fixation</option></select><small>Fixation scope can resolve many repeated samples with one decision. Keep Face region automatic when only the person identity needs correction.</small></label>
              <div class="face-review-grid">
                <label>Relation<select id="face-review-relation"><option value="">Keep automatic</option><option value="partner">partner</option><option value="self">self</option><option value="other_person">other_person</option></select></label>
                <label>Face region<select id="face-review-region"><option value="">Keep automatic</option><option value="left_eye">left_eye</option><option value="right_eye">right_eye</option><option value="nose">nose</option><option value="mouth">mouth</option><option value="other_face">other_face</option><option value="face">face</option><option value="no_face">no_face</option></select></label>
              </div>
              <label>Entity ID<input id="face-review-entity" placeholder="Keep automatic / candidate entity"></label>
              <label>Reviewer<input id="face-review-reviewer" value="local reviewer"></label>
              <label>Review note<textarea id="face-review-note" rows="2" placeholder="Optional reason or observation"></textarea></label>
            </div>
            <div class="face-review-actions">
              <button class="primary-button" data-face-review-action="accept_auto">Accept automatic</button>
              <button class="primary-button" data-face-review-action="override">Save correction</button>
              <button class="soft-button" data-face-review-action="exclude">Exclude sample</button>
              <button class="soft-button" data-face-review-action="defer">Review later</button>
              <button class="ghost-button" data-face-review-action="clear">Clear decision</button>
            </div>
            <div class="face-review-finalize">
              <label>Reviewed artifact label<input id="face-review-output-label" value="reviewed_face_aoi"></label>
              <button class="primary-button full" id="face-review-apply">Create reviewed Face AOI</button>
              <p id="face-review-finalize-help">Resolve every flagged sample before creating the reviewed artifact. The source artifact remains unchanged.</p>
            </div>
          </section>
          <section class="body-manual-review" id="body-manual-review" hidden>
            <div class="face-review-head"><div><div class="section-label">Human review · Body AOI</div><strong id="body-review-progress">No queue loaded</strong></div><span class="status-pill" id="body-review-status">—</span></div>
            <div class="body-review-geometry-gate" id="body-review-geometry-gate"></div>
            <div class="face-review-queue" id="body-review-queue"></div>
            <div class="face-review-auto" id="body-review-auto">Select a gaze sample in a flagged Body AOI frame.</div>
            <div class="face-review-guidance" id="body-review-guidance"></div>
            <div class="face-review-form">
              <label>Apply decision to<select id="body-review-apply-to"><option value="review_episode" selected>Entire target-stable review episode · recommended</option><option value="fixation">All flagged samples in this fixation</option><option value="frame">All flagged samples in this frame</option><option value="sample">Selected gaze sample</option></select><small>A review episode safely merges adjacent fixations only when self/partner, entity and coarse body region remain unchanged. Keep fields automatic when the C3D-derived target is correct.</small></label>
              <div class="face-review-grid">
                <label>Relation<select id="body-review-relation"><option value="">Keep automatic</option><option value="partner">partner</option><option value="self">self</option><option value="other">other</option></select></label>
                <label>Body region<select id="body-review-region"><option value="">Keep automatic</option><optgroup label="Coarse regions"><option value="head">head</option><option value="torso">torso</option><option value="left_arm">left_arm</option><option value="right_arm">right_arm</option><option value="left_hand">left_hand</option><option value="right_hand">right_hand</option><option value="left_leg">left_leg</option><option value="right_leg">right_leg</option></optgroup><optgroup label="Detailed regions"><option value="neck">neck</option><option value="pelvis">pelvis</option><option value="left_upper_arm">left_upper_arm</option><option value="left_forearm">left_forearm</option><option value="right_upper_arm">right_upper_arm</option><option value="right_forearm">right_forearm</option><option value="left_thigh">left_thigh</option><option value="left_lower_leg">left_lower_leg</option><option value="left_foot">left_foot</option><option value="right_thigh">right_thigh</option><option value="right_lower_leg">right_lower_leg</option><option value="right_foot">right_foot</option></optgroup><option value="no_body">no_body</option></select></label>
              </div>
              <label>Body entity ID<input id="body-review-entity" placeholder="Auto-resolve from self / partner role"></label>
              <label>Reviewer<input id="body-review-reviewer" value="local reviewer"></label>
              <label>Review note<textarea id="body-review-note" rows="2" placeholder="Optional reason or observation"></textarea></label>
            </div>
            <div class="face-review-actions body-review-actions">
              <button class="primary-button" data-body-review-action="accept_auto">Accept automatic</button>
              <button class="primary-button" data-body-review-action="override">Save correction</button>
              <button class="soft-button" data-body-review-action="exclude">Mark no body</button>
              <button class="soft-button" data-body-review-action="defer">Review later</button>
              <button class="ghost-button" data-body-review-action="clear">Clear decision</button>
            </div>
            <div class="face-review-finalize">
              <label>Reviewed artifact label<input id="body-review-output-label" value="reviewed_body_aoi"></label>
              <button class="primary-button full" id="body-review-apply">Create reviewed Body AOI</button>
              <p id="body-review-finalize-help">Resolve every flagged sample before creating the reviewed artifact. The original automatic Body AOI remains unchanged.</p>
            </div>
          </section>
          <div class="aoi-review-export">
            <label>Export label<input id="aoi-review-export-label" value="aoi_review_video"></label>
            <label class="check-row"><input id="aoi-review-export-audio" type="checkbox" checked><span>Preserve source audio</span></label>
            <button class="primary-button full" id="aoi-review-export">Export pre-rendered QC MP4</button>
            <p>Exported MP4 is for review and sharing. CSV/Parquet artifacts remain the analysis source.</p>
          </div>
          ${reviewVideos.length?`<div class="aoi-review-exports"><div class="section-label">Existing review videos</div>${reviewVideos.slice().reverse().slice(0,5).map(a=>`<div class="map-row"><div><strong>${esc(a.label)}</strong><small>${esc(a.id)}</small></div><span class="status-pill good">MP4</span></div>`).join('')}</div>`:''}
        </aside>
      </div>
    </section>`;
  }

  function fmtTime(seconds) {
    if (!Number.isFinite(seconds)) return '00:00:00.000';
    const totalMs = Math.max(0, Math.round(seconds*1000));
    const ms=totalMs%1000, totalSec=Math.floor(totalMs/1000), sec=totalSec%60,
      totalMin=Math.floor(totalSec/60), min=totalMin%60, hr=Math.floor(totalMin/60);
    return `${String(hr).padStart(2,'0')}:${String(min).padStart(2,'0')}:${String(sec).padStart(2,'0')}.${String(ms).padStart(3,'0')}`;
  }

  function currentRows() {
    return state.options?.modes?.[state.mode] || [];
  }

  function faceReviewRows() {
    return state.options?.modes?.face || [];
  }

  function faceReviewPendingCount(row) {
    const qc=row?.qc||{};
    const values=[qc.review_samples_remaining,qc.manual_review_pending_sample_count,qc.review_required_count];
    for(const value of values){
      if(value===null||value===undefined||value==='')continue;
      const number=Number(value);
      if(Number.isFinite(number)&&number>=0)return number;
    }
    return null;
  }

  function faceReviewNeedsAttention(row) {
    const pending=faceReviewPendingCount(row);
    return pending!==null?pending>0:Number(row?.qc?.review_required_fraction||0)>0;
  }

  function preferredHumanReviewArtifact() {
    const rows=faceReviewRows();
    if(!rows.length)return null;
    if(state.mode==='face'&&state.artifactId){
      const current=rows.find(row=>row.artifact_id===state.artifactId);
      if(current)return current;
    }
    const newest=[...rows].sort((a,b)=>String(b.created_at||'').localeCompare(String(a.created_at||'')));
    return newest.find(faceReviewNeedsAttention)||newest.find(row=>Number(row?.manual_review_draft?.decision_count||0)>0)||newest[0];
  }

  function updateHumanReviewEntry() {
    const entry=document.querySelector('#aoi-human-review-entry');
    const status=document.querySelector('#aoi-human-review-status');
    const count=document.querySelector('#aoi-human-review-count');
    const button=document.querySelector('#aoi-review-open-human-review');
    if(!entry||!status||!count||!button)return;
    const rows=faceReviewRows();
    const row=preferredHumanReviewArtifact();
    const activeManual=state.mode==='face'&&state.manifest&&state.artifactId===row?.artifact_id?state.manifest.manual_review:null;
    entry.classList.toggle('unavailable',!rows.length);
    if(!row){
      status.textContent='Generate a Face AOI artifact before starting human review.';
      count.textContent='No Face AOI';
      count.className='status-pill muted';
      button.disabled=true;
      button.textContent='Open human review';
      button.dataset.artifactId='';
      return;
    }
    const pending=activeManual?Number(activeManual.pending_sample_count||0):faceReviewPendingCount(row);
    const frames=activeManual?Number(activeManual.pending_frame_count||0):Number(row?.qc?.review_frames_remaining||0);
    const decisions=activeManual?Number(activeManual.decision_count||0):Number(row?.manual_review_draft?.decision_count||0);
    const total=activeManual?Number(activeManual.total_flagged_sample_count||0):Number(row?.qc?.review_samples_before_auto_resolution||0);
    const attention=activeManual?pending>0:faceReviewNeedsAttention(row);
    button.disabled=false;
    button.dataset.artifactId=row.artifact_id||'';
    if(attention){
      const pendingText=pending===null?'flagged samples':`${pending} flagged sample${pending===1?'':'s'}`;
      status.textContent=`${row.label||'Face AOI'} · ${pendingText}${frames>0?` in ${frames} frame${frames===1?'':'s'}`:''} still need a human decision.`;
      count.textContent=pending===null?'Review required':`${pending} pending`;
      count.className='status-pill warning';
      button.textContent=decisions>0?'Continue human review':'Review flagged samples';
    }else if(total>0||decisions>0){
      status.textContent=`${row.label||'Face AOI'} · the current human-review queue is complete. You can reopen it to audit saved decisions.`;
      count.textContent='Review complete';
      count.className='status-pill good';
      button.textContent='Open reviewed queue';
    }else{
      status.textContent=`${row.label||'Face AOI'} · no flagged sample is currently pending. Open the player for a manual spot-check.`;
      count.textContent='Spot-check ready';
      count.className='status-pill good';
      button.textContent='Open Face AOI review';
    }
  }

  async function openHumanReview() {
    const row=preferredHumanReviewArtifact();
    if(!row){toast('No Face AOI available','Generate a Face AOI artifact first, then open Human review.');return;}
    const mode=document.querySelector('#aoi-review-mode');
    if(mode)mode.value='face';
    state.mode='face';
    state.artifactId=row.artifact_id||'';
    state.videoArtifactId='';
    populateArtifacts(state.artifactId);
    await loadManifest();
    updateHumanReviewEntry();updateBodyHumanReviewEntry();
    requestAnimationFrame(()=>{
      const panel=document.querySelector('#face-manual-review');
      if(!panel)return;
      panel.hidden=false;
      panel.classList.add('human-review-focus');
      panel.scrollIntoView({behavior:'smooth',block:'start'});
      window.setTimeout(()=>panel.classList.remove('human-review-focus'),1600);
    });
  }

  function bodyReviewRows() {
    return state.options?.modes?.body || [];
  }

  function bodyReviewPendingCount(row) {
    const qc=row?.qc||{};
    const values=[qc.manual_review_pending_sample_count,qc.review_required_count,qc.review_samples_remaining];
    for(const value of values){
      if(value===null||value===undefined||value==='')continue;
      const number=Number(value);
      if(Number.isFinite(number)&&number>=0)return number;
    }
    const rows=Number(qc.output_rows||qc.input_gaze_rows||0);
    const fraction=Number(qc.review_required_fraction);
    if(Number.isFinite(rows)&&rows>=0&&Number.isFinite(fraction)&&fraction>=0)return Math.round(rows*fraction);
    return null;
  }

  function bodyReviewPendingUnitCount(row) {
    const qc=row?.qc||{};
    const values=[qc.manual_review_pending_review_unit_count,qc.review_units_remaining,qc.review_fixations_remaining];
    for(const value of values){
      if(value===null||value===undefined||value==='')continue;
      const number=Number(value);
      if(Number.isFinite(number)&&number>=0)return number;
    }
    return bodyReviewPendingCount(row);
  }

  function bodyReviewNeedsAttention(row) {
    const pending=bodyReviewPendingCount(row);
    return pending!==null?pending>0:Number(row?.qc?.review_required_fraction||0)>0;
  }

  function preferredBodyHumanReviewArtifact() {
    const rows=bodyReviewRows();
    if(!rows.length)return null;
    if(state.mode==='body'&&state.artifactId){
      const current=rows.find(row=>row.artifact_id===state.artifactId);
      if(current)return current;
    }
    const newest=[...rows].sort((a,b)=>String(b.created_at||'').localeCompare(String(a.created_at||'')));
    return newest.find(bodyReviewNeedsAttention)||newest.find(row=>Number(row?.manual_review_draft?.decision_count||0)>0)||newest[0];
  }

  function updateBodyHumanReviewEntry() {
    const entry=document.querySelector('#aoi-body-human-review-entry');
    const status=document.querySelector('#aoi-body-human-review-status');
    const count=document.querySelector('#aoi-body-human-review-count');
    const button=document.querySelector('#aoi-review-open-body-human-review');
    if(!entry||!status||!count||!button)return;
    const rows=bodyReviewRows();
    const row=preferredBodyHumanReviewArtifact();
    const activeManual=state.mode==='body'&&state.manifest&&state.artifactId===row?.artifact_id?state.manifest.manual_review:null;
    const sideButton=document.querySelector('#open-body-human-review-side');
    const sideCount=document.querySelector('#body-human-review-side-count');
    entry.classList.toggle('unavailable',!rows.length);
    if(!row){
      status.textContent='Generate a Body AOI artifact before starting human review.';
      count.textContent='No Body AOI';
      count.className='status-pill muted';
      button.disabled=true;
      button.textContent='Open Body review';
      button.dataset.artifactId='';
      return;
    }
    const pending=activeManual?Number(activeManual.pending_sample_count||0):bodyReviewPendingCount(row);
    const pendingUnits=activeManual?Number(activeManual.pending_review_unit_count??activeManual.pending_frame_count??0):bodyReviewPendingUnitCount(row);
    const frames=activeManual?Number(activeManual.pending_frame_count||0):0;
    const decisions=activeManual?Number(activeManual.decision_count||0):Number(row?.manual_review_draft?.decision_count||0);
    const total=activeManual?Number(activeManual.total_flagged_sample_count||0):pending;
    const attention=activeManual?pending>0:bodyReviewNeedsAttention(row);
    button.disabled=false;
    button.dataset.artifactId=row.artifact_id||'';
    if(attention){
      const eventText=pendingUnits===null?'review events':`${pendingUnits} review event${pendingUnits===1?'':'s'}`;
      const sampleText=pending===null?'flagged samples':`${pending} gaze sample${pending===1?'':'s'}`;
      status.textContent=`${row.label||'Body AOI'} · ${eventText} (${sampleText})${frames>0?` across ${frames} frame${frames===1?'':'s'}`:''} need self/partner or body-region verification.`;
      count.textContent=pendingUnits===null?'Review required':`${pendingUnits} events · ${pending} samples`;
      count.className='status-pill warning';
      button.textContent=decisions>0?'Continue Body review':'Review Body AOI';
    }else if((total||0)>0||decisions>0){
      status.textContent=`${row.label||'Body AOI'} · the Body human-review queue is complete. Reopen it to audit saved decisions.`;
      count.textContent='Review complete';
      count.className='status-pill good';
      button.textContent='Open reviewed queue';
    }else{
      status.textContent=`${row.label||'Body AOI'} · no flagged sample is pending. Open the player for a manual spot-check.`;
      count.textContent='Spot-check ready';
      count.className='status-pill good';
      button.textContent='Open Body AOI review';
    }
    if(sideButton&&sideCount&&row&&sideButton.dataset.artifactId===row.artifact_id){
      sideCount.textContent=attention?(pending===null?'Review events pending':`${pendingUnits} review event${pendingUnits===1?'':'s'} · ${pending} sample${pending===1?'':'s'}`):((total||0)>0||decisions>0?'Review complete':'Queue ready');
      sideButton.textContent=attention?'Review flagged Body AOI':((total||0)>0||decisions>0?'Open reviewed Body AOI':'Open Body AOI review');
    }
  }

  async function openBodyHumanReview() {
    const row=preferredBodyHumanReviewArtifact();
    if(!row){toast('No Body AOI available','Generate a Body AOI artifact first, then open Body human review.');return;}
    const mode=document.querySelector('#aoi-review-mode');
    if(mode)mode.value='body';
    state.mode='body';
    state.artifactId=row.artifact_id||'';
    state.videoArtifactId='';
    populateArtifacts(state.artifactId);
    await loadManifest();
    updateHumanReviewEntry();updateBodyHumanReviewEntry();
    requestAnimationFrame(()=>{
      const panel=document.querySelector('#body-manual-review');
      if(!panel)return;
      panel.hidden=false;
      panel.classList.add('human-review-focus');
      panel.scrollIntoView({behavior:'smooth',block:'start'});
      window.setTimeout(()=>panel.classList.remove('human-review-focus'),1600);
    });
  }

  function populateArtifacts(preferred='') {
    const select=document.querySelector('#aoi-review-artifact');
    if(!select)return;
    const rows=currentRows();
    const candidate=preferred||state.artifactId||state.initialArtifactId;
    const selected=rows.some(row=>row.artifact_id===candidate)?candidate:(rows.slice(-1)[0]?.artifact_id||'');
    select.innerHTML=optionHtml(rows,selected,'No artifact');
    select.value=selected;
    state.artifactId=select.value;
    populateVideos(state.videoArtifactId||state.initialVideoArtifactId);
  }

  function populateVideos(preferred='') {
    const select=document.querySelector('#aoi-review-video');
    if(!select)return;
    const row=currentRows().find(item=>item.artifact_id===state.artifactId);
    const compatible=new Set(row?.compatible_video_artifact_ids||[]);
    let videos=(state.options?.videos||[]).filter(video=>!compatible.size||compatible.has(video.artifact_id));
    if(state.interactionId||state.roleId||state.nativeTrimArtifactId){
      videos=videos.filter(video=>contextMatch(video));
    }
    const candidate=preferred||state.videoArtifactId||state.initialVideoArtifactId;
    const selected=videos.some(video=>video.artifact_id===candidate)?candidate:(row?.default_video_artifact_id&&videos.some(video=>video.artifact_id===row.default_video_artifact_id)?row.default_video_artifact_id:videos.slice(-1)[0]?.artifact_id||'');
    select.innerHTML=optionHtml(videos,selected,'No compatible video');
    select.value=selected;
    state.videoArtifactId=select.value;
  }

  function contextMatch(row) {
    const interaction=String(row?.interaction_id||'').toUpperCase();
    const role=String(row?.role_id||'').toUpperCase();
    const trim=String(row?.native_trim_artifact_id||'');
    const label=String(row?.label||'').toUpperCase();
    const interactionTokens=label.match(/(?:^|[^A-Z0-9])(I\d{1,6})(?![A-Z0-9])/g)||[];
    const roleTokens=label.match(/(?:^|[^A-Z0-9])(R\d{1,6})(?![A-Z0-9])/g)||[];
    if(state.interactionId){
      if(interaction&&interaction!==state.interactionId)return false;
      if(!interaction&&(!interactionTokens.length||!label.includes(state.interactionId)))return false;
    }
    if(state.roleId){
      if(role&&role!==state.roleId)return false;
      if(!role&&(!roleTokens.length||!label.includes(state.roleId)))return false;
    }
    if(state.nativeTrimArtifactId&&trim!==state.nativeTrimArtifactId)return false;
    return true;
  }

  function videoRect(video, canvas) {
    const width=canvas.clientWidth, height=canvas.clientHeight;
    const vw=video.videoWidth||width, vh=video.videoHeight||height;
    const scale=Math.min(width/vw,height/vh);
    return {x:(width-vw*scale)/2,y:(height-vh*scale)/2,w:vw*scale,h:vh*scale,scale,vw,vh};
  }

  function resizeCanvas(canvas) {
    const dpr=window.devicePixelRatio||1;
    const width=Math.max(1,canvas.clientWidth),height=Math.max(1,canvas.clientHeight);
    if(canvas.width!==Math.round(width*dpr)||canvas.height!==Math.round(height*dpr)){
      canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
    }
    const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);
    return ctx;
  }

  const faceColors={left_eye:'#eaff38',right_eye:'#65e55e',nose:'#4bc3ff',mouth:'#ff6fe0',other_face:'#d2d6a1',face:'#d2d6a1'};
  const bodyRegionMeta={
    head:{label:'Head',color:'#f4d35e'},torso:{label:'Torso',color:'#4cc9f0'},
    left_arm:{label:'Left arm',color:'#ff9f43'},right_arm:{label:'Right arm',color:'#ff6b9d'},
    left_hand:{label:'Left hand',color:'#ffc078'},right_hand:{label:'Right hand',color:'#ff9fce'},
    left_leg:{label:'Left leg',color:'#73d17b'},right_leg:{label:'Right leg',color:'#9a8cff'},
    no_body:{label:'No body',color:'#777d70'},no_data:{label:'No gaze samples',color:'#3f443c'},
    invalid_gaze:{label:'Invalid gaze',color:'#ef554a'},out_of_frame:{label:'Out of frame',color:'#ef776e'},
    unknown:{label:'Unknown',color:'#b2b7a7'},no_gaze:{label:'No gaze',color:'#777d70'},
  };
  const relationMeta={partner:{label:'Partner',color:'#48e6d1'},self:{label:'Self',color:'#ffbf55'},other:{label:'Other',color:'#d3d8c2'},none:{label:'No body',color:'#777d70'}};
  const bodyRegionKey=(value)=>String(value||'unknown').toLowerCase().replace(/^(self|partner|other)_/,'');
  const bodyRelationKey=(value,explicit='')=>{
    const relation=String(explicit||'').toLowerCase();if(['self','partner','other'].includes(relation))return relation;
    const social=String(value||'').toLowerCase();if(social.startsWith('self_'))return 'self';if(social.startsWith('partner_'))return 'partner';if(social.startsWith('other_'))return 'other';return 'none';
  };
  const bodyRegionLabel=(value)=>bodyRegionMeta[bodyRegionKey(value)]?.label||bodyRegionKey(value).replaceAll('_',' ');
  const bodyColor=(region,relation='')=>bodyRegionMeta[bodyRegionKey(region)]?.color||relationMeta[bodyRelationKey(region,relation)]?.color||'#d2d6a1';
  const bodyTargetLabel=(region,relation='')=>{const key=bodyRegionKey(region),rel=bodyRelationKey(region,relation);if(['no_body','no_data','invalid_gaze','out_of_frame','unknown','no_gaze'].includes(key)||rel==='none')return bodyRegionLabel(region);return `${relationMeta[rel]?.label||'Body'} · ${bodyRegionLabel(region)}`;};
  const faceRegionLabels={
    left_eye:'left_eye · subject left',
    right_eye:'right_eye · subject right',
    nose:'nose',mouth:'mouth',other_face:'face_other',face:'face',
    partner_left_eye:'partner_left_eye · subject left',
    partner_right_eye:'partner_right_eye · subject right',
    partner_nose:'partner_nose',partner_mouth:'partner_mouth',
    partner_face_unspecified:'partner_face_unspecified',
    partner_face_candidate:'partner_face_candidate · projected partner · review',
    self_left_eye:'self_left_eye · subject left',self_right_eye:'self_right_eye · subject right',
    self_face_unspecified:'self_face_unspecified',other_person_face:'other_person_face · no partner projection',
    ambiguous_person_face:'ambiguous_person_face',
  };
  const displayRegion=(region)=>{
    if(state.mode==='face')return faceRegionLabels[String(region||'')]||String(region||'');
    const key=bodyRegionKey(region);
    if(state.mode==='body')return bodyTargetLabel(region);
    if(bodyRegionMeta[key])return bodyTargetLabel(region);
    return String(region||'').replaceAll('_',' ');
  };
  function displayFaceEntity(entity,region) {
    const value=String(entity||'').trim();
    if(value&&value!=='—'&&value.toLowerCase()!=='nan')return value;
    const semantic=String(region||'').toLowerCase();
    if(semantic==='no_face')return 'No face';
    if(semantic==='no_face_detection')return 'Face not detected';
    if(semantic==='invalid_gaze')return 'Invalid gaze';
    if(semantic==='out_of_frame')return 'Out of frame';
    if(semantic.startsWith('partner_'))return 'Partner';
    if(semantic.startsWith('self_'))return 'Self';
    if(semantic.includes('ambiguous')||semantic.includes('unresolved')||semantic==='unknown')return 'Unresolved';
    return 'Unassigned';
  }

  function drawVideoOverlay() {
    const video=document.querySelector('#aoi-review-video-element');
    const canvas=document.querySelector('#aoi-review-overlay');
    if(!video||!canvas)return;
    const ctx=resizeCanvas(canvas), payload=state.currentPayload;
    if(!video.videoWidth)return;
    const rect=videoRect(video,canvas);
    const expectedFrame=state.presentedFrame>=0?state.presentedFrame:frameFromMediaTime(video.currentTime||0);
    const payloadFrame=payload?.output_frame_index==null?-1:Number(payload.output_frame_index);
    const exactFramePending=Boolean(video.seeking||state.pauseLockFrame>=0||state.pendingExplicitSeekFrame>=0||state.sourceSwitchPending);
    const overlaySynced=!exactFramePending&&payloadFrame===Number(expectedFrame);
    canvas.dataset.presentedFrame=String(expectedFrame);
    canvas.dataset.lastPresentedFrame=state.lastPresentedFrame>=0?String(state.lastPresentedFrame):'';
    canvas.dataset.pauseLockFrame=state.pauseLockFrame>=0?String(state.pauseLockFrame):'';
    canvas.dataset.videoCurrentFrame=String(frameFromMediaTime(video.currentTime||0));
    canvas.dataset.overlayFrame=payloadFrame>=0?String(payloadFrame):'';
    canvas.dataset.exactFramePending=exactFramePending?'true':'false';
    canvas.dataset.overlaySynced=overlaySynced?'true':'false';
    if(exactFramePending){
      state.lastOverlayFrame=-1;
      ctx.font='800 11px system-ui';
      ctx.fillStyle='rgba(26,31,20,.88)';ctx.fillRect(14,14,190,31);
      ctx.fillStyle='#eaff38';ctx.fillText('LOCKING PAUSED FRAME',27,35);
      return;
    }
    if(!overlaySynced){
      state.lastOverlayFrame=-1;
      if(state.manifest){
        ctx.font='800 11px system-ui';
        ctx.fillStyle='rgba(26,31,20,.88)';ctx.fillRect(14,14,166,31);
        ctx.fillStyle='#eaff38';ctx.fillText('SYNCING OVERLAY',27,35);
      }
      return;
    }
    state.lastOverlayFrame=payloadFrame;
    const coordinate=state.manifest?.coordinate_space||{};
    const sourceWidth=Number(coordinate.width)||rect.vw;
    const sourceHeight=Number(coordinate.height)||rect.vh;
    const scaleX=rect.w/Math.max(1,sourceWidth);
    const scaleY=rect.h/Math.max(1,sourceHeight);
    const mapPoint=(x,y)=>({x:rect.x+Number(x)*scaleX,y:rect.y+Number(y)*scaleY});
    ctx.save();ctx.beginPath();ctx.rect(rect.x,rect.y,rect.w,rect.h);ctx.clip();

    if(state.layers.face){
      for(const row of payload.face_regions||[]){
        try{
          const primary=row.include_in_primary_analysis!==false;
          if(!primary&&!state.layers.other)continue;
          const g=typeof row.geometry_json==='string'?JSON.parse(row.geometry_json):row.geometry_json;
          const social=row.face_social_region||row.region;const candidate=String(social||'').includes('candidate');
          ctx.strokeStyle=primary?(faceColors[row.region]||'#eaff38'):(candidate?'#43d2f5':'#ff9d2e');ctx.lineWidth=2;ctx.setLineDash(row.region==='other_face'||candidate?[7,5]:[]);
          let labelPoint=null;
          const polygonPoints=Array.isArray(g?.points)?g.points:[];
          if(String(row.geometry_type||'')==='polygon'||polygonPoints.length>=3){
            const mapped=polygonPoints.map(point=>mapPoint(point[0],point[1]));
            if(mapped.length<3)continue;
            ctx.beginPath();ctx.moveTo(mapped[0].x,mapped[0].y);for(const point of mapped.slice(1))ctx.lineTo(point.x,point.y);ctx.closePath();ctx.stroke();
            labelPoint={x:mapped.reduce((sum,point)=>sum+point.x,0)/mapped.length,y:mapped.reduce((sum,point)=>sum+point.y,0)/mapped.length};
          }else{
            const c=mapPoint(g.center_x,g.center_y);const rx=Number(g.radius_x)*scaleX,ry=Number(g.radius_y)*scaleY;
            ctx.beginPath();ctx.ellipse(c.x,c.y,rx,ry,0,0,Math.PI*2);ctx.stroke();
            labelPoint={x:c.x+rx,y:c.y};
          }
          ctx.setLineDash([]);
          ctx.font='600 11px system-ui';ctx.fillStyle=ctx.strokeStyle;ctx.fillText(primary?displayRegion(social):(candidate?displayRegion(social):`other · ${displayRegion(social)}`),labelPoint.x+4,labelPoint.y);
        }catch(_){ }
      }
    }

    if(state.layers.participant){
      for(const row of payload.participant_projections||[]){
        if(!row.projection_valid||!row.in_camera_view)continue;
        const a=mapPoint(row.projected_bbox_x1,row.projected_bbox_y1),b=mapPoint(row.projected_bbox_x2,row.projected_bbox_y2);
        const relation=String(row.relation_to_wearer||'tracked');
        ctx.strokeStyle=relation==='partner'?'#43f1ff':'#ffd24c';ctx.lineWidth=2;ctx.setLineDash([8,5]);
        ctx.strokeRect(a.x,a.y,b.x-a.x,b.y-a.y);ctx.setLineDash([]);
        ctx.font='700 11px system-ui';ctx.fillStyle=ctx.strokeStyle;
        ctx.fillText(`projected ${row.role_id||row.entity_id||'participant'} · ${relation}`,a.x+3,Math.max(14,a.y-5));
      }
    }

    if(state.layers.identity){
      for(const row of payload.face_identities||[]){
        const relation=String(row.target_relation||'other_person');
        if(relation==='other_person'&&!state.layers.other)continue;
        const a=mapPoint(row.face_box_x1,row.face_box_y1),b=mapPoint(row.face_box_x2,row.face_box_y2);
        const status=String(row.identity_status||'unknown');
        const matchedStatuses=new Set(['matched_participant','duplicate_inherits_matched_participant','temporal_bridge_matched_participant','temporal_edge_matched_participant']);
        const partnerCandidate=relation==='ambiguous_partner'||relation==='partner_candidate'||row.candidate_relation==='partner';
        let color='#ff9d2e';
        if(relation==='partner'&&matchedStatuses.has(status))color='#65e55e';
        else if(relation==='self'&&matchedStatuses.has(status))color='#ffd24c';
        else if(partnerCandidate)color='#43d2f5';
        else if(status.includes('ambiguous')||row.review_required)color='#ef554a';
        ctx.strokeStyle=color;ctx.lineWidth=2.5;ctx.strokeRect(a.x,a.y,b.x-a.x,b.y-a.y);
        const entity=row.matched_entity_id||row.candidate_entity_id||'other_person';const conf=row.identity_confidence==null?'—':Number(row.identity_confidence).toFixed(2);
        const relationLabel=partnerCandidate?'partner_candidate':relation;
        ctx.font='700 11px system-ui';ctx.fillStyle=color;ctx.fillText(`${entity} · ${relationLabel} · ${conf}`,a.x+3,Math.min(canvas.clientHeight-5,b.y+15));
      }
    }

    if(state.layers.trail){
      const points=(payload.trail||[]).filter(r=>r.gaze_x_px!=null&&r.gaze_y_px!=null).map(r=>mapPoint(r.gaze_x_px,r.gaze_y_px));
      if(points.length>1){ctx.strokeStyle='rgba(103,226,91,.75)';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(points[0].x,points[0].y);for(const p of points.slice(1))ctx.lineTo(p.x,p.y);ctx.stroke();}
      points.forEach((p,i)=>{ctx.fillStyle=`rgba(232,255,54,${0.15+0.75*(i+1)/Math.max(1,points.length)})`;ctx.beginPath();ctx.arc(p.x,p.y,2.3,0,Math.PI*2);ctx.fill();});
    }

    if(state.layers.gaze){
      (payload.samples||[]).forEach((sample,index)=>{
        if(sample.gaze_x_px==null||sample.gaze_y_px==null)return;
        const p=mapPoint(sample.gaze_x_px,sample.gaze_y_px);const selected=index===payload.sample_index;
        const bodyEffective=state.mode==='body'?effectiveBodyTarget(sample):null;
        const social=bodyEffective?.resolved?bodyEffective.region:(sample.body_social_region||sample.body_region||'unknown');
        const relation=bodyEffective?.resolved?bodyEffective.relation:(sample.body_relation||bodyRelationKey(social));
        const targetColor=state.mode==='body'?bodyColor(social,relation):'#eaff38';
        ctx.fillStyle=selected?'#ffffff':targetColor;ctx.strokeStyle=selected?targetColor:'rgba(30,35,20,.8)';ctx.lineWidth=selected?3:1.5;
        ctx.beginPath();ctx.arc(p.x,p.y,selected?7:4.5,0,Math.PI*2);ctx.fill();ctx.stroke();
        if(selected){
          ctx.beginPath();ctx.moveTo(p.x-12,p.y);ctx.lineTo(p.x+12,p.y);ctx.moveTo(p.x,p.y-12);ctx.lineTo(p.x,p.y+12);ctx.stroke();
          if(state.mode==='body'&&state.layers.body){
            const label=bodyTargetLabel(social,relation);ctx.font='700 10px system-ui';const width=ctx.measureText(label).width;
            const tx=Math.min(rect.x+rect.w-width-18,p.x+13),ty=Math.max(rect.y+16,p.y-12);
            ctx.fillStyle='rgba(20,24,17,.86)';ctx.fillRect(tx-5,ty-12,width+10,18);ctx.fillStyle=targetColor;ctx.fillText(label,tx,ty+1);
          }
        }
      });
    }
    ctx.restore();

    const summary=payload.summary||{};const selected=payload.selected_sample||{};
    const effective=state.mode==='face'?effectiveFaceTarget(selected):null;
    const bodyEffective=state.mode==='body'?effectiveBodyTarget(selected):null;
    const region = state.mode==='fusion' ? selected.final_target_region : state.mode==='body' ? (bodyEffective?.resolved?bodyEffective.region:(selected.body_social_region||selected.body_region)) : (effective?.resolved?effective.region:(selected.face_social_region||selected.face_region));
    const entity = state.mode==='fusion' ? selected.final_entity_id : state.mode==='body' ? (bodyEffective?.resolved?bodyEffective.entity:selected.body_entity_id) : (effective?.resolved?effective.entity:selected.face_entity_id);
    const confidence = state.mode==='fusion' ? selected.final_confidence : state.mode==='body' ? (bodyEffective?.resolved?bodyEffective.confidence:selected.body_confidence) : (effective?.resolved?effective.confidence:selected.face_aoi_confidence);
    if(state.layers.final){
      const labelRegion=region||summary.primary_region||'no_gaze';
      const labelEntity=state.mode==='face'?displayFaceEntity(entity||summary.primary_entity_id,labelRegion):(entity||summary.primary_entity_id||'—');
      const label=`${labelEntity} · ${displayRegion(labelRegion)}`;
      ctx.font='700 14px system-ui';const tw=ctx.measureText(label).width;ctx.fillStyle='rgba(30,34,21,.88)';ctx.fillRect(14,14,Math.min(canvas.clientWidth-28,tw+34),38);ctx.fillStyle='#eaff38';ctx.fillText(label,30,39);
    }
    if(state.layers.confidence&&confidence!=null){ctx.font='600 12px system-ui';ctx.fillStyle='rgba(30,34,21,.82)';ctx.fillRect(14,57,154,29);ctx.fillStyle='#f3f5e8';ctx.fillText(`confidence ${Number(confidence).toFixed(3)}`,27,77);}
    if(state.layers.review&&(summary.review_required||selected.review_required||selected.manual_review_resolved||effective?.resolved||bodyEffective?.resolved)){
      const resolved=Boolean(effective?.resolved||bodyEffective?.resolved||selected.manual_review_resolved);
      ctx.font='800 12px system-ui';ctx.fillStyle=resolved?'rgba(48,132,78,.94)':'rgba(186,56,46,.92)';ctx.fillRect(canvas.clientWidth-175,14,160,34);ctx.fillStyle='#fff';ctx.fillText(resolved?'HUMAN REVIEWED':'REVIEW REQUIRED',canvas.clientWidth-163,36);
    }
  }


  function augmentVisualTorsoFallback(parsedRows) {
    const parsed=[...(parsedRows||[])];
    const groups=new Map();
    const addPoint=(group,key,point)=>{if(Array.isArray(point)&&point.length===3&&point.every(value=>Number.isFinite(Number(value)))){(group[key]||(group[key]=[])).push(point.map(Number));}};
    const mean=points=>{if(!points?.length)return null;return [0,1,2].map(axis=>points.reduce((sum,point)=>sum+Number(point[axis]),0)/points.length);};
    const distance=(a,b)=>Math.hypot(Number(a[0])-Number(b[0]),Number(a[1])-Number(b[1]),Number(a[2])-Number(b[2]));
    for(const item of parsed){
      const row=item.row||{},g=item.g||{};const entity=String(row.entity_id||'body'),relation=String(row.entity_relation||row.relation||'other').toLowerCase();const key=`${entity}|${relation}`;
      const group=groups.get(key)||{entity,relation,hasTorso:false,hasPelvis:false,shoulders:[],hips:[],necks:[],heads:[],confidence:[]};
      const raw=String(row.region||'').toLowerCase(),coarse=String(row.coarse_region||raw).toLowerCase();
      if(raw==='torso')group.hasTorso=true;if(raw==='pelvis')group.hasPelvis=true;
      if(['left_upper_arm','right_upper_arm'].includes(raw))addPoint(group,'shoulders',g.start_mm);
      if(['left_thigh','right_thigh'].includes(raw))addPoint(group,'hips',g.start_mm);
      if(raw==='neck'){addPoint(group,'necks',g.end_mm);addPoint(group,'necks',g.start_mm);}
      if(raw==='head')addPoint(group,'heads',g.center_mm);
      const confidence=Number(row.confidence);if(Number.isFinite(confidence))group.confidence.push(confidence);
      groups.set(key,group);
    }
    for(const group of groups.values()){
      if(group.hasTorso)continue;
      let lower=mean(group.hips);let upper=mean(group.shoulders)||mean(group.necks);const head=mean(group.heads);
      if(!lower||(!upper&&!head))continue;
      if(!upper&&head)upper=lower.map((value,index)=>value+0.64*(head[index]-value));
      if(!upper||distance(upper,lower)<60||distance(upper,lower)>1200)continue;
      const shoulderWidth=group.shoulders.length>=2?distance(group.shoulders[0],group.shoulders[1]):0;
      const hipWidth=group.hips.length>=2?distance(group.hips[0],group.hips[1]):0;
      const inferredWidth=Math.max(shoulderWidth,hipWidth);const radius=Math.max(135,Math.min(220,inferredWidth?0.46*inferredWidth:185));
      const confidence=group.confidence.length?Math.max(0.35,Math.min(0.72,group.confidence.reduce((a,b)=>a+b,0)/group.confidence.length*0.72)):0.55;
      parsed.push({
        row:{entity_id:group.entity,entity_relation:group.relation,relation:group.relation,region:'torso',coarse_region:'torso',geometry_provenance:'visual_fallback',geometry_is_derived:true,confidence},
        g:{start_mm:upper,end_mm:lower,radius_mm:radius,visual_fallback:true},
      });
    }
    return parsed;
  }

  function drawWorldInset() {
    const canvas=document.querySelector('#aoi-review-world');if(!canvas)return;
    const ctx=resizeCanvas(canvas),payload=state.currentPayload;
    if(!payload||!state.layers.body){return;}
    const rows=payload.body_regions||[];const sample=payload.selected_sample||{};
    canvas.dataset.overlayFrame=payload.output_frame_index==null?'':String(payload.output_frame_index);
    canvas.dataset.gazeSampleUid=sample.gaze_sample_uid==null?'':String(sample.gaze_sample_uid);
    canvas.dataset.rayDirection=[sample.ray_direction_x,sample.ray_direction_y,sample.ray_direction_z].map(value=>value==null?'':Number(value).toFixed(6)).join(',');
    const effectiveBody=state.mode==='body'?effectiveBodyTarget(sample):null;
    let parsed=[];const points=[];let maximumRadiusMm=0;
    for(const row of rows){
      try{
        const g=typeof row.geometry_json==='string'?JSON.parse(row.geometry_json):row.geometry_json;
        if(!g)continue;parsed.push({row,g});
      }catch(_){ }
    }
    parsed=augmentVisualTorsoFallback(parsed);
    for(const {g} of parsed){
      const radius=Number(g.radius_mm||0);if(Number.isFinite(radius))maximumRadiusMm=Math.max(maximumRadiusMm,radius);
      for(const key of ['center_mm','start_mm','end_mm'])if(Array.isArray(g[key])&&g[key].length===3)points.push(g[key].map(Number));
    }
    const origin=sample.ray_origin_x_mm==null?null:[sample.ray_origin_x_mm,sample.ray_origin_y_mm,sample.ray_origin_z_mm].map(Number);
    const effectiveHitX=effectiveBody?.resolved?effectiveBodyField(sample,'body_hit_x_mm'):sample.body_hit_x_mm;
    const effectiveHitY=effectiveBody?.resolved?effectiveBodyField(sample,'body_hit_y_mm'):sample.body_hit_y_mm;
    const effectiveHitZ=effectiveBody?.resolved?effectiveBodyField(sample,'body_hit_z_mm'):sample.body_hit_z_mm;
    const hit=effectiveHitX==null?null:[effectiveHitX,effectiveHitY,effectiveHitZ].map(Number);
    if(origin&&origin.every(Number.isFinite))points.push(origin);if(hit&&hit.every(Number.isFinite))points.push(hit);
    if(!points.length){ctx.fillStyle='rgba(255,255,255,.56)';ctx.font='11px system-ui';ctx.fillText('No 3D body evidence for this frame',12,56);return;}

    const stable=state.bodyStableScale?state.manifest?.body_world_bounds:null;
    const frameBounds={};
    for(const [axis,index] of [['x',0],['y',1],['z',2]]){
      const values=points.map(point=>Number(point[index])).filter(Number.isFinite);let min=Math.min(...values),max=Math.max(...values);
      if(max-min<250){const centre=(max+min)/2;min=centre-125;max=centre+125;}const pad=Math.max(80,(max-min)*.10,maximumRadiusMm*1.15);
      frameBounds[axis]={min:min-pad,max:max+pad};
    }
    const bounds=stable||frameBounds;
    const views=state.bodyView==='split'?[{axis:'x',label:'X–Z'},{axis:'y',label:'Y–Z'}]:[{axis:state.bodyView==='yz'?'y':'x',label:state.bodyView==='yz'?'Y–Z':'X–Z'}];
    const width=canvas.clientWidth,height=canvas.clientHeight;const top=34,bottom=22,gap=8;
    const panelWidth=(width-gap*(views.length+1))/views.length;
    const selectedEntity=String(effectiveBody?.resolved?effectiveBody.entity:(sample.body_entity_id||''));
    const selectedRaw=String(effectiveBody?.resolved?effectiveBody.rawRegion:(sample.body_raw_region||sample.body_region||''));
    const selectedCoarse=String(effectiveBody?.resolved?effectiveBody.coarseRegion:(sample.body_coarse_region||sample.body_region||''));
    const entityAnchors=new Map();

    const niceScale=(mmPerPixel,pixels)=>{const target=mmPerPixel*pixels;const steps=[100,200,250,500,1000,2000,5000];return steps.find(value=>value>=target)||5000;};
    views.forEach((view,viewIndex)=>{
      const x0=gap+viewIndex*(panelWidth+gap),y0=top,w=panelWidth,h=Math.max(20,height-top-bottom);
      ctx.fillStyle='rgba(255,255,255,.025)';ctx.fillRect(x0,y0,w,h);ctx.strokeStyle='rgba(255,255,255,.10)';ctx.strokeRect(x0+.5,y0+.5,w-1,h-1);
      const horizontal=bounds[view.axis]||frameBounds[view.axis],vertical=bounds.z||frameBounds.z;
      const spanH=Math.max(1,horizontal.max-horizontal.min),spanV=Math.max(1,vertical.max-vertical.min);
      const scale=Math.min((w-22)/spanH,(h-24)/spanV);const usedW=spanH*scale,usedH=spanV*scale;
      const ox=x0+(w-usedW)/2,oy=y0+(h-usedH)/2;
      const map=point=>({x:ox+(Number(point[view.axis==='x'?0:1])-horizontal.min)*scale,y:oy+usedH-(Number(point[2])-vertical.min)*scale});
      ctx.setLineDash([]);ctx.lineWidth=1;ctx.strokeStyle='rgba(255,255,255,.07)';
      for(let i=1;i<4;i++){const gx=ox+usedW*i/4,gy=oy+usedH*i/4;ctx.beginPath();ctx.moveTo(gx,oy);ctx.lineTo(gx,oy+usedH);ctx.stroke();ctx.beginPath();ctx.moveTo(ox,gy);ctx.lineTo(ox+usedW,gy);ctx.stroke();}
      ctx.font='800 9px system-ui';ctx.fillStyle='rgba(237,241,225,.78)';ctx.fillText(view.label,x0+7,y0+12);
      ctx.font='600 7px system-ui';ctx.fillStyle='rgba(237,241,225,.42)';ctx.fillText(`${view.axis.toUpperCase()} →`,x0+w-28,y0+h-5);ctx.fillText('Z ↑',x0+5,y0+24);

      // Draw the head and torso as projected body surfaces before the
      // centre-lines.  Earlier versions rendered every capsule as a thin
      // segment, so the torso was technically present but visually
      // indistinguishable from an arm or leg.
      for(const {row,g} of parsed){
        const relation=String(row.entity_relation||row.relation||'other').toLowerCase();
        const raw=String(row.region||'unknown'),coarse=String(row.coarse_region||raw);
        if(!['head','torso'].includes(coarse))continue;
        const active=String(row.entity_id||'')===selectedEntity&&(raw===selectedRaw||coarse===selectedCoarse);
        const visualFallback=Boolean(g.visual_fallback||row.geometry_provenance==='visual_fallback');
        const surfaceColor=bodyColor(coarse,relation);const radiusPx=Math.max(2,Math.min(48,Number(g.radius_mm||20)*scale));
        ctx.save();ctx.strokeStyle=surfaceColor;ctx.fillStyle=surfaceColor;ctx.lineCap='round';ctx.setLineDash(visualFallback?[3,3]:[]);
        const relationAlpha=relation==='self'?.72:relation==='partner'?1:.62;
        ctx.globalAlpha=(active?.48:(coarse==='torso'?(visualFallback?.20:.30):.16))*relationAlpha;
        if(g.center_mm){const p=map(g.center_mm);ctx.beginPath();ctx.arc(p.x,p.y,radiusPx,0,Math.PI*2);ctx.fill();}
        else if(g.start_mm&&g.end_mm){const a=map(g.start_mm),b=map(g.end_mm);ctx.lineWidth=Math.max(4,2*radiusPx);ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();}
        ctx.restore();
      }
      for(const {row,g} of parsed){
        const relation=String(row.entity_relation||row.relation||'other').toLowerCase();
        const raw=String(row.region||'unknown'),coarse=String(row.coarse_region||raw);
        const active=String(row.entity_id||'')===selectedEntity&&(raw===selectedRaw||coarse===selectedCoarse);
        const visualFallback=Boolean(g.visual_fallback||row.geometry_provenance==='visual_fallback');
        const color=active?'#ffffff':bodyColor(coarse,relation);ctx.save();ctx.strokeStyle=color;ctx.fillStyle=color;ctx.lineCap='round';ctx.lineWidth=active?4:(coarse==='torso'?2.4:relation==='partner'?2.2:1.7);
        ctx.globalAlpha=active?1:(relation==='self'?.72:relation==='partner'?.92:.56);ctx.setLineDash(visualFallback?[3,3]:(relation==='self'?[5,4]:[]));
        if(g.center_mm){const p=map(g.center_mm),r=Math.max(2,Math.min(48,Number(g.radius_mm||20)*scale));ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.stroke();}
        else if(g.start_mm&&g.end_mm){const a=map(g.start_mm),b=map(g.end_mm);ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();}
        ctx.restore();
        const key=`${row.entity_id||'body'}|${relation}`;const anchorPoint=g.center_mm||g.start_mm;
        if(anchorPoint&&['head','torso'].includes(coarse)){const mapped=map(anchorPoint);const existing=entityAnchors.get(`${viewIndex}|${key}`)||[];existing.push(mapped);entityAnchors.set(`${viewIndex}|${key}`,existing);}
      }
      ctx.globalAlpha=1;ctx.setLineDash([]);
      if(origin&&sample.ray_direction_x!=null){
        const direction=[sample.ray_direction_x,sample.ray_direction_y,sample.ray_direction_z].map(Number);const distance=Number((effectiveBody?.resolved?effectiveBody.distance:null)||sample.body_distance_mm||sample.nearest_partner_hit_distance_mm||900);
        if(direction.every(Number.isFinite)){const end=origin.map((value,index)=>value+direction[index]*distance);const a=map(origin),b=map(end);ctx.strokeStyle='#6ef28a';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();}
      }
      if(hit){const p=map(hit);const hitRegion=effectiveBody?.resolved?effectiveBody.region:(sample.body_social_region||sample.body_region);const hitRelation=effectiveBody?.resolved?effectiveBody.relation:sample.body_relation;ctx.fillStyle='#ffffff';ctx.strokeStyle=bodyColor(hitRegion,hitRelation);ctx.lineWidth=3;ctx.beginPath();ctx.arc(p.x,p.y,5,0,Math.PI*2);ctx.fill();ctx.stroke();}
      const scaleMm=niceScale(1/scale,52);const scalePx=scaleMm*scale;ctx.strokeStyle='rgba(255,255,255,.68)';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x0+9,y0+h-10);ctx.lineTo(x0+9+scalePx,y0+h-10);ctx.stroke();ctx.font='600 7px system-ui';ctx.fillStyle='rgba(255,255,255,.62)';ctx.fillText(scaleMm>=1000?`${(scaleMm/1000).toFixed(scaleMm%1000?1:0)} m`:`${scaleMm} mm`,x0+11,y0+h-13);
    });
    ctx.globalAlpha=1;ctx.setLineDash([]);
    for(const [key,anchors] of entityAnchors.entries()){
      const parts=key.split('|');parts.shift();const relation=parts.pop()||'other';const entity=parts.join('|')||'body';
      const x=anchors.reduce((sum,p)=>sum+p.x,0)/anchors.length,y=Math.min(...anchors.map(p=>p.y))-5;
      ctx.font='700 8px system-ui';ctx.fillStyle=relationMeta[relation]?.color||'#e9edd8';ctx.fillText(`${relation} · ${entity}`,x-18,Math.max(44,y));
    }
    const subtitle=document.querySelector('#aoi-review-world-subtitle');
    if(subtitle){const region=effectiveBody?.resolved?effectiveBody.region:(sample.body_social_region||sample.body_region||'no_body');const relation=effectiveBody?.resolved?effectiveBody.relation:sample.body_relation;const distance=Number(effectiveBody?.resolved?effectiveBody.distance:sample.body_distance_mm);subtitle.textContent=`${bodyTargetLabel(region,relation)}${effectiveBody?.resolved?' · reviewed':''}${Number.isFinite(distance)?` · ${(distance/1000).toFixed(2)} m`:''}`;}
  }

  function bodyMapSvg(relation,counts={},selectedRelation='',selectedRegion='') {
    const regions=['head','torso','left_arm','right_arm','left_hand','right_hand','left_leg','right_leg'];
    const maximum=Math.max(1,...regions.map(region=>Number(counts[region]||0)));
    const attrs=region=>{const value=Number(counts[region]||0),ratio=value/maximum;const active=relation===selectedRelation&&region===bodyRegionKey(selectedRegion);return `style="--body-fill:${bodyColor(region,relation)};--body-alpha:${(0.16+0.84*ratio).toFixed(3)}" class="body-map-part ${active?'active':''}" data-region="${region}"`;};
    return `<div class="body-map-figure"><svg viewBox="0 0 100 150" role="img" aria-label="${relation} body AOI distribution"><circle ${attrs('head')} cx="50" cy="16" r="11"/><rect ${attrs('torso')} x="34" y="31" width="32" height="48" rx="12"/><path ${attrs('left_arm')} d="M36 39 L18 70"/><path ${attrs('right_arm')} d="M64 39 L82 70"/><circle ${attrs('left_hand')} cx="15" cy="76" r="6"/><circle ${attrs('right_hand')} cx="85" cy="76" r="6"/><path ${attrs('left_leg')} d="M43 76 L35 132"/><path ${attrs('right_leg')} d="M57 76 L65 132"/></svg><strong>${relationMeta[relation]?.label||relation}</strong><small>${Object.values(counts).reduce((sum,value)=>sum+Number(value||0),0)} hits</small></div>`;
  }

  function renderBodyOverview() {
    const root=document.querySelector('#body-aoi-overview');if(!root)return;
    const active=Boolean(state.manifest&&['body','fusion'].includes(state.mode)&&state.manifest.body_analytics);
    root.hidden=!active;if(!active){root.innerHTML='';return;}
    const analytics=state.manifest.body_analytics||{},sample=state.currentPayload?.selected_sample||{},effectiveBody=state.mode==='body'?effectiveBodyTarget(sample):null,frameMetrics=state.currentPayload?.summary?.body_metrics||{};
    const hitPct=100*Number(analytics.hit_fraction||0),partnerPct=100*Number(analytics.relation_percentages_of_all_samples?.partner||0),selfPct=100*Number(analytics.relation_percentages_of_all_samples?.self||0),reviewPct=100*Number(analytics.review_required_fraction||0),reviewBeforePct=100*Number(analytics.review_required_fraction_before_auto_resolution??analytics.review_required_fraction??0),autoResolved=Number(analytics.auto_review_resolved_count||0);
    const relationCounts=analytics.relation_counts||{},regionCounts=analytics.region_counts||{},perRelation=analytics.per_relation_region_counts||{};
    const relationTotal=Math.max(1,Number(analytics.sample_count||0));
    const relationSegments=[['partner',relationCounts.partner||0],['self',relationCounts.self||0],['other',relationCounts.other||0],['none',Math.max(0,relationTotal-Object.values(relationCounts).reduce((a,b)=>a+Number(b||0),0))]];
    const selectedRelation=effectiveBody?.resolved?effectiveBody.relation:bodyRelationKey(sample.body_social_region||sample.body_region,sample.body_relation),selectedRegion=bodyRegionKey(effectiveBody?.resolved?effectiveBody.coarseRegion:(sample.body_coarse_region||sample.body_region));
    const regionRows=Object.entries(regionCounts).sort((a,b)=>b[1]-a[1]).slice(0,8);
    const hitTotal=Math.max(1,Number(analytics.hit_count||0));
    const distance=Number(effectiveBody?.resolved?effectiveBody.distance:sample.body_distance_mm),secondDistance=Number(sample.second_hit_distance_mm),partnerDistance=Number(sample.nearest_partner_hit_distance_mm);
    const currentRegion=effectiveBody?.resolved?effectiveBody.region:(sample.body_social_region||sample.body_region),currentRelation=effectiveBody?.resolved?effectiveBody.relation:sample.body_relation,currentConfidence=effectiveBody?.resolved?effectiveBody.confidence:sample.body_confidence,currentStatus=effectiveBody?.resolved?effectiveBody.occlusionStatus:sample.occlusion_status,currentExact=effectiveBody?.resolved?effectiveBody.exactHit:sample.body_exact_hit;
    const currentDetails=sample&&Object.keys(sample).length?`<div class="body-current-target ${effectiveBody?.resolved?'reviewed':''}"><div><span>${effectiveBody?.resolved?'Reviewed gaze sample':'Selected gaze sample'}</span><strong>${esc(bodyTargetLabel(currentRegion,currentRelation))}${effectiveBody?.entity?` · ${esc(effectiveBody.entity)}`:''}</strong></div><div class="body-current-grid"><span><b>Distance</b>${Number.isFinite(distance)?`${(distance/1000).toFixed(2)} m`:'—'}</span><span><b>Intersection</b>${currentExact?'exact':effectiveBody?.geometryConsistent===false?'semantic override':sample.auto_review_resolved?'near-body · auto accepted':'near-body candidate'}</span><span><b>Visibility</b>${esc(String(currentStatus||'—').replaceAll('_',' '))}</span><span><b>Confidence</b>${currentConfidence==null?'—':Number(currentConfidence).toFixed(3)}</span></div>${effectiveBody?.resolved&&effectiveBody.geometryConsistent===false?`<div class="body-occlusion-note"><b>Human semantic correction.</b> The original ray-hit coordinates are not reused for the corrected target.</div>`:sample.self_occlusion?`<div class="body-occlusion-note"><b>Self occludes partner.</b> First hit ${esc(sample.first_hit_social_region||'self body')}${Number.isFinite(distance)?` at ${(distance/1000).toFixed(2)} m`:''}; partner candidate ${esc(sample.nearest_partner_hit_social_region||'partner body')}${Number.isFinite(partnerDistance)?` at ${(partnerDistance/1000).toFixed(2)} m`:''}.</div>`:''}${sample.second_hit_social_region?`<div class="body-second-hit"><span>Second surface</span><strong>${esc(bodyTargetLabel(sample.second_hit_social_region,sample.second_hit_relation))}${Number.isFinite(secondDistance)?` · ${(secondDistance/1000).toFixed(2)} m`:''}</strong></div>`:''}</div>`:'';
    root.innerHTML=`<div class="body-overview-head"><div><div class="section-label">Body AOI visual analytics</div><strong>Whole clip + current gaze evidence</strong></div><span class="status-pill">${Number(analytics.sample_count||0).toLocaleString()} gaze samples</span></div>
      <div class="body-overview-metrics"><div><span>Body hit</span><strong>${hitPct.toFixed(1)}%</strong><small>${Number(analytics.hit_count||0).toLocaleString()} samples</small></div><div><span>Partner</span><strong>${partnerPct.toFixed(1)}%</strong><small>of all samples</small></div><div><span>Self</span><strong>${selfPct.toFixed(1)}%</strong><small>of all samples</small></div><div><span>Review remaining</span><strong>${reviewPct.toFixed(1)}%</strong><small>${Number(analytics.review_required_count||0).toLocaleString()} pending${autoResolved?` · ${autoResolved.toLocaleString()} auto-cleared from ${reviewBeforePct.toFixed(1)}%`:''}</small></div></div>
      <div class="body-relation-bar" title="Body target relation over all gaze samples">${relationSegments.map(([relation,count])=>`<i class="${relation}" style="width:${Math.max(0,100*Number(count)/relationTotal)}%" title="${relation}: ${count}"></i>`).join('')}</div>
      <div class="body-map-pair">${bodyMapSvg('self',perRelation.self||{},selectedRelation,selectedRegion)}${bodyMapSvg('partner',perRelation.partner||{},selectedRelation,selectedRegion)}</div><div class="body-map-note">Left/right follow the viewed person’s anatomy. Opacity represents relative hits within self or partner.</div>
      <div class="body-region-bars"><div class="section-label">Body-region proportions · among body hits</div>${regionRows.map(([region,count])=>`<div class="body-region-row ${selectedRegion===bodyRegionKey(region)?'selected':''}"><span><i style="background:${bodyColor(region)}"></i>${esc(bodyRegionLabel(region))}</span><div><b style="width:${100*Number(count)/hitTotal}%"></b></div><strong>${(100*Number(count)/hitTotal).toFixed(1)}%</strong></div>`).join('')||'<div class="empty-tree">No body hits in this artifact.</div>'}</div>
      ${currentDetails}<div class="body-frame-evidence"><span>Current frame</span><b>${Number(frameMetrics.hit_count||0)} / ${Number(state.currentPayload?.summary?.gaze_sample_count||0)} body hits</b><small>${frameMetrics.self_occlusion_count?`${frameMetrics.self_occlusion_count} self-occluded sample(s)`:'No self occlusion in this frame'}</small></div>`;
  }

  function drawBodyTimeline() {
    const panel=document.querySelector('#body-aoi-timeline-panel'),canvas=document.querySelector('#body-aoi-timeline');if(!panel||!canvas)return;
    const timeline=state.manifest?.body_timeline;const active=Boolean(timeline&&['body','fusion'].includes(state.mode));panel.hidden=!active;if(!active)return;
    const ctx=resizeCanvas(canvas),width=canvas.clientWidth,height=canvas.clientHeight,bins=timeline.bins||[];ctx.fillStyle='#1c2018';ctx.fillRect(0,0,width,height);
    const plotTop=7,plotBottom=height-17,plotHeight=Math.max(1,plotBottom-plotTop),binWidth=width/Math.max(1,bins.length);
    bins.forEach((bin,index)=>{const relation=bin.dominant_relation||bodyRelationKey(bin.dominant_social_region),region=bin.dominant_region||bin.dominant_social_region;const x=index*binWidth;ctx.globalAlpha=bin.sample_count?(.30+.65*Math.max(.15,Number(bin.hit_fraction||0))):.20;ctx.fillStyle=(relationMeta[bodyRelationKey(region,relation)]?.color||bodyColor(region,relation));const barHeight=Math.max(3,plotHeight*Math.max(.08,Number(bin.hit_fraction||0)));ctx.fillRect(x,plotBottom-barHeight,Math.ceil(binWidth)+.5,barHeight);ctx.globalAlpha=1;if(Number(bin.review_fraction||0)>0){ctx.fillStyle='#ef554a';ctx.fillRect(x,plotTop,Math.ceil(binWidth)+.5,Math.max(1,3*Number(bin.review_fraction)));}if(Number(bin.self_occlusion_fraction||0)>0){ctx.fillStyle='#ffbf55';ctx.fillRect(x,plotBottom-2,Math.ceil(binWidth)+.5,2);}});
    const total=Math.max(1,Number(timeline.frame_count||1)),frame=Math.max(0,state.presentedFrame>=0?state.presentedFrame:state.currentFrame);const playX=Math.min(width-1,frame/(total-1||1)*width);ctx.strokeStyle='#ffffff';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(playX,0);ctx.lineTo(playX,height);ctx.stroke();ctx.fillStyle='rgba(240,243,229,.62)';ctx.font='600 8px system-ui';ctx.fillText('0:00',4,height-4);ctx.textAlign='center';ctx.fillText(fmtTime((Number(state.manifest.video?.duration_seconds||0))/2).slice(3),width/2,height-4);ctx.textAlign='right';ctx.fillText(fmtTime(Number(state.manifest.video?.duration_seconds||0)).slice(3),width-4,height-4);ctx.textAlign='left';
    if(state.bodyTimelineHover>=0&&bins[state.bodyTimelineHover]){const bin=bins[state.bodyTimelineHover],x=(state.bodyTimelineHover+.5)*binWidth;ctx.strokeStyle='rgba(255,255,255,.45)';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,height);ctx.stroke();}
  }

  const manualActionLabel={accept_auto:'Accepted automatic',override:'Corrected',exclude:'Excluded',defer:'Deferred',clear:'Cleared'};

  function reviewDecisionResolved(decision) {
    return Boolean(decision && ['accept_auto','override','exclude'].includes(decision.action));
  }

  function pendingReviewSample(row) {
    if(!row || !row.review_required)return false;
    const decision=row.manual_review_decision;
    return !reviewDecisionResolved(decision);
  }

  function firstPendingSampleIndex(payload=state.currentPayload) {
    return (payload?.samples||[]).findIndex(pendingReviewSample);
  }

  function automaticSampleLabel(sample) {
    if(!sample)return 'the automatic target';
    if(state.mode==='body'){
      const region=sample.body_social_region||sample.body_region||'unknown';
      const relation=sample.body_relation||bodyRelationKey(region);
      const entity=String(sample.body_entity_id||'').trim();
      return `${bodyTargetLabel(region,relation)}${entity?` · ${entity}`:''}`;
    }
    const region=sample.face_social_region||sample.face_region||'unknown';
    const entity=displayFaceEntity(sample.face_entity_id||sample.candidate_entity_id||sample.candidate_role,region);
    return `${entity} · ${displayRegion(region)}`;
  }

  function hasOwn(row,key) {
    return Boolean(row&&Object.prototype.hasOwnProperty.call(row,key));
  }

  function effectiveFaceField(sample,field) {
    const key=`effective_${field}`;
    return hasOwn(sample,key)?sample[key]:sample?.[field];
  }

  function effectiveFaceTarget(sample) {
    if(!sample)return {entity:null,region:'unknown',relation:'unknown',confidence:null,status:'unknown',resolved:false,source:'automatic'};
    return {
      entity:effectiveFaceField(sample,'face_entity_id')||null,
      region:effectiveFaceField(sample,'face_social_region')||effectiveFaceField(sample,'face_region')||'unknown',
      relation:effectiveFaceField(sample,'target_relation')||'unknown',
      confidence:effectiveFaceField(sample,'face_aoi_confidence'),
      status:effectiveFaceField(sample,'identity_status')||effectiveFaceField(sample,'assignment_status')||'unknown',
      resolved:Boolean(effectiveFaceField(sample,'manual_review_resolved')||sample.manual_review_resolved),
      source:effectiveFaceField(sample,'source')||'automatic',
      entityResolutionSource:effectiveFaceField(sample,'entity_resolution_source')||'automatic',
    };
  }

  function effectiveBodyField(sample,field) {
    const key=`effective_${field}`;
    return hasOwn(sample,key)?sample[key]:sample?.[field];
  }

  function effectiveBodyTarget(sample) {
    if(!sample)return {entity:null,region:'unknown',rawRegion:'unknown',coarseRegion:'unknown',relation:'none',confidence:null,status:'unknown',resolved:false,source:'automatic'};
    return {
      entity:effectiveBodyField(sample,'body_entity_id')||null,
      region:effectiveBodyField(sample,'body_social_region')||effectiveBodyField(sample,'body_region')||'unknown',
      rawRegion:effectiveBodyField(sample,'body_raw_region')||effectiveBodyField(sample,'body_region')||'unknown',
      coarseRegion:effectiveBodyField(sample,'body_coarse_region')||effectiveBodyField(sample,'body_region')||'unknown',
      relation:effectiveBodyField(sample,'body_relation')||bodyRelationKey(effectiveBodyField(sample,'body_social_region')||effectiveBodyField(sample,'body_region'))||'none',
      confidence:effectiveBodyField(sample,'body_confidence'),
      status:effectiveBodyField(sample,'assignment_status')||'unknown',
      exactHit:Boolean(effectiveBodyField(sample,'body_exact_hit')),
      distance:effectiveBodyField(sample,'body_distance_mm'),
      occlusionStatus:effectiveBodyField(sample,'occlusion_status')||'unknown',
      resolved:Boolean(effectiveBodyField(sample,'manual_review_resolved')||sample.manual_review_resolved),
      source:effectiveBodyField(sample,'source')||'automatic',
      entityResolutionSource:effectiveBodyField(sample,'entity_resolution_source')||'automatic',
      geometryConsistent:effectiveBodyField(sample,'geometry_consistent'),
    };
  }

  function bodyReviewTargetLabel(target) {
    const label=bodyTargetLabel(target?.region,target?.relation);
    return `${label}${target?.entity?` · ${target.entity}`:''}`;
  }

  function manualReviewRoleContext() {
    return state.currentPayload?.manual_review_role_context||state.manifest?.manual_review_role_context||{};
  }

  function normalizedReviewRelation(value) {
    const relation=String(value||'').trim();
    if(relation==='ambiguous_partner'||relation==='partner_candidate')return 'partner';
    return relation;
  }

  function suggestedEntityForRelation(relation) {
    const normalized=normalizedReviewRelation(relation);
    if(normalized!=='partner'&&normalized!=='self')return '';
    const context=manualReviewRoleContext();
    return String(context?.[`${normalized}_entity_id`]||'').trim();
  }

  function refreshReviewEntitySuggestion() {
    const entity=document.querySelector('#face-review-entity');
    const relation=document.querySelector('#face-review-relation');
    if(!entity)return;
    const selected=state.currentPayload?.selected_sample||{};
    const decision=state.currentPayload?.manual_review_decision||selected.manual_review_decision||null;
    const effective=effectiveFaceTarget(selected);
    const requested=relation?.value||decision?.override_relation||effective.relation||selected.target_relation||selected.candidate_relation||'';
    const suggestion=suggestedEntityForRelation(requested);
    entity.placeholder=suggestion?`Auto-resolved ${suggestion} from ${normalizedReviewRelation(requested)} role`:'Keep automatic / candidate entity';
    entity.dataset.suggestedEntity=suggestion;
  }

  function targetLabel(target) {
    const region=target?.region||'unknown';
    return `${displayFaceEntity(target?.entity,region)} · ${displayRegion(region)}`;
  }

  function renderFaceManualReview() {
    const panel=document.querySelector('#face-manual-review');
    if(!panel)return;
    const active=state.mode==='face';
    panel.hidden=!active;
    document.querySelectorAll('.face-review-nav').forEach(button=>button.hidden=!active);
    if(!active)return;
    const manual=state.currentPayload?.manual_review||state.manifest?.manual_review;
    const selected=state.currentPayload?.selected_sample||null;
    const decision=state.currentPayload?.manual_review_decision||selected?.manual_review_decision||null;
    const progress=document.querySelector('#face-review-progress');
    const status=document.querySelector('#face-review-status');
    const queue=document.querySelector('#face-review-queue');
    const automatic=document.querySelector('#face-review-auto');
    if(progress){
      const resolvedUnits=Math.max(0,Number(manual?.total_review_unit_count||0)-Number(manual?.pending_review_unit_count||0));
      progress.textContent=manual?`${resolvedUnits} / ${manual.total_review_unit_count??manual.total_flagged_sample_count??0} review events resolved · ${manual.pending_sample_count||0} samples remain`:'No queue loaded';
    }
    if(status){
      const text=(manual?.status||'not loaded').replaceAll('_',' ');
      status.textContent=text;
      status.className=`status-pill ${manual?.status==='complete'||manual?.status==='not_required'?'good':manual?.status==='in_progress'?'warning':''}`;
    }
    if(queue)queue.innerHTML=manual?`<span><b>${manual.pending_review_unit_count??manual.pending_frame_count??0}</b> pending review events</span><span><b>${manual.pending_frame_count||0}</b> affected frames</span><span><b>${manual.deferred_sample_count||0}</b> deferred samples</span>`:'<span>No review summary.</span>';
    if(automatic){
      if(!selected){
        automatic.innerHTML='<span>Select a gaze sample in the current frame.</span>';
      }else{
        const region=selected.face_social_region||selected.face_region||'unknown';
        const candidate=displayFaceEntity(selected.candidate_entity_id||selected.candidate_role||selected.face_entity_id,region);
        const relation=selected.target_relation||selected.candidate_relation||'unknown';
        const confidence=selected.face_aoi_confidence==null?'—':Number(selected.face_aoi_confidence).toFixed(3);
        const effective=effectiveFaceTarget(selected);
        const effectiveConfidence=effective.confidence==null?'—':Number(effective.confidence).toFixed(3);
        automatic.innerHTML=`<div><span>Automatic target</span><strong>${esc(candidate)} · ${esc(displayRegion(region))}</strong></div><div><span>Automatic relation / confidence</span><strong>${esc(relation)} · ${confidence}</strong></div><div><span>Automatic status</span><strong>${esc(selected.identity_status||selected.assignment_status||'unknown')}</strong></div>${decision?`<div class="face-review-existing"><span>Saved decision</span><strong>${esc(manualActionLabel[decision.action]||decision.action||'—')}</strong></div>`:''}${effective.resolved?`<div class="face-review-effective"><span>Effective reviewed target</span><strong>${esc(targetLabel(effective))}</strong></div><div><span>Reviewed relation / confidence</span><strong>${esc(effective.relation)} · ${effectiveConfidence}</strong></div><div><span>Entity resolution</span><strong>${esc(effective.entityResolutionSource||'automatic')}</strong></div><div><span>Reviewed status</span><strong>${esc(effective.status)}</strong></div>`:''}`;
      }
    }
    const relation=document.querySelector('#face-review-relation');
    const region=document.querySelector('#face-review-region');
    const entity=document.querySelector('#face-review-entity');
    const note=document.querySelector('#face-review-note');
    const reviewer=document.querySelector('#face-review-reviewer');
    const guidance=document.querySelector('#face-review-guidance');
    if(relation)relation.value=decision?.override_relation||'';
    if(region)region.value=decision?.override_region||'';
    if(entity)entity.value=decision?.override_entity_id||decision?.resolved_entity_id||'';
    refreshReviewEntitySuggestion();
    if(note)note.value=decision?.note||'';
    if(reviewer)reviewer.value=decision?.reviewer||reviewer.value||'local reviewer';
    if(guidance){
      const pendingIndex=firstPendingSampleIndex();
      if(!selected){
        guidance.className='face-review-guidance';
        guidance.innerHTML='<strong>No gaze sample selected.</strong><span>Select a red ! sample before reviewing.</span>';
      }else if(decision){
        guidance.className=`face-review-guidance ${decision.action==='defer'?'warning':'good'}`;
        guidance.innerHTML=decision.action==='defer'
          ?`<strong>Sample ${Number(state.currentPayload?.sample_index||0)+1} is deferred.</strong><span>It remains pending and continues to show the automatic target.</span>`
          :`<strong>Decision saved for sample ${Number(state.currentPayload?.sample_index||0)+1}.</strong><span>Samples in frame and the lime Reviewed primary target now show the effective review draft. The source artifact remains unchanged until Create reviewed Face AOI.</span>`;
      }else if(pendingReviewSample(selected)){
        guidance.className='face-review-guidance warning';
        guidance.innerHTML=`<strong>Selected sample ${Number(state.currentPayload?.sample_index||0)+1} is flagged.</strong><span>Choose Relation, Face region or Entity ID and use Save correction. To keep ${esc(automaticSampleLabel(selected))}, use Accept automatic.</span>`;
      }else if(pendingIndex>=0&&pendingIndex!==Number(state.currentPayload?.sample_index||0)){
        guidance.className='face-review-guidance info';
        guidance.innerHTML=`<strong>${esc(automaticSampleLabel(selected))} is already the automatic label for this selected sample.</strong><span>This sample is not the unresolved red ! item. Select sample ${pendingIndex+1} to resolve this frame. The lime Primary target is a frame aggregate, not this selected sample.</span>`;
      }else{
        guidance.className='face-review-guidance info';
        guidance.innerHTML=`<strong>${esc(automaticSampleLabel(selected))} is already the automatic label.</strong><span>Use Accept automatic to record confirmation. Save correction requires at least one changed field.</span>`;
      }
    }
    panel.querySelectorAll('[data-face-review-action]').forEach(button=>{
      button.disabled=!selected;
      button.classList.toggle('selected',Boolean(decision&&button.dataset.faceReviewAction===decision.action));
    });
    const output=document.querySelector('#face-review-output-label');
    if(output&&!output.dataset.touched){
      const base=state.manifest?.artifact?.label||'face_aoi';
      output.value=`${base} reviewed`;
    }
    const apply=document.querySelector('#face-review-apply');
    const help=document.querySelector('#face-review-finalize-help');
    const pending=Number(manual?.pending_sample_count||0);
    const decisions=Number(manual?.decision_count||0);
    if(apply){
      apply.disabled=!manual||pending>0||decisions===0;
      apply.title=pending>0?`${pending} flagged samples still need a final decision`:decisions===0?'Save at least one review decision first':'Create a new immutable reviewed Face AOI artifact';
    }
    if(help){
      help.textContent=pending>0?`${pending} flagged sample${pending===1?'':'s'} remain unresolved. Deferred samples still count as pending.`:decisions?`Review complete. Creating the artifact will preserve the source and apply ${decisions} saved decision${decisions===1?'':'s'}.`:'No flagged sample needs review in this artifact.';
    }
    const badge=document.querySelector('#aoi-review-badge');
    if(badge&&state.manifest){
      badge.textContent=`${modes.face.label} · ${manual?.pending_review_unit_count??manual?.pending_frame_count??0} pending groups · ${manual?.pending_frame_count||0} frames`;
    }
    updateHumanReviewEntry();updateBodyHumanReviewEntry();
  }

  function renderBodyManualReview() {
    const panel=document.querySelector('#body-manual-review');
    if(!panel)return;
    const active=state.mode==='body';
    panel.hidden=!active;
    if(!active)return;
    const manual=state.currentPayload?.manual_review||state.manifest?.manual_review;
    const selected=state.currentPayload?.selected_sample||null;
    const decision=state.currentPayload?.manual_review_decision||selected?.manual_review_decision||null;
    const effective=effectiveBodyTarget(selected);
    const integrity=state.manifest?.body_geometry_integrity||null;
    const reviewAllowed=state.manifest?.body_review_allowed!==false;
    const geometryGate=document.querySelector('#body-review-geometry-gate');
    if(geometryGate){
      const pct=value=>value==null?'—':`${(Number(value)*100).toFixed(1)}%`;
      const entityRows=(integrity?.entities||[]).filter(row=>row.required!==false).map(row=>`<div class="body-review-geometry-row"><strong>${esc(String(row.entity_relation||'person').toUpperCase())} · ${esc(row.entity_id||'unknown')}<small>${esc(row.marker_protocol?.name||'unknown marker profile')}</small></strong><span>body ${esc(pct(row.body_frame_coverage_fraction))}</span><span>torso ${esc(pct(row.torso_coverage_fraction))}</span><span>pelvis ${esc(pct(row.pelvis_coverage_fraction))}</span></div>`).join('');
      const failureRows=(integrity?.failures||[]).slice(0,4).map(row=>`<li>${esc(row.message||String(row.code||'geometry check failed').replaceAll('_',' '))}</li>`).join('');
      geometryGate.className=`body-review-geometry-gate ${reviewAllowed?'passed':'blocked'}`;
      geometryGate.innerHTML=`<div><strong>${reviewAllowed?'Body geometry accepted':'Human review blocked: torso geometry invalid'}</strong><span>${reviewAllowed?'Required self/partner streams passed coverage checks.':'Regenerate Body AOI before making manual decisions; otherwise chest-directed gaze can be mislabelled as head/neck uncertainty.'}</span></div>${entityRows?`<div class="body-review-geometry-rows">${entityRows}</div>`:''}${failureRows?`<ul>${failureRows}</ul>`:''}`;
    }
    const progress=document.querySelector('#body-review-progress');
    const status=document.querySelector('#body-review-status');
    const queue=document.querySelector('#body-review-queue');
    const automatic=document.querySelector('#body-review-auto');
    if(progress){
      const resolvedUnits=Math.max(0,Number(manual?.total_review_unit_count||0)-Number(manual?.pending_review_unit_count||0));
      progress.textContent=manual?`${resolvedUnits} / ${manual.total_review_unit_count??manual.total_flagged_sample_count??0} review events resolved · ${manual.pending_sample_count||0} samples remain`:'No queue loaded';
    }
    if(status){
      const text=(manual?.status||'not loaded').replaceAll('_',' ');
      status.textContent=text;
      status.className=`status-pill ${manual?.status==='complete'||manual?.status==='not_required'?'good':manual?.status==='in_progress'?'warning':''}`;
    }
    if(queue)queue.innerHTML=manual?`<span><b>${manual.pending_review_unit_count??manual.pending_frame_count??0}</b> pending review events</span><span><b>${manual.pending_frame_count||0}</b> affected frames</span><span><b>${manual.deferred_sample_count||0}</b> deferred samples</span>`:'<span>No review summary.</span>';
    if(automatic){
      if(!selected){
        automatic.innerHTML='<span>Select a gaze sample in the current Body AOI frame.</span>';
      }else{
        const region=selected.body_social_region||selected.body_region||'unknown';
        const relation=selected.body_relation||bodyRelationKey(region);
        const confidence=selected.body_confidence==null?'—':Number(selected.body_confidence).toFixed(3);
        const distance=Number(selected.body_distance_mm);
        const effectiveConfidence=effective.confidence==null?'—':Number(effective.confidence).toFixed(3);
        automatic.innerHTML=`<div><span>Automatic target</span><strong>${esc(bodyTargetLabel(region,relation))}</strong></div><div><span>Automatic entity / confidence</span><strong>${esc(selected.body_entity_id||'—')} · ${confidence}</strong></div><div><span>Automatic geometry</span><strong>${selected.body_exact_hit?'exact ray hit':selected.auto_review_resolved?'near-body · auto accepted':'near-body candidate'}${Number.isFinite(distance)?` · ${(distance/1000).toFixed(2)} m`:''} · ${esc(String(selected.occlusion_status||'unknown').replaceAll('_',' '))}</strong></div>${decision?`<div class="face-review-existing"><span>Saved decision</span><strong>${esc(manualActionLabel[decision.action]||decision.action||'—')}</strong></div>`:''}${effective.resolved?`<div class="face-review-effective"><span>Effective reviewed target</span><strong>${esc(bodyReviewTargetLabel(effective))}</strong></div><div><span>Reviewed confidence / status</span><strong>${effectiveConfidence} · ${esc(String(effective.status||'unknown').replaceAll('_',' '))}</strong></div><div><span>Entity resolution</span><strong>${esc(effective.entityResolutionSource||'automatic')}</strong></div><div><span>Geometry evidence</span><strong>${effective.geometryConsistent===false?'semantic correction; automatic hit geometry cleared':'consistent with automatic hit geometry'}</strong></div>`:''}`;
      }
    }
    const relation=document.querySelector('#body-review-relation');
    const region=document.querySelector('#body-review-region');
    const entity=document.querySelector('#body-review-entity');
    const note=document.querySelector('#body-review-note');
    const reviewer=document.querySelector('#body-review-reviewer');
    const guidance=document.querySelector('#body-review-guidance');
    if(relation)relation.value=decision?.override_relation||'';
    if(region)region.value=decision?.override_region||'';
    if(entity){
      entity.value=decision?.override_entity_id||decision?.resolved_entity_id||'';
      const requested=relation?.value||decision?.override_relation||selected?.body_relation||'';
      const suggestion=suggestedEntityForRelation(requested);
      entity.placeholder=suggestion?`Auto-resolved ${suggestion} from ${requested} role`:'Keep automatic / body entity';
      entity.dataset.suggestedEntity=suggestion;
    }
    if(note)note.value=decision?.note||'';
    if(reviewer)reviewer.value=decision?.reviewer||reviewer.value||'local reviewer';
    panel.querySelectorAll('select,input,textarea').forEach(control=>{if(control.id!=='body-review-output-label')control.disabled=!reviewAllowed;});
    if(guidance){
      const pendingIndex=firstPendingSampleIndex();
      if(!reviewAllowed){
        guidance.className='face-review-guidance warning';
        guidance.innerHTML='<strong>Do not review this Body AOI.</strong><span>Its torso/body geometry failed acceptance. Return to Gaze Targets and regenerate the artifact after fixing C3D assignment or marker coverage.</span>';
      }else if(!selected){
        guidance.className='face-review-guidance';
        guidance.innerHTML='<strong>No gaze sample selected.</strong><span>Select a red ! Body AOI sample before reviewing.</span>';
      }else if(decision){
        guidance.className=`face-review-guidance ${decision.action==='defer'?'warning':'good'}`;
        guidance.innerHTML=decision.action==='defer'
          ?`<strong>Sample ${Number(state.currentPayload?.sample_index||0)+1} is deferred.</strong><span>It remains pending and continues to show the automatic Body AOI target.</span>`
          :`<strong>Decision saved for sample ${Number(state.currentPayload?.sample_index||0)+1}.</strong><span>The cyan reviewed target is previewed immediately. The automatic artifact remains unchanged until Create reviewed Body AOI.</span>`;
      }else if(pendingReviewSample(selected)){
        guidance.className='face-review-guidance warning';
        guidance.innerHTML=`<strong>Selected sample ${Number(state.currentPayload?.sample_index||0)+1} is flagged.</strong><span>Correct Relation, Body region or Entity ID, mark it no_body, or accept ${esc(automaticSampleLabel(selected))}.</span>`;
      }else if(pendingIndex>=0&&pendingIndex!==Number(state.currentPayload?.sample_index||0)){
        guidance.className='face-review-guidance info';
        guidance.innerHTML=`<strong>${esc(automaticSampleLabel(selected))} is not the unresolved sample.</strong><span>Select sample ${pendingIndex+1}, which carries the red ! review flag.</span>`;
      }else{
        guidance.className='face-review-guidance info';
        guidance.innerHTML=`<strong>${esc(automaticSampleLabel(selected))} is the automatic label.</strong><span>Use Accept automatic to record confirmation. Save correction requires at least one changed field.</span>`;
      }
    }
    panel.querySelectorAll('[data-body-review-action]').forEach(button=>{
      button.disabled=!reviewAllowed||!selected;
      button.classList.toggle('selected',Boolean(decision&&button.dataset.bodyReviewAction===decision.action));
    });
    const output=document.querySelector('#body-review-output-label');
    if(output&&!output.dataset.touched){
      const base=state.manifest?.artifact?.label||'body_aoi';
      output.value=`${base} reviewed`;
    }
    const apply=document.querySelector('#body-review-apply');
    const help=document.querySelector('#body-review-finalize-help');
    const pending=Number(manual?.pending_sample_count||0);
    const decisions=Number(manual?.decision_count||0);
    if(apply){
      apply.disabled=!reviewAllowed||!manual||pending>0||decisions===0;
      apply.title=!reviewAllowed?'Body geometry did not pass torso acceptance; regenerate before review':pending>0?`${pending} flagged Body AOI samples still need a final decision`:decisions===0?'Save at least one Body AOI review decision first':'Create a new immutable reviewed Body AOI artifact';
    }
    if(help){
      help.textContent=!reviewAllowed?'This artifact is diagnosis-only because torso/body geometry did not pass acceptance. Regenerate Body AOI; do not finalize human labels from this result.':pending>0?`${pending} flagged Body AOI sample${pending===1?'':'s'} remain unresolved. Deferred samples still count as pending.`:decisions?`Body review complete. Creating the artifact will preserve the source and apply ${decisions} saved decision${decisions===1?'':'s'}.`:'No flagged Body AOI sample needs review in this artifact.';
    }
    const badge=document.querySelector('#aoi-review-badge');
    if(badge&&state.manifest)badge.textContent=`${modes.body.label} · ${manual?.pending_review_unit_count??manual?.pending_frame_count??0} pending groups · ${manual?.pending_frame_count||0} frames`;
    updateBodyHumanReviewEntry();
  }

  function renderManualReview() {
    document.querySelectorAll('.manual-review-nav').forEach(button=>button.hidden=!['face','body'].includes(state.mode));
    renderFaceManualReview();
    renderBodyManualReview();
  }

  async function saveManualDecision(action) {
    if(state.mode!=='face'||!state.currentPayload?.selected_sample)return;
    const selected=state.currentPayload.selected_sample;
    const requestedRelation=document.querySelector('#face-review-relation')?.value||null;
    const typedEntity=document.querySelector('#face-review-entity')?.value.trim()||'';
    const inferredRelation=normalizedReviewRelation(requestedRelation||selected.target_relation||selected.candidate_relation||'');
    const suggestedEntity=suggestedEntityForRelation(inferredRelation);
    const body={
      video_artifact_id:state.videoArtifactId,
      output_frame_index:state.currentPayload.output_frame_index,
      gaze_sample_uid:selected.gaze_sample_uid||null,
      apply_to:document.querySelector('#face-review-apply-to')?.value||'sample',
      action,
      override_relation:requestedRelation,
      override_entity_id:typedEntity||(action==='override'&&['partner','self'].includes(inferredRelation)?suggestedEntity:null),
      override_region:document.querySelector('#face-review-region')?.value||null,
      note:document.querySelector('#face-review-note')?.value.trim()||'',
      reviewer:document.querySelector('#face-review-reviewer')?.value.trim()||'local reviewer',
      confidence_threshold:state.confidenceThreshold,
    };
    if(action==='override'&&!body.override_relation&&!body.override_entity_id&&!body.override_region){
      const pendingIndex=firstPendingSampleIndex();
      if(!pendingReviewSample(selected)&&pendingIndex>=0&&pendingIndex!==state.sampleIndex){
        toast('Select the flagged gaze sample',`Sample ${state.sampleIndex+1} is already ${automaticSampleLabel(selected)}. Moving to red ! sample ${pendingIndex+1}.`);
        await loadFrame(state.currentFrame,pendingIndex);
      }else{
        toast('Choose a correction','Set Relation, Face region or Entity ID. To keep the automatic target, use Accept automatic.');
      }
      return;
    }
    const buttons=[...document.querySelectorAll('[data-face-review-action]')];
    buttons.forEach(button=>button.disabled=true);
    try{
      const result=await api(`/api/aoi-review/${encodeURIComponent(state.sessionId)}/face/${encodeURIComponent(state.artifactId)}/decision`,{method:'POST',body});
      if(state.manifest){
        state.manifest.manual_review=result.manual_review;
        state.manifest.event_frames=result.event_frames;
        state.manifest.event_counts=Object.fromEntries(Object.entries(result.event_frames||{}).map(([key,value])=>[key,(value||[]).length]));
      }
      clearFrameCaches();
      await loadFrame(state.currentFrame,state.sampleIndex);
      const count=(result.changed_sample_uids||[]).length;
      toast(action==='clear'?'Review decision cleared':'Face AOI review saved',`${count} gaze sample${count===1?'':'s'} updated. ${result.manual_review?.pending_sample_count||0} remain pending.`);
    }catch(error){showError(error,'Could not save Face AOI review decision');}
    finally{buttons.forEach(button=>button.disabled=!state.currentPayload?.selected_sample);renderManualReview();}
  }

  async function applyManualReview() {
    const manual=state.manifest?.manual_review;
    if(state.mode!=='face'||!manual)return;
    if(Number(manual.pending_sample_count||0)>0){toast('Review is incomplete',`${manual.pending_sample_count} flagged samples remain pending.`);return;}
    const button=document.querySelector('#face-review-apply');
    if(button){button.disabled=true;button.textContent='Queueing reviewed artifact…';}
    try{
      await api('/api/jobs/face-aoi-manual-review',{method:'POST',body:{session_id:state.sessionId,artifact_id:state.artifactId,label:document.querySelector('#face-review-output-label')?.value.trim()||null,require_complete:true}});
      toast('Reviewed Face AOI queued','A new immutable Face AOI artifact will be created from the saved human decisions; the source artifact remains unchanged.');
    }catch(error){showError(error,'Could not create reviewed Face AOI');}
    finally{if(button?.isConnected){button.textContent='Create reviewed Face AOI';renderManualReview();}}
  }

  async function saveBodyManualDecision(action) {
    if(state.mode!=='body'||!state.currentPayload?.selected_sample)return;
    if(state.manifest?.body_review_allowed===false){toast('Body review blocked','Regenerate Body AOI after torso geometry passes acceptance.');return;}
    const selected=state.currentPayload.selected_sample;
    const requestedRelation=document.querySelector('#body-review-relation')?.value||null;
    const requestedRegion=document.querySelector('#body-review-region')?.value||null;
    const typedEntity=document.querySelector('#body-review-entity')?.value.trim()||'';
    const inferredRelation=requestedRelation||selected.body_relation||bodyRelationKey(selected.body_social_region||selected.body_region);
    const suggestedEntity=suggestedEntityForRelation(inferredRelation);
    const body={
      video_artifact_id:state.videoArtifactId,
      output_frame_index:state.currentPayload.output_frame_index,
      gaze_sample_uid:selected.gaze_sample_uid||null,
      apply_to:document.querySelector('#body-review-apply-to')?.value||'review_episode',
      action,
      override_relation:requestedRelation,
      override_entity_id:typedEntity||(action==='override'&&['partner','self'].includes(inferredRelation)?suggestedEntity:null),
      override_region:requestedRegion,
      note:document.querySelector('#body-review-note')?.value.trim()||'',
      reviewer:document.querySelector('#body-review-reviewer')?.value.trim()||'local reviewer',
      confidence_threshold:state.confidenceThreshold,
    };
    if(action==='override'&&!body.override_relation&&!body.override_entity_id&&!body.override_region){
      const pendingIndex=firstPendingSampleIndex();
      if(!pendingReviewSample(selected)&&pendingIndex>=0&&pendingIndex!==state.sampleIndex){
        toast('Select the flagged Body AOI sample',`Moving to red ! sample ${pendingIndex+1}.`);
        await loadFrame(state.currentFrame,pendingIndex);
      }else{
        toast('Choose a Body AOI correction','Set Relation, Body region or Entity ID. To keep the automatic target, use Accept automatic.');
      }
      return;
    }
    const buttons=[...document.querySelectorAll('[data-body-review-action]')];
    buttons.forEach(button=>button.disabled=true);
    try{
      const result=await api(`/api/aoi-review/${encodeURIComponent(state.sessionId)}/body/${encodeURIComponent(state.artifactId)}/decision`,{method:'POST',body});
      if(state.manifest){
        state.manifest.manual_review=result.manual_review;
        state.manifest.event_frames=result.event_frames;
        state.manifest.event_counts=Object.fromEntries(Object.entries(result.event_frames||{}).map(([key,value])=>[key,(value||[]).length]));
      }
      clearFrameCaches();
      await loadFrame(state.currentFrame,state.sampleIndex);
      const count=(result.changed_sample_uids||[]).length;
      toast(action==='clear'?'Body AOI decision cleared':'Body AOI review saved',`${count} gaze sample${count===1?'':'s'} updated. ${result.manual_review?.pending_sample_count||0} remain pending.`);
    }catch(error){showError(error,'Could not save Body AOI review decision');}
    finally{buttons.forEach(button=>button.disabled=!state.currentPayload?.selected_sample);renderManualReview();}
  }

  async function applyBodyManualReview() {
    if(state.manifest?.body_review_allowed===false){toast('Body review blocked','This artifact is diagnosis-only. Regenerate Body AOI after torso geometry passes acceptance.');return;}
    const manual=state.manifest?.manual_review;
    if(state.mode!=='body'||!manual)return;
    if(Number(manual.pending_sample_count||0)>0){toast('Body review is incomplete',`${manual.pending_sample_count} flagged samples remain pending.`);return;}
    const button=document.querySelector('#body-review-apply');
    if(button){button.disabled=true;button.textContent='Queueing reviewed artifact…';}
    try{
      await api('/api/jobs/body-aoi-manual-review',{method:'POST',body:{session_id:state.sessionId,artifact_id:state.artifactId,label:document.querySelector('#body-review-output-label')?.value.trim()||null,require_complete:true}});
      toast('Reviewed Body AOI queued','A new immutable Body AOI artifact will be created from the saved human decisions; the automatic source artifact remains unchanged.');
    }catch(error){showError(error,'Could not create reviewed Body AOI');}
    finally{if(button?.isConnected){button.textContent='Create reviewed Body AOI';renderManualReview();}}
  }

  function renderInspector() {
    const payload=state.currentPayload;
    const summary=document.querySelector('#aoi-review-summary');
    const samplesRoot=document.querySelector('#aoi-review-samples');
    if(!summary||!samplesRoot)return;
    if(!payload){
      summary.innerHTML='<div class="empty-tree">No frame loaded.</div>';
      samplesRoot.innerHTML='';
      renderManualReview();
      return;
    }
    const s=payload.summary||{};
    const automaticSummary=payload.automatic_summary||s;
    const usesReviewDraft=['face','body'].includes(state.mode)&&Boolean(payload.summary_uses_manual_review);
    const selected=payload.selected_sample||{};
    const selectedEffective=state.mode==='face'?effectiveFaceTarget(selected):state.mode==='body'?effectiveBodyTarget(selected):null;
    const orientationNote=state.mode==='face'&&(selected.anatomical_orientation_source||selected.canonical_bilateral_convention)?`<div class="aoi-review-distribution"><span><b>Landmark geometry</b> ${selected.source_pixel_geometry_preserved?'source pixels preserved':(selected.landmark_geometry_mirror_applied?'mirrored':'as exported')}</span><span><b>Bilateral semantics</b> ${esc(selected.canonical_bilateral_convention||'legacy')} ${selected.bilateral_label_swap_applied?'· names exchanged':'· names kept'}</span><span><b>Source convention</b> ${esc(selected.source_bilateral_convention||selected.anatomical_orientation_source||'unknown')} · ${Number(selected.anatomical_orientation_confidence||0).toFixed(2)}</span></div>`:'';
    const selectedDecision=payload.manual_review_decision;
    const automaticPrimaryRegion=automaticSummary.primary_region||'no_gaze';
    const effectivePrimaryRegion=s.primary_region||'no_gaze';
    const automaticPrimaryEntity=state.mode==='face'?displayFaceEntity(automaticSummary.primary_entity_id,automaticPrimaryRegion):(automaticSummary.primary_entity_id||'—');
    const effectivePrimaryEntity=state.mode==='face'?displayFaceEntity(s.primary_entity_id,effectivePrimaryRegion):(s.primary_entity_id||'—');
    const automaticPrimary=`${automaticPrimaryEntity} · ${displayRegion(automaticPrimaryRegion)}`;
    const effectivePrimary=`${effectivePrimaryEntity} · ${displayRegion(effectivePrimaryRegion)}`;
    const primaryComparison=usesReviewDraft?`<div class="aoi-review-review-draft"><span><b>Review draft applied</b> ${Number(payload.resolved_decision_count_in_frame||0)} resolved sample${Number(payload.resolved_decision_count_in_frame||0)===1?'':'s'}</span><span><b>Automatic primary</b> ${esc(automaticPrimary)}</span></div>`:'';
    const reviewedTargetLabel=state.mode==='body'?bodyReviewTargetLabel(selectedEffective):targetLabel(selectedEffective);
    const decisionCard=selectedDecision?`<div class="success-card compact"><strong>Saved human decision</strong><p>${esc(manualActionLabel[selectedDecision.action]||selectedDecision.action)}${selectedEffective?.resolved?` → ${esc(reviewedTargetLabel)}`:''}${selectedDecision.note?` · ${esc(selectedDecision.note)}`:''}</p></div>`:'';
    const reviewWarning=s.review_required?'<div class="warning-card compact"><strong>Review required</strong><p>At least one sample in this frame is still low-confidence, ambiguous, conflicting or deferred.</p></div>':'';
    const coverage=payload.face_coverage_diagnostic;
    const coverageReasons=(coverage?.filter_reason_details||Object.entries(coverage?.filter_reason_counts||{}).map(([reason,count])=>({reason,count,explanation:''}))).slice(0,4);
    const coverageStatus=Object.entries(coverage?.filter_status_counts||{}).sort((a,b)=>b[1]-a[1]);
    const coverageMetrics=[];
    if(coverage?.identity_confidence!=null)coverageMetrics.push(`identity ${Number(coverage.identity_confidence).toFixed(2)}`);
    if(coverage?.projected_box_iou!=null)coverageMetrics.push(`2D/3D IoU ${Number(coverage.projected_box_iou).toFixed(3)}`);
    if(coverage?.normalized_center_error!=null)coverageMetrics.push(`centre error ${Number(coverage.normalized_center_error).toFixed(2)}× head scale`);
    if(coverage?.face_tolerance_ms!=null)coverageMetrics.push(`time gate ${Number(coverage.face_tolerance_ms).toFixed(1)} ms`);
    const coverageCard=state.mode==='face'&&coverage?`<div class="face-coverage-diagnostic ${esc(coverage.severity||'warning')}"><div><span>FaceMapper coverage</span><strong>${esc(coverage.title||coverage.code||'Coverage diagnostic')}</strong></div><p>${esc(coverage.detail||'')}</p>${coverage.recommendation?`<p class="face-coverage-recommendation"><b>What to check:</b> ${esc(coverage.recommendation)}</p>`:''}<div class="face-coverage-tags">${coverage.partner_projection_valid?'<span>3D partner projection present</span>':'<span>No valid partner projection</span>'}${coverageStatus.map(([key,value])=>`<span>${esc(key)} ${value}</span>`).join('')}${coverageMetrics.map(value=>`<span>${esc(value)}</span>`).join('')}</div>${coverageReasons.length?`<div class="face-coverage-reasons">${coverageReasons.map(row=>`<div><b>${esc(String(row.reason||'').replaceAll('_',' '))}${Number(row.count||0)>1?` ×${Number(row.count)}`:''}</b><span>${esc(row.explanation||'')}</span></div>`).join('')}</div>`:''}</div>`:'';
    summary.innerHTML=`<div class="section-label">Current frame</div><div class="aoi-review-stat-grid">
      <div><span>Video frame</span><strong id="aoi-live-video-frame">${payload.output_frame_index}</strong></div><div><span>Native frame</span><strong id="aoi-live-native-frame">${payload.native_frame_index}</strong></div>
      <div><span>Samples</span><strong id="aoi-live-sample-count">${s.gaze_sample_count||0}</strong></div><div><span>Mean confidence</span><strong id="aoi-live-confidence">${Number(s.mean_confidence||0).toFixed(3)}</strong></div>
    </div><div class="aoi-review-primary ${usesReviewDraft?'reviewed':''}"><span>${usesReviewDraft?'Reviewed frame primary target':'Frame primary target'}</span><strong id="aoi-live-primary-target">${esc(effectivePrimary)}</strong></div>
    ${primaryComparison}<div class="aoi-review-distribution">${Object.entries(s.distribution||{}).sort((a,b)=>b[1]-a[1]).map(([k,v])=>`<span><b>${esc(k)}</b> ${v}</span>`).join('')||'<span>No gaze samples</span>'}</div>${orientationNote}
    ${(payload.face_identities||[]).length?`<div class="aoi-review-distribution">${(payload.face_identities||[]).map(row=>{const candidate=row.target_relation==='ambiguous_partner'||row.candidate_relation==='partner';const entity=row.matched_entity_id||row.candidate_entity_id||'other_person';const relation=candidate?'partner_candidate':(row.target_relation||'other_person');return `<span><b>${esc(entity)}</b> ${esc(relation)} · ${row.identity_confidence==null?'—':Number(row.identity_confidence).toFixed(2)}</span>`}).join('')}</div>`:''}
    ${coverageCard}${decisionCard}${reviewWarning}`;
    const rows=payload.samples||[];
    samplesRoot.innerHTML=`<div class="section-label">Samples in frame</div>${rows.length?rows.map((row,index)=>{
      const faceEffective=state.mode==='face'?effectiveFaceTarget(row):null;
      const bodyEffective=state.mode==='body'?effectiveBodyTarget(row):null;
      const effective=faceEffective||bodyEffective;
      const resolved=Boolean(effective?.resolved);
      const region=state.mode==='fusion'?row.final_target_region:state.mode==='body'?(resolved?bodyEffective.region:(row.body_social_region||row.body_region)):(resolved?faceEffective.region:(row.face_social_region||row.face_region));
      const entity=state.mode==='fusion'?row.final_entity_id:state.mode==='body'?(resolved?bodyEffective.entity:row.body_entity_id):(resolved?faceEffective.entity:row.face_entity_id);
      const confidence=state.mode==='fusion'?row.final_confidence:state.mode==='body'?(resolved?bodyEffective.confidence:row.body_confidence):(resolved?faceEffective.confidence:row.face_aoi_confidence);
      const statusText=state.mode==='fusion'?row.fusion_status:(resolved?effective.status:(state.mode==='face'?(row.identity_status||row.assignment_status):row.assignment_status));
      const decision=row.manual_review_decision;
      const reviewMark=decision?`<i class="manual ${decision.action==='defer'?'deferred':'resolved'}" title="${esc(manualActionLabel[decision.action]||decision.action)}">${decision.action==='defer'?'…':'✓'}</i>`:row.review_required?'<i title="Review required">!</i>':'';
      const automaticRegion=state.mode==='body'?(row.body_social_region||row.body_region||'unknown'):(row.face_social_region||row.face_region||'unknown');
      const automaticTarget=state.mode==='body'?`${bodyTargetLabel(automaticRegion,row.body_relation)}${row.body_entity_id?` · ${row.body_entity_id}`:''}`:`${displayFaceEntity(row.face_entity_id||row.candidate_entity_id||row.candidate_role,automaticRegion)} · ${displayRegion(automaticRegion)}`;
      const reviewedTarget=state.mode==='body'?bodyReviewTargetLabel(bodyEffective):targetLabel(faceEffective);
      const audit=['face','body'].includes(state.mode)&&resolved?`<span class="aoi-review-automatic-label">automatic: ${esc(automaticTarget)}</span>`:'';
      const primaryText=state.mode==='body'?`${bodyTargetLabel(region,bodyEffective?.relation||row.body_relation)}${entity?` · ${entity}`:''}`:`${state.mode==='face'?displayFaceEntity(entity,region):(entity||'—')} · ${displayRegion(region||'unknown')}`;
      return `<button class="aoi-review-sample-row ${index===payload.sample_index?'active':''} ${decision?'has-manual-decision':''} ${resolved?'uses-reviewed-result':''}" data-review-sample="${index}" title="${resolved?`Reviewed: ${esc(reviewedTarget)} | Automatic: ${esc(automaticTarget)}`:''}"><span>${index+1}</span><div><strong>${esc(primaryText)}</strong><small>${esc(statusText||'')} · ${confidence==null?'—':Number(confidence).toFixed(3)}${decision?` · ${esc(manualActionLabel[decision.action]||decision.action)}`:''}${audit}</small></div>${reviewMark}</button>`;
    }).join(''):'<div class="empty-tree">No gaze samples mapped to this video frame.</div>'}`;
    samplesRoot.querySelectorAll('[data-review-sample]').forEach(button=>button.onclick=()=>loadFrame(payload.output_frame_index,Number(button.dataset.reviewSample)));
    const position=document.querySelector('#aoi-review-sample-position');
    if(position)position.textContent=rows.length?`${payload.sample_index+1} / ${rows.length}`:'0 / 0';
    renderManualReview();
  }

  function updatePlaybackStatus(text, tone='') {
    state.playbackStatus=String(text||'');
    const node=document.querySelector('#aoi-review-playback-status');
    if(node){node.textContent=state.playbackStatus;node.dataset.tone=tone||'';}
  }

  function updatePlaybackInspectorReadout() {
    const payload=state.currentPayload||{};
    const summary=payload.summary||{};
    const selected=payload.selected_sample||{};
    const values={
      '#aoi-live-video-frame':payload.output_frame_index,
      '#aoi-live-native-frame':payload.native_frame_index,
      '#aoi-live-sample-count':summary.gaze_sample_count??payload.sample_count??0,
      '#aoi-live-confidence':Number(summary.mean_confidence||0).toFixed(3),
    };
    for(const [selector,value] of Object.entries(values)){
      const node=document.querySelector(selector);if(node)node.textContent=value==null?'—':String(value);
    }
    const primary=document.querySelector('#aoi-live-primary-target');
    if(primary){
      if(state.mode==='body')primary.textContent=`${bodyTargetLabel(selected.body_social_region||selected.body_region,selected.body_relation)}${selected.body_entity_id?` · ${selected.body_entity_id}`:''}`;
      else if(state.mode==='face')primary.textContent=`${displayFaceEntity(selected.face_entity_id||selected.candidate_entity_id||selected.candidate_role,selected.face_social_region||selected.face_region)} · ${displayRegion(selected.face_social_region||selected.face_region||'unknown')}`;
      else primary.textContent=`${selected.final_entity_id||'—'} · ${displayRegion(selected.final_target_region||'unknown')}`;
    }
  }

  function renderPlaybackPayload() {
    // Keep the presented video and its scientific overlays in lock-step.  The
    // large analytics/inspector DOM is intentionally frozen while playing and
    // refreshed once on pause, eliminating the main-thread stall that made the
    // preview run far below the source frame rate. Only the tiny frame readout
    // is updated during playback so users can verify that video and evidence
    // are advancing together.
    drawVideoOverlay();
    drawWorldInset();
    drawBodyTimeline();
    updatePlaybackInspectorReadout();
    state.playbackFramesRendered+=1;
    state.playbackLastFrameAt=performance.now();
  }

  function renderPayload() { drawVideoOverlay();drawWorldInset();drawBodyTimeline();renderBodyOverview();renderInspector(); }

  function cacheFramePayload(payload) {
    if(!payload||payload.output_frame_index==null)return;
    const frame=Number(payload.output_frame_index);
    const compact=payload.payload_profile==='compact_playback';
    const cache=compact?state.playbackFrameCache:state.frameCache;
    const order=compact?state.playbackFrameCacheOrder:state.frameCacheOrder;
    cache.set(frame,payload);
    const priorIndex=order.indexOf(frame);
    if(priorIndex>=0)order.splice(priorIndex,1);
    order.push(frame);
    const maximum=compact?240:72;
    while(order.length>maximum){
      const expired=order.shift();
      cache.delete(expired);
    }
  }

  function clearFrameCaches() {
    state.frameCache.clear();
    state.frameCacheOrder=[];
    state.playbackFrameCache.clear();
    state.playbackFrameCacheOrder=[];
    state.playbackBufferedStart=-1;
    state.playbackBufferedEnd=-1;
  }

  function applyFramePayload(payload, sampleIndex=null, autoSelectPending=true, playback=false) {
    if(!payload)return;
    if(!playback&&sampleIndex==null&&autoSelectPending&&['face','body'].includes(state.mode)){
      const pendingIndex=firstPendingSampleIndex(payload);
      if(pendingIndex>=0&&pendingIndex!==payload.sample_index){
        payload.sample_index=pendingIndex;
        payload.selected_sample=payload.samples[pendingIndex];
        payload.manual_review_decision=payload.selected_sample?.manual_review_decision||null;
      }
    }
    state.currentPayload=payload;state.currentFrame=Number(payload.output_frame_index);state.sampleIndex=Number(payload.sample_index||0);
    cacheFramePayload(payload);
    if(playback)renderPlaybackPayload();else renderPayload();
  }

  function frameRequestUrl(frameIndex,sampleIndex=0) {
    const trailFrames=Math.max(1,Math.round(Number(state.manifest?.video?.fps||30)*0.25));
    return `/api/aoi-review/${encodeURIComponent(state.sessionId)}/${state.mode}/${encodeURIComponent(state.artifactId)}/frame?video_artifact_id=${encodeURIComponent(state.videoArtifactId)}&output_frame_index=${Math.max(0,frameIndex)}&sample_index=${Math.max(0,Number(sampleIndex)||0)}&trail_frames=${trailFrames}&confidence_threshold=${state.confidenceThreshold}`;
  }

  function frameWindowRequestUrl(startFrame,frameCount) {
    // Playback windows keep only a short trail.  Long per-frame trails make the
    // buffer several times larger and can starve real-time overlay delivery.
    const trailFrames=Math.max(2,Math.min(6,Math.round(Number(state.manifest?.video?.fps||30)*0.08)));
    return `/api/aoi-review/${encodeURIComponent(state.sessionId)}/${state.mode}/${encodeURIComponent(state.artifactId)}/frame-window?video_artifact_id=${encodeURIComponent(state.videoArtifactId)}&start_frame_index=${Math.max(0,startFrame)}&frame_count=${Math.max(1,frameCount)}&trail_frames=${trailFrames}&confidence_threshold=${state.confidenceThreshold}&compact_playback=true`;
  }

  async function fetchFramePayload(frameIndex,sampleIndex=0) {
    return api(frameRequestUrl(frameIndex,sampleIndex));
  }

  async function fetchFrameWindow(startFrame,frameCount) {
    return api(frameWindowRequestUrl(startFrame,frameCount));
  }

  function playbackWindowSize() {
    const fps=Math.max(1,Number(state.manifest?.video?.fps||30));
    // Compact packets make a short rolling buffer sufficient.  A 0.7-second
    // window starts quickly and is continuously replenished instead of blocking
    // the Play action on up to 120 full inspector frames.
    return Math.max(18,Math.min(36,Math.round(fps*1.0)));
  }

  function stopPlaybackFrameRequests() {
    state.playbackGeneration+=1;
    state.playbackQueuedFrame=null;
    state.playbackRequestInFlight=false;
    state.playbackPromise=null;
    state.playbackBufferedStart=-1;
    state.playbackBufferedEnd=-1;
  }

  function primePlaybackBuffer(frameIndex) {
    if(!state.manifest)return Promise.resolve();
    const total=Math.max(1,Number(state.manifest.video.frame_count||1));
    const frame=Math.max(0,Math.min(total-1,Number(frameIndex)||0));
    state.playbackQueuedFrame=frame;
    return pumpPlaybackBuffer();
  }

  function pumpPlaybackBuffer() {
    if(!state.manifest)return Promise.resolve();
    if(state.playbackRequestInFlight)return state.playbackPromise||Promise.resolve();
    const generation=state.playbackGeneration;
    state.playbackRequestInFlight=true;
    const task=(async()=>{
      try{
        while(generation===state.playbackGeneration&&state.playbackQueuedFrame!=null){
          const total=Math.max(1,Number(state.manifest.video.frame_count||1));
          const target=Math.max(0,Math.min(total-1,Number(state.playbackQueuedFrame)||0));
          state.playbackQueuedFrame=null;
          if(state.playbackFrameCache.has(target)&&target>=state.playbackBufferedStart&&target<state.playbackBufferedEnd)continue;
          const count=playbackWindowSize();
          const overlap=Math.max(2,Math.round(Number(state.manifest.video.fps||30)*0.12));
          const start=Math.max(0,Math.min(target-overlap,Math.max(0,total-count)));
          const result=await fetchFrameWindow(start,count);
          if(generation!==state.playbackGeneration)return;
          for(const payload of result.frames||[])cacheFramePayload(payload);
          state.playbackBufferedStart=Number(result.start_frame_index??start);
          state.playbackBufferedEnd=Number(result.end_frame_index_exclusive??(start+(result.frames||[]).length));
          const current=Number(state.presentedFrame);
          const currentPayload=state.playbackFrameCache.get(current)||state.frameCache.get(current);
          const video=document.querySelector('#aoi-review-video-element');
          if(currentPayload&&video&&!video.paused){applyFramePayload(currentPayload,null,false,true);updatePlaybackStatus('Playing · live 3D overlay','live');}
          if(!currentPayload&&current>=0&&video&&!video.paused)state.playbackQueuedFrame=current;
        }
      }catch(error){
        if(generation===state.playbackGeneration)showError(error,'Could not synchronize AOI overlay');
      }finally{
        if(generation===state.playbackGeneration){
          state.playbackRequestInFlight=false;
          state.playbackPromise=null;
        }
      }
    })();
    state.playbackPromise=task;
    return task.then(()=>{
      if(generation===state.playbackGeneration&&state.playbackQueuedFrame!=null)return pumpPlaybackBuffer();
    });
  }

  async function ensurePlaybackReady(frameIndex) {
    if(!state.manifest)return;
    const total=Math.max(1,Number(state.manifest.video.frame_count||1));
    const fps=Math.max(1,Number(state.manifest.video.fps||30));
    const frame=Math.max(0,Math.min(total-1,Number(frameIndex)||0));
    const readyAhead=Math.min(total,frame+Math.max(8,Math.round(fps*0.35)));
    if(state.playbackFrameCache.has(frame)&&state.playbackBufferedStart<=frame&&state.playbackBufferedEnd>=readyAhead)return;
    await primePlaybackBuffer(frame);
  }

  async function requestPlaybackFrame(frameIndex) {
    if(!state.manifest)return;
    const total=Math.max(1,Number(state.manifest.video.frame_count||1));
    const frame=Math.max(0,Math.min(total-1,Number(frameIndex)||0));
    state.presentedFrame=frame;
    const cached=state.playbackFrameCache.get(frame)||state.frameCache.get(frame);
    if(cached){
      applyFramePayload(cached,null,false,true);
      const lead=Math.max(5,Math.round(Number(state.manifest.video.fps||30)*0.35));
      if(state.playbackBufferedEnd<0||frame+lead>=state.playbackBufferedEnd){
        primePlaybackBuffer(Math.min(total-1,frame+lead));
      }
      return;
    }
    state.playbackQueuedFrame=frame;
    drawBodyTimeline();
    updatePlaybackStatus('Playing · overlay buffer catching up','buffering');
    if(!state.playbackRequestInFlight)pumpPlaybackBuffer();
  }

  async function loadFrame(frameIndex, sampleIndex=null) {
    if(!state.manifest)return;
    stopPlaybackFrameRequests();
    const serial=++state.requestSerial;
    const requestedFrame=Math.max(0,Math.min(Number(state.manifest.video.frame_count||1)-1,Number(frameIndex)||0));
    const requestedSample=sampleIndex==null?0:Math.max(0,Number(sampleIndex)||0);
    try{
      const payload=await fetchFramePayload(requestedFrame,requestedSample);if(serial!==state.requestSerial)return;
      state.presentedFrame=requestedFrame;
      applyFramePayload(payload,sampleIndex,sampleIndex==null);
      updatePlaybackStatus('Paused · full review frame','');
      primePlaybackBuffer(requestedFrame);
    }
    catch(error){if(serial===state.requestSerial)showError(error,'Could not load AOI frame');}
  }

  function videoSourceDescriptor(mode=state.videoSourceMode) {
    const sources=state.manifest?.video_sources||{};
    if(sources[mode])return sources[mode];
    if(mode==='privacy'&&state.manifest){
      return {available:true,url:state.manifest.video_url,frame_pts_ns:state.manifest.video?.frame_pts_ns||[],source_start_sec:0,source_end_sec:state.manifest.video?.duration_seconds||0};
    }
    return null;
  }

  function framePtsNs(mode=state.videoSourceMode) {
    const count=Math.max(1,Number(state.manifest?.video?.frame_count||1));
    const source=videoSourceDescriptor(mode);
    const candidate=source?.frame_pts_ns;
    if(Array.isArray(candidate)&&candidate.length===count)return candidate.map(Number);
    const local=state.manifest?.video?.frame_pts_ns;
    if(Array.isArray(local)&&local.length===count)return local.map(Number);
    const fps=Math.max(1e-6,Number(state.manifest?.video?.fps||30));
    return Array.from({length:count},(_,index)=>Math.round(index*1e9/fps));
  }

  function localFramePtsNs() {
    const count=Math.max(1,Number(state.manifest?.video?.frame_count||1));
    const local=state.manifest?.video?.frame_pts_ns;
    if(Array.isArray(local)&&local.length===count)return local.map(Number);
    const fps=Math.max(1e-6,Number(state.manifest?.video?.fps||30));
    return Array.from({length:count},(_,index)=>Math.round(index*1e9/fps));
  }

  function frameFromPts(time,pts) {
    const count=Math.max(1,Number(state.manifest?.video?.frame_count||1));
    const target=Math.max(0,Number(time)||0)*1e9;
    if(Array.isArray(pts)&&pts.length===count){
      let lo=0,hi=pts.length-1;
      while(lo<=hi){
        const mid=(lo+hi)>>1;
        if(Number(pts[mid])<=target+500)lo=mid+1;else hi=mid-1;
      }
      return Math.max(0,Math.min(count-1,hi));
    }
    return Math.max(0,Math.min(count-1,Math.floor((Number(time)||0)*Number(state.manifest?.video?.fps||30)+1e-6)));
  }

  function frameFromMediaTime(time) {
    return frameFromPts(time,framePtsNs());
  }

  function frameFromLocalTime(time) {
    return frameFromPts(time,localFramePtsNs());
  }

  function localTimeForFrame(frameIndex) {
    if(!state.manifest)return 0;
    const count=Math.max(1,Number(state.manifest.video.frame_count||1));
    const frame=Math.max(0,Math.min(count-1,Number(frameIndex)||0));
    const pts=localFramePtsNs();
    return Number(pts[frame]||0)/1e9;
  }

  function mediaTimeForFrame(frameIndex) {
    if(!state.manifest)return 0;
    const count=Math.max(1,Number(state.manifest.video.frame_count||1));
    const frame=Math.max(0,Math.min(count-1,Number(frameIndex)||0));
    const pts=framePtsNs();
    const start=Number(pts[frame]||0)/1e9;
    const fallback=1/Math.max(1e-6,Number(state.manifest.video.fps||30));
    let next=frame+1;
    while(next<count&&Number(pts[next]||0)<=Number(pts[frame]||0)+500)next+=1;
    const source=videoSourceDescriptor();
    const end=next<count?Number(pts[next]||0)/1e9:Number(source?.source_end_sec||start+fallback);
    const span=Math.max(0,Math.min(fallback,Math.max(0,end-start)));
    return start+span*0.25;
  }

  function localDuration() {
    return Math.max(0,Number(state.manifest?.video?.duration_seconds||0));
  }

  function updateVideoSourceUI() {
    const button=document.querySelector('#aoi-review-blur-toggle');
    const warning=document.querySelector('#aoi-review-unblurred-warning');
    const wrap=document.querySelector('#aoi-review-video-wrap');
    const source=videoSourceDescriptor('unblurred');
    const available=Boolean(source?.available);
    const unblurred=state.videoSourceMode==='unblurred';
    if(button){
      button.disabled=!state.manifest||!available||state.sourceSwitchPending;
      button.classList.toggle('active',unblurred);
      button.textContent=state.sourceSwitchPending?'Switching video…':unblurred?'Blur off · original':'Blur on · privacy';
      button.title=available
        ?(unblurred?'Return to the privacy-blurred review clip':'Show the original unblurred Native scene video for local annotation only')
        :`Original video unavailable${source?.reason?` · ${source.reason.replaceAll('_',' ')}`:''}`;
      button.setAttribute('aria-pressed',unblurred?'true':'false');
    }
    if(warning)warning.hidden=!unblurred;
    if(wrap)wrap.classList.toggle('showing-unblurred',unblurred);
  }

  function seekVideoForSourceSwitch(video,frame,exactTime) {
    return new Promise((resolve,reject)=>{
      let settled=false;
      const finish=(error=null)=>{
        if(settled)return;
        settled=true;
        window.clearTimeout(timer);
        video.removeEventListener('seeked',onSeeked);
        video.removeEventListener('error',onError);
        if(error)reject(error);else resolve();
      };
      const onSeeked=()=>finish();
      const onError=()=>finish(new Error('Could not seek the selected review video'));
      const timer=window.setTimeout(()=>finish(new Error('Timed out while locking the selected review frame')),12000);
      if(Math.abs(Number(video.currentTime||0)-exactTime)<1e-6&&!video.seeking){
        state.pendingExplicitSeekFrame=-1;
        finalizePausedFrame(frame,exactTime);
        finish();
        return;
      }
      video.addEventListener('seeked',onSeeked,{once:true});
      video.addEventListener('error',onError,{once:true});
      video.currentTime=exactTime;
    });
  }

  async function switchVideoSource(mode,{frame=null,resume=null,quiet=false}={}) {
    const video=document.querySelector('#aoi-review-video-element');
    if(!video||!state.manifest)return;
    const source=videoSourceDescriptor(mode);
    if(!source?.available||!source?.url){
      if(!quiet)toast('Original video unavailable','The selected Privacy clip cannot resolve its original Native scene video.');
      return;
    }
    const targetFrame=Math.max(0,Math.min(Number(state.manifest.video.frame_count||1)-1,
      frame==null?(state.currentFrame>=0?state.currentFrame:(state.presentedFrame>=0?state.presentedFrame:0)):Number(frame)||0));
    const shouldResume=resume==null?!video.paused:Boolean(resume);
    const serial=++state.sourceSwitchSerial;
    state.sourceSwitchPending=true;
    state.videoSourceMode=mode;
    state.pauseLockFrame=-1;
    state.pauseLockSerial+=1;
    state.pendingExplicitSeekFrame=targetFrame;
    state.presentedFrame=targetFrame;
    stopPresentedFrameSync();
    stopPlaybackFrameRequests();
    if(!video.paused)video.pause();
    updateVideoSourceUI();
    renderPayload();
    try{
      await new Promise((resolve,reject)=>{
        const loaded=()=>{cleanup();resolve();};
        const failed=()=>{cleanup();reject(new Error(mode==='unblurred'?'Could not open the original Native video':'Could not open the Privacy clip'));};
        const cleanup=()=>{video.removeEventListener('loadedmetadata',loaded);video.removeEventListener('error',failed);};
        video.addEventListener('loadedmetadata',loaded,{once:true});
        video.addEventListener('error',failed,{once:true});
        video.src=source.url;
        video.load();
      });
      if(serial!==state.sourceSwitchSerial)return;
      const seek=document.querySelector('#aoi-review-seek');
      if(seek){seek.max=String(localDuration()||1);seek.value=String(localTimeForFrame(targetFrame));}
      const exactTime=Math.max(0,Math.min(video.duration||Infinity,mediaTimeForFrame(targetFrame)));
      updateTransportReadout(localTimeForFrame(targetFrame));
      await seekVideoForSourceSwitch(video,targetFrame,exactTime);
      if(serial!==state.sourceSwitchSerial)return;
      if(shouldResume){
        primePlaybackBuffer(targetFrame);
        const pending=video.play();
        if(pending?.catch)await pending;
      }
    }catch(error){
      if(mode==='unblurred'&&serial===state.sourceSwitchSerial){
        await switchVideoSource('privacy',{frame:targetFrame,resume:shouldResume,quiet:true});
      }
      if(!quiet)showError(error,'Could not switch AOI review video');
    }finally{
      if(serial===state.sourceSwitchSerial){state.sourceSwitchPending=false;updateVideoSourceUI();renderPayload();}
    }
  }

  async function loadManifest() {
    const empty=document.querySelector('#aoi-review-empty');
    const video=document.querySelector('#aoi-review-video-element');
    state.artifactId=document.querySelector('#aoi-review-artifact')?.value||'';
    state.videoArtifactId=document.querySelector('#aoi-review-video')?.value||'';
    state.confidenceThreshold=Number(document.querySelector('#aoi-review-threshold')?.value||0.65);
    if(!state.artifactId||!state.videoArtifactId){
      stopPlaybackFrameRequests();clearFrameCaches();
      state.manifest=null;state.currentPayload=null;state.currentFrame=-1;state.presentedFrame=-1;state.lastPresentedFrame=-1;state.lastPresentedMediaTime=null;state.pauseLockFrame=-1;state.pauseLockSerial+=1;state.pendingExplicitSeekFrame=-1;state.videoSourceMode='privacy';state.sourceSwitchPending=false;
      if(video){video.removeAttribute('src');video.load();}
      updateVideoSourceUI();
      if(empty)empty.style.display='flex';
      document.querySelector('#aoi-review-shell')?.classList.remove('body-review-active');
      const bodyOverview=document.querySelector('#body-aoi-overview');if(bodyOverview)bodyOverview.hidden=true;
      const bodyTimeline=document.querySelector('#body-aoi-timeline-panel');if(bodyTimeline)bodyTimeline.hidden=true;
      renderInspector();
      updateHumanReviewEntry();updateBodyHumanReviewEntry();
      return;
    }
    try{
      const url=`/api/aoi-review/${encodeURIComponent(state.sessionId)}/${state.mode}/${encodeURIComponent(state.artifactId)}/manifest?video_artifact_id=${encodeURIComponent(state.videoArtifactId)}&confidence_threshold=${state.confidenceThreshold}`;
      state.manifest=await api(url);
      stopPlaybackFrameRequests();
      clearFrameCaches();state.lastOverlayFrame=-1;
      state.currentPayload=null;state.currentFrame=-1;state.presentedFrame=-1;state.lastPresentedFrame=-1;state.lastPresentedMediaTime=null;state.pauseLockFrame=-1;state.pauseLockSerial+=1;state.pendingExplicitSeekFrame=-1;state.sampleIndex=0;
      renderPayload();
      if(empty)empty.style.display='none';
      state.videoSourceMode='privacy';
      const manual=state.manifest.manual_review;
      const badge=document.querySelector('#aoi-review-badge');
      if(badge)badge.textContent=state.mode==='face'&&manual?`${modes[state.mode].label} · ${manual.pending_review_unit_count??manual.pending_frame_count} pending groups · ${manual.pending_frame_count} frames`:`${modes[state.mode].label} · ${state.manifest.sample_count} samples`;
      document.querySelector('#aoi-review-world-wrap')?.classList.toggle('hidden',!state.manifest.available_layers.body_world_inset);
      document.querySelector('#aoi-review-shell')?.classList.toggle('face-review-active',state.mode==='face');
      document.querySelector('#aoi-review-shell')?.classList.toggle('body-review-active',['body','fusion'].includes(state.mode));
      const bodyKey=document.querySelector('#body-aoi-video-key');if(bodyKey)bodyKey.hidden=!['body','fusion'].includes(state.mode);
      const bodyTimeline=document.querySelector('#body-aoi-timeline-panel');if(bodyTimeline)bodyTimeline.hidden=!state.manifest.available_layers.body_timeline;
      renderManualReview();
      updateVideoSourceUI();
      const pending=['face','body'].includes(state.mode)?(state.manifest.event_frames?.pending_review||[]):[];
      const initialFrame=pending.length?pending[0]:0;
      state.presentedFrame=initialFrame;
      await switchVideoSource('privacy',{frame:initialFrame,resume:false,quiet:true});
      await loadFrame(initialFrame,null);
    }catch(error){
      showError(error,'Could not open AOI Review Player');
      if(empty){empty.style.display='flex';empty.querySelector('strong').textContent=error.message;}
      state.manifest=null;state.currentPayload=null;renderInspector();updateHumanReviewEntry();updateBodyHumanReviewEntry();
    }
  }

  function updateTransportReadout(shownTime) {
    const video=document.querySelector('#aoi-review-video-element');
    const readout=document.querySelector('#aoi-review-time');
    const seek=document.querySelector('#aoi-review-seek');
    if(!video)return;
    const value=Math.max(0,Number(shownTime)||0);
    if(readout)readout.textContent=`${fmtTime(value)} / ${fmtTime(localDuration())}`;
    if(seek&&!seek.matches(':active'))seek.value=String(value);
  }

  function finalizePausedFrame(frameIndex, mediaTime=null) {
    if(!state.manifest)return;
    const total=Math.max(1,Number(state.manifest.video.frame_count||1));
    const frame=Math.max(0,Math.min(total-1,Number(frameIndex)||0));
    const shownMediaTime=mediaTime==null?mediaTimeForFrame(frame):Math.max(0,Number(mediaTime)||0);
    state.pauseLockFrame=-1;
    state.presentedFrame=frame;
    state.lastPresentedFrame=frame;
    state.lastPresentedMediaTime=shownMediaTime;
    updateTransportReadout(localTimeForFrame(frame));
    const cached=state.frameCache.get(frame);
    if(cached&&cached.payload_profile!=='compact_playback'){
      applyFramePayload(cached,null,true);
      primePlaybackBuffer(frame);
    }else{
      updatePlaybackStatus('Paused · loading full review frame','buffering');
      loadFrame(frame,null).finally?.(()=>updatePlaybackStatus('Paused · full review frame',''));
    }
  }

  function lockPausedPresentedFrame(video, candidateFrame=null) {
    if(!video||!video.paused||!state.manifest||video.seeking)return;
    const total=Math.max(1,Number(state.manifest.video.frame_count||1));
    const fallback=frameFromMediaTime(video.currentTime||0);
    const frame=Math.max(0,Math.min(total-1,
      candidateFrame==null
        ?(state.lastPresentedFrame>=0?state.lastPresentedFrame:(state.presentedFrame>=0?state.presentedFrame:fallback))
        :Number(candidateFrame)||0
    ));
    const exactTime=Math.max(0,Math.min(video.duration||Infinity,mediaTimeForFrame(frame)));
    state.presentedFrame=frame;
    // currentTime and the compositor's visible paused frame are not guaranteed
    // to identify the same decoded image.  Always seek to an interior point of
    // the selected frame so the video pixels and the review payload are frozen
    // on one deterministic frame before annotation begins.
    if(Math.abs(Number(video.currentTime||0)-exactTime)>1e-6||video.seeking){
      state.pauseLockFrame=frame;
      const serial=++state.pauseLockSerial;
      video.currentTime=exactTime;
      window.setTimeout(()=>{
        if(serial!==state.pauseLockSerial||state.pauseLockFrame!==frame||!video.paused||video.seeking)return;
        finalizePausedFrame(frame,exactTime);
      },180);
      renderPayload();
      return;
    }
    finalizePausedFrame(frame,exactTime);
  }

  function seekReviewFrame(frameIndex) {
    const video=document.querySelector('#aoi-review-video-element');
    if(!video||!state.manifest)return;
    const total=Math.max(1,Number(state.manifest.video.frame_count||1));
    const frame=Math.max(0,Math.min(total-1,Number(frameIndex)||0));
    state.pauseLockFrame=-1;
    state.pauseLockSerial+=1;
    state.pendingExplicitSeekFrame=frame;
    stopPresentedFrameSync();
    stopPlaybackFrameRequests();
    if(!video.paused)video.pause();
    state.presentedFrame=frame;
    renderPayload();
    const exactTime=Math.max(0,Math.min(video.duration||Infinity,mediaTimeForFrame(frame)));
    if(Math.abs(Number(video.currentTime||0)-exactTime)<1e-6&&!video.seeking){
      state.pendingExplicitSeekFrame=-1;
      finalizePausedFrame(frame,exactTime);
      return;
    }
    video.currentTime=exactTime;
  }

  function updateTime(mediaTime=null,{presented=false}={}) {
    const video=document.querySelector('#aoi-review-video-element');if(!video)return;
    const shownMediaTime=mediaTime==null?Number(video.currentTime||0):Number(mediaTime||0);
    const source=videoSourceDescriptor();
    if(state.videoSourceMode==='unblurred'&&source?.available){
      const start=Number(source.source_start_sec||0),end=Number(source.source_end_sec||Infinity);
      if(shownMediaTime<start-0.002&&!video.seeking){video.currentTime=mediaTimeForFrame(0);return;}
      if(shownMediaTime>=end-0.0005&&!video.paused){
        video.pause();seekReviewFrame(Math.max(0,Number(state.manifest?.video?.frame_count||1)-1));return;
      }
    }
    const frame=frameFromMediaTime(shownMediaTime);
    updateTransportReadout(localTimeForFrame(frame));
    state.presentedFrame=frame;
    if(presented||video.paused){
      state.lastPresentedFrame=frame;
      state.lastPresentedMediaTime=shownMediaTime;
    }
    if(video.paused){
      if(frame!==state.currentFrame||Number(state.currentPayload?.output_frame_index)!==frame)loadFrame(frame,null);
      else renderPayload();
    }else{
      requestPlaybackFrame(frame);
    }
  }

  function schedulePresentedFrameSync(){
    const video=document.querySelector('#aoi-review-video-element');if(!video||video.paused||video.ended)return;
    if(typeof video.requestVideoFrameCallback==='function'){
      state.videoFrameCallbackId=video.requestVideoFrameCallback((_,metadata)=>{
        const mediaTime=metadata?.mediaTime ?? video.currentTime;
        updateTime(mediaTime,{presented:true});
        schedulePresentedFrameSync();
      });
    }
  }

  function stopPresentedFrameSync(){
    const video=document.querySelector('#aoi-review-video-element');
    if(video&&state.videoFrameCallbackId!=null&&typeof video.cancelVideoFrameCallback==='function')video.cancelVideoFrameCallback(state.videoFrameCallbackId);
    state.videoFrameCallbackId=null;
  }

  function navigateEvent(kind,direction) {
    const rows=state.manifest?.event_frames?.[kind]||[];if(!rows.length){toast('No matching review frames',`No ${kind.replaceAll('_',' ')} frames were found.`);return;}
    const current=Math.max(0,state.currentFrame);let target;
    if(direction>0)target=rows.find(x=>x>current)??rows[0];else target=[...rows].reverse().find(x=>x<current)??rows[rows.length-1];
    seekReviewFrame(target);
  }

  function nativeFullscreenElement() {
    return document.fullscreenElement || document.webkitFullscreenElement || null;
  }

  function reviewCard() {
    return document.querySelector('#aoi-review-player-card');
  }

  function reviewFullscreenActive() {
    const card=reviewCard();
    return Boolean(card && (nativeFullscreenElement()===card || card.classList.contains('aoi-review-fullscreen-fallback')));
  }

  function updateFullscreenUI() {
    const card=reviewCard(), active=reviewFullscreenActive();
    const button=document.querySelector('#aoi-review-fullscreen');
    const exitButton=document.querySelector('#aoi-review-fullscreen-exit');
    if(card)card.classList.toggle('is-fullscreen',active);
    document.body.classList.toggle('aoi-review-fullscreen-open',active);
    if(button){
      button.textContent=active?'↙':'⛶';
      button.title=active?'Exit fullscreen':'Enter fullscreen';
      button.setAttribute('aria-label',button.title);
    }
    if(exitButton)exitButton.hidden=!active;
    requestAnimationFrame(renderPayload);
  }

  async function enterReviewFullscreen() {
    const card=reviewCard();if(!card)return;
    if(reviewFullscreenActive())return;
    const request=card.requestFullscreen||card.webkitRequestFullscreen;
    if(request){
      try{await request.call(card);updateFullscreenUI();return;}catch(_){ }
    }
    card.classList.add('aoi-review-fullscreen-fallback');
    updateFullscreenUI();
  }

  async function exitReviewFullscreen() {
    const card=reviewCard();if(!card)return;
    card.classList.remove('aoi-review-fullscreen-fallback');
    const active=nativeFullscreenElement();
    if(active===card){
      const exit=document.exitFullscreen||document.webkitExitFullscreen;
      if(exit){try{await exit.call(document);}catch(_){ }}
    }
    updateFullscreenUI();
  }

  async function toggleReviewFullscreen() {
    if(reviewFullscreenActive())await exitReviewFullscreen();else await enterReviewFullscreen();
  }

  async function togglePlayback(player) {
    if(!player)return;
    if(player.paused||player.ended){
      const button=document.querySelector('#aoi-review-play');
      if(button){button.disabled=false;button.textContent='Ⅱ Pause';button.setAttribute('aria-label','Pause review video');}
      try{
        let frame=frameFromMediaTime(player.currentTime||0);
        const total=Math.max(1,Number(state.manifest?.video?.frame_count||1));
        if(frame>=total-1){seekReviewFrame(0);frame=0;}
        state.presentedFrame=frame;
        updatePlaybackStatus('Preparing short live-overlay buffer','buffering');
        await ensurePlaybackReady(frame);
        const pending=player.play();
        if(pending?.catch)await pending;
      }catch(error){
        if(button?.isConnected){button.textContent='▶ Play';button.setAttribute('aria-label','Play review video');}
        updatePlaybackStatus('Paused · playback could not start','error');
        showError(error,'Could not play AOI review video');
      }
    }else player.pause();
  }

  async function exportVideo() {
    if(!state.artifactId||!state.videoArtifactId)return;
    const button=document.querySelector('#aoi-review-export');if(button){button.disabled=true;button.textContent='Checking encoder…';}
    try{
      const check=await api('/api/media-tools/check',{method:'POST',body:{mode:'encoder',preserve_audio:document.querySelector('#aoi-review-export-audio')?.checked!==false}});if(!check.ready)throw new Error(check.error);
      await api('/api/jobs/aoi-review-video',{method:'POST',body:{session_id:state.sessionId,mode:state.mode,artifact_id:state.artifactId,video_artifact_id:state.videoArtifactId,label:document.querySelector('#aoi-review-export-label')?.value.trim()||'aoi_review_video',confidence_threshold:state.confidenceThreshold,draw_face_regions:state.layers.face,draw_trail:state.layers.trail,draw_confidence:state.layers.confidence,preserve_audio:document.querySelector('#aoi-review-export-audio')?.checked!==false,ffmpeg_path:check.preflight.ffmpeg.path}});
      toast('AOI review video queued','The dynamic player remains available while the QC MP4 renders.');await refreshState({quiet:false});
    }catch(error){showError(error,'Could not export AOI review video');}
    finally{if(button?.isConnected){button.disabled=false;button.textContent='Export pre-rendered QC MP4';}}
  }

  async function bind({sessionId,interactionId='',roleId='',nativeTrimArtifactId='',initialMode='',initialArtifactId='',initialVideoArtifactId=''}) {
    const shell=document.querySelector('#aoi-review-shell');
    if(!shell)return;
    state.sessionId=sessionId;
    state.interactionId=String(interactionId||'').toUpperCase();
    state.roleId=String(roleId||'').toUpperCase();
    state.nativeTrimArtifactId=String(nativeTrimArtifactId||'');
    state.initialMode=initialMode||'';
    state.initialArtifactId=initialArtifactId||'';
    state.initialVideoArtifactId=initialVideoArtifactId||'';
    try{
      state.options=await api(`/api/aoi-review/${encodeURIComponent(sessionId)}/options`);
      const original=Object.fromEntries(Object.entries(state.options.modes||{}).map(([key,value])=>[key,[...(value||[])]]));
      if(state.interactionId||state.roleId||state.nativeTrimArtifactId){
        for(const modeKey of Object.keys(state.options.modes||{})){
          state.options.modes[modeKey]=(state.options.modes[modeKey]||[]).filter(contextMatch);
        }
        state.options.videos=(state.options.videos||[]).filter(contextMatch);
      }
      if(state.initialArtifactId){
        for(const modeKey of Object.keys(original)){
          const requested=(original[modeKey]||[]).find(row=>row.artifact_id===state.initialArtifactId);
          if(requested&&contextMatch(requested)){
            state.initialMode=modeKey;
            if(!(state.options.modes[modeKey]||[]).some(row=>row.artifact_id===requested.artifact_id))state.options.modes[modeKey].push(requested);
            break;
          }
        }
      }
    }catch(error){showError(error,'Could not load AOI review options');return;}
    const mode=document.querySelector('#aoi-review-mode');
    const artifact=document.querySelector('#aoi-review-artifact');
    const video=document.querySelector('#aoi-review-video');
    const availableModes=Object.keys(modes).filter(key=>(state.options.modes?.[key]||[]).length);
    const requestedMode=state.initialMode&&availableModes.includes(state.initialMode)?state.initialMode:'';
    const artifactMode=availableModes.find(key=>(state.options.modes[key]||[]).some(row=>row.artifact_id===state.initialArtifactId));
    state.mode=artifactMode||requestedMode||availableModes[0]||'face';
    mode.value=state.mode;
    mode.onchange=()=>{state.mode=mode.value;state.artifactId='';state.videoArtifactId='';populateArtifacts();updateHumanReviewEntry();updateBodyHumanReviewEntry();loadManifest();};
    artifact.onchange=()=>{state.artifactId=artifact.value;state.videoArtifactId='';populateVideos();updateHumanReviewEntry();updateBodyHumanReviewEntry();loadManifest();};
    video.onchange=()=>{state.videoArtifactId=video.value;loadManifest();};
    document.querySelector('#aoi-review-threshold').onchange=()=>loadManifest();
    document.querySelectorAll('[data-review-layer]').forEach(input=>input.onchange=()=>{state.layers[input.dataset.reviewLayer]=input.checked;renderPayload();});
    document.querySelectorAll('[data-body-view]').forEach(button=>button.onclick=()=>{state.bodyView=button.dataset.bodyView;document.querySelectorAll('[data-body-view]').forEach(row=>row.classList.toggle('active',row.dataset.bodyView===state.bodyView));drawWorldInset();});
    const humanReviewButton=document.querySelector('#aoi-review-open-human-review');
    if(humanReviewButton)humanReviewButton.onclick=openHumanReview;
    const bodyHumanReviewButton=document.querySelector('#aoi-review-open-body-human-review');
    if(bodyHumanReviewButton)bodyHumanReviewButton.onclick=openBodyHumanReview;
    updateHumanReviewEntry();updateBodyHumanReviewEntry();
    const timelineCanvas=document.querySelector('#body-aoi-timeline');
    if(timelineCanvas){
      timelineCanvas.onmousemove=event=>{const bins=state.manifest?.body_timeline?.bins||[];if(!bins.length)return;const rect=timelineCanvas.getBoundingClientRect();const index=Math.max(0,Math.min(bins.length-1,Math.floor((event.clientX-rect.left)/Math.max(1,rect.width)*bins.length)));state.bodyTimelineHover=index;const bin=bins[index],tooltip=document.querySelector('#body-aoi-timeline-tooltip');if(tooltip){tooltip.hidden=false;tooltip.style.left=`${Math.min(rect.width-190,Math.max(6,event.clientX-rect.left+8))}px`;tooltip.innerHTML=`<strong>${esc(bodyTargetLabel(bin.dominant_social_region,bin.dominant_relation))}</strong><span>${fmtTime(bin.start_seconds)}–${fmtTime(bin.end_seconds)} · ${(100*Number(bin.hit_fraction||0)).toFixed(1)}% body hit</span><span>${(100*Number(bin.review_fraction||0)).toFixed(1)}% review · ${(100*Number(bin.self_occlusion_fraction||0)).toFixed(1)}% self occlusion</span>`;}drawBodyTimeline();};
      timelineCanvas.onmouseleave=()=>{state.bodyTimelineHover=-1;const tooltip=document.querySelector('#body-aoi-timeline-tooltip');if(tooltip)tooltip.hidden=true;drawBodyTimeline();};
      timelineCanvas.onclick=event=>{const rect=timelineCanvas.getBoundingClientRect(),total=Math.max(1,Number(state.manifest?.video?.frame_count||1));seekReviewFrame(Math.round(Math.max(0,Math.min(1,(event.clientX-rect.left)/Math.max(1,rect.width)))*(total-1)));};
    }
    const player=document.querySelector('#aoi-review-video-element');
    player.ontimeupdate=()=>{if(typeof player.requestVideoFrameCallback!=='function')updateTime(null,{presented:!player.paused});};
    player.onplay=()=>{
      state.pauseLockFrame=-1;
      state.pendingExplicitSeekFrame=-1;
      state.pauseLockSerial+=1;
      const playButton=document.querySelector('#aoi-review-play');if(playButton){playButton.textContent='Ⅱ Pause';playButton.setAttribute('aria-label','Pause review video');}
      updatePlaybackStatus('Playing · live overlay','live');
      stopPresentedFrameSync();
      requestPlaybackFrame(frameFromMediaTime(player.currentTime||0));
      schedulePresentedFrameSync();
    };
    player.onpause=()=>{
      const playButton=document.querySelector('#aoi-review-play');if(playButton){playButton.textContent='▶ Play';playButton.setAttribute('aria-label','Play review video');}
      updatePlaybackStatus('Paused · loading full review frame','buffering');
      stopPresentedFrameSync();
      stopPlaybackFrameRequests();
      const candidate=state.lastPresentedFrame>=0?state.lastPresentedFrame:(state.presentedFrame>=0?state.presentedFrame:frameFromMediaTime(player.currentTime||0));
      const serial=++state.pauseLockSerial;
      window.setTimeout(()=>{
        // Frame-navigation and scrub actions set currentTime immediately after pause().
        // In that case the seeked handler owns the target; never snap back to the old frame.
        if(serial!==state.pauseLockSerial||!player.paused||player.seeking||state.pendingExplicitSeekFrame>=0)return;
        lockPausedPresentedFrame(player,candidate);
      },0);
    };
    player.onseeked=()=>{
      if(state.pendingExplicitSeekFrame>=0&&player.paused){
        const frame=state.pendingExplicitSeekFrame;
        state.pendingExplicitSeekFrame=-1;
        finalizePausedFrame(frame,mediaTimeForFrame(frame));
        return;
      }
      if(state.pauseLockFrame>=0&&player.paused){
        const frame=state.pauseLockFrame;
        state.pauseLockSerial+=1;
        finalizePausedFrame(frame,mediaTimeForFrame(frame));
        return;
      }
      updateTime(null,{presented:player.paused});
    };
    document.querySelector('#aoi-review-play').onclick=()=>togglePlayback(player);
    document.querySelector('#aoi-review-prev-frame').onclick=()=>{if(!state.manifest)return;seekReviewFrame(Math.max(0,state.currentFrame-1));};
    document.querySelector('#aoi-review-next-frame').onclick=()=>{if(!state.manifest)return;seekReviewFrame(Math.min(state.manifest.video.frame_count-1,state.currentFrame+1));};
    document.querySelector('#aoi-review-prev-sample').onclick=()=>state.currentPayload&&loadFrame(state.currentFrame,Math.max(0,state.sampleIndex-1));
    document.querySelector('#aoi-review-next-sample').onclick=()=>state.currentPayload&&loadFrame(state.currentFrame,Math.min(Math.max(0,state.currentPayload.sample_count-1),state.sampleIndex+1));
    document.querySelector('#aoi-review-seek').oninput=event=>{seekReviewFrame(frameFromLocalTime(Number(event.target.value)||0));};
    player.onclick=()=>togglePlayback(player);
    const blurToggle=document.querySelector('#aoi-review-blur-toggle');
    if(blurToggle)blurToggle.onclick=()=>switchVideoSource(state.videoSourceMode==='unblurred'?'privacy':'unblurred');
    document.querySelector('#aoi-review-fullscreen').onclick=toggleReviewFullscreen;
    document.querySelector('#aoi-review-fullscreen-exit').onclick=exitReviewFullscreen;
    if(!state.fullscreenBound){
      document.addEventListener('fullscreenchange',updateFullscreenUI);
      document.addEventListener('webkitfullscreenchange',updateFullscreenUI);
      state.fullscreenBound=true;
    }
    if(!state.keyboardBound){
      document.addEventListener('keydown',event=>{
        const card=reviewCard();if(!card||!card.isConnected)return;
        const target=event.target;const typing=target?.matches?.('input,textarea,select,[contenteditable="true"]');
        if(event.key==='Escape'&&card.classList.contains('aoi-review-fullscreen-fallback')){event.preventDefault();exitReviewFullscreen();return;}
        if((event.key===' '||event.code==='Space')&&reviewFullscreenActive()&&!typing){event.preventDefault();togglePlayback(document.querySelector('#aoi-review-video-element'));}
      });
      state.keyboardBound=true;
    }
    if(!state.resizeBound){window.addEventListener('resize',renderPayload,{passive:true});state.resizeBound=true;}
    updateFullscreenUI();
    document.querySelectorAll('[data-review-nav]').forEach(button=>button.onclick=()=>navigateEvent(button.dataset.reviewNav,Number(button.dataset.direction)));
    document.querySelectorAll('[data-face-review-action]').forEach(button=>button.onclick=()=>saveManualDecision(button.dataset.faceReviewAction));
    document.querySelectorAll('[data-body-review-action]').forEach(button=>button.onclick=()=>saveBodyManualDecision(button.dataset.bodyReviewAction));
    const reviewRelation=document.querySelector('#face-review-relation');
    if(reviewRelation)reviewRelation.onchange=refreshReviewEntitySuggestion;
    const bodyReviewRelation=document.querySelector('#body-review-relation');
    if(bodyReviewRelation)bodyReviewRelation.onchange=renderBodyManualReview;
    const outputLabel=document.querySelector('#face-review-output-label');
    if(outputLabel)outputLabel.oninput=()=>{outputLabel.dataset.touched='true';};
    const bodyOutputLabel=document.querySelector('#body-review-output-label');
    if(bodyOutputLabel)bodyOutputLabel.oninput=()=>{bodyOutputLabel.dataset.touched='true';};
    const applyReview=document.querySelector('#face-review-apply');
    if(applyReview)applyReview.onclick=applyManualReview;
    const applyBodyReview=document.querySelector('#body-review-apply');
    if(applyBodyReview)applyBodyReview.onclick=applyBodyManualReview;
    document.querySelector('#aoi-review-export').onclick=exportVideo;
    populateArtifacts(state.initialArtifactId||state.artifactId);
    await loadManifest();
  }

  return {markup,bind,openHumanReview,openBodyHumanReview};
})();
