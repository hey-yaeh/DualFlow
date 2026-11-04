(() => {
  'use strict';

  const ROLE_COLORS = ['#effd3f', '#78c7ff', '#7def59', '#ffcf62', '#ff8fc8', '#caa5ff'];
  const ETM_COLORS = ['#ffffff', '#b7c1a4', '#f7eab2', '#c7e8ff', '#ffd2e6', '#e4d5ff'];

  function markup({
    sessionId = '', assignmentId = '', gazeId = '', peerGazeId = '', previewMode = 'single', pairingToleranceMs = '',
  } = {}) {
    const assignmentUrl = assignmentId
      ? `/api/preview/etm-assignment/${encodeURIComponent(sessionId)}/${encodeURIComponent(assignmentId)}`
      : '';
    const gazeUrl = gazeId
      ? `/api/preview/gaze-rays/${encodeURIComponent(sessionId)}/${encodeURIComponent(gazeId)}`
      : '';
    let dyadicUrl = '';
    if (gazeId && peerGazeId && gazeId !== peerGazeId) {
      dyadicUrl = `/api/preview/gaze-rays-dyadic/${encodeURIComponent(sessionId)}/${encodeURIComponent(gazeId)}/${encodeURIComponent(peerGazeId)}`;
      if (pairingToleranceMs !== '' && Number.isFinite(Number(pairingToleranceMs))) {
        dyadicUrl += `?pairing_tolerance_ms=${encodeURIComponent(String(pairingToleranceMs))}`;
      }
    }
    const mode = previewMode === 'dyadic' && dyadicUrl ? 'dyadic' : 'single';
    return `<section class="ray-stage gaze3d-live-stage" id="gaze3d-live-preview"
      data-assignment-url="${assignmentUrl}" data-gaze-url="${gazeUrl}" data-dyadic-url="${dyadicUrl}" data-preview-mode="${mode}">
      <canvas class="gaze3d-preview-canvas" tabindex="0" aria-label="Interactive Vicon-world ETM and gaze preview"></canvas>
      <div class="gaze3d-preview-loading"><span class="gaze3d-loading-dot"></span><strong>Loading real Vicon geometry…</strong><small>Existing assignment and gaze artifacts are previewed without recomputation.</small></div>
      <div class="gaze3d-preview-toolbar" aria-label="3D preview controls">
        <div class="gaze3d-mode-buttons ${dyadicUrl ? '' : 'hidden'}">
          <button type="button" data-preview-mode="single" class="${mode === 'single' ? 'active' : ''}">Single</button>
          <button type="button" data-preview-mode="dyadic" class="${mode === 'dyadic' ? 'active' : ''}">Dyadic</button>
        </div>
        <div class="gaze3d-view-buttons">
          <button type="button" data-view="perspective" class="active">Perspective</button>
          <button type="button" data-view="front">Front</button>
          <button type="button" data-view="top">Top</button>
        </div>
        <label><input type="checkbox" data-layer="trails" checked> Trails</label>
        <label><input type="checkbox" data-layer="alternatives"> Alternatives</label>
        <label class="gaze3d-gaze-layer gaze3d-single-layer"><input type="checkbox" data-layer="gaze" checked> Gaze ray</label>
        <label class="gaze3d-role-ray gaze3d-dyadic-layer hidden" data-role-index="0"><input type="checkbox" checked> <span>R01 ray</span></label>
        <label class="gaze3d-role-ray gaze3d-dyadic-layer hidden" data-role-index="1"><input type="checkbox" checked> <span>R02 ray</span></label>
        <button type="button" data-action="reset" title="Reset view">Reset</button>
      </div>
      <div class="gaze3d-preview-legend" aria-label="Preview legend"></div>
      <div class="gaze3d-dyadic-qc hidden" aria-live="polite"></div>
      <div class="gaze3d-live-readout" aria-live="polite"></div>
      <div class="gaze3d-preview-controls">
        <button type="button" class="gaze3d-play" data-action="play" aria-label="Play preview">▶</button>
        <button type="button" class="gaze3d-step" data-action="previous" aria-label="Previous sample">‹</button>
        <input type="range" class="gaze3d-scrubber" min="0" max="0" value="0" step="1" aria-label="Preview frame">
        <button type="button" class="gaze3d-step" data-action="next" aria-label="Next sample">›</button>
        <span class="gaze3d-timecode">—</span>
        <select class="gaze3d-speed" aria-label="Preview speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select>
      </div>
      <div class="gaze3d-preview-help">Drag to orbit · mouse wheel to zoom · Shift-drag to pan · Space to play/pause</div>
    </section>`;
  }

  function clamp(value, minimum, maximum) { return Math.max(minimum, Math.min(maximum, value)); }
  function finitePoint(point) { return Array.isArray(point) && point.length === 3 && point.every(Number.isFinite); }
  function combineBounds(...boundsRows) {
    const valid = boundsRows.filter(row => row && finitePoint(row.min) && finitePoint(row.max));
    if (!valid.length) return {min: [-500, -500, -500], max: [500, 500, 500], centre: [0, 0, 0], span: [1000, 1000, 1000]};
    const minimum = [0, 1, 2].map(axis => Math.min(...valid.map(row => Number(row.min[axis]))));
    const maximum = [0, 1, 2].map(axis => Math.max(...valid.map(row => Number(row.max[axis]))));
    const span = maximum.map((value, axis) => Math.max(1, value - minimum[axis]));
    return {min: minimum, max: maximum, centre: minimum.map((value, axis) => (value + maximum[axis]) / 2), span};
  }
  function roleSort(a, b) {
    const order = role => role === 'R01' ? 0 : role === 'R02' ? 1 : 2;
    return order(a) - order(b) || String(a).localeCompare(String(b));
  }
  function roleColor(role, roles = []) {
    const ordered = [...roles].sort(roleSort);
    const index = Math.max(0, ordered.indexOf(role));
    return ROLE_COLORS[index % ROLE_COLORS.length];
  }
  function nearestByTime(samples, timeSec, key = 'c3d_time_sec') {
    if (!samples?.length) return null;
    if (!Number.isFinite(timeSec)) return samples[0];
    let low = 0, high = samples.length - 1;
    while (low < high) {
      const middle = Math.floor((low + high) / 2);
      const value = Number(samples[middle][key]);
      if (!Number.isFinite(value) || value < timeSec) low = middle + 1;
      else high = middle;
    }
    const a = samples[low];
    const b = samples[Math.max(0, low - 1)];
    if (!b || !Number.isFinite(Number(b[key]))) return a;
    return Math.abs(Number(a[key]) - timeSec) < Math.abs(Number(b[key]) - timeSec) ? a : b;
  }

  class Preview {
    constructor(root) {
      this.root = root;
      this.canvas = root.querySelector('canvas');
      this.context = this.canvas.getContext('2d');
      this.loading = root.querySelector('.gaze3d-preview-loading');
      this.legend = root.querySelector('.gaze3d-preview-legend');
      this.readout = root.querySelector('.gaze3d-live-readout');
      this.dyadicQc = root.querySelector('.gaze3d-dyadic-qc');
      this.scrubber = root.querySelector('.gaze3d-scrubber');
      this.timecode = root.querySelector('.gaze3d-timecode');
      this.playButton = root.querySelector('[data-action="play"]');
      this.speed = root.querySelector('.gaze3d-speed');
      this.assignment = null;
      this.gaze = null;
      this.dyadic = null;
      this.mode = root.dataset.previewMode === 'dyadic' ? 'dyadic' : 'single';
      this.index = 0;
      this.playing = false;
      this.playCursor = 0;
      this.lastTick = 0;
      this.view = {yaw: -0.58, pitch: -0.40, zoom: 1, panX: 0, panY: 0};
      this.layers = {trails: true, alternatives: false, gaze: true, roleRays: {}};
      this.drag = null;
      this.resizeObserver = null;
      this.animationFrame = 0;
      this.init();
    }

    async init() {
      this.bindControls();
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(this.root);
      this.resize();
      const assignmentUrl = this.root.dataset.assignmentUrl;
      const gazeUrl = this.root.dataset.gazeUrl;
      const dyadicUrl = this.root.dataset.dyadicUrl;
      if (!assignmentUrl && !gazeUrl && !dyadicUrl) {
        this.fail('Generate calibrated rays to load real Vicon geometry.');
        return;
      }
      const requests = [];
      if (assignmentUrl) requests.push(fetch(assignmentUrl, {cache: 'no-store'}).then(this.readResponse).then(data => { this.assignment = data; }).catch(error => ({error, type: 'assignment'})));
      if (gazeUrl) requests.push(fetch(gazeUrl, {cache: 'no-store'}).then(this.readResponse).then(data => { this.gaze = data; }).catch(error => ({error, type: 'gaze'})));
      if (dyadicUrl) requests.push(fetch(dyadicUrl, {cache: 'no-store'}).then(this.readResponse).then(data => { this.dyadic = data; }).catch(error => ({error, type: 'dyadic'})));
      const results = await Promise.all(requests);
      const failures = results.filter(Boolean);
      if (!this.assignment && !this.gaze && !this.dyadic) {
        const message = failures[0]?.error?.message || 'The preview data could not be loaded.';
        this.fail(message);
        return;
      }
      if (this.mode === 'dyadic' && !this.dyadic) this.mode = 'single';
      this.loading?.remove();
      this.root.classList.add('ready');
      this.configureRoleLayers();
      this.updateModeUi();
      this.configureTimeline();
      this.renderLegend();
      this.renderDyadicQc();
      this.draw();
    }

    readResponse = async response => {
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || data.message || `Preview request failed (${response.status})`);
      return data;
    };

    fail(message) {
      if (this.loading) {
        this.loading.innerHTML = `<strong>Preview unavailable</strong><small>${String(message || 'Unknown error')}</small>`;
        this.loading.classList.add('error');
      }
      this.drawEmpty();
    }

    configureRoleLayers() {
      const roles = this.dyadic?.role_order || Object.keys(this.dyadic?.roles || {}).sort(roleSort);
      this.root.querySelectorAll('.gaze3d-role-ray').forEach((label, index) => {
        const role = roles[index] || `Role ${index + 1}`;
        label.dataset.role = role;
        label.querySelector('span').textContent = `${role} ray`;
        label.style.setProperty('--role-ray-color', roleColor(role, roles));
        this.layers.roleRays[role] = label.querySelector('input')?.checked !== false;
        label.querySelector('input')?.addEventListener('change', event => {
          this.layers.roleRays[role] = event.currentTarget.checked;
          this.draw();
        });
      });
    }

    bindControls() {
      this.root.querySelectorAll('[data-preview-mode]').forEach(button => {
        button.addEventListener('click', () => {
          if (button.dataset.previewMode === 'dyadic' && !this.dyadic) return;
          this.mode = button.dataset.previewMode;
          this.index = 0; this.playCursor = 0;
          this.updateModeUi(); this.configureTimeline(); this.renderLegend(); this.renderDyadicQc(); this.draw();
        });
      });
      this.root.querySelectorAll('[data-view]').forEach(button => {
        button.addEventListener('click', () => {
          this.root.querySelectorAll('[data-view]').forEach(row => row.classList.toggle('active', row === button));
          const view = button.dataset.view;
          if (view === 'front') this.view = {...this.view, yaw: 0, pitch: 0, zoom: 1, panX: 0, panY: 0};
          else if (view === 'top') this.view = {...this.view, yaw: 0, pitch: -Math.PI / 2 + 0.001, zoom: 1, panX: 0, panY: 0};
          else this.view = {...this.view, yaw: -0.58, pitch: -0.40, zoom: 1, panX: 0, panY: 0};
          this.draw();
        });
      });
      this.root.querySelectorAll('[data-layer]').forEach(input => {
        input.addEventListener('change', () => { this.layers[input.dataset.layer] = input.checked; this.draw(); });
      });
      this.root.querySelector('[data-action="reset"]')?.addEventListener('click', () => {
        this.view = {yaw: -0.58, pitch: -0.40, zoom: 1, panX: 0, panY: 0};
        this.root.querySelectorAll('[data-view]').forEach(row => row.classList.toggle('active', row.dataset.view === 'perspective'));
        this.draw();
      });
      this.playButton?.addEventListener('click', () => this.togglePlay());
      this.root.querySelector('[data-action="previous"]')?.addEventListener('click', () => this.step(-1));
      this.root.querySelector('[data-action="next"]')?.addEventListener('click', () => this.step(1));
      this.scrubber?.addEventListener('input', () => { this.index = Number(this.scrubber.value) || 0; this.playCursor = this.index; this.draw(); });
      this.canvas.addEventListener('pointerdown', event => {
        this.canvas.setPointerCapture?.(event.pointerId);
        this.drag = {x: event.clientX, y: event.clientY, yaw: this.view.yaw, pitch: this.view.pitch, panX: this.view.panX, panY: this.view.panY, pan: event.shiftKey || event.button === 1};
      });
      this.canvas.addEventListener('pointermove', event => {
        if (!this.drag) return;
        const dx = event.clientX - this.drag.x, dy = event.clientY - this.drag.y;
        if (this.drag.pan) { this.view.panX = this.drag.panX + dx; this.view.panY = this.drag.panY + dy; }
        else { this.view.yaw = this.drag.yaw + dx * 0.008; this.view.pitch = clamp(this.drag.pitch + dy * 0.008, -Math.PI / 2 + 0.02, Math.PI / 2 - 0.02); }
        this.draw();
      });
      const endDrag = () => { this.drag = null; };
      this.canvas.addEventListener('pointerup', endDrag);
      this.canvas.addEventListener('pointercancel', endDrag);
      this.canvas.addEventListener('wheel', event => { event.preventDefault(); this.view.zoom = clamp(this.view.zoom * Math.exp(-event.deltaY * 0.001), 0.25, 6); this.draw(); }, {passive: false});
      this.canvas.addEventListener('dblclick', () => { this.view = {yaw: -0.58, pitch: -0.40, zoom: 1, panX: 0, panY: 0}; this.draw(); });
      this.canvas.addEventListener('keydown', event => {
        if (event.code === 'Space') { event.preventDefault(); this.togglePlay(); }
        else if (event.key === 'ArrowLeft') { event.preventDefault(); this.step(event.shiftKey ? -10 : -1); }
        else if (event.key === 'ArrowRight') { event.preventDefault(); this.step(event.shiftKey ? 10 : 1); }
      });
    }

    updateModeUi() {
      this.root.querySelectorAll('[data-preview-mode]').forEach(button => button.classList.toggle('active', button.dataset.previewMode === this.mode));
      this.root.querySelectorAll('.gaze3d-single-layer').forEach(row => row.classList.toggle('hidden', this.mode === 'dyadic'));
      this.root.querySelectorAll('.gaze3d-dyadic-layer').forEach(row => row.classList.toggle('hidden', this.mode !== 'dyadic' || !this.dyadic));
      this.dyadicQc?.classList.toggle('hidden', this.mode !== 'dyadic' || !this.dyadic);
    }

    timeline() {
      if (this.mode === 'dyadic' && this.dyadic?.samples?.length) return {type: 'dyadic', samples: this.dyadic.samples};
      if (this.assignment?.samples?.length) return {type: 'assignment', samples: this.assignment.samples};
      if (this.gaze?.samples?.length) return {type: 'gaze', samples: this.gaze.samples};
      return {type: 'none', samples: []};
    }
    configureTimeline() {
      const samples = this.timeline().samples;
      this.index = clamp(this.index, 0, Math.max(0, samples.length - 1)); this.playCursor = this.index;
      this.scrubber.max = String(Math.max(0, samples.length - 1)); this.scrubber.value = String(this.index); this.scrubber.disabled = samples.length <= 1;
    }
    step(amount) { const samples = this.timeline().samples; if (!samples.length) return; this.index = (this.index + amount + samples.length) % samples.length; this.playCursor = this.index; this.scrubber.value = String(this.index); this.draw(); }
    togglePlay() {
      const samples = this.timeline().samples; if (samples.length <= 1) return;
      this.playing = !this.playing; this.playButton.textContent = this.playing ? '❚❚' : '▶'; this.playButton.setAttribute('aria-label', this.playing ? 'Pause preview' : 'Play preview');
      if (this.playing) { this.lastTick = performance.now(); this.animationFrame = requestAnimationFrame(time => this.tick(time)); }
      else if (this.animationFrame) { cancelAnimationFrame(this.animationFrame); this.animationFrame = 0; }
    }
    timelineRateHz() {
      const timeline = this.timeline(), samples = timeline.samples;
      if (samples.length < 2) return 1;
      const key = timeline.type === 'dyadic' ? 'time_sec' : timeline.type === 'assignment' ? 'time_sec' : (samples[0].local_time_sec != null ? 'local_time_sec' : 'c3d_time_sec');
      const deltas = [];
      for (let i = 1; i < Math.min(samples.length, 120); i += 1) {
        const delta = Number(samples[i]?.[key]) - Number(samples[i - 1]?.[key]);
        if (Number.isFinite(delta) && delta > 0) deltas.push(delta);
      }
      if (!deltas.length) return 1;
      deltas.sort((a, b) => a - b);
      return 1 / deltas[Math.floor(deltas.length / 2)];
    }
    tick(time) {
      if (!this.playing) return;
      const samples = this.timeline().samples; const delta = Math.min(0.12, Math.max(0, (time - this.lastTick) / 1000)); this.lastTick = time;
      this.playCursor += delta * this.timelineRateHz() * Number(this.speed.value || 1); if (this.playCursor >= samples.length) this.playCursor %= samples.length;
      this.index = Math.floor(this.playCursor); this.scrubber.value = String(this.index); this.draw(); this.animationFrame = requestAnimationFrame(next => this.tick(next));
    }
    resize() {
      const rect = this.canvas.getBoundingClientRect(); const ratio = Math.min(2, window.devicePixelRatio || 1);
      const width = Math.max(1, Math.round(rect.width * ratio)), height = Math.max(1, Math.round(rect.height * ratio));
      if (this.canvas.width !== width || this.canvas.height !== height) { this.canvas.width = width; this.canvas.height = height; }
      this.draw();
    }
    bounds() { return combineBounds(this.assignment?.bounds_mm, this.gaze?.bounds_mm, this.dyadic?.bounds_mm); }
    projector() {
      const rect = this.canvas.getBoundingClientRect(), ratio = this.canvas.width / Math.max(1, rect.width), width = this.canvas.width, height = this.canvas.height;
      const bounds = this.bounds(), centre = bounds.centre, maxSpan = Math.max(...bounds.span, 1), scale = Math.min(width, height) * 0.68 / maxSpan * this.view.zoom;
      const cy = Math.cos(this.view.yaw), sy = Math.sin(this.view.yaw), cp = Math.cos(this.view.pitch), sp = Math.sin(this.view.pitch);
      return point => {
        if (!finitePoint(point)) return null;
        const x = point[0] - centre[0], y = point[1] - centre[1], z = point[2] - centre[2];
        const x1 = cy * x - sy * y, y1 = sy * x + cy * y, y2 = cp * y1 - sp * z, z2 = sp * y1 + cp * z;
        const perspective = clamp(1 + y2 / (maxSpan * 4), 0.65, 1.5);
        return {x: width / 2 + x1 * scale / perspective + this.view.panX * ratio, y: height / 2 - z2 * scale / perspective + this.view.panY * ratio, depth: y2, ratio};
      };
    }
    drawEmpty() {
      const ctx = this.context, width = this.canvas.width, height = this.canvas.height; ctx.clearRect(0, 0, width, height);
      const gradient = ctx.createLinearGradient(0, 0, width, height); gradient.addColorStop(0, '#34372a'); gradient.addColorStop(1, '#1e2119'); ctx.fillStyle = gradient; ctx.fillRect(0, 0, width, height);
    }
    draw() {
      if (!this.context || !this.canvas.width) return;
      const ctx = this.context, width = this.canvas.width, height = this.canvas.height; ctx.clearRect(0, 0, width, height);
      const gradient = ctx.createLinearGradient(0, 0, width, height); gradient.addColorStop(0, '#34372a'); gradient.addColorStop(1, '#171a14'); ctx.fillStyle = gradient; ctx.fillRect(0, 0, width, height);
      const project = this.projector(); this.drawGrid(ctx, project);
      const timeline = this.timeline(), sample = timeline.samples[this.index] || null;
      if (this.assignment?.samples?.length) this.drawAssignment(ctx, project, sample, timeline.type);
      if (this.mode === 'dyadic' && this.dyadic?.samples?.length) this.drawDyadicGaze(ctx, project, sample);
      else if (this.layers.gaze && this.gaze?.samples?.length) this.drawSingleGaze(ctx, project, sample, timeline.type);
      this.updateReadout(sample, timeline.type);
    }
    drawGrid(ctx, project) {
      const bounds = this.bounds(), z = bounds.min[2], divisions = 8; ctx.lineWidth = Math.max(1, this.canvas.width / 1600);
      for (let i = 0; i <= divisions; i += 1) {
        const fraction = i / divisions, x = bounds.min[0] + bounds.span[0] * fraction, y = bounds.min[1] + bounds.span[1] * fraction;
        const a = project([x, bounds.min[1], z]), b = project([x, bounds.max[1], z]), c = project([bounds.min[0], y, z]), d = project([bounds.max[0], y, z]);
        ctx.strokeStyle = i === Math.floor(divisions / 2) ? 'rgba(239,253,63,.18)' : 'rgba(193,199,170,.10)';
        if (a && b) { ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); }
        if (c && d) { ctx.beginPath(); ctx.moveTo(c.x, c.y); ctx.lineTo(d.x, d.y); ctx.stroke(); }
      }
      const centre = bounds.centre, axisLength = Math.max(...bounds.span) * 0.16, origin = project([centre[0], centre[1], z]); if (!origin) return;
      [['X', [centre[0] + axisLength, centre[1], z], '#effd3f'], ['Y', [centre[0], centre[1] + axisLength, z], '#7def59'], ['Z', [centre[0], centre[1], z + axisLength], '#78c7ff']].forEach(([label, point, color]) => {
        const end = project(point); if (!end) return; ctx.strokeStyle = color; ctx.lineWidth = 2 * origin.ratio; ctx.beginPath(); ctx.moveTo(origin.x, origin.y); ctx.lineTo(end.x, end.y); ctx.stroke(); ctx.fillStyle = color; ctx.font = `${10 * origin.ratio}px system-ui`; ctx.fillText(label, end.x + 4 * origin.ratio, end.y - 4 * origin.ratio);
      });
    }
    drawTrail(ctx, project, points, color, endIndex) {
      if (!this.layers.trails || !points.length) return; const start = Math.max(0, endIndex - 150); ctx.strokeStyle = color; ctx.globalAlpha = 0.34; ctx.lineWidth = Math.max(1.2, this.canvas.width / 1200); ctx.beginPath(); let started = false;
      for (let i = start; i <= endIndex; i += 1) { const screen = project(points[i]); if (!screen) { started = false; continue; } if (!started) { ctx.moveTo(screen.x, screen.y); started = true; } else ctx.lineTo(screen.x, screen.y); }
      ctx.stroke(); ctx.globalAlpha = 1;
    }
    assignmentTargetTime(timelineSample, timelineType) {
      if (!timelineSample) return {time: NaN, local: false};
      if (timelineType === 'dyadic') return {time: Number(timelineSample.time_sec), local: true};
      if (timelineType === 'gaze') return {time: Number(timelineSample.c3d_time_sec), local: false};
      return {time: Number(timelineSample.time_sec), local: false};
    }
    drawAssignment(ctx, project, timelineSample, timelineType) {
      const samples = this.assignment.samples; let assignmentIndex = this.index;
      if (timelineType !== 'assignment' && timelineSample) { const target = this.assignmentTargetTime(timelineSample, timelineType); assignmentIndex = this.nearestAssignmentIndex(target.time, target.local); }
      const sample = samples[assignmentIndex] || samples[0], roles = Object.keys(this.assignment.roles || {}), etms = Object.keys(this.assignment.etms || {});
      roles.forEach((role, index) => this.drawTrail(ctx, project, samples.map(row => row.roles?.[role] || null), ROLE_COLORS[index % ROLE_COLORS.length], assignmentIndex));
      etms.forEach((etm, index) => this.drawTrail(ctx, project, samples.map(row => row.etms?.[etm] || null), ETM_COLORS[index % ETM_COLORS.length], assignmentIndex));
      if (this.layers.alternatives) etms.forEach(etm => roles.forEach(role => { if (this.assignment.etms?.[etm]?.assigned_role_id !== role) this.drawLink(ctx, project, sample.etms?.[etm], sample.roles?.[role], 'rgba(255,255,255,.22)', true, null); }));
      roles.forEach((role, index) => { const metadata = this.assignment.roles?.[role] || {}, etm = metadata.assigned_etm_id, distance = sample.assigned_distances_mm?.[role]; this.drawLink(ctx, project, sample.roles?.[role], sample.etms?.[etm], ROLE_COLORS[index % ROLE_COLORS.length], false, Number.isFinite(distance) ? `${distance.toFixed(1)} mm` : null); });
      roles.forEach((role, index) => this.drawNode(ctx, project, sample.roles?.[role], role, ROLE_COLORS[index % ROLE_COLORS.length], 'role'));
      etms.forEach((etm, index) => this.drawNode(ctx, project, sample.etms?.[etm], etm, ETM_COLORS[index % ETM_COLORS.length], 'etm'));
    }
    nearestAssignmentIndex(timeSec, local = false) {
      const samples = this.assignment?.samples || []; if (!samples.length || !Number.isFinite(timeSec)) return 0;
      const first = Number(samples[0].time_sec) || 0;
      const valueAt = index => Number(samples[index].time_sec) - (local ? first : 0);
      let low = 0, high = samples.length - 1; while (low < high) { const middle = Math.floor((low + high) / 2); if (valueAt(middle) < timeSec) low = middle + 1; else high = middle; }
      const previous = Math.max(0, low - 1); return Math.abs(valueAt(low) - timeSec) < Math.abs(valueAt(previous) - timeSec) ? low : previous;
    }
    drawSingleGaze(ctx, project, timelineSample, timelineType) {
      const targetTime = timelineType === 'assignment' ? Number(timelineSample?.time_sec) : Number(timelineSample?.c3d_time_sec);
      const gazeSample = timelineType === 'gaze' ? timelineSample : nearestByTime(this.gaze.samples, targetTime, 'c3d_time_sec');
      if (gazeSample) this.drawRay(ctx, project, gazeSample, '#effd3f');
    }
    drawDyadicGaze(ctx, project, timelineSample) {
      if (!timelineSample?.roles) return;
      const roles = this.dyadic.role_order || Object.keys(timelineSample.roles).sort(roleSort);
      roles.forEach(role => {
        if (this.layers.roleRays[role] === false) return;
        const sample = timelineSample.roles[role]; if (!sample?.available) return;
        this.drawRay(ctx, project, sample, roleColor(role, roles), role);
      });
    }
    drawRay(ctx, project, sample, color, label = '') {
      const origin = project(sample.origin_mm), endpoint = project(sample.endpoint_mm); if (!origin || !endpoint) return;
      const ratio = origin.ratio; ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = 3 * ratio; ctx.shadowColor = color; ctx.shadowBlur = 10 * ratio; ctx.beginPath(); ctx.moveTo(origin.x, origin.y); ctx.lineTo(endpoint.x, endpoint.y); ctx.stroke(); ctx.shadowBlur = 0;
      const angle = Math.atan2(endpoint.y - origin.y, endpoint.x - origin.x), head = 10 * ratio; ctx.fillStyle = color; ctx.beginPath(); ctx.moveTo(endpoint.x, endpoint.y); ctx.lineTo(endpoint.x - head * Math.cos(angle - .45), endpoint.y - head * Math.sin(angle - .45)); ctx.lineTo(endpoint.x - head * Math.cos(angle + .45), endpoint.y - head * Math.sin(angle + .45)); ctx.closePath(); ctx.fill();
      ctx.fillStyle = '#ffffff'; ctx.beginPath(); ctx.arc(origin.x, origin.y, 5 * ratio, 0, Math.PI * 2); ctx.fill();
      if (label) { ctx.font = `700 ${9 * ratio}px system-ui`; ctx.fillStyle = color; ctx.fillText(label, endpoint.x + 7 * ratio, endpoint.y - 7 * ratio); }
      ctx.restore();
    }
    drawLink(ctx, project, a, b, color, dashed, label) {
      const start = project(a), end = project(b); if (!start || !end) return; const ratio = start.ratio; ctx.save(); ctx.strokeStyle = color; ctx.globalAlpha = dashed ? .55 : .86; ctx.lineWidth = (dashed ? 1.2 : 2.2) * ratio; if (dashed) ctx.setLineDash([6 * ratio, 5 * ratio]); ctx.beginPath(); ctx.moveTo(start.x, start.y); ctx.lineTo(end.x, end.y); ctx.stroke(); const screenDistance = Math.hypot(end.x - start.x, end.y - start.y);
      if (label && screenDistance > 82 * ratio) { const x = (start.x + end.x) / 2, y = (start.y + end.y) / 2; ctx.setLineDash([]); ctx.globalAlpha = 1; ctx.font = `${9 * ratio}px system-ui`; const width = ctx.measureText(label).width + 10 * ratio; ctx.fillStyle = 'rgba(25,28,20,.78)'; ctx.beginPath(); ctx.roundRect(x - width / 2, y - 10 * ratio, width, 18 * ratio, 6 * ratio); ctx.fill(); ctx.fillStyle = color; ctx.textAlign = 'center'; ctx.fillText(label, x, y + 3 * ratio); ctx.textAlign = 'start'; }
      ctx.restore();
    }
    drawNode(ctx, project, point, label, color, kind) {
      const screen = project(point); if (!screen) return; const ratio = screen.ratio; ctx.save(); ctx.shadowColor = color; ctx.shadowBlur = 9 * ratio; ctx.fillStyle = color; ctx.strokeStyle = '#202319'; ctx.lineWidth = 2 * ratio;
      if (kind === 'etm') { const size = 8 * ratio; ctx.beginPath(); ctx.moveTo(screen.x, screen.y - size); ctx.lineTo(screen.x + size, screen.y); ctx.lineTo(screen.x, screen.y + size); ctx.lineTo(screen.x - size, screen.y); ctx.closePath(); ctx.fill(); ctx.stroke(); }
      else { ctx.beginPath(); ctx.arc(screen.x, screen.y, 9 * ratio, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); }
      ctx.shadowBlur = 0; ctx.font = `700 ${10 * ratio}px system-ui`; const textWidth = ctx.measureText(label).width, labelX = kind === 'role' ? screen.x - textWidth - 23 * ratio : screen.x + 11 * ratio; ctx.fillStyle = 'rgba(22,25,18,.84)'; ctx.beginPath(); ctx.roundRect(labelX, screen.y - 10 * ratio, textWidth + 12 * ratio, 20 * ratio, 6 * ratio); ctx.fill(); ctx.fillStyle = '#f6f7ee'; ctx.fillText(label, labelX + 6 * ratio, screen.y + 4 * ratio); ctx.restore();
    }
    renderLegend() {
      const rows = [];
      Object.keys(this.assignment?.roles || {}).forEach((role, index) => rows.push(`<span><i style="--legend:${ROLE_COLORS[index % ROLE_COLORS.length]}"></i>${role} head</span>`));
      Object.keys(this.assignment?.etms || {}).forEach((etm, index) => rows.push(`<span><i class="diamond" style="--legend:${ETM_COLORS[index % ETM_COLORS.length]}"></i>${etm}</span>`));
      if (this.mode === 'dyadic' && this.dyadic) {
        const roles = this.dyadic.role_order || Object.keys(this.dyadic.roles || {}).sort(roleSort);
        roles.forEach(role => rows.push(`<span><i style="--legend:${roleColor(role, roles)}"></i>${role} gaze ray</span>`));
      } else if (this.gaze?.samples?.length) rows.push('<span><i style="--legend:#effd3f"></i>Calibrated gaze</span>');
      this.legend.innerHTML = rows.join('');
    }
    renderDyadicQc() {
      if (!this.dyadicQc || !this.dyadic) return;
      const pct = `${(Number(this.dyadic.common_valid_pair_fraction || 0) * 100).toFixed(1)}%`;
      const delta = this.dyadic.median_pair_delta_ms == null ? '—' : `${Number(this.dyadic.median_pair_delta_ms).toFixed(1)} ms`;
      this.dyadicQc.innerHTML = `<b>Dyadic overlap ${Number(this.dyadic.common_overlap_duration_sec || 0).toFixed(1)} s</b><span>Both rays valid ${pct} · median pair Δ ${delta} · tolerance ${Number(this.dyadic.pairing_tolerance_ms || 0).toFixed(1)} ms</span>`;
    }
    updateReadout(sample, timelineType) {
      if (!sample) return;
      if (timelineType === 'dyadic') {
        const roles = this.dyadic.role_order || Object.keys(sample.roles || {}).sort(roleSort);
        this.timecode.textContent = `${Number(sample.time_sec).toFixed(3)} s · shared local time`;
        const cards = roles.map(role => {
          const row = sample.roles?.[role] || {}, color = roleColor(role, roles), meta = this.dyadic.roles?.[role] || {};
          const detail = row.available
            ? `sample ${Number(row.sample_time_sec).toFixed(3)} s · Δ ${Number(row.delta_ms).toFixed(1)} ms · ${meta.calibration_hardware_id || 'calibrated ETM'}`
            : `${row.reason || 'ray unavailable'}${row.delta_ms == null ? '' : ` · nearest Δ ${Number(row.delta_ms).toFixed(1)} ms`}`;
          return `<div class="${row.available ? '' : 'unavailable'}"><i style="--readout:${color}"></i><span><b>${role} ray · ${row.available ? 'valid' : 'unavailable'}</b><small>${detail}</small></span></div>`;
        }).join('');
        const pair = `<div class="gaze3d-pair-readout ${sample.both_valid ? 'valid' : 'unavailable'}"><i style="--readout:${sample.both_valid ? '#ffffff' : '#d39b75'}"></i><span><b>Paired sample · ${sample.both_valid ? 'both valid' : 'partial'}</b><small>${sample.pair_delta_ms == null ? 'No two-sided pair at this preview time' : `R01/R02 sample Δ ${Number(sample.pair_delta_ms).toFixed(1)} ms`}</small></span></div>`;
        this.readout.innerHTML = cards + pair;
        return;
      }
      const time = timelineType === 'assignment' ? Number(sample.time_sec) : Number(sample.c3d_time_sec), frame = timelineType === 'assignment' ? sample.frame : sample.c3d_frame_float;
      this.timecode.textContent = `${Number.isFinite(time) ? time.toFixed(3) + ' s' : '—'} · ${frame == null ? 'frame —' : 'frame ' + Number(frame).toFixed(Number.isInteger(frame) ? 0 : 2)}`;
      if (this.assignment?.samples?.length) {
        const assignmentIndex = timelineType === 'assignment' ? this.index : this.nearestAssignmentIndex(time, false), assignmentSample = this.assignment.samples[assignmentIndex];
        this.readout.innerHTML = Object.entries(this.assignment.roles || {}).map(([role, metadata], index) => { const current = assignmentSample?.assigned_distances_mm?.[role], currentText = Number.isFinite(current) ? `${current.toFixed(1)} mm now` : 'missing at this frame', median = Number(metadata.median_head_distance_mm); return `<div><i style="--readout:${ROLE_COLORS[index % ROLE_COLORS.length]}"></i><span><b>${role} → ${metadata.assigned_etm_id || '—'}</b><small>${currentText} · median ${Number.isFinite(median) ? median.toFixed(1) + ' mm' : '—'} · ${metadata.confidence || 'review'}</small></span></div>`; }).join('');
      } else if (this.gaze) this.readout.innerHTML = `<div><i style="--readout:#effd3f"></i><span><b>${this.gaze.wearer_role_id || 'Gaze 3D'}</b><small>${this.gaze.source_valid_ray_count || 0} valid rays · ${this.gaze.sample_count || 0} preview samples</small></span></div>`;
    }
  }

  function bind() {
    const root = document.querySelector('#gaze3d-live-preview');
    if (!root || root.dataset.bound === 'true') return;
    root.dataset.bound = 'true'; new Preview(root);
  }

  window.OculoGazePreview = {markup, bind};
})();
