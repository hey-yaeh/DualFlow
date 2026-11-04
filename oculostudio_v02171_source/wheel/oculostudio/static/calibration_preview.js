(() => {
  'use strict';

  const COLORS = {
    board: '#effd3f',
    tracker_1: '#78c7ff',
    tracker_2: '#ff9d68',
    tracker_3: '#caa5ff',
    tracker_4: '#7def59',
    world: '#c6c9b8',
    camera: '#ff7bd5',
  };
  const instances = new WeakMap();

  function markup({sessionId = ''} = {}) {
    return `<section class="ray-stage calibration-live-stage" id="calibration-live-preview" data-session-id="${String(sessionId || '')}">
      <canvas class="calibration-preview-canvas" tabindex="0" aria-label="Interactive calibration-session C3D preview"></canvas>
      <div class="calibration-preview-loading"><span class="gaze3d-loading-dot"></span><strong>Select calibration-session sources</strong><small>Choose a board C3D and one or more tracker candidates in the right panel. The preview reads real marker trajectories.</small></div>
      <div class="calibration-preview-toolbar" aria-label="Calibration preview controls">
        <div class="calibration-view-buttons">
          <button type="button" data-view="perspective" class="active">Perspective</button>
          <button type="button" data-view="front">Front</button>
          <button type="button" data-view="top">Top</button>
        </div>
        <label><input type="checkbox" data-layer="trails" checked> Trails</label>
        <label><input type="checkbox" data-layer="world" checked> World</label>
        <label class="calibration-camera-layer"><input type="checkbox" data-layer="camera" checked> Camera transform</label>
        <button type="button" data-action="reset" title="Reset view">Reset</button>
        <button type="button" data-action="reload" title="Reload selected sources">Reload</button>
      </div>
      <div class="calibration-preview-legend" aria-label="Calibration preview legend"></div>
      <div class="calibration-live-readout" aria-live="polite"></div>
      <div class="calibration-preview-controls">
        <button type="button" class="calibration-play" data-action="play" aria-label="Play preview">▶</button>
        <button type="button" class="calibration-step" data-action="previous" aria-label="Previous sample">‹</button>
        <input type="range" class="calibration-scrubber" min="0" max="0" value="0" step="1" aria-label="Calibration preview frame">
        <button type="button" class="calibration-step" data-action="next" aria-label="Next sample">›</button>
        <span class="calibration-timecode">—</span>
        <select class="calibration-speed" aria-label="Preview speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select>
      </div>
      <div class="calibration-preview-help">Drag to orbit · mouse wheel to zoom · Shift-drag to pan · Space to play/pause</div>
    </section>`;
  }

  function clamp(value, minimum, maximum) {
    return Math.max(minimum, Math.min(maximum, value));
  }

  function finitePoint(point) {
    return Array.isArray(point) && point.length === 3 && point.every(value => Number.isFinite(Number(value)));
  }

  function distance(a, b) {
    if (!finitePoint(a) || !finitePoint(b)) return null;
    return Math.hypot(Number(a[0]) - Number(b[0]), Number(a[1]) - Number(b[1]), Number(a[2]) - Number(b[2]));
  }

  function fmtTime(value) {
    const total = Math.max(0, Number(value) || 0);
    const minutes = Math.floor(total / 60);
    const seconds = total - minutes * 60;
    return `${String(minutes).padStart(2, '0')}:${seconds.toFixed(3).padStart(6, '0')}`;
  }

  function roleColor(role, index = 0) {
    return COLORS[role] || [COLORS.tracker_1, COLORS.tracker_2, COLORS.tracker_3, COLORS.tracker_4][index % 4];
  }

  function convexHull(points) {
    if (points.length <= 2) return points.slice();
    const sorted = points.slice().sort((a, b) => a.x === b.x ? a.y - b.y : a.x - b.x);
    const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    const lower = [];
    for (const point of sorted) {
      while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) lower.pop();
      lower.push(point);
    }
    const upper = [];
    for (let i = sorted.length - 1; i >= 0; i -= 1) {
      const point = sorted[i];
      while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) upper.pop();
      upper.push(point);
    }
    lower.pop(); upper.pop();
    return lower.concat(upper);
  }

  class CalibrationPreview {
    constructor(root) {
      this.root = root;
      this.canvas = root.querySelector('canvas');
      this.context = this.canvas.getContext('2d');
      this.loading = root.querySelector('.calibration-preview-loading');
      this.legend = root.querySelector('.calibration-preview-legend');
      this.readout = root.querySelector('.calibration-live-readout');
      this.scrubber = root.querySelector('.calibration-scrubber');
      this.timecode = root.querySelector('.calibration-timecode');
      this.playButton = root.querySelector('[data-action="play"]');
      this.speed = root.querySelector('.calibration-speed');
      this.data = null;
      this.config = {};
      this.index = 0;
      this.playing = false;
      this.playCursor = 0;
      this.lastTick = 0;
      this.animationFrame = 0;
      this.request = null;
      this.view = {yaw: -0.58, pitch: -0.42, zoom: 1, panX: 0, panY: 0};
      this.layers = {trails: true, world: true, camera: true};
      this.drag = null;
      this.resizeObserver = new ResizeObserver(() => this.resize());
      this.resizeObserver.observe(root);
      this.bindControls();
      this.resize();
    }

    readResponse = async response => {
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || data.message || `Calibration preview failed (${response.status})`);
      return data;
    };

    async load(config = {}) {
      const normalized = {
        sessionId: String(config.sessionId || this.root.dataset.sessionId || ''),
        boardId: String(config.boardId || ''),
        trackerIds: [...new Set((config.trackerIds || []).map(String).filter(Boolean))],
        worldId: String(config.worldId || ''),
        profileId: String(config.profileId || ''),
      };
      const key = JSON.stringify(normalized);
      if (key === JSON.stringify(this.config) && this.data) return;
      this.config = normalized;
      if (this.request) this.request.abort();
      this.request = new AbortController();
      this.setLoading('Loading real calibration geometry…', 'Board, tracker, world and profile data are sampled without changing the source files.');
      if (!normalized.sessionId || (!normalized.boardId && !normalized.trackerIds.length && !normalized.worldId)) {
        this.data = null;
        this.fail('Select a calibration-board C3D and one or more tracker candidates.');
        return;
      }
      const params = new URLSearchParams();
      if (normalized.boardId) params.set('board_asset_id', normalized.boardId);
      normalized.trackerIds.forEach(value => params.append('tracker_asset_id', value));
      if (normalized.worldId) params.set('world_asset_id', normalized.worldId);
      if (normalized.profileId) params.set('profile_artifact_id', normalized.profileId);
      params.set('maximum_samples', '600');
      try {
        const url = `/api/preview/calibration/${encodeURIComponent(normalized.sessionId)}?${params.toString()}`;
        this.data = await fetch(url, {cache: 'no-store', signal: this.request.signal}).then(this.readResponse);
        this.loading?.remove();
        this.loading = null;
        this.root.classList.add('ready');
        this.index = 0;
        this.playCursor = 0;
        this.configureTimeline();
        this.renderLegend();
        const cameraLayer = this.root.querySelector('.calibration-camera-layer');
        if (cameraLayer) cameraLayer.classList.toggle('hidden', !this.data?.profile?.available);
        this.draw();
      } catch (error) {
        if (error?.name === 'AbortError') return;
        this.data = null;
        this.fail(error?.message || 'Calibration preview could not be loaded.');
      }
    }

    setLoading(title, detail = '') {
      if (!this.loading || !this.loading.isConnected) {
        this.loading = document.createElement('div');
        this.loading.className = 'calibration-preview-loading';
        this.root.appendChild(this.loading);
      }
      this.loading.classList.remove('error');
      this.loading.innerHTML = `<span class="gaze3d-loading-dot"></span><strong>${String(title)}</strong><small>${String(detail)}</small>`;
    }

    fail(message) {
      this.setLoading('Preview unavailable', message);
      this.loading?.classList.add('error');
      this.drawEmpty();
    }

    bindControls() {
      this.root.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => {
        this.root.querySelectorAll('[data-view]').forEach(row => row.classList.toggle('active', row === button));
        const view = button.dataset.view;
        if (view === 'front') this.view = {...this.view, yaw: 0, pitch: 0, zoom: 1, panX: 0, panY: 0};
        else if (view === 'top') this.view = {...this.view, yaw: 0, pitch: -Math.PI / 2 + 0.001, zoom: 1, panX: 0, panY: 0};
        else this.view = {...this.view, yaw: -0.58, pitch: -0.42, zoom: 1, panX: 0, panY: 0};
        this.draw();
      }));
      this.root.querySelectorAll('[data-layer]').forEach(input => input.addEventListener('change', () => {
        this.layers[input.dataset.layer] = input.checked;
        this.draw();
      }));
      this.root.querySelector('[data-action="reset"]')?.addEventListener('click', () => {
        this.view = {yaw: -0.58, pitch: -0.42, zoom: 1, panX: 0, panY: 0};
        this.root.querySelectorAll('[data-view]').forEach(row => row.classList.toggle('active', row.dataset.view === 'perspective'));
        this.draw();
      });
      this.root.querySelector('[data-action="reload"]')?.addEventListener('click', () => {
        const current = {...this.config}; this.config = {}; this.load(current);
      });
      this.playButton?.addEventListener('click', () => this.togglePlay());
      this.root.querySelector('[data-action="previous"]')?.addEventListener('click', () => this.step(-1));
      this.root.querySelector('[data-action="next"]')?.addEventListener('click', () => this.step(1));
      this.scrubber?.addEventListener('input', () => {
        this.index = Number(this.scrubber.value) || 0;
        this.playCursor = this.index;
        this.draw();
      });
      this.canvas.addEventListener('pointerdown', event => {
        this.canvas.setPointerCapture?.(event.pointerId);
        this.drag = {x: event.clientX, y: event.clientY, yaw: this.view.yaw, pitch: this.view.pitch, panX: this.view.panX, panY: this.view.panY, pan: event.shiftKey || event.button === 1};
      });
      this.canvas.addEventListener('pointermove', event => {
        if (!this.drag) return;
        const dx = event.clientX - this.drag.x, dy = event.clientY - this.drag.y;
        if (this.drag.pan) {
          this.view.panX = this.drag.panX + dx;
          this.view.panY = this.drag.panY + dy;
        } else {
          this.view.yaw = this.drag.yaw + dx * 0.008;
          this.view.pitch = clamp(this.drag.pitch + dy * 0.008, -Math.PI / 2 + 0.02, Math.PI / 2 - 0.02);
        }
        this.draw();
      });
      const stopDrag = () => { this.drag = null; };
      this.canvas.addEventListener('pointerup', stopDrag);
      this.canvas.addEventListener('pointercancel', stopDrag);
      this.canvas.addEventListener('wheel', event => {
        event.preventDefault();
        this.view.zoom = clamp(this.view.zoom * Math.exp(-event.deltaY * 0.001), 0.2, 8);
        this.draw();
      }, {passive: false});
      this.canvas.addEventListener('dblclick', () => this.root.querySelector('[data-action="reset"]')?.click());
      this.canvas.addEventListener('keydown', event => {
        if (event.key === ' ') { event.preventDefault(); this.togglePlay(); }
        else if (event.key === 'ArrowLeft') { event.preventDefault(); this.step(event.shiftKey ? -10 : -1); }
        else if (event.key === 'ArrowRight') { event.preventDefault(); this.step(event.shiftKey ? 10 : 1); }
      });
    }

    configureTimeline() {
      const samples = this.data?.samples || [];
      this.index = clamp(this.index, 0, Math.max(0, samples.length - 1));
      this.scrubber.max = String(Math.max(0, samples.length - 1));
      this.scrubber.value = String(this.index);
      this.scrubber.disabled = samples.length <= 1;
    }

    step(amount) {
      const samples = this.data?.samples || [];
      if (!samples.length) return;
      this.index = (this.index + amount + samples.length) % samples.length;
      this.playCursor = this.index;
      this.scrubber.value = String(this.index);
      this.draw();
    }

    togglePlay() {
      const samples = this.data?.samples || [];
      if (samples.length <= 1) return;
      this.playing = !this.playing;
      this.playButton.textContent = this.playing ? '❚❚' : '▶';
      if (this.playing) {
        this.lastTick = performance.now();
        this.animationFrame = requestAnimationFrame(time => this.tick(time));
      } else if (this.animationFrame) {
        cancelAnimationFrame(this.animationFrame); this.animationFrame = 0;
      }
    }

    tick(time) {
      if (!this.playing) return;
      const samples = this.data?.samples || [];
      const delta = Math.min(0.12, Math.max(0, (time - this.lastTick) / 1000));
      this.lastTick = time;
      this.playCursor += delta * 30 * Number(this.speed.value || 1);
      if (this.playCursor >= samples.length) this.playCursor %= samples.length;
      this.index = Math.floor(this.playCursor);
      this.scrubber.value = String(this.index);
      this.draw();
      this.animationFrame = requestAnimationFrame(next => this.tick(next));
    }

    resize() {
      const rect = this.canvas.getBoundingClientRect();
      const ratio = Math.min(2, window.devicePixelRatio || 1);
      const width = Math.max(1, Math.round(rect.width * ratio));
      const height = Math.max(1, Math.round(rect.height * ratio));
      if (this.canvas.width !== width || this.canvas.height !== height) {
        this.canvas.width = width; this.canvas.height = height;
      }
      this.draw();
    }

    bounds() {
      const row = this.data?.bounds_mm;
      if (row && finitePoint(row.min) && finitePoint(row.max)) return row;
      return {min: [-500, -500, -500], max: [500, 500, 500], centre: [0, 0, 0], span: [1000, 1000, 1000]};
    }

    projector() {
      const rect = this.canvas.getBoundingClientRect();
      const ratio = this.canvas.width / Math.max(1, rect.width);
      const width = this.canvas.width, height = this.canvas.height;
      const bounds = this.bounds();
      const centre = bounds.centre || bounds.min.map((value, axis) => (Number(value) + Number(bounds.max[axis])) / 2);
      const span = bounds.span || bounds.min.map((value, axis) => Math.max(1, Number(bounds.max[axis]) - Number(value)));
      const maxSpan = Math.max(...span.map(Number), 1);
      const scale = Math.min(width, height) * 0.70 / maxSpan * this.view.zoom;
      const cy = Math.cos(this.view.yaw), sy = Math.sin(this.view.yaw);
      const cp = Math.cos(this.view.pitch), sp = Math.sin(this.view.pitch);
      return point => {
        if (!finitePoint(point)) return null;
        const x = Number(point[0]) - Number(centre[0]);
        const y = Number(point[1]) - Number(centre[1]);
        const z = Number(point[2]) - Number(centre[2]);
        const x1 = cy * x - sy * y;
        const y1 = sy * x + cy * y;
        const y2 = cp * y1 - sp * z;
        const z2 = sp * y1 + cp * z;
        const perspective = clamp(1 + y2 / (maxSpan * 4), 0.65, 1.5);
        return {x: width / 2 + x1 * scale / perspective + this.view.panX * ratio, y: height / 2 - z2 * scale / perspective + this.view.panY * ratio, depth: y2, ratio};
      };
    }

    drawEmpty() {
      const ctx = this.context, width = this.canvas.width, height = this.canvas.height;
      ctx.clearRect(0, 0, width, height);
      const gradient = ctx.createLinearGradient(0, 0, width, height);
      gradient.addColorStop(0, '#34372a'); gradient.addColorStop(1, '#171a14');
      ctx.fillStyle = gradient; ctx.fillRect(0, 0, width, height);
    }

    draw() {
      if (!this.context || !this.canvas.width) return;
      const ctx = this.context, width = this.canvas.width, height = this.canvas.height;
      ctx.clearRect(0, 0, width, height);
      const gradient = ctx.createLinearGradient(0, 0, width, height);
      gradient.addColorStop(0, '#34372a'); gradient.addColorStop(1, '#171a14');
      ctx.fillStyle = gradient; ctx.fillRect(0, 0, width, height);
      if (!this.data?.samples?.length) { this.drawGrid(ctx, this.projector()); return; }
      const project = this.projector();
      this.drawGrid(ctx, project);
      const sample = this.data.samples[this.index] || this.data.samples[0];
      const sourceEntries = Object.entries(this.data.sources || {});
      sourceEntries.forEach(([assetId, meta], index) => {
        if (meta.role === 'world' && !this.layers.world) return;
        this.drawSource(ctx, project, assetId, meta, sample, index);
      });
      if (this.layers.camera && sample.camera) this.drawCamera(ctx, project, sample.camera, sample);
      this.updateReadout(sample);
    }

    drawGrid(ctx, project) {
      const bounds = this.bounds();
      const z = Number(bounds.min?.[2] || 0);
      const divisions = 8;
      ctx.lineWidth = Math.max(1, this.canvas.width / 1600);
      for (let i = 0; i <= divisions; i += 1) {
        const fraction = i / divisions;
        const x = Number(bounds.min[0]) + Number(bounds.span[0]) * fraction;
        const y = Number(bounds.min[1]) + Number(bounds.span[1]) * fraction;
        const a = project([x, bounds.min[1], z]), b = project([x, bounds.max[1], z]);
        const c = project([bounds.min[0], y, z]), d = project([bounds.max[0], y, z]);
        ctx.strokeStyle = i === Math.floor(divisions / 2) ? 'rgba(239,253,63,.18)' : 'rgba(193,199,170,.10)';
        if (a && b) { ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); }
        if (c && d) { ctx.beginPath(); ctx.moveTo(c.x, c.y); ctx.lineTo(d.x, d.y); ctx.stroke(); }
      }
    }

    drawTrail(ctx, project, assetId, color) {
      if (!this.layers.trails) return;
      const samples = this.data.samples || [];
      const start = Math.max(0, this.index - 150);
      ctx.strokeStyle = color; ctx.globalAlpha = 0.32; ctx.lineWidth = Math.max(1.1, this.canvas.width / 1300);
      ctx.beginPath(); let started = false;
      for (let i = start; i <= this.index; i += 1) {
        const point = samples[i]?.sources?.[assetId]?.centre_mm;
        const screen = project(point);
        if (!screen) { started = false; continue; }
        if (!started) { ctx.moveTo(screen.x, screen.y); started = true; }
        else ctx.lineTo(screen.x, screen.y);
      }
      ctx.stroke(); ctx.globalAlpha = 1;
    }

    drawSource(ctx, project, assetId, meta, sample, index) {
      const row = sample.sources?.[assetId]; if (!row) return;
      const color = roleColor(meta.role, index);
      this.drawTrail(ctx, project, assetId, color);
      const points = (row.markers_mm || []).map((point, markerIndex) => {
        const screen = project(point); return screen ? {...screen, markerIndex} : null;
      }).filter(Boolean);
      if (!points.length) return;
      const hull = convexHull(points);
      if (hull.length >= 3) {
        ctx.fillStyle = `${color}18`; ctx.strokeStyle = `${color}99`; ctx.lineWidth = Math.max(1.2, this.canvas.width / 1200);
        ctx.beginPath(); hull.forEach((point, hullIndex) => hullIndex ? ctx.lineTo(point.x, point.y) : ctx.moveTo(point.x, point.y)); ctx.closePath(); ctx.fill(); ctx.stroke();
      }
      const centre = project(row.centre_mm);
      if (centre) {
        ctx.strokeStyle = `${color}65`; ctx.lineWidth = Math.max(0.8, this.canvas.width / 1800);
        points.forEach(point => { ctx.beginPath(); ctx.moveTo(centre.x, centre.y); ctx.lineTo(point.x, point.y); ctx.stroke(); });
      }
      points.forEach(point => {
        ctx.fillStyle = color; ctx.beginPath(); ctx.arc(point.x, point.y, 3.2 * point.ratio, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = 'rgba(0,0,0,.45)'; ctx.lineWidth = point.ratio; ctx.stroke();
      });
      if (centre) {
        ctx.fillStyle = '#171a14'; ctx.strokeStyle = color; ctx.lineWidth = 2 * centre.ratio;
        ctx.beginPath(); ctx.arc(centre.x, centre.y, 7 * centre.ratio, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        ctx.fillStyle = color; ctx.font = `bold ${9 * centre.ratio}px system-ui`;
        ctx.fillText(meta.asset_label, centre.x + 10 * centre.ratio, centre.y - 9 * centre.ratio);
      }
    }

    drawCamera(ctx, project, camera, sample) {
      const origin = project(camera.origin_mm); if (!origin) return;
      const axes = camera.axes_mm || {};
      const colors = {x: '#ff7a74', y: '#7def59', z: '#78c7ff'};
      Object.entries(colors).forEach(([axis, color]) => {
        const end = project(axes[axis]); if (!end) return;
        ctx.strokeStyle = color; ctx.lineWidth = 3 * origin.ratio;
        ctx.beginPath(); ctx.moveTo(origin.x, origin.y); ctx.lineTo(end.x, end.y); ctx.stroke();
        ctx.fillStyle = color; ctx.font = `bold ${9 * origin.ratio}px system-ui`; ctx.fillText(axis.toUpperCase(), end.x + 4 * origin.ratio, end.y - 4 * origin.ratio);
      });
      ctx.fillStyle = COLORS.camera; ctx.beginPath(); ctx.arc(origin.x, origin.y, 5.5 * origin.ratio, 0, Math.PI * 2); ctx.fill();
      const boardEntry = Object.entries(this.data.sources || {}).find(([, meta]) => meta.role === 'board');
      const boardCentre = boardEntry ? sample.sources?.[boardEntry[0]]?.centre_mm : null;
      const target = project(boardCentre);
      if (target) {
        ctx.strokeStyle = 'rgba(255,123,213,.55)'; ctx.setLineDash([7 * origin.ratio, 5 * origin.ratio]); ctx.lineWidth = 1.5 * origin.ratio;
        ctx.beginPath(); ctx.moveTo(origin.x, origin.y); ctx.lineTo(target.x, target.y); ctx.stroke(); ctx.setLineDash([]);
      }
    }

    renderLegend() {
      const rows = Object.values(this.data?.sources || {}).map((meta, index) => {
        const color = roleColor(meta.role, index);
        return `<span><i style="--legend:${color}"></i>${String(meta.asset_label || meta.role)}</span>`;
      });
      if (this.data?.profile) rows.push(`<span><i class="diamond" style="--legend:${COLORS.camera}"></i>${String(this.data.profile.artifact_label || 'Camera transform')}</span>`);
      this.legend.innerHTML = rows.join('');
    }

    updateReadout(sample) {
      if (!sample) { this.readout.innerHTML = ''; this.timecode.textContent = '—'; return; }
      this.timecode.textContent = `${fmtTime(sample.time_sec)} · ${this.index + 1}/${this.data.samples.length}`;
      const rows = [];
      Object.entries(this.data.sources || {}).forEach(([assetId, meta], index) => {
        const current = sample.sources?.[assetId] || {};
        rows.push(`<div><i style="--readout:${roleColor(meta.role, index)}"></i><span><b>${String(meta.asset_label || meta.role)}</b><small>${Number(current.valid_marker_count || 0)}/${Number(meta.marker_count || 0)} markers · frame ${current.frame ?? '—'} · ${(Number(meta.valid_marker_fraction || 0) * 100).toFixed(1)}% overall valid</small></span></div>`);
      });
      if (this.data.profile) {
        const profile = this.data.profile;
        const detail = profile.available
          ? `${profile.hardware_display_name || profile.hardware_id || 'Hardware'} · camera origin ${Array.isArray(profile.camera_origin_tracker_mm) ? profile.camera_origin_tracker_mm.map(value => Number(value).toFixed(1)).join(', ') : '—'} mm${profile.direct_reprojection_rms_px_median != null ? ` · reproj ${Number(profile.direct_reprojection_rms_px_median).toFixed(2)} px` : ''}`
          : profile.reason || 'Profile camera transform cannot be placed with the selected tracker';
        rows.push(`<div><i style="--readout:${COLORS.camera}"></i><span><b>${String(profile.artifact_label || 'Calibration profile')}</b><small>${detail}</small></span></div>`);
      }
      const boardEntry = Object.entries(this.data.sources || {}).find(([, meta]) => meta.role === 'board');
      const trackerEntries = Object.entries(this.data.sources || {}).filter(([, meta]) => String(meta.role).startsWith('tracker_'));
      if (boardEntry) {
        const boardCentre = sample.sources?.[boardEntry[0]]?.centre_mm;
        trackerEntries.forEach(([assetId, meta]) => {
          const trackerCentre = sample.sources?.[assetId]?.centre_mm;
          const value = distance(boardCentre, trackerCentre);
          if (Number.isFinite(value)) rows.push(`<div><i style="--readout:${roleColor(meta.role)}"></i><span><b>Board ↔ ${String(meta.asset_label)}</b><small>${value.toFixed(1)} mm at ${fmtTime(sample.time_sec)}</small></span></div>`);
        });
      }
      this.readout.innerHTML = rows.join('');
    }
  }

  function currentConfig() {
    const root = document.querySelector('#calibration-live-preview');
    if (!root) return null;
    return {
      sessionId: root.dataset.sessionId,
      boardId: document.querySelector('#cal-board')?.value || '',
      trackerIds: [...document.querySelectorAll('.cal-tracker-check:checked')].map(row => row.value),
      worldId: document.querySelector('#cal-world')?.value || '',
      profileId: document.querySelector('#cal-preview-profile')?.value || '',
    };
  }

  function mount() {
    const root = document.querySelector('#calibration-live-preview');
    if (!root) return null;
    let instance = instances.get(root);
    if (!instance) { instance = new CalibrationPreview(root); instances.set(root, instance); }
    const config = currentConfig(); if (config) instance.load(config);
    return instance;
  }

  function update() {
    const instance = mount();
    const config = currentConfig();
    if (instance && config) instance.load(config);
  }

  window.OculoCalibrationPreview = {markup, mount, update};
})();
