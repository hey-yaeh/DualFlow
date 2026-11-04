"""Trim TSD + Face Mapper using an already-edited Native MP4's start/duration.

Independent of global TimeGraph, 3D rays, C3D and body annotation. Point CSV
fields are preserved verbatim. Interval endpoints are clipped explicitly and
original endpoints/statistics remain traceable. There is no video encoder.
"""
from __future__ import annotations

import csv
import html
import io
import json
import math
import re
import shutil
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import numpy as np

from oculostudio import __version__
from oculostudio.adapters.archive import SourceView
from oculostudio.adapters.native_reference_clock import (
    NativeClock, build_clock, load_clock, native_options, open_member, source_stamp,
)
from oculostudio.adapters.pupil import find_cloud_candidates
from oculostudio.domain.models import ArtifactKind
from oculostudio.domain.video_input_time import parse_video_time, parse_native_trim_times
from oculostudio.errors import AssetError, ValidationError
from oculostudio.storage import ProjectStore
from oculostudio.storage.artifact_store import ArtifactStore
from oculostudio.util import (CancellationToken, atomic_write_json, emit,
                             new_id, sha256_file, utc_now)

SCHEMA = 'oculostudio-native-reference-trim-1.0'
POINT_EXTRA = ['oculo_source_row_index', 'oculo_native_video_ns', 'oculo_clip_video_ns',
               'oculo_nearest_native_frame_index', 'oculo_frame_match_error_ns']
FACE_EXTRA = ['oculo_source_row_index', 'oculo_native_frame_index',
              'oculo_source_frame_ordinal_in_window', 'oculo_native_video_ns',
              'oculo_clip_video_ns', 'oculo_frame_match_error_ns', 'oculo_boundary_context']
EVENT_EXTRA = ['oculo_source_row_index', 'oculo_original_start_utc_ns', 'oculo_original_end_utc_ns',
               'oculo_original_duration_ms', 'oculo_retained_start_utc_ns', 'oculo_retained_end_utc_ns',
               'oculo_retained_duration_ns', 'oculo_clip_start_video_ns', 'oculo_clip_end_video_ns',
               'oculo_event_truncated', 'oculo_statistics_scope']


def fmt_ns(ns: int) -> str:
    sign = '-' if ns < 0 else ''
    sec, frac = divmod(abs(int(ns)), 1_000_000_000)
    h, sec = divmod(sec, 3600); m, sec = divmod(sec, 60)
    return f'{sign}{h:02d}:{m:02d}:{sec:02d}.{frac:09d}'


def col_key(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def pick(columns, *keys) -> str | None:
    lookup = {col_key(c): c for c in columns if c is not None}
    return next((lookup[k] for k in keys if k in lookup), None)


def int_ns(value: str, *, where: str) -> int:
    if not re.fullmatch(r'[+-]?\d+', str(value).strip()):
        raise ValidationError(f'Invalid integer nanosecond timestamp in {where}: {str(value)[:60]!r}')
    n = int(value)
    if not -(2**63) < n < 2**63:
        raise ValidationError(f'Timestamp exceeds int64 in {where}')
    return n


@contextmanager
def csv_input(source: str | Path, name: str):
    with open_member(source, name) as binary:
        text = io.TextIOWrapper(binary, encoding='utf-8-sig', newline='')
        try:
            reader = csv.DictReader(text)
            columns = reader.fieldnames or []
            if not columns or len(columns) != len(set(columns)):
                raise ValidationError(f'Missing or duplicate CSV column names: {name}')
            if any(col_key(c).startswith('oculo') for c in columns):
                raise ValidationError(f'Input already contains OculoStudio trim columns: {name}. Use original untrimmed exports.')
            yield reader, columns
        finally:
            text.detach()


def _identities(source: str | Path, name: str, rid: str, *, required: bool = True) -> tuple[list[str], set[str]]:
    sections = set(); ids = set()
    with csv_input(source, name) as (reader, columns):
        ridcol = pick(columns, 'recordingid')
        sidcol = pick(columns, 'sectionid')
        if not ridcol:
            if required:
                raise ValidationError(f'{name} has no recording id; cannot safely select a recording')
            return [], set()
        for row in reader:
            ids.add(row[ridcol].strip())
            if row[ridcol].strip() == rid and sidcol:
                sections.add(row[sidcol])
    return sorted(ids), sections


def resolve_sources(native_path: str, tsd_path: str, fm_path: str, video_member: str | None) -> dict:
    options = native_options(native_path)
    if video_member:
        options = [o for o in options if o['video_member'] == video_member]
    if len(options) != 1:
        raise ValidationError('Choose the exact NRD scene-video part before previewing')
    native = options[0]; rid = native['recording_id']
    with SourceView(tsd_path) as view:
        cloud = [c for c in find_cloud_candidates(view) if c.recording_id == rid]
        if len(cloud) != 1:
            raise ValidationError(f'TSD has {len(cloud)} recordings matching NRD recording id {rid}; expected one')
        c = cloud[0]
        prefix = c.root + '/' if c.root else ''
        names = view.names()
        members = sorted(n for n in names if n.startswith(prefix) and
                         ('' if PurePosixPath(n).parent.as_posix() == '.' else PurePosixPath(n).parent.as_posix()) == c.root)
        # Select global section metadata separately from per-recording tables.
        sections = [n for n in names if PurePosixPath(n).name.lower() == 'sections.csv']
        members = sorted(set(members + sections))
    with SourceView(fm_path) as view:
        fm_names = view.names()
        if len(fm_names) != len(set(fm_names)):
            raise ValidationError('Face Mapper archive contains duplicate filenames')
        fm_sections = [n for n in fm_names if PurePosixPath(n).name.lower() == 'sections.csv']
        if len(fm_sections) != 1:
            raise ValidationError('Face Mapper must contain one unambiguous sections.csv')
        ids, section_ids = _identities(fm_path, fm_sections[0], rid)
        if rid not in ids:
            raise ValidationError('Face Mapper does not contain the NRD/TSD recording id; no data will be exported')
        fm_csvs = [n for n in fm_names if n.lower().endswith('.csv')]
        basenames = [PurePosixPath(n).name.lower() for n in fm_csvs]
        if len(basenames) != len(set(basenames)):
            raise ValidationError('Face Mapper has ambiguous tables with the same basename')
        if not any(n in basenames for n in ('face_detections.csv', 'face_positions.csv')):
            raise ValidationError('No Face Mapper face_detections.csv / face_positions.csv was found')
    return {'recording_id': rid, 'native': native, 'cloud_root': c.root,
            'cloud_members': members, 'fm_members': sorted(fm_csvs),
            'fm_recording_count': len(ids), 'fm_section_ids': sorted(section_ids),
            'gaze_member': c.gaze_name}


@dataclass
class Window:
    clock: NativeClock
    start: int
    end: int
    utc_start: int
    utc_end: int
    frame_ids: np.ndarray
    face_tolerance: int
    event_policy: str

    @classmethod
    def create(cls, clock, start, duration, tolerance_ms, policy):
        end = start + duration
        if start < 0 or duration <= 0 or end > clock.end_media_ns:
            raise ValidationError(f'Requested window {fmt_ns(start)} → {fmt_ns(end)} is outside '
                                  f'NRD duration {fmt_ns(clock.end_media_ns)}. No silent clamping is performed.')
        if policy not in {'clip', 'inside'}:
            raise ValidationError('Event policy must be clip or inside')
        if not math.isfinite(tolerance_ms) or not 0 < tolerance_ms <= 15:
            raise ValidationError('Face-frame matching tolerance must be in (0, 15] ms')
        return cls(clock, start, end, clock.utc_at(start), clock.utc_at(end),
                   clock.overlapping_frames(start, end), int(round(tolerance_ms*1e6)), policy)


class TablePass:
    """Streaming dry-run/export. All original non-edited CSV cells remain strings."""
    def __init__(self, window: Window, rid: str, output: Path | None, token: CancellationToken, progress=None):
        self.w = window; self.rid = rid; self.output = output; self.token = token; self.progress = progress
        self.tables: list[dict] = []
        self.warnings: list[str] = []
        self.gaze = []
        self.eye = []
        self.fm_gaze = defaultdict(list)
        self.fm_detections_frames = set()
        self.frame_lookup = {int(f): k for k, f in enumerate(window.frame_ids)}
        self.omitted = []

    def point_extra(self, stamp: int, source_index: int) -> dict:
        pos = self.w.clock.media_at(stamp)
        near, err = self.w.clock.nearest_capture(stamp)
        return {'oculo_source_row_index': source_index, 'oculo_native_video_ns': pos,
                'oculo_clip_video_ns': pos - self.w.start,
                'oculo_nearest_native_frame_index': near, 'oculo_frame_match_error_ns': err}

    def process(self, source: str, member: str, group: str, *, unique_root: bool):
        base = PurePosixPath(member).name
        basename = base.lower()
        stats = {'group': group, 'member': member, 'output_file': f'{group}/{base}',
                 'input_rows': 0, 'selected_recording_rows': 0, 'output_rows': 0,
                 'truncated_events': 0, 'boundary_context_rows': 0,
                 'unmatched_face_rows_in_window': 0, 'context_only_rows': 0}
        with csv_input(source, member) as (reader, columns):
            ridcol = pick(columns, 'recordingid')
            tscol = pick(columns, 'timestampns', 'timestamp')
            scol = pick(columns, 'starttimestampns', 'sectionstarttimens', 'starttimens')
            ecol = pick(columns, 'endtimestampns', 'sectionendtimens', 'endtimens')
            dcol = pick(columns, 'durationms')
            isface = basename in {'face_detections.csv', 'face_positions.csv'}
            if not ridcol and not unique_root:
                raise ValidationError(f'Cannot safely select recording in {member}: no recording id')
            if isface and not tscol:
                raise ValidationError('Face detections require an explicit nanosecond timestamp column')
            if scol and ecol:
                kind, extras = 'interval', EVENT_EXTRA
            elif tscol:
                kind, extras = ('face_frame', FACE_EXTRA) if isface else ('point', POINT_EXTRA)
            elif basename in {'labels.csv', 'template.csv'}:
                kind, extras = 'recording_metadata', ['oculo_source_row_index']
            else:
                self.omitted.append({'group': group, 'member': member,
                                     'reason': 'unrecognized CSV schema, not copied through untrimmed'})
                self.warnings.append(f'{member}: unknown table schema omitted; it was NOT copied as if trimmed.')
                return
            stats['kind'] = kind
            writer = context_writer = rejected_writer = None
            handles = []
            try:
                if self.output:
                    target = self.output / group / base
                    target.parent.mkdir(parents=True, exist_ok=True)
                    h = target.open('w', encoding='utf-8', newline=''); handles.append(h)
                    writer = csv.DictWriter(h, fieldnames=columns+extras, lineterminator='\n')
                    writer.writeheader()
                    if isface:
                        folder = self.output / 'context'; folder.mkdir(exist_ok=True)
                        h2 = (folder / base).open('w', encoding='utf-8', newline=''); handles.append(h2)
                        context_writer = csv.DictWriter(h2, fieldnames=columns+FACE_EXTRA, lineterminator='\n')
                        context_writer.writeheader()
                        folder = self.output / 'audit'; folder.mkdir(exist_ok=True)
                        h3 = (folder / ('unmatched_'+base)).open('w', encoding='utf-8', newline=''); handles.append(h3)
                        rejected_writer = csv.DictWriter(h3, fieldnames=columns+['oculo_reject_reason'], lineterminator='\n')
                        rejected_writer.writeheader()
                for source_index, row in enumerate(reader):
                    if source_index % 4096 == 0:
                        self.token.raise_if_cancelled()
                    stats['input_rows'] += 1
                    if None in row or any(v is None for v in row.values()):
                        raise ValidationError(f'Malformed CSV row in {member}, row {source_index}')
                    if ridcol:
                        if not row[ridcol].strip():
                            raise ValidationError(f'Blank recording id in {member}, row {source_index}')
                        if row[ridcol].strip() != self.rid:
                            continue
                    stats['selected_recording_rows'] += 1
                    extra = None
                    if kind == 'point':
                        t = int_ns(row[tscol], where=f'{member}:{source_index}')
                        if not self.w.utc_start <= t < self.w.utc_end:
                            continue
                        extra = self.point_extra(t, source_index)
                        if basename == 'gaze.csv' and group == 'tsd':
                            self.gaze.append((t, source_index))
                        elif basename == '3d_eye_states.csv' and group == 'tsd':
                            self.eye.append(t)
                        elif basename == 'gaze_on_face.csv' and group == 'face_mapper':
                            flag_col = pick(columns, 'gazeonface')
                            self.fm_gaze[t].append((source_index, row.get(flag_col, '')))
                    elif kind == 'interval':
                        a = int_ns(row[scol], where=member)
                        b = int_ns(row[ecol], where=member)
                        if b <= a:
                            raise ValidationError(f'Non-positive event interval in {member}, row {source_index}')
                        if b <= self.w.utc_start or a >= self.w.utc_end:
                            continue
                        aa, bb = max(a, self.w.utc_start), min(b, self.w.utc_end)
                        truncated = aa != a or bb != b
                        if self.w.event_policy == 'inside' and truncated and basename != 'sections.csv':
                            continue
                        extra = {'oculo_source_row_index': source_index,
                                 'oculo_original_start_utc_ns': row[scol], 'oculo_original_end_utc_ns': row[ecol],
                                 'oculo_original_duration_ms': row[dcol] if dcol else '',
                                 'oculo_retained_start_utc_ns': aa, 'oculo_retained_end_utc_ns': bb,
                                 'oculo_retained_duration_ns': bb-aa,
                                 'oculo_clip_start_video_ns': self.w.clock.media_at(aa) - self.w.start,
                                 'oculo_clip_end_video_ns': self.w.clock.media_at(bb) - self.w.start,
                                 'oculo_event_truncated': str(truncated).lower(),
                                 'oculo_statistics_scope': 'original_event_not_recomputed' if truncated else 'original_event'}
                        row[scol], row[ecol] = str(aa), str(bb)
                        if dcol and truncated:
                            row[dcol] = format(Decimal(bb-aa) / Decimal(1_000_000), 'f')
                        stats['truncated_events'] += int(truncated)
                    elif kind == 'face_frame':
                        t = int_ns(row[tscol], where=member)
                        index, err, match = self.w.clock.face_match(t, self.w.face_tolerance)
                        if index < 0:
                            if self.w.utc_start <= t < self.w.utc_end:
                                stats['unmatched_face_rows_in_window'] += 1
                                if rejected_writer:
                                    rejected_writer.writerow({**row, 'oculo_reject_reason': match})
                            continue
                        belongs = index in self.frame_lookup
                        context_only = (not belongs and
                            (index == int(self.w.frame_ids[0])-1 or index == int(self.w.frame_ids[-1])+1))
                        if not belongs and not context_only:
                            continue
                        native_pos = int(self.w.clock.media_ns[index])
                        boundary = not self.w.utc_start <= t < self.w.utc_end
                        extra = {'oculo_source_row_index': source_index,
                                 'oculo_native_frame_index': index,
                                 'oculo_source_frame_ordinal_in_window': self.frame_lookup.get(index, ''),
                                 'oculo_native_video_ns': native_pos,
                                 'oculo_clip_video_ns': native_pos - self.w.start,
                                 'oculo_frame_match_error_ns': err,
                                 'oculo_boundary_context': str(boundary or context_only).lower()}
                        if context_only:
                            stats['context_only_rows'] += 1
                            if context_writer: context_writer.writerow({**row, **extra})
                            continue
                        stats['boundary_context_rows'] += int(boundary)
                        self.fm_detections_frames.add(index)
                    else:
                        extra = {'oculo_source_row_index': source_index}
                    stats['output_rows'] += 1
                    if writer:
                        writer.writerow({**row, **extra})
            finally:
                for h in handles: h.close()
        if stats['unmatched_face_rows_in_window']:
            self.warnings.append(f"{member}: {stats['unmatched_face_rows_in_window']} face detections in the time window "
                                 'could not be uniquely matched to a Native frame; see audit/unmatched_*.csv.')
        self.tables.append(stats)

    def run(self, sources: dict, selection: dict) -> dict:
        tasks = [('tsd', selection['cloud_members'], True), ('face_mapper', selection['fm_members'], False)]
        total = sum(sum(n.endswith('.csv') for n in names) for _, names, _ in tasks)
        step = 0
        for group, members, trusted in tasks:
            seen = set()
            for member in members:
                self.token.raise_if_cancelled()
                base = PurePosixPath(member).name
                if member.endswith('.csv'):
                    if base.lower() in seen:
                        raise ValidationError(f'Ambiguous duplicated output table: {base}')
                    seen.add(base.lower())
                    emit(self.progress, .36+.49*step/max(total, 1), f'{"Exporting" if self.output else "Checking"} {group} · {base}')
                    self.process(sources[group], member, group, unique_root=trusted and base.lower()!='sections.csv')
                    step += 1
                elif group == 'tsd' and base.lower() in {'info.json', 'scene_camera.json'}:
                    # This remains original recording metadata, not a forged vendor
                    # recording/section. A separate trim manifest defines the subset.
                    if self.output:
                        out = self.output / group / base; out.parent.mkdir(exist_ok=True)
                        with SourceView(sources[group]) as view: out.write_bytes(view.read_bytes(member))
                else:
                    self.omitted.append({'group': group, 'member': member,
                                         'reason': 'not needed for data-only trimming; no video copied'})
        if not self.gaze:
            raise ValidationError('The selected Native video window contains no TSD gaze samples. Check your cut times.')
        if not any(t['group']=='face_mapper' and t['member'].endswith('sections.csv') and t['output_rows']
                   for t in self.tables):
            raise ValidationError('Face Mapper has no section overlapping this Native video window')
        counts = Counter(t for t, _ in self.gaze)
        unique_match = sum(1 for t, _ in self.gaze if len(self.fm_gaze.get(t, []))==1)
        missing = sum(1 for t, _ in self.gaze if not self.fm_gaze.get(t))
        ambiguous = len(self.gaze)-unique_match-missing
        orphan = sum(len(v) for t, v in self.fm_gaze.items() if t not in counts)
        if missing or ambiguous or orphan:
            self.warnings.append(f'Exact gaze/on-face join: missing={missing}, ambiguous={ambiguous}, '
                                 f'FM-only={orphan}. Original rows are preserved; no labels are guessed.')
        if not self.fm_detections_frames:
            self.warnings.append('No face detections overlap the retained Native frames. This is not evidence of gaze on the environment.')
        if self.output:
            path = self.output / 'audit'; path.mkdir(exist_ok=True)
            with (path/'gaze_on_face_exact_join.csv').open('w', encoding='utf-8', newline='') as h:
                wr = csv.writer(h, lineterminator='\n')
                wr.writerow(['timestamp [ns]', 'tsd_source_row_index', 'fm_source_row_index',
                             'pupil_gaze_on_face', 'fm_match_count', 'join_status'])
                for t, row_idx in self.gaze:
                    matches = self.fm_gaze.get(t, [])
                    wr.writerow([t, row_idx, matches[0][0] if len(matches)==1 else '',
                                 matches[0][1] if len(matches)==1 else '', len(matches),
                                 'exact' if len(matches)==1 else ('missing' if not matches else 'ambiguous')])
        return {'tables': self.tables, 'omitted': self.omitted, 'warnings': self.warnings,
                'gaze_rows': len(self.gaze), 'eye_state_rows': len(self.eye),
                'eye_state_timestamps_equal_gaze': [t for t, _ in self.gaze] == self.eye if self.eye else None,
                'gaze_duplicates': sum(n-1 for n in counts.values() if n>1),
                'fm_exact_gaze_matches': unique_match, 'fm_missing_gaze': missing,
                'fm_ambiguous_gaze': ambiguous, 'fm_orphan_gaze': orphan,
                'source_frames_in_window': len(self.w.frame_ids),
                'source_frames_with_faces': len(self.fm_detections_frames)}


def _assets(store: ProjectStore, sid: str, n: str, t: str, f: str) -> dict:
    s = store.snapshot().sessions[sid]
    result = {}
    for role, id_, expected in [('native', n, 'pupil_native'), ('tsd', t, 'pupil_cloud'), ('face_mapper', f, 'face_mapper')]:
        a = s.assets[id_]
        if a.kind != expected:
            raise ValidationError(f'{role} asset has kind {a.kind}; expected {expected}')
        p = Path(a.source_path).resolve()
        if p.is_dir() and (p == store.root or p in store.root.parents):
            raise ValidationError('Project output cannot be placed inside a source recording directory')
        result[role] = a
    return result


def _public_preview(plan: dict) -> dict:
    c, w, report = plan['clock_summary'], plan['window'], plan['report']
    return {'preview_id': plan['preview_id'], 'recording_id': plan['recording_id'],
            'session_id': plan['session_id'], 'request': plan['request'],
            'timecode_interpretation': plan['request'].get('timecode_interpretation'),
            'native_video_member': plan['selection']['native']['video_member'],
            'native_video_duration': fmt_ns(int(c['duration_ns'])),
            'native_video_frame_count': c['decoded_frames'],
            'native_start': fmt_ns(int(w['start_media_ns'])), 'native_end': fmt_ns(int(w['end_media_ns'])),
            'retained_duration': fmt_ns(int(w['end_media_ns'])-int(w['start_media_ns'])),
            'utc_start_ns': str(w['utc_start_ns']), 'utc_end_ns': str(w['utc_end_ns']),
            'recording_start_offset_sec': c.get('recording_start_offset_sec'),
            'fm_recording_count': plan['selection']['fm_recording_count'],
            'report': report,
            'notice': 'Data only: no video is copied, cut, encoded or marked as validated. No 3D prerequisites.'}


def preview_native_reference_trim(store: ProjectStore, *, session_id: str, native_asset_id: str,
                                  timeseries_asset_id: str, facemapper_asset_id: str,
                                  start_timecode: str, duration_timecode: str,
                                  timecode_fps: int = 30, label: str = 'Native data clip', native_video_member: str | None = None,
                                  interval_policy: str = 'clip', face_tolerance_ms: float = 5.0,
                                  ffprobe_path: str | None = None, progress=None,
                                  cancellation: CancellationToken | None = None) -> dict:
    token = cancellation or CancellationToken.create()
    interpretation = parse_native_trim_times(start_timecode, duration_timecode, timecode_fps=timecode_fps)
    start = int(interpretation['start']['nanoseconds'])
    duration = int(interpretation['duration']['nanoseconds'])
    records = _assets(store, session_id, native_asset_id, timeseries_asset_id, facemapper_asset_id)
    sources = {k: a.source_path for k, a in records.items()}
    stamps = {k: source_stamp(p) for k, p in sources.items()}
    emit(progress, .02, 'Matching Native, TSD and Face Mapper by recording id')
    selection = resolve_sources(sources['native'], sources['tsd'], sources['face_mapper'], native_video_member)
    clock = build_clock(sources['native'], store.cache_dir(), video_member=selection['native']['video_member'],
                        ffprobe_path=ffprobe_path, cancellation=token, progress=progress)
    w = Window.create(clock, start, duration, face_tolerance_ms, interval_policy)
    report = TablePass(w, selection['recording_id'], None, token, progress).run(sources, selection)
    # Same keys but NOT same section IDs: section IDs belong to an enrichment export.
    if any(source_stamp(sources[k]) != stamps[k] for k in stamps):
        raise ValidationError('An input changed during preview; run preview again')
    preview_id = new_id('nativeplan')
    root = store.cache_dir()/'native_trim_plans'; root.mkdir(exist_ok=True)
    info = clock.metadata.copy()
    rst = info.get('recording_start_utc_ns')
    if rst and re.fullmatch(r'\d+', rst):
        info['recording_start_offset_sec'] = str(Decimal(int(clock.capture_ns[0])-int(rst))/Decimal(1e9))
    request = {'label': label, 'start_timecode': start_timecode, 'duration_timecode': duration_timecode,
               'timecode_fps': timecode_fps, 'timecode_interpretation': interpretation,
               'interval_policy': interval_policy, 'face_tolerance_ms': face_tolerance_ms,
               'native_asset_id': native_asset_id, 'timeseries_asset_id': timeseries_asset_id,
               'facemapper_asset_id': facemapper_asset_id,
               'native_video_member': selection['native']['video_member'], 'ffprobe_path': ffprobe_path}
    plan = {'schema_version': SCHEMA, 'preview_id': preview_id, 'session_id': session_id,
            'project_id': store.project.id, 'created_at': utc_now(), 'request': request,
            'recording_id': selection['recording_id'], 'sources': sources, 'source_stamps': stamps,
            'asset_ids': {k: a.id for k, a in records.items()}, 'selection': selection,
            'clock_index': str(clock.index_path), 'clock_summary': info,
            'clock_index_sha256': sha256_file(clock.index_path),
            'window': {'start_media_ns': str(w.start), 'end_media_ns': str(w.end),
                       'utc_start_ns': str(w.utc_start), 'utc_end_ns': str(w.utc_end)},
            'report': report}
    atomic_write_json(root/f'{preview_id}.json', plan)
    emit(progress, 1, 'Preview complete; confirm the cut times and row counts before export')
    return _public_preview(plan)


def _write_frame_map(output: Path, w: Window):
    fields = ['native_frame_index', 'source_frame_ordinal_in_window', 'native_video_pts_ns',
              'native_capture_utc_ns', 'native_anchored_display_utc_ns', 'clip_video_pts_ns',
              'retained_display_start_ns', 'retained_display_end_ns', 'boundary_frame']
    with (output/'native_frame_map.csv').open('w', encoding='utf-8', newline='') as h:
        wr = csv.writer(h, lineterminator='\n'); wr.writerow(fields)
        for ordinal, i in enumerate(w.frame_ids):
            i = int(i); a = int(w.clock.media_ns[i])
            b = int(w.clock.media_ns[i+1]) if i+1<len(w.clock.media_ns) else w.clock.end_media_ns
            wr.writerow([i, ordinal, a, int(w.clock.capture_ns[i]), int(w.clock.capture_ns[0])+a,
                         a-w.start, max(a,w.start)-w.start, min(b,w.end)-w.start,
                         str(a<w.start or b>w.end).lower()])
    with (output/'native_clock_full.csv').open('w', encoding='utf-8', newline='') as h:
        wr=csv.writer(h, lineterminator='\n'); wr.writerow(['native_frame_index','native_video_pts_ns','native_capture_utc_ns'])
        wr.writerows((i,int(a),int(b)) for i,(a,b) in enumerate(zip(w.clock.media_ns,w.clock.capture_ns)))


def _zip_subset(source: Path, output_zip: Path, token: CancellationToken):
    with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for file in sorted(source.rglob('*')):
            token.raise_if_cancelled()
            if file.is_file(): z.write(file, file.relative_to(source).as_posix())


def _report_html(preview: dict) -> str:
    req = preview.get('request', {})
    interpretation = req.get('timecode_interpretation')
    timecode_note = ''
    if interpretation:
        timecode_note = (f"<p><b>Editor inputs:</b> {html.escape(req['start_timecode'])} + "
                         f"{html.escape(req['duration_timecode'])} (retained duration)<br>"
                         f"<b>Timecode FPS:</b> {int(req['timecode_fps'])} (non-drop; FF interpretation only). "
                         "NRD frames still use actual MP4 PTS and the companion .time index.</p>")
    rows = ''.join('<tr>'+''.join(f'<td>{html.escape(str(t[k]))}</td>' for k in
                                 ['group','member','selected_recording_rows','output_rows','kind'])+'</tr>'
                   for t in preview['report']['tables'])
    warnings = ''.join(f'<li>{html.escape(w)}</li>' for w in preview['report']['warnings']) or '<li>None reported by these checks.</li>'
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Native-reference data trim</title>
<style>body{{font:15px/1.6 system-ui;background:#f4f3eb;color:#282c21;margin:40px auto;max-width:1150px;padding:0 25px}}h1{{color:#4d5130}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:8px;border:1px solid #d0d1c2;text-align:left}}.card{{padding:20px;background:white;border-radius:18px;margin:20px 0}}code{{overflow-wrap:anywhere}}</style>
<h1>OculoStudio — Native-reference data trim</h1><p>v{__version__} · No video re-encoding · No 3D prerequisites</p>
<div class="card"><b>Recording:</b> <code>{html.escape(preview['recording_id'])}</code><br>
<b>NRD MP4:</b> {html.escape(preview['native_video_member'])}<br>
<b>Keep:</b> {preview['native_start']} → {preview['native_end']} · duration {preview['retained_duration']}<br>
{timecode_note}<b>UTC interval:</b> <code>[{preview['utc_start_ns']}, {preview['utc_end_ns']})</code></div>
<table><thead><tr><th>Group</th><th>Source table</th><th>Rows in recording</th><th>Retained rows</th><th>Rule</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Warnings</h2><ul>{warnings}</ul>
<h2>Interpretation</h2><p>This is a derived data subset, not a reconstructed vendor recording. Point timestamps and coordinates are retained. Partially clipped events retain their original endpoints and original statistics in explicit columns. No pupil/gaze resampling is performed.</p>
<p>Face geometry follows Native source frames whose presentation intervals overlap the requested window. Boundary geometry is flagged, and one adjacent frame on either side may be stored separately in <code>context/</code>. These are not additional gaze samples. Source-frame ordinals are NOT verified edited-video frame numbers.</p>
<p>Your already-edited MP4 was not supplied to this data pipeline. The assumption is a simple cut without a speed change. The software cannot certify your editor's frame snapping, retiming, mirroring or crop. This release does not turn the previous 3D-dependent semantic page into a standalone facial-AOI player.</p></html>'''


def export_native_reference_trim(store: ProjectStore, *, session_id: str, preview_id: str,
                                 progress=None, cancellation: CancellationToken | None = None) -> dict:
    token = cancellation or CancellationToken.create()
    if not re.fullmatch(r'nativeplan_[a-f0-9]{12}', preview_id):
        raise ValidationError('Invalid Native trim preview id')
    plan_path = store.cache_dir()/'native_trim_plans'/f'{preview_id}.json'
    if not plan_path.exists():
        raise ValidationError('Preview not found in this project; run alignment preview first')
    plan = json.loads(plan_path.read_text('utf-8'))
    if plan['project_id'] != store.project.id or plan['session_id'] != session_id:
        raise ValidationError('Preview belongs to another project/session')
    req = plan['request']
    current_assets = _assets(store, session_id, req['native_asset_id'], req['timeseries_asset_id'], req['facemapper_asset_id'])
    if any(str(Path(a.source_path).resolve()) != str(Path(plan['sources'][role]).resolve())
           for role, a in current_assets.items()):
        raise ValidationError('Registered source paths changed after preview; run preview again')
    for role, path in plan['sources'].items():
        token.raise_if_cancelled()
        if source_stamp(path) != plan['source_stamps'][role]:
            raise ValidationError(f'{role} source changed after preview; run preview again')
    if sha256_file(plan['clock_index']) != plan['clock_index_sha256']:
        raise ValidationError('Native clock cache changed after preview; re-index the source')
    request = plan['request']
    clock = load_clock(plan['clock_index'])
    start = int(plan['window']['start_media_ns']); end = int(plan['window']['end_media_ns'])
    w = Window.create(clock, start, end-start, request['face_tolerance_ms'], request['interval_policy'])
    artifact_id = new_id('artifact')
    category = store.derived_dir(session_id, 'native_reference_trim')
    staging = category/('.partial_'+artifact_id); staging.mkdir()
    final = category/artifact_id
    try:
        result = TablePass(w, plan['recording_id'], staging, token, progress).run(plan['sources'], plan['selection'])
        # Any row-count changes, even with an unchanged path stamp, invalidate
        # a preview rather than producing a silently different export.
        if result != plan['report']:
            raise ValidationError('Export result differs from the verified preview; no artifact was published')
        _write_frame_map(staging, w)
        public = _public_preview(plan)
        manifest = {'schema_version': SCHEMA, 'producer_version': __version__, 'recording_id': plan['recording_id'],
                    'request': request, 'window': plan['window'], 'native_reference': plan['clock_summary'],
                    'source_stamps': plan['source_stamps'], 'report': result,
                    'point_interval_policy': '[UTC_start, UTC_end)',
                    'geometry_interval_policy': 'Native_frame_presentation_overlap_plus_separate_context',
                    'event_policy': request['interval_policy'],
                    'video_modified': False, 'gaze_resampled': False, 'three_d_required': False,
                    'edited_video_frame_correspondence_verified': False,
                    'output_is_vendor_native_recording': False}
        atomic_write_json(staging/'trim_manifest.json', manifest)
        (staging/'trim_report.html').write_text(_report_html(public), 'utf-8')
        note = 'OculoStudio derived subset; see trim_manifest.json. Original absolute timestamps are preserved for point samples.\n'
        for folder in ['tsd','face_mapper']:
            (staging/folder/'OCULOSTUDIO_SUBSET.txt').write_text(note, 'utf-8')
            atomic_write_json(staging/folder/'oculostudio_trim.json', {
                'schema_version': SCHEMA, 'recording_id': plan['recording_id'], 'window': plan['window'],
                'event_policy': request['interval_policy'], 'video_included': False,
                'recording_info_retains_original_metadata': True})
        emit(progress, .87, 'Packaging selected TSD and Face Mapper only')
        _zip_subset(staging/'tsd', staging/'TSD_trimmed.zip', token)
        _zip_subset(staging/'face_mapper', staging/'FaceMapper_trimmed.zip', token)
        for role, path in plan['sources'].items():
            if source_stamp(path) != plan['source_stamps'][role]:
                raise ValidationError('An input changed during export; no artifact was published')
        token.raise_if_cancelled()
        staging.rename(final)
        # Do not create a fake global Clip or change existing 3D time maps.
        source_asset_ids = list(plan['asset_ids'].values()) + list(request.get('additional_source_asset_ids') or [])
        source_asset_ids = list(dict.fromkeys(str(value) for value in source_asset_ids if value))
        parent_artifact_ids = list(dict.fromkeys(
            str(value) for value in (request.get('parent_artifact_ids') or []) if value
        ))
        provenance = {
            'recording_id': plan['recording_id'],
            'video_modified': False,
            'native_frame_clock_sha256': plan['clock_index_sha256'],
            **dict(request.get('provenance_extra') or {}),
        }
        artifact = ArtifactStore(store).register(session_id=session_id, kind=ArtifactKind.NATIVE_REFERENCE_TRIM,
            label=request['label'] or 'Native-reference data clip', output_root=final,
            source_asset_ids=source_asset_ids, parent_artifact_ids=parent_artifact_ids,
            parameters={**request, 'preview_id': preview_id,
                'time_reference': 'selected_NRD_scene_video', 'window': plan['window']},
            qc={k: v for k,v in result.items() if k not in ['tables','omitted']},
            provenance=provenance, artifact_id=artifact_id)
        result_payload = {'artifact_id': artifact.id, 'output_dir': str(final), 'tsd_zip': str(final/'TSD_trimmed.zip'),
                'facemapper_zip': str(final/'FaceMapper_trimmed.zip'), 'report_html': str(final/'trim_report.html'),
                'manifest': str(final/'trim_manifest.json'), 'preview': public}
        store.update(lambda project: project.sessions[session_id].metadata.__setitem__(
            'native_reference_trim_last', {'request': request, 'preview': public, 'result': result_payload}))
        emit(progress, 1, 'TSD + Face Mapper trimmed; original and edited videos are unchanged')
        return result_payload
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        # A failed registration must not leave a folder presented as a complete artifact.
        if final.exists() and artifact_id not in store.project.sessions[session_id].artifacts:
            shutil.rmtree(final, ignore_errors=True)
        raise
