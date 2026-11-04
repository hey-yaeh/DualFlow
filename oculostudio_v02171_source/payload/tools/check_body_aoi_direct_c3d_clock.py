from __future__ import annotations
import json, tempfile
from pathlib import Path
import pandas as pd
from oculostudio.demo import create_demo_project
from oculostudio.domain.models import ArtifactKind
from oculostudio.services.gaze3d import generate_gaze_rays
from oculostudio.services.body_aoi import create_body_aoi_annotations
from oculostudio.storage.project_store import ProjectStore
from oculostudio.storage.artifact_store import ArtifactStore


def run(root: Path):
    base = create_demo_project(root, run_processing=False)
    store = ProjectStore.open(root)
    sid = base['session_id']; assets = base['assets']
    rays_result = generate_gaze_rays(
        store, session_id=sid, timeseries_asset_id=assets['cloud'],
        tracker_asset_id=assets['tracker'], profile_artifact_id=base['profile_artifact_id'],
        scope='clip', clip_id=base['clip_id'], max_pose_gap_sec=0.05,
    )
    rays_path = Path(rays_result['csv'])
    rays = pd.read_csv(rays_path, low_memory=False)

    trim_root = store.derived_dir(sid, 'native_reference_trim', 'direct_c3d_clock_regression')
    (trim_root/'tsd').mkdir(parents=True, exist_ok=True)
    trim_gaze = pd.DataFrame({
        'timestamp [ns]': rays['timestamp [ns]'],
        'gaze x [px]': rays['gaze x [px]'],
        'gaze y [px]': rays['gaze y [px]'],
        'worn': rays['worn'],
        'oculo_source_row_index': rays['gaze3d_input_row_index'],
        'oculo_clip_video_ns': ((rays['gaze3d_c3d_time_sec'] - rays['gaze3d_c3d_time_sec'].iloc[0])*1e9).round().astype('int64'),
    })
    trim_gaze.to_csv(trim_root/'tsd'/'gaze.csv', index=False)
    (trim_root/'trim_manifest.json').write_text(json.dumps({
        'recording_id': 'demo-recording',
        'native_reference': {'width': 1920, 'height': 1080},
    }), 'utf-8')
    trim = ArtifactStore(store).register(
        session_id=sid, kind=ArtifactKind.NATIVE_REFERENCE_TRIM,
        label='C3D-defined regression trim', output_root=trim_root,
        source_asset_ids=[assets['cloud'], assets['body'], assets['tracker']],
        parameters={
            'automatic_c3d_defined': True,
            'window_source': 'common_body_c3d_overlap',
            'body_asset_id': assets['body'],
            'body_intervals': {'R02': {'asset_id': assets['body']}},
        },
    )

    # Reproduce the user project: Gaze 3D is already valid, but there is no
    # unnecessary Timeseries -> project Global path for downstream Body AOI.
    store.update(lambda project: project.sessions[sid].time_maps.clear())

    result = create_body_aoi_annotations(
        store, session_id=sid,
        gaze_rays_artifact_id=rays_result['artifact']['id'],
        timeseries_asset_id=assets['cloud'], body_asset_ids=[assets['body']],
        partner_body_asset_id=assets['body'], include_self_body=False,
        entity_marker_prefixes={assets['body']: 'P02'},
        native_trim_artifact_id=trim.id,
        geometry_acceptance_policy='required',
        sample_clock_chunk_size=512,
        output_parquet=False,
        label='Direct gaze-ray C3D clock regression',
    )
    manifest = json.loads((Path(result['samples_csv']).parent/'body_aoi_manifest.json').read_text('utf-8'))
    samples = pd.read_csv(result['samples_csv'], low_memory=False)
    qc = manifest['qc']
    checks = {
        'rows_preserved': len(samples) == len(rays),
        'uses_direct_gaze_ray_c3d_time': qc.get('body_time_alignment_mode') == 'gaze_ray_c3d_time',
        'global_path_not_required': qc.get('global_timegraph_required') is False,
        'clock_column_audited': qc.get('timebase_summary', {}).get('gaze_ray_c3d_clock_column') == 'gaze3d_c3d_time_sec',
        'body_geometry_gate_passed': bool(qc.get('body_geometry_gate_passed')),
        'partner_hits_present': bool(samples['assignment_status'].astype(str).eq('body_3d_exact_partner').any()),
    }
    return checks, {'qc': {k: qc.get(k) for k in ['alignment_mode','body_time_alignment_mode','alignment_basis','global_timegraph_required','body_geometry_gate_passed']}, 'rows': len(samples)}

if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='oculo_body_direct_c3d_') as td:
        checks, details = run(Path(td))
    failed=[k for k,v in checks.items() if not v]
    print(json.dumps({'checks':checks,'details':details,'failed':failed,'ok':not failed},indent=2))
    raise SystemExit(1 if failed else 0)
