from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARSER = argparse.ArgumentParser()
PARSER.add_argument('--output', default=str(ROOT / 'docs' / 'VALIDATION_v0_21_54_BODY_SEMANTIC_REDUCTION.json'))
ARGS = PARSER.parse_args()

from oculostudio.services.aoi_review import _body_review_episode_groups
from oculostudio.services.body_aoi import _apply_body_auto_review_reduction


def row(**updates):
    base = {
        'gaze_sample_uid': 'sample',
        'source_gaze_row_index': 0,
        'timestamp_ns': 1_000_000_000,
        'fixation_id': 'F1',
        'native_frame_index': 0,
        'scene_frame_index': 0,
        'review_preview_available': True,
        'ray_valid': True,
        'review_required': True,
        'assignment_status': 'body_3d_uncertainty_cone_partner',
        'body_relation': 'partner',
        'body_entity_id': 'P02',
        'body_raw_region': 'head',
        'body_coarse_region': 'head',
        'body_region': 'head',
        'body_social_region': 'partner_head',
        'body_exact_hit': False,
        'body_angular_miss_deg': 0.25,
        'body_confidence': 0.84,
        'first_hit_primitive_confidence': 0.82,
        'candidate_hit_count': 2,
        'second_hit_relation': 'partner',
        'second_hit_entity_id': 'P02',
        'second_hit_raw_region': 'neck',
        'second_hit_coarse_region': 'head',
        'second_hit_angular_miss_deg': 0.27,
        'second_hit_primitive_confidence': 0.79,
        'self_occlusion': False,
    }
    base.update(updates)
    return base


def reduced(rows, policy='semantic_guided'):
    frame = pd.DataFrame(rows)
    out, summary, audit = _apply_body_auto_review_reduction(frame, policy)
    return out, summary, audit


checks = {}
notes = {}

# The same exported target (head) may be represented by raw head and neck
# primitives.  Semantic mode should not waste a human decision on that raw
# primitive boundary, while the previous evidence-guided mode remains strict.
semantic, semantic_summary, _ = reduced([row()])
legacy, _, _ = reduced([row()], 'evidence_guided')
checks['same_coarse_surface_resolved'] = bool(semantic.loc[0, 'auto_review_resolved'])
checks['same_coarse_surface_source_auditable'] = semantic.loc[0, 'auto_review_resolution_source'] == 'same_coarse_surface'
checks['previous_policy_stays_strict'] = bool(legacy.loc[0, 'review_required'])

# A true head/torso ambiguity must remain in the queue.
conflict, _, _ = reduced([row(
    second_hit_raw_region='torso', second_hit_coarse_region='torso',
    second_hit_angular_miss_deg=0.27,
)])
checks['different_coarse_regions_remain_pending'] = bool(conflict.loc[0, 'review_required'])

# Self-occlusion is always a human decision.
occluded, _, _ = reduced([row(self_occlusion=True)])
checks['self_occlusion_remains_pending'] = bool(occluded.loc[0, 'review_required'])

# An exact hit immediately beside a split fixation can safely anchor the same
# target; it may not anchor a different target.
anchor_rows = [
    row(
        gaze_sample_uid='exact-head', timestamp_ns=1_000_000_000,
        fixation_id='F_exact', review_required=False,
        assignment_status='body_3d_exact_partner', body_exact_hit=True,
        candidate_hit_count=1, body_angular_miss_deg=0.0,
        second_hit_relation='', second_hit_entity_id='',
        second_hit_raw_region='', second_hit_coarse_region='',
        second_hit_angular_miss_deg=float('nan'),
    ),
    row(
        gaze_sample_uid='near-head', timestamp_ns=1_120_000_000,
        fixation_id='F_split', scene_frame_index=4, native_frame_index=4, body_angular_miss_deg=0.51,
        candidate_hit_count=1, second_hit_angular_miss_deg=float('nan'),
    ),
]
anchored, _, _ = reduced(anchor_rows)
checks['adjacent_exact_anchor_resolves_split_fixation'] = not bool(anchored.loc[1, 'review_required'])
checks['adjacent_exact_anchor_is_audited'] = anchored.loc[1, 'auto_review_resolution_source'] == 'adjacent_exact_anchor'

anchor_conflict_rows = anchor_rows + [
    row(
        gaze_sample_uid='exact-torso', timestamp_ns=1_080_000_000,
        fixation_id='F_other', scene_frame_index=2, native_frame_index=2, review_required=False,
        assignment_status='body_3d_exact_partner', body_exact_hit=True,
        body_raw_region='torso', body_coarse_region='torso', body_region='torso',
        body_social_region='partner_torso', candidate_hit_count=1,
        body_angular_miss_deg=0.0, second_hit_relation='', second_hit_entity_id='',
        second_hit_raw_region='', second_hit_coarse_region='',
        second_hit_angular_miss_deg=float('nan'),
    ),
]
anchor_conflict, _, _ = reduced(anchor_conflict_rows)
checks['conflicting_exact_anchor_does_not_resolve'] = bool(anchor_conflict.loc[1, 'review_required'])

# Review-unit reduction: adjacent fixation IDs with an identical person/entity/
# coarse body target become one target-stable review episode. A changed target
# remains a separate event.
episode_samples = pd.DataFrame([
    row(gaze_sample_uid='a1', fixation_id='F1', timestamp_ns=1_000_000_000, native_frame_index=0),
    row(gaze_sample_uid='a2', fixation_id='F1', timestamp_ns=1_050_000_000, native_frame_index=1),
    row(gaze_sample_uid='b1', fixation_id='F2', timestamp_ns=1_180_000_000, native_frame_index=4),
    row(gaze_sample_uid='b2', fixation_id='F2', timestamp_ns=1_230_000_000, native_frame_index=5),
    row(
        gaze_sample_uid='c1', fixation_id='F3', timestamp_ns=1_360_000_000,
        native_frame_index=8, body_raw_region='torso', body_coarse_region='torso',
        body_region='torso', body_social_region='partner_torso',
        second_hit_raw_region='pelvis', second_hit_coarse_region='torso',
    ),
])
video_map = pd.DataFrame({
    'native_frame_index': list(range(12)),
    'output_frame_index': list(range(12)),
})
episodes = _body_review_episode_groups(SimpleNamespace(samples=episode_samples, video_map=video_map))
checks['adjacent_same_target_fixations_merge'] = len(episodes) == 2 and episodes[0]['base_unit_count'] == 2
checks['changed_body_target_stays_separate'] = len(episodes) == 2 and episodes[1]['target'] == ('partner', 'P02', 'torso')
checks['episode_keeps_all_original_samples'] = len(episodes[0]['sample_uids']) == 4

notes['semantic_summary'] = semantic_summary
notes['episode_count'] = len(episodes)
notes['episodes'] = [
    {
        'target': episode.get('target'),
        'base_unit_count': episode.get('base_unit_count'),
        'sample_count': len(episode.get('sample_uids') or []),
    }
    for episode in episodes
]

report = {
    'schema_version': 'oculostudio-validation-body-semantic-review-1.0',
    'version': '0.21.71',
    'checks': checks,
    'notes': notes,
    'passed': sum(bool(value) for value in checks.values()),
    'total': len(checks),
}
report['ok'] = report['passed'] == report['total']
out = Path(ARGS.output)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
print(json.dumps(report, indent=2, default=str))
raise SystemExit(0 if report['ok'] else 1)
