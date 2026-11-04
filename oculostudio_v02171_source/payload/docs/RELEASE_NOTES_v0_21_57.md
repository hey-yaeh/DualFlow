# OculoStudio v0.21.57 — C3D-Defined Pupil Clips

## Purpose

The normal workflow no longer asks the user to type ego-video start and end times when the interaction body C3D files already preserve the authoritative Vicon interval.

The new default path is:

```text
Original NRD + TSD + FaceMapper
        ↓
Calibration: wearer assignment + precision-audited Pupil↔Vicon TimeGraphs
        ↓
Common R01/R02 body-C3D Vicon interval
        ↓ inverse role-specific TimeGraph
R01 and R02 Pupil clock intervals
        ↓ exact NRD .time / MP4 PTS mapping
Paired Native-reference trim artifacts
        ↓
Privacy → Gaze 3D → Face/Body AOI
```

## Main changes

- Calibration now precedes the normal Trim stage.
- The physical interaction window is the half-open overlap of the R01 and R02 body C3D ranges.
- Each role's accepted affine TimeGraph is inverted to map that common Vicon interval into the correct Pupil recording clock.
- Exact NRD boundaries are resolved from the companion `.time` array and actual MP4 presentation timestamps, not `time × nominal FPS`.
- TSD and FaceMapper rows are cut using the same role-specific Pupil interval.
- R01 and R02 trims are published transactionally. If either role fails, neither trim and no pair artifact is published.
- A new immutable `aligned_clip_set` binds both role trims to one common Vicon interval and one synchronized Calibration handoff.
- The source NRD MP4 is not rewritten or re-encoded. The trim is a virtual interval plus exact frame map and trimmed tables; Privacy may encode a physical clip later.
- Manual start/end fields remain available only in a collapsed recovery section.

## Validation and safety gates

The automatic clip job fails closed when:

- the selected TSD is not the Timeseries source used by the accepted synchronization;
- the R01/R02 body C3Ds have no common physical interval or insufficient overlap;
- the assigned ETM C3D does not cover the common interval;
- the mapped Pupil interval falls outside the NRD clock beyond the configured tolerance;
- NRD, TSD and FaceMapper do not resolve to the same Pupil recording;
- either role fails during export.

Each output records:

- common Vicon start/end and duration;
- role-specific Pupil start/end;
- exact NRD media interval and nearest frame indices;
- start/end boundary error;
- accepted TimeGraph and synchronization artifact;
- body and assigned ETM source assets;
- no-manual-timecode and no-source-reencode provenance.

## Upgrade procedure

1. Install v0.21.57 and open the project.
2. Complete Calibration wearer assignment, precision-audited synchronization and the synchronized Gaze 3D handoff.
3. In Calibration Phase 5, verify the automatically selected R01/R02 NRD, synchronized TSD and FaceMapper sources.
4. Click **Create C3D-defined R01 + R02 clips**.
5. Open **Trim** to inspect physical duration, exact NRD interval and boundary error.
6. Continue to Privacy and Gaze 3D using the new role-specific trim artifacts.

Old trim artifacts are not deleted. New artifacts are immutable children with explicit TimeGraph lineage.
