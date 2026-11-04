# OculoStudio v0.21.56 — Precision-Audited Clock Synchronization

## Why this release was necessary

v0.21.55 could report `0.0 ppm` drift and `0.0 ms` P95 residual after a manually co-start-trimmed recording. Co-start trimming can legitimately simplify offset recovery, but it does not prove zero clock drift. Two implementation details made the metrics look more certain than the evidence supported:

1. local window lags were quantized to the comparison sample grid; and
2. the same windows used to fit the affine map were also used to summarize residuals.

## Synchronization changes

- **Sub-sample local offsets.** A quadratic fit around each correlation peak refines lag below one comparison sample and retains peak width/curvature as uncertainty evidence.
- **Independent validation.** Every accepted motion window is held out once. The affine mapping is fitted to the remaining windows and predicts the held-out local offset. Reported P95 residuals are leave-one-window-out residuals.
- **Finite drift uncertainty.** Drift is displayed with a 95% confidence interval, estimator resolution, and the distributed-window time span.
- **No false exact zero.** Residuals below resolution are labelled `precision-limited`; they are not interpreted as mathematical zero.
- **Co-start is a prior, not proof.** The UI explicitly notes when the relative start is already close because of manual trimming while continuing to estimate drift from independent windows.
- **Evidence preview.** Calibration can display aligned Pupil IMU and assigned ETM C3D motion, local offsets over time, fitted drift, accepted/rejected windows, lag uncertainty, and cross-validated residuals. Ego video is not read.
- **Fail-closed lineage.** v0.21.55 automatic-sync artifacts remain viewable but are not production-ready because they lack the precision-audit fields. Rerun synchronization to create v0.21.56 TimeGraphs.

## Compatibility

Projects and existing artifacts are preserved. Hardware profiles and wearer assignment are unchanged. After accepting a new TimeGraph, regenerate Gaze 3D and downstream AOI artifacts so their lineage references the audited mapping.
