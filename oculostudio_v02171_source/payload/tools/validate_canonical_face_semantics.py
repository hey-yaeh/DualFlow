from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: validate_canonical_face_semantics.py <wheel-tree> <FaceMapper.zip>")
        return 2
    wheel_tree = Path(sys.argv[1]).resolve()
    source_zip = Path(sys.argv[2]).resolve()
    sys.path.insert(0, str(wheel_tree))
    from oculostudio.services.facemapper_canonical import (
        BilateralColumns,
        canonicalize_bilateral_semantics,
    )

    with zipfile.ZipFile(source_zip) as archive:
        with archive.open("face_detections.csv") as handle:
            raw = pd.read_csv(handle)
    columns = BilateralColumns(
        "eye left x [px]", "eye left y [px]",
        "eye right x [px]", "eye right y [px]",
        "mouth left x [px]", "mouth left y [px]",
        "mouth right x [px]", "mouth right y [px]",
    )
    canonical, audit = canonicalize_bilateral_semantics(
        raw, columns=columns, source_convention="pupil_image_side"
    )

    checks = {
        "row_count_preserved": len(raw) == len(canonical),
        "subject_left_eye_is_source_image_right": np.allclose(
            canonical[columns.left_eye_x], raw[columns.right_eye_x], equal_nan=True
        ) and np.allclose(canonical[columns.left_eye_y], raw[columns.right_eye_y], equal_nan=True),
        "subject_right_eye_is_source_image_left": np.allclose(
            canonical[columns.right_eye_x], raw[columns.left_eye_x], equal_nan=True
        ) and np.allclose(canonical[columns.right_eye_y], raw[columns.left_eye_y], equal_nan=True),
        "subject_left_mouth_is_source_image_right": np.allclose(
            canonical[columns.mouth_left_x], raw[columns.mouth_right_x], equal_nan=True
        ),
        "subject_right_mouth_is_source_image_left": np.allclose(
            canonical[columns.mouth_right_x], raw[columns.mouth_left_x], equal_nan=True
        ),
        "nose_unchanged": np.allclose(
            canonical["nose x [px]"], raw["nose x [px]"], equal_nan=True
        ) and np.allclose(canonical["nose y [px]"], raw["nose y [px]"], equal_nan=True),
        "face_box_unchanged": all(
            np.allclose(canonical[column], raw[column], equal_nan=True)
            for column in ("p1 x [px]", "p1 y [px]", "p2 x [px]", "p2 y [px]")
        ),
        "geometry_mirror_false": not bool(audit["geometry_mirror_applied"]),
        "semantic_swap_true": bool(audit["bilateral_label_swap_applied"]),
    }
    payload = {"checks": checks, "audit": audit}
    print(json.dumps(payload, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
