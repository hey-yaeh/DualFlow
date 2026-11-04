from __future__ import annotations

# These labels are generic object names commonly used in the project's Vicon
# exports. They do not encode a wearer identity and can be edited in the GUI.
DEFAULT_CLAPPER_TOP_LABELS = [f"clapperboard_top:M_{index}" for index in range(1, 5)]
DEFAULT_CLAPPER_BOTTOM_LABELS = [f"clapperboard_down:M_{index}" for index in range(1, 5)]

# An empty mapping tells the automatic identity/calibration path to infer the
# 4x4 AprilTag-board layout from the board C3D geometry.
EMPTY_BOARD_MAPPING = {"tags": {}, "outer": {}}
