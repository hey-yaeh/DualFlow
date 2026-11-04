from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from oculostudio.domain.models import TimeAnchor, TimeMap, TimeMapQuality
from oculostudio.errors import TimeGraphError, ValidationError


def _weighted_affine_fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Fit y = scale*x + intercept on already-localized nanosecond deltas."""
    total = float(np.sum(weights))
    if total <= 0:
        raise ValidationError("Anchor weights sum to zero")
    x0 = float(np.sum(weights * x) / total)
    y0 = float(np.sum(weights * y) / total)
    xc = x - x0
    yc = y - y0
    denominator = float(np.sum(weights * xc * xc))
    if denominator <= 0:
        return 1.0, y0 - x0
    scale = float(np.sum(weights * xc * yc) / denominator)
    intercept = y0 - scale * x0
    return scale, intercept


def _equivalent_offset(source_origin_ns: int, target_origin_ns: int, scale: float) -> float:
    # Diagnostic/backward-compatible intercept. It is never used to evaluate a
    # centered map because it may lose a fraction of a nanosecond at epoch scale.
    return float(target_origin_ns) - float(scale) * float(source_origin_ns)


def fit_time_map(
    source_domain_id: str,
    target_domain_id: str,
    anchors: Iterable[TimeAnchor],
    *,
    method: str = "robust_affine",
    huber_delta_ns: float = 5_000_000.0,
    max_iterations: int = 12,
) -> TimeMap:
    points = list(anchors)
    if not points:
        raise ValidationError("At least one synchronization anchor is required")

    # Subtract exact integer anchors before float conversion. This preserves
    # sub-millisecond differences even for Unix-epoch nanosecond clocks.
    source_origin = int(points[0].source_ns)
    target_reference = int(points[0].target_ns)
    x_int = np.asarray([int(p.source_ns) - source_origin for p in points], dtype=np.int64)
    y_int = np.asarray([int(p.target_ns) - target_reference for p in points], dtype=np.int64)
    x = x_int.astype(np.float64)
    y = y_int.astype(np.float64)
    base_weights = np.asarray([p.weight for p in points], dtype=np.float64)

    if len(points) == 1:
        scale, intercept = 1.0, 0.0
        iterations = 0
    else:
        weights = base_weights.copy()
        scale, intercept = _weighted_affine_fit(x, y, weights)
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            residual = y - (scale * x + intercept)
            abs_residual = np.abs(residual)
            robust = np.ones_like(abs_residual)
            outside = abs_residual > huber_delta_ns
            robust[outside] = huber_delta_ns / np.maximum(abs_residual[outside], 1.0)
            new_weights = base_weights * robust
            new_scale, new_intercept = _weighted_affine_fit(x, y, new_weights)
            if abs(new_scale - scale) < 1e-13 and abs(new_intercept - intercept) < 0.1:
                scale, intercept = new_scale, new_intercept
                break
            scale, intercept = new_scale, new_intercept
    if scale <= 0 or not np.isfinite(scale) or not np.isfinite(intercept):
        raise ValidationError("Synchronization anchors produced an invalid time map")

    target_origin = target_reference + int(round(intercept))
    predicted = target_origin + np.rint(scale * x).astype(np.int64)
    truth = np.asarray([int(p.target_ns) for p in points], dtype=np.int64)
    residual = (truth - predicted).astype(np.float64)
    abs_residual = np.abs(residual)
    quality = TimeMapQuality(
        anchor_count=len(points),
        rms_residual_ns=float(np.sqrt(np.mean(residual**2))),
        max_abs_residual_ns=float(np.max(abs_residual)),
        median_abs_residual_ns=float(np.median(abs_residual)),
        robust_iterations=iterations,
        warning=(
            "Only one anchor is available; clock drift cannot be estimated."
            if len(points) == 1
            else None
        ),
    )
    return TimeMap(
        source_domain_id=source_domain_id,
        target_domain_id=target_domain_id,
        scale=scale,
        offset_ns=_equivalent_offset(source_origin, target_origin, scale),
        source_origin_ns=source_origin,
        target_origin_ns=target_origin,
        anchors=points,
        quality=quality,
        valid_source_start_ns=min(int(p.source_ns) for p in points),
        valid_source_end_ns=max(int(p.source_ns) for p in points),
        method=method,
    )


def invert_time_map(mapping: TimeMap) -> TimeMap:
    scale = 1.0 / mapping.scale
    if mapping.source_origin_ns is not None and mapping.target_origin_ns is not None:
        source_origin = mapping.target_origin_ns
        target_origin = mapping.source_origin_ns
        offset = _equivalent_offset(source_origin, target_origin, scale)
    else:
        source_origin = None
        target_origin = None
        offset = -mapping.offset_ns / mapping.scale
    return TimeMap(
        id=f"{mapping.id}:inverse",
        source_domain_id=mapping.target_domain_id,
        target_domain_id=mapping.source_domain_id,
        scale=scale,
        offset_ns=offset,
        source_origin_ns=source_origin,
        target_origin_ns=target_origin,
        anchors=[
            TimeAnchor(
                id=f"{anchor.id}:inverse",
                source_ns=anchor.target_ns,
                target_ns=anchor.source_ns,
                label=anchor.label,
                weight=anchor.weight,
            )
            for anchor in mapping.anchors
        ],
        quality=mapping.quality,
        valid_source_start_ns=None,
        valid_source_end_ns=None,
        method=f"inverse({mapping.method})",
        created_at=mapping.created_at,
    )


def _representative_source_origin(mapping: TimeMap) -> int:
    if mapping.source_origin_ns is not None:
        return int(mapping.source_origin_ns)
    if mapping.anchors:
        return int(mapping.anchors[0].source_ns)
    if mapping.valid_source_start_ns is not None:
        return int(mapping.valid_source_start_ns)
    return 0


def compose_time_maps(first: TimeMap, second: TimeMap) -> TimeMap:
    """Return second(first(t)) while retaining an exact integer reference."""
    if first.target_domain_id != second.source_domain_id:
        raise TimeGraphError("Time maps cannot be composed: domains do not connect")
    scale = second.scale * first.scale
    source_origin = _representative_source_origin(first)
    middle_origin = int(apply_time_map(first, source_origin))
    target_origin = int(apply_time_map(second, middle_origin))
    return TimeMap(
        source_domain_id=first.source_domain_id,
        target_domain_id=second.target_domain_id,
        scale=scale,
        offset_ns=_equivalent_offset(source_origin, target_origin, scale),
        source_origin_ns=source_origin,
        target_origin_ns=target_origin,
        anchors=[],
        quality=TimeMapQuality(
            anchor_count=min(first.quality.anchor_count, second.quality.anchor_count),
            rms_residual_ns=float(
                np.hypot(second.scale * first.quality.rms_residual_ns, second.quality.rms_residual_ns)
            ),
            max_abs_residual_ns=(
                abs(second.scale) * first.quality.max_abs_residual_ns
                + second.quality.max_abs_residual_ns
            ),
            median_abs_residual_ns=(
                abs(second.scale) * first.quality.median_abs_residual_ns
                + second.quality.median_abs_residual_ns
            ),
        ),
        method=f"compose({first.method},{second.method})",
    )


def apply_time_map(mapping: TimeMap, values_ns: int | np.ndarray) -> int | np.ndarray:
    values = np.asarray(values_ns)
    if mapping.source_origin_ns is not None and mapping.target_origin_ns is not None:
        # The subtraction happens as integer arithmetic before conversion.
        if np.issubdtype(values.dtype, np.integer):
            delta = values.astype(np.int64, copy=False) - np.int64(mapping.source_origin_ns)
        else:
            delta = values.astype(np.float64) - float(mapping.source_origin_ns)
        result = np.int64(mapping.target_origin_ns) + np.rint(mapping.scale * delta).astype(np.int64)
    else:
        # Backward compatibility for projects saved before centered clock maps.
        numeric = values.astype(np.float64)
        result = np.rint(mapping.scale * numeric + mapping.offset_ns).astype(np.int64)
    if np.ndim(values_ns) == 0:
        return int(result)
    return result


@dataclass
class TimeGraph:
    domains: set[str]
    mappings: list[TimeMap]

    @classmethod
    def from_session(cls, session) -> "TimeGraph":
        return cls(set(session.time_domains), list(session.time_maps.values()))

    def resolve(self, source_domain_id: str, target_domain_id: str) -> TimeMap:
        if source_domain_id == target_domain_id:
            return TimeMap(
                source_domain_id=source_domain_id,
                target_domain_id=target_domain_id,
                scale=1.0,
                offset_ns=0.0,
                source_origin_ns=0,
                target_origin_ns=0,
                method="identity",
            )
        adjacency: dict[str, list[TimeMap]] = {domain: [] for domain in self.domains}
        for mapping in self.mappings:
            adjacency.setdefault(mapping.source_domain_id, []).append(mapping)
            adjacency.setdefault(mapping.target_domain_id, []).append(invert_time_map(mapping))
        queue: deque[tuple[str, TimeMap | None]] = deque([(source_domain_id, None)])
        visited = {source_domain_id}
        while queue:
            domain, accumulated = queue.popleft()
            for edge in adjacency.get(domain, []):
                if edge.target_domain_id in visited:
                    continue
                combined = edge if accumulated is None else compose_time_maps(accumulated, edge)
                if edge.target_domain_id == target_domain_id:
                    return combined
                visited.add(edge.target_domain_id)
                queue.append((edge.target_domain_id, combined))
        raise TimeGraphError(
            f"No time-map path connects '{source_domain_id}' to '{target_domain_id}'"
        )

    def convert(self, source_domain_id: str, target_domain_id: str, values_ns):
        return apply_time_map(self.resolve(source_domain_id, target_domain_id), values_ns)
