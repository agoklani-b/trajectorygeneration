import numpy as np
from typing import Dict, Tuple


def closest_point_on_path(point: np.ndarray, path: np.ndarray) -> Tuple[np.ndarray, int, float]:
    """Return closest point on a polyline, the segment index, and the segment parameter t in [0, 1]."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3:
        raise ValueError("path must have shape (N, 3)")
    if len(path) == 0:
        raise ValueError("path is empty")

    point = np.asarray(point, dtype=float)
    best_dist2 = float("inf")
    best_point = path[0]
    best_idx = 0
    best_t = 0.0

    if len(path) == 1:
        return best_point, best_idx, best_t

    for i in range(len(path) - 1):
        a = path[i]
        b = path[i + 1]
        ab = b - a
        denom = np.dot(ab, ab)
        if denom < 1e-12:
            t = 0.0
        else:
            t = np.dot(point - a, ab) / denom
        t_clamped = float(np.clip(t, 0.0, 1.0))
        proj = a + t_clamped * ab
        dist2 = float(np.sum((proj - point) ** 2))
        if dist2 < best_dist2:
            best_dist2 = dist2
            best_point = proj
            best_idx = i
            best_t = t_clamped
    return best_point, best_idx, best_t


def bezier_correction_to_path(
    current_position: np.ndarray,
    planned_path: np.ndarray,
    pull_strength: float = 0.35,
    num_points: int = 25,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Bridge the drone's current position back to the planned path with a smooth cubic Bezier.

    Args:
        current_position: 3-vector of the drone position in world coordinates.
        planned_path: Nx3 array of the nominal path (waypoints in world coordinates).
        pull_strength: Scale for how aggressively the curve bends toward the path (0-1 typical).
        num_points: Number of interpolated points on the correction curve (>=2).

    Returns:
        corrected_path: Concatenation of the correction curve and the remainder of the planned path.
        meta: Metadata with `segment_index`, `distance`, and `nearest_point`.
    """
    if num_points < 2:
        raise ValueError("num_points must be at least 2")

    planned_path = np.asarray(planned_path, dtype=float)
    if planned_path.ndim != 2 or planned_path.shape[1] != 3:
        raise ValueError("planned_path must have shape (N, 3)")
    if len(planned_path) < 2:
        raise ValueError("planned_path must have at least two waypoints")

    current_position = np.asarray(current_position, dtype=float).reshape(3)

    nearest_point, seg_idx, t = closest_point_on_path(current_position, planned_path)
    to_path = nearest_point - current_position
    dist = float(np.linalg.norm(to_path))

    # Approximate path tangent at the rejoin point for smoother heading
    a = planned_path[seg_idx]
    b = planned_path[min(seg_idx + 1, len(planned_path) - 1)]
    tangent = b - a
    if np.linalg.norm(tangent) > 1e-9:
        tangent = tangent / np.linalg.norm(tangent)

    p0 = current_position
    p3 = nearest_point
    p1 = p0 + pull_strength * to_path
    p2 = p3 - pull_strength * to_path + 0.5 * dist * tangent

    t_vals = np.linspace(0.0, 1.0, num_points)
    curve = (
        (1 - t_vals)[:, None] ** 3 * p0
        + 3 * (1 - t_vals)[:, None] ** 2 * t_vals[:, None] * p1
        + 3 * (1 - t_vals)[:, None] * t_vals[:, None] ** 2 * p2
        + t_vals[:, None] ** 3 * p3
    )

    remainder = planned_path[seg_idx + 1 :]
    if remainder.size > 0 and np.linalg.norm(remainder[0] - p3) < 1e-6:
        remainder = remainder[1:]

    corrected_path = np.vstack([curve, remainder])
    meta = {
        "segment_index": int(seg_idx),
        "t_on_segment": float(t),
        "distance": dist,
        "nearest_point": nearest_point,
    }
    return corrected_path, meta
