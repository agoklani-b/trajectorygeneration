import numpy as np
from typing import Dict, Tuple, Optional


def _project_to_segment(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Project point onto segment ab and return the clamped projection, t in [0,1], and segment length."""
    ab = b - a
    seg_len = float(np.linalg.norm(ab))
    denom = seg_len * seg_len
    if denom < 1e-12:
        t = 0.0
    else:
        t = float(np.dot(point - a, ab) / denom)
    t_clamped = float(np.clip(t, 0.0, 1.0))
    proj = a + t_clamped * ab
    return proj, t_clamped, seg_len


def closest_point_on_path(point: np.ndarray, path: np.ndarray) -> Tuple[np.ndarray, int, float]:
    """Return closest point on a polyline, the segment index, and the segment parameter t in [0, 1]."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3:
        raise ValueError("path must have shape (N, 3)")
    if len(path) == 0:
        raise ValueError("path is empty")

    point = np.asarray(point, dtype=float)
    best = (float("inf"), path[0], 0, 0.0)
    for i in range(len(path) - 1):
        proj, t_clamped, _ = _project_to_segment(point, path[i], path[i + 1])
        dist2 = float(np.sum((proj - point) ** 2))
        if dist2 < best[0]:
            best = (dist2, proj, i, t_clamped)
    return best[1], best[2], best[3]


def forward_point_on_path(
    point: np.ndarray,
    path: np.ndarray,
    tolerance: float = 1e-3,
    min_forward_progress: float = 0.0,
    lookahead_distance: float = 0.5,
) -> Tuple[np.ndarray, int, float, float]:
    """
    Choose a rejoin point that maintains forward progress toward the goal.

    Strategy:
    - Compute the closest point on the path and its along-path curvilinear position s.
    - Among segments with s >= s_closest - tolerance, choose the nearest projection.
    - Fall back to the absolute closest point if no forward segment is found.
    Returns (point, segment_index, t_on_segment, s_position).
    """
    path = np.asarray(path, dtype=float)
    n = len(path)
    if n < 2:
        return path[0], 0, 0.0, 0.0

    seg_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_len = cum_lengths[-1]

    point = np.asarray(point, dtype=float)
    closest_any = None
    for i in range(n - 1):
        proj, t_clamped, seg_len = _project_to_segment(point, path[i], path[i + 1])
        s_here = cum_lengths[i] + t_clamped * seg_len
        dist2 = float(np.sum((proj - point) ** 2))
        if closest_any is None or dist2 < closest_any[0]:
            closest_any = (dist2, proj, i, t_clamped, s_here)

    s_closest = closest_any[4]
    desired_s = min(total_len, s_closest + max(min_forward_progress, lookahead_distance))

    # find segment containing desired_s
    seg_idx = None
    for i in range(n - 1):
        if desired_s + tolerance <= cum_lengths[i]:
            continue
        if desired_s <= cum_lengths[i + 1] + tolerance:
            seg_idx = i
            break
    if seg_idx is None:
        seg_idx = n - 2

    seg_len = seg_lengths[seg_idx]
    if seg_len < 1e-9:
        return path[seg_idx], seg_idx, 0.0, cum_lengths[seg_idx]
    t = float(np.clip((desired_s - cum_lengths[seg_idx]) / seg_len, 0.0, 1.0))
    rejoin = path[seg_idx] + t * (path[seg_idx + 1] - path[seg_idx])
    return rejoin, seg_idx, t, desired_s


def bezier_correction_to_path(
    current_position: np.ndarray,
    planned_path: np.ndarray,
    pull_strength: float = 0.35,
    num_points: int = 25,
    min_forward_progress: float = 0.1,
    forward_push: float = 0.2,
    lookahead_distance: float = 0.6,
    occupancy_inflated: Optional[np.ndarray] = None,
    voxel_size: Optional[float] = None,
    origin: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Bridge the drone's current position back to the planned path with a smooth cubic Bezier.

    Args:
        current_position: 3-vector of the drone position in world coordinates.
        planned_path: Nx3 array of the nominal path (waypoints in world coordinates).
        pull_strength: Scale for how aggressively the curve bends toward the path (0-1 typical).
        num_points: Number of interpolated points on the correction curve (>=2).
        min_forward_progress: Minimum along-path distance (m) beyond the nearest point to target.
        forward_push: Scale for forward bias along path tangent in Bezier control points.
        lookahead_distance: Additional along-path distance (m) to bias the rejoin target forward.
        occupancy_inflated: Optional inflated occupancy grid to collision-check the correction curve.
        voxel_size: Required if occupancy_inflated is provided.
        origin: Required if occupancy_inflated is provided.

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

    nearest_point, seg_idx, t, s_pos = forward_point_on_path(
        current_position,
        planned_path,
        min_forward_progress=min_forward_progress,
        lookahead_distance=lookahead_distance,
    )
    to_path = nearest_point - current_position
    dist = float(np.linalg.norm(to_path))

    # Approximate path tangent at the rejoin point for smoother heading
    a = planned_path[seg_idx]
    b = planned_path[min(seg_idx + 1, len(planned_path) - 1)]
    tangent = b - a
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm > 1e-9:
        tangent = tangent / tangent_norm
    else:
        tangent = np.array([0.0, 0.0, 0.0])

    p0 = current_position
    p3 = nearest_point
    forward_term = forward_push * dist * tangent
    p1 = p0 + pull_strength * to_path + 0.4 * forward_term
    p2 = p3 - pull_strength * to_path + forward_term

    t_vals = np.linspace(0.0, 1.0, num_points)
    curve = (
        (1 - t_vals)[:, None] ** 3 * p0
        + 3 * (1 - t_vals)[:, None] ** 2 * t_vals[:, None] * p1
        + 3 * (1 - t_vals)[:, None] * t_vals[:, None] ** 2 * p2
        + t_vals[:, None] ** 3 * p3
    )

    if occupancy_inflated is not None:
        if voxel_size is None or origin is None:
            raise ValueError("voxel_size and origin are required when occupancy_inflated is provided.")
        dense = curve if len(curve) >= num_points else np.linspace(p0, p3, max(num_points * 2, 50))
        if not _collision_free(dense, occupancy_inflated, voxel_size, origin):
            return None, {
                "segment_index": int(seg_idx),
                "t_on_segment": float(t),
                "s_on_path": float(s_pos),
                "distance": dist,
                "nearest_point": nearest_point,
                "collision": True,
            }

    remainder = planned_path[seg_idx + 1 :]
    if remainder.size > 0 and np.linalg.norm(remainder[0] - p3) < 1e-6:
        remainder = remainder[1:]

    corrected_path = np.vstack([curve, remainder])
    meta = {
        "segment_index": int(seg_idx),
        "t_on_segment": float(t),
        "s_on_path": float(s_pos),
        "distance": dist,
        "nearest_point": nearest_point,
    }
    return corrected_path, meta
def _world_to_idx(xyz: np.ndarray, origin: np.ndarray, voxel_size: float) -> np.ndarray:
    return np.floor((xyz - origin) / voxel_size).astype(int)


def _in_bounds(idxs: np.ndarray, shape) -> np.ndarray:
    return (
        (idxs[:, 0] >= 0)
        & (idxs[:, 0] < shape[0])
        & (idxs[:, 1] >= 0)
        & (idxs[:, 1] < shape[1])
        & (idxs[:, 2] >= 0)
        & (idxs[:, 2] < shape[2])
    )


def _collision_free(points: np.ndarray, occupancy: np.ndarray, voxel_size: float, origin: np.ndarray) -> bool:
    """Check that all points are in free space (occupancy already inflated)."""
    idxs = _world_to_idx(points, np.asarray(origin, dtype=float), voxel_size)
    inb = _in_bounds(idxs, occupancy.shape)
    if not np.all(inb):
        return False
    return not np.any(occupancy[idxs[:, 0], idxs[:, 1], idxs[:, 2]])
