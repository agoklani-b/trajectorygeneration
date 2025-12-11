import numpy as np
from typing import Optional


def _world_to_idx(xyz: np.ndarray, origin: np.ndarray, voxel_size: float) -> np.ndarray:
    """Convert world coordinates to integer voxel indices."""
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


def _catmull_rom_segment(p0, p1, p2, p3, t: np.ndarray) -> np.ndarray:
    """Uniform Catmull-Rom spline segment from p1 to p2."""
    t = t[:, None]
    return (
        0.5
        * (
            (2 * p1)
            + (-p0 + p2) * t
            + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
            + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3
        )
    )


def smooth_path_catmull_rom(
    path_world: np.ndarray,
    inflated_occupancy: np.ndarray,
    voxel_size: float,
    origin: np.ndarray,
    samples_per_segment: int = 8,
) -> Optional[np.ndarray]:
    """
    Smooth a piecewise-linear path with a Catmull-Rom spline while checking collisions.

    Returns the smoothed path if all sampled points are free in the inflated occupancy grid,
    otherwise returns None to signal failure (caller can fall back to the original path).
    """
    if path_world is None or len(path_world) < 2:
        return None
    pts = np.asarray(path_world, dtype=float)
    if len(pts) < 4:
        # Not enough points to spline; just return original
        return pts

    padded = np.vstack([pts[0], pts, pts[-1]])
    ts = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
    smoothed = []
    for i in range(len(pts) - 1):
        p0, p1, p2, p3 = padded[i], padded[i + 1], padded[i + 2], padded[i + 3]
        seg = _catmull_rom_segment(p0, p1, p2, p3, ts)
        smoothed.append(seg)
    smoothed.append(pts[-1][None, :])
    smoothed_path = np.vstack(smoothed)

    idxs = _world_to_idx(smoothed_path, np.asarray(origin, dtype=float), voxel_size)
    inb = _in_bounds(idxs, inflated_occupancy.shape)
    if not np.all(inb):
        return None
    occupied = inflated_occupancy[idxs[:, 0], idxs[:, 1], idxs[:, 2]]
    if np.any(occupied):
        return None
    return smoothed_path
