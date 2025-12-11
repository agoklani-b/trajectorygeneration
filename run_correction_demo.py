import argparse
import sys
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")  # ensure headless rendering
import matplotlib.pyplot as plt
import numpy as np

from traj_gen import AStar3D, bezier_correction_to_path, smooth_path_catmull_rom


def generate_random_grid(shape: Tuple[int, int, int], obstacle_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a random occupancy grid with the given fill probability."""
    return (rng.random(shape) < obstacle_prob).astype(float)


def pick_free_cell(free_mask: np.ndarray, rng: np.random.Generator) -> Tuple[int, int, int]:
    """Pick a random free voxel index from a boolean free-mask."""
    free = np.argwhere(free_mask)
    if free.size == 0:
        raise RuntimeError("No free cells to choose from.")
    idx = rng.integers(0, free.shape[0])
    return tuple(free[idx].tolist())


def sample_start_and_goal(
    free_mask: np.ndarray,
    rng: np.random.Generator,
    voxel_size: float,
    origin: np.ndarray,
    min_z_layers: int = 1,
    min_dist: float = 1.0,
    max_attempts: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample start/goal in free space with separation and vertical difference for better visuals."""
    free = np.argwhere(free_mask)
    if free.size == 0:
        raise RuntimeError("No free cells to choose from.")
    attempts = 0
    while attempts < max_attempts:
        s_idx = free[rng.integers(0, len(free))]
        g_idx = free[rng.integers(0, len(free))]
        if np.array_equal(s_idx, g_idx):
            attempts += 1
            continue
        if abs(s_idx[2] - g_idx[2]) < min_z_layers:
            attempts += 1
            continue
        s_world = origin + (s_idx + 0.5) * voxel_size
        g_world = origin + (g_idx + 0.5) * voxel_size
        if np.linalg.norm(s_world - g_world) < min_dist:
            attempts += 1
            continue
        return s_world, g_world
    # fallback
    s_idx = free[rng.integers(0, len(free))]
    g_idx = free[rng.integers(0, len(free))]
    while np.array_equal(s_idx, g_idx):
        g_idx = free[rng.integers(0, len(free))]
    return origin + (s_idx + 0.5) * voxel_size, origin + (g_idx + 0.5) * voxel_size


def sample_drift_positions(path: np.ndarray, num_drifts: int, sigma: float, rng: np.random.Generator) -> List[np.ndarray]:
    """Pick positions near the path by adding randomized 3D offsets to random waypoints (excluding endpoints)."""
    if len(path) < 3:
        raise ValueError("Path too short to sample drift positions.")
    idxs = rng.integers(1, len(path) - 1, size=num_drifts)
    offsets = []
    for _ in range(num_drifts):
        direction = rng.normal(size=3)
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = direction / norm
        mag = rng.uniform(0.5 * sigma, sigma)
        offsets.append(direction * mag)
    return [path[i] + offsets[j] for j, i in enumerate(idxs)]


def set_axes_equal(ax, pts: np.ndarray) -> None:
    """Set 3D axes limits to equal scales based on given points."""
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    ranges = maxs - mins
    max_range = ranges.max()
    centers = (maxs + mins) / 2
    half = max_range / 2
    ax.set_xlim(centers[0] - half, centers[0] + half)
    ax.set_ylim(centers[1] - half, centers[1] + half)
    ax.set_zlim(centers[2] - half, centers[2] + half)


def visualize(path_world: np.ndarray, drifts: List[np.ndarray], corrections: List[np.ndarray], save_path: str) -> None:
    """Plot nominal path, drifted positions, and correction curves."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(path_world[:, 0], path_world[:, 1], path_world[:, 2], "-", c="#1f77b4", linewidth=1.5, label="Nominal path")
    ax.scatter(path_world[0, 0], path_world[0, 1], path_world[0, 2], c="green", marker="o", s=60, label="Start")
    ax.scatter(path_world[-1, 0], path_world[-1, 1], path_world[-1, 2], c="magenta", marker="x", s=80, label="Goal")

    correction_labeled = False
    drift_labeled = False
    for drift, corrected in zip(drifts, corrections):
        if corrected is None:
            continue
        curve = corrected  # correction + remainder; we highlight full curve for clarity
        ax.plot(
            curve[:, 0],
            curve[:, 1],
            curve[:, 2],
            "--",
            c="#ff7f0e",
            linewidth=1.2,
            alpha=0.9,
            label="Correction" if not correction_labeled else "",
        )
        correction_labeled = True
        ax.scatter(drift[0], drift[1], drift[2], c="#ff7f0e", marker="^", s=60, label="Drifted pose" if not drift_labeled else "")
        drift_labeled = True

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title("Path correction demo")
    handles, labels = ax.get_legend_handles_labels()
    dedup = {}
    for h, l in zip(handles, labels):
        if l and l not in dedup:
            dedup[l] = h
    ax.legend(dedup.values(), dedup.keys())
    pts = np.vstack([path_world] + [c for c in corrections if c is not None] + [np.array(drifts)])
    set_axes_equal(ax, pts)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"Saved visualization to {save_path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Visualize online path correction from drifted poses.")
    parser.add_argument("--shape", type=int, nargs=3, default=[30, 30, 10], help="Grid shape (x y z)")
    parser.add_argument("--voxel-size", type=float, default=0.2, help="Meters per voxel")
    parser.add_argument("--obstacle-prob", type=float, default=0.2, help="Probability a voxel is occupied")
    parser.add_argument("--robot-radius", type=float, default=0.25, help="Collision tolerance radius [m]")
    parser.add_argument("--connectivity", type=int, choices=[6, 18, 26], default=26, help="Neighborhood connectivity")
    parser.add_argument("--seed", type=int, default=3, help="Random seed")
    parser.add_argument("--num-drifts", type=int, default=3, help="How many drifted poses to correct")
    parser.add_argument("--drift-sigma", type=float, default=1.0, help="Scale for sampling drift around path [m]")
    parser.add_argument("--correction-pull", type=float, default=0.35, help="Bezier pull strength toward the path")
    parser.add_argument("--correction-steps", type=int, default=25, help="Number of samples along the correction curve")
    parser.add_argument("--correction-lookahead", type=float, default=0.6, help="Along-path distance (m) to bias rejoin forward")
    parser.add_argument("--correction-min-progress", type=float, default=0.1, help="Minimum forward progress (m) before rejoining")
    parser.add_argument("--correction-forward-push", type=float, default=0.3, help="Forward bias along tangent for Bezier controls")
    parser.add_argument("--save-path", type=str, default="correction_demo.png", help="Where to save the visualization")
    parser.add_argument("--smooth-samples", type=int, default=8, help="Samples per segment for path smoothing")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    occupancy = generate_random_grid(tuple(args.shape), args.obstacle_prob, rng)
    origin = np.zeros(3)

    planner = AStar3D(
        voxel_size=args.voxel_size,
        occ_threshold=0.5,
        robot_radius=args.robot_radius,
        connectivity=args.connectivity,
        heuristic_weight=1.0,
        max_iterations=500000,
        origin=tuple(origin.tolist()),
    )

    inflated = planner._inflate(occupancy)
    free_mask = ~inflated

    try:
        start_world, goal_world = sample_start_and_goal(
            free_mask,
            rng,
            args.voxel_size,
            origin,
            min_z_layers=2,
            min_dist=2.0,
        )
    except RuntimeError:
        print("No path found; free space too limited.")
        return 1

    path = planner.plan(occupancy, start_world, goal_world)
    if path is None:
        print("No path found; try lowering obstacle-prob or robot-radius.")
        return 1
    path_world = np.vstack(path)
    print(f"Found nominal path with {len(path_world)} waypoints.")

    smoothed_path_world = smooth_path_catmull_rom(
        path_world,
        inflated,
        voxel_size=args.voxel_size,
        origin=origin,
        samples_per_segment=args.smooth_samples,
    )
    if smoothed_path_world is None:
        smoothed_path_world = path_world
        print("Smoothing failed (collision or bounds); using original path.")
    else:
        print(f"Smoothed path has {len(smoothed_path_world)} points.")

    drifts = sample_drift_positions(smoothed_path_world, args.num_drifts, args.drift_sigma, rng)
    corrections = []
    valid_drifts = []
    for i, drift in enumerate(drifts):
        corrected, meta = bezier_correction_to_path(
            current_position=drift,
            planned_path=smoothed_path_world,
            pull_strength=args.correction_pull,
            num_points=args.correction_steps,
            min_forward_progress=args.correction_min_progress,
            forward_push=args.correction_forward_push,
            lookahead_distance=args.correction_lookahead,
            occupancy_inflated=inflated,
            voxel_size=args.voxel_size,
            origin=origin,
        )
        if corrected is None:
            print(
                f"[Drift {i}] correction rejected due to collision; skipping visualization. "
                f"(segment {meta['segment_index']}, t={meta['t_on_segment']:.2f})"
            )
            continue
        valid_drifts.append(drift)
        corrections.append(corrected)
        print(
            f"[Drift {i}] distance to path: {meta['distance']:.2f} m, "
            f"rejoin segment {meta['segment_index']} at t={meta['t_on_segment']:.2f}"
        )

    if len(corrections) == 0:
        print("No collision-free corrections found; nothing to visualize.")
        return 1

    visualize(smoothed_path_world, valid_drifts, corrections, args.save_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
