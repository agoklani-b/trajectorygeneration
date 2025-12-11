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

    start_idx = (0, 0, 0)
    goal_idx = (args.shape[0] - 1, args.shape[1] - 1, max(0, args.shape[2] // 2 - 1))
    start_world = origin + (np.array(start_idx) + 0.5) * args.voxel_size
    goal_world = origin + (np.array(goal_idx) + 0.5) * args.voxel_size

    path = planner.plan(occupancy, start_world, goal_world)
    if path is None:
        print("No path found; try lowering obstacle-prob or robot-radius.")
        return 1
    path_world = np.vstack(path)
    print(f"Found nominal path with {len(path_world)} waypoints.")

    inflated = planner._inflate(occupancy)
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
