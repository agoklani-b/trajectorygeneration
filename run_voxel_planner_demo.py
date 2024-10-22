import argparse
import sys
from typing import Tuple

import matplotlib
matplotlib.use("Agg")  # force non-interactive backend for headless environments
import matplotlib.pyplot as plt
try:
    import imageio.v2 as imageio
except Exception:
    imageio = None
import numpy as np

from traj_gen.voxel_path_planner import AStar3D


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


def visualize(
    occupancy: np.ndarray,
    path_world: np.ndarray,
    voxel_size: float,
    origin: np.ndarray,
    title: str = "",
    save_path: str = None,
    save_gif: str = None,
    num_views: int = 12,
) -> None:
    """Show occupancy voxels and path in 3D, save single view and optional rotating GIF."""

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    filled = np.argwhere(occupancy > 0.5)
    if filled.size > 0:
        # centers of occupied voxels
        occ_xyz = origin + (filled + 0.5) * voxel_size
        ax.scatter(occ_xyz[:, 0], occ_xyz[:, 1], occ_xyz[:, 2], c="red", marker="s", alpha=0.4, label="Obstacles")

    if path_world is not None and len(path_world) > 0:
        ax.plot(
            path_world[:, 0],
            path_world[:, 1],
            path_world[:, 2],
            "-",
            c="#1f77b4",
            linewidth=1.5,
            label="Path",
        )
        # direction arrows along path
        if len(path_world) > 1:
            stride = max(1, len(path_world) // 10)
            p0 = path_world[::stride]
            p1 = np.roll(path_world, -stride, axis=0)[::stride]
            vec = p1 - p0
            ax.quiver(
                p0[:, 0], p0[:, 1], p0[:, 2],
                vec[:, 0], vec[:, 1], vec[:, 2],
                length=1.0, normalize=True, color="#1f77b4", linewidth=0.8, arrow_length_ratio=0.3
            )
        ax.scatter(path_world[0, 0], path_world[0, 1], path_world[0, 2], c="green", marker="o", s=60, label="Start")
        ax.scatter(path_world[-1, 0], path_world[-1, 1], path_world[-1, 2], c="magenta", marker="x", s=80, label="Goal")

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(title or "Voxel Planner Demo")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved visualization to {save_path}")

    # Optional rotating GIF for better spatial intuition
    if save_gif:
        if imageio is None:
            print("imageio not available; skipping GIF export.")
        else:
            frames = []
            elev = 20
            for i in range(num_views):
                azim = i * (360.0 / num_views)
                ax.view_init(elev=elev, azim=azim)
                fig.canvas.draw()
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(buf)
            imageio.mimsave(save_gif, frames, fps=6)
            print(f"Saved rotating view to {save_gif}")

    if not save_path and not save_gif:
        plt.show()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Random voxel A* demo with visualization.")
    parser.add_argument("--shape", type=int, nargs=3, default=[30, 30, 10], help="Grid shape (x y z)")
    parser.add_argument("--voxel-size", type=float, default=0.2, help="Meters per voxel")
    parser.add_argument("--obstacle-prob", type=float, default=0.2, help="Probability a voxel is occupied")
    parser.add_argument("--robot-radius", type=float, default=0.25, help="Collision tolerance radius [m]")
    parser.add_argument("--connectivity", type=int, choices=[6, 18, 26], default=26, help="Neighborhood connectivity")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--max-iters", type=int, default=500000, help="Max A* iterations (None for unlimited)")
    parser.add_argument("--heuristic-weight", type=float, default=1.0, help="Heuristic weight (>=1 for admissibility)")
    parser.add_argument("--save-gif", type=str, default="voxel_demo.gif", help="Path to save rotating GIF (set empty to skip)")
    parser.add_argument("--num-views", type=int, default=18, help="Number of views around the path for GIF")
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    occupancy = generate_random_grid(tuple(args.shape), args.obstacle_prob, rng)

    voxel_size = args.voxel_size
    origin = np.zeros(3)

    planner = AStar3D(
        voxel_size=voxel_size,
        occ_threshold=0.5,
        robot_radius=args.robot_radius,
        connectivity=args.connectivity,
        heuristic_weight=args.heuristic_weight,
        max_iterations=args.max_iters,
        origin=tuple(origin.tolist()),
    )

    inflated = planner._inflate(occupancy)
    free_mask = ~inflated

    # Try a few draws for start/goal in free inflated space
    attempts = 0
    start_world = None
    goal_world = None
    while attempts < 20 and (start_world is None or goal_world is None or np.allclose(start_world, goal_world)):
        start_idx = pick_free_cell(free_mask, rng)
        goal_idx = pick_free_cell(free_mask, rng)
        start_world = origin + (np.array(start_idx) + 0.5) * voxel_size
        goal_world = origin + (np.array(goal_idx) + 0.5) * voxel_size
        attempts += 1

    if start_world is None or goal_world is None:
        print("Could not sample free start/goal; grid too full.")
        return 1

    path = planner.plan(occupancy, start_world, goal_world)
    if path is None:
        print("No path found; try lowering obstacle-prob or robot-radius, or increasing max-iters.")
        return 1

    path_world = np.vstack(path)
    print(f"Found path with {len(path_world)} waypoints.")

    save_gif = args.save_gif if args.save_gif else None
    visualize(
        occupancy,
        path_world,
        voxel_size,
        origin,
        title="Voxel Planner Demo",
        save_path="voxel_demo.png",
        save_gif=save_gif,
        num_views=args.num_views,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
