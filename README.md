# Trajectory Generation

Minimal 3D voxel-based path planning utilities with a quick demo for generating collision-free routes through random occupancy grids.

## What this repository contains
- `traj_gen/voxel_path_planner.py`: A lightweight A* implementation over voxel grids with optional inflation for robot radius.
- `run_voxel_planner_demo.py`: Script that samples a random map, plans a path, and saves a static view plus an optional rotating GIF.
- Sample outputs: `voxel_demo.png` and `voxel_demo.gif`.

## Getting started
1. Create and activate a Python 3.8+ environment.
2. Install dependencies (and the package itself in editable mode):
   ```bash
   pip install -e .
   ```
3. Run the demo to generate a random world, plan a path, and export visuals:
   ```bash
   python run_voxel_planner_demo.py --shape 30 30 10 --obstacle-prob 0.2 --robot-radius 0.25 --save-gif voxel_demo.gif
   ```

Key flags: `--connectivity {6,18,26}` picks neighbor set, `--max-iters` caps A* expansions, and `--heuristic-weight` scales the heuristic (>=1.0 keeps it admissible).

## Development notes
- Packaging metadata lives in `setup.py`.
- The planner expects occupancy grids shaped `(x, y, z)` with world coordinates `origin + (idx + 0.5) * voxel_size`.
- Feel free to swap the random map generator with real occupancy data to exercise the planner on real scenarios.

## License
MIT
