# Trajectory Generation

Lightweight 3D voxel-based planning utilities plus a live correction helper to keep a vehicle on its nominal path even when it drifts due to disturbances.

## Contents
- `traj_gen/voxel_path_planner.py`: 3D A* over occupancy grids with inflation for robot radius and configurable connectivity (6/18/26).
- `traj_gen/path_correction.py`: Projects a live pose onto the planned path and builds a short cubic Bezier that bends back to the closest point without fully recomputing the route.
- `run_voxel_planner_demo.py`: End-to-end demo that samples a random map, plans a path, optionally generates a correction curve from a given current position, and saves a static view plus an optional rotating GIF.
- Artifacts: `voxel_demo.png`, `voxel_demo.gif` show a typical random run.

## Quickstart
1) Python 3.8+ environment.
2) Install:
```bash
pip install -e .
```
3) Plan a path on a random map:
```bash
python run_voxel_planner_demo.py --shape 30 30 10 --obstacle-prob 0.2 --robot-radius 0.25 --save-gif voxel_demo.gif
```
4) Simulate online correction from a drifting pose:
```bash
python run_voxel_planner_demo.py --current-position 1.0 2.0 0.5 --correction-pull 0.35 --correction-steps 25
```

Key flags: `--connectivity {6,18,26}` selects neighbor set; `--max-iters` caps expansions; `--heuristic-weight` scales the heuristic (>=1.0 keeps it admissible).

## How the correction works
- We continuously project the live drone position onto the nominal path (closest point on the polyline).
- A cubic Bezier bridge pulls from the live pose toward that point, with tunable aggressiveness (`pull_strength`) and sampling density (`num_points`).
- The output path is simply: correction curve + remainder of the nominal path, so the vehicle smoothly rejoins and continues.

### API snippets
```python
import numpy as np
from traj_gen import AStar3D, bezier_correction_to_path

# plan
planner = AStar3D(voxel_size=0.2, robot_radius=0.25, connectivity=26)
path = planner.plan(occupancy_grid, start_xyz, goal_xyz)
path_world = np.vstack(path)

# live correction
current_xyz = np.array([1.0, 2.0, 0.5])
corrected_path, meta = bezier_correction_to_path(
    current_position=current_xyz,
    planned_path=path_world,
    pull_strength=0.35,
    num_points=25,
)
```

## Notes
- Occupancy grid convention: `occupancy[x, y, z]`; world position = `origin + (idx + 0.5) * voxel_size`.
- `setup.py` contains packaging metadata and dependencies (numpy, scipy, matplotlib, imageio).
- Swap the random map generator with real occupancy data to test against actual environments.

## License
MIT
