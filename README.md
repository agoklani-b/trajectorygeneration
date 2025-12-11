# Trajectory Generation

Lightweight 3D voxel planning utilities with smoothing and online correction to keep a vehicle on its intended path, even under drift.

## Features
- **Voxel A\***: `traj_gen/voxel_path_planner.py` plans in 3D occupancy grids with robot-radius inflation and 6/18/26-connectivity.
- **Path smoothing**: `traj_gen/path_smoothing.py` converts the waypoint polyline to a Catmull–Rom spline while collision-checking against the inflated map.
- **Online correction**: `traj_gen/path_correction.py` projects a live pose to a forward point on the path (avoids backtracking), collision-checks the curve, and builds a forward-biased Bezier bridge back.
- **Demos**:
  - `run_voxel_planner_demo.py`: plan, optional smoothing, optional live correction, and static/GIF visualization.
  - `run_correction_demo.py`: generate multiple drifted poses, correct each, and visualize the curves.
- **Artifacts**: `voxel_demo.png`, `voxel_demo.gif`, `correction_demo.png` show typical runs.

## Setup
1) Python 3.8+ environment  
2) Install:
```bash
pip install -e .
```

## Demo commands
- Plan + smooth + export visuals (nominal only):
```bash
python run_voxel_planner_demo.py --shape 30 30 10 --obstacle-prob 0.2 --robot-radius 0.25 --smooth --save-gif voxel_demo.gif
```
- Add a live correction from a drifting pose:
```bash
python run_voxel_planner_demo.py --current-position 1.0 2.0 0.5 --correction-pull 0.35 --correction-steps 25 --smooth
```
- Visualize multiple drift corrections at once:
```bash
python run_correction_demo.py --num-drifts 3 --drift-sigma 0.6 --correction-pull 0.35 --smooth-samples 10
```

Key flags: `--connectivity {6,18,26}` neighbor set; `--max-iters` expansion cap; `--heuristic-weight` heuristic scale (>=1 for admissibility); `--smooth-samples` Catmull–Rom density.

## How smoothing and correction work
- Smoothing samples a Catmull–Rom spline per segment; if any sample hits the inflated occupancy grid, the smoothed path is discarded and the original polyline is used.
- Live correction (in `run_correction_demo.py`) finds a forward point on the (smoothed) path, then generates a cubic Bezier curve that rejoins smoothly—leaving the nominal voxel demo to show the baseline path only.

## API snippet
```python
import numpy as np
from traj_gen import AStar3D, smooth_path_catmull_rom, bezier_correction_to_path

planner = AStar3D(voxel_size=0.2, robot_radius=0.25, connectivity=26)
path = planner.plan(occupancy_grid, start_xyz, goal_xyz)
path_world = np.vstack(path)
inflated = planner._inflate(occupancy_grid)
smooth_path = smooth_path_catmull_rom(path_world, inflated, voxel_size=0.2, origin=np.zeros(3)) or path_world

current_xyz = np.array([1.0, 2.0, 0.5])
corrected_path, meta = bezier_correction_to_path(
    current_xyz,
    smooth_path,
    pull_strength=0.35,
    num_points=25,
    min_forward_progress=0.1,    # meters of along-path progress before rejoining
    lookahead_distance=0.6,      # push rejoin target forward along the path
    forward_push=0.3,            # bias control points along path tangent
    occupancy_inflated=inflated, # collision check the correction curve
    voxel_size=0.2,
    origin=np.zeros(3),
)
# meta includes the segment index/t chosen, biased toward forward progress
```

## Notes
- Occupancy grid convention: `occupancy[x, y, z]`; world position = `origin + (idx + 0.5) * voxel_size`.
- Dependencies are listed in `setup.py` (numpy, scipy, matplotlib, imageio).
- Swap the random map generator for real occupancy data to test on actual environments.

## License
MIT
