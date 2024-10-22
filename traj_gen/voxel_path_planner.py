import heapq
import math
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation


def spherical_kernel(radius_vox: int) -> np.ndarray:
    """Create a boolean kernel of points within radius in voxel units."""
    if radius_vox <= 0:
        return np.ones((1, 1, 1), dtype=bool)
    rng = np.arange(-radius_vox, radius_vox + 1)
    xx, yy, zz = np.meshgrid(rng, rng, rng, indexing="ij")
    dist2 = xx**2 + yy**2 + zz**2
    return dist2 <= radius_vox**2


class AStar3D:
    """
    3D A* planner over a voxel occupancy grid.

    Grid convention: occupancy[x, y, z] with world position = origin + (idx + 0.5) * voxel_size.
    Occupancy values > occ_threshold are considered filled.
    """

    def __init__(
        self,
        voxel_size: float,
        occ_threshold: float = 0.5,
        robot_radius: float = 0.0,
        connectivity: int = 26,
        heuristic_weight: float = 1.0,
        max_iterations: Optional[int] = None,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        self.voxel_size = voxel_size
        self.occ_threshold = occ_threshold
        self.robot_radius = robot_radius
        self.connectivity = connectivity
        self.heuristic_weight = heuristic_weight
        self.max_iterations = max_iterations
        self.origin = np.asarray(origin, dtype=float)

        if connectivity not in (6, 18, 26):
            raise ValueError("connectivity must be one of 6, 18, 26")
        if heuristic_weight < 1.0:
            raise ValueError("heuristic_weight should be >= 1.0 for admissibility")

        self._neighbor_offsets = self._build_offsets(connectivity)

    @staticmethod
    def _build_offsets(connectivity: int) -> List[Tuple[int, int, int]]:
        offsets = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == dy == dz == 0:
                        continue
                    if connectivity == 6 and sum(abs(v) for v in (dx, dy, dz)) != 1:
                        continue
                    if connectivity == 18 and sum(abs(v) for v in (dx, dy, dz)) > 2:
                        continue
                    offsets.append((dx, dy, dz))
        return offsets

    def _inflate(self, occupancy: np.ndarray) -> np.ndarray:
        radius_vox = int(math.ceil(self.robot_radius / self.voxel_size))
        kernel = spherical_kernel(radius_vox)
        inflated = binary_dilation(occupancy > self.occ_threshold, structure=kernel)
        return inflated

    def _world_to_idx(self, xyz: np.ndarray) -> Optional[Tuple[int, int, int]]:
        idx = np.floor((xyz - self.origin) / self.voxel_size).astype(int)
        return tuple(idx.tolist())

    def _idx_to_world(self, idx: Tuple[int, int, int]) -> np.ndarray:
        return self.origin + (np.array(idx, dtype=float) + 0.5) * self.voxel_size

    def _in_bounds(self, idx: Tuple[int, int, int], shape: Tuple[int, int, int]) -> bool:
        x, y, z = idx
        return 0 <= x < shape[0] and 0 <= y < shape[1] and 0 <= z < shape[2]

    def plan(
        self,
        occupancy: np.ndarray,
        start_xyz: np.ndarray,
        goal_xyz: np.ndarray,
    ) -> Optional[List[np.ndarray]]:
        """Plan a path from start_xyz to goal_xyz avoiding inflated occupancy."""
        inflated = self._inflate(occupancy)
        shape = inflated.shape

        start_idx = self._world_to_idx(start_xyz)
        goal_idx = self._world_to_idx(goal_xyz)
        if not self._in_bounds(start_idx, shape) or not self._in_bounds(goal_idx, shape):
            return None
        if inflated[start_idx] or inflated[goal_idx]:
            return None

        def heuristic(a, b):
            return self.heuristic_weight * np.linalg.norm(np.array(a) - np.array(b))

        g_score = {start_idx: 0.0}
        came_from = {}
        counter = 0
        open_set = [(heuristic(start_idx, goal_idx), counter, start_idx)]

        while open_set:
            _, _, current = heapq.heappop(open_set)
            if current == goal_idx:
                path_idx = [current]
                while current in came_from:
                    current = came_from[current]
                    path_idx.append(current)
                path_idx.reverse()
                return [self._idx_to_world(c) for c in path_idx]

            if self.max_iterations is not None and counter > self.max_iterations:
                break

            cx, cy, cz = current
            for dx, dy, dz in self._neighbor_offsets:
                nxt = (cx + dx, cy + dy, cz + dz)
                if not self._in_bounds(nxt, shape) or inflated[nxt]:
                    continue
                step_cost = math.sqrt(dx * dx + dy * dy + dz * dz)
                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score.get(nxt, float("inf")):
                    came_from[nxt] = current
                    g_score[nxt] = tentative_g
                    counter += 1
                    f = tentative_g + heuristic(nxt, goal_idx)
                    heapq.heappush(open_set, (f, counter, nxt))
        return None
