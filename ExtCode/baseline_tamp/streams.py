from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


def _point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        intersects = ((y1 > y) != (y2 > y)) and (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
        )
        if intersects:
            inside = not inside
    return inside


@dataclass(frozen=True)
class Shelf:
    element_id: str
    center: Point
    length: float
    width: float
    orientation_rad: float
    polygon: Tuple[Point, ...]

    @property
    def short_axis(self) -> Point:
        return (math.cos(self.orientation_rad), math.sin(self.orientation_rad))

    @property
    def long_axis(self) -> Point:
        return (-math.sin(self.orientation_rad), math.cos(self.orientation_rad))

    def approach_candidates(self, package_point: Point, offset: float) -> Tuple[Point, Point]:
        sx, sy = self.short_axis
        px, py = package_point
        left = (px + sx * offset, py + sy * offset)
        right = (px - sx * offset, py - sy * offset)
        return left, right


@dataclass(frozen=True)
class LayoutModel:
    layout_name: str
    bounds: Dict[str, float]
    shelves: Tuple[Shelf, ...]

    @classmethod
    def from_layouts_file(cls, layouts_path: Path, layout_name: str) -> "LayoutModel":
        data = json.loads(layouts_path.read_text(encoding="utf-8"))
        layouts = data.get("layouts", {})
        if layout_name not in layouts:
            raise KeyError(f"Layout '{layout_name}' not found in {layouts_path}")

        model = layouts[layout_name].get("warehouse_geometry_model", {})
        summary = model.get("global_summary", {})
        size = summary.get("warehouse_size_m", {"x": 12.0, "y": 12.0})
        bounds = {
            "x_min": -float(size.get("x", 12.0)) / 2.0,
            "x_max": float(size.get("x", 12.0)) / 2.0,
            "y_min": -float(size.get("y", 12.0)) / 2.0,
            "y_max": float(size.get("y", 12.0)) / 2.0,
        }

        shelves: List[Shelf] = []
        for obs in model.get("obstacles", []):
            if obs.get("type") != "Shelf":
                continue
            center_raw = obs.get("center_point_m", [0.0, 0.0])
            dims = obs.get("dimensions_m", {})
            orientation = float(obs.get("orientation_rad", 0.0))
            polygon_raw = obs.get("obstacle_polygon_m", [])

            polygon: List[Point] = []
            for p in polygon_raw:
                if len(p) >= 2:
                    polygon.append((float(p[0]), float(p[1])))

            shelves.append(
                Shelf(
                    element_id=str(obs.get("element_id", "Shelf")),
                    center=(float(center_raw[0]), float(center_raw[1])),
                    length=float(dims.get("length", 8.0)),
                    width=float(dims.get("width", 0.6)),
                    orientation_rad=orientation,
                    polygon=tuple(polygon),
                )
            )

        return cls(layout_name=layout_name, bounds=bounds, shelves=tuple(shelves))

    def get_shelf(self, element_id: str) -> Shelf:
        for shelf in self.shelves:
            if shelf.element_id == element_id:
                return shelf
        raise KeyError(f"Shelf '{element_id}' not found in layout '{self.layout_name}'")


class GridMotionPlanner:
    def __init__(self, layout: LayoutModel, grid_resolution: float = 0.1):
        self.layout = layout
        self.grid_resolution = float(grid_resolution)

        self.x_min = layout.bounds["x_min"]
        self.x_max = layout.bounds["x_max"]
        self.y_min = layout.bounds["y_min"]
        self.y_max = layout.bounds["y_max"]

        self.cols = int(math.ceil((self.x_max - self.x_min) / self.grid_resolution))
        self.rows = int(math.ceil((self.y_max - self.y_min) / self.grid_resolution))

        self._base_occupancy = self._build_base_occupancy()
        self._inflation_cache: Dict[float, np.ndarray] = {}

    def _build_base_occupancy(self) -> np.ndarray:
        occ = np.zeros((self.rows, self.cols), dtype=np.bool_)
        for r in range(self.rows):
            for c in range(self.cols):
                point = self.grid_to_world((r, c))
                for shelf in self.layout.shelves:
                    if _point_in_polygon(point, shelf.polygon):
                        occ[r, c] = True
                        break

        # 边界设为障碍，避免路径贴边导致几何误差
        occ[0, :] = True
        occ[-1, :] = True
        occ[:, 0] = True
        occ[:, -1] = True
        return occ

    def _occupancy_with_inflation(self, inflation_m: float) -> np.ndarray:
        key = round(float(inflation_m), 3)
        cached = self._inflation_cache.get(key)
        if cached is not None:
            return cached

        cells = int(math.ceil(max(0.0, inflation_m) / self.grid_resolution))
        inflated = self._base_occupancy.copy()

        if cells > 0:
            occupied_idx = np.argwhere(self._base_occupancy)
            offsets = [
                (dr, dc)
                for dr in range(-cells, cells + 1)
                for dc in range(-cells, cells + 1)
                if math.hypot(dr, dc) * self.grid_resolution <= inflation_m + 1e-9
            ]
            for r, c in occupied_idx:
                for dr, dc in offsets:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        inflated[nr, nc] = True

        self._inflation_cache[key] = inflated
        return inflated

    def world_to_grid(self, point: Point) -> Tuple[int, int]:
        x, y = point
        c = int((x - self.x_min) / self.grid_resolution)
        r = int((y - self.y_min) / self.grid_resolution)
        c = min(self.cols - 1, max(0, c))
        r = min(self.rows - 1, max(0, r))
        return r, c

    def grid_to_world(self, rc: Tuple[int, int]) -> Point:
        r, c = rc
        x = self.x_min + (c + 0.5) * self.grid_resolution
        y = self.y_min + (r + 0.5) * self.grid_resolution
        return float(x), float(y)

    def _nearest_free_cell(self, start: Tuple[int, int], occ: np.ndarray, max_radius: int = 15) -> Optional[Tuple[int, int]]:
        sr, sc = start
        if not occ[sr, sc]:
            return start

        for radius in range(1, max_radius + 1):
            r0 = max(0, sr - radius)
            r1 = min(self.rows - 1, sr + radius)
            c0 = max(0, sc - radius)
            c1 = min(self.cols - 1, sc + radius)

            for r in range(r0, r1 + 1):
                for c in (c0, c1):
                    if not occ[r, c]:
                        return r, c
            for c in range(c0, c1 + 1):
                for r in (r0, r1):
                    if not occ[r, c]:
                        return r, c
        return None

    def _line_free(self, a: Tuple[int, int], b: Tuple[int, int], occ: np.ndarray) -> bool:
        r0, c0 = a
        r1, c1 = b

        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc

        r, c = r0, c0
        while True:
            if occ[r, c]:
                return False
            if r == r1 and c == c1:
                return True

            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r += sr
            if e2 < dr:
                err += dr
                c += sc

    def _smooth_path(self, path: List[Point], occ: np.ndarray) -> List[Point]:
        if len(path) <= 2:
            return path

        smoothed = [path[0]]
        i = 0
        while i < len(path) - 1:
            anchor = self.world_to_grid(path[i])
            jump = i + 1
            for j in range(len(path) - 1, i, -1):
                candidate = self.world_to_grid(path[j])
                if self._line_free(anchor, candidate, occ):
                    jump = j
                    break
            smoothed.append(path[jump])
            i = jump
        return smoothed

    def plan(
        self,
        start: Point,
        goal: Point,
        inflation_m: float,
        max_steps: int = 200000,
    ) -> Optional[List[Point]]:
        occ = self._occupancy_with_inflation(inflation_m)
        start_cell = self._nearest_free_cell(self.world_to_grid(start), occ)
        goal_cell = self._nearest_free_cell(self.world_to_grid(goal), occ)

        if start_cell is None or goal_cell is None:
            return None

        neighbors = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

        g_score = {start_cell: 0.0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            return math.hypot(a[0] - b[0], a[1] - b[1])

        open_heap: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (heuristic(start_cell, goal_cell), 0.0, start_cell))
        visited = set()
        steps = 0

        while open_heap and steps < max_steps:
            steps += 1
            _, g, cur = heapq.heappop(open_heap)
            if cur in visited:
                continue
            visited.add(cur)

            if cur == goal_cell:
                path_cells = [cur]
                while cur in came_from:
                    cur = came_from[cur]
                    path_cells.append(cur)
                path_cells.reverse()
                path = [self.grid_to_world(cell) for cell in path_cells]
                return self._smooth_path(path, occ)

            for dr, dc in neighbors:
                nr = cur[0] + dr
                nc = cur[1] + dc
                nxt = (nr, nc)
                if nr < 0 or nr >= self.rows or nc < 0 or nc >= self.cols:
                    continue
                if occ[nr, nc]:
                    continue
                step_cost = math.sqrt(2.0) if dr != 0 and dc != 0 else 1.0
                ng = g + step_cost
                if ng >= g_score.get(nxt, float("inf")):
                    continue
                g_score[nxt] = ng
                came_from[nxt] = cur
                f = ng + heuristic(nxt, goal_cell)
                heapq.heappush(open_heap, (f, ng, nxt))

        return None

    def path_has_collision(self, path: Sequence[Point], inflation_m: float) -> bool:
        if not path:
            return True
        occ = self._occupancy_with_inflation(inflation_m)
        for idx in range(1, len(path)):
            a = self.world_to_grid(path[idx - 1])
            b = self.world_to_grid(path[idx])
            if not self._line_free(a, b, occ):
                return True
        return False

    @staticmethod
    def path_length(path: Sequence[Point]) -> float:
        if len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(path)):
            total += math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
        return float(total)


class TAMPStreams:
    def __init__(
        self,
        planner: GridMotionPlanner,
        inflation_profile: Dict[str, float],
        approach_offset_m: float,
    ):
        self.planner = planner
        self.inflation_profile = inflation_profile
        self.approach_offset_m = float(approach_offset_m)

    def _inflation_for(self, size_class: str) -> float:
        norm = size_class.lower()
        if norm not in self.inflation_profile:
            norm = "unloaded"
        return float(self.inflation_profile[norm])

    def sample_motion(self, start: Point, goal: Point) -> Optional[List[Point]]:
        return self.planner.plan(start, goal, self._inflation_for("unloaded"))

    def sample_motion_loaded(self, start: Point, goal: Point, size_class: str) -> Optional[List[Point]]:
        return self.planner.plan(start, goal, self._inflation_for(size_class))

    def sample_approach_pose(self, package_point: Point, shelf: Shelf) -> Tuple[Point, Point]:
        return shelf.approach_candidates(package_point, self.approach_offset_m)

    def check_collision(self, path: Sequence[Point], size_class: str = "unloaded") -> bool:
        inflation = self._inflation_for(size_class)
        return not self.planner.path_has_collision(path, inflation)

    def to_pddlstream_map(self):
        def _as_point(val) -> Point:
            return float(val[0]), float(val[1])

        def gen_motion(p1, p2) -> Iterator[Tuple[List[Point]]]:
            path = self.sample_motion(_as_point(p1), _as_point(p2))
            if path is None:
                return iter(())
            return iter(((path,),))

        def gen_motion_loaded(p1, p2, size_class) -> Iterator[Tuple[List[Point]]]:
            path = self.sample_motion_loaded(_as_point(p1), _as_point(p2), str(size_class))
            if path is None:
                return iter(())
            return iter(((path,),))

        def gen_collision_free(path, size_class="unloaded") -> Iterator[Tuple[bool]]:
            ok = self.check_collision(path, str(size_class))
            if ok:
                return iter(((True,),))
            return iter(())

        return {
            "sample-motion": gen_motion,
            "sample-motion-loaded": gen_motion_loaded,
            "check-collision": gen_collision_free,
        }
