from __future__ import annotations

import heapq
import math
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .integration import BaselinePlan, PackageTarget, Point, Waypoint, deduplicate_points, path_length
from ..baseline_tamp.streams import GridMotionPlanner, LayoutModel

Cell = Tuple[int, int]


class HeuristicBaseline:
    """Nearest-neighbour task ordering plus load-aware A* motion planning."""

    def __init__(
        self,
        layout: LayoutModel,
        inflation_profile: Optional[Dict[str, float]] = None,
        grid_resolution: float = 0.1,
        approach_offset_m: float = 0.9,
        approach_clearance_m: Optional[float] = None,
        repair_obstacle_radius_m: float = 0.35,
        safety_diagnostics: Optional[Dict[str, float]] = None,
    ):
        self.layout = layout
        self.grid = GridMotionPlanner(layout, grid_resolution)
        self.inflation_profile = inflation_profile or {
            "unloaded": 0.55,
            "small": 0.60,
            "medium": 0.68,
            "large": 0.78,
        }
        self.approach_offset_m = float(approach_offset_m)
        self.approach_clearance_m = (
            None if approach_clearance_m is None else float(approach_clearance_m)
        )
        self.repair_obstacle_radius_m = float(repair_obstacle_radius_m)
        self.safety_diagnostics = dict(safety_diagnostics or {})
        self._dynamic_obstacles: List[Point] = []

    def _inflation(self, size_class: str) -> float:
        return float(self.inflation_profile.get(size_class, self.inflation_profile["unloaded"]))

    def _ordered_packages(self, start: Point, packages: Sequence[PackageTarget]) -> List[PackageTarget]:
        current = start
        remaining = list(packages)
        ordered: List[PackageTarget] = []
        while remaining:
            urgent = [pkg for pkg in remaining if pkg.urgent]
            candidates = urgent or remaining
            chosen = min(candidates, key=lambda pkg: (math.dist(current, pkg.position), pkg.package_id))
            ordered.append(chosen)
            remaining.remove(chosen)
            current = chosen.position
        return ordered

    def _candidate_goals(self, package: PackageTarget) -> Sequence[Point]:
        if package.shelf_id:
            shelf = self.layout.get_shelf(package.shelf_id)
            if self.approach_clearance_m is not None:
                dx = package.position[0] - shelf.center[0]
                dy = package.position[1] - shelf.center[1]
                along = dx * shelf.long_axis[0] + dy * shelf.long_axis[1]
                along = max(-shelf.length / 2.0, min(shelf.length / 2.0, along))
                centerline = (
                    shelf.center[0] + shelf.long_axis[0] * along,
                    shelf.center[1] + shelf.long_axis[1] * along,
                )
                offset = shelf.width / 2.0 + self.approach_clearance_m
                return shelf.approach_candidates(centerline, offset)
            return shelf.approach_candidates(package.position, self.approach_offset_m)
        return (package.position,)

    def _occupancy(self, inflation: float) -> np.ndarray:
        occupancy = self.grid._occupancy_with_inflation(inflation).copy()
        radius_cells = int(math.ceil(self.repair_obstacle_radius_m / self.grid.grid_resolution))
        for point in self._dynamic_obstacles:
            row, col = self.grid.world_to_grid(point)
            for dr in range(-radius_cells, radius_cells + 1):
                for dc in range(-radius_cells, radius_cells + 1):
                    if dr * dr + dc * dc > radius_cells * radius_cells:
                        continue
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < self.grid.rows and 0 <= nc < self.grid.cols:
                        occupancy[nr, nc] = True
        return occupancy

    def _plan_cells(self, start: Point, goal: Point, inflation: float) -> Optional[List[Point]]:
        occupancy = self._occupancy(inflation)
        start_radius = max(1, int(math.ceil(0.60 / self.grid.grid_resolution)))
        goal_radius = max(1, int(math.ceil(0.30 / self.grid.grid_resolution)))
        start_cell = self.grid._nearest_free_cell(
            self.grid.world_to_grid(start), occupancy, max_radius=start_radius
        )
        goal_cell = self.grid._nearest_free_cell(
            self.grid.world_to_grid(goal), occupancy, max_radius=goal_radius
        )
        if start_cell is None or goal_cell is None:
            return None
        open_set = [(math.dist(start_cell, goal_cell), 0.0, start_cell)]
        came_from: Dict[Cell, Cell] = {}
        costs = {start_cell: 0.0}
        visited: Set[Cell] = set()
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        while open_set:
            _, cost, cell = heapq.heappop(open_set)
            if cell in visited:
                continue
            visited.add(cell)
            if cell == goal_cell:
                cells = [cell]
                while cell in came_from:
                    cell = came_from[cell]
                    cells.append(cell)
                cells.reverse()
                return self.grid._smooth_path([self.grid.grid_to_world(item) for item in cells], occupancy)
            for dr, dc in neighbors:
                nxt = (cell[0] + dr, cell[1] + dc)
                if not (0 <= nxt[0] < self.grid.rows and 0 <= nxt[1] < self.grid.cols):
                    continue
                if occupancy[nxt] or (dr and dc and (occupancy[cell[0] + dr, cell[1]] or occupancy[cell[0], cell[1] + dc])):
                    continue
                new_cost = cost + (math.sqrt(2.0) if dr and dc else 1.0)
                if new_cost >= costs.get(nxt, float("inf")):
                    continue
                costs[nxt] = new_cost
                came_from[nxt] = cell
                heapq.heappush(open_set, (new_cost + math.dist(nxt, goal_cell), new_cost, nxt))
        return None

    def plan_path(self, start: Point, goal: Point, size_class: str = "unloaded") -> Optional[List[Point]]:
        return self._plan_cells(start, goal, self._inflation(size_class))

    def repair_path(self, start: Point, goal: Point, collision_point: Optional[Point]) -> Optional[List[Point]]:
        if collision_point is not None:
            self._dynamic_obstacles.append(collision_point)
        return self.plan_path(start, goal, "unloaded")

    def plan(self, start: Point, packages: Sequence[PackageTarget]) -> BaselinePlan:
        started = time.perf_counter()
        current = start
        records: List[Waypoint] = []
        sequence: List[str] = []
        expanded_segments = 0
        for package in self._ordered_packages(start, packages):
            options = []
            for goal in self._candidate_goals(package):
                segment = self.plan_path(current, goal, "unloaded")
                if segment:
                    options.append((path_length(segment), segment))
            if not options:
                return BaselinePlan(
                    method="heuristic_astar",
                    success=False,
                    waypoints=records,
                    pickup_sequence=sequence,
                    planning_time_s=time.perf_counter() - started,
                    failure_reason=f"unreachable_package:{package.package_id}",
                    diagnostics={"planned_segments": expanded_segments},
                )
            segment = min(options, key=lambda item: item[0])[1]
            points = deduplicate_points(segment)
            if records and points and math.dist(records[-1].coordinates_m, points[0]) < 1e-6:
                points = points[1:]
            for index, point in enumerate(points):
                is_pickup = index == len(points) - 1
                records.append(Waypoint(point, "pickup" if is_pickup else "transit", package.package_id if is_pickup else None))
            current = segment[-1]
            sequence.append(package.package_id)
            expanded_segments += 1
        points = [start] + [item.coordinates_m for item in records]
        collision_free = not self.grid.path_has_collision(points, self._inflation("unloaded"))
        return BaselinePlan(
            method="heuristic_astar",
            success=True,
            waypoints=records,
            pickup_sequence=sequence,
            planning_time_s=time.perf_counter() - started,
            collision_free=collision_free,
            diagnostics={
                "planned_segments": expanded_segments,
                "path_length_m": path_length(points),
                "robot_safety": self.safety_diagnostics,
            },
        )
