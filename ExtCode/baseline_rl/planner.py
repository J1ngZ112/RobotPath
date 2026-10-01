from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .integration import BaselinePlan, PackageTarget, Point, Waypoint, path_length
from ..baseline_tamp.streams import GridMotionPlanner, LayoutModel

Cell = Tuple[int, int]
State = Tuple[int, int, int, int]
GoalState = Tuple[int, int, int, int]


class QLearningBaseline:
    """A reproducible, goal-conditioned Dyna-Q baseline.

    The observation contains the local obstacle mask, goal direction, distance
    bin, and previous action. A one-step safety shield rejects actions known to
    enter occupied cells; path selection itself is learned from interaction.
    """

    ACTIONS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))

    def __init__(
        self,
        layout: LayoutModel,
        inflation_m: float = 0.55,
        grid_resolution: float = 0.2,
        approach_offset_m: float = 0.9,
        approach_clearance_m: Optional[float] = None,
        seed: int = 0,
        learning_rate: float = 0.18,
        discount: float = 0.97,
        safety_diagnostics: Optional[Dict[str, float]] = None,
    ):
        self.layout = layout
        self.grid = GridMotionPlanner(layout, grid_resolution)
        self.inflation_m = float(inflation_m)
        self.approach_offset_m = float(approach_offset_m)
        self.approach_clearance_m = (
            None if approach_clearance_m is None else float(approach_clearance_m)
        )
        self.seed = int(seed)
        self.learning_rate = float(learning_rate)
        self.discount = float(discount)
        self.safety_diagnostics = dict(safety_diagnostics or {})
        self.q: DefaultDict[State, np.ndarray] = defaultdict(lambda: np.zeros(len(self.ACTIONS), dtype=np.float64))
        self.goal_q: DefaultDict[GoalState, np.ndarray] = defaultdict(
            lambda: np.zeros(len(self.ACTIONS), dtype=np.float64)
        )
        self._occupancy = self.grid._occupancy_with_inflation(self.inflation_m).copy()
        self._trained_episodes = 0
        self._rng = random.Random(self.seed)

    def _free_cells(self) -> List[Cell]:
        return [tuple(map(int, item)) for item in np.argwhere(~self._occupancy)]

    def _valid_actions(self, cell: Cell) -> List[int]:
        valid = []
        for action, (dr, dc) in enumerate(self.ACTIONS):
            row, col = cell[0] + dr, cell[1] + dc
            if not (0 <= row < self.grid.rows and 0 <= col < self.grid.cols):
                continue
            if self._occupancy[row, col]:
                continue
            if dr and dc and (self._occupancy[cell[0] + dr, cell[1]] or self._occupancy[cell[0], cell[1] + dc]):
                continue
            valid.append(action)
        return valid

    @staticmethod
    def _goal_direction(cell: Cell, goal: Cell) -> int:
        angle = math.atan2(goal[0] - cell[0], goal[1] - cell[1])
        return int(round(angle / (math.pi / 4.0))) % 8

    def _obstacle_mask(self, cell: Cell) -> int:
        mask = 0
        for index, (dr, dc) in enumerate(self.ACTIONS):
            row, col = cell[0] + dr, cell[1] + dc
            blocked = not (0 <= row < self.grid.rows and 0 <= col < self.grid.cols) or self._occupancy[row, col]
            if blocked:
                mask |= 1 << index
        return mask

    def _state(self, cell: Cell, goal: Cell, previous_action: int) -> State:
        distance = math.dist(cell, goal)
        distance_bin = min(7, int(distance // 3))
        return self._obstacle_mask(cell), self._goal_direction(cell, goal), distance_bin, previous_action

    def _step(self, cell: Cell, action: int) -> Cell:
        dr, dc = self.ACTIONS[action]
        return cell[0] + dr, cell[1] + dc

    def train(self, episodes: int = 12000, max_steps: int = 180) -> Dict[str, float]:
        free = self._free_cells()
        if len(free) < 2:
            raise ValueError("Layout has fewer than two navigable grid cells")
        successes = 0
        started = time.perf_counter()
        for episode in range(max(0, int(episodes))):
            cell, goal = self._rng.sample(free, 2)
            previous_action = 8
            epsilon = max(0.03, 0.9 * (1.0 - episode / max(1, episodes)))
            for _ in range(max_steps):
                state = self._state(cell, goal, previous_action)
                valid = self._valid_actions(cell)
                if not valid:
                    break
                if self._rng.random() < epsilon:
                    action = self._rng.choice(valid)
                else:
                    best = max(self.q[state][item] for item in valid)
                    action = self._rng.choice([item for item in valid if self.q[state][item] == best])
                nxt = self._step(cell, action)
                old_distance = math.dist(cell, goal)
                new_distance = math.dist(nxt, goal)
                reached = nxt == goal
                reward = 40.0 if reached else -1.0 + 2.0 * (old_distance - self.discount * new_distance)
                next_state = self._state(nxt, goal, action)
                next_valid = self._valid_actions(nxt)
                next_value = max((self.q[next_state][item] for item in next_valid), default=0.0)
                target = reward if reached else reward + self.discount * next_value
                self.q[state][action] += self.learning_rate * (target - self.q[state][action])
                cell, previous_action = nxt, action
                if reached:
                    successes += 1
                    break
        self._trained_episodes += max(0, int(episodes))
        return {
            "episodes": max(0, int(episodes)),
            "success_rate": successes / max(1, episodes),
            "training_time_s": time.perf_counter() - started,
            "q_states": len(self.q),
        }

    def _nearest_free(self, point: Point, max_projection_m: float = 0.30) -> Optional[Cell]:
        max_radius = max(1, int(math.ceil(max_projection_m / self.grid.grid_resolution)))
        return self.grid._nearest_free_cell(
            self.grid.world_to_grid(point), self._occupancy, max_radius=max_radius
        )

    @staticmethod
    def _goal_state(cell: Cell, goal: Cell) -> GoalState:
        return goal[0], goal[1], cell[0], cell[1]

    def _train_for_goal(self, start: Cell, goal: Cell, episodes: int = 500, max_steps: int = 300) -> float:
        free = self._free_cells()
        successes = 0
        maximum_distance = max(math.dist(cell, goal) for cell in free)
        for episode in range(episodes):
            curriculum_radius = 2.0 + maximum_distance * episode / max(1, episodes - 1)
            curriculum = [cell for cell in free if math.dist(cell, goal) <= curriculum_radius]
            cell = start if self._rng.random() < 0.55 else self._rng.choice(curriculum)
            epsilon = max(0.04, 0.75 * (1.0 - episode / max(1, episodes)))
            trajectory = []
            for _ in range(max_steps):
                state = self._goal_state(cell, goal)
                valid = self._valid_actions(cell)
                if not valid:
                    break
                if self._rng.random() < epsilon:
                    action = self._rng.choice(valid)
                else:
                    best = max(self.goal_q[state][item] for item in valid)
                    action = self._rng.choice([item for item in valid if self.goal_q[state][item] == best])
                nxt = self._step(cell, action)
                reached = nxt == goal
                old_distance = math.dist(cell, goal)
                new_distance = math.dist(nxt, goal)
                reward = 60.0 if reached else -1.0 + 2.0 * (old_distance - self.discount * new_distance)
                trajectory.append((state, action, reward))
                next_state = self._goal_state(nxt, goal)
                next_valid = self._valid_actions(nxt)
                next_value = max((self.goal_q[next_state][item] for item in next_valid), default=0.0)
                target = reward if reached else reward + self.discount * next_value
                self.goal_q[state][action] += self.learning_rate * (target - self.goal_q[state][action])
                cell = nxt
                if reached:
                    successes += 1
                    episode_return = 0.0
                    for visited_state, visited_action, visited_reward in reversed(trajectory):
                        episode_return = visited_reward + self.discount * episode_return
                        self.goal_q[visited_state][visited_action] += 0.3 * (
                            episode_return - self.goal_q[visited_state][visited_action]
                        )
                    break
        # Dyna-Q replay over the deterministic grid model propagates sparse
        # terminal reward through long detours much faster than real rollouts.
        for _ in range(200):
            replay_cells = list(free)
            self._rng.shuffle(replay_cells)
            for cell in replay_cells:
                state = self._goal_state(cell, goal)
                for action in self._valid_actions(cell):
                    nxt = self._step(cell, action)
                    reached = nxt == goal
                    old_distance = math.dist(cell, goal)
                    new_distance = math.dist(nxt, goal)
                    reward = 60.0 if reached else -1.0 + 2.0 * (
                        old_distance - self.discount * new_distance
                    )
                    next_valid = self._valid_actions(nxt)
                    next_value = max(
                        (self.goal_q[self._goal_state(nxt, goal)][item] for item in next_valid),
                        default=0.0,
                    )
                    target = reward if reached else reward + self.discount * next_value
                    self.goal_q[state][action] += 0.6 * (
                        target - self.goal_q[state][action]
                    )
        return successes / max(1, episodes)

    def _rollout(self, start: Cell, goal: Cell, goal_specific: bool, max_steps: int) -> Optional[List[Point]]:
        cell = start
        cells = [cell]
        visits = defaultdict(int)
        previous_action = 8
        for _ in range(max_steps):
            if cell == goal:
                points = [self.grid.grid_to_world(item) for item in cells]
                return self.grid._smooth_path(points, self._occupancy)
            valid = self._valid_actions(cell)
            if not valid:
                return None
            values = self.goal_q[self._goal_state(cell, goal)] if goal_specific else self.q[
                self._state(cell, goal, previous_action)
            ]
            best_value = max(values[action] for action in valid)
            best_actions = [action for action in valid if math.isclose(values[action], best_value, abs_tol=1e-9)]
            action = min(
                best_actions,
                key=lambda item: (visits[self._step(cell, item)], math.dist(self._step(cell, item), goal), item),
            )
            nxt = self._step(cell, action)
            visits[nxt] += 1
            if visits[nxt] > 6:
                return None
            cell, previous_action = nxt, action
            cells.append(cell)
        return None

    def plan_path(self, start: Point, goal: Point, max_steps: int = 800) -> Optional[List[Point]]:
        start_cell = self._nearest_free(start, max_projection_m=0.60)
        goal_cell = self._nearest_free(goal, max_projection_m=0.30)
        if start_cell is None or goal_cell is None:
            return None
        path = self._rollout(start_cell, goal_cell, goal_specific=False, max_steps=max_steps)
        if path:
            return path
        self._train_for_goal(start_cell, goal_cell)
        return self._rollout(start_cell, goal_cell, goal_specific=True, max_steps=max_steps)

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

    def _ordered_packages(self, start: Point, packages: Sequence[PackageTarget]) -> List[PackageTarget]:
        # The RL baseline learns motion. A deterministic urgent-first ordering
        # keeps task constraints equal across methods and isolates navigation.
        current = start
        remaining = list(packages)
        ordered = []
        while remaining:
            urgent = [package for package in remaining if package.urgent]
            candidates = urgent or remaining
            chosen = min(
                candidates,
                key=lambda package: (math.dist(current, package.position), package.package_id),
            )
            ordered.append(chosen)
            remaining.remove(chosen)
            current = chosen.position
        return ordered

    def repair_path(self, start: Point, goal: Point, collision_point: Optional[Point]) -> Optional[List[Point]]:
        if collision_point is not None:
            row, col = self.grid.world_to_grid(collision_point)
            radius = max(1, int(math.ceil(0.35 / self.grid.grid_resolution)))
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    if dr * dr + dc * dc <= radius * radius:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < self.grid.rows and 0 <= nc < self.grid.cols:
                            self._occupancy[nr, nc] = True
        self.train(episodes=1500, max_steps=120)
        return self.plan_path(start, goal)

    def plan(self, start: Point, packages: Sequence[PackageTarget], training_episodes: int = 12000) -> BaselinePlan:
        started = time.perf_counter()
        training = self.train(training_episodes)
        current = start
        records: List[Waypoint] = []
        sequence: List[str] = []
        for package in self._ordered_packages(start, packages):
            options = []
            for goal in self._candidate_goals(package):
                segment = self.plan_path(current, goal)
                if segment:
                    options.append((path_length(segment), segment))
            if not options:
                return BaselinePlan(
                    method="dyna_q_learning",
                    success=False,
                    waypoints=records,
                    pickup_sequence=sequence,
                    planning_time_s=time.perf_counter() - started,
                    failure_reason=f"policy_failed:{package.package_id}",
                    diagnostics={"training": training},
                )
            segment = min(options, key=lambda item: item[0])[1]
            # Consecutive packages can resolve to the same coarse RL grid cell.
            # Keep the endpoint before removing a duplicated segment start; the
            # remaining path may legitimately become empty in that case.
            segment_endpoint = segment[-1]
            new_points = segment
            if records and math.dist(records[-1].coordinates_m, segment[0]) < 1e-6:
                new_points = segment[1:]
            for index, point in enumerate(new_points):
                pickup = index == len(new_points) - 1
                records.append(Waypoint(point, "pickup" if pickup else "transit", package.package_id if pickup else None))
            if not new_points:
                # Preserve task semantics even though no base motion is needed.
                records.append(Waypoint(segment_endpoint, "pickup", package.package_id))
            current = segment_endpoint
            sequence.append(package.package_id)
        points = [start] + [waypoint.coordinates_m for waypoint in records]
        return BaselinePlan(
            method="dyna_q_learning",
            success=True,
            waypoints=records,
            pickup_sequence=sequence,
            planning_time_s=time.perf_counter() - started,
            collision_free=not self.grid.path_has_collision(points, self.inflation_m),
            diagnostics={
                "training": training,
                "goal_specific_q_states": len(self.goal_q),
                "path_length_m": path_length(points),
                "robot_safety": self.safety_diagnostics,
            },
        )
