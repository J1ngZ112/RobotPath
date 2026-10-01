from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..baseline_tamp.streams import LayoutModel

Point = Tuple[float, float]


@dataclass(frozen=True)
class PackageTarget:
    package_id: str
    position: Point
    shelf_id: str
    size_class: str = "small"
    urgent: bool = False

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "PackageTarget":
        position = value.get("position", value.get("position_m", value.get("coordinates_m")))
        if not position or len(position) < 2:
            raise ValueError(f"Package {value!r} has no 2D position")
        return cls(
            str(value.get("package_id", value.get("name", value.get("id", "package")))),
            (float(position[0]), float(position[1])),
            str(value.get("shelf_id", value.get("shelf", ""))),
            str(value.get("size_class", "small")).lower(),
            bool(value.get("urgent", False)),
        )


@dataclass(frozen=True)
class Waypoint:
    coordinates_m: Point
    purpose: str = "transit"
    package_id: Optional[str] = None


@dataclass
class BaselinePlan:
    method: str
    success: bool
    waypoints: List[Waypoint] = field(default_factory=list)
    pickup_sequence: List[str] = field(default_factory=list)
    planning_time_s: float = 0.0
    collision_free: bool = False
    failure_reason: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def waypoint_points(self) -> List[Point]:
        return [waypoint.coordinates_m for waypoint in self.waypoints]

    def to_dict(self) -> Dict[str, Any]:
        rows = [
            {"waypoint_id": index, "coordinates_m": list(waypoint.coordinates_m),
             "purpose": waypoint.purpose, "package_id": waypoint.package_id}
            for index, waypoint in enumerate(self.waypoints)
        ]
        return {
            "baseline_method": self.method,
            "success": self.success,
            "planning_time_s": round(float(self.planning_time_s), 6),
            "failure_reason": self.failure_reason,
            "pickup_plan": {"sequence": [
                {"order": index + 1, "package_id": package_id}
                for index, package_id in enumerate(self.pickup_sequence)
            ]},
            "waypoint_plan": {"waypoints": rows, "waypoint_count": len(rows)},
            "collision_prediction": {
                "collision_detected": not self.collision_free,
                "source": "rl_safety_shield",
            },
            "diagnostics": self.diagnostics,
        }

    def write_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def load_packages(path: Path) -> List[PackageTarget]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("packages", raw.get("packages_to_pickup", []))
    if not isinstance(raw, list):
        raise ValueError("Package input must be a JSON list or contain a 'packages' list")
    return [PackageTarget.from_dict(item) for item in raw]


def path_length(points: Sequence[Point]) -> float:
    return sum(math.dist(a, b) for a, b in zip(points, points[1:]))


class BaselineExecutionAdapter:
    def __init__(self, controller, repair_callback: Callable, max_replans: int = 3):
        self.controller = controller
        self.repair_callback = repair_callback
        self.max_replans = max(0, int(max_replans))
        self.replan_count = 0
        self.collision_events = 0
        self.failures: List[Dict[str, Any]] = []
        self._terminal_callback = None

    def execute(self, plan: BaselinePlan, on_terminal_failure=None) -> None:
        if not plan.success or not plan.waypoints:
            raise ValueError("Cannot execute an unsuccessful or empty baseline plan")
        self.replan_count = self.collision_events = 0
        self.failures = []
        self._terminal_callback = on_terminal_failure
        self.controller.set_failure_callback(self._on_failure)
        self.controller.start_waypoint_navigation(plan.waypoint_points())

    def _on_failure(self, info: Dict[str, Any]) -> None:
        self.failures.append(dict(info))
        if info.get("type") == "collision":
            self.collision_events += 1
        remaining = list(info.get("remaining_waypoints") or [])
        current_raw = info.get("current_position")
        if self.replan_count >= self.max_replans or not remaining or not current_raw:
            if self._terminal_callback:
                self._terminal_callback(info)
            return
        details = info.get("details") or {}
        raw_collision = details.get("collision_position", details.get("first_contact_position"))
        collision = None if not raw_collision else (float(raw_collision[0]), float(raw_collision[1]))
        current = (float(current_raw[0]), float(current_raw[1]))
        goal = (float(remaining[0][0]), float(remaining[0][1]))
        repaired = self.repair_callback(current, goal, collision)
        if not repaired:
            if self._terminal_callback:
                self._terminal_callback(info)
            return
        suffix = [(float(point[0]), float(point[1])) for point in remaining[1:]]
        if repaired[0] == current:
            repaired = repaired[1:]
        self.replan_count += 1
        self.controller.start_waypoint_navigation(repaired + suffix)


def _package_size(raw: Dict[str, Any]) -> str:
    if raw.get("size_class"):
        return str(raw["size_class"]).lower()
    largest = max(float(value) for value in (raw.get("scale") or [0.4, 0.4, 0.4])[:3])
    return "large" if largest >= 0.47 else "medium" if largest >= 0.38 else "small"


def _distance_to_shelf(point: Point, shelf) -> float:
    dx, dy = point[0] - shelf.center[0], point[1] - shelf.center[1]
    short = max(0.0, abs(dx * shelf.short_axis[0] + dy * shelf.short_axis[1]) - shelf.width / 2.0)
    long = max(0.0, abs(dx * shelf.long_axis[0] + dy * shelf.long_axis[1]) - shelf.length / 2.0)
    return math.hypot(short, long)


def packages_from_scene(layout: LayoutModel, rows: Sequence[Dict[str, Any]]) -> List[PackageTarget]:
    packages = []
    for index, raw in enumerate(rows):
        position = raw.get("position") or raw.get("position_m")
        if not position or len(position) < 2:
            continue
        point = (float(position[0]), float(position[1]))
        shelf_id = str(raw.get("shelf_id", ""))
        if not shelf_id and layout.shelves:
            shelf_id = min(
                layout.shelves,
                key=lambda shelf: (_distance_to_shelf(point, shelf), math.dist(point, shelf.center)),
            ).element_id
        packages.append(PackageTarget(
            str(raw.get("package_id", raw.get("name", index))), point, shelf_id,
            _package_size(raw), bool(raw.get("urgent", False)),
        ))
    return packages


def compute_robot_safety(
    robot_size: Optional[Dict[str, Any]],
    physical_margin_m: float = 0.03,
    tracking_margin_m: float = 0.02,
    docking_margin_m: float = 0.05,
) -> Dict[str, float]:
    spec = robot_size or {}
    radius = None
    circle = spec.get("bounding_circle") or {}
    if isinstance(circle, dict) and circle.get("radius") is not None:
        radius = float(circle["radius"])
    polygon = spec.get("polygon") or []
    if radius is None and polygon:
        valid = [point for point in polygon if point and len(point) >= 2]
        if valid:
            radius = max(math.hypot(float(point[0]), float(point[1])) for point in valid)
    if radius is None:
        radius = 0.5 * math.hypot(
            float(spec.get("width", 0.8)), float(spec.get("length", 1.0))
        )
    planning_inflation = radius + physical_margin_m + tracking_margin_m
    return {
        "footprint_radius_m": float(radius),
        "physical_margin_m": float(physical_margin_m),
        "tracking_margin_m": float(tracking_margin_m),
        "docking_margin_m": float(docking_margin_m),
        "planning_inflation_m": float(planning_inflation),
        "approach_clearance_from_shelf_surface_m": float(planning_inflation + docking_margin_m),
    }


def build_runtime_plan(
    layouts_path: Path, layout_name: str, robot_position: Point, rows,
    seed: int = 0, training_episodes: int = 12000,
    robot_size: Optional[Dict[str, Any]] = None,
):
    from .planner import QLearningBaseline

    layout = LayoutModel.from_layouts_file(layouts_path, layout_name)
    packages = packages_from_scene(layout, rows)
    if not packages:
        raise ValueError("No valid packages were found in the current scene")
    safety = compute_robot_safety(robot_size)
    planner = QLearningBaseline(
        layout,
        inflation_m=safety["planning_inflation_m"],
        approach_clearance_m=safety["approach_clearance_from_shelf_surface_m"],
        seed=seed,
        safety_diagnostics=safety,
    )
    print(
        "RL safety: "
        f"robot={str((robot_size or {}).get('robot_id', 'unknown'))}, "
        f"radius={safety['footprint_radius_m']:.3f} m, "
        f"inflation={safety['planning_inflation_m']:.3f} m, "
        f"surface_clearance={safety['approach_clearance_from_shelf_surface_m']:.3f} m"
    )
    return planner, planner.plan(robot_position, packages, training_episodes)


class IsaacBaselineSession:
    def __init__(self, controller, planner, plan: BaselinePlan, max_replans: int = 3):
        self.plan = plan
        self.planner = planner
        self.executor = BaselineExecutionAdapter(controller, planner.repair_path, max_replans)

    def start(self, on_terminal_failure=None) -> None:
        self.executor.execute(self.plan, on_terminal_failure)

    def metrics(self) -> Dict[str, Any]:
        return {
            "method": self.plan.method,
            "planning_time_s": self.plan.planning_time_s,
            "planned_collision_free": self.plan.collision_free,
            "replan_count": self.executor.replan_count,
            "collision_events": self.executor.collision_events,
            "failures": self.executor.failures,
        }


def run_cli(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the RL mobile-picking baseline offline")
    parser.add_argument("--layouts", default="Experiment/layouts.json")
    parser.add_argument("--layout", required=True)
    parser.add_argument("--packages", required=True)
    parser.add_argument("--start", nargs=2, type=float, default=(0.0, 0.0))
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--training-episodes", type=int, default=12000)
    args = parser.parse_args(argv)
    planner, plan = build_runtime_plan(
        Path(args.layouts), args.layout, tuple(args.start),
        [vars(package) for package in load_packages(Path(args.packages))],
        args.seed, args.training_episodes,
    )
    plan.write_json(Path(args.output))
    print(json.dumps({"success": plan.success, "output": args.output, "diagnostics": plan.diagnostics}, indent=2))
    return 0 if plan.success else 2
