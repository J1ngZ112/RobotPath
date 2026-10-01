from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


Point = Tuple[float, float]
SCHEMA_VERSION = "baseline_decoupled_v1"


def resolve_waypoint_path(raw_path: str, project_root: Path) -> Path:
    """Resolve a UI-entered waypoint path relative to the extension root."""
    value = str(raw_path or "").strip().strip('"').strip("'").strip()
    if not value:
        raise ValueError("Waypoint JSON path is empty")
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if not path.is_absolute():
        path = Path(project_root) / path
    return path.resolve()


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_experiment_input(
    path: Path,
    method: str,
    layout_name: str,
    plan,
    warehouse_path: Path,
    robot_context: Dict[str, Any],
    packages: Sequence[Dict[str, Any]],
) -> Path:
    """Export planning output and scene context without running either model."""
    warehouse = _read_json(Path(warehouse_path))
    plan_data = plan.to_dict()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "stage": "planning_output_for_offline_collision_prediction_and_repair",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_method": method,
        "layout_name": layout_name,
        "planning_result": plan_data,
        # Kept at the top level so the existing predictor can consume this file.
        "waypoint_plan": plan_data["waypoint_plan"],
        "environment": warehouse,
        "robot": {
            "robot_id": robot_context.get("robot_id"),
            "size": robot_context.get("robot_size"),
            "initial_state": robot_context.get("robot_info"),
        },
        "packages": list(packages),
        "expected_model_output": {
            "schema_version": SCHEMA_VERSION,
            "source_baseline_method": method,
            "collision_prediction": {
                "collision_detected": "boolean",
                "collision_segment": "[start_index, end_index] or null",
                "collision_object": "string or null",
                "collision_position": "[x, y] or null",
            },
            "repair_result": {
                "repair_applied": "boolean",
                "repair_status": "model_proposal | algorithm_fallback | not_needed | failed",
            },
            "waypoint_plan": {
                "waypoints": [
                    {
                        "waypoint_id": 0,
                        "coordinates_m": [0.0, 0.0],
                        "purpose": "transit | pickup | collision_repair",
                        "package_id": None,
                    }
                ]
            },
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_model_output(
    path: Path,
    expected_method: str | None = None,
) -> tuple[Dict[str, Any], List[Point]]:
    """Load a repaired waypoint JSON; method metadata is optional for unified execution."""
    payload = _read_json(Path(path))
    version = payload.get("schema_version")
    if version is not None and version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version {version!r}; expected {SCHEMA_VERSION!r}"
        )
    source_method = payload.get("source_baseline_method", payload.get("baseline_method"))
    if expected_method is not None and source_method != expected_method:
        raise ValueError(
            f"Model output is for {source_method!r}, not {expected_method!r}"
        )
    waypoint_plan = payload.get("waypoint_plan")
    rows = waypoint_plan.get("waypoints") if isinstance(waypoint_plan, dict) else None
    if rows is None:
        rows = payload.get("waypoints")
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "Repaired JSON must contain a non-empty waypoint_plan.waypoints or waypoints list"
        )

    points: List[Point] = []
    for index, row in enumerate(rows):
        coordinates = row.get("coordinates_m") if isinstance(row, dict) else row
        if not isinstance(coordinates, (list, tuple)) or len(coordinates) < 2:
            raise ValueError(f"Waypoint {index} has invalid coordinates_m")
        try:
            point = float(coordinates[0]), float(coordinates[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Waypoint {index} coordinates must be numeric") from exc
        if not all(math.isfinite(value) for value in point):
            raise ValueError(f"Waypoint {index} coordinates must be finite")
        if not points or point != points[-1]:
            points.append(point)
    if not points:
        raise ValueError("Model output contains no executable waypoints")
    return payload, points


def write_simulation_result(
    path: Path,
    method: str,
    model_output_path: Path,
    success: bool,
    controller,
    failure_info: Dict[str, Any] | None = None,
) -> Path:
    result = {
        "schema_version": SCHEMA_VERSION,
        "stage": "simulation_validation_result",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_baseline_method": method,
        "model_output_path": str(model_output_path),
        "success": bool(success),
        "failure": failure_info,
        "simulation_time_s": controller.get_simulation_time(),
        "executed_trajectory": controller.get_executed_trajectory(),
        "contact_diagnostics": controller.get_contact_diagnostics(),
        "motion_config": controller.get_motion_config(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return path


def _json_default(value):
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
