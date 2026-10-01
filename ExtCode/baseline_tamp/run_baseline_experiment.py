from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

if __package__ in (None, ""):
    import sys

    MODULE_DIR = Path(__file__).resolve().parent
    PDDL_DIR = MODULE_DIR / "pddl"
    for p in (MODULE_DIR, PDDL_DIR):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    from pddl.problem_generator import generate_problem_for_trial
    from streams import GridMotionPlanner, LayoutModel, TAMPStreams
else:
    from .pddl.problem_generator import generate_problem_for_trial
    from .streams import GridMotionPlanner, LayoutModel, TAMPStreams


Point = Tuple[float, float]
SIZE_RANK = {"unloaded": 0, "small": 1, "medium": 2, "large": 3}
RANK_TO_SIZE = {0: "unloaded", 1: "small", 2: "medium", 3: "large"}


@dataclass(frozen=True)
class PackageInstance:
    package_id: str
    position: Point
    shelf_id: str
    size_class: str
    urgent: bool
    mass_kg: float
    pose_name: str


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _parse_csv_ints(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _expand_size_distribution(size_distribution: dict, num_packages: int, rng: random.Random) -> List[str]:
    size_list: List[str] = []
    for size_key in ("small", "medium", "large"):
        size_list.extend([size_key] * int(size_distribution.get(size_key, 0)))

    if len(size_list) < num_packages:
        fallback = ["small", "medium", "large"]
        while len(size_list) < num_packages:
            size_list.append(rng.choice(fallback))
    elif len(size_list) > num_packages:
        rng.shuffle(size_list)
        size_list = size_list[:num_packages]

    rng.shuffle(size_list)
    return size_list


def _generate_packages(layout: LayoutModel, scenario: dict, seed: int) -> List[PackageInstance]:
    rng = random.Random(seed)
    num_packages = int(scenario["num_packages"])

    sizes = _expand_size_distribution(scenario.get("size_distribution", {}), num_packages, rng)
    urgent_count = min(int(scenario.get("urgent_count", 0)), num_packages)
    urgent_indices = set(rng.sample(range(num_packages), k=urgent_count))

    packages: List[PackageInstance] = []
    for i in range(num_packages):
        shelf = rng.choice(layout.shelves)

        # 在货架长轴方向采样包裹位置，避免贴近端点
        margin = min(0.8, max(0.2, shelf.length * 0.12))
        span = max(0.2, shelf.length / 2.0 - margin)
        along = rng.uniform(-span, span)
        lx, ly = shelf.long_axis

        px = shelf.center[0] + lx * along
        py = shelf.center[1] + ly * along
        size_class = sizes[i]

        mass_base = {"small": 5.0, "medium": 9.0, "large": 14.0}[size_class]
        mass_noise = rng.uniform(-0.8, 0.8)

        packages.append(
            PackageInstance(
                package_id=f"{i:02d}",
                position=(round(px, 3), round(py, 3)),
                shelf_id=shelf.element_id,
                size_class=size_class,
                urgent=i in urgent_indices,
                mass_kg=round(mass_base + mass_noise, 3),
                pose_name=f"p_pkg_{i:02d}",
            )
        )

    return packages


def _heuristic(current: Point, remaining_indices: Sequence[int], packages: Sequence[PackageInstance]) -> float:
    if not remaining_indices:
        return 0.0
    nearest = min(_distance(current, packages[i].position) for i in remaining_indices)
    # 简单启发: 当前点到最近剩余点 + 轻量剩余项
    return nearest + 0.18 * len(remaining_indices)


def _symbolic_search_sequence(
    packages: Sequence[PackageInstance],
    start_pose: Point,
    urgent_first: bool,
    timeout_s: float,
    max_expansions: Optional[int],
) -> Tuple[Optional[List[int]], int, bool]:
    n = len(packages)
    if n == 0:
        return [], 0, False

    full_mask = (1 << n) - 1
    open_heap: List[Tuple[float, float, int, int, int, Tuple[int, ...]]] = []
    best_cost: Dict[Tuple[int, int], float] = {}

    counter = 0
    start_h = _heuristic(start_pose, list(range(n)), packages)
    open_heap.append((start_h, 0.0, counter, 0, -1, tuple()))
    best_cost[(0, -1)] = 0.0

    expanded_nodes = 0
    t0 = time.perf_counter()

    while open_heap:
        if time.perf_counter() - t0 > timeout_s:
            return None, expanded_nodes, True
        if max_expansions is not None and max_expansions > 0 and expanded_nodes >= max_expansions:
            return None, expanded_nodes, True

        _, g, _, mask, last_idx, seq = heapq.heappop(open_heap)
        expanded_nodes += 1

        state_key = (mask, last_idx)
        if g > best_cost.get(state_key, float("inf")) + 1e-9:
            continue

        if mask == full_mask:
            return list(seq), expanded_nodes, False

        remaining = [i for i in range(n) if not (mask & (1 << i))]
        available = remaining
        if urgent_first:
            remaining_urgent = [i for i in remaining if packages[i].urgent]
            if remaining_urgent:
                available = remaining_urgent

        current = start_pose if last_idx == -1 else packages[last_idx].position
        available = sorted(
            available,
            key=lambda i: (SIZE_RANK.get(packages[i].size_class, 1), _distance(current, packages[i].position), packages[i].package_id),
        )

        for i in available:
            new_mask = mask | (1 << i)
            step = _distance(current, packages[i].position)
            new_g = g + step
            new_key = (new_mask, i)

            if new_g >= best_cost.get(new_key, float("inf")):
                continue

            best_cost[new_key] = new_g
            remaining_after = [j for j in range(n) if not (new_mask & (1 << j))]
            h = _heuristic(packages[i].position, remaining_after, packages)
            counter += 1
            heapq.heappush(open_heap, (new_g + h, new_g, counter, new_mask, i, seq + (i,)))

    return None, expanded_nodes, True


def _symbolic_greedy_sequence(
    packages: Sequence[PackageInstance],
    start_pose: Point,
    urgent_first: bool,
    timeout_s: float,
) -> Tuple[Optional[List[int]], int, bool]:
    n = len(packages)
    if n == 0:
        return [], 0, False

    remaining = set(range(n))
    sequence: List[int] = []
    current = start_pose
    expanded_nodes = 0
    t0 = time.perf_counter()

    while remaining:
        if time.perf_counter() - t0 > timeout_s:
            return None, expanded_nodes, True

        candidates = list(remaining)
        if urgent_first:
            urgent_candidates = [i for i in candidates if packages[i].urgent]
            if urgent_candidates:
                candidates = urgent_candidates

        expanded_nodes += len(candidates)
        best = min(
            candidates,
            key=lambda i: (
                SIZE_RANK.get(packages[i].size_class, 1),
                _distance(current, packages[i].position),
                packages[i].package_id,
            ),
        )

        sequence.append(best)
        remaining.remove(best)
        current = packages[best].position

    return sequence, expanded_nodes, False


def _choose_best_approach_path(
    streams: TAMPStreams,
    current_pose: Point,
    candidates: Sequence[Point],
    move_size: str,
) -> Tuple[Optional[List[Point]], int, int]:
    best_path: Optional[List[Point]] = None
    best_len = float("inf")
    motion_calls = 0
    failed_calls = 0

    for cand in candidates:
        motion_calls += 1
        if move_size == "unloaded":
            path = streams.sample_motion(current_pose, cand)
        else:
            path = streams.sample_motion_loaded(current_pose, cand, move_size)

        if not path:
            failed_calls += 1
            continue

        path_len = streams.planner.path_length(path)
        if path_len < best_len:
            best_len = path_len
            best_path = path

    return best_path, motion_calls, failed_calls


def _stitch_path(full_path: List[Point], segment: Sequence[Point]) -> None:
    if not segment:
        return
    if not full_path:
        full_path.extend(segment)
        return

    if _distance(full_path[-1], segment[0]) < 1e-6:
        full_path.extend(segment[1:])
    else:
        full_path.extend(segment)


def _estimate_drop_pose(layout: LayoutModel) -> Point:
    return (0.0, round(layout.bounds["y_max"] - 1.0, 3))


def _run_trial(
    scenario: dict,
    layout: LayoutModel,
    streams: TAMPStreams,
    seed: int,
    start_pose: Point,
    planning_timeout_s: float,
    max_symbolic_expansions: Optional[int],
    symbolic_mode: str,
    planner_name: str,
) -> dict:
    trial_id = f"{scenario['id']}_seed_{seed}"
    t0 = time.perf_counter()

    packages = _generate_packages(layout, scenario, seed)
    urgent_first = bool(scenario.get("constraints", {}).get("urgent_first", False))
    load_dependent_inflation = bool(scenario.get("constraints", {}).get("load_dependent_inflation", False))

    symbolic_timeout_sec = planning_timeout_s * 0.75
    if symbolic_mode == "greedy":
        sequence, expanded_nodes, symbolic_timeout = _symbolic_greedy_sequence(
            packages=packages,
            start_pose=start_pose,
            urgent_first=urgent_first,
            timeout_s=symbolic_timeout_sec,
        )
    else:
        sequence, expanded_nodes, symbolic_timeout = _symbolic_search_sequence(
            packages=packages,
            start_pose=start_pose,
            urgent_first=urgent_first,
            timeout_s=symbolic_timeout_sec,
            max_expansions=max_symbolic_expansions,
        )

    motion_calls_total = 0
    motion_calls_failed = 0
    replan_count = 0
    collision_events = 0
    total_path: List[Point] = [start_pose]
    status_reason = "ok"

    if sequence is None:
        planning_time = time.perf_counter() - t0
        status_reason = "symbolic_timeout"
        return {
            "trial_id": trial_id,
            "seed": seed,
            "scenario_id": scenario["id"],
            "layout": scenario["layout"],
            "num_packages": int(scenario["num_packages"]),
            "planner_name": planner_name,
            "symbolic_mode": symbolic_mode,
            "planning_time_s": round(planning_time, 4),
            "solved_within_timeout": 0,
            "execution_success": 0,
            "end_to_end_success": 0,
            "symbolic_expanded_nodes": expanded_nodes,
            "motion_calls_total": motion_calls_total,
            "motion_calls_failed": motion_calls_failed,
            "replan_count": replan_count,
            "total_path_length_m": 0.0,
            "collision_events": collision_events,
            "status_reason": status_reason,
            "pickup_sequence": [],
            "packages": [
                {
                    "package_id": p.package_id,
                    "position": [p.position[0], p.position[1]],
                    "shelf_id": p.shelf_id,
                    "size_class": p.size_class,
                    "urgent": p.urgent,
                    "mass_kg": p.mass_kg,
                    "pose_name": p.pose_name,
                }
                for p in packages
            ],
        }

    current_pose = start_pose
    carried_rank = 0

    for seq_idx in sequence:
        if time.perf_counter() - t0 > planning_timeout_s:
            status_reason = "planning_timeout"
            break

        pkg = packages[seq_idx]
        shelf = layout.get_shelf(pkg.shelf_id)
        candidates = streams.sample_approach_pose(pkg.position, shelf)

        if load_dependent_inflation and carried_rank > 0:
            move_size = RANK_TO_SIZE[carried_rank]
        else:
            move_size = "unloaded"

        best_segment, calls, failed = _choose_best_approach_path(
            streams=streams,
            current_pose=current_pose,
            candidates=candidates,
            move_size=move_size,
        )
        motion_calls_total += calls
        motion_calls_failed += failed

        if not best_segment:
            status_reason = "motion_no_path"
            break

        _stitch_path(total_path, best_segment)
        current_pose = best_segment[-1]
        carried_rank = max(carried_rank, SIZE_RANK.get(pkg.size_class, 1))

    if status_reason == "ok":
        drop_pose = _estimate_drop_pose(layout)
        move_size = RANK_TO_SIZE[carried_rank] if (load_dependent_inflation and carried_rank > 0) else "unloaded"
        final_segment, calls, failed = _choose_best_approach_path(
            streams=streams,
            current_pose=current_pose,
            candidates=[drop_pose],
            move_size=move_size,
        )
        motion_calls_total += calls
        motion_calls_failed += failed

        if not final_segment:
            status_reason = "dropoff_no_path"
        else:
            _stitch_path(total_path, final_segment)

    planning_time = time.perf_counter() - t0
    solved_within_timeout = int(status_reason == "ok" and planning_time <= planning_timeout_s)

    if solved_within_timeout:
        final_size_for_collision = RANK_TO_SIZE[carried_rank] if load_dependent_inflation else "unloaded"
        if not streams.check_collision(total_path, final_size_for_collision):
            collision_events = 1
            status_reason = "collision_check_failed"

    execution_success = int(solved_within_timeout == 1 and collision_events == 0)
    end_to_end_success = execution_success

    return {
        "trial_id": trial_id,
        "seed": seed,
        "scenario_id": scenario["id"],
        "layout": scenario["layout"],
        "num_packages": int(scenario["num_packages"]),
        "planner_name": planner_name,
        "symbolic_mode": symbolic_mode,
        "planning_time_s": round(planning_time, 4),
        "solved_within_timeout": solved_within_timeout,
        "execution_success": execution_success,
        "end_to_end_success": end_to_end_success,
        "symbolic_expanded_nodes": expanded_nodes,
        "motion_calls_total": motion_calls_total,
        "motion_calls_failed": motion_calls_failed,
        "replan_count": replan_count,
        "total_path_length_m": round(streams.planner.path_length(total_path), 4),
        "collision_events": collision_events,
        "status_reason": status_reason,
        "pickup_sequence": [packages[i].package_id for i in sequence],
        "packages": [
            {
                "package_id": p.package_id,
                "position": [p.position[0], p.position[1]],
                "shelf_id": p.shelf_id,
                "size_class": p.size_class,
                "urgent": p.urgent,
                "mass_kg": p.mass_kg,
                "pose_name": p.pose_name,
            }
            for p in packages
        ],
    }


def _summarize(trials: Sequence[dict]) -> dict:
    total = len(trials)
    if total == 0:
        return {
            "total_trials": 0,
            "success_rate": 0.0,
            "timeout_rate": 0.0,
            "avg_planning_time_s": 0.0,
        }

    success = sum(int(t.get("end_to_end_success", 0)) for t in trials)
    timeout_like = sum(1 for t in trials if t.get("solved_within_timeout", 0) == 0)
    avg_plan = sum(float(t.get("planning_time_s", 0.0)) for t in trials) / total

    return {
        "total_trials": total,
        "success_rate": round(success / total, 4),
        "timeout_rate": round(timeout_like / total, 4),
        "avg_planning_time_s": round(avg_plan, 4),
    }


def _write_trials_csv(trials: Sequence[dict], output_csv: Path) -> None:
    if not trials:
        return

    headers = [
        "trial_id",
        "seed",
        "scenario_id",
        "layout",
        "num_packages",
        "planner_name",
        "symbolic_mode",
        "planning_time_s",
        "solved_within_timeout",
        "execution_success",
        "end_to_end_success",
        "symbolic_expanded_nodes",
        "motion_calls_total",
        "motion_calls_failed",
        "replan_count",
        "total_path_length_m",
        "collision_events",
        "status_reason",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for t in trials:
            writer.writerow({k: t.get(k) for k in headers})


def _select_scenarios(all_scenarios: Sequence[dict], scenario_filter: Optional[str]) -> List[dict]:
    if not scenario_filter:
        return list(all_scenarios)

    wanted = {x.strip() for x in scenario_filter.split(",") if x.strip()}
    return [s for s in all_scenarios if s.get("id") in wanted]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run traditional TAMP baseline trials for STP.")
    parser.add_argument(
        "--scenario-config",
        default="Experiment/tamp_baseline_scenarios_wh_table.json",
    )
    parser.add_argument("--layouts", default="Experiment/layouts.json")
    parser.add_argument("--output-dir", default="Experiment/baseline_results")
    parser.add_argument("--robot", default=None, help="Override robot profile in scenario config")
    parser.add_argument("--scenarios", default=None, help="Comma-separated scenario ids")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds override")
    parser.add_argument("--grid-resolution", type=float, default=0.1)
    parser.add_argument(
        "--max-symbolic-expansions",
        type=int,
        default=0,
        help="Maximum symbolic expanded nodes. Use 0 or negative for no limit.",
    )
    parser.add_argument(
        "--symbolic-mode",
        default="astar",
        choices=["astar", "greedy"],
        help="Symbolic sequence method: astar (PDDLStream-style) or greedy.",
    )
    parser.add_argument("--planner-name", default="traditional_tamp_astar")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    scenario_path = (repo_root / args.scenario_config).resolve()
    layouts_path = (repo_root / args.layouts).resolve()
    output_dir = (repo_root / args.output_dir).resolve()

    scenario_cfg = _load_json(scenario_path)
    all_scenarios = scenario_cfg.get("scenarios", [])
    selected_scenarios = _select_scenarios(all_scenarios, args.scenarios)
    if not selected_scenarios:
        raise RuntimeError("No scenarios selected. Check --scenarios argument.")

    if args.seeds:
        seeds = _parse_csv_ints(args.seeds)
    else:
        seeds = [int(x) for x in scenario_cfg.get("fixed_settings", {}).get("seeds", [])]
    if not seeds:
        raise RuntimeError("No seeds configured. Check scenario config or --seeds argument.")

    robot_name = args.robot or scenario_cfg.get("default_robot", "cyli_robo")
    robot_profile = scenario_cfg.get("robot_profiles", {}).get(robot_name)
    if not robot_profile:
        raise RuntimeError(f"Robot profile '{robot_name}' not found in scenario config.")

    effective_clearance = float(robot_profile.get("effective_clearance_m", 0.55))
    inflation_profile = robot_profile.get("inflation_m", {})
    planning_timeout_s = float(scenario_cfg.get("fixed_settings", {}).get("planning_timeout_sec", 60))
    start_pose_xy = tuple(float(v) for v in scenario_cfg.get("fixed_settings", {}).get("start_pose_xy", [0.0, 0.0]))
    max_symbolic_expansions: Optional[int] = args.max_symbolic_expansions if args.max_symbolic_expansions > 0 else None

    timestamp = _now_stamp()
    run_dir = output_dir / f"run_{timestamp}"
    pddl_dir = run_dir / "pddl_problems"
    run_dir.mkdir(parents=True, exist_ok=True)
    pddl_dir.mkdir(parents=True, exist_ok=True)

    trials: List[dict] = []

    for scenario in selected_scenarios:
        layout_name = str(scenario["layout"])
        layout = LayoutModel.from_layouts_file(layouts_path, layout_name)

        approach_offset = max(
            effective_clearance + 0.05,
            float(inflation_profile.get("unloaded", effective_clearance)),
        )

        planner = GridMotionPlanner(layout=layout, grid_resolution=args.grid_resolution)
        streams = TAMPStreams(
            planner=planner,
            inflation_profile={
                "unloaded": float(inflation_profile.get("unloaded", effective_clearance)),
                "small": float(inflation_profile.get("small", effective_clearance + 0.05)),
                "medium": float(inflation_profile.get("medium", effective_clearance + 0.12)),
                "large": float(inflation_profile.get("large", effective_clearance + 0.22)),
            },
            approach_offset_m=approach_offset,
        )

        for seed in seeds:
            trial = _run_trial(
                scenario=scenario,
                layout=layout,
                streams=streams,
                seed=seed,
                start_pose=start_pose_xy,
                planning_timeout_s=planning_timeout_s,
                max_symbolic_expansions=max_symbolic_expansions,
                symbolic_mode=args.symbolic_mode,
                planner_name=args.planner_name,
            )
            trials.append(trial)

            pddl_trial = {
                "trial_id": trial["trial_id"],
                "start_pose_name": "p_start",
                "drop_pose_name": "p_drop",
                "packages": [
                    {
                        "package_id": p["package_id"],
                        "pose_name": p["pose_name"],
                        "size_class": p["size_class"],
                        "urgent": p["urgent"],
                    }
                    for p in trial["packages"]
                ],
            }
            generate_problem_for_trial(pddl_trial, pddl_dir / f"{trial['trial_id']}.pddl")

            print(
                f"[trial] {trial['trial_id']} | success={trial['end_to_end_success']} "
                f"| plan_time={trial['planning_time_s']}s | expanded={trial['symbolic_expanded_nodes']}"
            )

    summary = _summarize(trials)
    result = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "scenario_config": str(scenario_path),
            "layouts_file": str(layouts_path),
            "robot_profile": robot_name,
            "planner_name": args.planner_name,
            "symbolic_mode": args.symbolic_mode,
            "grid_resolution": args.grid_resolution,
            "planning_timeout_sec": planning_timeout_s,
            "start_pose_xy": list(start_pose_xy),
            "max_symbolic_expansions": max_symbolic_expansions,
            "selected_scenarios": [s.get("id") for s in selected_scenarios],
            "seeds": seeds,
            "summary": summary,
        },
        "trials": trials,
    }

    json_path = run_dir / "baseline_trials.json"
    csv_path = run_dir / "baseline_trials.csv"

    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_trials_csv(trials, csv_path)

    print("\n[done] baseline run completed")
    print(f"- json: {json_path}")
    print(f"- csv:  {csv_path}")
    print(f"- pddl problems: {pddl_dir}")


if __name__ == "__main__":
    main()
