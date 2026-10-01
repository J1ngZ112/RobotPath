from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _round_f(v: float, ndigits: int = 3) -> float:
    return round(float(v), ndigits)


def _layout_signature_from_model(warehouse_geometry_model: dict) -> Tuple[Tuple[float, float, float, float, float], ...]:
    obstacles = warehouse_geometry_model.get("obstacles", []) if isinstance(warehouse_geometry_model, dict) else []
    rows: List[Tuple[float, float, float, float, float]] = []

    for obs in obstacles:
        if obs.get("type") != "Shelf":
            continue
        center = obs.get("center_point_m", [0.0, 0.0])
        dims = obs.get("dimensions_m", {})
        ori = obs.get("orientation_rad", 0.0)
        rows.append(
            (
                _round_f(center[0]),
                _round_f(center[1]),
                _round_f(dims.get("length", 0.0)),
                _round_f(dims.get("width", 0.0)),
                _round_f(ori, 4),
            )
        )

    return tuple(sorted(rows))


def _load_layout_signature_map(layouts_path: Path) -> Dict[Tuple[Tuple[float, float, float, float, float], ...], str]:
    if not layouts_path.exists():
        return {}

    raw = _load_json(layouts_path)
    out: Dict[Tuple[Tuple[float, float, float, float, float], ...], str] = {}

    for layout_name, layout_data in raw.get("layouts", {}).items():
        model = layout_data.get("warehouse_geometry_model", {})
        sig = _layout_signature_from_model(model)
        if sig:
            out[sig] = str(layout_name)
    return out


def _infer_layout_name_from_metadata(raw: dict, layout_signatures: Dict[Tuple[Tuple[float, float, float, float, float], ...], str]) -> str:
    metadata = raw.get("metadata", {}) if isinstance(raw, dict) else {}
    if not isinstance(metadata, dict):
        return "unknown"

    layout_blob = metadata.get("warehouse_layout", {})
    if not isinstance(layout_blob, dict):
        return "unknown"

    # 优先使用直接字段
    layout_name = layout_blob.get("layout_name")
    if layout_name:
        return str(layout_name)

    # 兼容当前数据结构: warehouse_layout.warehouse_geometry_model
    model = layout_blob.get("warehouse_geometry_model")
    if isinstance(model, dict):
        sig = _layout_signature_from_model(model)
        if sig in layout_signatures:
            return layout_signatures[sig]

    return "unknown"


def _path_length(points: Sequence[Sequence[float]]) -> float:
    if len(points) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(points)):
        total += math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
    return float(total)


def _to_int01(value) -> int:
    return 1 if bool(value) else 0


def _as_float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percentile(values: Sequence[float], p: float) -> Optional[float]:
    if not values:
        return None
    arr = sorted(values)
    idx = int(round((len(arr) - 1) * p))
    idx = max(0, min(len(arr) - 1, idx))
    return float(arr[idx])


def _normalize_baseline_trials(raw: dict, source_path: Path) -> List[dict]:
    if isinstance(raw, dict):
        trials = raw.get("trials", [])
    elif isinstance(raw, list):
        trials = raw
    else:
        trials = []

    normalized: List[dict] = []
    for idx, t in enumerate(trials):
        rec = {
            "id": str(t.get("trial_id", f"baseline_{idx:04d}")),
            "planner": str(t.get("planner_name", "traditional_tamp_astar")),
            "scenario_id": str(t.get("scenario_id", "unknown")),
            "num_packages": t.get("num_packages"),
            "planning_time_s": _as_float_or_none(t.get("planning_time_s")),
            "solved_within_timeout": _to_int01(t.get("solved_within_timeout", 0)),
            "execution_success": _to_int01(t.get("execution_success", 0)),
            "end_to_end_success": _to_int01(t.get("end_to_end_success", 0)),
            "collision_events": int(t.get("collision_events", 0) or 0),
            "total_path_length_m": float(t.get("total_path_length_m", 0.0) or 0.0),
            "source": source_path.name,
        }
        normalized.append(rec)
    return normalized


def _normalize_llm_trials(
    raw: dict,
    source_path: Path,
    planner_name: str,
    layout_signatures: Optional[Dict[Tuple[Tuple[float, float, float, float, float], ...], str]] = None,
) -> List[dict]:
    normalized: List[dict] = []

    # 1) 标准 trial 结构
    if isinstance(raw, dict) and "trials" in raw:
        trials = raw.get("trials", [])
        for idx, t in enumerate(trials):
            normalized.append(
                {
                    "id": str(t.get("trial_id", f"llm_trial_{idx:04d}")),
                    "planner": planner_name,
                    "scenario_id": str(t.get("scenario_id", "unknown")),
                    "num_packages": t.get("num_packages"),
                    "planning_time_s": _as_float_or_none(t.get("planning_time_s")),
                    "solved_within_timeout": _to_int01(t.get("solved_within_timeout", t.get("end_to_end_success", 0))),
                    "execution_success": _to_int01(t.get("execution_success", t.get("end_to_end_success", 0))),
                    "end_to_end_success": _to_int01(t.get("end_to_end_success", 0)),
                    "collision_events": int(t.get("collision_events", 0) or 0),
                    "total_path_length_m": float(t.get("total_path_length_m", 0.0) or 0.0),
                    "source": source_path.name,
                }
            )
        return normalized

    # 2) 现有 collision_dataset 结构
    if isinstance(raw, dict) and "sequences" in raw:
        sequences = raw.get("sequences", [])
        layout_name = _infer_layout_name_from_metadata(raw, layout_signatures or {})

        for seq in sequences:
            status = str(seq.get("navigation_status", "unknown"))
            success = int(status == "success")
            timeout = int(status == "timeout")
            solved = int(1 - timeout)
            waypoints = seq.get("waypoint_sequence", [])

            normalized.append(
                {
                    "id": str(seq.get("test_id", f"llm_seq_{len(normalized):04d}")),
                    "planner": planner_name,
                    "scenario_id": layout_name,
                    "num_packages": None,
                    "planning_time_s": _as_float_or_none(seq.get("planning_time_s")),
                    "solved_within_timeout": solved,
                    "execution_success": success,
                    "end_to_end_success": success,
                    "collision_events": int(seq.get("collision_detected", False)),
                    "total_path_length_m": round(_path_length(waypoints), 4),
                    "source": source_path.name,
                }
            )
        return normalized

    return normalized


def _summarize_by_planner(records: Sequence[dict]) -> List[dict]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        grouped[r["planner"]].append(r)

    rows: List[dict] = []
    for planner, items in grouped.items():
        n = len(items)
        success = sum(int(x["end_to_end_success"]) for x in items)
        timeout = sum(1 for x in items if int(x["solved_within_timeout"]) == 0)
        collision = sum(int(x["collision_events"]) for x in items)

        plan_times = [x["planning_time_s"] for x in items if x["planning_time_s"] is not None]
        path_lens = [float(x["total_path_length_m"]) for x in items]

        rows.append(
            {
                "planner": planner,
                "trial_count": n,
                "success_rate": round(success / n if n else 0.0, 4),
                "timeout_rate": round(timeout / n if n else 0.0, 4),
                "collision_rate": round(collision / n if n else 0.0, 4),
                "avg_planning_time_s": round(sum(plan_times) / len(plan_times), 4) if plan_times else None,
                "p95_planning_time_s": round(_percentile(plan_times, 0.95), 4) if plan_times else None,
                "avg_path_length_m": round(sum(path_lens) / len(path_lens), 4) if path_lens else 0.0,
                "planning_time_samples": len(plan_times),
            }
        )

    rows.sort(key=lambda x: x["planner"])
    return rows


def _summarize_by_scenario(records: Sequence[dict]) -> List[dict]:
    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in records:
        grouped[(r["planner"], r["scenario_id"])] .append(r)

    rows: List[dict] = []
    for (planner, scenario), items in sorted(grouped.items()):
        n = len(items)
        success = sum(int(x["end_to_end_success"]) for x in items)
        timeout = sum(1 for x in items if int(x["solved_within_timeout"]) == 0)
        rows.append(
            {
                "planner": planner,
                "scenario_id": scenario,
                "trial_count": n,
                "success_rate": round(success / n if n else 0.0, 4),
                "timeout_rate": round(timeout / n if n else 0.0, 4),
            }
        )
    return rows


def _write_csv(rows: Sequence[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown_report(summary_planner: Sequence[dict], summary_scenario: Sequence[dict], output_path: Path) -> None:
    lines = [
        "# Baseline vs LLM Comparison",
        "",
        f"Generated at: {datetime.now().isoformat()}",
        "",
        "## Planner Summary",
        "",
        "| Planner | Trials | Success Rate | Timeout Rate | Collision Rate | Avg Planning Time(s) | P95 Planning Time(s) | Avg Path Length(m) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in summary_planner:
        lines.append(
            "| {planner} | {trial_count} | {success_rate} | {timeout_rate} | {collision_rate} | {avg_planning_time_s} | {p95_planning_time_s} | {avg_path_length_m} |".format(
                planner=row.get("planner"),
                trial_count=row.get("trial_count"),
                success_rate=row.get("success_rate"),
                timeout_rate=row.get("timeout_rate"),
                collision_rate=row.get("collision_rate"),
                avg_planning_time_s=row.get("avg_planning_time_s"),
                p95_planning_time_s=row.get("p95_planning_time_s"),
                avg_path_length_m=row.get("avg_path_length_m"),
            )
        )

    lines.extend(
        [
            "",
            "## Scenario Summary",
            "",
            "| Planner | Scenario | Trials | Success Rate | Timeout Rate |",
            "|---|---|---:|---:|---:|",
        ]
    )

    for row in summary_scenario:
        lines.append(
            "| {planner} | {scenario_id} | {trial_count} | {success_rate} | {timeout_rate} |".format(
                planner=row.get("planner"),
                scenario_id=row.get("scenario_id"),
                trial_count=row.get("trial_count"),
                success_rate=row.get("success_rate"),
                timeout_rate=row.get("timeout_rate"),
            )
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def _try_make_plots(records: Sequence[dict], output_dir: Path) -> List[str]:
    messages: List[str] = []
    try:
        import matplotlib.pyplot as plt
    except Exception:
        messages.append("matplotlib not installed, skipped png plot generation")
        return messages

    planners = sorted({r["planner"] for r in records})

    # success rate bar
    success_rates = []
    for planner in planners:
        rows = [r for r in records if r["planner"] == planner]
        success = sum(r["end_to_end_success"] for r in rows)
        success_rates.append(success / len(rows) if rows else 0.0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(planners, success_rates)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Success Rate")
    ax.set_title("End-to-End Success Rate by Planner")
    fig.tight_layout()
    fig.savefig(output_dir / "success_rate_by_planner.png", dpi=150)
    plt.close(fig)

    # planning time vs packages
    time_rows = [r for r in records if r["planning_time_s"] is not None and r.get("num_packages") is not None]
    if time_rows:
        grouped: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
        for r in time_rows:
            grouped[r["planner"]][int(r["num_packages"])].append(float(r["planning_time_s"]))

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for planner in sorted(grouped.keys()):
            xs = sorted(grouped[planner].keys())
            ys = [sum(grouped[planner][x]) / len(grouped[planner][x]) for x in xs]
            ax.plot(xs, ys, marker="o", label=planner)

        ax.set_xlabel("Number of Packages")
        ax.set_ylabel("Planning Time (s)")
        ax.set_title("Planning Time vs Task Length")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "planning_time_vs_packages.png", dpi=150)
        plt.close(fig)

    return messages


def _find_latest_collision_dataset(repo_root: Path) -> Optional[Path]:
    for relative_dir in (("data", "collision_dataset"), ("Experiment", "collision_dataset")):
        dataset_dir = repo_root.joinpath(*relative_dir)
        files = sorted(dataset_dir.glob("collision_sequences_final_*.json"))
        if files:
            return files[-1]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare baseline results against LLM-side results.")
    parser.add_argument("--baseline", required=True, help="Path to baseline_trials.json")
    parser.add_argument("--llm", default=None, help="Path to LLM result json (optional)")
    parser.add_argument("--llm-planner-name", default="llm_system")
    parser.add_argument("--skip-llm", action="store_true", help="Skip loading any LLM-side results")
    parser.add_argument("--output-dir", default="Experiment/baseline_results")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    baseline_path = (repo_root / args.baseline).resolve()
    output_root = (repo_root / args.output_dir).resolve()
    layouts_path = repo_root / "Experiment" / "layouts.json"
    layout_signatures = _load_layout_signature_map(layouts_path)

    llm_path: Optional[Path] = None
    if not args.skip_llm:
        if args.llm:
            llm_path = (repo_root / args.llm).resolve()
        else:
            llm_path = _find_latest_collision_dataset(repo_root)

    run_dir = output_root / f"compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    baseline_raw = _load_json(baseline_path)
    records = _normalize_baseline_trials(baseline_raw, baseline_path)

    if llm_path and llm_path.exists():
        llm_raw = _load_json(llm_path)
        records.extend(
            _normalize_llm_trials(
                llm_raw,
                llm_path,
                planner_name=args.llm_planner_name,
                layout_signatures=layout_signatures,
            )
        )

    if not records:
        raise RuntimeError("No comparable records were found in provided files.")

    summary_planner = _summarize_by_planner(records)
    summary_scenario = _summarize_by_scenario(records)

    combined_path = run_dir / "combined_trials.json"
    summary_planner_csv = run_dir / "summary_by_planner.csv"
    summary_scenario_csv = run_dir / "summary_by_scenario.csv"
    report_md = run_dir / "comparison_report.md"

    combined_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(records, run_dir / "combined_trials.csv")
    _write_csv(summary_planner, summary_planner_csv)
    _write_csv(summary_scenario, summary_scenario_csv)
    _write_markdown_report(summary_planner, summary_scenario, report_md)

    plot_msgs = _try_make_plots(records, run_dir)

    print("[done] comparison completed")
    print(f"- combined records: {combined_path}")
    print(f"- summary planner:  {summary_planner_csv}")
    print(f"- summary scenario: {summary_scenario_csv}")
    print(f"- markdown report:  {report_md}")
    if args.skip_llm:
        print("- llm source:       skipped (--skip-llm)")
    elif llm_path and llm_path.exists():
        print(f"- llm source:       {llm_path}")
    for m in plot_msgs:
        print(f"- note: {m}")


if __name__ == "__main__":
    main()
