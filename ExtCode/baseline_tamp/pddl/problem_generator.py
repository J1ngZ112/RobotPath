from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


DOMAIN_NAME = "stp_traditional_tamp"


@dataclass(frozen=True)
class PackageSpec:
    package_id: str
    pose_name: str
    size_class: str
    urgent: bool


def _sanitize_symbol(raw: str) -> str:
    s = raw.strip().replace(" ", "_").replace("-", "_")
    if not s:
        s = "x"
    if s[0].isdigit():
        s = f"n_{s}"
    return "".join(ch for ch in s if ch.isalnum() or ch == "_")


def _size_fact(package_name: str, size_class: str) -> str:
    norm = size_class.lower()
    if norm == "large":
        return f"(SizeLarge {package_name})"
    if norm == "medium":
        return f"(SizeMedium {package_name})"
    return f"(SizeSmall {package_name})"


def build_problem_text(
    problem_name: str,
    start_pose_name: str,
    drop_pose_name: str,
    packages: Iterable[PackageSpec],
) -> str:
    pkgs = list(packages)

    pose_names: List[str] = [start_pose_name, drop_pose_name]
    grasp_names: List[str] = []
    pkg_names: List[str] = []
    init_facts: List[str] = [
        f"(AtRobot {start_pose_name})",
        "(HandEmpty)",
        "(CarriedSize sz_small)",
    ]

    has_urgent = any(p.urgent for p in pkgs)
    if has_urgent:
        init_facts.append("(UrgentPhaseOpen)")

    goal_facts: List[str] = []

    for pkg in pkgs:
        pkg_name = _sanitize_symbol(f"pkg_{pkg.package_id}")
        pose_name = _sanitize_symbol(pkg.pose_name)
        grasp_name = _sanitize_symbol(f"g_{pkg.package_id}")

        pkg_names.append(pkg_name)
        pose_names.append(pose_name)
        grasp_names.append(grasp_name)

        init_facts.append(f"(AtPkg {pkg_name} {pose_name})")
        init_facts.append(f"(CanPick {pkg_name} {pose_name} {grasp_name})")
        init_facts.append(f"(CanPlace {pkg_name} {drop_pose_name})")
        init_facts.append(_size_fact(pkg_name, pkg.size_class))

        if pkg.urgent:
            init_facts.append(f"(Urgent {pkg_name})")
        else:
            init_facts.append(f"(NonUrgent {pkg_name})")

        goal_facts.append(f"(Picked {pkg_name})")

    pose_obj = " ".join(sorted(set(pose_names)))
    pkg_obj = " ".join(sorted(set(pkg_names)))
    grasp_obj = " ".join(sorted(set(grasp_names)))

    lines = [
        f"(define (problem {_sanitize_symbol(problem_name)})",
        f"  (:domain {DOMAIN_NAME})",
        "  (:objects",
        f"    {pose_obj} - pose",
        f"    {pkg_obj} - package",
        f"    {grasp_obj} - grasp",
        "    tr_dummy - traj",
        "    sz_small sz_medium sz_large - size",
        "  )",
        "  (:init",
    ]

    for fact in init_facts:
        lines.append(f"    {fact}")

    lines.extend(
        [
            "  )",
            "  (:goal (and",
        ]
    )

    for fact in goal_facts:
        lines.append(f"    {fact}")

    lines.extend(
        [
            "  ))",
            ")",
        ]
    )

    return "\n".join(lines)


def generate_problem_for_trial(trial: dict, output_path: Path) -> Path:
    packages = trial.get("packages", [])
    package_specs: List[PackageSpec] = []
    for idx, pkg in enumerate(packages):
        package_specs.append(
            PackageSpec(
                package_id=str(pkg.get("package_id", idx)),
                pose_name=str(pkg.get("pose_name", f"p_pkg_{idx:02d}")),
                size_class=str(pkg.get("size_class", "small")),
                urgent=bool(pkg.get("urgent", False)),
            )
        )

    problem_name = str(trial.get("trial_id", "trial"))
    start_pose = _sanitize_symbol(str(trial.get("start_pose_name", "p_start")))
    drop_pose = _sanitize_symbol(str(trial.get("drop_pose_name", "p_drop")))

    text = build_problem_text(problem_name, start_pose, drop_pose, package_specs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def _load_trial(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "trial" in raw:
        return raw["trial"]
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PDDL problem file for one trial instance.")
    parser.add_argument("--trial-json", required=True, help="Path to a trial JSON file.")
    parser.add_argument("--output", required=True, help="Output path of generated problem.pddl.")
    args = parser.parse_args()

    trial = _load_trial(Path(args.trial_json))
    output = generate_problem_for_trial(trial, Path(args.output))
    print(f"Generated PDDL problem: {output}")


if __name__ == "__main__":
    main()
