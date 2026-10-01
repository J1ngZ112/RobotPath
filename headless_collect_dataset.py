#!/usr/bin/env python3
"""Headless Isaac Sim collision dataset collection entrypoint."""

import argparse
import os
import sys

from isaacsim import SimulationApp


def parse_args():
    parser = argparse.ArgumentParser(description="Collect STP collision data in Isaac Sim headless mode.")
    parser.add_argument("--layout", default=None, help="Layout name from Experiment/layouts.json. Defaults to the first layout.")
    parser.add_argument("--mode", choices=["astar", "file"], default="astar", help="Sequence source.")
    parser.add_argument("--num-sequences", type=int, default=300, help="Number of A* sequences to generate.")
    parser.add_argument(
        "--collision-probability",
        type=float,
        default=0.7,
        help="Independent probability that each generated sequence contains an injected static collision scenario.",
    )
    parser.add_argument("--lateral-offset-weight", type=float, default=0.30)
    parser.add_argument("--corner-cutting-weight", type=float, default=0.30)
    parser.add_argument("--docking-overtravel-weight", type=float, default=0.25)
    parser.add_argument("--wall-drift-weight", type=float, default=0.15)
    parser.add_argument(
        "--hard-safe-probability",
        type=float,
        default=0.5,
        help="Probability that a safe draw becomes a footprint-safe near-miss route.",
    )
    parser.add_argument("--collision-penetration-min", type=float, default=0.06)
    parser.add_argument("--collision-penetration-max", type=float, default=0.22)
    parser.add_argument("--fault-transition-waypoints-min", type=int, default=1)
    parser.add_argument("--fault-transition-waypoints-max", type=int, default=3)
    parser.add_argument("--hard-safe-margin-min", type=float, default=0.05)
    parser.add_argument("--hard-safe-margin-max", type=float, default=0.18)
    parser.add_argument("--collision-index-min", type=int, default=4)
    parser.add_argument("--collision-index-max", type=int, default=27)
    parser.add_argument("--waypoint-count-min", type=int, default=15)
    parser.add_argument("--waypoint-count-max", type=int, default=30)
    parser.add_argument("--waypoint-file", default=None, help="Waypoint file for --mode file. Defaults to Experiment/SimWaypoint.txt.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to data/collision_dataset.")
    parser.add_argument("--save-interval", type=int, default=5, help="Partial save interval.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Sequence-generation seed. Omit to generate a new random seed per collection.",
    )
    parser.add_argument("--max-linear-speed", type=float, default=0.8, help="Base linear speed limit in m/s.")
    parser.add_argument("--max-linear-acceleration", type=float, default=1.0, help="Base linear acceleration limit in m/s^2.")
    parser.add_argument("--max-angular-speed", type=float, default=1.2, help="Base angular speed limit in rad/s.")
    parser.add_argument("--max-angular-acceleration", type=float, default=2.0, help="Base angular acceleration limit in rad/s^2.")
    parser.add_argument(
        "--trajectory-sample-interval",
        type=float,
        default=0.5,
        help="Executed-trajectory sampling interval in simulation seconds.",
    )
    parser.add_argument("--collision-force-threshold", type=float, default=0.01)
    parser.add_argument("--strong-collision-impulse-threshold", type=float, default=0.5)
    parser.add_argument("--collision-confirmation-events", type=int, default=2)
    parser.add_argument("--collision-confirmation-window", type=float, default=0.15)
    parser.add_argument("--max-recorded-contact-events", type=int, default=20)
    parser.add_argument("--reset-x", type=float, default=0.0)
    parser.add_argument("--reset-y", type=float, default=0.0)
    parser.add_argument("--reset-yaw", type=float, default=0.0)
    parser.add_argument("--max-startup-frames", type=int, default=120, help="Frames to warm up after scene load.")
    parser.add_argument("--width", type=int, default=640, help="Headless app width.")
    parser.add_argument("--height", type=int, default=480, help="Headless app height.")
    parser.add_argument("--renderer", default="RayTracedLighting", help="Isaac Sim renderer setting.")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Keep timeline synchronized to wall time instead of running offline as fast as possible.",
    )
    return parser.parse_args()


async def run_collection(args, repo_root):
    # Imports must happen after SimulationApp starts.
    import carb.settings
    import omni.kit.app
    import omni.timeline

    ext_root = repo_root
    if ext_root not in sys.path:
        sys.path.insert(0, ext_root)

    from ExtCode.SceneManager import SceneManager
    from ExtCode.Controller import RobotController
    from ExtCode.CollisionData import CollisionDataCollector

    scene_manager = SceneManager("data/Assets/Warehouse.usd")
    if args.layout:
        if not scene_manager.load_layout_config(args.layout):
            raise RuntimeError(f"Layout not found or failed to load: {args.layout}")

    if not scene_manager.load_scene():
        raise RuntimeError("Failed to load scene.")

    for _ in range(max(0, args.max_startup_frames)):
        await omni.kit.app.get_app().next_update_async()

    # SingleArticulation.initialize() requires the physics simulation view.
    # Playing the timeline creates that backend on the next application update.
    timeline = omni.timeline.get_timeline_interface()
    settings = carb.settings.get_settings()
    settings.set("/app/player/useFixedTimeStepping", True)
    timeline.set_play_every_frame(not args.realtime)
    settings.set("/app/runLoops/main/rateLimitEnabled", args.realtime)
    timeline.play()
    timeline.commit()
    settings.set("/app/runLoops/main/rateLimitEnabled", args.realtime)
    for _ in range(3):
        await omni.kit.app.get_app().next_update_async()

    robot_id, robot_cfg = scene_manager.get_robot_config()
    shape_info = robot_cfg.get("shape_info", {})
    robot_controller = RobotController(
        robot_prim_path=robot_cfg.get("prim_path", "/World/RidgebackFranka"),
        sensor_path=robot_cfg.get("sensor_path"),
        robot_width=float(shape_info.get("width", 0.8)),
        robot_length=float(shape_info.get("length", 1.0)),
        max_linear_speed=args.max_linear_speed,
        max_linear_acceleration=args.max_linear_acceleration,
        max_angular_speed=args.max_angular_speed,
        max_angular_acceleration=args.max_angular_acceleration,
        trajectory_sample_interval=args.trajectory_sample_interval,
        collision_force_threshold=args.collision_force_threshold,
        collision_confirmation_events=args.collision_confirmation_events,
        collision_confirmation_window=args.collision_confirmation_window,
        strong_collision_impulse_threshold=args.strong_collision_impulse_threshold,
        max_recorded_contact_events=args.max_recorded_contact_events,
    )
    if not robot_controller.initialize():
        raise RuntimeError(f"Failed to initialize robot controller for {robot_id}.")

    output_dir = args.output_dir or os.path.join(ext_root, "data", "collision_dataset")
    collector = CollisionDataCollector(
        robot_controller,
        output_dir=output_dir,
        reset_position=(args.reset_x, args.reset_y, args.reset_yaw),
        layout_config=scene_manager.current_layout_data,
        robot_config=robot_cfg,
        random_seed=args.seed,
        collection_runtime_config={
            "headless": True,
            "offline_fast_mode": not args.realtime,
            "fixed_time_stepping": True,
            "viewport_updates_disabled": not args.realtime,
            "renderer": args.renderer,
        },
    )

    if args.mode == "file":
        waypoint_file = args.waypoint_file or os.path.join(ext_root, "Experiment", "SimWaypoint.txt")
        sequences = collector.load_waypoint_sequences_from_file(waypoint_file)
    else:
        sequences = collector.generate_industrial_picking_sequences(
            {
                "num_sequences": args.num_sequences,
                "collision_probability": args.collision_probability,
                "lateral_localization_offset_weight": args.lateral_offset_weight,
                "shelf_corner_cutting_weight": args.corner_cutting_weight,
                "docking_overtravel_weight": args.docking_overtravel_weight,
                "wall_localization_drift_weight": args.wall_drift_weight,
                "hard_safe_probability_given_safe": args.hard_safe_probability,
                "collision_penetration_min_m": args.collision_penetration_min,
                "collision_penetration_max_m": args.collision_penetration_max,
                "fault_transition_waypoints_min": args.fault_transition_waypoints_min,
                "fault_transition_waypoints_max": args.fault_transition_waypoints_max,
                "hard_safe_margin_min_m": args.hard_safe_margin_min,
                "hard_safe_margin_max_m": args.hard_safe_margin_max,
                "collision_index_min": args.collision_index_min,
                "collision_index_max": args.collision_index_max,
                "waypoint_count_min": args.waypoint_count_min,
                "waypoint_count_max": args.waypoint_count_max,
            }
        )

    if not sequences:
        raise RuntimeError("No waypoint sequences available for collection.")

    try:
        await collector.collect_dataset(
            sequences,
            robot_id=robot_id,
            save_interval=args.save_interval,
        )
    finally:
        robot_controller.stop_navigation()
        timeline.stop()


def main():
    args = parse_args()
    repo_root = os.path.dirname(os.path.abspath(__file__))

    sim_app = SimulationApp(
        {
            "headless": True,
            "disable_viewport_updates": not args.realtime,
            "width": args.width,
            "height": args.height,
            "renderer": args.renderer,
        }
    )

    try:
        from omni.kit.async_engine import run_coroutine

        collection_task = run_coroutine(run_collection(args, repo_root))
        while not collection_task.done():
            sim_app.update()

        # Propagate collection failures to the process exit status.
        collection_task.result()
    finally:
        sim_app.close()


if __name__ == "__main__":
    main()
