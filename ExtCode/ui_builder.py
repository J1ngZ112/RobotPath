"""
Omniverse UI Framework:
  https://docs.omniverse.nvidia.com/kit/docs/omni.ui/latest/Overview.html

Isaac Sim UI Utilities extension:
  https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.ui/docs/index.html
"""
import os
from PIL import Image
import asyncio
import json
from pathlib import Path
import omni.kit.app
import omni.timeline
import omni.kit.ui
import omni.ui as ui
from pxr import PhysxSchema, UsdPhysics
import isaacsim.gui.components as ui_utils
from isaacsim.sensors.camera import Camera
from isaacsim.sensors.physics import _sensor, ContactSensor
from .SceneManager import SceneManager
from .Controller import RobotController
from .LLMPlanner import LM, Scene_Prompt, get_plan_prompt, replan_from_failure
from .Prediector import RemoteCollisionPredictor
from .Utils import *
from .CollisionData import CollisionDataCollector
from .baseline_heuristic.integration import (
    build_runtime_plan as build_heuristic_runtime_plan,
)
from .baseline_rl.integration import (
    build_runtime_plan as build_rl_runtime_plan,
)
from .baseline_experiment_io import (
    load_model_output,
    resolve_waypoint_path,
    write_experiment_input,
    write_simulation_result,
)

EXTENSION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILE_PATH = os.path.join(EXTENSION_ROOT, "ExtCode", "ui_builder.py")
SCENE_PATH = "data/Assets/Warehouse.usd"
EXPERIMENT_ROOT = os.path.join(EXTENSION_ROOT, "Experiment")
HEURISTIC_WAYPOINT_ROOT = os.path.join(EXPERIMENT_ROOT, "heuristic_waypoint")
RL_WAYPOINT_ROOT = os.path.join(EXPERIMENT_ROOT, "RL_waypoint")
PICK_WAYPOINT_PATH = os.path.join(EXPERIMENT_ROOT, "Pick_WayPoint.txt")
REPAIRED_WAYPOINT_PATH = os.path.join(EXPERIMENT_ROOT, "repaired_waypoint_plan.json")
SIMULATION_RESULT_PATH = os.path.join(EXPERIMENT_ROOT, "simulation_validation_result.json")
IMAGE_SAVE_PATH = EXPERIMENT_ROOT
WAREHOUSE_MODEL_PATH = os.path.join(EXPERIMENT_ROOT, "Warehouse.txt")
CHECK_REPORT_PATH = os.path.join(EXPERIMENT_ROOT, "check_report.json")


class UIBuilder:
    """Manage extension UI"""

    def __init__(self, window_title, menu_path=None):
        self._menu = None
        self._window = None
        self._menu_path = menu_path
        self._window_title = window_title
        
        self._scene_manager = SceneManager(SCENE_PATH)
        self._active_robot_id = "ridgeback_franka"
        self._robot_controller = RobotController()
        self._robot_controller.set_failure_callback(self.on_navigation_failed)

        self._target_x_model = ui.SimpleFloatModel(5.0)
        self._target_y_model = ui.SimpleFloatModel(5.0)
        self._navigation_path_model = ui.SimpleStringModel(
            "Experiment/repaired_waypoint_plan.json"
        )
        
        # 布局选择相关
        self._layout_names = []
        self._current_layout_name = ""
        self._load_available_layouts()
        
        # 保存 UI 引用以防止垃圾回收
        self._layout_combo = None
        self._combo_subscription = None

        # create menu
        if self._menu_path:
            self._menu = omni.kit.ui.get_editor_menu().add_item(self._menu_path, self.on_toggle, toggle=True, value=False)

        # 添加数据收集器
        self._collision_collector = None

        # 远程预测器客户端
        self._predictor = RemoteCollisionPredictor()
        self._check_in_progress = False
        self._baseline_in_progress = False
        self._validation_context = None

        # 使用当前布局中的机器人配置重建控制器
        self._rebuild_robot_controller_from_layout()

    def _get_active_robot_context(self):
        """Collect the currently active robot's static spec and runtime pose."""
        robot_id, robot_cfg = self._scene_manager.get_robot_config(self._active_robot_id)
        prim_path = robot_cfg.get("prim_path", "/World/RidgebackFranka")
        shape_info = robot_cfg.get("shape_info", {})

        return {
            "robot_id": robot_id,
            "robot_cfg": robot_cfg,
            "robot_size": {
                "width": float(shape_info.get("width", 0.8)),
                "length": float(shape_info.get("length", 0.8)),
                "polygon": shape_info.get("polygon"),
                "bounding_circle": shape_info.get("bounding_circle"),
                "robot_id": robot_id,
            },
            "robot_info": get_robot_info(prim_path),
        }

    def _rebuild_robot_controller_from_layout(self):
        """根据当前布局配置创建对应机器人控制器。"""
        robot_id, robot_cfg = self._scene_manager.get_robot_config()
        prim_path = robot_cfg.get("prim_path", "/World/RidgebackFranka")
        sensor_path = robot_cfg.get("sensor_path", f"{prim_path}/base_link/Contact_Sensor")
        shape_info = robot_cfg.get("shape_info", {})
        width = float(shape_info.get("width", 0.8))
        length = float(shape_info.get("length", 1.0))

        self._robot_controller = RobotController(
            robot_prim_path=prim_path,
            sensor_path=sensor_path,
            robot_width=width,
            robot_length=length,
        )
        self._robot_controller.set_failure_callback(self.on_navigation_failed)
        self._active_robot_id = robot_id
        print(f"Robot controller set to '{robot_id}' ({prim_path})")

    def _load_available_layouts(self):
        """从 layouts.json 加载布局名称"""
        try:
            layouts_file = os.path.join(EXPERIMENT_ROOT, "layouts.json")
            if os.path.exists(layouts_file):
                with open(layouts_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._layout_names = list(data.get("layouts", {}).keys())
            
            if not self._layout_names:
                self._layout_names = ["Default"]
            
            # 默认选中第一个
            if self._layout_names:
                self._current_layout_name = self._layout_names[0]
                
        except Exception as e:
            print(f"Error loading layouts: {e}")
            self._layout_names = ["Default"]
            self._current_layout_name = "Default"

    def on_toggle(self, *args, **kwargs):
        """Toggle window visibility"""
        self.build_ui()
        if self._window is not None:
            self._window.visible = not self._window.visible

    def build_ui(self):
        """Build the Graphical User Interface (GUI) in the underlying windowing system"""
        if not self._window:
            self._window = ui.Window(title=self._window_title, visible=False)
            with self._window.frame:
                with ui.VStack(spacing=6):
                    header = ui_utils.setup_ui_headers(ext_id="MyExtension", file_path=FILE_PATH , title="Spatial Trajectory Planning", overview="My Extension for Spatial Trajectory Planning.")
                    
                    with ui.CollapsableFrame("World Controls", height=120):
                        with ui.VStack(spacing=6):
                            # 添加布局选择器
                            with ui.HStack(height=22):
                                ui.Label("Layout:", width=60, height=22)
                                
                                # 1. 获取当前索引 (如果找不到则默认为0)
                                current_idx = self._layout_names.index(self._current_layout_name) if self._current_layout_name in self._layout_names else 0
                                
                                # 2. 创建下拉框并保存引用
                                self._layout_combo = ui.ComboBox(current_idx, *self._layout_names, height=22)
                                
                                # 3. 定义回调：直接通过索引获取名称
                                def on_layout_changed(model, item):
                                    idx = model.get_item_value_model(item, 0).as_int
                                    if 0 <= idx < len(self._layout_names):
                                        self._current_layout_name = self._layout_names[idx]
                                        print(f"Layout selected: {self._current_layout_name}")
                                
                                # 4. 绑定回调并保存订阅对象
                                self._combo_subscription = self._layout_combo.model.subscribe_item_changed_fn(on_layout_changed)

                            load_btn = ui_utils.btn_builder(label="Load Scene", text="LOAD", on_clicked_fn=self.load_scene)
                            reset_btn = ui_utils.btn_builder(label="Reset Scene", text="RESET", on_clicked_fn=self.reset_scene)
                            add_btn = ui_utils.btn_builder(label="Add Package", text="Add Package", on_clicked_fn=self.add_package)
                            ui_utils.btn_builder(label="Env Awareness", text="AWARE", on_clicked_fn=self.aware_env_clicked)
                            test_btn = ui_utils.btn_builder(label="Test", text="Test", on_clicked_fn=self.test_button_clicked)

                    with ui.CollapsableFrame("Robot Controls", height=170):
                        with ui.VStack(spacing=6):
                            # with ui.VStack(spacing=4):
                            #     ui.Label("Target Position", height=30)
                            #     with ui.HStack(spacing=50):
                            #         with ui.HStack(spacing=4):
                            #             ui.Label("X:", width=20)
                            #             ui.FloatDrag(model=self._target_x_model, min=-12.0, max=12.0, step=0.1, width=100)
                            #         with ui.HStack(spacing=4):
                            #             ui.Label("Y:", width=20)
                            #             ui.FloatDrag(model=self._target_y_model, min=-12.0, max=12.0, step=0.1, width=100)

                                ui_utils.btn_builder(label="Generate Plan", text="PLAN", on_clicked_fn=self.generate_plan_clicked)
                                ui_utils.btn_builder(label="Check Plan", text="CHECK", on_clicked_fn=self.check_plan_clicked)
                                with ui.HStack(height=24, spacing=6):
                                    ui.Label("Waypoint JSON:", width=100, height=24)
                                    ui.StringField(
                                        model=self._navigation_path_model,
                                        height=24,
                                    )
                                ui_utils.btn_builder(label="Start Navigation", text="START", on_clicked_fn=self.start_navigation_clicked)
                                ui_utils.btn_builder(label="Stop Navigation", text="STOP", on_clicked_fn=self.stop_navigation_clicked)
                                ui_utils.btn_builder(label="Heuristic Baseline", text="A* BASELINE", on_clicked_fn=self.run_heuristic_baseline_clicked)
                                ui_utils.btn_builder(label="RL Baseline", text="RL BASELINE", on_clicked_fn=self.run_rl_baseline_clicked)

                    # 添加新的数据收集面板
                    with ui.CollapsableFrame("Dataset Collection", height=120):
                        with ui.VStack(spacing=6):
                            ui_utils.btn_builder(
                                label="Collect File Dataset", 
                                text="COLLECT FILE", 
                                on_clicked_fn=self.collect_file_dataset_clicked
                            )
                            ui_utils.btn_builder(
                                label="Collect A Dataset", 
                                text="COLLECT A", 
                                on_clicked_fn=self.collect_a_dataset_clicked
                            )
                            ui_utils.btn_builder(
                                label="Stop Collection", 
                                text="STOP", 
                                on_clicked_fn=self.stop_collection_clicked
                            )
    
    def load_scene(self):
        """Load the scene using scene manager"""
        self._validation_context = None
        # 直接使用当前名称，无需索引转换
        print(f"Loading layout: {self._current_layout_name}")
        self._scene_manager.load_layout_config(self._current_layout_name)
        self._scene_manager.load_scene()
        self._rebuild_robot_controller_from_layout()

    def reset_scene(self):
        """Reset the scene using scene manager"""
        self._validation_context = None
        # 直接使用当前名称
        self._scene_manager.load_layout_config(self._current_layout_name)
        self._scene_manager.reset_scene()
        self._rebuild_robot_controller_from_layout()

    def add_package(self):
        """Add package to scene using scene manager"""
        self._scene_manager.add_packages(count=10)

    def move_robot(self):
        """Obtain the target and command the robot to move"""
        target_x = self._target_x_model.get_value_as_float()
        target_y = self._target_y_model.get_value_as_float()
        self._robot_controller.move_to_target(target_x, target_y)

    def aware_env_clicked(self):
        """Captures an image, sends it to the LLM for analysis, and saves the result."""
        async def aware_env_async():
            print("Starting warehouse model generation...")
            
            if not self._scene_manager.overview_camera:
                print("Error: Overview camera not found. Please load the scene first.")
                return

            # 1. 捕获并保存图像
            image_path = os.path.join(EXPERIMENT_ROOT, "Overview_Capture.png")
            capture_result = await capture_image(self._scene_manager.overview_camera, output_path=image_path)
            
            if not capture_result or not capture_result.get('success'):
                error_msg = capture_result.get('error', 'Unknown error') if capture_result else 'Unknown error'
                print(f"Image capture failed. Error: {error_msg}")
                return
            
            # 2. 调用多模态模型进行分析
            try:
                print("Analyzing warehouse layout with LLM...")
                # 使用 Scene_Prompt 作为系统提示初始化 LLM
                llm = LM(sys_prompt=Scene_Prompt)
                user_prompt = "Analyze the warehouse layout from the provided image."
                model_response = llm.reasoning(user_prompt, image_path=image_path, json_only=True)
                
                # 4. 保存结果到文件
                output_file_path = os.path.join(EXPERIMENT_ROOT, "Warehouse.txt")
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    # 确保响应是字符串格式
                    if isinstance(model_response, dict):
                        json.dump(model_response, f, indent=2)
                    else:
                        f.write(str(model_response))

                print(f"LLM analysis complete. Environment Model saved to: {output_file_path}")

            except Exception as e:
                print(f"An error occurred during LLM analysis: {e}")

        # 在UI事件循环中安全地执行异步任务
        asyncio.ensure_future(aware_env_async())


    def generate_plan_clicked(self):
        """
        Generates a plan using the LLM based on the current environment and task,
        and saves it to a file.
        """
        async def generate_plan_async():
            print("Starting LLM planning sequence...")
            # 1. 加载环境模型
            warehouse_model_file_path = os.path.join(EXPERIMENT_ROOT, "Warehouse.txt")
            try:
                with open(warehouse_model_file_path, 'r', encoding='utf-8') as f:
                    warehouse_geometry_model = f.read()
                print("Successfully loaded warehouse geometry model.")
            except FileNotFoundError:
                print(f"Error: Warehouse model file not found at {warehouse_model_file_path}")
                print("Please run 'Env Awareness' first to generate the model.")
                return
            except Exception as e:
                print(f"Error reading warehouse model file: {e}")
                return

            # 2. 获取动态信息
            packages_info = get_packages_info()
            robot_info = self._get_active_robot_context().get("robot_info")
            print("Successfully retrieved current package and robot info.")

            # 3. 生成规划提示并调用 LLM
            try:
                print("Generating plan with LLM...")
                plan_prompt = get_plan_prompt(warehouse_geometry_model, packages_info, robot_info, mission_package_ids=None)
                llm = LM()
                model_response = llm.reasoning(plan_prompt, image_path=None, json_only=True)

                if not isinstance(model_response, dict):
                    print("LLM planning failed. Response was not a valid JSON object.")
                    print("LLM Response:", model_response)
                    return
                
                # 4. 保存规划结果
                plan_output_path = os.path.join(EXPERIMENT_ROOT, "Pick_WayPoint.txt")
                with open(plan_output_path, 'w', encoding='utf-8') as f:
                    json.dump(model_response, f, indent=2)
                
                print(f"LLM planning successful. Plan saved to: {plan_output_path}")

            except Exception as e:
                print(f"An error occurred during LLM planning: {e}")

        asyncio.ensure_future(generate_plan_async())

    def check_plan_clicked(self):
        """Sends the current plan to the remote predictor and saves a check report."""
        async def check_plan_async():
            if self._check_in_progress:
                print("CHECK request is already running. Please wait for the current prediction to finish.")
                return

            self._check_in_progress = True
            print("Starting remote collision check...")
            try:
                robot_context = self._get_active_robot_context()
                robot_info = robot_context.get("robot_info")
                robot_size = robot_context.get("robot_size")

                if not os.path.exists(PICK_WAYPOINT_PATH):
                    report = {
                        "error": f"Plan file not found: {PICK_WAYPOINT_PATH}",
                        "predictor_url": self._predictor.predictor_url,
                    }
                    self._predictor.save_report(report, CHECK_REPORT_PATH)
                    print(report["error"])
                    print(f"Report saved to: {CHECK_REPORT_PATH}")
                    return

                if not os.path.exists(WAREHOUSE_MODEL_PATH):
                    report = {
                        "error": f"Warehouse model not found: {WAREHOUSE_MODEL_PATH}",
                        "predictor_url": self._predictor.predictor_url,
                    }
                    self._predictor.save_report(report, CHECK_REPORT_PATH)
                    print(report["error"])
                    print(f"Report saved to: {CHECK_REPORT_PATH}")
                    return

                if not robot_info.get("success", True):
                    print(f"Warning: failed to read runtime robot pose: {robot_info.get('error')}")

                report = await self._predictor.check_plan(
                    plan_path=PICK_WAYPOINT_PATH,
                    warehouse_path=WAREHOUSE_MODEL_PATH,
                    robot_size=robot_size,
                    robot_info=robot_info,
                )
                report["local_context"] = {
                    "layout_name": self._current_layout_name,
                    "active_robot_id": self._active_robot_id,
                    "pick_waypoint_path": PICK_WAYPOINT_PATH,
                    "warehouse_model_path": WAREHOUSE_MODEL_PATH,
                    "report_path": CHECK_REPORT_PATH,
                }

                self._predictor.save_report(report, CHECK_REPORT_PATH)
                if report.get("error"):
                    print(f"Prediction failed: {report['error']}")
                else:
                    response = report.get("response", {})
                    backend = report.get("backend")
                    collision_detected = response.get("collision_detected")
                    collision_idx = response.get("collision_waypoint_index")
                    collision_reason = response.get("collision_reason")

                    print("Prediction completed.")
                    print(f"  Backend: {backend}")
                    print(f"  Predictor URL: {report.get('predictor_url')}")
                    print(f"  Collision detected: {collision_detected}")
                    if collision_idx is not None:
                        print(f"  Collision waypoint index: {collision_idx}")
                    if collision_reason:
                        print(f"  Collision reason: {collision_reason}")
                print(f"Report saved to: {CHECK_REPORT_PATH}")
            finally:
                self._check_in_progress = False

        asyncio.ensure_future(check_plan_async())

    def start_navigation_clicked(self):
        """Load the unified repaired waypoint JSON and run simulation validation."""
        raw_path = self._navigation_path_model.as_string
        result_path = Path(SIMULATION_RESULT_PATH)

        try:
            plan_path = resolve_waypoint_path(raw_path, Path(EXTENSION_ROOT))
            print(f"Loading repaired waypoint sequence: {plan_path}")
            payload, waypoints = load_model_output(plan_path)
            source_method = payload.get(
                "source_baseline_method",
                payload.get("baseline_method", "unspecified"),
            )
            self._validation_context = {
                "method": str(source_method),
                "model_output_path": plan_path,
                "result_path": result_path,
            }
            self._robot_controller.set_failure_callback(self._on_validation_failure)
            self._robot_controller.set_completion_callback(self._on_validation_complete)
            self._robot_controller.start_waypoint_navigation(waypoints)
            print(
                f"START executing {len(waypoints)} repaired waypoints from {plan_path}. "
                "No prediction, repair, or replanning will run during validation."
            )
        except Exception as exc:
            self._validation_context = None
            print(f"Cannot start repaired waypoint validation: {exc}")

    def run_heuristic_baseline_clicked(self):
        self._run_baseline_clicked("heuristic")

    def run_rl_baseline_clicked(self):
        self._run_baseline_clicked("rl")

    def _run_baseline_clicked(self, method):
        if self._baseline_in_progress:
            print("A baseline is already training/planning. Please wait for it to finish.")
            return
        self._baseline_in_progress = True

        async def run_async():
            try:
                packages = get_packages_info()
                if not isinstance(packages, list) or not packages:
                    print(f"Cannot run {method} baseline: no scene packages are available")
                    return
                robot_context = self._get_active_robot_context()
                robot_info = robot_context.get("robot_info", {})
                robot_size = robot_context.get("robot_size", {})
                position = robot_info.get("position")
                if not position or len(position) < 2:
                    print(f"Cannot run {method} baseline: robot position is unavailable")
                    return
                layouts_path = os.path.join(EXPERIMENT_ROOT, "layouts.json")
                print(f"Training/planning {method} baseline for {len(packages)} packages...")
                build_plan = build_heuristic_runtime_plan if method == "heuristic" else build_rl_runtime_plan
                _, plan = await asyncio.to_thread(
                    build_plan,
                    Path(layouts_path),
                    self._current_layout_name,
                    (float(position[0]), float(position[1])),
                    packages,
                    robot_size=robot_size,
                )
                output_dir = Path(
                    HEURISTIC_WAYPOINT_ROOT if method == "heuristic" else RL_WAYPOINT_ROOT
                )
                output_path = output_dir / f"baseline_{method}_experiment_input.json"
                if not plan.success:
                    print(f"{method} baseline planning failed: {plan.failure_reason}")
                    return
                write_experiment_input(
                    output_path,
                    method,
                    self._current_layout_name,
                    plan,
                    Path(WAREHOUSE_MODEL_PATH),
                    robot_context,
                    packages,
                )
                print(
                    f"{method} planning export complete: {len(plan.waypoints)} waypoints. "
                    f"No prediction, repair, or simulation was started. Input JSON: {output_path}"
                )
                print(
                    "Run your collision prediction/repair model offline, save its final JSON as: "
                    f"{REPAIRED_WAYPOINT_PATH}, then click START."
                )
            except Exception as exc:
                print(f"{method} baseline error: {exc}")
                import traceback
                traceback.print_exc()
            finally:
                self._baseline_in_progress = False

        asyncio.ensure_future(run_async())

    def _on_validation_failure(self, failure_info):
        context = self._validation_context
        if not context:
            return
        output = write_simulation_result(
            context["result_path"],
            context["method"],
            context["model_output_path"],
            False,
            self._robot_controller,
            failure_info,
        )
        print(
            f"Simulation validation failed: {failure_info.get('type')} at "
            f"{failure_info.get('current_position')}; result: {output}"
        )
        self._validation_context = None

    def _on_validation_complete(self, completion_info):
        context = self._validation_context
        if not context:
            return
        output = write_simulation_result(
            context["result_path"],
            context["method"],
            context["model_output_path"],
            True,
            self._robot_controller,
        )
        print(
            f"Simulation validation succeeded in {completion_info.get('simulation_time_s', 0.0):.3f}s; "
            f"result: {output}"
        )
        self._validation_context = None

    def on_navigation_failed(self, failure_info):
        """
        导航失败时的回调函数,触发重新规划
        
        Args:
            failure_info (dict): 包含失败类型、当前位置、剩余航点等信息的字典
        """
        print("\n" + "="*60)
        print("NAVIGATION FAILURE DETECTED")
        print("="*60)
        print(f"Failure Type: {failure_info['type']}")
        print(f"Current Position: ({failure_info['current_position'][0]:.3f}, {failure_info['current_position'][1]:.3f})")
        print(f"Failed Waypoint: {failure_info['failed_waypoint_index']}/{failure_info['total_waypoints']}")
        print(f"Failed Waypoint Coordinates: {failure_info['failed_waypoint']}")
        print("="*60)
        
        # 启动异步重新规划
        asyncio.ensure_future(self._handle_replan_async(failure_info))

    async def _handle_replan_async(self, failure_info):
        """
        处理重新规划的异步包装函数
        
        Args:
            failure_info (dict): 失败信息
        """
        try:
            # 调用LLMPlanner中的重规划函数
            replan_result = await replan_from_failure(failure_info, EXTENSION_ROOT)
            
            if not replan_result['success']:
                print(f"Replanning failed: {replan_result['error']}")
                return
            
            # 如果没有剩余包裹,任务完成
            if not replan_result['remaining_packages']:
                print("Mission completed - no further action needed.")
                return
            
            # 获取新的航点
            new_waypoints = replan_result['new_waypoints']
            if not new_waypoints:
                print("No waypoints generated in replan.")
                return
            
            # 等待一小段时间,确保系统稳定
            await asyncio.sleep(1.5)
            
            # 自动开始执行新的规划
            print("Resuming navigation with replanned waypoints...")
            self._robot_controller.start_waypoint_navigation(new_waypoints)
            
        except Exception as e:
            print(f"Error handling replan: {e}")
            import traceback
            traceback.print_exc()

    def stop_navigation_clicked(self):
        """Click Stop Navigation button"""
        self._robot_controller.stop_navigation()
        self._validation_context = None
        self._robot_controller.set_completion_callback(None)

    def cleanup(self):
        """Clean up window and menu"""
        # destroy window
        if self._window is not None:
            self._window.destroy()
            self._window = None
        # destroy menu
        if self._menu is not None:
            try:
                omni.kit.ui.get_editor_menu().remove_item(self._menu)
            except Exception:
                pass
            self._menu = None

    def test_button_clicked(self):
        """Test physical collision detection and position reporting"""
        async def test_physical_collision_async():
            import time
            print("\n" + "="*60)
            print("PHYSICAL COLLISION DETECTION TEST (INTERACTIVE)")
            print("="*60)
            
            try:
                from .NavMonitor import NavigationMonitor
                from .Utils import get_robot_info
                
                # 1. 获取机器人信息用于初始化
                robot_context = self._get_active_robot_context()
                robot_info = robot_context.get("robot_info")
                if not robot_info.get('success', True):
                    print(f"Error getting robot info: {robot_info.get('error')}")
                    return
                
                # 获取当前位置作为初始参考
                initial_pos = (robot_info['position'][0], robot_info['position'][1])
                
                # 2. 初始化监控器
                print("Initializing NavigationMonitor with PhysX subscription...")
                # 注意：sensor_path 只要包含机器人的路径即可，NavMonitor会自动解析出 Prim 路径
                monitor = NavigationMonitor(
                    sensor_path=robot_context.get("robot_cfg", {}).get("sensor_path", "/World/RidgebackFranka/base_link/Contact_Sensor"), 
                    collision_force_threshold=0.0001 # 设置较低的阈值以便更容易触发测试
                )
                
                # 3. 启动监控 (这将订阅 PhysX 事件)
                monitor.start_monitoring(initial_pos)
                
                start_time = time.time()
                duration = 30.0
                
                while time.time() - start_time < duration:
                    # 检查物理碰撞
                    # 我们传入 initial_pos 仅作为回退，主要依赖 PhysX 回调数据
                    result = monitor.check_sensor_collision(current_position=initial_pos)
                    
                    if result['failed']:
                        print(f"\n[COLLISION DETECTED] Time: {time.time() - start_time:.1f}s")
                        print(f"  Target Object: {result.get('colliding_object', 'Unknown')}")
                        print(f"  Contact Point: {result.get('collision_position')}")
                        print(f"  Impulse:       {result.get('force'):.6f}")
                        print(f"  Message:       {result.get('message')}")
                        
                        # 稍微暂停一下，避免控制台被刷屏
                        await asyncio.sleep(0.5)
                    
                    # 保持检测频率
                    await asyncio.sleep(0.1)
                
                monitor.stop_monitoring()
                print("\n" + "="*60)
                print("TEST COMPLETED")
                print("="*60)

            except Exception as e:
                print(f"Error during physical collision test: {e}")
                import traceback
                traceback.print_exc()
        
        # 启动异步测试
        asyncio.ensure_future(test_physical_collision_async())

    def collect_file_dataset_clicked(self):
        """从文件读取航点序列并收集碰撞数据"""
        async def collect_async():
            # 将当前布局数据传递给收集器
            layout_data = self._scene_manager.current_layout_data
            robot_id, robot_cfg = self._scene_manager.get_robot_config()

            collector = CollisionDataCollector(
                self._robot_controller,
                output_dir=os.path.join(EXTENSION_ROOT, "data", "collision_dataset"),
                reset_position=(0.0, 0.0, 0.0),
                layout_config=layout_data,
                robot_config=robot_cfg,
            )
            self._collision_collector = collector
            
            # 从文件加载航点序列
            waypoint_file = os.path.join(EXPERIMENT_ROOT, "SimWaypoint.txt")
            sequences = collector.load_waypoint_sequences_from_file(waypoint_file)
            
            if not sequences:
                print(f"No waypoint sequences found in {waypoint_file}")
                self._collision_collector = None
                return
            
            print(f"Loaded {len(sequences)} sequences from file")
            print(f"Starting file-based dataset collection...")
            
            await collector.collect_dataset(
                sequences,
                robot_id=self._active_robot_id,
                save_interval=5,
            )
            self._collision_collector = None
        
        asyncio.ensure_future(collect_async())
    
    def collect_a_dataset_clicked(self):
        """生成工业拣货路线并收集碰撞数据"""
        async def collect_async():
            # 将当前布局数据传递给收集器
            layout_data = self._scene_manager.current_layout_data
            robot_id, robot_cfg = self._scene_manager.get_robot_config()

            collector = CollisionDataCollector(
                self._robot_controller,
                output_dir=os.path.join(EXTENSION_ROOT, "data", "collision_dataset"),
                reset_position=(0.0, 0.0, 0.0),
                layout_config=layout_data,
                robot_config=robot_cfg,
            )
            self._collision_collector = collector
            
            # 生成符合工业场景的拣货路线
            sequences = collector.generate_industrial_picking_sequences({
                'num_sequences': 300,
                'collision_probability': 0.70,
                'lateral_localization_offset_weight': 0.30,
                'shelf_corner_cutting_weight': 0.30,
                'docking_overtravel_weight': 0.25,
                'wall_localization_drift_weight': 0.15,
                'hard_safe_probability_given_safe': 0.50,
                'collision_penetration_min_m': 0.06,
                'collision_penetration_max_m': 0.22,
                'fault_transition_waypoints_min': 1,
                'fault_transition_waypoints_max': 3,
                'hard_safe_margin_min_m': 0.05,
                'hard_safe_margin_max_m': 0.18,
                'collision_index_min': 4,
                'collision_index_max': 27,
                'waypoint_count_min': 15,
                'waypoint_count_max': 30,
            })
            
            print(f"Generated {len(sequences)} industrial picking sequences")
            print(f"Starting industrial route dataset collection...")
            
            await collector.collect_dataset(
                sequences,
                robot_id=self._active_robot_id,
                save_interval=5,
            )
            self._collision_collector = None
        
        asyncio.ensure_future(collect_async())

    def stop_collection_clicked(self):
        """停止数据收集"""
        if self._collision_collector:
            self._collision_collector.save_dataset(partial=True)
            print("Dataset collection stopped by user")
        self._robot_controller.stop_navigation()
