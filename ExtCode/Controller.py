import math
import time
import numpy as np
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from omni.physx import get_physx_interface
from isaacsim.sensors.physics import _sensor
from .NavMonitor import NavigationMonitor

class RobotController:
    """控制场景中的Ridgeback机器人,并能沿航点导航。"""
    def __init__(
        self,
        robot_prim_path="/World/RidgebackFranka",
        sensor_path=None,
        robot_width=0.8,
        robot_length=1.0,
        max_linear_speed=1.5,
        max_linear_acceleration=3.0,
        max_angular_speed=2.5,
        max_angular_acceleration=6.0,
        heading_tolerance=math.radians(2.0),
        trajectory_sample_interval=0.5,
        collision_force_threshold=0.01,
        collision_confirmation_events=2,
        collision_confirmation_window=0.15,
        strong_collision_impulse_threshold=0.5,
        max_recorded_contact_events=20,
        motion_mode="simple",
    ):
        self._robot_prim_path = robot_prim_path
        self._sensor_path = sensor_path or f"{robot_prim_path}/base_link/Contact_Sensor"
        self._robot_width = float(robot_width)
        self._robot_length = float(robot_length)
        self._max_linear_speed = float(max_linear_speed)
        self._max_linear_acceleration = float(max_linear_acceleration)
        self._max_angular_speed = float(max_angular_speed)
        self._max_angular_acceleration = float(max_angular_acceleration)
        self._heading_tolerance = float(heading_tolerance)
        self._motion_mode = "simple"
        self.set_motion_mode(motion_mode)
        self._commanded_linear_speed = 0.0
        self._commanded_angular_speed = 0.0
        self._commanded_x = None
        self._commanded_y = None
        self._commanded_yaw = None
        self._simulation_time = 0.0
        self._trajectory = []
        self._trajectory_sample_interval = max(0.05, float(trajectory_sample_interval))
        self._last_trajectory_sample_time = -float("inf")
        self._robot = None
        self._x_joint_idx = None
        self._y_joint_idx = None
        self._rz_joint_idx = None
        self._is_initialized = False

        # 导航状态属性
        self._waypoints = []
        self._current_waypoint_index = 0
        self._is_navigating = False
        self._navigation_completed = False  # 添加这个标志
        self._physics_subscription = None
        self._arrival_threshold = 0.1
        
        # 导航监控器 (包含传感器碰撞检测、超时检测、卡住检测)
        self._monitor = NavigationMonitor(
            timeout=60.0, 
            stuck_check_interval=8.0, 
            stuck_threshold=0.02,
            collision_force_threshold=collision_force_threshold,
            sensor_path=self._sensor_path,
            robot_base_width=self._robot_width,
            robot_base_length=self._robot_length,
            collision_confirmation_events=collision_confirmation_events,
            collision_confirmation_window=collision_confirmation_window,
            strong_collision_impulse_threshold=strong_collision_impulse_threshold,
            max_recorded_contact_events=max_recorded_contact_events,
        )
        
        # 失败回调
        self._on_navigation_failed = None
        self._on_navigation_completed = None

    def set_failure_callback(self, callback):
        """设置导航失败时的回调函数"""
        self._on_navigation_failed = callback

    def set_motion_mode(self, motion_mode):
        """Select waypoint motion behavior: ``simple`` or simultaneous ``hard``."""
        normalized = str(motion_mode).strip().lower()
        if normalized not in {"simple", "hard"}:
            raise ValueError("motion_mode must be 'simple' or 'hard'")
        self._motion_mode = normalized

    def get_motion_mode(self):
        return self._motion_mode

    def set_completion_callback(self, callback):
        """Set an optional callback invoked after all waypoints are reached."""
        self._on_navigation_completed = callback

    def get_motion_config(self):
        return {
            "control_mode": self._motion_mode,
            "motion_mode": self._motion_mode,
            "max_linear_speed_mps": self._max_linear_speed,
            "max_linear_acceleration_mps2": self._max_linear_acceleration,
            "max_angular_speed_radps": self._max_angular_speed,
            "max_angular_acceleration_radps2": self._max_angular_acceleration,
            "heading_tolerance_rad": self._heading_tolerance,
            "arrival_threshold_m": self._arrival_threshold,
            "trajectory_sample_interval_seconds": self._trajectory_sample_interval,
        }

    def get_executed_trajectory(self):
        return [sample.copy() for sample in self._trajectory]

    def get_simulation_time(self):
        return float(self._simulation_time)

    def get_contact_diagnostics(self):
        getter = getattr(self._monitor, 'get_contact_diagnostics', None)
        return getter() if callable(getter) else {}

    def get_collision_confirmation_config(self):
        getter = getattr(self._monitor, 'get_collision_confirmation_config', None)
        return getter() if callable(getter) else {}

    def _reset_motion_state(self, clear_pose=False):
        self._commanded_linear_speed = 0.0
        self._commanded_angular_speed = 0.0
        if clear_pose:
            self._commanded_x = None
            self._commanded_y = None
            self._commanded_yaw = None

    def _sync_command_pose(self, joint_positions):
        self._commanded_x = float(joint_positions[self._x_joint_idx])
        self._commanded_y = float(joint_positions[self._y_joint_idx])
        self._commanded_yaw = float(joint_positions[self._rz_joint_idx])

    @staticmethod
    def _wrap_angle(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _approach(current, target, max_delta):
        if current < target:
            return min(target, current + max_delta)
        return max(target, current - max_delta)

    def _handle_navigation_failure(self, failure_type, current_x, current_y, details=None):
        """
        处理导航失败
        
        Args:
            failure_type (str): 失败类型 ('timeout', 'stuck', 'collision')
            current_x (float): 当前X坐标
            current_y (float): 当前Y坐标
            details (dict, optional): 额外的失败详情
        """
        print(f"Navigation failed: {failure_type} at ({current_x:.2f}, {current_y:.2f})")
        
        failure_info = {
            "type": failure_type,
            "timestamp": time.time(),
            "current_position": (current_x, current_y),
            "failed_waypoint_index": self._current_waypoint_index,
            "failed_waypoint": self._waypoints[self._current_waypoint_index] if self._current_waypoint_index < len(self._waypoints) else None,
            "remaining_waypoints": self._waypoints[self._current_waypoint_index:],
            "completed_waypoints": self._waypoints[:self._current_waypoint_index],
            "total_waypoints": len(self._waypoints),
            "details": details or {}
        }
        
        self.stop_navigation()
        
        if self._on_navigation_failed:
            try:
                self._on_navigation_failed(failure_info)
            except Exception as e:
                print(f"Error in failure callback: {e}")

    def initialize(self):
        """初始化机器人控制器"""
        if self._is_initialized:
            return True
        try:
            self._robot = SingleArticulation(prim_path=self._robot_prim_path)
            self._robot.initialize()
            dof_names = self._robot.dof_names
            self._x_joint_idx = dof_names.index("dummy_base_prismatic_x_joint")
            self._y_joint_idx = dof_names.index("dummy_base_prismatic_y_joint")
            self._rz_joint_idx = dof_names.index("dummy_base_revolute_z_joint")
            self._is_initialized = True
            print(f"RobotController prim <{self._robot_prim_path}> initialized successfully.")
            return True
        except Exception as e:
            print(f"Initialize RobotController failed: {e}")
            self._robot = None
            self._is_initialized = False
            return False

    def reset(self):
        """重置控制器,并停止任何正在进行的导航。"""
        self.stop_navigation()
        self._robot = None
        self._is_initialized = False
        print("RobotController state has been reset.")

    def start_waypoint_navigation(self, waypoints, motion_mode=None):
        """
        开始沿给定的航点列表进行导航。
        
        Args:
            waypoints (list): 一个包含(x, y)坐标元组的列表。
            motion_mode (str, optional): Override the configured mode for this run.
        """
        if motion_mode is not None:
            self.set_motion_mode(motion_mode)
        if not waypoints:
            print("Waypoints list is empty. Cannot start navigation.")
            return
        if not self._is_initialized:
            if not self.initialize():
                print("Initialization failed. Cannot start navigation.")
                return
        
        self._waypoints = waypoints
        self._current_waypoint_index = 0
        self._is_navigating = True
        self._navigation_completed = False
        self._reset_motion_state(clear_pose=True)
        self._simulation_time = 0.0
        self._trajectory = []
        self._last_trajectory_sample_time = -float("inf")
        
        # 获取初始位置并启动监控(包括碰撞检测)
        joint_positions = self._robot.get_joint_positions()
        if joint_positions is not None:
            self._sync_command_pose(joint_positions)
            initial_pose = (
                joint_positions[self._x_joint_idx],
                joint_positions[self._y_joint_idx],
                joint_positions[self._rz_joint_idx],
            )
            self._monitor.start_monitoring(initial_pose)
        
        # 订阅物理更新事件以运行导航循环
        if self._physics_subscription is None:
            self._physics_subscription = get_physx_interface().subscribe_physics_step_events(self._navigation_update)
        
        print(f"Start Navigation: {len(self._waypoints)} Waypoints")
        print("Collision detection enabled (physical sensor)")

    def stop_navigation(self):
        """停止当前的航点导航任务。"""
        # Disable callbacks before unsubscribing or clearing their state. In fast
        # mode, another physics callback can already be queued in this window.
        self._is_navigating = False
        if self._physics_subscription:
            self._physics_subscription.unsubscribe()
            self._physics_subscription = None
        
        # 停止监控
        self._monitor.stop_monitoring()
        
        self._waypoints = []
        self._current_waypoint_index = 0
        self._reset_motion_state(clear_pose=True)
        print("Navigation stopped")

    def _navigation_update(self, dt: float):
        if (
            not self._is_navigating
            or not self._is_initialized
            or not self._waypoints
            or self._current_waypoint_index >= len(self._waypoints)
        ):
            return

        self._monitor.advance_simulation_time(dt)
        self._simulation_time += max(0.0, float(dt))

        # 获取当前目标航点
        target_x, target_y = self._waypoints[self._current_waypoint_index]

        # 获取机器人当前位置
        joint_positions = self._robot.get_joint_positions()
        if joint_positions is None:
            self.reset()
            return
        current_x = joint_positions[self._x_joint_idx]
        current_y = joint_positions[self._y_joint_idx]
        current_pos = (current_x, current_y)
        current_pose = (current_x, current_y, joint_positions[self._rz_joint_idx])
        if self._simulation_time - self._last_trajectory_sample_time >= self._trajectory_sample_interval:
            self._trajectory.append({
                "simulation_time": round(self._simulation_time, 6),
                "x": round(float(current_x), 6),
                "y": round(float(current_y), 6),
                "yaw": round(float(joint_positions[self._rz_joint_idx]), 6),
                "commanded_linear_speed": round(float(self._commanded_linear_speed), 6),
                "commanded_angular_speed": round(float(self._commanded_angular_speed), 6),
                "commanded_x": round(float(self._commanded_x), 6),
                "commanded_y": round(float(self._commanded_y), 6),
                "commanded_yaw": round(float(self._commanded_yaw), 6),
            })
            self._last_trajectory_sample_time = self._simulation_time

        # 执行所有失败检测 (传感器碰撞、超时、卡住)
        check_result = self._monitor.check_all(current_pose)
        if check_result['failed']:
            self._handle_navigation_failure(
                check_result['failure_type'], 
                current_x, 
                current_y,
                check_result['details']
            )
            return

        # 检查是否已到达当前航点
        distance_to_target = math.hypot(target_x - current_x, target_y - current_y)
        if distance_to_target < self._arrival_threshold:
            print(f"Reached waypoint {self._current_waypoint_index}: ({target_x:.3f}, {target_y:.3f})")
            self._current_waypoint_index += 1
            self._reset_motion_state()
            self._sync_command_pose(joint_positions)
            
            # 重置监控器的航点计时器
            self._monitor.reset_waypoint_timer(
                current_pose, waypoint_index=self._current_waypoint_index
            )
            
            # 检查是否已完成所有航点
            if self._current_waypoint_index >= len(self._waypoints):
                print("=" * 60)
                print("NAVIGATION COMPLETE - ALL WAYPOINTS REACHED")
                print("=" * 60)
                self._navigation_completed = True  # 设置完成标志
                completion_callback = self._on_navigation_completed
                completion_info = {
                    "completed_waypoints": list(self._waypoints),
                    "total_waypoints": len(self._waypoints),
                    "simulation_time_s": self.get_simulation_time(),
                }
                self.stop_navigation()
                if completion_callback:
                    try:
                        completion_callback(completion_info)
                    except Exception as exc:
                        print(f"Error in completion callback: {exc}")
                return
            else:
                # 更新目标为下一个航点
                target_x, target_y = self._waypoints[self._current_waypoint_index]
                print(f"Next waypoint {self._current_waypoint_index}: ({target_x:.3f}, {target_y:.3f})")

        # 移动到当前(或下一个)航点
        self.move_to_target(target_x, target_y, dt)

    def move_to_target(self, target_x, target_y, dt=1.0 / 60.0):
        """
        命令机器人移动到单个目标 (x, y) 位置
        
        Args:
            target_x (float): 目标X坐标
            target_y (float): 目标Y坐标
        """
        if not self._is_initialized:
            return

        try:
            joint_positions = self._robot.get_joint_positions()
            if joint_positions is None:
                self.reset()
                return

            delta_x = target_x - joint_positions[self._x_joint_idx]
            delta_y = target_y - joint_positions[self._y_joint_idx]
            distance = math.hypot(delta_x, delta_y)
            if distance <= 1e-9:
                self._reset_motion_state()
                return

            dt = max(0.0, float(dt))
            if dt <= 0.0:
                return

            target_angle = math.atan2(delta_y, delta_x)
            current_angle = float(joint_positions[self._rz_joint_idx])
            angle_error = self._wrap_angle(target_angle - current_angle)

            if self._commanded_x is None:
                self._sync_command_pose(joint_positions)

            action = ArticulationAction(joint_positions=np.full(self._robot.num_dof, np.nan))

            command_angle_error = self._wrap_angle(target_angle - self._commanded_yaw)
            angular_limit = min(self._max_angular_speed, abs(command_angle_error) / dt)
            self._commanded_angular_speed = self._approach(
                self._commanded_angular_speed,
                angular_limit,
                self._max_angular_acceleration * dt,
            )
            angle_step = min(abs(command_angle_error), self._commanded_angular_speed * dt)
            if angle_step > 0.0:
                self._commanded_yaw += math.copysign(angle_step, command_angle_error)

            command_delta_x = target_x - self._commanded_x
            command_delta_y = target_y - self._commanded_y
            command_distance = math.hypot(command_delta_x, command_delta_y)
            if self._motion_mode == "simple" and abs(angle_error) > self._heading_tolerance:
                # Rotate in place until the measured heading is within tolerance.
                self._commanded_linear_speed = 0.0
            else:
                stopping_speed = math.sqrt(
                    max(0.0, 2.0 * self._max_linear_acceleration * command_distance)
                )
                desired_speed = min(self._max_linear_speed, stopping_speed)
                if self._motion_mode == "hard":
                    # Translation and yaw update in the same step. Reducing the
                    # speed for a large heading error avoids commanding reverse motion.
                    desired_speed *= max(0.0, math.cos(angle_error))
                self._commanded_linear_speed = self._approach(
                    self._commanded_linear_speed,
                    desired_speed,
                    self._max_linear_acceleration * dt,
                )
                if command_distance > 1e-9:
                    distance_step = min(command_distance, self._commanded_linear_speed * dt)
                    self._commanded_x += command_delta_x / command_distance * distance_step
                    self._commanded_y += command_delta_y / command_distance * distance_step

            # Some articulation configurations ignore a sparse action containing NaNs
            # on the base joints. Always hold or command the complete planar pose.
            action.joint_positions[self._x_joint_idx] = self._commanded_x
            action.joint_positions[self._y_joint_idx] = self._commanded_y
            action.joint_positions[self._rz_joint_idx] = self._commanded_yaw
            
            self._robot.apply_action(action)
        except Exception as e:
            print(f"move_to_target failed: {e}")
            self.reset()
