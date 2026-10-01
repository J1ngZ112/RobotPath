import math
import re
from collections import deque
from omni.physx import get_physx_simulation_interface
from pxr import PhysicsSchemaTools

class NavigationMonitor:
    """导航监控器,负责检测导航过程中的各种失败情况"""
    
    def __init__(self, timeout=60.0, stuck_check_interval=5.0, stuck_threshold=0.05, 
                 collision_force_threshold=0.01, sensor_path="/World/RidgebackFranka/base_link/Contact_Sensor",
                 robot_base_width=0.8, robot_base_length=1.0,
                 collision_confirmation_events=2,
                 collision_confirmation_window=0.15,
                 strong_collision_impulse_threshold=0.5,
                 max_recorded_contact_events=20):
        self.timeout = timeout
        self.stuck_check_interval = stuck_check_interval
        self.stuck_threshold = stuck_threshold
        self.collision_force_threshold = collision_force_threshold
        self.robot_base_width = float(robot_base_width)
        self.robot_base_length = float(robot_base_length)
        self._collision_confirmation_threshold = max(1, int(collision_confirmation_events))
        self._collision_confirmation_window = max(0.0, float(collision_confirmation_window))
        self._strong_collision_impulse_threshold = max(
            self.collision_force_threshold, float(strong_collision_impulse_threshold)
        )
        self._max_recorded_contact_events = max(1, int(max_recorded_contact_events))
        self.sensor_path = sensor_path # 注意：这里其实只需要机器人的Prim路径即可，不需要具体的Sensor路径
        # 提取机器人的Prim路径 (假设sensor挂在base_link下)
        self.monitored_link_path = sensor_path.split("/Contact_Sensor")[0]
        base_link_marker = "/base_link/"
        if base_link_marker in sensor_path:
            self.robot_prim_path = sensor_path.split(base_link_marker, 1)[0]
        else:
            self.robot_prim_path = self.monitored_link_path
        
        # 状态变量
        self._waypoint_start_time = None
        self._last_position = None
        self._last_check_time = None
        self._is_monitoring = False
        
        # PhysX 接触报告订阅
        self._contact_report_sub = None
        self._latest_collision_data = None # 存储最新的碰撞数据
        self._first_collision_data = None # 存储本次导航中的首次接触数据
        
        self._collision_detected_count = 0
        self._simulation_time = 0.0
        self._collision_event_id = 0
        self._contact_report_id = 0
        self._last_checked_collision_event_id = 0
        self._max_impulse_collision_data = None
        self._confirmed_collision_data = None
        self._raw_contact_event_count = 0
        self._contact_events = []
        self._recent_contact_events = deque()
        self._max_consecutive_contact_events = 0
        self._current_waypoint_index = 0

    def advance_simulation_time(self, dt):
        """Advance the monitor clock using the physics-step delta."""
        if not self._is_monitoring:
            return
        try:
            self._simulation_time += max(0.0, float(dt))
        except (TypeError, ValueError):
            pass

    @staticmethod
    def _contact_object_key(actor_path):
        path = str(actor_path or '')
        match = re.search(r'(/World/Shelf[_-]?0*\d+)', path, re.IGNORECASE)
        return match.group(1).lower().replace('_', '-') if match else path
    
    def start_monitoring(self, initial_position):
        """启动监控"""
        self._is_monitoring = True
        self._simulation_time = 0.0
        self._waypoint_start_time = self._simulation_time
        self._last_position = initial_position
        self._last_check_time = self._simulation_time
        self._collision_detected_count = 0
        self._latest_collision_data = None
        self._first_collision_data = None
        self._max_impulse_collision_data = None
        self._collision_event_id = 0
        self._contact_report_id = 0
        self._last_checked_collision_event_id = 0
        self._confirmed_collision_data = None
        self._raw_contact_event_count = 0
        self._contact_events = []
        self._recent_contact_events = deque()
        self._max_consecutive_contact_events = 0
        self._current_waypoint_index = 0
        
        # 订阅 PhysX 接触报告
        try:
            self._contact_report_sub = get_physx_simulation_interface().subscribe_contact_report_events(self._on_contact_report_event)
            print(f"PhysX contact report subscription started. Monitoring collisions for: {self.robot_prim_path}")
        except Exception as e:
            print(f"Warning: Could not subscribe to contact reports: {e}")
            self._contact_report_sub = None
        
        print(f"Navigation monitoring started at position: ({initial_position[0]:.3f}, {initial_position[1]:.3f})")

    def stop_monitoring(self):
        """停止监控"""
        self._is_monitoring = False
        self._waypoint_start_time = None
        self._last_position = None
        self._last_check_time = None
        self._collision_detected_count = 0
        self._latest_collision_data = None
        self._first_collision_data = None
        
        # 取消订阅
        if self._contact_report_sub is not None and hasattr(self._contact_report_sub, "unsubscribe"):
            try:
                self._contact_report_sub.unsubscribe()
            except Exception:
                pass
        self._contact_report_sub = None
        print("Navigation monitoring stopped")

    def _on_contact_report_event(self, contact_headers, contact_data):
        """PhysX 接触报告回调函数"""
        if not self._is_monitoring:
            return
        self._contact_report_id += 1

        # 遍历所有接触头信息
        for contact_header in contact_headers:
            # 获取碰撞的两个 Actor 的路径
            actor0_path = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor0))
            actor1_path = str(PhysicsSchemaTools.intToSdfPath(contact_header.actor1))
            
            root_prefix = self.robot_prim_path + "/"
            actor0_is_robot = actor0_path == self.robot_prim_path or actor0_path.startswith(root_prefix)
            actor1_is_robot = actor1_path == self.robot_prim_path or actor1_path.startswith(root_prefix)

            # Internal articulation contacts are not environment collisions.
            if actor0_is_robot and actor1_is_robot:
                continue

            is_robot_collision = actor0_is_robot or actor1_is_robot
            other_actor_path = actor1_path if actor0_is_robot else actor0_path
            
            # 如果是机器人的碰撞，处理详细数据
            if is_robot_collision:
                contact_data_offset = contact_header.contact_data_offset
                num_contact_data = contact_header.num_contact_data
                
                max_impulse = 0.0
                best_contact_pos = None
                
                # 遍历该 Header 下的所有接触点
                for index in range(contact_data_offset, contact_data_offset + num_contact_data):
                    c_data = contact_data[index]
                    impulse_vec = c_data.impulse
                    impulse_mag = impulse_vec[0]**2 + impulse_vec[1]**2 + impulse_vec[2]**2
                    
                    if impulse_mag > max_impulse:
                        max_impulse = impulse_mag
                        best_contact_pos = (c_data.position[0], c_data.position[1])
                
                impulse = math.sqrt(max_impulse)
                if best_contact_pos and impulse > self.collision_force_threshold:
                    self._collision_event_id += 1
                    collision_record = {
                        'position': best_contact_pos,
                        'impulse': impulse,
                        'other_actor': other_actor_path,
                        'timestamp': self._simulation_time,
                        'event_id': self._collision_event_id,
                        'report_id': self._contact_report_id,
                        'waypoint_index': self._current_waypoint_index,
                        'object_key': self._contact_object_key(other_actor_path),
                    }
                    self._raw_contact_event_count += 1
                    if len(self._contact_events) < self._max_recorded_contact_events:
                        self._contact_events.append(collision_record.copy())
                    self._latest_collision_data = collision_record
                    if self._first_collision_data is None:
                        self._first_collision_data = collision_record.copy()
                    if (
                        self._max_impulse_collision_data is None
                        or impulse > self._max_impulse_collision_data['impulse']
                    ):
                        self._max_impulse_collision_data = collision_record.copy()

                    self._recent_contact_events.append(collision_record.copy())
                    window_start = self._simulation_time - self._collision_confirmation_window
                    while (
                        self._recent_contact_events
                        and self._recent_contact_events[0]['timestamp'] < window_start
                    ):
                        self._recent_contact_events.popleft()
                    same_actor_events = []
                    seen_report_ids = set()
                    for event in reversed(self._recent_contact_events):
                        if event['object_key'] != collision_record['object_key']:
                            continue
                        if event['report_id'] in seen_report_ids:
                            continue
                        seen_report_ids.add(event['report_id'])
                        same_actor_events.append(event)
                    same_actor_events.reverse()
                    self._collision_detected_count = len(same_actor_events)
                    self._max_consecutive_contact_events = max(
                        self._max_consecutive_contact_events, len(same_actor_events)
                    )
                    if (
                        self._confirmed_collision_data is None
                        and (
                            impulse >= self._strong_collision_impulse_threshold
                            or len(same_actor_events) >= self._collision_confirmation_threshold
                        )
                    ):
                        first_event = same_actor_events[0] if same_actor_events else collision_record
                        self._confirmed_collision_data = collision_record.copy()
                        self._confirmed_collision_data['first_confirming_event'] = first_event.copy()
                        self._confirmed_collision_data['confirmation_reason'] = (
                            'strong_impulse'
                            if impulse >= self._strong_collision_impulse_threshold
                            else 'repeated_contact'
                        )

    def reset_waypoint_timer(self, current_position, waypoint_index=None):
        """
        重置航点计时器(到达新航点时调用)
        """
        self._waypoint_start_time = self._simulation_time
        self._last_position = current_position
        self._last_check_time = self._simulation_time
        if waypoint_index is not None:
            self._current_waypoint_index = max(0, int(waypoint_index))

    def reset_collision_state(self):
        """Reset contact state once at the beginning of a navigation sequence."""
        self._collision_detected_count = 0
        self._latest_collision_data = None
        self._first_collision_data = None
        self._max_impulse_collision_data = None
        self._confirmed_collision_data = None
        self._collision_event_id = 0
        self._contact_report_id = 0
        self._last_checked_collision_event_id = 0
        self._raw_contact_event_count = 0
        self._contact_events = []
        self._recent_contact_events = deque()
        self._max_consecutive_contact_events = 0
        self._current_waypoint_index = 0

    def get_contact_diagnostics(self):
        return {
            'raw_contact_event_count': self._raw_contact_event_count,
            'recorded_contact_event_count': len(self._contact_events),
            'contact_event_log_truncated': self._raw_contact_event_count > len(self._contact_events),
            'max_consecutive_contact_events': self._max_consecutive_contact_events,
            'subthreshold_contact_detected': (
                self._raw_contact_event_count > 0 and self._confirmed_collision_data is None
            ),
            'contact_events': [event.copy() for event in self._contact_events],
        }

    def get_collision_confirmation_config(self):
        return {
            'minimum_impulse': self.collision_force_threshold,
            'strong_impulse_threshold': self._strong_collision_impulse_threshold,
            'repeated_contact_events': self._collision_confirmation_threshold,
            'confirmation_window_seconds': self._collision_confirmation_window,
        }

    def check_timeout(self):
        """检查超时"""
        if not self._is_monitoring or self._waypoint_start_time is None:
            return {'failed': False, 'type': None, 'elapsed_time': 0, 'message': None}
        
        elapsed = self._simulation_time - self._waypoint_start_time
        
        if elapsed > self.timeout:
            return {
                'failed': True,
                'type': 'timeout',
                'elapsed_time': elapsed,
                'message': f"Waypoint timeout: {elapsed:.1f}s elapsed (limit: {self.timeout}s)"
            }
        
        return {'failed': False, 'type': None, 'elapsed_time': elapsed, 'message': None}
    
    def check_stuck(self, current_position):
        """检查机器人是否卡住"""
        if not self._is_monitoring:
            return {'failed': False, 'type': None, 'distance_moved': 0, 'time_elapsed': 0, 'message': None}
        
        current_time = self._simulation_time
        current_x, current_y = current_position[:2]
        
        if self._last_check_time is None or self._last_position is None:
            self._last_check_time = current_time
            self._last_position = current_position
            return {'failed': False, 'type': None, 'distance_moved': 0, 'time_elapsed': 0, 'message': None}
        
        time_elapsed = current_time - self._last_check_time
        
        if time_elapsed >= self.stuck_check_interval:
            distance_moved = math.hypot(
                current_x - self._last_position[0],
                current_y - self._last_position[1]
            )
            angular_distance = 0.0
            if len(current_position) >= 3 and len(self._last_position) >= 3:
                yaw_delta = (current_position[2] - self._last_position[2] + math.pi) % (2 * math.pi) - math.pi
                rotation_radius = 0.5 * max(self.robot_base_width, self.robot_base_length)
                angular_distance = abs(yaw_delta) * rotation_radius
            motion_progress = max(distance_moved, angular_distance)
            
            if motion_progress < self.stuck_threshold:
                return {
                    'failed': True,
                    'type': 'stuck',
                    'distance_moved': distance_moved,
                    'angular_distance_m': angular_distance,
                    'time_elapsed': time_elapsed,
                    'message': (
                        f"Robot stuck: progress {motion_progress:.3f}m equivalent in "
                        f"{time_elapsed:.1f}s (threshold: {self.stuck_threshold}m)"
                    )
                }
            
            # 更新检测基准
            self._last_position = current_position
            self._last_check_time = current_time
        
        return {'failed': False, 'type': None, 'distance_moved': 0, 'time_elapsed': 0, 'message': None}
    
    def check_sensor_collision(self, current_position=None):
        """Return the collision already confirmed by the PhysX callback."""
        if not self._is_monitoring or self._confirmed_collision_data is None:
            return {'failed': False, 'type': None, 'force': 0, 'message': None}
        collision_data = self._confirmed_collision_data
        pos = collision_data['position']
        first_collision = collision_data.get(
            'first_confirming_event', self._first_collision_data or collision_data
        )
        max_impulse_collision = self._max_impulse_collision_data or collision_data
        impulse = collision_data['impulse']
        other_obj = collision_data['other_actor']
        return {
            'failed': True,
            'type': 'sensor_collision',
            'force': impulse,
            'first_contact_position': first_collision.get('position', pos),
            'first_contact_impulse': first_collision.get('impulse', impulse),
            'first_contact_timestamp': first_collision.get('timestamp'),
            'collision_position': pos,
            'colliding_object': other_obj,
            'max_impulse_contact_position': max_impulse_collision.get('position', pos),
            'max_impulse': max_impulse_collision.get('impulse', impulse),
            'confirmation_reason': collision_data.get('confirmation_reason'),
            'message': (
                f"Physical collision with {other_obj} at "
                f"({pos[0]:.2f}, {pos[1]:.2f})! Impulse: {impulse:.4f}"
            ),
        }

    def check_all(self, current_position):
        """
        执行所有失败检测
        
        Args:
            current_position (tuple): 当前位置 (x, y)
        
        Returns:
            dict: 包含失败状态和详细信息的字典
        """
        if not self._is_monitoring:
            return {'failed': False, 'failure_type': None, 'details': {}, 'message': None}

        # 1. 检查传感器碰撞 (物理接触 - 会触发失败)
        sensor_result = self.check_sensor_collision(current_position)
        if sensor_result['failed']:
            return {
                'failed': True,
                'failure_type': 'collision',
                'details': sensor_result,
                'message': sensor_result['message']
            }

        # 2. 检查超时
        timeout_result = self.check_timeout()
        if timeout_result['failed']:
            return {
                'failed': True,
                'failure_type': 'timeout',
                'details': timeout_result,
                'message': timeout_result['message']
            }

        # 3. 检查卡住
        stuck_result = self.check_stuck(current_position)
        if stuck_result['failed']:
            return {
                'failed': True,
                'failure_type': 'stuck',
                'details': stuck_result,
                'message': stuck_result['message']
            }
        
        return {'failed': False, 'failure_type': None, 'details': {}, 'message': None}
