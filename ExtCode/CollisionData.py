import random
import time
import os
import json
import logging
import asyncio
import re
import numpy as np
import heapq
import math
from datetime import datetime
from pxr import Gf
from .Utils import get_robot_info


def convert_to_serializable(obj):
    """
    递归地将 numpy 类型转换为 Python 原生类型，以便 JSON 序列化
    """
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


class CollisionDataCollector:
    """收集碰撞数据集的工具类 - 基于 A* 航点序列"""

    SCENARIO_TYPES = (
        "lateral_localization_offset",
        "shelf_corner_cutting",
        "docking_overtravel",
        "wall_localization_drift",
    )
    DEFAULT_SCENARIO_WEIGHTS = {
        "lateral_localization_offset": 0.30,
        "shelf_corner_cutting": 0.30,
        "docking_overtravel": 0.25,
        "wall_localization_drift": 0.15,
    }
    MAX_FAULT_WAYPOINTS = 4

    @staticmethod
    def _resolve_random_seed(random_seed):
        if random_seed is not None:
            return int(random_seed)
        return random.SystemRandom().randrange(0, 2 ** 32)
    
    def __init__(self, robot_controller, output_dir=None,
                 reset_position=(0.0, 0.0, 0.0), layout_config=None, robot_config=None,
                 random_seed=None, collection_runtime_config=None):
        self.robot_controller = robot_controller
        if output_dir is None:
            extension_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(extension_root, "data", "collision_dataset")
        self.output_dir = output_dir
        self.dataset = []
        self.robot_id = "unknown"
        self.current_test_id = 0
        self.reset_position = reset_position  # 机器人重置位置
        self.raw_layout_config = layout_config
        self.robot_config = robot_config
        self.random_seed = self._resolve_random_seed(random_seed)
        self._rng = random.Random(self.random_seed)
        self._sequence_generation_info = []
        self._sequence_generation_config = {}
        self._skipped_reset_sequences = 0
        self._requested_sequence_count = 0
        self.collection_runtime_config = collection_runtime_config or {}
        
        # 导航相关参数
        self.grid_map = None
        self.grid_resolution = 0.1  # 10cm 精度
        self.grid_origin = (0, 0)
        self.grid_dims = (0, 0)

        # 碰撞检测状态
        self._collision_occurred = False
        self._collision_info = {}
        self._navigation_failure_occurred = False
        self._navigation_failure_info = {}
        
        # 停止标志
        self._stop_requested = False
        
        # 文件名时间戳 - 在初始化时生成，整个收集过程使用同一个
        self._session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if layout_config:
            self.warehouse_layout = self._parse_layout_config(layout_config)
            # 初始化导航网格
            self._build_occupancy_grid()
        else:
            # 默认回退
            self.warehouse_layout = self._get_default_layout()
            self._build_occupancy_grid()
        
        # 计算拣货点 (初始计算，后续生成序列时可能会动态重算)
        self._calculate_pickup_points(
            margin=0.8, required_clearance=self._get_planning_radius()
        )
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

    def _get_default_layout(self):
        """返回默认的硬编码布局 (为了兼容性)"""
        return {
            'shelves': [
                {'id': 'Shelf-1', 'center': (-4.5, 0.0), 'size': (0.6, 8.0), 'rot': 0},
                {'id': 'Shelf-2', 'center': (-1.5, 0.0), 'size': (0.6, 8.0), 'rot': 0},
                {'id': 'Shelf-3', 'center': (1.5, 0.0), 'size': (0.6, 8.0), 'rot': 0},
                {'id': 'Shelf-4', 'center': (4.5, 0.0), 'size': (0.6, 8.0), 'rot': 0},
            ],
            'start_zone': (0.0, 0.0),
            'dropoff_zone': (0.0, 5.0),
            'bounds': {'x_min': -6.0, 'x_max': 6.0, 'y_min': -6.0, 'y_max': 6.0},
            'pickup_points': [],
            'aisles': {} 
        }

    def _calculate_pickup_points(
        self, margin=0.6, points_per_side=4, required_clearance=None
    ):
        """
        计算拣货点，沿着货架长边随机分布
        points_per_side: 单侧的点数 (总数 = points_per_side * 2)
        """
        self.warehouse_layout['pickup_points'] = []
        if required_clearance is None:
            required_clearance = self._get_planning_radius()
        required_clearance = max(0.0, float(required_clearance))
        seen_positions = set()
        for shelf in self.warehouse_layout['shelves']:
            cx, cy = shelf['center']
            w, l = shelf['size'] 
            rot = shelf.get('rot', 0.0)
            
            offset_dist = w/2 + margin
            
            # 在长边 (local Y) 上分布点
            # 留出两端的一些余量，避免太靠边缘
            padding = 1.0 
            valid_length_half = max(0.1, l/2 - padding)
            
            requested = max(1, int(points_per_side))
            max_attempts = 1 if points_per_side <= 0 else max(20, requested * 12)
            for side_name, side_sign in (('left', -1.0), ('right', 1.0)):
                accepted = 0
                attempts = 0
                while accepted < requested and attempts < max_attempts:
                    attempts += 1
                    y_off = 0.0 if points_per_side <= 0 else self._rng.uniform(
                        -valid_length_half, valid_length_half
                    )
                    local_x = side_sign * offset_dist
                    px = cx + local_x * np.cos(rot) - y_off * np.sin(rot)
                    py = cy + local_x * np.sin(rot) + y_off * np.cos(rot)
                    position = (round(px, 2), round(py, 2))
                    position_key = (shelf['id'], side_name, position)
                    if position_key in seen_positions:
                        continue
                    if not self._is_world_point_navigable(
                        position, required_clearance
                    ):
                        continue
                    seen_positions.add(position_key)
                    self.warehouse_layout['pickup_points'].append({
                        'shelf_id': shelf['id'],
                        'side': side_name,
                        'position': position,
                    })
                    accepted += 1

    def _build_occupancy_grid(self):
        """基于布局构建栅格地图，用于 A* 寻路"""
        try:
            bounds = self.warehouse_layout['bounds']
            padding = 0.1 # padding 减小，防止机器人走出边界或认为角落缝隙可通行
            width_m = bounds['x_max'] - bounds['x_min'] + 2 * padding
            height_m = bounds['y_max'] - bounds['y_min'] + 2 * padding
            
            cols = int(np.ceil(width_m / self.grid_resolution))
            rows = int(np.ceil(height_m / self.grid_resolution))
            
            self.grid_map = np.zeros((rows, cols), dtype=np.int8)
            self.grid_origin = (bounds['x_min'] - padding, bounds['y_min'] - padding)
            self.grid_dims = (rows, cols)
            
            print(f"Building navigation grid: {cols}x{rows}, Origin: {self.grid_origin}")
            
            # Prefer the exact world-space collider polygons captured from USD.
            for shelf in self.warehouse_layout['shelves']:
                collider_polygons = self._shelf_collision_world_polygons(shelf)
                if collider_polygons:
                    for polygon in collider_polygons:
                        min_x = min(point[0] for point in polygon)
                        max_x = max(point[0] for point in polygon)
                        min_y = min(point[1] for point in polygon)
                        max_y = max(point[1] for point in polygon)
                        min_i = max(0, int(np.floor(
                            (min_x - self.grid_origin[0]) / self.grid_resolution
                        )))
                        max_i = min(cols, int(np.ceil(
                            (max_x - self.grid_origin[0]) / self.grid_resolution
                        )))
                        min_j = max(0, int(np.floor(
                            (min_y - self.grid_origin[1]) / self.grid_resolution
                        )))
                        max_j = min(rows, int(np.ceil(
                            (max_y - self.grid_origin[1]) / self.grid_resolution
                        )))
                        for c in range(min_i, max_i):
                            for r in range(min_j, max_j):
                                point = (
                                    self.grid_origin[0] + (c + 0.5) * self.grid_resolution,
                                    self.grid_origin[1] + (r + 0.5) * self.grid_resolution,
                                )
                                if self._point_in_polygon(point, polygon):
                                    self.grid_map[r, c] = 1
                    continue

                # Backward-compatible fallback for layouts without USD colliders.
                cx, cy = shelf['center']
                w, l = shelf['size'] # local dims
                rot = shelf.get('rot', 0)
                
                # 计算各种旋转角度下的最大包围盒
                cos_r = abs(np.cos(rot))
                sin_r = abs(np.sin(rot))
                max_w = w * cos_r + l * sin_r
                max_l = w * sin_r + l * cos_r
                
                # 确定搜索范围
                search_padding = 0.5
                min_i = int((cx - max_w/2 - search_padding - self.grid_origin[0]) / self.grid_resolution)
                max_i = int((cx + max_w/2 + search_padding - self.grid_origin[0]) / self.grid_resolution)
                min_j = int((cy - max_l/2 - search_padding - self.grid_origin[1]) / self.grid_resolution)
                max_j = int((cy + max_l/2 + search_padding - self.grid_origin[1]) / self.grid_resolution)
                
                min_i = max(0, min_i); max_i = min(cols, max_i)
                min_j = max(0, min_j); max_j = min(rows, max_j)

                for c in range(min_i, max_i):
                    for r in range(min_j, max_j):
                        # 网格中心坐标
                        wx = self.grid_origin[0] + (c + 0.5) * self.grid_resolution
                        wy = self.grid_origin[1] + (r + 0.5) * self.grid_resolution
                        
                        # 转局部坐标
                        dx = wx - cx
                        dy = wy - cy
                        lx = dx * np.cos(-rot) - dy * np.sin(-rot)
                        ly = dx * np.sin(-rot) + dy * np.cos(-rot)
                        
                        if abs(lx) < w/2 and abs(ly) < l/2:
                            self.grid_map[r, c] = 1 # 物理占据
                            
        except Exception as e:
            print(f"Error building grid: {e}")
            self.grid_map = np.zeros((50, 50), dtype=np.int8) # Fallback

    def _world_to_grid(self, pos):
        """世界坐标转网格坐标 (row, col)"""
        c = int((pos[0] - self.grid_origin[0]) / self.grid_resolution)
        r = int((pos[1] - self.grid_origin[1]) / self.grid_resolution)
        rows, cols = self.grid_dims
        return min(rows-1, max(0, r)), min(cols-1, max(0, c))

    def _is_world_point_navigable(self, point, clearance):
        """检查未截断的世界坐标是否满足边界和静态障碍物净空。"""
        clearance = max(0.0, float(clearance))
        bounds = self.warehouse_layout['bounds']
        x, y = float(point[0]), float(point[1])
        if not (
            bounds['x_min'] + clearance <= x <= bounds['x_max'] - clearance
            and bounds['y_min'] + clearance <= y <= bounds['y_max'] - clearance
        ):
            return False
        if self.grid_map is None:
            return True
        inflation_cells = int(np.ceil(clearance / self.grid_resolution))
        cell = self._world_to_grid((x, y))
        return self._is_line_safe(cell, cell, inflation_cells)

    def _grid_to_world(self, r, c):
        """网格坐标转世界坐标"""
        wx = self.grid_origin[0] + (c + 0.5) * self.grid_resolution
        wy = self.grid_origin[1] + (r + 0.5) * self.grid_resolution
        return (round(wx, 2), round(wy, 2))

    def _is_line_safe(self, start_grid, end_grid, inflation_cells):
        """检查网格坐标系下的两点连线是否安全（Bresenham算法变体）"""
        r0, c0 = start_grid
        r1, c1 = end_grid
        
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        
        r, c = r0, c0
        
        step_r = 1 if r0 < r1 else -1
        step_c = 1 if c0 < c1 else -1
        
        # Bresenham's algorithm error term
        err = (dr if dr > dc else -dc) / 2.0
        
        rows, cols = self.grid_dims
        
        while True:
            if not (0 <= r < rows and 0 <= c < cols):
                return False

            # Check collision
            if self.grid_map[r, c] == 1:
                return False
            
            # Check inflation (only if needed)
            if inflation_cells > 0:
                 if (
                     r - inflation_cells < 0
                     or r + inflation_cells >= rows
                     or c - inflation_cells < 0
                     or c + inflation_cells >= cols
                 ):
                     return False
                 r_min = max(0, r - inflation_cells)
                 r_max = min(rows, r + inflation_cells + 1)
                 c_min = max(0, c - inflation_cells)
                 c_max = min(cols, c + inflation_cells + 1)
                 # Check if any cell in the inflation window is occupied
                 if np.any(self.grid_map[r_min:r_max, c_min:c_max] == 1):
                     return False
            
            if r == r1 and c == c1:
                break
                
            e2 = err
            if e2 > -dr:
                err -= dc
                r += step_r
            if e2 < dc:
                err += dr
                c += step_c
                
        return True

    def _smooth_path(self, path, inflation_cells):
        """使用视线检查简化路径"""
        if len(path) <= 2: return path
        
        smoothed = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            last_valid_idx = current_idx + 1
            
            # 贪婪搜索：从终点往回找第一个可见点
            for i in range(len(path) - 1, current_idx, -1):
                p_start = path[current_idx]
                p_end = path[i]
                
                start_grid = self._world_to_grid(p_start)
                end_grid = self._world_to_grid(p_end)
                
                if self._is_line_safe(start_grid, end_grid, inflation_cells):
                    last_valid_idx = i
                    break
            
            smoothed.append(path[last_valid_idx])
            current_idx = last_valid_idx
            
        return smoothed

    def _generate_astar_path(self, start, end, inflation_m):
        """A* 寻路，支持动态膨胀半径和路径平滑"""
        if self.grid_map is None:
             return []

        if not self._is_world_point_navigable(start, inflation_m):
            print(f"No safe A* path: start {start} violates {inflation_m:.2f}m clearance")
            return []
        if not self._is_world_point_navigable(end, inflation_m):
            print(f"No safe A* path: end {end} violates {inflation_m:.2f}m clearance")
            return []

        start_r, start_c = self._world_to_grid(start)
        end_r, end_c = self._world_to_grid(end)
        
        # 2. 计算膨胀格子数
        inflation_cells = int(np.ceil(inflation_m / self.grid_resolution))

        if not self._is_line_safe((start_r, start_c), (start_r, start_c), inflation_cells):
            print(f"No safe A* path: start {start} violates {inflation_m:.2f}m clearance")
            return []
        if not self._is_line_safe((end_r, end_c), (end_r, end_c), inflation_cells):
            print(f"No safe A* path: end {end} violates {inflation_m:.2f}m clearance")
            return []
        
        # 3. A* Setup
        open_set = []
        heapq.heappush(open_set, (0, start_r, start_c))
        came_from = {}
        g_score = {(start_r, start_c): 0}
        
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        path_found = False
        visited = set()
        
        max_steps = 20000 
        steps = 0
        rows, cols = self.grid_dims
        
        while open_set and steps < max_steps:
            steps += 1
            _, cur_r, cur_c = heapq.heappop(open_set)
            
            if (cur_r, cur_c) == (end_r, end_c):
                path_found = True
                break
            
            if (cur_r, cur_c) in visited:
                continue
            visited.add((cur_r, cur_c))
            
            for dr, dc in dirs:
                nr, nc = cur_r + dr, cur_c + dc
                
                if 0 <= nr < rows and 0 <= nc < cols:
                    if self.grid_map[nr, nc] == 1:
                        continue 
                    
                    # 膨胀检测
                    too_close = False
                    if inflation_cells > 0:
                        r_min = max(0, nr-inflation_cells)
                        r_max = min(rows, nr+inflation_cells+1)
                        c_min = max(0, nc-inflation_cells)
                        c_max = min(cols, nc+inflation_cells+1)
                        if np.any(self.grid_map[r_min:r_max, c_min:c_max] == 1):
                            too_close = True
                    
                    if too_close:
                        continue
                        
                    move_cost = 1.414 if dr!=0 and dc!=0 else 1.0
                    new_g = g_score[(cur_r, cur_c)] + move_cost
                    
                    if (nr, nc) not in g_score or new_g < g_score[(nr, nc)]:
                        g_score[(nr, nc)] = new_g
                        # Diagonal distance heuristic
                        dx = abs(nr - end_r)
                        dy = abs(nc - end_c)
                        h = (dx + dy) + (1.414 - 2) * min(dx, dy)
                        heapq.heappush(open_set, (new_g + h, nr, nc))
                        came_from[(nr, nc)] = (cur_r, cur_c)
        
        if path_found:
            path = []
            curr = (end_r, end_c)
            while curr in came_from:
                path.append(self._grid_to_world(curr[0], curr[1]))
                curr = came_from[curr]
            path.append(self._grid_to_world(start_r, start_c))
            path.reverse()
            
            # 使用视线检查优化路径
            optimized_path = self._smooth_path(path, inflation_cells)
            return optimized_path
        else:
            print(f"No safe A* path found from {start} to {end}")
            return []

    def request_stop(self):
        """请求停止数据收集"""
        self._stop_requested = True
        print("Stop requested - will stop after current sequence")
        
    def reset_robot_to_start(self):
        """强制重置机器人到起始位置"""
        try:
            self.robot_controller.stop_navigation()
            
            if not self.robot_controller._is_initialized:
                self.robot_controller.initialize()
            
            robot = self.robot_controller._robot
            if robot is None:
                print("Warning: Robot not available for reset")
                return False
            
            x_idx = self.robot_controller._x_joint_idx
            y_idx = self.robot_controller._y_joint_idx
            rz_idx = self.robot_controller._rz_joint_idx
            
            current_positions = robot.get_joint_positions()
            if current_positions is None:
                return False
            
            new_positions = current_positions.copy()
            new_positions[x_idx] = float(self.reset_position[0])
            new_positions[y_idx] = float(self.reset_position[1])
            new_positions[rz_idx] = float(self.reset_position[2]) if len(self.reset_position) > 2 else 0.0
            
            robot.set_joint_positions(new_positions)
            
            zero_velocities = np.zeros_like(current_positions)
            robot.set_joint_velocities(zero_velocities)
            
            print(f"Robot reset to position: {self.reset_position}")
            return True
            
        except Exception as e:
            print(f"Failed to reset robot: {e}")
            return False

    async def _wait_for_reset_settle(self, max_checks=100, required_stable_checks=3):
        """Wait until reset pose and joint velocities are stable before collection."""
        stable_checks = 0
        for _ in range(max_checks):
            robot = self.robot_controller._robot
            if robot is None:
                return False
            positions = robot.get_joint_positions()
            velocities = robot.get_joint_velocities()
            if positions is not None and velocities is not None:
                x_idx = self.robot_controller._x_joint_idx
                y_idx = self.robot_controller._y_joint_idx
                rz_idx = self.robot_controller._rz_joint_idx
                target_yaw = float(self.reset_position[2]) if len(self.reset_position) > 2 else 0.0
                pose_error = max(
                    abs(float(positions[x_idx]) - float(self.reset_position[0])),
                    abs(float(positions[y_idx]) - float(self.reset_position[1])),
                    abs(float(positions[rz_idx]) - target_yaw),
                )
                velocity_norm = float(np.max(np.abs(velocities)))
                if pose_error <= 0.01 and velocity_norm <= 0.01:
                    stable_checks += 1
                    if stable_checks >= required_stable_checks:
                        return True
                else:
                    stable_checks = 0
            await asyncio.sleep(0.02)
        return False
            
    def load_waypoint_sequences_from_file(self, file_path):
        """从文件加载航点序列"""
        sequences = []
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            return sequences
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            if not content: return sequences
            
            try:
                data = json.loads(content)
                if isinstance(data, dict) and 'sequences' in data:
                    for seq in data['sequences']:
                        clean_seq = [(float(p[0]), float(p[1])) for p in seq]
                        if len(clean_seq) >= 2: sequences.append(clean_seq)
                elif isinstance(data, list):
                    if len(data) > 0 and isinstance(data[0], list):
                         for seq in data:
                                clean_seq = [(float(p[0]), float(p[1])) for p in seq]
                                if len(clean_seq) >= 2: sequences.append(clean_seq)
            except json.JSONDecodeError:
                # 简单文本格式支持
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('#'): continue
                    sequence = []
                    parts = line.replace('(', '').replace(')', '').split(';')
                    for part in parts:
                        if ',' in part:
                            coords = part.split(',')
                            if len(coords) >= 2:
                                try:
                                    x = float(coords[0].strip()); y = float(coords[1].strip())
                                    sequence.append((x, y))
                                except ValueError: continue
                    if len(sequence) >= 2: sequences.append(sequence)
            print(f"Loaded {len(sequences)} sequences from file")
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
        return sequences

    def _get_robot_dimensions(self):
        shape_info = (self.robot_config or {}).get('shape_info', {})
        return (
            max(0.05, float(shape_info.get('width', 0.8))),
            max(0.05, float(shape_info.get('length', 0.8))),
        )

    def _get_footprint_support_radius(self, world_direction, robot_yaw):
        """Project the configured footprint onto a world-space direction."""
        dx, dy = float(world_direction[0]), float(world_direction[1])
        direction_norm = math.hypot(dx, dy)
        if direction_norm <= 1e-12:
            return self._get_planning_radius()
        dx /= direction_norm
        dy /= direction_norm

        cos_yaw = math.cos(float(robot_yaw))
        sin_yaw = math.sin(float(robot_yaw))
        local_dx = cos_yaw * dx + sin_yaw * dy
        local_dy = -sin_yaw * dx + cos_yaw * dy
        shape_info = (self.robot_config or {}).get('shape_info', {})
        polygon = shape_info.get('polygon') or []
        projections = []
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                projections.append(
                    float(point[0]) * local_dx + float(point[1]) * local_dy
                )
            except (TypeError, ValueError):
                continue
        if projections:
            return max(0.0, max(projections))

        width, length = self._get_robot_dimensions()
        return 0.5 * math.hypot(length * local_dx, width * local_dy)

    def _get_collision_arrival_compensation(self):
        """Distance needed beyond first contact despite waypoint arrival tolerance."""
        arrival_threshold = 0.1
        controller = getattr(self, 'robot_controller', None)
        get_motion_config = getattr(controller, 'get_motion_config', None)
        if callable(get_motion_config):
            try:
                arrival_threshold = max(
                    0.0,
                    float(get_motion_config().get('arrival_threshold_m', 0.1)),
                )
            except (AttributeError, TypeError, ValueError):
                pass
        # Leave enough travel after first contact for three distinct PhysX reports.
        return arrival_threshold + 0.03

    @staticmethod
    def _extend_segment_endpoint(start, end, extension_distance):
        dx = float(end[0]) - float(start[0])
        dy = float(end[1]) - float(start[1])
        segment_length = math.hypot(dx, dy)
        if segment_length <= 1e-12:
            return tuple(end)
        scale = max(0.0, float(extension_distance)) / segment_length
        return (
            round(float(end[0]) + dx * scale, 3),
            round(float(end[1]) + dy * scale, 3),
        )

    @staticmethod
    def _shelf_local_to_world(shelf, local_x, local_y):
        cx, cy = shelf['center']
        rot = float(shelf.get('rot', 0.0))
        cos_r = math.cos(rot)
        sin_r = math.sin(rot)
        return (
            round(cx + local_x * cos_r - local_y * sin_r, 3),
            round(cy + local_x * sin_r + local_y * cos_r, 3),
        )

    @staticmethod
    def _world_to_shelf_local(shelf, point):
        cx, cy = shelf['center']
        rot = -float(shelf.get('rot', 0.0))
        dx = float(point[0]) - cx
        dy = float(point[1]) - cy
        return (
            dx * math.cos(rot) - dy * math.sin(rot),
            dx * math.sin(rot) + dy * math.cos(rot),
        )

    @classmethod
    def _segment_intersects_expanded_shelf(cls, start, end, shelf, clearance):
        """Return whether a centerline segment enters a shelf's expanded AABB."""
        x0, y0 = cls._world_to_shelf_local(shelf, start)
        x1, y1 = cls._world_to_shelf_local(shelf, end)
        half_width = shelf['size'][0] / 2.0 + max(0.0, float(clearance))
        half_length = shelf['size'][1] / 2.0 + max(0.0, float(clearance))
        dx = x1 - x0
        dy = y1 - y0
        t_min, t_max = 0.0, 1.0
        for origin, delta, lower, upper in (
            (x0, dx, -half_width, half_width),
            (y0, dy, -half_length, half_length),
        ):
            if abs(delta) <= 1e-12:
                if origin < lower or origin > upper:
                    return False
                continue
            t0 = (lower - origin) / delta
            t1 = (upper - origin) / delta
            if t0 > t1:
                t0, t1 = t1, t0
            t_min = max(t_min, t0)
            t_max = min(t_max, t1)
            if t_min > t_max:
                return False
        return True

    def _segment_footprint_intersects_shelf(self, start, end, shelf):
        """Check a translating footprint against a shelf for the segment heading."""
        dx = float(end[0]) - float(start[0])
        dy = float(end[1]) - float(start[1])
        if math.hypot(dx, dy) <= 1e-12:
            return False
        robot_yaw = math.atan2(dy, dx)
        shelf_yaw = float(shelf.get('rot', 0.0))
        normal = (math.cos(shelf_yaw), math.sin(shelf_yaw))
        tangent = (-math.sin(shelf_yaw), math.cos(shelf_yaw))
        normal_support = max(
            self._get_footprint_support_radius(normal, robot_yaw),
            self._get_footprint_support_radius((-normal[0], -normal[1]), robot_yaw),
        )
        tangent_support = max(
            self._get_footprint_support_radius(tangent, robot_yaw),
            self._get_footprint_support_radius((-tangent[0], -tangent[1]), robot_yaw),
        )

        x0, y0 = self._world_to_shelf_local(shelf, start)
        x1, y1 = self._world_to_shelf_local(shelf, end)
        half_width = shelf['size'][0] / 2.0 + normal_support
        half_length = shelf['size'][1] / 2.0 + tangent_support
        segment_dx = x1 - x0
        segment_dy = y1 - y0
        t_min, t_max = 0.0, 1.0
        for origin, delta, lower, upper in (
            (x0, segment_dx, -half_width, half_width),
            (y0, segment_dy, -half_length, half_length),
        ):
            if abs(delta) <= 1e-12:
                if origin < lower or origin > upper:
                    return False
                continue
            t0 = (lower - origin) / delta
            t1 = (upper - origin) / delta
            if t0 > t1:
                t0, t1 = t1, t0
            t_min = max(t_min, t0)
            t_max = min(t_max, t1)
            if t_min > t_max:
                return False
        return True

    @staticmethod
    def _convex_hull_2d(points):
        unique = sorted({
            (round(float(point[0]), 9), round(float(point[1]), 9))
            for point in points
        })
        if len(unique) <= 1:
            return unique

        def cross(origin, first, second):
            return (
                (first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0])
            )

        lower = []
        for point in unique:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 1e-12:
                lower.pop()
            lower.append(point)
        upper = []
        for point in reversed(unique):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 1e-12:
                upper.pop()
            upper.append(point)
        return lower[:-1] + upper[:-1]

    @staticmethod
    def _point_in_polygon(point, polygon):
        if len(polygon) < 3:
            return False
        x, y = float(point[0]), float(point[1])
        inside = False
        previous = polygon[-1]
        for current in polygon:
            x0, y0 = float(previous[0]), float(previous[1])
            x1, y1 = float(current[0]), float(current[1])
            cross = (x - x0) * (y1 - y0) - (y - y0) * (x1 - x0)
            if (
                abs(cross) <= 1e-9
                and min(x0, x1) - 1e-9 <= x <= max(x0, x1) + 1e-9
                and min(y0, y1) - 1e-9 <= y <= max(y0, y1) + 1e-9
            ):
                return True
            if (y0 > y) != (y1 > y):
                intersection_x = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
                if intersection_x >= x:
                    inside = not inside
            previous = current
        return inside

    @staticmethod
    def _line_segments_intersect(first_start, first_end, second_start, second_end):
        def orientation(a, b, c):
            return (
                (float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1]))
                - (float(b[1]) - float(a[1])) * (float(c[0]) - float(a[0]))
            )

        def on_segment(a, point, b):
            return (
                min(a[0], b[0]) - 1e-9 <= point[0] <= max(a[0], b[0]) + 1e-9
                and min(a[1], b[1]) - 1e-9 <= point[1] <= max(a[1], b[1]) + 1e-9
            )

        o1 = orientation(first_start, first_end, second_start)
        o2 = orientation(first_start, first_end, second_end)
        o3 = orientation(second_start, second_end, first_start)
        o4 = orientation(second_start, second_end, first_end)
        if o1 * o2 < 0.0 and o3 * o4 < 0.0:
            return True
        return (
            (abs(o1) <= 1e-9 and on_segment(first_start, second_start, first_end))
            or (abs(o2) <= 1e-9 and on_segment(first_start, second_end, first_end))
            or (abs(o3) <= 1e-9 and on_segment(second_start, first_start, second_end))
            or (abs(o4) <= 1e-9 and on_segment(second_start, first_end, second_end))
        )

    @classmethod
    def _segment_intersects_polygon(cls, start, end, polygon):
        if len(polygon) < 3:
            return False
        if cls._point_in_polygon(start, polygon) or cls._point_in_polygon(end, polygon):
            return True
        return any(
            cls._line_segments_intersect(
                start, end, polygon[index], polygon[(index + 1) % len(polygon)]
            )
            for index in range(len(polygon))
        )

    def _segment_footprint_intersects_polygon(self, start, end, polygon):
        dx = float(end[0]) - float(start[0])
        dy = float(end[1]) - float(start[1])
        if math.hypot(dx, dy) <= 1e-12 or len(polygon) < 3:
            return False
        yaw = math.atan2(dy, dx)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        robot_polygon = (
            (self.robot_config or {}).get('shape_info', {}).get('polygon') or []
        )
        rotated_footprint = []
        for point in robot_polygon:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            local_x, local_y = float(point[0]), float(point[1])
            rotated_footprint.append((
                local_x * cos_yaw - local_y * sin_yaw,
                local_x * sin_yaw + local_y * cos_yaw,
            ))
        if len(rotated_footprint) < 3:
            radius = self._get_planning_radius()
            rotated_footprint = [
                (
                    radius * math.cos(2.0 * math.pi * index / 16.0),
                    radius * math.sin(2.0 * math.pi * index / 16.0),
                )
                for index in range(16)
            ]
        expanded = self._convex_hull_2d([
            (float(obstacle[0]) - footprint[0], float(obstacle[1]) - footprint[1])
            for obstacle in polygon
            for footprint in rotated_footprint
        ])
        return self._segment_intersects_polygon(start, end, expanded)

    @staticmethod
    def _shelf_collision_aabbs(shelf):
        geometry = shelf.get('collision_geometry') or {}
        aabbs = []
        for collider in geometry.get('colliders') or []:
            minimum = collider.get('aabb_min_m')
            maximum = collider.get('aabb_max_m')
            if not (
                isinstance(minimum, (list, tuple))
                and isinstance(maximum, (list, tuple))
                and len(minimum) >= 2
                and len(maximum) >= 2
            ):
                continue
            aabbs.append((
                float(minimum[0]), float(minimum[1]),
                float(maximum[0]), float(maximum[1]),
                collider.get('prim_path'),
            ))
        return aabbs

    def _shelf_collision_local_bounds(self, shelf):
        local_polygons = self._shelf_collision_local_polygons(shelf)
        if not local_polygons:
            half_width = shelf['size'][0] / 2.0
            half_length = shelf['size'][1] / 2.0
            return {
                'x_min': -half_width, 'x_max': half_width,
                'y_min': -half_length, 'y_max': half_length,
                'source': 'fallback_layout_geometry',
                'collider_count': 0,
            }
        geometry = shelf.get('collision_geometry') or {}
        source = geometry.get('source')
        if not source:
            has_exact_polygon = any(
                collider.get('polygon_m') or collider.get('shelf_local_polygon_m')
                for collider in geometry.get('colliders') or []
            )
            source = (
                'usd_collision_api_world_polygon'
                if has_exact_polygon else 'usd_collision_api_world_aabb'
            )
        return {
            'x_min': min(point[0] for polygon in local_polygons for point in polygon),
            'x_max': max(point[0] for polygon in local_polygons for point in polygon),
            'y_min': min(point[1] for polygon in local_polygons for point in polygon),
            'y_max': max(point[1] for polygon in local_polygons for point in polygon),
            'source': source,
            'collider_count': len(local_polygons),
        }

    def _shelf_collision_local_polygons(self, shelf):
        geometry = shelf.get('collision_geometry') or {}
        polygons = []
        for collider in geometry.get('colliders') or []:
            polygon = collider.get('shelf_local_polygon_m')
            parsed = []
            if isinstance(polygon, list):
                for point in polygon:
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        parsed.append((float(point[0]), float(point[1])))
            if len(parsed) >= 3:
                polygons.append(parsed)
                continue
            world_polygon = collider.get('polygon_m')
            parsed_world = []
            if isinstance(world_polygon, list):
                for point in world_polygon:
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        parsed_world.append((float(point[0]), float(point[1])))
            if len(parsed_world) >= 3:
                polygons.append([
                    self._world_to_shelf_local(shelf, point)
                    for point in parsed_world
                ])
                continue
            minimum = collider.get('shelf_local_aabb_min_m')
            maximum = collider.get('shelf_local_aabb_max_m')
            if (
                isinstance(minimum, (list, tuple))
                and isinstance(maximum, (list, tuple))
                and len(minimum) >= 2
                and len(maximum) >= 2
            ):
                polygons.append([
                    (float(minimum[0]), float(minimum[1])),
                    (float(maximum[0]), float(minimum[1])),
                    (float(maximum[0]), float(maximum[1])),
                    (float(minimum[0]), float(maximum[1])),
                ])
                continue
            world_minimum = collider.get('aabb_min_m')
            world_maximum = collider.get('aabb_max_m')
            if (
                isinstance(world_minimum, (list, tuple))
                and isinstance(world_maximum, (list, tuple))
                and len(world_minimum) >= 2
                and len(world_maximum) >= 2
            ):
                polygons.append([
                    self._world_to_shelf_local(shelf, point)
                    for point in (
                        (float(world_minimum[0]), float(world_minimum[1])),
                        (float(world_maximum[0]), float(world_minimum[1])),
                        (float(world_maximum[0]), float(world_maximum[1])),
                        (float(world_minimum[0]), float(world_maximum[1])),
                    )
                ])
        return polygons

    def _shelf_collision_world_polygons(self, shelf):
        geometry = shelf.get('collision_geometry') or {}
        polygons = []
        for collider in geometry.get('colliders') or []:
            polygon = collider.get('polygon_m')
            parsed = []
            if isinstance(polygon, list):
                for point in polygon:
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        parsed.append((float(point[0]), float(point[1])))
            if len(parsed) >= 3:
                polygons.append(parsed)
        if polygons:
            return polygons
        return [
            [
                (box[0], box[1]), (box[2], box[1]),
                (box[2], box[3]), (box[0], box[3]),
            ]
            for box in self._shelf_collision_aabbs(shelf)
        ]

    def _shelf_collision_local_boxes(self, shelf):
        geometry = shelf.get('collision_geometry') or {}
        boxes = []
        for collider in geometry.get('colliders') or []:
            minimum = collider.get('shelf_local_aabb_min_m')
            maximum = collider.get('shelf_local_aabb_max_m')
            if (
                isinstance(minimum, (list, tuple))
                and isinstance(maximum, (list, tuple))
                and len(minimum) >= 2
                and len(maximum) >= 2
            ):
                boxes.append((
                    float(minimum[0]), float(minimum[1]),
                    float(maximum[0]), float(maximum[1]),
                    collider.get('prim_path'),
                ))
                continue
            world_minimum = collider.get('aabb_min_m')
            world_maximum = collider.get('aabb_max_m')
            if not (
                isinstance(world_minimum, (list, tuple))
                and isinstance(world_maximum, (list, tuple))
                and len(world_minimum) >= 2
                and len(world_maximum) >= 2
            ):
                continue
            local_points = [
                self._world_to_shelf_local(shelf, point)
                for point in (
                    (world_minimum[0], world_minimum[1]),
                    (world_minimum[0], world_maximum[1]),
                    (world_maximum[0], world_minimum[1]),
                    (world_maximum[0], world_maximum[1]),
                )
            ]
            boxes.append((
                min(point[0] for point in local_points),
                min(point[1] for point in local_points),
                max(point[0] for point in local_points),
                max(point[1] for point in local_points),
                collider.get('prim_path'),
            ))
        return boxes

    def _collision_box_as_shelf(self, shelf, box):
        x_min, y_min, x_max, y_max, _ = box
        center = self._shelf_local_to_world(
            shelf, (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
        )
        return {
            'center': center,
            'size': (x_max - x_min, y_max - y_min),
            'rot': float(shelf.get('rot', 0.0)),
        }

    def _segment_footprint_intersects_collision_geometry(self, start, end, shelf):
        polygons = self._shelf_collision_world_polygons(shelf)
        if not polygons:
            return self._segment_footprint_intersects_shelf(start, end, shelf)
        for polygon in polygons:
            if self._segment_footprint_intersects_polygon(start, end, polygon):
                return True
        return False

    def _segment_centerline_intersects_collision_geometry(self, start, end, shelf):
        polygons = self._shelf_collision_world_polygons(shelf)
        if not polygons:
            return self._segment_intersects_expanded_shelf(start, end, shelf, 0.0)
        for polygon in polygons:
            if self._segment_intersects_polygon(start, end, polygon):
                return True
        return False

    def _fault_segment_targets_only(self, points, target_shelf, footprint_radius):
        """Require body-level contact with the target without centerline penetration."""
        segments = list(zip(points, points[1:]))
        if not segments:
            return False
        if not any(
            self._segment_footprint_intersects_collision_geometry(a, b, target_shelf)
            for a, b in segments
        ):
            return False
        if any(
            self._segment_centerline_intersects_collision_geometry(a, b, target_shelf)
            for a, b in segments
        ):
            return False
        for other_shelf in self.warehouse_layout.get('shelves', []):
            if other_shelf is target_shelf:
                continue
            if any(
                self._segment_footprint_intersects_collision_geometry(a, b, other_shelf)
                for a, b in segments
            ):
                return False
        bounds = self.warehouse_layout['bounds']
        return all(
            bounds['x_min'] + footprint_radius <= point[0] <= bounds['x_max'] - footprint_radius
            and bounds['y_min'] + footprint_radius <= point[1] <= bounds['y_max'] - footprint_radius
            for point in points
        )

    @staticmethod
    def _append_segment(sequence, segment):
        if not segment:
            return
        start_index = 1 if sequence and np.hypot(
            segment[0][0] - sequence[-1][0], segment[0][1] - sequence[-1][1]
        ) <= 0.1 else 0
        sequence.extend(segment[start_index:])

    @staticmethod
    def _interpolate_transition(start, end, waypoint_count):
        """Return a gradual transition whose final waypoint is the contact target."""
        waypoint_count = max(1, int(waypoint_count))
        return [
            (
                round(start[0] + (end[0] - start[0]) * index / waypoint_count, 3),
                round(start[1] + (end[1] - start[1]) * index / waypoint_count, 3),
            )
            for index in range(1, waypoint_count + 1)
        ]

    def _segments_clear_of_shelves(self, points, footprint_radius):
        segments = list(zip(points, points[1:]))
        return not any(
            self._segment_intersects_expanded_shelf(
                segment_start, segment_end, shelf, footprint_radius
            )
            for shelf in self.warehouse_layout.get('shelves', [])
            for segment_start, segment_end in segments
        )

    def _generate_safe_sequence(self, footprint_radius, target_waypoint_count):
        path_inflation = footprint_radius
        pickup_margin = footprint_radius + 0.25
        self._calculate_pickup_points(margin=pickup_margin, points_per_side=4)
        pickup_pool = self.warehouse_layout['pickup_points']
        if not pickup_pool:
            return None, None

        num_targets = min(self._rng.randint(5, 10), len(pickup_pool))
        targets = self._rng.sample(pickup_pool, num_targets)
        ordered_targets = []
        current_pos = self.warehouse_layout['start_zone']
        while targets:
            nearest = min(
                targets,
                key=lambda item: np.hypot(
                    item['position'][0] - current_pos[0],
                    item['position'][1] - current_pos[1],
                ),
            )
            ordered_targets.append(nearest)
            current_pos = nearest['position']
            targets.remove(nearest)

        sequence = [self.warehouse_layout['start_zone']]
        current_pos = sequence[0]
        for target in ordered_targets:
            segment = self._generate_astar_path(
                current_pos, target['position'], inflation_m=path_inflation
            )
            if not segment:
                return None, None
            self._append_segment(sequence, segment)
            current_pos = target['position']

        sequence = self._remove_duplicate_waypoints(sequence)
        sequence = self._densify_path_to_count(sequence, target_waypoint_count)
        if not sequence:
            return None, None
        final_target = ordered_targets[-1]
        sequence[-1] = tuple(final_target['position'])
        return sequence, {
            'scenario_type': 'safe',
            'expected_collision': False,
            'target_obstacle_id': None,
            'injection': None,
            'safe_profile': 'conservative_astar',
            'requested_waypoint_count': target_waypoint_count,
            'final_task_type': 'package_pickup',
            'final_pickup_shelf_id': final_target['shelf_id'],
            'final_pickup_position': list(final_target['position']),
            'path_inflation_m': round(path_inflation, 4),
            'footprint_planning_radius_m': round(footprint_radius, 4),
        }

    def _generate_hard_safe_sequence(
        self, footprint_radius, target_waypoint_count
    ):
        """Generate an executable near-miss route that remains footprint-safe."""
        bounds = self.warehouse_layout['bounds']
        shelves = self.warehouse_layout.get('shelves', [])
        if target_waypoint_count < 8:
            return None, None
        margin_min = self._generation_randomization['hard_safe_margin_min_m']
        margin_max = self._generation_randomization['hard_safe_margin_max_m']

        for _ in range(80):
            margin = self._rng.uniform(margin_min, margin_max)
            clearance = footprint_radius + margin
            near_miss_type = self._rng.choice(
                ('shelf_side_near_miss', 'wall_near_miss')
            )
            target_id = None

            if near_miss_type == 'shelf_side_near_miss' and shelves:
                shelf = self._rng.choice(shelves)
                half_width = shelf['size'][0] / 2.0
                half_length = shelf['size'][1] / 2.0
                side = self._rng.choice((-1.0, 1.0))
                direction = self._rng.choice((-1.0, 1.0))
                travel = self._rng.uniform(0.6, 1.5)
                tangent_limit = max(0.3, half_length - 0.8)
                tangent_start = self._rng.uniform(-tangent_limit, tangent_limit)
                tangent_end = max(
                    -tangent_limit,
                    min(tangent_limit, tangent_start + direction * travel),
                )
                if abs(tangent_end - tangent_start) < 0.35:
                    continue
                approach = self._shelf_local_to_world(
                    shelf, side * (half_width + clearance), tangent_start
                )
                near_end = self._shelf_local_to_world(
                    shelf, side * (half_width + clearance), tangent_end
                )
                target_id = shelf['id']
                near_miss_details = {
                    'near_miss_type': near_miss_type,
                    'near_obstacle_id': target_id,
                    'planned_surface_margin_m': round(margin, 4),
                    'parallel_travel_m': round(abs(tangent_end - tangent_start), 4),
                    'shelf_side': 'left' if side < 0 else 'right',
                }
            else:
                wall = self._rng.choice(('x_min', 'x_max', 'y_min', 'y_max'))
                direction = self._rng.choice((-1.0, 1.0))
                travel = self._rng.uniform(0.6, 1.5)
                if wall.startswith('x_'):
                    inward = 1.0 if wall == 'x_min' else -1.0
                    boundary = bounds[wall]
                    tangent_min = bounds['y_min'] + footprint_radius + 0.5
                    tangent_max = bounds['y_max'] - footprint_radius - 0.5
                    tangent_start = self._rng.uniform(tangent_min, tangent_max)
                    tangent_end = max(
                        tangent_min,
                        min(tangent_max, tangent_start + direction * travel),
                    )
                    approach = (boundary + inward * clearance, tangent_start)
                    near_end = (boundary + inward * clearance, tangent_end)
                else:
                    inward = 1.0 if wall == 'y_min' else -1.0
                    boundary = bounds[wall]
                    tangent_min = bounds['x_min'] + footprint_radius + 0.5
                    tangent_max = bounds['x_max'] - footprint_radius - 0.5
                    tangent_start = self._rng.uniform(tangent_min, tangent_max)
                    tangent_end = max(
                        tangent_min,
                        min(tangent_max, tangent_start + direction * travel),
                    )
                    approach = (tangent_start, boundary + inward * clearance)
                    near_end = (tangent_end, boundary + inward * clearance)
                if abs(tangent_end - tangent_start) < 0.35:
                    continue
                approach = tuple(round(value, 3) for value in approach)
                near_end = tuple(round(value, 3) for value in near_end)
                target_id = f'warehouse_wall_{wall}'
                near_miss_details = {
                    'near_miss_type': near_miss_type,
                    'near_obstacle_id': target_id,
                    'planned_surface_margin_m': round(margin, 4),
                    'parallel_travel_m': round(abs(tangent_end - tangent_start), 4),
                    'wall': wall,
                }

            if not self._segments_clear_of_shelves(
                [approach, near_end], footprint_radius
            ):
                continue
            if not all(
                bounds['x_min'] + footprint_radius <= point[0] <= bounds['x_max'] - footprint_radius
                and bounds['y_min'] + footprint_radius <= point[1] <= bounds['y_max'] - footprint_radius
                for point in (approach, near_end)
            ):
                continue

            prefix_count = self._rng.randint(4, target_waypoint_count - 3)
            prefix, completed_pickups = self._generate_normal_pickup_prefix(
                approach, prefix_count, footprint_radius
            )
            if not prefix:
                continue
            remaining_count = target_waypoint_count - len(prefix) - 1
            suffix, final_target = self._generate_task_suffix(
                near_end, remaining_count + 1, footprint_radius
            )
            if not suffix:
                continue
            sequence = [*prefix, near_end, *suffix[1:]]
            if len(sequence) != target_waypoint_count:
                continue
            return sequence, {
                'scenario_type': 'safe',
                'expected_collision': False,
                'target_obstacle_id': None,
                'injection': None,
                'safe_profile': 'hard_near_miss',
                'near_miss': near_miss_details,
                'near_miss_waypoint_index': len(prefix),
                'completed_pickups_before_near_miss': completed_pickups,
                'requested_waypoint_count': target_waypoint_count,
                'final_task_type': 'package_pickup',
                'final_pickup_shelf_id': final_target['shelf_id'],
                'final_pickup_position': list(final_target['position']),
                'path_inflation_m': round(footprint_radius, 4),
                'footprint_planning_radius_m': round(footprint_radius, 4),
            }
        return None, None

    @staticmethod
    def _densify_path_to_count(path, target_count):
        """Add waypoints on existing safe segments without changing the polyline."""
        points = [tuple(point) for point in path]
        target_count = int(target_count)
        if len(points) < 2 or len(points) > target_count or target_count < 2:
            return None
        while len(points) < target_count:
            segment_index = max(
                range(len(points) - 1),
                key=lambda index: math.hypot(
                    points[index + 1][0] - points[index][0],
                    points[index + 1][1] - points[index][1],
                ),
            )
            start = points[segment_index]
            end = points[segment_index + 1]
            midpoint = (
                round((start[0] + end[0]) / 2.0, 3),
                round((start[1] + end[1]) / 2.0, 3),
            )
            points.insert(segment_index + 1, midpoint)
        return points

    def _generate_task_suffix(self, nominal_start, target_count, footprint_radius):
        """Generate a safe unexecuted mission suffix ending at a package pickup."""
        if target_count < 2:
            return None, None
        self._calculate_pickup_points(
            margin=footprint_radius + 0.25, points_per_side=4
        )
        candidates = list(self.warehouse_layout.get('pickup_points', []))
        self._rng.shuffle(candidates)
        for target in candidates:
            path = self._generate_astar_path(
                nominal_start, target['position'], inflation_m=footprint_radius
            )
            if not path:
                continue
            suffix = self._densify_path_to_count(
                self._remove_duplicate_waypoints(path), target_count
            )
            if suffix:
                suffix[-1] = tuple(target['position'])
                return suffix, target
        return None, None

    def _generate_normal_pickup_prefix(
        self, nominal_end, target_count, footprint_radius
    ):
        """Build a normal multi-pick route before entering the injected fault."""
        if target_count < 2:
            return None, None
        max_prior_pickups = min(3, max(0, (target_count - 2) // 2))
        min_prior_pickups = 1 if max_prior_pickups > 0 else 0
        requested_pickups = self._rng.randint(
            min_prior_pickups, max_prior_pickups
        )
        self._calculate_pickup_points(
            margin=footprint_radius + 0.25, points_per_side=4
        )
        candidates = list(self.warehouse_layout.get('pickup_points', []))
        self._rng.shuffle(candidates)
        sequence = [self.warehouse_layout['start_zone']]
        current_pos = sequence[0]
        completed_pickups = []

        while candidates and len(completed_pickups) < requested_pickups:
            target = candidates.pop()
            segment = self._generate_astar_path(
                current_pos, target['position'], inflation_m=footprint_radius
            )
            if not segment:
                continue
            segment[-1] = tuple(target['position'])
            self._append_segment(sequence, segment)
            current_pos = target['position']
            completed_pickups.append({
                'shelf_id': target['shelf_id'],
                'position': list(target['position']),
            })

        if len(completed_pickups) != requested_pickups:
            return None, None
        final_segment = self._generate_astar_path(
            current_pos, nominal_end, inflation_m=footprint_radius
        )
        if not final_segment:
            return None, None
        final_segment[-1] = tuple(nominal_end)
        self._append_segment(sequence, final_segment)
        sequence = self._densify_path_to_count(sequence, target_count)
        if not sequence:
            return None, None
        sequence[-1] = tuple(nominal_end)
        return sequence, completed_pickups

    def _generate_shelf_fault_sequence(
        self, scenario_type, footprint_radius, planned_fault_waypoint_index,
        target_waypoint_count,
    ):
        shelves = self.warehouse_layout.get('shelves', [])
        if not shelves:
            return None, None
        robot_width, _ = self._get_robot_dimensions()
        safe_clearance = footprint_radius + 0.25
        path_inflation = footprint_radius
        arrival_compensation = self._get_collision_arrival_compensation()
        transition_min = self._generation_randomization['transition_waypoints_min']
        transition_max = self._generation_randomization['transition_waypoints_max']
        penetration_min = self._generation_randomization['penetration_min_m']
        penetration_max = self._generation_randomization['penetration_max_m']

        for _ in range(40):
            shelf = self._rng.choice(shelves)
            collision_bounds = self._shelf_collision_local_bounds(shelf)
            x_min = collision_bounds['x_min']
            x_max = collision_bounds['x_max']
            y_min = collision_bounds['y_min']
            y_max = collision_bounds['y_max']
            half_length = (y_max - y_min) / 2.0
            tangent_center = (y_min + y_max) / 2.0
            side = self._rng.choice((-1.0, 1.0))
            surface_x = x_min if side < 0 else x_max

            if scenario_type == 'lateral_localization_offset':
                tangent_limit = max(0.3, half_length - 0.8)
                tangent_start = self._rng.uniform(
                    tangent_center - tangent_limit,
                    tangent_center + tangent_limit,
                )
                direction = self._rng.choice((-1.0, 1.0))
                requested_travel = self._rng.uniform(0.7, 1.8)
                tangent_contact = max(
                    tangent_center - tangent_limit,
                    min(
                        tangent_center + tangent_limit,
                        tangent_start + direction * requested_travel,
                    ),
                )
                if abs(tangent_contact - tangent_start) < 0.4:
                    continue
                penetration = self._rng.uniform(penetration_min, penetration_max)
                contact_clearance = max(0.03, robot_width / 2.0 - penetration)
                approach = self._shelf_local_to_world(
                    shelf, surface_x + side * safe_clearance, tangent_start
                )
                contact_target = self._shelf_local_to_world(
                    shelf,
                    surface_x + side * contact_clearance,
                    tangent_contact,
                )
                transition_count = self._rng.randint(transition_min, transition_max)
                injected = self._interpolate_transition(
                    approach, contact_target, transition_count
                )
                continuation_travel = self._rng.uniform(0.3, 0.8)
                tangent_end = max(
                    tangent_center - tangent_limit,
                    min(
                        tangent_center + tangent_limit,
                        tangent_contact + direction * continuation_travel,
                    ),
                )
                if abs(tangent_end - tangent_contact) >= 0.15:
                    injected.append(self._shelf_local_to_world(
                        shelf,
                        surface_x + side * contact_clearance,
                        tangent_end,
                    ))
                injection = {
                    'lateral_offset_m': round(safe_clearance - contact_clearance, 4),
                    'collision_penetration_m': round(penetration, 4),
                    'executed_surface_clearance_m': round(contact_clearance, 4),
                    'parallel_travel_before_contact_m': round(
                        abs(tangent_contact - tangent_start), 4
                    ),
                    'parallel_travel_total_m': round(abs(tangent_end - tangent_start), 4),
                    'shelf_side': 'left' if side < 0 else 'right',
                    'fault_transition_waypoint_count': transition_count,
                    'planned_contact_offset_in_fault_segment': transition_count - 1,
                }
            elif scenario_type == 'docking_overtravel':
                tangent = self._rng.uniform(
                    y_min + min(1.0, max(0.0, half_length - 0.2)),
                    y_max - min(1.0, max(0.0, half_length - 0.2)),
                )
                penetration = self._rng.uniform(penetration_min, penetration_max)
                approach = self._shelf_local_to_world(
                    shelf, surface_x + side * safe_clearance, tangent
                )
                nominal_contact = self._shelf_local_to_world(
                    shelf, surface_x, tangent
                )
                motion_direction = (
                    nominal_contact[0] - approach[0],
                    nominal_contact[1] - approach[1],
                )
                robot_yaw = math.atan2(motion_direction[1], motion_direction[0])
                contact_support = self._get_footprint_support_radius(
                    motion_direction, robot_yaw
                )
                contact_clearance = max(
                    0.03,
                    contact_support - penetration - arrival_compensation,
                )
                contact_target = self._shelf_local_to_world(
                    shelf, surface_x + side * contact_clearance, tangent
                )
                transition_count = self._rng.randint(transition_min, transition_max)
                injected = self._interpolate_transition(
                    approach, contact_target, transition_count
                )
                injection = {
                    'overtravel_m': round(safe_clearance - contact_clearance, 4),
                    'collision_penetration_m': round(penetration, 4),
                    'target_surface_clearance_m': round(contact_clearance, 4),
                    'footprint_support_radius_m': round(contact_support, 4),
                    'arrival_compensation_m': round(arrival_compensation, 4),
                    'collision_geometry_source': collision_bounds['source'],
                    'target_collider_count': collision_bounds['collider_count'],
                    'shelf_side': 'left' if side < 0 else 'right',
                    'fault_transition_waypoint_count': transition_count,
                    'planned_contact_offset_in_fault_segment': transition_count - 1,
                }
            elif scenario_type == 'shelf_corner_cutting':
                end = self._rng.choice((-1.0, 1.0))
                surface_y = y_min if end < 0 else y_max
                entry_backoff = self._rng.uniform(0.5, 1.5)
                approach = self._shelf_local_to_world(
                    shelf,
                    surface_x + side * safe_clearance,
                    surface_y - end * entry_backoff,
                )
                normal_clearance = self._rng.uniform(
                    self._generation_randomization['corner_clearance_min_m'],
                    self._generation_randomization['corner_clearance_max_m'],
                )
                tangent_clearance = self._rng.uniform(
                    self._generation_randomization['corner_clearance_min_m'],
                    self._generation_randomization['corner_clearance_max_m'],
                )
                exit_distance = self._rng.uniform(0.3, 0.9)
                nominal_contact_target = self._shelf_local_to_world(
                    shelf,
                    surface_x + side * normal_clearance,
                    surface_y + end * tangent_clearance,
                )
                contact_target = self._extend_segment_endpoint(
                    approach,
                    nominal_contact_target,
                    arrival_compensation,
                )
                transition_count = self._rng.randint(transition_min, transition_max)
                injected = self._interpolate_transition(
                    approach, contact_target, transition_count
                )
                injected.append(self._shelf_local_to_world(
                    shelf,
                    surface_x + side * normal_clearance,
                    surface_y + end * exit_distance,
                ))
                local_normal_delta = safe_clearance - normal_clearance
                local_tangent_delta = entry_backoff + tangent_clearance
                entry_angle_deg = math.degrees(math.atan2(
                    abs(local_normal_delta), abs(local_tangent_delta)
                ))
                corner_radial_clearance = math.hypot(
                    normal_clearance, tangent_clearance
                )
                injection = {
                    'corner_normal_clearance_m': round(normal_clearance, 4),
                    'corner_tangent_clearance_m': round(tangent_clearance, 4),
                    'corner_radial_clearance_m': round(corner_radial_clearance, 4),
                    'corner_cut_depth_m': round(
                        max(0.0, footprint_radius - corner_radial_clearance), 4
                    ),
                    'corner_exit_distance_m': round(exit_distance, 4),
                    'entry_backoff_m': round(entry_backoff, 4),
                    'entry_angle_from_shelf_tangent_deg': round(entry_angle_deg, 3),
                    'arrival_compensation_m': round(arrival_compensation, 4),
                    'nominal_contact_position': list(nominal_contact_target),
                    'compensated_contact_position': list(contact_target),
                    'collision_geometry_source': collision_bounds['source'],
                    'target_collider_count': collision_bounds['collider_count'],
                    'shelf_side': 'left' if side < 0 else 'right',
                    'shelf_end': 'lower' if end < 0 else 'upper',
                    'fault_transition_waypoint_count': transition_count,
                    'planned_contact_offset_in_fault_segment': transition_count - 1,
                }
            else:
                return None, None

            injection.setdefault(
                'collision_geometry_source', collision_bounds['source']
            )
            injection.setdefault(
                'target_collider_count', collision_bounds['collider_count']
            )

            if not self._fault_segment_targets_only(
                [approach, *injected], shelf, footprint_radius
            ):
                continue
            sequence, completed_pickups = self._generate_normal_pickup_prefix(
                approach,
                planned_fault_waypoint_index,
                footprint_radius,
            )
            if not sequence:
                continue
            suffix_count = (
                target_waypoint_count
                - planned_fault_waypoint_index
                - len(injected)
            )
            suffix, final_target = self._generate_task_suffix(
                approach, suffix_count, footprint_radius
            )
            if not suffix:
                continue
            sequence.extend(injected)
            sequence.extend(suffix)
            if len(sequence) == target_waypoint_count:
                return sequence, {
                    'scenario_type': scenario_type,
                    'expected_collision': True,
                    'target_obstacle_id': shelf['id'],
                    'planned_fault_waypoint_index': planned_fault_waypoint_index,
                    'planned_contact_waypoint_index': (
                        planned_fault_waypoint_index
                        + injection['planned_contact_offset_in_fault_segment']
                    ),
                    'completed_pickups_before_fault': completed_pickups,
                    'requested_waypoint_count': target_waypoint_count,
                    'final_task_type': 'package_pickup',
                    'final_pickup_shelf_id': final_target['shelf_id'],
                    'final_pickup_position': list(final_target['position']),
                    'injection': injection,
                    'path_inflation_m': round(path_inflation, 4),
                    'footprint_planning_radius_m': round(footprint_radius, 4),
                }
        return None, None

    def _generate_wall_fault_sequence(
        self, footprint_radius, planned_fault_waypoint_index,
        target_waypoint_count,
    ):
        bounds = self.warehouse_layout['bounds']
        robot_width, _ = self._get_robot_dimensions()
        safe_clearance = footprint_radius + 0.25
        path_inflation = footprint_radius
        walls = ('x_min', 'x_max', 'y_min', 'y_max')
        transition_min = self._generation_randomization['transition_waypoints_min']
        transition_max = self._generation_randomization['transition_waypoints_max']
        penetration_min = self._generation_randomization['penetration_min_m']
        penetration_max = self._generation_randomization['penetration_max_m']

        for _ in range(60):
            wall = self._rng.choice(walls)
            penetration = self._rng.uniform(penetration_min, penetration_max)
            contact_clearance = max(0.03, robot_width / 2.0 - penetration)
            transition_count = self._rng.randint(transition_min, transition_max)
            requested_travel = self._rng.uniform(0.7, 1.8)
            if wall.startswith('x_'):
                inward = 1.0 if wall == 'x_min' else -1.0
                boundary = bounds[wall]
                tangent_min = bounds['y_min'] + 1.0
                tangent_max = bounds['y_max'] - 1.0
                tangent_start = self._rng.uniform(tangent_min, tangent_max)
                direction = self._rng.choice((-1.0, 1.0))
                tangent_contact = max(
                    tangent_min,
                    min(tangent_max, tangent_start + direction * requested_travel),
                )
                if abs(tangent_contact - tangent_start) < 0.4:
                    continue
                approach = (boundary + inward * safe_clearance, tangent_start)
                contact_target = (
                    boundary + inward * contact_clearance, tangent_contact
                )
            else:
                inward = 1.0 if wall == 'y_min' else -1.0
                boundary = bounds[wall]
                tangent_min = bounds['x_min'] + 1.0
                tangent_max = bounds['x_max'] - 1.0
                tangent_start = self._rng.uniform(tangent_min, tangent_max)
                direction = self._rng.choice((-1.0, 1.0))
                tangent_contact = max(
                    tangent_min,
                    min(tangent_max, tangent_start + direction * requested_travel),
                )
                if abs(tangent_contact - tangent_start) < 0.4:
                    continue
                approach = (tangent_start, boundary + inward * safe_clearance)
                contact_target = (
                    tangent_contact, boundary + inward * contact_clearance
                )

            approach = tuple(round(value, 3) for value in approach)
            contact_target = tuple(round(value, 3) for value in contact_target)
            injected = self._interpolate_transition(
                approach, contact_target, transition_count
            )
            continuation_travel = self._rng.uniform(0.3, 0.8)
            tangent_end = max(
                tangent_min,
                min(tangent_max, tangent_contact + direction * continuation_travel),
            )
            if abs(tangent_end - tangent_contact) >= 0.15:
                if wall.startswith('x_'):
                    injected.append((
                        round(boundary + inward * contact_clearance, 3),
                        round(tangent_end, 3),
                    ))
                else:
                    injected.append((
                        round(tangent_end, 3),
                        round(boundary + inward * contact_clearance, 3),
                    ))
            if any(
                not self._is_line_safe(
                    self._world_to_grid(segment_start),
                    self._world_to_grid(segment_end),
                    inflation_cells=0,
                )
                for segment_start, segment_end in zip(injected, injected[1:])
            ):
                continue
            if any(
                self._segment_intersects_expanded_shelf(
                    segment_start, segment_end, shelf, footprint_radius
                )
                for shelf in self.warehouse_layout.get('shelves', [])
                for segment_start, segment_end in zip(
                    [approach, *injected[:-1]], injected
                )
            ):
                continue
            sequence, completed_pickups = self._generate_normal_pickup_prefix(
                approach,
                planned_fault_waypoint_index,
                footprint_radius,
            )
            if not sequence:
                continue
            suffix_count = (
                target_waypoint_count
                - planned_fault_waypoint_index
                - len(injected)
            )
            suffix, final_target = self._generate_task_suffix(
                approach, suffix_count, footprint_radius
            )
            if not suffix:
                continue
            sequence.extend(injected)
            sequence.extend(suffix)
            if len(sequence) == target_waypoint_count:
                return sequence, {
                    'scenario_type': 'wall_localization_drift',
                    'expected_collision': True,
                    'target_obstacle_id': f'warehouse_wall_{wall}',
                    'planned_fault_waypoint_index': planned_fault_waypoint_index,
                    'planned_contact_waypoint_index': (
                        planned_fault_waypoint_index + transition_count - 1
                    ),
                    'completed_pickups_before_fault': completed_pickups,
                    'requested_waypoint_count': target_waypoint_count,
                    'final_task_type': 'package_pickup',
                    'final_pickup_shelf_id': final_target['shelf_id'],
                    'final_pickup_position': list(final_target['position']),
                    'injection': {
                        'wall': wall,
                        'lateral_drift_m': round(safe_clearance - contact_clearance, 4),
                        'collision_penetration_m': round(penetration, 4),
                        'executed_wall_clearance_m': round(contact_clearance, 4),
                        'parallel_travel_before_contact_m': round(
                            abs(tangent_contact - tangent_start), 4
                        ),
                        'parallel_travel_total_m': round(
                            abs(tangent_end - tangent_start), 4
                        ),
                        'fault_transition_waypoint_count': transition_count,
                        'planned_contact_offset_in_fault_segment': transition_count - 1,
                    },
                    'path_inflation_m': round(path_inflation, 4),
                    'footprint_planning_radius_m': round(footprint_radius, 4),
                }
        return None, None

    def _generate_scenario_sequence(
        self, scenario_type, footprint_radius, planned_fault_waypoint_index=None,
        target_waypoint_count=None,
    ):
        if scenario_type == 'safe':
            return self._generate_safe_sequence(
                footprint_radius, target_waypoint_count
            )
        if scenario_type == 'hard_safe':
            return self._generate_hard_safe_sequence(
                footprint_radius, target_waypoint_count
            )
        if scenario_type == 'wall_localization_drift':
            return self._generate_wall_fault_sequence(
                footprint_radius, planned_fault_waypoint_index,
                target_waypoint_count,
            )
        return self._generate_shelf_fault_sequence(
            scenario_type, footprint_radius, planned_fault_waypoint_index,
            target_waypoint_count,
        )

    def generate_industrial_picking_sequences(self, config=None):
        """Generate independent safe/fault samples with normal task prefixes."""
        config = config or {}
        num_seq = max(0, int(config.get('num_sequences', 50)))
        collision_probability = min(
            1.0, max(0.0, float(config.get('collision_probability', 0.7)))
        )
        scenario_weights = {
            scenario: float(config.get(
                f'{scenario}_weight', self.DEFAULT_SCENARIO_WEIGHTS[scenario]
            ))
            for scenario in self.SCENARIO_TYPES
        }
        if any(weight < 0.0 for weight in scenario_weights.values()):
            raise ValueError("Collision scenario weights cannot be negative.")
        if collision_probability > 0.0 and sum(scenario_weights.values()) <= 0.0:
            raise ValueError("At least one collision scenario weight must be positive.")
        collision_index_min = max(4, int(config.get('collision_index_min', 4)))
        collision_index_max = max(
            collision_index_min, int(config.get('collision_index_max', 27))
        )
        waypoint_count_min = max(15, int(config.get('waypoint_count_min', 15)))
        waypoint_count_max = max(
            waypoint_count_min, int(config.get('waypoint_count_max', 30))
        )
        hard_safe_probability = min(1.0, max(0.0, float(config.get(
            'hard_safe_probability_given_safe', 0.5
        ))))
        penetration_min = max(
            0.02, float(config.get('collision_penetration_min_m', 0.06))
        )
        penetration_max = max(
            penetration_min,
            float(config.get('collision_penetration_max_m', 0.22)),
        )
        transition_min = min(
            3, max(1, int(config.get('fault_transition_waypoints_min', 1)))
        )
        transition_max = min(
            3,
            max(
                transition_min,
                int(config.get('fault_transition_waypoints_max', 3)),
            ),
        )
        hard_safe_margin_min = max(
            0.03, float(config.get('hard_safe_margin_min_m', 0.05))
        )
        hard_safe_margin_max = max(
            hard_safe_margin_min,
            float(config.get('hard_safe_margin_max_m', 0.18)),
        )
        self._generation_randomization = {
            'penetration_min_m': penetration_min,
            'penetration_max_m': penetration_max,
            'transition_waypoints_min': transition_min,
            'transition_waypoints_max': transition_max,
            'hard_safe_margin_min_m': hard_safe_margin_min,
            'hard_safe_margin_max_m': hard_safe_margin_max,
            'corner_clearance_min_m': 0.04,
            'corner_clearance_max_m': 0.09,
        }
        schedule = []
        planned_indices = []
        scenario_names = list(scenario_weights)
        scenario_weight_values = [scenario_weights[name] for name in scenario_names]
        for _ in range(num_seq):
            target_waypoint_count = self._rng.randint(
                waypoint_count_min, waypoint_count_max
            )
            if self._rng.random() >= collision_probability:
                safe_type = (
                    'hard_safe'
                    if self._rng.random() < hard_safe_probability else 'safe'
                )
                schedule.append((safe_type, None, target_waypoint_count))
                continue
            scenario_type = self._rng.choices(
                scenario_names, weights=scenario_weight_values, k=1
            )[0]
            effective_index_max = min(
                collision_index_max,
                target_waypoint_count - self.MAX_FAULT_WAYPOINTS - 2,
            )
            if effective_index_max < collision_index_min:
                raise ValueError(
                    "Waypoint count range leaves no room for a collision and final pickup."
                )
            planned_index = self._rng.randint(
                collision_index_min, effective_index_max
            )
            planned_indices.append(planned_index)
            schedule.append((
                scenario_type,
                planned_index,
                target_waypoint_count,
            ))
        self._rng.shuffle(schedule)

        scenario_counts = {
            scenario: sum(1 for item in schedule if item[0] == scenario)
            for scenario in self.SCENARIO_TYPES
        }
        hazard_count = sum(scenario_counts.values())
        safe_count = num_seq - hazard_count
        hard_safe_count = sum(1 for item in schedule if item[0] == 'hard_safe')

        requested_index_counts = {
            index: planned_indices.count(index)
            for index in range(collision_index_min, collision_index_max + 1)
            if planned_indices.count(index) > 0
        }

        self._sequence_generation_info = []
        self._sequence_generation_config = {
            'collision_probability': collision_probability,
            'safe_probability': round(1.0 - collision_probability, 6),
            'selection_strategy': 'independent_bernoulli_per_sequence',
            'scenario_weights_given_collision': scenario_weights,
            'hard_safe_probability_given_safe': hard_safe_probability,
            'realized_hard_safe_count': hard_safe_count,
            'fault_parameter_randomization': self._generation_randomization.copy(),
            'planned_fault_segment_start_range': [collision_index_min, collision_index_max],
            'waypoint_count_range': [waypoint_count_min, waypoint_count_max],
            'collision_index_sampling': (
                'random_fault_segment_start_plus_random_contact_offset'
            ),
            'requested_planned_fault_segment_start_counts': requested_index_counts,
            'realized_generation_counts': {'safe': safe_count, **scenario_counts},
        }
        footprint_radius = self._get_planning_radius()
        sequences = []
        attempts = 0
        max_attempts = max(num_seq * 80, 80)
        last_reported_count = 0
        print(
            f"Generating sequences from independent probability draws: "
            f"{hazard_count} injected fault scenarios, {safe_count} safe scenarios"
        )

        while schedule and attempts < max_attempts:
            attempts += 1
            (
                scenario_type,
                planned_fault_waypoint_index,
                target_waypoint_count,
            ) = schedule[-1]
            sequence, generation_info = self._generate_scenario_sequence(
                scenario_type,
                footprint_radius,
                planned_fault_waypoint_index,
                target_waypoint_count,
            )
            if sequence and generation_info:
                sequences.append(sequence)
                self._sequence_generation_info.append(generation_info)
                schedule.pop()
                if len(sequences) > last_reported_count and len(sequences) % 10 == 0:
                    print(f"  Generated {len(sequences)}/{num_seq} ({scenario_type})")
                    last_reported_count = len(sequences)

        if schedule:
            remaining = {}
            for scenario, _, _ in schedule:
                remaining[scenario] = remaining.get(scenario, 0) + 1
            print(
                f"Warning: generated {len(sequences)}/{num_seq} valid sequences after "
                f"{attempts} attempts; remaining scheduled scenarios: {remaining}"
            )
        return sequences

    def _get_planning_radius(self):
        """Return a conservative 2-D footprint radius for path clearance."""
        shape_info = (self.robot_config or {}).get('shape_info', {})
        circle = shape_info.get('bounding_circle') or {}
        radius = circle.get('radius')
        if radius is not None:
            try:
                return max(0.0, float(radius))
            except (TypeError, ValueError):
                pass
        width = float(shape_info.get('width', 0.8))
        length = float(shape_info.get('length', 0.8))
        return 0.5 * math.hypot(width, length)

    def _generate_aisle_path(self, start, end):
        """兼容性包装: 使用 A* 代替硬编码通道"""
        return self._generate_astar_path(start, end, inflation_m=0.6)

    def _add_noise_to_waypoint(self, waypoint, noise_level, bounds):
        """添加噪声到航点"""
        x, y = waypoint
        noisy_x = x + self._rng.gauss(0, noise_level)
        noisy_y = y + self._rng.gauss(0, noise_level)
        noisy_x = max(bounds['x_min'], min(bounds['x_max'], noisy_x))
        noisy_y = max(bounds['y_min'], min(bounds['y_max'], noisy_y))
        return (round(noisy_x, 2), round(noisy_y, 2))

    def _remove_duplicate_waypoints(self, sequence, threshold=0.1):
        """移除重复或过于接近的航点"""
        if len(sequence) <= 1: return sequence
        cleaned = [sequence[0]]
        for wp in sequence[1:]:
            last_wp = cleaned[-1]
            distance = np.hypot(wp[0] - last_wp[0], wp[1] - last_wp[1])
            if distance > threshold:
                cleaned.append(wp)
        return cleaned

    def _on_navigation_failed(self, failure_info):
        """导航失败回调：区分真实碰撞、卡住和超时"""
        info = failure_info.copy() if failure_info else {}
        self._navigation_failure_occurred = True
        self._navigation_failure_info = info
        if info.get('type') == 'collision':
            self._collision_occurred = True
            self._collision_info = info
    
    def _extract_collision_position(self):
        """从碰撞信息中安全提取首次 PhysX 接触点，旧数据缺失时逐级回退"""
        if not self._collision_info:
            return None

        def to_xy(value):
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                try:
                    return [float(value[0]), float(value[1])]
                except (TypeError, ValueError):
                    return None
            return None

        details = self._collision_info.get('details', {})
        contact_pos = None
        if isinstance(details, dict):
            contact_pos = (
                to_xy(details.get('first_contact_position')) or
                to_xy(details.get('collision_position')) or
                to_xy(details.get('max_impulse_contact_position'))
            )
        if contact_pos is not None:
            return contact_pos

        return to_xy(self._collision_info.get('current_position'))

    def _extract_xy_from_details(self, key):
        """从 failure details 中提取二维坐标字段"""
        details = self._collision_info.get('details', {}) if self._collision_info else {}
        value = details.get(key) if isinstance(details, dict) else None
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return [float(value[0]), float(value[1])]
            except (TypeError, ValueError):
                return None
        return None

    def _extract_collision_segment(self, failed_idx, sequence_length):
        """用失败航点索引近似首次碰撞所在航段"""
        try:
            idx = int(failed_idx)
        except (TypeError, ValueError):
            return None
        if sequence_length < 2:
            return None
        if idx <= 0:
            return [0, 1]
        if idx >= sequence_length:
            return [sequence_length - 2, sequence_length - 1]
        return [idx - 1, idx]

    def _normalize_collision_object_id(self, collision_object):
        """从 PhysX actor path 中提取更稳定的障碍物 id"""
        text = str(collision_object or "")
        match = re.search(r"Shelf[_\-\s]*0*(\d+)", text, re.IGNORECASE)
        if match:
            return f"Shelf-{int(match.group(1))}"
        return text if text and text != "Unknown" else None

    @staticmethod
    def _classify_collection_outcome(test_result, generation_info):
        """Separate the physical label from fault-injection quality."""
        if test_result.get('collision_detected'):
            return 'collision', True
        if test_result.get('navigation_status') != 'success':
            return 'invalid_navigation', False

        diagnostics = test_result.get('contact_diagnostics') or {}
        raw_contact_count = int(diagnostics.get('raw_contact_event_count') or 0)
        if generation_info.get('expected_collision') is True:
            if raw_contact_count > 0:
                return 'ambiguous_contact', False
            return 'fault_injection_miss', False
        if raw_contact_count > 0:
            return 'unexpected_subthreshold_contact', False
        return 'safe', True

    def _get_robot_size(self):
        """获取机器人尺寸及完整 shape_info（含 12 方向射线距离）。"""
        size = {"width": 0.8, "length": 0.8}
        shape_info = None

        # 优先从 robot_config 获取完整 shape_info
        if self.robot_config:
            si = self.robot_config.get("shape_info", {})
            if si:
                size["width"] = float(si.get("width", size["width"]))
                size["length"] = float(si.get("length", size["length"]))
                # 保存全部形状表达和固定方向射线特征。
                shape_info = {
                    "polygon": si.get("polygon"),
                    "hull_3d": si.get("hull_3d"),
                    "obb_2d": si.get("obb_2d"),
                    "bounding_circle": si.get("bounding_circle"),
                    "polar_ray_distances_12": si.get("polar_ray_distances_12"),
                }

        # 回退：从控制器获取真实尺寸
        if shape_info is None:
            try:
                if hasattr(self.robot_controller, '_monitor'):
                    monitor = getattr(self.robot_controller, '_monitor', None)
                    if monitor:
                        size["width"] = float(getattr(monitor, 'robot_base_width', size["width"]))
                        size["length"] = float(getattr(monitor, 'robot_base_length', size["length"]))
            except Exception as e:
                print(f"Warning: Could not read robot size: {e}")

        return {"size": size, "shape_info": shape_info}

    async def test_waypoint_sequence(self, waypoint_sequence, generation_info=None):
        """测试一条完整的航点序列"""
        self.current_test_id += 1
        generation_info = generation_info or {
            'scenario_type': 'external',
            'expected_collision': None,
        }
        scenario_type = generation_info.get('scenario_type', 'external')
        
        clean_sequence = []
        for wp in waypoint_sequence:
            clean_sequence.append([float(wp[0]), float(wp[1])])
        
        test_result = {
            'test_id': self.current_test_id,
            'timestamp': datetime.now().isoformat(),
            'waypoint_sequence': clean_sequence,
            'sequence_length': len(clean_sequence),
            'collision_detected': False,
            'scenario_type': scenario_type,
            'collision_type': None,
            'target_obstacle_id': generation_info.get('target_obstacle_id'),
            'collision_matches_target': None,
            'collision_waypoint_index': None,
            'collision_segment': None,
            'collision_position': None,
            'first_contact_position': None,
            'max_impulse_contact_position': None,
            'first_contact_impulse': None,
            'max_impulse': None,
            'first_contact_simulation_time': None,
            'robot_center_at_collision': None,
            'collision_object': None,
            'collision_object_id': None,
            'collision_confirmation_reason': None,
            'reached_waypoint_count': 0,
            'navigation_status': 'unknown',
            'failure_type': None,
            'failure_message': None,
            'duration': 0.0,
            'wall_duration_seconds': 0.0,
            'simulation_duration_seconds': 0.0,
            'generation': generation_info,
            'executed_trajectory': [],
            'contact_diagnostics': {},
            'collection_outcome': 'unknown',
            'training_eligible': False,
        }
        
        start_time = time.time()
        self._collision_occurred = False
        self._collision_info = {}
        self._navigation_failure_occurred = False
        self._navigation_failure_info = {}
        
        # 注册回调
        original_callback = getattr(self.robot_controller, '_on_navigation_failed', None)
        self.robot_controller.set_failure_callback(self._on_navigation_failed)
        
        try:
            print(f"\n[Sequence {self.current_test_id}] Testing {len(waypoint_sequence)} waypoints")
            
            self.robot_controller.start_waypoint_navigation(list(waypoint_sequence))
            
            check_interval = 0.1
            while True:
                await asyncio.sleep(check_interval)
                
                if self._collision_occurred:
                    test_result['collision_detected'] = True
                    test_result['collision_type'] = (
                        scenario_type if scenario_type != 'safe' else 'unexpected_collision'
                    )
                    test_result['navigation_status'] = 'collision'
                    
                    failed_idx = self._collision_info.get('failed_waypoint_index')
                    if failed_idx is None:
                        failed_idx = int(getattr(self.robot_controller, '_current_waypoint_index', 0))

                    test_result['collision_waypoint_index'] = failed_idx
                    test_result['collision_segment'] = self._extract_collision_segment(
                        failed_idx,
                        len(clean_sequence),
                    )
                    test_result['reached_waypoint_count'] = failed_idx
                    test_result['collision_position'] = self._extract_collision_position()
                    test_result['first_contact_position'] = self._extract_xy_from_details('first_contact_position')
                    test_result['max_impulse_contact_position'] = (
                        self._extract_xy_from_details('max_impulse_contact_position') or
                        self._extract_xy_from_details('collision_position')
                    )
                    current_pos = self._collision_info.get('current_position')
                    if isinstance(current_pos, (list, tuple)) and len(current_pos) >= 2:
                        test_result['robot_center_at_collision'] = [float(current_pos[0]), float(current_pos[1])]
                    
                    details = self._collision_info.get('details', {})
                    test_result['first_contact_impulse'] = details.get('first_contact_impulse')
                    test_result['max_impulse'] = details.get('max_impulse', details.get('force'))
                    test_result['first_contact_simulation_time'] = details.get('first_contact_timestamp')
                    test_result['collision_confirmation_reason'] = details.get(
                        'confirmation_reason'
                    )
                    test_result['collision_object'] = str(details.get('colliding_object', 'Unknown'))
                    test_result['collision_object_id'] = self._normalize_collision_object_id(
                        test_result['collision_object']
                    )
                    target_obstacle_id = test_result.get('target_obstacle_id')
                    if target_obstacle_id:
                        if scenario_type == 'wall_localization_drift':
                            test_result['collision_matches_target'] = '/World/Wall' in test_result['collision_object']
                        else:
                            test_result['collision_matches_target'] = (
                                test_result['collision_object_id'] == target_obstacle_id
                            )

                    self.robot_controller.stop_navigation()
                    print(f"  COLLISION at waypoint {failed_idx}")
                    break

                if self._navigation_failure_occurred:
                    failure_type = self._navigation_failure_info.get('type', 'unknown')
                    test_result['collision_detected'] = False
                    test_result['navigation_status'] = failure_type
                    failed_idx = self._navigation_failure_info.get('failed_waypoint_index')
                    if failed_idx is None:
                        failed_idx = int(getattr(self.robot_controller, '_current_waypoint_index', 0))
                    test_result['reached_waypoint_count'] = failed_idx
                    test_result['failure_type'] = failure_type
                    test_result['failure_message'] = self._navigation_failure_info.get('details', {}).get(
                        'message',
                        self._navigation_failure_info.get('message')
                    )
                    print(f"  {failure_type.upper()}")
                    break
                
                if not getattr(self.robot_controller, '_is_navigating', False):
                    test_result['reached_waypoint_count'] = int(getattr(self.robot_controller, '_current_waypoint_index', 0))
                    
                    if getattr(self.robot_controller, '_navigation_completed', False):
                        test_result['navigation_status'] = 'success'
                        test_result['reached_waypoint_count'] = len(waypoint_sequence)
                        print("  SUCCESS - All waypoints reached")
                    else:
                        test_result['navigation_status'] = 'timeout'
                        test_result['failure_type'] = 'timeout'
                        print("  TIMEOUT/STUCK")
                    break
            
            test_result['duration'] = round(time.time() - start_time, 2)
            test_result['wall_duration_seconds'] = test_result['duration']
            
        except Exception as e:
            print(f"  ERROR: {e}")
            test_result['navigation_status'] = 'error'
            test_result['error_message'] = str(e)
            
        finally:
            if hasattr(self.robot_controller, 'get_executed_trajectory'):
                test_result['executed_trajectory'] = self.robot_controller.get_executed_trajectory()
            if hasattr(self.robot_controller, 'get_simulation_time'):
                test_result['simulation_duration_seconds'] = round(
                    self.robot_controller.get_simulation_time(), 6
                )
            if hasattr(self.robot_controller, 'get_contact_diagnostics'):
                test_result['contact_diagnostics'] = (
                    self.robot_controller.get_contact_diagnostics()
                )
            outcome, training_eligible = self._classify_collection_outcome(
                test_result, generation_info
            )
            test_result['collection_outcome'] = outcome
            test_result['training_eligible'] = training_eligible
            self.robot_controller.set_failure_callback(original_callback)
            self.robot_controller.stop_navigation()
        
        return test_result

    async def collect_dataset(self, waypoint_sequences, robot_id="unknown", save_interval=5):
        """批量收集碰撞数据集"""
        self._stop_requested = False
        self.robot_id = robot_id or "unknown"
        save_interval = max(1, int(save_interval))
        self._skipped_reset_sequences = 0
        self._requested_sequence_count = len(waypoint_sequences)
        
        print("\n" + "="*70)
        print(f"COLLISION DATASET COLLECTION - {len(waypoint_sequences)} SEQUENCES")
        print(f"Robot ID: {self.robot_id}")
        print("="*70)
        
        for i, sequence in enumerate(waypoint_sequences):
            if self._stop_requested:
                print(f"Collection stopped by user.")
                break
            
            print(f"\n--- Sequence {i+1}/{len(waypoint_sequences)} ---")
            reset_ready = False
            for _ in range(3):
                if self.reset_robot_to_start() and await self._wait_for_reset_settle():
                    reset_ready = True
                    break
            if not reset_ready:
                self._skipped_reset_sequences += 1
                print("  RESET ERROR - sequence skipped to protect dataset integrity")
                continue
            
            generation_info = (
                self._sequence_generation_info[i]
                if i < len(self._sequence_generation_info)
                else {'scenario_type': 'external', 'expected_collision': None}
            )
            result = await self.test_waypoint_sequence(sequence, generation_info=generation_info)
            self.dataset.append(result)
            
            if (i + 1) % save_interval == 0:
                self.save_dataset(partial=True)
                print(f"Progress saved.")
            
            await asyncio.sleep(0.3)
        
        self.save_dataset(partial=False)
        self.print_statistics()
        print("COLLECTION COMPLETED")

    def save_dataset(self, partial=False):
        """保存数据集到JSON文件"""
        if partial:
            filename = f"collision_sequences_partial_{self._session_timestamp}.json"
        else:
            filename = f"collision_sequences_final_{self._session_timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        total = len(self.dataset)
        collisions = sum(1 for d in self.dataset if d['collision_detected'])
        safe = sum(1 for d in self.dataset if d['navigation_status'] == 'success')
        actual_collision_index_counts = {}
        collection_outcome_counts = {}
        for item in self.dataset:
            outcome = str(item.get('collection_outcome') or 'unknown')
            collection_outcome_counts[outcome] = (
                collection_outcome_counts.get(outcome, 0) + 1
            )
            if not item.get('collision_detected'):
                continue
            collision_index = item.get('collision_waypoint_index')
            if collision_index is None:
                continue
            key = str(int(collision_index))
            actual_collision_index_counts[key] = actual_collision_index_counts.get(key, 0) + 1
        
        robot_size_data = self._get_robot_size()
        wall_seconds = sum(float(item.get('wall_duration_seconds') or 0.0) for item in self.dataset)
        simulation_seconds = sum(float(item.get('simulation_duration_seconds') or 0.0) for item in self.dataset)
        dataset_output = {
            'metadata': {
                'total_sequences': total,
                'collision_sequences': collisions,
                'safe_sequences': safe,
                'training_eligible_sequences': sum(
                    1 for item in self.dataset if item.get('training_eligible')
                ),
                'collection_outcome_counts': collection_outcome_counts,
                'requested_sequences': self._requested_sequence_count,
                'skipped_reset_sequences': self._skipped_reset_sequences,
                'actual_collision_waypoint_index_counts': actual_collision_index_counts,
                'collection_date': datetime.now().isoformat(),
                'collection_config': {
                    'random_seed': self.random_seed,
                    'grid_resolution_m': self.grid_resolution,
                    'clock_source': 'physics_dt',
                    'collision_confirmation': (
                        self.robot_controller.get_collision_confirmation_config()
                        if hasattr(
                            self.robot_controller,
                            'get_collision_confirmation_config',
                        ) else None
                    ),
                    'scenario_generation': self._sequence_generation_config,
                    'motion': self.robot_controller.get_motion_config()
                    if hasattr(self.robot_controller, 'get_motion_config') else None,
                    'runtime': self.collection_runtime_config,
                    'performance': {
                        'wall_duration_seconds': round(wall_seconds, 3),
                        'simulation_duration_seconds': round(simulation_seconds, 3),
                        'real_time_factor': round(simulation_seconds / wall_seconds, 4)
                        if wall_seconds > 0.0 else None,
                    },
                },
                'robot_info': {
                    'id': self.robot_id,
                    'size': robot_size_data.get('size', {'width': 0.8, 'length': 0.8}),
                    'shape_info': robot_size_data.get('shape_info'),
                },
                'warehouse_layout': self.raw_layout_config
            },
            'sequences': self.dataset
        }
        
        dataset_output = convert_to_serializable(dataset_output)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset_output, f, indent=2, ensure_ascii=False)
        
        if not partial:
            p_file = os.path.join(self.output_dir, f"collision_sequences_partial_{self._session_timestamp}.json")
            if os.path.exists(p_file): os.remove(p_file)
        
        return filepath

    def _parse_layout_config(self, config):
        """解析 layouts.json，支持旋转"""
        try:
            model = config.get("warehouse_geometry_model", {})
            summary = model.get("global_summary", {})
            obstacles = model.get("obstacles", [])
            
            size = summary.get("warehouse_size_m", {"x": 12.0, "y": 12.0})
            w, h = size["x"], size["y"]
            
            start_zone = (0.0, 0.0) # 默认为中心
            
            layout = {
                'shelves': [],
                'bounds': {'x_min': -w/2, 'x_max': w/2, 'y_min': -h/2, 'y_max': h/2},
                'start_zone': start_zone,
                'dropoff_zone': (0.0, h/2 - 1.0),
                'pickup_points': []
            }
            
            for obs in obstacles:
                if obs.get("type") == "Shelf":
                    center = obs.get("center_point_m", [0, 0])
                    dims = obs.get("dimensions_m", {"length": 1, "width": 1})
                    rot = obs.get("orientation_rad", 0.0)
                    
                    # 统一存储为 size=(local_x, local_y)
                    layout['shelves'].append({
                        'id': obs.get('element_id', 'unknown'),
                        'center': (float(center[0]), float(center[1])),
                        'size': (float(dims['width']), float(dims['length'])),
                        'rot': float(rot),
                        'collision_geometry': obs.get('collision_geometry') or {},
                    })
            
            return layout
        except Exception as e:
            print(f"Error parsing layout config: {e}")
            return None
        
    def print_statistics(self):
        """打印数据集统计信息"""
        total = len(self.dataset)
        if total == 0:
            print("No data collected yet.")
            return
            
        collisions = sum(1 for d in self.dataset if d['collision_detected'])
        safe = sum(1 for d in self.dataset if d['navigation_status'] == 'success')
        timeout = sum(1 for d in self.dataset if d['navigation_status'] == 'timeout')
        incomplete = sum(1 for d in self.dataset if d['navigation_status'] == 'incomplete')
        collision_type_counts = {}
        collection_outcome_counts = {}
        actual_collision_indices = []
        for item in self.dataset:
            outcome = str(item.get('collection_outcome') or 'unknown')
            collection_outcome_counts[outcome] = (
                collection_outcome_counts.get(outcome, 0) + 1
            )
            collision_type = item.get('collision_type')
            if collision_type:
                collision_type_counts[collision_type] = collision_type_counts.get(collision_type, 0) + 1
            if item.get('collision_detected') and item.get('collision_waypoint_index') is not None:
                actual_collision_indices.append(int(item['collision_waypoint_index']))
        wall_seconds = sum(float(d.get('wall_duration_seconds') or 0.0) for d in self.dataset)
        simulation_seconds = sum(float(d.get('simulation_duration_seconds') or 0.0) for d in self.dataset)
        
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        print(f"Total sequences tested:    {total}")
        print(f"Collision sequences:       {collisions} ({collisions/total*100:.1f}%)")
        print(f"Safe (complete) sequences: {safe} ({safe/total*100:.1f}%)")
        print(f"Timeout sequences:         {timeout} ({timeout/total*100:.1f}%)")
        print(f"Incomplete sequences:      {incomplete} ({incomplete/total*100:.1f}%)")
        print(
            "Training-eligible sequences: "
            f"{sum(1 for item in self.dataset if item.get('training_eligible'))}"
        )
        print("Collection outcomes:")
        for outcome, count in sorted(collection_outcome_counts.items()):
            print(f"  {outcome}: {count}")
        for collision_type, count in sorted(collision_type_counts.items()):
            print(f"  {collision_type}: {count}")
        if actual_collision_indices:
            print(
                f"Actual collision waypoint index range: "
                f"{min(actual_collision_indices)}-{max(actual_collision_indices)}"
            )
        if wall_seconds > 0.0:
            print(f"Real-time factor:          {simulation_seconds / wall_seconds:.2f}x")
        print("="*50)
