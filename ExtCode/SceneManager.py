"""
Scene management utilities for Isaac Sim extension
"""
import os
import math
import json
import random
import numpy as np
import omni.usd
from pxr import Gf, UsdPhysics, UsdGeom
from omni.physx.scripts import utils
from isaacsim.core.utils.prims import create_prim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.sensors.camera import Camera
from .Geometry2D import convex_hull_2d, cube_local_corners, transformed_xy_convex_hull

class SceneManager:
    """Manage scene operations like loading, resetting, and adding objects"""

    # Native local-X width and local-Y length of the shelf USD assets.
    _SHELF_NATIVE_DIMENSIONS_M = {
        "Shelf_01.usd": (0.691, 8.81),
        "Shelf_02.usd": (0.691, 8.81),
        "Shelf_03.usd": (0.691, 8.81),
        "Shelf_04.usd": (0.691, 8.81),
    }
    
    def __init__(self, scene_path=None):
        ext_code_dir = os.path.dirname(os.path.abspath(__file__))
        extension_root = os.path.dirname(ext_code_dir)

        configured_asset_root = os.environ.get("ROBOTPATH_ASSET_ROOT")
        self.base_assets_path = os.path.abspath(
            configured_asset_root or os.path.join(extension_root, "data", "Assets")
        )
        configured_scene_path = os.environ.get("ROBOTPATH_SCENE_PATH")
        scene_path = configured_scene_path or scene_path or "Warehouse.usd"
        if os.path.isabs(scene_path):
            self.scene_path = os.path.abspath(scene_path)
        elif scene_path.replace("\\", "/").startswith("data/Assets/"):
            self.scene_path = os.path.join(self.base_assets_path, os.path.basename(scene_path))
        else:
            self.scene_path = os.path.abspath(os.path.join(extension_root, scene_path))
        self.package_counter = 0
        self.overview_camera = None

        # 布局配置相关
        self.layouts_file = os.path.join(extension_root, "Experiment", "layouts.json")
        self.current_layout_data = None
        self.global_robot_config = {}
        self.current_robot_id = "ridgeback_franka"
        self.current_robot_config = self._get_fallback_robot_config()
        # 默认加载第一个布局配置
        self.load_layout_config()

    def required_asset_paths(self):
        """Return the external files needed to assemble the selected scene."""
        required = [self.scene_path]
        layout = self.current_layout_data or {}
        obstacles = layout.get("warehouse_geometry_model", {}).get("obstacles", [])
        shelf_files = {
            obs.get("usd_file", "Shelf_01.usd")
            for obs in obstacles
            if obs.get("type") == "Shelf"
        }
        if not shelf_files:
            shelf_files = {"Shelf_01.usd", "Shelf_02.usd", "Shelf_03.usd", "Shelf_04.usd"}
        required.extend(os.path.join(self.base_assets_path, name) for name in sorted(shelf_files))
        robot_cfg = self.get_robot_config()[1]
        required.append(os.path.join(self.base_assets_path, robot_cfg.get("usd_file", "ridgeback_franka.usd")))
        return required

    def missing_assets(self):
        """Return missing files and a short remediation hint for users."""
        return [path for path in self.required_asset_paths() if not os.path.isfile(path)]

    def check_assets(self, report=True):
        """Check required external assets before opening a USD stage."""
        missing = self.missing_assets()
        if missing and report:
            print("RobotPath cannot load the complete Warehouse scene.")
            print("Set ROBOTPATH_ASSET_ROOT to a licensed asset directory, or set ROBOTPATH_SCENE_PATH.")
            for path in missing:
                print(f"Missing asset: {path}")
        return not missing

    def _get_fallback_robot_config(self):
        """默认机器人配置,用于兼容旧版 layouts.json。"""
        return {
            "usd_file": "ridgeback_franka.usd",
            "prim_path": "/World/RidgebackFranka",
            "sensor_path": "/World/RidgebackFranka/base_link/Contact_Sensor",
            "shape_info": {"width": 0.8, "length": 1.0},
        }

    def get_robot_config(self, robot_id=None):
        """从全局配置中解析机器人配置,若缺失则回退到默认配置。"""
        root_config = self.global_robot_config or {}
        robots = root_config.get("robots", {})
        default_robot_id = robot_id

        if default_robot_id is None and self.current_layout_data:
            default_robot_id = self.current_layout_data.get("default_robot")

        if default_robot_id is None:
            default_robot_id = root_config.get("default_robot", "ridgeback_franka")

        fallback = self._get_fallback_robot_config()
        selected = robots.get(default_robot_id)
        if not selected:
            if default_robot_id != "ridgeback_franka":
                print(f"Robot '{default_robot_id}' not found. Fallback to ridgeback_franka.")
            return "ridgeback_franka", fallback

        # 合并缺省字段,避免配置不完整时崩溃。
        fallback_shape = fallback.get("shape_info", {})
        selected_shape = selected.get("shape_info", {})
        merged = {
            "usd_file": selected.get("usd_file", fallback["usd_file"]),
            "prim_path": selected.get("prim_path", fallback["prim_path"]),
            "sensor_path": selected.get("sensor_path", fallback["sensor_path"]),
            "shape_info": {
                "width": float(selected_shape.get("width", fallback_shape.get("width", 0.8))),
                "length": float(selected_shape.get("length", fallback_shape.get("length", 1.0))),
                "polygon": selected_shape.get("polygon", fallback_shape.get("polygon")),
                "hull_3d": selected_shape.get("hull_3d", fallback_shape.get("hull_3d")),
                "obb_2d": selected_shape.get("obb_2d", fallback_shape.get("obb_2d")),
                "bounding_circle": selected_shape.get(
                    "bounding_circle", fallback_shape.get("bounding_circle")
                ),
                "polar_ray_distances_12": selected_shape.get(
                    "polar_ray_distances_12", fallback_shape.get("polar_ray_distances_12")
                ),
            },
        }
        return default_robot_id, merged

    def load_layout_config(self, layout_name=None):
        """加载指定的布局配置"""
        try:
            if not os.path.exists(self.layouts_file):
                print(f"Layouts file not found: {self.layouts_file}")
                return False

            with open(self.layouts_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.global_robot_config = data.get("robot_config", {})
            
            layouts = data.get("layouts", {})
            
            # 如果未指定名称，默认使用第一个
            if layout_name is None and layouts:
                layout_name = list(layouts.keys())[0]
            
            if layout_name and layout_name in layouts:
                self.current_layout_data = layouts[layout_name]
                return True
            else:
                print(f"Layout '{layout_name}' not found.")
                return False
        except Exception as e:
            print(f"Error loading layout config: {e}")
            return False

    def load_scene(self):
        """Load the scene from USD file"""
        try:
            if not self.check_assets():
                return False
            stage = omni.usd.get_context().get_stage()
            if stage:
                omni.usd.get_context().close_stage()
            
            result = omni.usd.get_context().open_stage(self.scene_path)
            if result:
                print(f"Successfully loaded scene from: {self.scene_path}")
                stage = omni.usd.get_context().get_stage()
                # 添加场景组件&设定参数
                self.add_shelves(stage)
                self.capture_shelf_collision_geometry(stage)
                self.current_robot_id, self.current_robot_config = self.get_robot_config()
                self.add_robot(stage, self.current_robot_config)
                self.overview_camera = self.add_camera(prim_path="/World/OverviewCamera", 
                                                       position=(0, 0, 4.5), 
                                                       orientation=(-90, 90, 0))
                return True
            else:
                print(f"Failed to load scene from: {self.scene_path}")
                return False
        except Exception as e:
            print(f"Error loading USD file: {str(e)}")
            return False

    def reset_scene(self):
        """Reset the scene"""
        try:
            result = self.load_scene()
            if result:
                print("Scene reset successful")
            return result
        except Exception as e:
            print(f"Error resetting scene: {str(e)}")
            return False

    def add_to_scene(self, stage, usd_path, target_prim_path, position=(0, 0, 0), 
                    rotation=(0, 0, 0), scale=(1, 1, 1), add_physics=True):
        """Add USD asset to scene"""
        try:
            if not os.path.isfile(usd_path):
                print(f"Missing USD asset for {target_prim_path}: {usd_path}")
                return False
            orientation_quat = euler_angles_to_quat(np.array(rotation), degrees=True)

            xform_prim = create_prim(
                prim_path=target_prim_path,
                prim_type="Xform",
                position=position,
                orientation=orientation_quat,
                scale=scale,
                usd_path=usd_path,
            )

            # about物理属性
            if add_physics:
                utils.setRigidBody(xform_prim, "convexHull", False)
                mass_api = UsdPhysics.MassAPI.Apply(xform_prim)
                mass_api.CreateMassAttr(1.0)
            
            print(f"Added {target_prim_path} at {position}")
            return True
        except Exception as e:
            print(f"Error adding usd: {str(e)}")
            return False

    def add_shelves(self, stage):
        """Add shelves to the scene based on current layout"""
        if not self.current_layout_data:
            print("No layout data loaded. Using default hardcoded shelves.")
            self._add_default_shelves(stage)
            return

        try:
            obstacles = self.current_layout_data.get("warehouse_geometry_model", {}).get("obstacles", [])
            
            for obs in obstacles:
                if obs.get("type") == "Shelf":
                    usd_file = obs.get("usd_file", "Shelf_01.usd")
                    usd_path = os.path.join(self.base_assets_path, usd_file)
                    prim_path = obs.get("prim_path", f"/World/{obs.get('element_id')}")
                    
                    # JSON中的位置是2D [x, y]，我们需要3D [x, y, 0]
                    center_2d = obs.get("center_point_m", [0, 0])
                    position = (center_2d[0], center_2d[1], 0.0)
                    
                    # JSON中的旋转是弧度，add_to_scene 内部会处理欧拉角转四元数
                    # 注意：add_to_scene 默认期望的是角度(degrees=True)，所以需要转换
                    rot_rad = obs.get("orientation_rad", 0.0)
                    rot_deg = math.degrees(rot_rad)
                    rotation = (0, 0, rot_deg)

                    scale = (1.0, 1.0, 1.0)
                    asset_dims = obs.get("asset_dimensions_m", {})
                    native_dims = self._SHELF_NATIVE_DIMENSIONS_M.get(usd_file)
                    if asset_dims:
                        native_dims = (
                            float(asset_dims["width"]),
                            float(asset_dims["length"]),
                        )
                    target_dims = obs.get("dimensions_m", {})
                    if native_dims and target_dims:
                        native_width, native_length = native_dims
                        target_width = float(target_dims.get("width", native_width))
                        target_length = float(target_dims.get("length", native_length))
                        scale = (
                            target_width / native_width,
                            target_length / native_length,
                            1.0,
                        )
                    
                    self.add_to_scene(
                        stage,
                        usd_path,
                        prim_path,
                        position,
                        rotation=rotation,
                        scale=scale,
                        add_physics=False,
                    )
                    
        except Exception as e:
            print(f"Error adding shelves from layout: {e}")
            self._add_default_shelves(stage)

    @staticmethod
    def _convex_hull_2d(points):
        """Return a counter-clockwise convex hull without a repeated endpoint."""
        return convex_hull_2d(points)

    @classmethod
    def _collider_world_polygon(cls, prim, prim_to_world, minimum, maximum):
        """Extract an exact Cube rectangle or a transformed Mesh XY convex hull."""
        local_points = []
        geometry_source = 'usd_world_aabb_fallback'
        if prim.IsA(UsdGeom.Cube):
            size = UsdGeom.Cube(prim).GetSizeAttr().Get()
            local_points = cube_local_corners(size if size is not None else 2.0)
            geometry_source = 'usd_cube_transformed_rectangle'
        elif prim.IsA(UsdGeom.Mesh):
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
            local_points = [tuple(point) for point in points]
            geometry_source = 'usd_mesh_world_xy_convex_hull'

        if not local_points:
            world_points = [
                (float(x), float(y))
                for x in (minimum[0], maximum[0])
                for y in (minimum[1], maximum[1])
            ]
            polygon = cls._convex_hull_2d(world_points)
        else:
            polygon = transformed_xy_convex_hull(
                local_points,
                lambda point: prim_to_world.Transform(Gf.Vec3d(*point)),
            )
        if len(polygon) < 3:
            polygon = cls._convex_hull_2d([
                (float(x), float(y))
                for x in (minimum[0], maximum[0])
                for y in (minimum[1], maximum[1])
            ])
            geometry_source = 'usd_world_aabb_fallback'
        return polygon, geometry_source

    def capture_shelf_collision_geometry(self, stage):
        """Snapshot enabled shelf CollisionAPI polygons in world XY coordinates."""
        if not stage or not self.current_layout_data:
            return
        obstacles = self.current_layout_data.get(
            'warehouse_geometry_model', {}
        ).get('obstacles', [])
        bbox_cache = UsdGeom.BBoxCache(
            0.0,
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
        xform_cache = UsdGeom.XformCache(0.0)
        all_prims = list(stage.Traverse())
        for obstacle in obstacles:
            if obstacle.get('type') != 'Shelf':
                continue
            root_path = obstacle.get('prim_path')
            root = stage.GetPrimAtPath(root_path) if root_path else None
            colliders = []
            if root and root.IsValid():
                prefix = root_path.rstrip('/') + '/'
                for prim in all_prims:
                    prim_path = str(prim.GetPath())
                    if prim_path != root_path and not prim_path.startswith(prefix):
                        continue
                    if not prim.HasAPI(UsdPhysics.CollisionAPI):
                        continue
                    collision_api = UsdPhysics.CollisionAPI(prim)
                    enabled_attr = collision_api.GetCollisionEnabledAttr()
                    if (
                        enabled_attr
                        and enabled_attr.HasAuthoredValueOpinion()
                        and enabled_attr.Get() is False
                    ):
                        continue
                    try:
                        aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
                        minimum = aligned.GetMin()
                        maximum = aligned.GetMax()
                    except Exception:
                        continue
                    if maximum[0] <= minimum[0] or maximum[1] <= minimum[1]:
                        continue
                    center = obstacle.get('center_point_m', [0.0, 0.0])
                    shelf_rotation = -float(obstacle.get('orientation_rad', 0.0))
                    cos_r = math.cos(shelf_rotation)
                    sin_r = math.sin(shelf_rotation)
                    try:
                        prim_to_world = xform_cache.GetLocalToWorldTransform(prim)
                    except Exception:
                        prim_to_world = Gf.Matrix4d(1.0)
                    polygon, geometry_source = self._collider_world_polygon(
                        prim, prim_to_world, minimum, maximum
                    )
                    local_polygon = []
                    for world_x, world_y in polygon:
                        dx = world_x - float(center[0])
                        dy = world_y - float(center[1])
                        local_polygon.append((
                            dx * cos_r - dy * sin_r,
                            dx * sin_r + dy * cos_r,
                        ))
                    colliders.append({
                        'prim_path': prim_path,
                        'prim_type': prim.GetTypeName(),
                        'geometry_source': geometry_source,
                        'polygon_m': [list(point) for point in polygon],
                        'shelf_local_polygon_m': [
                            [float(point[0]), float(point[1])]
                            for point in local_polygon
                        ],
                        'aabb_min_m': [float(minimum[0]), float(minimum[1])],
                        'aabb_max_m': [float(maximum[0]), float(maximum[1])],
                        'shelf_local_aabb_min_m': [
                            min(point[0] for point in local_polygon),
                            min(point[1] for point in local_polygon),
                        ],
                        'shelf_local_aabb_max_m': [
                            max(point[0] for point in local_polygon),
                            max(point[1] for point in local_polygon),
                        ],
                    })
            obstacle['collision_geometry'] = {
                'source': 'usd_collision_api_world_polygon',
                'collider_count': len(colliders),
                'colliders': colliders,
                'fallback_used': not bool(colliders),
            }
            if not colliders:
                print(f"Warning: no enabled CollisionAPI geometry found under {root_path}")

    def _add_default_shelves(self, stage):
        """Original hardcoded shelf logic as fallback"""
        shelf_configs = [
            {"path": f"{self.base_assets_path}/Shelf_01.usd", "target": "/World/Shelf_01", "position": (4.5, 0.0, 0.0)},
            {"path": f"{self.base_assets_path}/Shelf_02.usd", "target": "/World/Shelf_02", "position": (1.5, 0.0, 0.0)},
            {"path": f"{self.base_assets_path}/Shelf_03.usd", "target": "/World/Shelf_03", "position": (-1.5, 0.0, 0.0)},
            {"path": f"{self.base_assets_path}/Shelf_04.usd", "target": "/World/Shelf_04", "position": (-4.5, 0.0, 0.0)}
        ]
        
        for config in shelf_configs:
            self.add_to_scene(stage, config["path"], config["target"], config["position"], add_physics=False)

    def add_packages(self, count=1):
        """Add packages to scene on random shelves and levels based on layout"""
        try:
            stage = omni.usd.get_context().get_stage()
            if not stage:
                print("No active stage to add package")
                return
            
            # 从布局中获取货架信息
            shelves = []
            if self.current_layout_data:
                obstacles = self.current_layout_data.get("warehouse_geometry_model", {}).get("obstacles", [])
                for obs in obstacles:
                    if obs.get("type") == "Shelf":
                        shelves.append(obs)
            
            if not shelves:
                print("No shelves found in layout for package placement.")
                return
            
            for _ in range(count):               
                random_scale = (
                    random.uniform(0.3, 0.5),
                    random.uniform(0.3, 0.5),
                    random.uniform(0.3, 0.5)
                )
                
                volume = random_scale[0] * random_scale[1] * random_scale[2]
                density_factor = 10.0 
                package_mass = volume * density_factor

                # 随机选择货架
                target_shelf = random.choice(shelves)
                
                # 获取货架属性
                center = target_shelf.get("center_point_m", [0, 0])
                dims = target_shelf.get("dimensions_m", {"length": 8.0, "width": 0.6})
                orientation = target_shelf.get("orientation_rad", 0.0)
                
                # 随机层级 (1-3层)
                level = random.randint(0, 2)
                z_height = level * 0.75
                
                # 在货架局部坐标系内生成随机位置
                # 假设 length 是长边(Y)，width 是短边(X)
                half_len = dims["length"] / 2.0
                half_width = dims["width"] / 2.0
                
                local_y = random.uniform(-half_len + 0.5, half_len - 0.5) # 沿长边
                local_x = random.uniform(-half_width + 0.1, half_width - 0.1) # 沿短边
                
                # 旋转到世界坐标系
                cos_a = math.cos(orientation)
                sin_a = math.sin(orientation)
                
                # 2D 旋转变换
                world_x = center[0] + (local_x * cos_a - local_y * sin_a)
                world_y = center[1] + (local_x * sin_a + local_y * cos_a)
                
                package_position = (world_x, world_y, z_height)
                
                # 添加包裹到场景中
                package_usd_path = os.path.join(self.base_assets_path, "Props", "SM_CardBoxD_01_892.usd")
                package_prim_path = f"/World/Package_{self.package_counter:02d}"
                
                self.add_to_scene(
                    stage, 
                    package_usd_path, 
                    package_prim_path, 
                    position=package_position,  
                    rotation=(0, 0, math.degrees(orientation)), 
                    scale=random_scale,
                    add_physics=True
                )
                
                # 设置包裹质量
                package_prim = stage.GetPrimAtPath(package_prim_path)
                if package_prim:
                    mass_api = UsdPhysics.MassAPI.Apply(package_prim)
                    mass_api.CreateMassAttr(package_mass)
                    print(f"Package_{self.package_counter:02d}: scale={random_scale}, mass={package_mass:.3f}kg")
                
                self.package_counter += 1
        except Exception as e:
            print(f"Error adding package: {str(e)}")

    def add_robot(self, stage, robot_config):
        """Add configured robot to the scene."""
        try:
            usd_file = robot_config.get("usd_file", "ridgeback_franka.usd")
            robot_usd_path = os.path.join(self.base_assets_path, usd_file)
            target_prim_path = robot_config.get("prim_path", "/World/RidgebackFranka")

            spawn_cfg = robot_config.get("spawn", None)
            if spawn_cfg:
                position = tuple(spawn_cfg.get("position", [0.0, 0.0, 0.0]))
                rotation = tuple(spawn_cfg.get("rotation", [0.0, 0.0, 0.0]))
                scale = tuple(spawn_cfg.get("scale", [1.0, 1.0, 1.0]))
            else:
                position = (0.0, 0.0, 0.0)
                rotation = (0.0, 0.0, 0.0)
                scale = (1.0, 1.0, 1.0)
            
            self.add_to_scene(
                stage, 
                robot_usd_path, 
                target_prim_path, 
                position=position,
                rotation=rotation,
                scale=scale,
                add_physics=False  
            )
            print(f"Loaded robot '{self.current_robot_id}' from {usd_file}")
        except Exception as e:
            print(f"Error adding robot: {str(e)}")

    def add_ridgeback_franka(self, stage):
        """Backward-compatible wrapper for old call sites."""
        self.current_robot_id = "ridgeback_franka"
        self.current_robot_config = self._get_fallback_robot_config()
        self.add_robot(stage, self.current_robot_config)

    def add_camera(self, prim_path="/World/Camera", position=(0, 0, 4.5), 
                   orientation=(0, 0, 0), resolution=(1200, 1200), frequency=30):
        try:
            # Convert euler angles to quaternion, which is used by the Camera class
            orientation_quat = euler_angles_to_quat(np.array(orientation), degrees=True)
            
            camera = Camera(
                prim_path=prim_path,
                position=np.array(position),
                frequency=frequency,
                resolution=resolution,
                orientation=orientation_quat
            )
            camera.set_projection_mode("orthographic")
            camera.set_vertical_aperture(12.0)         
            camera.set_horizontal_aperture(12.0)
            camera.initialize()
            print(f"Successfully added and initialized Camera at prim path: {prim_path}")
            # Return the camera object so it can be initialized and used in the main script
            return camera
        except Exception as e:
            print(f"Error adding camera: {str(e)}")
            return None
