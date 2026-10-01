import os, json, re, asyncio, math
from PIL import Image
import numpy as np
import omni.usd
import omni.timeline
import omni.kit.app
from pxr import UsdGeom, UsdPhysics, Usd, Gf
from isaacsim.sensors.camera import Camera
from isaacsim.core.prims import XFormPrim, RigidPrim

def extract_json(text):
    # 优先匹配 ```json 或 ``` 包裹的 JSON
    codeblock_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
    m = re.findall(codeblock_pattern, text, re.DOTALL | re.IGNORECASE)
    if m:
        json_str = m[-1].strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return json_str  # 返回原始字符串以便调试
    
    # 备用：找到文本中最后一个较大的花括号块（尽力匹配嵌套）
    brace_pattern = r'\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}'
    m2 = re.findall(brace_pattern, text, re.DOTALL)
    if m2:
        json_str = m2[-1].strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return json_str
    
    # 未找到 JSON，返回原始文本以便排查
    return text

def get_robot_info(robot_prim_path: str = "/World/RidgebackFranka"):
    try:
        robot_prim = XFormPrim(robot_prim_path)
        if not robot_prim.is_valid():
            return {"success": False, "error": f"Prim not found: {robot_prim_path}"}
        position, orientation = robot_prim.get_world_poses()
        return {
            "success": True,
            "robot_prim_path": robot_prim_path,
            "position": list(position[0]),
            "orientation": list(orientation[0]),
        }
    except Exception as exc:
        return {"success": False, "error": f"Robot info error: {exc}"}

def get_packages_info(package_prefix: str = "/World/Package_"):
    try:
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return {"success": False, "error": "Unable to get current stage"}

        packages = []
        for prim in stage.Traverse():
            path_str = prim.GetPath().pathString
            if not path_str.startswith(package_prefix):
                continue
            suffix = path_str[len(package_prefix):]
            if not suffix or "/" in suffix:
                continue

            rigid = RigidPrim(path_str)
            prim_wrapper = rigid if rigid.is_valid() else XFormPrim(path_str)
            if not prim_wrapper.is_valid():
                continue

            position, orientation = prim_wrapper.get_world_poses()
            scale = prim_wrapper.get_world_scales()
            mass = rigid.get_masses() if rigid.is_valid() else None

            packages.append(
                {
                    "name": suffix,
                    "position": list(position[0]),
                    "orientation": list(orientation[0]),
                    "scale": list(scale[0]),
                    "mass": float(mass) if mass is not None else None,
                }
            )

        return packages
    except Exception as exc:
        return {"success": False, "error": f"Package info error: {exc}"}

def get_static_obstacles(warehouse_file_path=None):
    """
    从Warehouse.txt文件中读取静态障碍物信息
    
    Args:
        warehouse_file_path (str, optional): Warehouse.txt文件路径
    
    Returns:
        list: 包含障碍物信息的列表
    """
    try:
        # 如果没有提供路径,使用默认路径
        if warehouse_file_path is None:
            ext_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            warehouse_file_path = os.path.join(ext_root, "Experiment", "Warehouse.txt")
        
        # 检查文件是否存在
        if not os.path.exists(warehouse_file_path):
            print(f"Warning: Warehouse file not found at {warehouse_file_path}")
            return []
        
        # 读取文件内容
        with open(warehouse_file_path, 'r', encoding='utf-8') as f:
            warehouse_data = json.load(f)
        
        geometry_model = warehouse_data.get("warehouse_geometry_model", {})
        obstacles_data = geometry_model.get("obstacles", [])
        
        if not obstacles_data:
            print("Warning: No obstacles found in warehouse file")
            return []
        
        obstacles = []
        
        for obs_data in obstacles_data:
            # 解析中心点坐标 (格式: "[-4.5, 0.0]")
            center_str = obs_data.get("center_point_m", "[0, 0]")
            center = json.loads(center_str)
            
            # 获取尺寸
            dimensions = obs_data.get("dimensions_m", {})
            length = dimensions.get("length", 0)
            width = dimensions.get("width", 0)
            
            # 获取障碍物多边形顶点
            polygon = obs_data.get("obstacle_polygon_m", [])
            
            # 计算轴对齐边界框(AABB)
            if polygon:
                x_coords = [p[0] for p in polygon]
                y_coords = [p[1] for p in polygon]
                x_min = min(x_coords)
                x_max = max(x_coords)
                y_min = min(y_coords)
                y_max = max(y_coords)
            else:
                # 如果没有多边形,从中心点和尺寸计算
                half_length = length / 2.0
                half_width = width / 2.0
                
                orientation = obs_data.get("orientation_rad", 0.0)
                
                if abs(orientation) < 0.01:  # 基本对齐坐标轴
                    x_min = center[0] - half_width
                    x_max = center[0] + half_width
                    y_min = center[1] - half_length
                    y_max = center[1] + half_length
                else:
                    # 如果有旋转,计算旋转后的边界框
                    cos_theta = math.cos(orientation)
                    sin_theta = math.sin(orientation)
                    
                    # 四个角点
                    corners = [
                        (half_width, half_length),
                        (-half_width, half_length),
                        (-half_width, -half_length),
                        (half_width, -half_length)
                    ]
                    
                    rotated_corners = []
                    for cx, cy in corners:
                        rx = cx * cos_theta - cy * sin_theta + center[0]
                        ry = cx * sin_theta + cy * cos_theta + center[1]
                        rotated_corners.append((rx, ry))
                    
                    x_min = min(c[0] for c in rotated_corners)
                    x_max = max(c[0] for c in rotated_corners)
                    y_min = min(c[1] for c in rotated_corners)
                    y_max = max(c[1] for c in rotated_corners)
            
            obstacle_info = {
                "path": f"/World/{obs_data.get('element_id', 'Unknown')}",
                "element_id": obs_data.get("element_id"),
                "type": obs_data.get("type", "Unknown"),
                "position": [center[0], center[1], 0.0],  # 添加Z坐标
                "dimensions": {
                    "length": length,
                    "width": width
                },
                "orientation_rad": obs_data.get("orientation_rad", 0.0),
                "polygon": polygon,
                "bbox_min": [x_min, y_min, 0.0],
                "bbox_max": [x_max, y_max, 2.0],  # 假设高度为2米
                "footprint": {
                    "x_min": x_min,
                    "x_max": x_max,
                    "y_min": y_min,
                    "y_max": y_max
                }
            }
            
            obstacles.append(obstacle_info)
        
        return obstacles
        
    except FileNotFoundError:
        print(f"Error: Warehouse file not found at {warehouse_file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse warehouse file: {e}")
        return []
    except Exception as e:
        print(f"Error loading static obstacles: {e}")
        import traceback
        traceback.print_exc()
        return []

async def capture_image(camera_source, output_path="auto", return_data=False):
    prims_to_hide = ["/World/Beam", "/World/Ceiling", "/World/LampCeiling", "/World/Pillar", "/World/Wall","/World/RidgebackFranka"]
    try:
        set_prim_visibility(prims_to_hide, visible=False)
        await asyncio.sleep(3.6)
        # update_async
        try:
            if hasattr(camera_source, 'get_rgb'):
                camera = camera_source
            elif hasattr(camera_source, 'overview_camera'):
                camera = camera_source.overview_camera
            else:
                return {'success': False, 'error': 'Invalid camera source'}
            
            if not camera:
                return {'success': False, 'error': 'Camera is None. Load scene first.'}
            
            timeline = omni.timeline.get_timeline_interface()
            if not timeline.is_playing():
                timeline.play()
                for _ in range(10):
                    await omni.kit.app.get_app().next_update_async()
            
            # camera.initialize()
            rgb_data = camera.get_rgb()
            if rgb_data is None:
                return {'success': False, 'error': 'Failed to get RGB data'}
            
            rgb_array = rgb_data.numpy() if hasattr(rgb_data, 'numpy') else rgb_data
            if rgb_array.dtype != 'uint8':
                rgb_array = (rgb_array * 255).astype('uint8')
            
            result = {'success': True, 'path': None, 'data': rgb_array if return_data else None, 'error': None}
            
            if output_path:
                if output_path == "auto":
                    ext_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    experiment_dir = os.path.join(ext_root, "Experiment")
                    os.makedirs(experiment_dir, exist_ok=True)
                    output_path = os.path.join(experiment_dir, "Overview_Capture.png")
                else:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                Image.fromarray(rgb_array, 'RGB').save(output_path)
                result['path'] = output_path
                print(f"Image captured successfully and saved to: {output_path}")
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': f'Capture error: {str(e)}'}
    finally:
        for _ in range(10):
            await omni.kit.app.get_app().next_update_async()
        set_prim_visibility(prims_to_hide, visible=True)


def set_prim_visibility(prim_paths, visible=True):
    try:
        stage = omni.usd.get_context().get_stage()
        if not stage:
            return {'success': False, 'error': 'Unable to get current stage'}
        
        if isinstance(prim_paths, str):
            prim_paths = [prim_paths]
            is_single = True
        else:
            is_single = False
        
        results = []
        success_count = 0
        
        for prim_path in prim_paths:
            try:
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    result = {'success': False, 'prim_path': prim_path, 'error': f'Prim not found at path: {prim_path}'}
                else:
                    imageable = UsdGeom.Imageable(prim)
                    
                    if visible:
                        vis_attr = imageable.GetVisibilityAttr()
                        if vis_attr:
                            vis_attr.Set(UsdGeom.Tokens.inherited)
                    else:
                        imageable.MakeInvisible()
                    
                    result = {'success': True, 'prim_path': prim_path, 'visible': visible, 'error': None}
                    success_count += 1
                
                results.append(result)
                
            except Exception as e:
                result = {'success': False, 'prim_path': prim_path, 'error': f'Error processing {prim_path}: {str(e)}'}
                results.append(result)
        
        if is_single:
            return results[0]
        else:
            return {
                'success': success_count == len(prim_paths),
                'total': len(prim_paths),
                'success_count': success_count,
                'results': results,
                'error': None if success_count == len(prim_paths) else f'Only {success_count}/{len(prim_paths)} operations succeeded'
            }
        
    except Exception as e:
        return {'success': False, 'error': f'Visibility error: {str(e)}'}


def extract_remaining_packages(failed_waypoint_index, original_plan, verbose=False):
    """
    从失败航点开始,提取所有未拾取的包裹ID
    """
    try:
        remaining_packages = []
        
        waypoints_data = original_plan.get("waypoint_plan", {}).get("waypoints", [])
        if not waypoints_data:
            return {
                'success': False,
                'remaining_packages': [],
                'total_packages': 0,
                'completed_packages': 0,
                'error': 'No waypoint data found in plan'
            }
        
        # 从失败航点开始,遍历剩余的所有航点
        for i in range(failed_waypoint_index, len(waypoints_data)):
            waypoint_info = waypoints_data[i]
            description = waypoint_info.get("description", "")
            
            # 检查描述中是否包含 "pick up Package" 关键字
            if "pick up Package" in description:
                # 提取包裹ID,例如从 "Optimal approach to pick up Package 06" 中提取 "06"
                match = re.search(r'Package\s+(\d+)', description)
                if match:
                    package_id = match.group(1)
                    if package_id not in remaining_packages:  # 避免重复
                        remaining_packages.append(package_id)
                        if verbose:
                            print(f"  → Package {package_id} needs to be picked (Waypoint {i})")
        
        # 计算总包裹数和已完成数
        pickup_sequence = original_plan.get("pickup_plan", {}).get("pickup_sequence", [])
        total_packages = len(pickup_sequence)
        completed_packages = total_packages - len(remaining_packages)
        
        return {
            'success': True,
            'remaining_packages': remaining_packages,
            'total_packages': total_packages,
            'completed_packages': completed_packages,
            'error': None
        }
        
    except Exception as e:
        return {
            'success': False,
            'remaining_packages': [],
            'total_packages': 0,
            'completed_packages': 0,
            'error': f'Error extracting remaining packages: {str(e)}'
        }
