try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from google import genai
    from google.genai import types
    _GENAI_IMPORT_ERROR = None
except Exception as e:
    genai = None
    types = None
    _GENAI_IMPORT_ERROR = e
import os, json, re
from .Utils import *

Scene_Prompt = """
# **Task: Automated Warehouse Geometry Modeling via Computer Vision**

## 1. Role and Goal

 **You are to act as a top-tier** **Computer Vision Engineer and AI Analyst**. Your mindset must be strictly code- and algorithm-driven, not based on visual intuition.

  **Your core mission is to** **write and execute Python code** **to perform a high-precision analysis of a provided top-down warehouse image. You must accomplish all steps** **programmatically, not through visual estimation**, to generate a structured JSON report that is geometrically sound, highly accurate, and ready for use in a robotic path-planning system.

## 2. Core Principles: Mandated Programmatic and Independent Execution

 **Primary Principle: Mandated Programmatic Method (Code-First Execution)**
Every numerical value you output (center points, vertex coordinates, etc.) **must be the direct result of Python code execution**. Your thought process must clearly demonstrate your code logic, intermediate calculations, and final results.

 **Principle 1: Strict Prohibition of Visual Estimation**
[CRITICAL INSTRUCTION] **You are strictly forbidden from "eyeballing," "guessing," or "intuitively assessing" any position, size, or coordinate.** **Treat the image as a pure numerical matrix of pixels. All your operations must be mathematical and code-based.**

 **Principle 2: Mandatory Use of Professional Tools**
[CRITICAL INSTRUCTION] **You must use professional Python computer vision libraries, specifically** **OpenCV (**cv2**) and Pillow (**PIL**)**, for this task. All obstacle identification, localization, and measurement must be performed by calling functions from these libraries.

 **Principle 3: Derivation over Imitation**
[CRITICAL INSTRUCTION] **You must operate on a "first principles" basis. The numerical values in the JSON example exist only to illustrate the format. You must derive all geometric data directly from your** **code-based analysis** **of the image.**

**Principle 4: Rejection of Pattern-Based Assumptions**
[CRITICAL INSTRUCTION] **The placement of objects in the warehouse is completely irregular.** **You must treat every detected contour as a completely independent entity.** **The calculation for one obstacle must never inform the calculation for another.**

## 3. Input Data & Coordinate System Reference

### 3.1. Input Data

* **You will be provided with a single top-down warehouse image.**

### 3.2. Coordinate Systems & Transformation Formulas [For Reference]

* **Physical World Coordinate System (World Frame):**

  * **Origin** **(0.0, 0.0)**: The exact physical center of the warehouse.
  * **X-Axis: Positive to the right. Y-Axis: Positive upwards. Unit: Meters (m).**
* **Image Pixel Coordinate System (Pixel Frame):**

  * **Origin** **(0, 0)**: The top-left corner of the image.
  * **p_x** **Axis: Positive to the right.** **p_y** **Axis: Positive downwards. Unit: Pixels (px).**
* **Core Transformation Formulas (To be applied in the workflow):**

  * **Scale_X = Image_Width_px / 12.0**
  * **Scale_Y = Image_Height_px / 12.0**
  * **w_x = (p_x_center - (Image_Width_px / 2)) / Scale_X**
  * **w_y = ((Image_Height_px / 2) - p_y_center) / Scale_Y**

## 4. Mandatory Algorithmic Workflow

**You must strictly follow the sequential steps below, writing and executing code for each. Show your code snippets and key outputs in your thought process.**

### Step 1: Image Property Analysis and Calibration

* **Load Image**: Use **Pillow** **or** **cv2** **to** **load the provided input image**.
* **Get Dimensions**: Obtain and print the exact image width (**Image_Width_px**) and height (**Image_Height_px**).
* **Calculate Scale Factors**: Based on the known physical dimensions (12.0m x 12.0m), calculate and state the values for **Scale_X** **and** **Scale_Y**.

### Step 2: Color-Based Obstacle Detection

* **Color Space Conversion**: Convert the image from the BGR (OpenCV default) or RGB color space to the HSV color space.
* **Define HSV Thresholds**: Define a reasonable HSV range to accurately isolate the yellow color of the shelves.
* **Create Binary Mask**: Use the **cv2.inRange** **function with your HSV thresholds to create a binary mask.**
* **(Optional but Recommended) Morphological Operations**: Use **cv2.morphologyEx** **with the** **MORPH_CLOSE** **operation to fill small holes in the mask and ensure solid detections.**
* **Find Contours**: Use the **cv2.findContours** **function on the final mask to find all separate obstacle outlines (contours).**
* **Calculate Bounding Boxes**: For **each** **contour found, use** **cv2.boundingRect** **to calculate its pixel-level bounding box** **(x, y, w, h)**.
* **Verification**: Print the list of detected bounding boxes **[(x1, y1, w1, h1), (x2, y2, w2, h2), ...]** **for verification.**

### Step 3: Coordinate Calculation and Transformation

* **Iterate through** **each bounding box** **found in the previous step and perform the following calculations independently:**
* **Calculate Pixel Center**:
  p_x_center = x + w / 2
  p_y_center = y + h / 2
* **Apply Transformation Formulas**: Use the **Image_Width_px**, **Image_Height_px**, **Scale_X**, and **Scale_Y** **from Step 1 to convert the pixel center** **(p_x_center, p_y_center)** **into the physical world coordinates** **(w_x, w_y)**.
* **Show Your Work**: **You must show the full calculation for at least one shelf in your thought process** **to prove you have followed the formulas. E.g.,** **Shelf-1 BBox: (x=..., y=..., w=..., h=...). Pixel Center: (px=..., py=...). Calculation: w_x = (...); w_y = (...)**.

### Step 4: Physical Geometry Modeling

* **For each calculated physical center point** **(w_x, w_y)**:
* **Define Properties**: Use the known shelf dimensions (length: 8.0m, width: 0.6m) and the specified orientation **orientation_rad = 0.0** **(long side parallel to the Y-axis).**
* **Calculate Vertices**: Centered on **(w_x, w_y)**, calculate the four corner vertices in the physical coordinate system.
* **Enforce Clockwise Order**: The vertices **must** **be listed in a consistent clockwise order (e.g., starting from the top-right vertex).**

### Step 5: Final JSON Output Assembly

* **Assemble Data**: Consolidate all calculated results into the specified JSON format.
* **Update Total Count**: Ensure the **total_obstacles** **field matches the exact** **number of contours** **you detected.**
* **Formatting**: Ensure the entire vertex array for **obstacle_polygon_m** **is on a** **single line**.

## 5. Output Format

**Your final answer must be** **only** **the JSON code block conforming to the structure below.**

```
{
  "warehouse_geometry_model": {
    "global_summary": {
      "warehouse_size_m": { "x": 12.0, "y": 12.0 },
      "coordinate_system_notes": "Origin at image center. X+ right, Y+ up.",
      "total_obstacles": 0
    },
    "obstacles": [
      {
        "element_id": "Shelf-1",
        "type": "Shelf",
        "is_ground_truth": false,
        "center_point_m": "[w_x_calculated, w_y_calculated]",
        "dimensions_m": { "length": 8.0, "width": 0.6 },
        "orientation_rad": 0.0,
        "obstacle_polygon_m": [ [w_x_v1, w_y_v1], [w_x_v2, w_y_v2], [w_x_v3, w_y_v3], [w_x_v4, w_y_v4] ]
      }
    ]
  }
}
```
"""

def get_plan_prompt(warehouse_geometry_model, packages_info, robot_info, mission_package_ids):
    """
    Generates a formatted prompt string for the LLM planner.

    Args:
        warehouse_geometry_model (str): A JSON string representing the warehouse layout.
        package_pickup_task (str): A JSON string representing the package pickup task.

    Returns:
        str: The complete, formatted prompt for the LLM.
    """
    if mission_package_ids is None:
      mission_package_ids = [pkg['name'] for pkg in packages_info]
    
    return f"""
# [ROLE]
You are an expert robotics software engineer specializing in logistics optimization and motion planning for Autonomous Mobile Robots (AMRs) in warehouse environments. Your task is to process raw sensor and simulation data to create a complete, executable plan.

# [CONTEXT]
You are tasked with creating an operational plan for an AMR. You will be given three pieces of critical information:
1.  **Warehouse Environment Model**: A JSON object describing the static layout of the warehouse, including its dimensions and all obstacles (shelves). The coordinate system origin is at the center of the warehouse, with X+ pointing right and Y+ pointing up.
2.  **Robot and Mission Parameters**: A JSON object containing all dynamic information for the current task. This includes:
    *   `packages_to_pickup`: A master list of all available packages in the environment.
    *   `mission_package_ids`: A specific list of package IDs that must be collected in *this* mission. **You will only plan for these packages.**

## 1. Warehouse Environment Model (JSON)

{warehouse_geometry_model}

## 2. Robot and Mission Parameters (JSON)

{{
  "mission_id": "M-2048",
  "robot_specifications": {{
    "body_dimensions_m": {{ "length": 0.9, "width": 0.9 }},
    "start_position_m": {robot_info['position']}
  }},
  "packages_to_pickup": {packages_info},
  "mission_package_ids": {mission_package_ids},
  "drop_off_position_m": [-5.0, 5.0]
}}


# [GOAL]
Your mission is to process the raw data and generate a complete, two-part operational plan for the packages listed in `mission_package_ids` ONLY:
1.  An optimized **pickup sequence** for the mission packages.
2.  A detailed, **collision-free waypoint path** for the robot to follow that sequence, intelligently selecting the most efficient side of a shelf for each pickup.

# [STEP-BY-STEP INSTRUCTIONS]

## Step 1: Data Interpretation and Pre-processing
First, you must **filter** the master `packages_to_pickup` list to work only with the packages whose `name` appears in the `mission_package_ids` list. For each package in this mission-specific subset, derive the following simplified planning attributes:
*   **`package_id`**: The `name` of the package.
*   **`location_m`**: The package's 2D location `[x, y]` from its `position` array.
*   **`weight_kg`**: The `mass` value.
*   **`size_category`**: A categorical size derived from its `scale` array. Calculate volume (`scale[0] * scale[1] * scale[2]`) and classify:
    *   `volume < 0.06 m³`: **"small"**
    *   `0.06 m³ <= volume < 0.09 m³`: **"medium"**
    *   `volume >= 0.09 m³`: **"large"**

## Step 2: Determine the Optimal Pickup Sequence
Using the processed data for the **mission packages**, determine the most efficient and stable pickup order. Your reasoning must balance two factors:
*   **Logistical Efficiency (Geographic Clustering)**: Minimize total travel distance by grouping nearby packages.
*   **Stability and Stacking (Heavy/Large First Principle)**: To create a stable base, heavier and larger items **MUST be picked up FIRST**. Prioritize "large" or "medium" packages with higher `weight_kg` early in the sequence. Lighter, smaller packages should be picked up later.

## Step 3: Generate the Collision-Free Waypoint Path
Based on your determined pickup sequence, generate a precise list of waypoints. The path MUST adhere to the following strict rules:

### 3a. Core Constraint: Robot Collision Avoidance
*   The robot's body is **0.9m x 0.9m** (radius `0.45m`). With an additional safety buffer of **0.15m**, the **centerline of the robot's path must ALWAYS be at least `0.6m` away from any obstacle polygon boundary**.

### 3b. Critical Logic: Determine Optimal Pickup Approach Points
For each package in your sequence, you must decide on the most efficient approach point. This involves a crucial choice:
*   **Dual-Sided Access**: Since shelves are narrow, a package can be picked from the aisle on **either side** of the shelf.
*   **Decision Process for Each Pickup**:
    1.  Let the robot's current location (or previous waypoint) be `P_current`. Let the target package be at `P_target`.
    2.  Identify the shelf the package is on and the two adjacent aisles (e.g., Aisle-Left and Aisle-Right).
    3.  Calculate **Approach Point A**: The valid pickup point in Aisle-Left. This is offset from the package location by **0.6m** into the aisle.
    4.  Calculate **Approach Point B**: The valid pickup point in Aisle-Right, also offset by **0.6m**.
    5.  **Select the Best Point**: Calculate the path distance from `P_current` to `Approach Point A` and from `P_current` to `Approach Point B`. Choose the approach point that results in the shorter travel distance. This is your next waypoint.
*   **Example**: The robot is at `P_current = [-0.9, 3.0]`. The next target package is at `P_target = [1.2, -4.0]` on Shelf-2 (which spans x from 0.905 to 1.505).
    *   `Approach Point A` (Left Aisle): `[1.2 - 0.6, -4.0] = [0.6, -4.0]`.
    *   `Approach Point B` (Right Aisle): `[1.2 + 0.6, -4.0] = [1.8, -4.0]`.
    *   The robot decides: The path from `[-0.9, 3.0]` to `[0.6, -4.0]` is significantly shorter than going all the way around to `[1.8, -4.0]`. It will select `Approach Point A`.

### 3c. Construct the Final Waypoint List
*   The path must start at `start_position_m`.
*   For each package in the sequence, determine its optimal `pickup_approach_point_m` using the logic in 3b and add it to the path.
*   Add intermediate waypoints as needed to navigate around corners and down aisles, always respecting the **0.6m total clearance**.
*   The path must end at the `drop_off_position_m`.
*   The "description" for a pickup waypoint should clearly state that it is an "approach point".

# [OUTPUT FORMAT]
You MUST provide your final plan in a single, well-formed JSON object. Do not include any text or explanations outside of this JSON structure.

{{
  "pickup_plan": {{
    "pickup_sequence": [
      {{ "order": 1, "package_id": "..." }},
      {{ "order": 2, "package_id": "..." }}
    ],
    "reasoning": "The plan prioritizes picking up the heaviest/largest mission packages first for stability. The path is highly optimized by dynamically selecting the most efficient approach side for each shelf pickup, minimizing travel time while adhering to all safety constraints."
  }},
  "waypoint_plan": {{
    "waypoints": [
      {{ "waypoint_id": 0, "coordinates_m": [0.0, 0.0], "description": "Start Position" }},
      {{ "waypoint_id": 1, "coordinates_m": [ ... ], "description": "Navigate..." }},
      {{ "waypoint_id": 2, "coordinates_m": [ ... ], "description": "Optimal approach to pick up Package ..." }}
    ]
  }}
}}

"""

#def get_replan_prompt()

class LM:
    def __init__(self, key="AIzaSyChpMrBrNV7lDFjMQv6HXGfuwI4GM4GHdY", sys_prompt=None):
        if genai is None or types is None:
            raise RuntimeError(
                "google-genai is unavailable or incompatible in Isaac Sim Python. "
                "Please install a compatible version (recommended: google-genai<1.70)."
            ) from _GENAI_IMPORT_ERROR
        self.client = genai.Client(api_key=key)
        self.sys_prompt = sys_prompt
        self.model_name = "gemini-2.5-pro"

    def reasoning(self, prompt, image_path=None, json_only=True):
        content_parts = []
        
        if self.sys_prompt:
            content_parts.append(f"System Instructions: {self.sys_prompt}\n\n")
        
        # 如果提供了图像路径,上传图像文件
        if image_path:
            try:
                if not os.path.exists(image_path):
                    return f"Error: Image file not found at {image_path}"
                
                my_file = self.client.files.upload(file=image_path)
                content_parts.append(my_file)
                print(f"[LM] Successfully uploaded image: {image_path}")
            except Exception as e:
                return f"Error: Failed to upload image {image_path}. Details: {str(e)}"
        
        # 添加用户提示
        content_parts.append(prompt)
        
        # 配置代码执行工具
        config = types.GenerateContentConfig(
            tools=[types.Tool(code_execution=types.ToolCodeExecution())]
        )
        
        try:
            # 生成内容
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=content_parts,
                config=config
            )
            
            # 仅收集模型的文本输出部分（避免从代码执行重复抓取 JSON）
            text_parts = []
            
            # 优先使用第一个候选（通常为主响应）
            if not response.candidates:
                return "Error: No candidates returned from model."
            candidate = response.candidates[0]
            
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text is not None:
                    text_parts.append(part.text)
            
            response_text = "".join(text_parts)
            
            if json_only:
                return extract_json(response_text)
            
            # 如果不要求只取 JSON，则同时保留文本、代码与执行输出
            full_response = [response_text]
            for part in candidate.content.parts:
                if hasattr(part, 'executable_code') and part.executable_code is not None:
                    code = part.executable_code.code
                    full_response.append(f"\n[Generated Code]:\n```python\n{code}\n```\n")
                if hasattr(part, 'code_execution_result') and part.code_execution_result is not None:
                    output = part.code_execution_result.output
                    full_response.append(f"\n[Execution Output]:\n{output}\n")
            
            return "".join(full_response)
            
        except Exception as e:
            return f"Error during content generation: {str(e)}"

async def replan_from_failure(failure_info, extension_root):
    """
    基于导航失败信息进行重新规划
    
    Args:
        failure_info (dict): 失败信息,包含:
            - type: 失败类型
            - current_position: 当前位置 (x, y)
            - failed_waypoint_index: 失败的航点索引
            - failed_waypoint: 失败的航点坐标
            - total_waypoints: 总航点数
            - completed_waypoints: 已完成的航点列表
        extension_root (str): 扩展根目录路径
    
    Returns:
        dict: 重规划结果,包含:
            - success (bool): 是否成功
            - new_waypoints (list): 新的航点列表 [(x, y), ...]
            - remaining_packages (list): 剩余包裹ID列表
            - plan_data (dict): 完整的规划JSON数据
            - error (str): 错误信息(如果有)
    """
    try:
        print("\n" + "="*60)
        print("REPLANNING PROCESS INITIATED")
        print("="*60)
        
        # 1. 获取当前状态信息
        current_position = failure_info['current_position']
        failed_waypoint_index = failure_info['failed_waypoint_index']
        
        print(f"Failure Analysis:")
        print(f"  Type: {failure_info['type']}")
        print(f"  Failed at waypoint: {failed_waypoint_index}/{failure_info['total_waypoints']}")
        print(f"  Current position: ({current_position[0]:.3f}, {current_position[1]:.3f})")
        
        # 2. 加载原始计划
        plan_path = os.path.join(extension_root, "Experiment", "Pick_WayPoint.txt")
        try:
            with open(plan_path, 'r', encoding='utf-8') as f:
                original_plan = json.load(f)
        except Exception as e:
            error_msg = f"Error loading original plan: {e}"
            print(f"{error_msg}")
            return {
                'success': False,
                'new_waypoints': [],
                'remaining_packages': [],
                'plan_data': None,
                'error': error_msg
            }
        
        # 3. 从失败航点开始提取所有未拾取的包裹
        print(f"\nAnalyzing remaining pickup tasks...")
        extraction_result = extract_remaining_packages(failed_waypoint_index, original_plan, verbose=True)
        
        if not extraction_result['success']:
            print(f"{extraction_result['error']}")
            return {
                'success': False,
                'new_waypoints': [],
                'remaining_packages': [],
                'plan_data': None,
                'error': extraction_result['error']
            }
        
        remaining_packages = extraction_result['remaining_packages']
        
        if not remaining_packages:
            print("\n" + "="*60)
            print("✓ ALL PACKAGES COLLECTED")
            print("="*60)
            print("No remaining packages to pick up.")
            print("Mission can be considered complete!")
            print("="*60 + "\n")
            return {
                'success': True,
                'new_waypoints': [],
                'remaining_packages': [],
                'plan_data': None,
                'error': 'Mission complete - no packages remaining'
            }
        
        # 4. 显示任务进度
        print(f"\nMission Progress:")
        print(f"  Total packages: {extraction_result['total_packages']}")
        print(f"  Completed: {extraction_result['completed_packages']}")
        print(f"  Remaining: {len(remaining_packages)}")
        print(f"  Remaining IDs: {remaining_packages}")
        
        # 5. 加载环境模型
        warehouse_model_file_path = os.path.join(extension_root, "Experiment", "Warehouse.txt")
        try:
            with open(warehouse_model_file_path, 'r', encoding='utf-8') as f:
                warehouse_geometry_model = f.read()
        except FileNotFoundError:
            error_msg = f"Warehouse model file not found at {warehouse_model_file_path}"
            print(f"Error: {error_msg}")
            return {
                'success': False,
                'new_waypoints': [],
                'remaining_packages': remaining_packages,
                'plan_data': None,
                'error': error_msg
            }
        except Exception as e:
            error_msg = f"Error reading warehouse model file: {e}"
            print(f"{error_msg}")
            return {
                'success': False,
                'new_waypoints': [],
                'remaining_packages': remaining_packages,
                'plan_data': None,
                'error': error_msg
            }
        
        # 6. 获取最新的包裹信息
        packages_info = get_packages_info()
        
        # 构造新的机器人信息,使用当前失败位置作为起点
        robot_info = {
            'position': list(current_position) + [0.0]  # 添加z坐标
        }
        
        print(f"\nReplanning Configuration:")
        print(f"  New start position: ({current_position[0]:.3f}, {current_position[1]:.3f})")
        print(f"  Target packages: {remaining_packages}")
        
        # 7. 生成新的规划提示并调用LLM
        print("\n" + "-"*60)
        print("Generating new plan with LLM...")
        print("-"*60)
        
        plan_prompt = get_plan_prompt(
            warehouse_geometry_model, 
            packages_info, 
            robot_info, 
            mission_package_ids=remaining_packages
        )
        
        llm = LM()
        model_response = llm.reasoning(plan_prompt, image_path=None, json_only=True)
        
        if not isinstance(model_response, dict):
            error_msg = "LLM replanning failed. Response was not a valid JSON object."
            print(f"{error_msg}")
            print("LLM Response:", model_response)
            return {
                'success': False,
                'new_waypoints': [],
                'remaining_packages': remaining_packages,
                'plan_data': None,
                'error': error_msg
            }
        
        # 8. 保存新的规划结果
        replan_output_path = os.path.join(extension_root, "Experiment", "Replan_WayPoint.txt")
        with open(replan_output_path, 'w', encoding='utf-8') as f:
            json.dump(model_response, f, indent=2)
        
        print(f"✓ Replanning successful!")
        print(f"  New plan saved to: {replan_output_path}")
        
        # 9. 显示新规划的摘要
        new_waypoints_data = model_response.get("waypoint_plan", {}).get("waypoints", [])
        new_sequence = model_response.get("pickup_plan", {}).get("pickup_sequence", [])
        
        print(f"\nNew Plan Summary:")
        print(f"  Pickup sequence: {[pkg['package_id'] for pkg in new_sequence]}")
        print(f"  Total waypoints: {len(new_waypoints_data)}")
        
        # 10. 提取新的航点列表
        if not new_waypoints_data:
            error_msg = "New waypoint data is missing or empty."
            print(f"Error: {error_msg}")
            return {
                'success': False,
                'new_waypoints': [],
                'remaining_packages': remaining_packages,
                'plan_data': model_response,
                'error': error_msg
            }
        
        new_waypoints = [tuple(w["coordinates_m"]) for w in new_waypoints_data]
        
        print("\n" + "="*60)
        print("REPLANNING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"New start position: ({current_position[0]:.3f}, {current_position[1]:.3f})")
        print(f"Waypoints generated: {len(new_waypoints)}")
        print(f"Packages to collect: {len(remaining_packages)}")
        print("="*60 + "\n")
        
        return {
            'success': True,
            'new_waypoints': new_waypoints,
            'remaining_packages': remaining_packages,
            'plan_data': model_response,
            'error': None
        }
        
    except Exception as e:
        import traceback
        error_msg = f"Error during replanning: {str(e)}"
        print(f"{error_msg}")
        traceback.print_exc()
        return {
            'success': False,
            'new_waypoints': [],
            'remaining_packages': [],
            'plan_data': None,
            'error': error_msg
        }
