import asyncio
import json
import math
import os
import re
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .Utils import extract_json

try:
    from shapely.affinity import rotate as shapely_rotate
    from shapely.affinity import translate as shapely_translate
    from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point, Polygon
    from shapely.ops import nearest_points
except Exception:
    shapely_rotate = None
    shapely_translate = None
    GeometryCollection = None
    LineString = None
    MultiPoint = None
    Point = None
    Polygon = None
    nearest_points = None


class RemoteCollisionPredictor:
    """Remote collision predictor client for waypoint checks."""

    def __init__(
        self,
        predictor_url: Optional[str] = None,
        timeout_s: Optional[float] = None,
        auth_token: Optional[str] = None,
        layout_path: Optional[str] = None,
    ):
        self.predictor_url = predictor_url or os.environ.get(
            "STP_PREDICTOR_URL", "http://127.0.0.1:8888/v1/chat/completions"
        )
        self.timeout_s = self._coerce_timeout(timeout_s)
        self.auth_token = auth_token if auth_token is not None else os.environ.get(
            "STP_PREDICTOR_TOKEN", ""
        )
        self.backend = os.environ.get("STP_PREDICTOR_BACKEND", "").strip().lower()
        self.vllm_model = os.environ.get("STP_VLLM_MODEL", "colpre")
        self.vllm_max_tokens = self._coerce_int(
            os.environ.get("STP_VLLM_MAX_TOKENS"), 256
        )
        self.vllm_temperature = self._coerce_float(
            os.environ.get("STP_VLLM_TEMPERATURE"), 0.0
        )
        extension_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.layout_path = layout_path or os.environ.get(
            "STP_LAYOUTS_PATH", os.path.join(extension_root, "Experiment", "layouts.json")
        )

    def _coerce_timeout(self, explicit_timeout: Optional[float]) -> float:
        if explicit_timeout is not None:
            try:
                return float(explicit_timeout)
            except (TypeError, ValueError):
                pass
        try:
            return float(os.environ.get("STP_PREDICTOR_TIMEOUT", "30"))
        except (TypeError, ValueError):
            return 30.0

    def _coerce_int(self, value: Optional[Any], default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _coerce_float(self, value: Optional[Any], default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _read_text_file(self, file_path: str) -> Optional[str]:
        if not file_path or not os.path.exists(file_path):
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return None

    def _load_json_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        raw_text = self._read_text_file(file_path)
        if raw_text is None:
            return None

        try:
            parsed = json.loads(raw_text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

        extracted = extract_json(raw_text)
        if isinstance(extracted, dict):
            return extracted
        if isinstance(extracted, str):
            try:
                parsed = json.loads(extracted)
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None
        return None

    def _get_default_robot_spec(self, layout_path: Optional[str] = None) -> Dict[str, Any]:
        path = layout_path or self.layout_path
        data = self._load_json_file(path) if path else None
        robot_cfg = data.get("robot_config", {}) if data else {}
        robots = robot_cfg.get("robots", {})
        default_robot_id = robot_cfg.get("default_robot")

        default_robot = robots.get(default_robot_id) if default_robot_id else None
        if default_robot is None and robots:
            default_robot_id = sorted(robots.keys())[0]
            default_robot = robots.get(default_robot_id)

        shape_info = default_robot.get("shape_info", {}) if default_robot else {}
        width = float(shape_info.get("width", 0.8))
        length = float(shape_info.get("length", 0.8))
        polygon = shape_info.get("polygon")

        return {
            "width": width,
            "length": length,
            "polygon": polygon,
            "robot_id": default_robot_id,
        }

    def _normalize_robot_size(
        self, robot_size: Optional[Dict[str, Any]], layout_path: Optional[str]
    ) -> Dict[str, Any]:
        defaults = self._get_default_robot_spec(layout_path)
        if not robot_size:
            return defaults

        width = float(robot_size.get("width", defaults.get("width", 0.8)))
        length = float(robot_size.get("length", defaults.get("length", 0.8)))
        polygon = robot_size.get("polygon", defaults.get("polygon"))
        robot_id = robot_size.get("robot_id", defaults.get("robot_id"))

        return {
            "width": width,
            "length": length,
            "polygon": polygon,
            "robot_id": robot_id,
        }

    def get_robot_size(self, layout_path: Optional[str] = None) -> Dict[str, Any]:
        return self._get_default_robot_spec(layout_path)

    def _extract_waypoints(self, plan_data: Dict[str, Any]) -> List[List[float]]:
        waypoints = []
        waypoints_data = plan_data.get("waypoint_plan", {}).get("waypoints", [])
        for wp in waypoints_data:
            coords = wp.get("coordinates_m", [])
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                try:
                    waypoints.append([float(coords[0]), float(coords[1])])
                except (TypeError, ValueError):
                    continue
        return waypoints

    def _parse_point(self, value: Any) -> List[float]:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                nums = re.findall(r"-?\d+(?:\.\d+)?", value)
                value = nums

        if not isinstance(value, (list, tuple)):
            return [0.0, 0.0]

        point: List[float] = []
        for v in list(value)[:2]:
            try:
                point.append(float(v))
            except (TypeError, ValueError):
                point.append(0.0)

        while len(point) < 2:
            point.append(0.0)
        return point

    def _to_float_list(self, value: Any) -> Optional[List[float]]:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        try:
            return [float(v) for v in list(value)]
        except Exception:
            return None

    def _format_footprint(self, polygon: Any) -> str:
        if not isinstance(polygon, list) or not polygon:
            return "  - Not provided."

        lines = []
        for idx, point in enumerate(polygon):
            xy = self._parse_point(point)
            lines.append(f"  - Vertex {idx}: ({xy[0]:.3f}, {xy[1]:.3f})")
        return "\n".join(lines)

    def _format_polygon_inline(self, polygon: Any) -> str:
        if not isinstance(polygon, list) or not polygon:
            return "Not provided"
        points = []
        for point in polygon:
            xy = self._parse_point(point)
            points.append(f"({xy[0]:.3f}, {xy[1]:.3f})")
        return "[" + ", ".join(points) + "]"

    def _obstacle_polygon(self, obs: Dict[str, Any]) -> List[List[float]]:
        polygon = obs.get("obstacle_polygon_m", [])
        parsed = []
        if isinstance(polygon, list):
            for point in polygon:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    try:
                        parsed.append([float(point[0]), float(point[1])])
                    except (TypeError, ValueError):
                        continue
        if len(parsed) >= 3:
            return parsed

        center = self._parse_point(obs.get("center_point_m", [0, 0]))
        dims = obs.get("dimensions_m", {})
        width = float(dims.get("width", 0))
        length = float(dims.get("length", dims.get("height", 0)))
        half_w = width / 2.0
        half_l = length / 2.0
        return [
            [center[0] - half_w, center[1] - half_l],
            [center[0] + half_w, center[1] - half_l],
            [center[0] + half_w, center[1] + half_l],
            [center[0] - half_w, center[1] + half_l],
        ]

    def _point_segment_distance(self, point: List[float], a: List[float], b: List[float]) -> float:
        px, py = point
        ax, ay = a
        bx, by = b
        dx = bx - ax
        dy = by - ay
        denom = dx * dx + dy * dy
        if denom <= 1e-12:
            return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
        cx = ax + t * dx
        cy = ay + t * dy
        return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5

    def _segments_intersect(self, a: List[float], b: List[float], c: List[float], d: List[float]) -> bool:
        def orient(p, q, r):
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

        def on_segment(p, q, r):
            return (
                min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9 and
                min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9
            )

        o1 = orient(a, b, c)
        o2 = orient(a, b, d)
        o3 = orient(c, d, a)
        o4 = orient(c, d, b)
        if o1 * o2 < 0 and o3 * o4 < 0:
            return True
        if abs(o1) < 1e-9 and on_segment(a, c, b):
            return True
        if abs(o2) < 1e-9 and on_segment(a, d, b):
            return True
        if abs(o3) < 1e-9 and on_segment(c, a, d):
            return True
        if abs(o4) < 1e-9 and on_segment(c, b, d):
            return True
        return False

    def _segment_distance(self, a: List[float], b: List[float], c: List[float], d: List[float]) -> float:
        if self._segments_intersect(a, b, c, d):
            return 0.0
        return min(
            self._point_segment_distance(a, c, d),
            self._point_segment_distance(b, c, d),
            self._point_segment_distance(c, a, b),
            self._point_segment_distance(d, a, b),
        )

    def _segment_polygon_distance(self, start: List[float], end: List[float], polygon: List[List[float]]) -> float:
        if len(polygon) < 2:
            return float("inf")
        distances = []
        for idx, point in enumerate(polygon):
            next_point = polygon[(idx + 1) % len(polygon)]
            distances.append(self._segment_distance(start, end, point, next_point))
        return min(distances) if distances else float("inf")

    def _build_segment_geometry_info(
        self, waypoints: List[List[float]], warehouse_data: Optional[Dict[str, Any]]
    ) -> str:
        if len(waypoints) < 2:
            return "## Path Segment Geometry Features\nNo path segments provided.\n"

        obstacles = []
        warehouse_model = warehouse_data.get("warehouse_geometry_model", {}) if warehouse_data else {}
        for obs in warehouse_model.get("obstacles", []):
            obstacles.append((obs.get("element_id", "unknown"), self._obstacle_polygon(obs)))

        lines = ["## Path Segment Geometry Features"]
        for idx in range(len(waypoints) - 1):
            start = waypoints[idx]
            end = waypoints[idx + 1]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = (dx * dx + dy * dy) ** 0.5
            heading = math.degrees(math.atan2(dy, dx)) if length > 1e-9 else 0.0
            nearest_id = "none"
            nearest_distance = float("inf")
            for obs_id, polygon in obstacles:
                dist = self._segment_polygon_distance(start, end, polygon)
                if dist < nearest_distance:
                    nearest_distance = dist
                    nearest_id = obs_id
            distance_text = "inf" if nearest_distance == float("inf") else f"{nearest_distance:.3f}"
            lines.append(
                f"- Segment {idx}->{idx + 1}: start=({start[0]:.2f}, {start[1]:.2f}), "
                f"end=({end[0]:.2f}, {end[1]:.2f}), length={length:.3f}m, "
                f"heading={heading:.1f}deg, nearest_obstacle={nearest_id}, "
                f"centerline_clearance={distance_text}m"
            )
        return "\n".join(lines) + "\n"

    def _canonical_obstacle_name(self, value: Any) -> str:
        text = str(value or "").lower()
        text = text.replace("_", "-")
        match = re.search(r"shelf[-_ ]*0*(\d+)", text)
        if match:
            return f"shelf-{int(match.group(1))}"
        return re.sub(r"[^a-z0-9]+", "-", text).strip("-")

    def _coerce_segment(
        self, prediction: Dict[str, Any], waypoints: List[List[float]]
    ) -> Optional[Tuple[int, int]]:
        if len(waypoints) < 2:
            return None

        raw_segment = prediction.get("collision_segment")
        if isinstance(raw_segment, str):
            nums = re.findall(r"-?\d+", raw_segment)
            raw_segment = [int(n) for n in nums[:2]]

        if isinstance(raw_segment, (list, tuple)) and len(raw_segment) >= 2:
            try:
                a = int(raw_segment[0])
                b = int(raw_segment[1])
                if a == b:
                    b = a + 1
                a, b = sorted((a, b))
                if 0 <= a < len(waypoints) - 1 and 1 <= b < len(waypoints):
                    return a, b
            except (TypeError, ValueError):
                pass

        idx = prediction.get("collision_waypoint_index")
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            return None

        if idx <= 0:
            return 0, 1
        if idx >= len(waypoints):
            return len(waypoints) - 2, len(waypoints) - 1
        return idx - 1, idx

    def _iter_obstacles(
        self, warehouse_data: Optional[Dict[str, Any]]
    ) -> List[Tuple[str, List[List[float]]]]:
        warehouse_model = warehouse_data.get("warehouse_geometry_model", {}) if warehouse_data else {}
        obstacles = []
        for obs in warehouse_model.get("obstacles", []):
            obs_id = str(obs.get("element_id", "unknown"))
            polygon = self._obstacle_polygon(obs)
            if len(polygon) >= 3:
                obstacles.append((obs_id, polygon))
        return obstacles

    def _select_obstacle_for_prediction(
        self,
        prediction: Dict[str, Any],
        waypoints: List[List[float]],
        segment: Tuple[int, int],
        warehouse_data: Optional[Dict[str, Any]],
    ) -> Optional[Tuple[str, List[List[float]], str]]:
        obstacles = self._iter_obstacles(warehouse_data)
        if not obstacles:
            return None

        raw_name = prediction.get("collision_object") or prediction.get("collision_object_id")
        canonical = self._canonical_obstacle_name(raw_name)
        if canonical:
            for obs_id, polygon in obstacles:
                if self._canonical_obstacle_name(obs_id) == canonical:
                    return obs_id, polygon, "model_object"
            for obs_id, polygon in obstacles:
                if canonical in self._canonical_obstacle_name(obs_id) or self._canonical_obstacle_name(obs_id) in canonical:
                    return obs_id, polygon, "model_object_partial"

        start = waypoints[segment[0]]
        end = waypoints[segment[1]]
        nearest = min(
            obstacles,
            key=lambda item: self._segment_polygon_distance(start, end, item[1]),
        )
        return nearest[0], nearest[1], "nearest_segment_obstacle"

    def _iter_segment_candidates(
        self, primary: Optional[Tuple[int, int]], waypoint_count: int
    ) -> List[Tuple[int, int]]:
        candidates: List[Tuple[int, int]] = []

        def add(a: int, b: int) -> None:
            if waypoint_count < 2:
                return
            if a == b:
                b = a + 1
            a, b = sorted((a, b))
            if 0 <= a < waypoint_count - 1 and 1 <= b < waypoint_count:
                pair = (a, b)
                if pair not in candidates:
                    candidates.append(pair)

        if primary is not None:
            add(primary[0], primary[1])
            add(primary[0] - 1, primary[1] - 1)
            add(primary[0] + 1, primary[1] + 1)

        for idx in range(waypoint_count - 1):
            add(idx, idx + 1)
        return candidates

    def _robot_polygon_at(
        self,
        robot_polygon: List[List[float]],
        x: float,
        y: float,
        heading_rad: float,
    ):
        poly = Polygon(robot_polygon)
        rotated = shapely_rotate(poly, math.degrees(heading_rad), origin=(0, 0), use_radians=False)
        return shapely_translate(rotated, xoff=x, yoff=y)

    def _representative_contact_point(self, geometry, robot_poly, obstacle_poly) -> Optional[List[float]]:
        if geometry is None or geometry.is_empty:
            try:
                p1, p2 = nearest_points(robot_poly.boundary, obstacle_poly.boundary)
                return [round((p1.x + p2.x) / 2.0, 4), round((p1.y + p2.y) / 2.0, 4)]
            except Exception:
                return None

        points = []
        if isinstance(geometry, Point):
            points.append(geometry)
        elif isinstance(geometry, MultiPoint):
            points.extend(list(geometry.geoms))
        elif isinstance(geometry, LineString):
            coords = list(geometry.coords)
            if coords:
                mid = geometry.interpolate(0.5, normalized=True)
                points.append(mid)
        elif isinstance(geometry, GeometryCollection):
            for geom in geometry.geoms:
                pt = self._representative_contact_point(geom, robot_poly, obstacle_poly)
                if pt is not None:
                    return pt

        if points:
            pt = points[0]
            return [round(pt.x, 4), round(pt.y, 4)]

        try:
            centroid = geometry.centroid
            return [round(centroid.x, 4), round(centroid.y, 4)]
        except Exception:
            return None

    def _refine_collision_position_with_shapely(
        self,
        prediction: Dict[str, Any],
        waypoints: List[List[float]],
        robot_size: Dict[str, Any],
        warehouse_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if Polygon is None:
            return {
                "position": None,
                "source": "model_raw",
                "error": "Shapely is not installed in this Python environment.",
            }

        if not prediction.get("collision_detected"):
            return {"position": None, "source": "not_collision", "error": None}

        robot_polygon = robot_size.get("polygon")
        if not isinstance(robot_polygon, list) or len(robot_polygon) < 3:
            return {"position": None, "source": "model_raw", "error": "Robot footprint polygon is missing."}

        primary_segment = self._coerce_segment(prediction, waypoints)
        if primary_segment is None:
            return {"position": None, "source": "model_raw", "error": "Collision segment is unavailable."}

        def refine_once(segment: Tuple[int, int], allow_nearest_object: bool):
            original_object = prediction.get("collision_object")
            if allow_nearest_object:
                prediction["collision_object"] = None
            selected = self._select_obstacle_for_prediction(prediction, waypoints, segment, warehouse_data)
            prediction["collision_object"] = original_object
            if selected is None:
                return None, "Obstacle polygon is unavailable."

            obstacle_id, obstacle_polygon, obstacle_source = selected
            try:
                obstacle_poly = Polygon(obstacle_polygon)
                if not obstacle_poly.is_valid:
                    obstacle_poly = obstacle_poly.buffer(0)
            except Exception as exc:
                return None, f"Invalid obstacle polygon: {exc}"

            start = waypoints[segment[0]]
            end = waypoints[segment[1]]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = (dx * dx + dy * dy) ** 0.5
            if length <= 1e-9:
                return None, "Collision segment length is zero."

            heading = math.atan2(dy, dx)

            def pose_at(t: float):
                return start[0] + dx * t, start[1] + dy * t

            def intersects_at(t: float) -> bool:
                x, y = pose_at(t)
                robot_poly = self._robot_polygon_at(robot_polygon, x, y, heading)
                return robot_poly.intersects(obstacle_poly)

            step_count = max(8, int(math.ceil(length / 0.02)))
            prev_t = 0.0
            first_t = None
            if intersects_at(0.0):
                first_t = 0.0
            else:
                for step in range(1, step_count + 1):
                    t = step / step_count
                    if intersects_at(t):
                        first_t = t
                        break
                    prev_t = t

            if first_t is None:
                return None, "Candidate segment does not geometrically intersect the selected obstacle."

            low = prev_t if first_t > 0.0 else 0.0
            high = first_t
            if high > low:
                for _ in range(16):
                    mid = (low + high) / 2.0
                    if intersects_at(mid):
                        high = mid
                    else:
                        low = mid

            x, y = pose_at(high)
            robot_poly = self._robot_polygon_at(robot_polygon, x, y, heading)
            boundary_intersection = robot_poly.boundary.intersection(obstacle_poly.boundary)
            contact_point = self._representative_contact_point(boundary_intersection, robot_poly, obstacle_poly)
            if contact_point is None:
                overlap = robot_poly.intersection(obstacle_poly)
                contact_point = self._representative_contact_point(overlap, robot_poly, obstacle_poly)
            if contact_point is None:
                contact_point = [round(x, 4), round(y, 4)]

            source = "geometry_refined_shapely"
            if segment != primary_segment or allow_nearest_object:
                source = "geometry_refined_shapely_fallback"
            return {
                "position": contact_point,
                "source": source,
                "error": None,
                "collision_segment": [segment[0], segment[1]],
                "collision_object": obstacle_id,
                "collision_object_source": obstacle_source,
                "first_intersection_t": round(high, 6),
                "fallback_used": source.endswith("_fallback"),
                "fallback_mode": "nearest_object" if allow_nearest_object else "model_object",
            }, None

        attempts = []
        for segment in self._iter_segment_candidates(primary_segment, len(waypoints)):
            for allow_nearest_object in (False, True):
                refined, error = refine_once(segment, allow_nearest_object)
                attempts.append({
                    "collision_segment": [segment[0], segment[1]],
                    "object_mode": "nearest_object" if allow_nearest_object else "model_object",
                    "success": refined is not None,
                    "error": error,
                })
                if refined is not None:
                    refined["attempts"] = attempts
                    return refined

        return {
            "position": None,
            "source": "model_raw",
            "error": "No candidate segment/object geometrically intersected.",
            "collision_segment": [primary_segment[0], primary_segment[1]],
            "attempts": attempts,
        }

    def _apply_collision_geometry_refinement(
        self,
        response: Dict[str, Any],
        waypoints: List[List[float]],
        robot_size: Dict[str, Any],
        warehouse_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not isinstance(response, dict):
            return response

        refinement = self._refine_collision_position_with_shapely(
            response,
            waypoints,
            robot_size,
            warehouse_data,
        )

        if refinement.get("position") is not None:
            response["model_collision_position"] = response.get("collision_position")
            response["collision_position"] = refinement["position"]
            response["collision_position_source"] = refinement["source"]
        else:
            response.setdefault("collision_position_source", refinement.get("source", "model_raw"))

        if refinement.get("collision_segment") is not None:
            response["collision_segment"] = refinement["collision_segment"]
        if refinement.get("collision_object") is not None:
            response["collision_object"] = refinement["collision_object"]
        response["geometry_refinement"] = refinement
        return self._populate_template_explanation(response)

    def _populate_template_explanation(self, response: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(response, dict):
            return response

        if response.get("collision_detected"):
            idx = response.get("collision_waypoint_index")
            segment = response.get("collision_segment")
            obstacle = response.get("collision_object") or "the selected obstacle"
            response.setdefault(
                "collision_reason",
                "The path is predicted to contact "
                + str(obstacle)
                + (f" on segment {segment}" if segment is not None else "")
                + (f" near waypoint {idx}." if idx is not None else "."),
            )
            response.setdefault(
                "avoidance_suggestion",
                "Re-plan the affected segment with a larger clearance margin and reduce speed near the predicted contact area.",
            )
            response.setdefault(
                "analysis",
                "Collision localization is based on the predicted segment/object and deterministic geometry refinement.",
            )
        else:
            response.setdefault("collision_reason", None)
            response.setdefault("avoidance_suggestion", None)
            response.setdefault("analysis", "The waypoint sequence is predicted to be safe. No collision risk detected.")
        return response

    def _build_env_info(
        self,
        warehouse_data: Optional[Dict[str, Any]],
        robot_size: Dict[str, Any],
    ) -> str:
        if not warehouse_data:
            return "## Environment Information\nWarehouse layout not provided.\n"

        warehouse_model = warehouse_data.get("warehouse_geometry_model", {})
        global_summary = warehouse_model.get("global_summary", {})
        warehouse_size = global_summary.get("warehouse_size_m", {"x": 0, "y": 0})
        obstacles = warehouse_model.get("obstacles", [])
        obstacle_count = global_summary.get("total_obstacles", len(obstacles))

        env_desc = (
            "## Environment Information\n"
            f"- Warehouse Size: {warehouse_size.get('x', 0)}m x {warehouse_size.get('y', 0)}m\n"
            "- Coordinate System: X-axis positive to the right, Y-axis positive down.\n"
            f"- Number of Obstacles: {obstacle_count}\n\n"
            "## Obstacle Details\n"
        )

        for obs in obstacles:
            center = self._parse_point(obs.get("center_point_m", [0, 0]))
            dims = obs.get("dimensions_m", {})
            polygon = obs.get("obstacle_polygon_m", [])
            x_coords = [float(pt[0]) for pt in polygon if isinstance(pt, (list, tuple)) and len(pt) >= 2]
            y_coords = [float(pt[1]) for pt in polygon if isinstance(pt, (list, tuple)) and len(pt) >= 2]
            if x_coords and y_coords:
                bbox_min = [min(x_coords), min(y_coords)]
                bbox_max = [max(x_coords), max(y_coords)]
            else:
                width = float(dims.get("width", 0))
                length = float(dims.get("length", dims.get("height", 0)))
                half_w = width / 2.0
                half_l = length / 2.0
                bbox_min = [center[0] - half_w, center[1] - half_l]
                bbox_max = [center[0] + half_w, center[1] + half_l]

            width = float(dims.get("width", 0))
            height = float(dims.get("height", dims.get("length", 0)))

            env_desc += (
                f"- {obs.get('element_id', 'unknown')} ({obs.get('type', 'unknown')}):\n"
                f"  - Center: ({center[0]}, {center[1]})\n"
                f"  - Dimensions: {width}m x {height}m\n"
                f"  - Bounding Box: X[{bbox_min[0]}, {bbox_max[0]}], Y[{bbox_min[1]}, {bbox_max[1]}]\n"
                f"  - Obstacle Polygon Vertices (m, ordered): {self._format_polygon_inline(self._obstacle_polygon(obs))}\n"
            )

        env_desc += (
            "\n## Robot Information\n"
            f"- Width: {robot_size.get('width', 0.8)}m\n"
            f"- Length: {robot_size.get('length', 0.8)}m\n"
            "- Footprint Vertices (m, robot frame, ordered):\n"
            f"{self._format_footprint(robot_size.get('polygon'))}\n"
        )

        return env_desc

    def _build_payload(
        self,
        waypoints: List[List[float]],
        robot_size: Dict[str, Any],
        robot_info: Optional[Dict[str, Any]],
        warehouse_data: Optional[Dict[str, Any]],
        env_info: str,
        plan_path: str,
        warehouse_path: str,
    ) -> Dict[str, Any]:
        robot_pose = None
        if robot_info and robot_info.get("success", True):
            position = self._to_float_list(robot_info.get("position"))
            orientation = self._to_float_list(robot_info.get("orientation"))
            robot_pose = {
                "position": position,
                "orientation": orientation,
            }

        return {
            "waypoints": waypoints,
            "robot": {
                "width": robot_size.get("width", 0.8),
                "length": robot_size.get("length", 0.8),
                "polygon": robot_size.get("polygon"),
                "robot_id": robot_size.get("robot_id"),
                "pose": robot_pose,
            },
            "environment": {
                "env_info": env_info,
                "warehouse_model": warehouse_data,
            },
            "source": {
                "plan_path": plan_path,
                "warehouse_path": warehouse_path,
            },
        }

    def _format_waypoints_for_prompt(self, waypoints: List[List[float]]) -> str:
        return "\n".join(
            f"  Waypoint {idx}: ({wp[0]:.2f}, {wp[1]:.2f})"
            for idx, wp in enumerate(waypoints)
        )

    def _build_inference_prompt(
        self,
        env_info: str,
        waypoints: List[List[float]],
        warehouse_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        waypoint_text = self._format_waypoints_for_prompt(waypoints)
        segment_geometry = self._build_segment_geometry_info(waypoints, warehouse_data)
        return f"""You are an expert in warehouse robot navigation safety analysis. Please predict if a collision will occur based on the environment information and waypoint sequence provided below.

{env_info}

## Waypoint Sequence to Analyze (Total {len(waypoints)} waypoints)
{waypoint_text}

{segment_geometry}

## Task
Analyze the waypoint sequence above and predict:
1. Whether a collision will occur (true/false)
2. If a collision occurs, prioritize predicting the path segment [start_waypoint_index, end_waypoint_index] where first contact occurs.
3. If a collision occurs, prioritize predicting the most likely obstacle element_id involved in the collision.
4. If a collision occurs, predict the index of the waypoint where it happens.
5. If a collision occurs, predict the approximate coordinate of the collision.
6. The collision_segment and collision_object fields are the primary localization outputs. Keep them consistent with the collision_position.

Please output the prediction results in the following JSON format:
{{
    "collision_detected": bool,
    "collision_segment": [int, int] or null,
    "collision_object": string or null,
    "collision_waypoint_index": int or null,
    "collision_position": [x, y] or null
}}
"""

    def _build_vllm_request(self, prompt: str) -> Dict[str, Any]:
        request = {
            "model": self.vllm_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.vllm_max_tokens,
            "temperature": self.vllm_temperature,
        }
        if self.vllm_temperature <= 0.0:
            request["top_p"] = 1.0
        return request

    def _detect_backend(self) -> str:
        if self.backend in {"vllm", "chat", "openai"}:
            return "vllm"
        if self.backend in {"predict", "custom"}:
            return "predict"

        url = (self.predictor_url or "").lower()
        if "/v1/chat/completions" in url:
            return "vllm"
        return "predict"

    def _extract_vllm_prediction(
        self, response_data: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        assistant_text = None
        try:
            assistant_text = response_data["choices"][0]["message"]["content"]
        except Exception:
            assistant_text = None

        if not isinstance(assistant_text, str):
            return None, None

        parsed = extract_json(assistant_text)
        if isinstance(parsed, dict):
            return assistant_text, parsed
        return assistant_text, None

    def _post_json_sync(
        self, url: str, payload: Dict[str, Any], timeout_s: float, auth_token: str
    ) -> Tuple[int, str]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, method="POST")
        request.add_header("Content-Type", "application/json")
        if auth_token:
            request.add_header("Authorization", f"Bearer {auth_token}")

        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                status = response.status
                body = response.read().decode("utf-8", errors="replace")
                return status, body
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return e.code, body

    async def _post_json_async(
        self, url: str, payload: Dict[str, Any], timeout_s: float, auth_token: str
    ) -> Tuple[int, str]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._post_json_sync, url, payload, timeout_s, auth_token
        )

    async def check_plan(
        self,
        plan_path: str,
        warehouse_path: str,
        robot_size: Optional[Dict[str, Any]] = None,
        robot_info: Optional[Dict[str, Any]] = None,
        layout_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        backend = self._detect_backend()
        report["backend"] = backend

        robot_size = self._normalize_robot_size(robot_size, layout_path)

        plan_data = self._load_json_file(plan_path)
        if plan_data is None:
            report["error"] = f"Plan file not found or invalid: {plan_path}"
            return report

        waypoints = self._extract_waypoints(plan_data)
        if not waypoints:
            report["error"] = "No valid waypoints found in plan file."
            return report

        warehouse_data = self._load_json_file(warehouse_path)
        if warehouse_data is None:
            report["error"] = f"Warehouse file not found or invalid: {warehouse_path}"
            return report

        env_info = self._build_env_info(warehouse_data, robot_size)
        payload = self._build_payload(
            waypoints,
            robot_size,
            robot_info,
            warehouse_data,
            env_info,
            plan_path,
            warehouse_path,
        )

        report["request"] = payload
        report["request_summary"] = {
            "waypoint_count": len(waypoints),
            "robot_id": robot_size.get("robot_id"),
            "robot_width_m": robot_size.get("width"),
            "robot_length_m": robot_size.get("length"),
            "plan_path": plan_path,
            "warehouse_path": warehouse_path,
        }
        prompt = self._build_inference_prompt(env_info, waypoints, warehouse_data)
        report["prompt"] = prompt

        if not self.predictor_url:
            report["error"] = "Predictor URL is not configured."
            return report

        if backend == "vllm":
            predictor_request = self._build_vllm_request(prompt)
        else:
            predictor_request = payload
        report["predictor_request"] = predictor_request

        try:
            status, response_text = await self._post_json_async(
                self.predictor_url,
                predictor_request,
                self.timeout_s,
                self.auth_token,
            )
        except Exception as exc:
            report["error"] = str(exc)
            return report

        try:
            response_data = json.loads(response_text)
        except Exception:
            response_data = extract_json(response_text)

        report["predictor_raw_response"] = response_data if isinstance(response_data, dict) else {
            "raw_response": response_text
        }

        report["http_status"] = status
        report["predictor_url"] = self.predictor_url
        if status < 200 or status >= 300:
            report["error"] = f"Predictor request failed with status {status}."
            report["response"] = report["predictor_raw_response"]
            return report

        if not isinstance(response_data, dict):
            report["error"] = "Predictor returned a non-JSON response."
            report["response"] = {"raw_response": response_text}
            return report

        if backend == "vllm":
            assistant_text, parsed_prediction = self._extract_vllm_prediction(response_data)
            report["assistant_text"] = assistant_text
            if isinstance(parsed_prediction, dict):
                report["response"] = self._apply_collision_geometry_refinement(
                    parsed_prediction,
                    waypoints,
                    robot_size,
                    warehouse_data,
                )
            else:
                report["response"] = {"raw_response": assistant_text or response_data}
            if parsed_prediction is None:
                report["error"] = "Could not parse prediction JSON from vLLM response."
        else:
            report["response"] = self._apply_collision_geometry_refinement(
                response_data,
                waypoints,
                robot_size,
                warehouse_data,
            )
        return report

    def save_report(self, report: Dict[str, Any], report_path: str) -> None:
        if not report_path:
            return
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
