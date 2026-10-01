import numpy as np
import math
from scipy.spatial import ConvexHull
from pxr import Usd, UsdGeom, Gf
import omni.usd

def get_prim_local_points(prim):
    """
    Extract or generate local points for a given USD Prim.
    Supports Mesh, Cube, Sphere, Cylinder, and Capsule.
    """
    points = []
    
    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        if pts:
            points.extend(pts)
            
    elif prim.IsA(UsdGeom.Cube):
        cube = UsdGeom.Cube(prim)
        size = cube.GetSizeAttr().Get()
        if size is None: size = 2.0 # USD default
        s = size / 2.0
        # Generate 8 corner points of the cube
        points = [
            ( s,  s,  s), ( s,  s, -s), ( s, -s,  s), ( s, -s, -s),
            (-s,  s,  s), (-s,  s, -s), (-s, -s,  s), (-s, -s, -s)
        ]
        
    elif prim.IsA(UsdGeom.Cylinder) or prim.IsA(UsdGeom.Capsule):
        # We handle cylinder and capsule similarly by approximating the boundary
        geom = UsdGeom.Cylinder(prim) if prim.IsA(UsdGeom.Cylinder) else UsdGeom.Capsule(prim)
        radius = geom.GetRadiusAttr().Get()
        height = geom.GetHeightAttr().Get()
        axis = geom.GetAxisAttr().Get()
        
        r = radius if radius is not None else 1.0
        h = (height / 2.0) if height is not None else 1.0
        
        # Generate points around the top and bottom circles (e.g., 16 points)
        angles = np.linspace(0, 2 * math.pi, 16, endpoint=False)
        for a in angles:
            x, y = r * math.cos(a), r * math.sin(a)
            if axis == 'X':
                points.extend([(h, x, y), (-h, x, y)])
            elif axis == 'Y':
                points.extend([(x, h, y), (x, -h, y)])
            else: # Default is 'Z'
                points.extend([(x, y, h), (x, y, -h)])
                
    elif prim.IsA(UsdGeom.Sphere):
        sphere = UsdGeom.Sphere(prim)
        radius = sphere.GetRadiusAttr().Get()
        r = radius if radius is not None else 1.0
        # Generate points on 3 orthogonal planes to approximate sphere bounds
        angles = np.linspace(0, 2 * math.pi, 16, endpoint=False)
        for a in angles:
            points.append((r * math.cos(a), r * math.sin(a), 0))
            points.append((r * math.cos(a), 0, r * math.sin(a)))
            points.append((0, r * math.cos(a), r * math.sin(a)))

    return points

def get_robot_3d_points(root_prim_path):
    """
    Traverse all geometry primitives and return 3D points in the unscaled root frame.
    This keeps native XYZ coordinates (no 2D projection).
    """
    stage = omni.usd.get_context().get_stage()
    root_prim = stage.GetPrimAtPath(root_prim_path)
    
    if not root_prim.IsValid():
        print(f"[Error] Root prim not found: {root_prim_path}")
        return None

    # Use default time code to force real-time evaluation (bypasses cache)
    time_code = Usd.TimeCode.Default()

    # 1. Get Live World Transform of the Root Prim
    root_xform = UsdGeom.Xformable(root_prim)
    root_to_world_mat = root_xform.ComputeLocalToWorldTransform(time_code)
    
    # 2. Extract ONLY Rotation and Translation from Root
    root_translation = root_to_world_mat.ExtractTranslation()
    root_rotation = root_to_world_mat.ExtractRotation()
    
    # Create a new transform matrix with scale locked to 1.0
    unscaled_root_mat = Gf.Matrix4d(root_rotation, root_translation)
    world_to_root_unscaled_mat = unscaled_root_mat.GetInverse()

    all_points_3d = []

    # Traverse all descendants
    for prim in stage.TraverseAll():
        if prim.GetPath().HasPrefix(root_prim_path):
            local_points = get_prim_local_points(prim)
            
            if not local_points:
                continue
                
            # 3. Get Live World Transform of the Child Prim (includes its own scale)
            prim_xform = UsdGeom.Xformable(prim)
            prim_to_world_mat = prim_xform.ComputeLocalToWorldTransform(time_code)
            
            # 4. Transform to Unscaled Root Space
            prim_to_root_mat = prim_to_world_mat * world_to_root_unscaled_mat
            
            for p in local_points:
                p_vec = Gf.Vec3f(float(p[0]), float(p[1]), float(p[2]))
                p_root = prim_to_root_mat.Transform(p_vec)
                all_points_3d.append([p_root[0], p_root[1], p_root[2]])

    if not all_points_3d:
        print("[Error] No geometric points found.")
        return None

    return all_points_3d

def get_robot_2d_convex_hull(root_prim_path):
    """
    Traverse all geometry primitives, compute their LIVE world transforms (bypassing cache),
    and project them back to an UNSCALED local coordinate frame.
    """
    all_points_3d = get_robot_3d_points(root_prim_path)
    if all_points_3d is None:
        return None

    all_points_2d = [[p[0], p[1]] for p in all_points_3d]

    if len(all_points_2d) < 3:
        print("[Error] Not enough geometric points found to create a shape.")
        return None

    points_array = np.array(all_points_2d)

    # Compute 2D Convex Hull
    hull = ConvexHull(points_array)
    hull_points = points_array[hull.vertices]
    
    hull_points_closed = np.vstack((hull_points, hull_points[0]))
    return hull_points_closed

def get_projected_2d_footprint(root_prim_path):
    """
    Return projected 2D convex hull points without perimeter resampling.
    """
    hull_points_closed = get_robot_2d_convex_hull(root_prim_path)
    if hull_points_closed is None:
        return None

    return hull_points_closed.tolist()

def ray_intersect_segment(ray_angle, p1, p2):
    """Helper function to compute intersection distance of a ray and a line segment."""
    # Ray direction vector D
    D = np.array([np.cos(ray_angle), np.sin(ray_angle)])
    # Segment vector V
    V = p2 - p1
    
    # 2D Cross product: D x V
    cross_D_V = (D[0] * V[1]) - (D[1] * V[0])
    
    # If cross product is close to 0, ray and segment are parallel
    if abs(cross_D_V) < 1e-6:
        return float('inf')
        
    # t is the distance along the ray (from origin 0,0)
    cross_p1_V = (p1[0] * V[1]) - (p1[1] * V[0])
    t = cross_p1_V / cross_D_V
    # u is the parameter along the line segment (must be between 0 and 1)
    cross_p1_D = (p1[0] * D[1]) - (p1[1] * D[0])
    u = cross_p1_D / cross_D_V
    
    # Valid intersection occurs if t >= 0 (forward ray) and 0 <= u <= 1 (inside segment)
    if t >= 0 and 0 <= u <= 1:
        return t
        
    return float('inf')

def get_polar_ray_distances(root_prim_path, num_rays=12):
    hull_points_closed = get_robot_2d_convex_hull(root_prim_path)
    if hull_points_closed is None: return None

    angles = np.linspace(0, 2 * math.pi, num_rays, endpoint=False)
    distances = []

    for angle in angles:
        min_dist = float('inf')
        for i in range(len(hull_points_closed) - 1):
            p1 = hull_points_closed[i]
            p2 = hull_points_closed[i+1]
            dist = ray_intersect_segment(angle, p1, p2)
            if dist < min_dist:
                min_dist = dist
        distances.append(round(min_dist, 4))
        
    return distances

# ==========================================
# Example Execution
# ==========================================
ROBOT_ROOT_PATH = "/World/Robot" 

print(f"--- Processing Robot: {ROBOT_ROOT_PATH} ---")

points_3d = get_robot_3d_points(ROBOT_ROOT_PATH)
if points_3d:
    print(f"\n[Scheme 0] Native 3D Points (count={len(points_3d)}):")
    print(np.round(points_3d, 3).tolist())

footprint_2d = get_projected_2d_footprint(ROBOT_ROOT_PATH)
if footprint_2d:
    print("\n[Scheme 1] Projected 2D Footprint (Convex Hull):")
    formatted_points = np.round(footprint_2d, 3)
    print(formatted_points.tolist())

ray_distances_12 = get_polar_ray_distances(ROBOT_ROOT_PATH, num_rays=12)
if ray_distances_12:
    print("\n[Scheme 3] 12-Direction Polar Ray Distances:")
    print(ray_distances_12)
