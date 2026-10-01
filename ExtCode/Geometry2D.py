"""Small dependency-free helpers for world-space XY collider geometry."""


def convex_hull_2d(points):
    """Return a counter-clockwise convex hull without a repeated endpoint."""
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


def cube_local_corners(size):
    """Return all local-space corners of a USD Cube with the authored size."""
    half = float(size) / 2.0
    return [
        (x, y, z)
        for x in (-half, half)
        for y in (-half, half)
        for z in (-half, half)
    ]


def transformed_xy_convex_hull(points, transform_point):
    """Transform 3-D points to world space and project their XY convex hull."""
    world_xy = []
    for point in points:
        world = transform_point(point)
        world_xy.append((float(world[0]), float(world[1])))
    return convex_hull_2d(world_xy)
