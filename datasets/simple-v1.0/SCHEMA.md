# Simple v1.0 schema / 数据字典

JSONL files contain one JSON object per line; join by `sample_id`, never row
number. IDs are SHA-256-derived opaque identifiers and carry no time ordering.
All positions and dimensions are meters; angles are radians. World XY origin
is warehouse center, X+ right, Y+ up. Robot footprints use robot-local XY
coordinates. Robot length is along local X and width along local Y.

## Inputs

`sample_id`: stable ID. `motion_mode`: always `simple`.
`scene_id`: key in scenes.json. `waypoints_m`: ordered array of [x,y] pairs.
`robot.id`: recorded robot category. `robot.size_m`: width/length.
`robot.footprint_local_m`: original `polygon` and/or `polygon_components` when
available. Each polygon is an ordered array of [x,y] vertices. No rounding,
simplification, hull regeneration or coordinate conversion is performed.

## Scenes

`warehouse_size_m`: full extent {x,y}. `obstacles`: shelves with `element_id`,
`center_point_m`, `dimensions_m`, `orientation_rad`, `obstacle_polygon_m`.
Polygons use world XY. Scalar dimensions preserve source conventions: for
shelves, at zero orientation the long axis is world Y (local width along X,
length along Y). Robot and shelf dimension conventions must not be conflated.
Prefer the explicit polygon when using recorded obstacle geometry. Simplified
layout geometry is not equivalent to the excluded physical collision meshes.

## Labels

- `collision`: true when a collision was recorded, false for completed
  noncollision navigation retained in this release.
- `collision_waypoint_index`: integer k; endpoints are `waypoints_m[k-1]`
  and `waypoints_m[k]`. Thus k ranges from 1 to N-1 using zero-based arrays.
- `collision_segment`: explicit zero-based endpoint indices [k-1,k].
- `collision_position_m`: recorded contact point [x,y], **not robot center**.
- `robot_center_at_collision_m`: recorded robot center at collision.
- `collision_object_id`: original object identifier. Shelf IDs match scene
  `element_id`. Wall collisions retain an original `/World/...` prim identifier;
  these are simulator object IDs, not filesystem paths. They are not necessarily
  keys in the shelf-only obstacles list. No wall IDs are silently merged.
- `collision_fields_valid`: true for collision samples. For negative records,
  all collision-only fields above are null; false validity is not a bad sample.

Labels preserve source positive values, including contacts associated with a
turn or segment transition. Do not infer an exact executed heading solely from
the planned segment. No new repair, successful execution or geometric clearance
annotation is generated. Generation targets and executed trajectories are
intentionally separate from the public input schema to avoid outcome leakage.

## 划分与来源

`sample_id` 用于连接输入和标签；`scene_id` 引用场景。全部坐标单位为米，角度为弧度。
碰撞索引 k 对应数组中第 k-1 到第 k 个路径点，`collision_segment` 明确保存两个零基索引。
`collision_position_m` 是接触点，不是机器人中心。负例碰撞专属字段统一为 null。
墙壁对象保留原始仿真对象 ID，不能当作货架列表中必然存在的对象。

`splits.json` 含六个互不重叠的集合；`examples.json` 是 train 中记录的副本。
`exclusions.json` 中记录不属于任何发布划分。原始记录包含 2,700 条未记录扩展
诊断字段的旧样本；发布沿用既有资格规则，不捏造诊断值。源文件校验值只用于
溯源，最终包的文件校验应使用 SHA256SUMS。
