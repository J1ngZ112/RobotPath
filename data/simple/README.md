# Simple 原始采集数据 / Raw Simple data

本目录直接提供 24 个原始采集 JSON 文件，共 **7,200 条记录**：
4,745 条碰撞、2,455 条未碰撞。Simple 表示机器人先转向、后移动。
文件名、结构、字段和数值均与原始采集文件一致，没有拆分或重新编码。
标注正确性已由作者验证。Hard 数据、模型权重和训练代码不在本次发布范围内。

每个文件包含：

- `metadata`：机器人、仓库布局、采集配置与文件级统计。
- `sequences`：路径点、碰撞标注、执行轨迹和诊断信息。

其中 170 条记录标记为 `training_eligible=false`，原因为
`fault_injection_miss`。保留这些原始记录，但复用原实验数据口径时应排除，
剩余 7,030 条。旧记录可能缺少 `training_eligible`；原加载规则默认视为 true。
故障注入未命中不表示标注错误。生成提示与执行结果不应作为碰撞预测输入。

原迁移测试文件（均保留在本目录，不另行移动）：

| 文件 | 用途 | 数量 |
|---|---|---:|
| `collision_sequences_final_20260821_120455.json` | 布局迁移 | 300 |
| `collision_sequences_final_20260824_123844.json` | 联合迁移 | 300 |
| `collision_sequences_final_20260825_005353.json` | 机器人迁移 | 300 |

其余 21 个文件为主集合。不要将上述迁移测试文件混入训练。
坐标单位为米、角度为弧度；`collision_position` 为接触点，
`robot_center_at_collision` 为机器人中心。`collision_segment=[k-1,k]`
对应 `waypoint_sequence` 的两个零基索引。部分墙壁对象使用 `/World/...`
仿真对象路径，不是本地文件路径。

从仓库根目录读取，不需要 Isaac Sim 或 API key：

```python
import json
from pathlib import Path

path = Path("data/simple/collision_sequences_final_20260807_164648.json")
with path.open(encoding="utf-8") as f:
    data = json.load(f)
print(data["metadata"]["robot_info"])
print(len(data["sequences"]))
eligible = [s for s in data["sequences"] if s.get("training_eligible", True)]
```

数据使用 [CC BY 4.0](LICENSE)。作者已确认数据中几何的再分发许可，
许可依据待补充。JSON 中保留采集得到的几何描述，但不包含 USD、材质、纹理文件。
这些数据不提供完整仿真场景，也不意味着私有模型结果可直接复现。
原文件可能含采集时间和仿真路径，当前个人 GitHub 仓库不是匿名审稿入口。

## English

These are 24 unmodified acquisition JSON files: 7,200 records, including
4,745 collisions and 2,455 noncollisions. Simple means rotate first, then
translate. Read `metadata` and `sequences` directly with Python's `json` module.
No simulator, GPU or API key is needed to read the files.

Keep the original 170 `training_eligible=false` fault-injection misses for
provenance, but exclude them to obtain the 7,030-record eligible population.
Missing eligibility flags default to true under the original loading policy.
The three transfer files listed above are held out; all other files form the
main collection. Do not use generation hints or execution outcomes as inputs.

Data is CC BY 4.0. The author confirmed geometry redistribution permission;
supporting evidence will be added. Model weights, training code, Hard data and
original asset files are not included. This release does not reproduce private
model inference or supply a runnable simulation scene.
