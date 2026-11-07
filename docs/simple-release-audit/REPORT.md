# Simple 首版数据发布审计

审计日期：2026-11-07。结论：具备组织离线诊断数据包的结构基础；候选包已确定为 7,030 条。作者于本次审计后确认，数据标注正确性已经验证，标注正确；不再将标注正确性验证列为发布前待办。发布仍需完成字段导出、许可确认与打包，论文结果复现不在本次审计范围内。

## 实际范围与快照

逐文件读取 `data/collision_dataset` 与 `LLM-RobotPath/collision_dataset`。前者顶层 24 个 Simple 文件，与后者顶层 21 个及 `Unseen` 3 个文件的 metadata 和 records 内容一致；只需发布一份，不能叠加计数。后者分组更清晰，建议作为打包来源。24 个文件共 307,929,521 字节（约 294 MiB），打包前的 SHA-256 见 `source_manifest.json`。该清单同时列出排除的 Hard 文件，不能把整张清单当作发布白名单。

| 子集 | 原始记录 | 碰撞 | 建议发布记录 |
|---|---:|---:|---:|
| 主集合（21 文件） | 6,300 | 4,092 | 6,130 |
| 布局迁移（20260821_120455） | 300 | 216 | 300 |
| 联合迁移（20260824_123844） | 300 | 230 | 300 |
| 机器人迁移（20260825_005353） | 300 | 207 | 300 |
| 合计 | 7,200 | 4,745 | 7,030 |

候选包共 4,745 条碰撞、2,285 条成功未碰撞。排除的 170 条均为 `training_eligible=false`、`collection_outcome=fault_injection_miss`，虽然 navigation_status 为 success，也不应悄悄放回论文使用的训练集合。逐条排除清单见 `excluded_records.json`。原始文件不修改。

主集合机器人为 rect_robo、cyli_robo、ridgeback，各 2,100 条原始记录。迁移集合含 ridgeback 300 条、square_robo 600 条；Simple 总计 4 个 robot ID，不能直接沿用整个 Simple+Hard 的“五种机器人”说法。按障碍物 ID、中心、尺寸和朝向去重，主集合有 6 种布局，Unseen 有 2 种，其中 1 种与主集合共享；它们是不同迁移任务，而非全部未见布局。

## 已完成检查

- 24 个文件的 motion.control_mode 均为 rotate_then_translate_accumulated_pose_servo；采样轨迹中未发现线速度和角速度命令同时超过 1e-6。此结论仅覆盖记录的采样命令，不证明每个物理步都无转向漂移。
- 7,200 条均有路径、标签、执行状态和轨迹；路径为有限二维坐标，长度字段一致；碰撞位置有效，对象 ID 非空，碰撞索引在范围内，collision_segment 符合 [index-1,index]；未发现结构告警。
- 未发现非碰撞 timeout/stuck、轨迹时间倒退、超出 ±6m 的路径点，或精确重复路径/完整几何输入。
- `contact_diagnostics`、`collection_outcome`、`training_eligible`、`collision_confirmation_reason` 仅有 4,500 条提供，另外 2,700 条缺失。应保留“旧版未记录”的语义，不能伪造诊断值。现有加载器将缺失 eligibility 当作 true，本候选名单保持这一历史行为。
- 检查 Simple JSON 字符串中的 Google key 格式、私钥标记、邮箱、HTTP URL 和常见 /Users、/home、/mnt、/tmp 路径，未命中。有限规则扫描不等同于全面匿名认证。
- 标注正确性：作者确认已完成验证且标注正确。本报告记录该结论；本次审计脚本未重新运行 Isaac Sim，也未独立复核该验证。作者未在本次沟通中提供验证方法、日志或指标，因此不补写具体方法或数值。
- 本次结构审计未覆盖近重复路径、几何容差、沿轨迹的实际控制行为或修复成功率；这些范围说明不否定作者已完成的标注验证，也不将其扩展为对修复成功率的验证。

## 固定划分

核验已有 `outputs/audit_20260919/split_manifest.json`：train 4,903、validation 610、test 617，共 6,130 条；无缺失 ID、无不合格记录、无集合交叉，恰好覆盖主集合候选记录。该清单可保留，不必重新随机划分。`candidate_splits.json` 已保存候选划分，并将额外 900 条分为三个独立迁移测试集，各 300 条。

此处只验证 ID/资格/覆盖；未重新验证旧报告的 dataset_fingerprint 算法。主集合采用样本级随机划分，不应作为未见布局泛化证据。训练与额外迁移集无精确路径重复，但机器人迁移集使用已见布局是预期设计。

旧报告 `dataset_evaluation/REPORT.md` 描述 11,999 条 Hard，不能当作本次 Simple 数据卡。论文概览图中的“900 items Layout Transfer”也需要改为三个任务或解释其合并口径。

## 确定的发布包内容

建议包名 `robotpath-simple-v1.0`，以离线碰撞诊断为首版范围，以下为交付结构；目前仅完成审计与候选名单，尚未生成正式数据包。

```
robotpath-simple-v1.0/
  README.md                  # 数据卡、统计、来源、限制和快速读取
  LICENSE-DATA               # 经确认的数据许可证，不自动套用代码 MIT
  SCHEMA.md                  # 单位、坐标系、索引、空值与有效性约定
  inputs/*.jsonl             # 7,030 条任务输入，使用稳定匿名 sample_id
  labels/*.jsonl             # 标签，独立于预测输入
  scenes/*.json              # 任务必需、经过许可范围确认的几何
  splits/*.json              # 原有 3 个划分 + 3 个迁移测试集
  examples/                  # 从发布集合选少量正负样本，标明所属划分
  tools/                     # 标准库加载器、格式校验和离线指标脚本
  provenance.json            # 采集版本、原始快照与匿名 ID 映射规则
  exclusions.json            # 170 条排除记录和原因，不包含原始轨迹
  SHA256SUMS                 # 最终导出文件的校验值
```

预测输入只提供路径、机器人任务必需几何、场景几何及模式。标签包括 collision_detected、collision_waypoint_index、collision_segment、collision_position、collision_object_id 和有效性掩码。成功负类的碰撞索引/对象/位置不参与指标。

`scenario_type`、`collision_type`、`target_obstacle_id`、`generation.expected_collision`、注入参数、navigation_status、contact_diagnostics 和 executed_trajectory 会暴露目标或执行结果，不得混入预测 prompt。执行轨迹可另做可选诊断包；首版核心包不需要收录。修复缓存中的合成解不作为物理验证的 ground truth。

保留 test_id 与 source_file 的复合来源键供内部追溯，公开使用稳定匿名 sample_id；转换后重写划分并检查一一对应。时间戳和内部路径仅在非公开溯源映射保留，除非任务确实需要。

## 不发布内容与剩余条件

1. 排除全部 Hard（13,199 条）、repair_cache、训练缓存 JSONL、日志、模型输出、USD/MDL/纹理和完整 LLM-RobotPath 源目录。当前 LLM-RobotPath 整目录为 Git 未跟踪状态，不能 git add 整个目录。
2. 数据含 24 处机器人 hull_3d、72 处 obstacle_polygon_m、42 处 collision_geometry，其中有明确的 usd_mesh_world_xy_convex_hull 来源。首版核心不需要 hull_3d 和逐部件 collider，建议省略；用于任务输入的二维足迹/多边形仍须核实适用条款，不能因导出成 JSON 就认定不涉及资产许可。
3. 若用自定义粗几何替换资产提取几何，应另标版本并重做评测；不能声称与论文原始输入完全等价。当前审计不作许可结论。
4. 确认数据许可证、完成允许字段导出、准备无 Isaac Sim/API key 的加载器和指标脚本、验证最终 checksum/划分后再发布。匿名镜像还需检查作者、路径、提交历史及下载链接；个人 GitHub 仓库本身不是匿名入口。
5. 在获得合法几何发布范围前，已可以据此清单准备导出脚本和离线评测，但不应宣布完整 Simple 数据包已公开。

复跑结构审计：在仓库根目录运行 `python3 tools/audit_simple_release.py`。输出 summary.json、source_manifest.json、record_issues.json 和 excluded_records.json。candidate_splits.json 是基于已核验的旧清单及本次迁移集手动命名的快照；源文件变化后需同步重新生成并审核。


## 发布实施更新（2026-11-07）

作者选择 CC BY 4.0 并确认二维几何已获再分发许可，许可依据稍后补充。
首版范围调整为数据、标签、固定划分、数据说明、加载示例和完整性校验工具；
不包括模型、训练代码或评分工具。正式导出位于 `datasets/simple-v1.0/`，
独立压缩包位于 `downloads/`。保留原始二维足迹和障碍物多边形，不导出三维凸包、
逐部件碰撞几何和原始资产。上文描述的是初始审计阶段，最新范围以数据包 README、
SCHEMA、NOTICE 和 provenance.json 为准。原始 ID 的本地映射仅存于忽略目录 outputs。
