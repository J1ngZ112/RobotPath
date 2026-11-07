# RobotPath-Bench Simple v1.0

A downloadable, model-independent release of **7,030 labeled simulated robot
paths**: 4,745 collision records and 2,285 successful noncollision records.
Simple means **rotate first, then translate**. Annotation correctness has been
verified by the dataset author. This release provides data, labels, fixed
splits and standard-library loading/validation tools. It does not provide
model weights, model training code, inference reproduction or scoring scripts.

## Quick start

Requires Python 3.10 or newer. No extra packages, Isaac Sim, assets, GPU or API
key are required to read this dataset. From the repository root:

```bash
python3 datasets/simple-v1.0/tools/validate_dataset.py
python3 datasets/simple-v1.0/tools/load_dataset.py --split test --limit 2
```

After extracting a standalone archive, run `python3 tools/validate_dataset.py`
and `python3 tools/load_dataset.py --split test` from its directory. Import
`iter_samples` from `tools/load_dataset.py` for programmatic use; each yielded
item has separate `input`, `scene` and `label` dictionaries.

## Fixed splits

| Split | Records | Collision | Noncollision |
|---|---:|---:|---:|
| train | 4,903 | 3,273 | 1,630 |
| validation | 610 | 408 | 202 |
| test | 617 | 411 | 206 |
| layout_transfer | 300 | 216 | 84 |
| robot_transfer | 300 | 207 | 93 |
| joint_transfer | 300 | 230 | 70 |

The original main split is preserved, not resampled. The main split is a
sample-level random split and is not evidence of unseen-layout generalization.
The three transfer sets are separate tasks, not a single layout-transfer set.
There are four robot IDs across this release. Hard data is not included.
The paper's overview figure describes a broader dataset than this release.

## Contents

- `inputs/records.jsonl`: paths, robot footprints, mode and scene references.
- `labels/records.jsonl`: collision annotations; never feed these as inputs.
- `scenes.json`: deduplicated numerical scene descriptions.
- `splits.json`: six disjoint lists of stable sample IDs.
- `examples.json`: one positive and one negative drawn from train (not extra records).
- `exclusions.json`: 170 excluded IDs with `fault_injection_miss` reason.
- `provenance.json`: counts, source checksums and export transformations.
- `SCHEMA.md`, `LICENSE-DATA`, `NOTICE.md`, `SHA256SUMS` and `tools/`.

The source contained 7,200 Simple records. We preserve the existing exclusion
of 170 fault-injection misses; those are not claimed to be annotation errors.
Source filenames, timestamps, generation hints, execution trajectories and
runtime diagnostic fields are not exported. The private model directory and
training code are not included. No synthetic repair targets are presented as
physical ground truth.

## Scope and licensing

Data and dataset documentation: **CC BY 4.0** (see LICENSE-DATA).
Python tools: **MIT** (see LICENSE-CODE).
Geometry redistribution permission was confirmed by the author; supporting
evidence is pending addition (NOTICE.md). USD and other original assets remain
outside this release. Raw numeric footprint and polygon values are preserved;
this is not a recreated or simplified replacement scene. However, no claim is
made that these loader outputs duplicate the private model's prompt processing
or reproduce the paper's results. Wall labels retain simulator object IDs;
complete wall meshes are not available. This is an offline dataset, not a
standalone simulator scene. The public GitHub repository is not anonymous.

## 中文说明

首版公开 7,030 条 Simple 数据，包含 4,745 条碰撞、2,285 条成功未碰撞记录。
Simple 表示先转向后移动。标注正确性已由作者验证；工具仅检查数据完整性。
保留既有训练/验证/测试划分，并分别提供布局迁移、机器人迁移和联合迁移测试。
排除 170 条故障注入未命中记录以保持原实验口径，不将其称为错误标注。

无需 Isaac Sim 或 API key，使用上面的 Python 命令即可加载、校验。
模型权重、训练代码、Hard 数据、原始资产和修复标签不在本次发布范围内。
数据采用 CC BY 4.0，工具采用 MIT；二维几何许可已由作者确认，依据待补充。
字段、单位、索引及空值约定见 SCHEMA.md。数据公开不代表私有模型结果可复现。
