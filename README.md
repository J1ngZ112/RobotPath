# RobotPath

RobotPath is an Isaac Sim extension and experiment codebase for mobile-robot
path planning, collision checking, replanning, and collision-data collection.
The repository contains the project-authored Python code, layout/configuration
examples, experiment scripts, and a small asset-free data-format example.

## Release scope

The first public release is **code-only**. It intentionally does not include
`data/Assets/` or the full `data/collision_dataset/`. The Warehouse scene was
assembled by the project authors, but its USD files may reference NVIDIA,
Fab, or other third-party meshes, robots, materials, and textures. A custom
scene composition does not by itself grant permission to redistribute those
underlying files. See [`docs/ASSET_MANIFEST.md`](docs/ASSET_MANIFEST.md) for
the authorization record.

The extension can still be enabled because its Python module and UI dependency
are present. Loading the Warehouse scene requires the external USD files, so a
missing asset directory produces a clear asset report and the scene will not
load completely. Prepare a compatible, legally obtained asset directory or
wait for a separately authorized asset release.

## Requirements

- Isaac Sim 4.x (use the version used for the associated paper experiments)
- Python shipped with Isaac Sim
- A licensed set of compatible Warehouse, shelf, robot, prop, material, and
  texture assets

## Configure external assets

The default asset directory is `data/Assets` inside the checkout. Since that
directory is excluded from this release, set an external directory before
starting Isaac Sim:

```bash
export ROBOTPATH_ASSET_ROOT=/path/to/licensed/robotpath-assets
# Optional when the scene file has a different name or location:
export ROBOTPATH_SCENE_PATH=/path/to/licensed/robotpath-assets/Warehouse.usd
```

Both the graphical extension and `headless_collect_dataset.py` use these
variables. The startup check reports the missing `Warehouse.usd`, shelf USD,
and robot USD paths. Referenced materials and textures must also be present in
the locations encoded by the USD files.

## Run

1. Open the repository as an Isaac Sim extension and enable `RobotPath` from
   **Window > Extensions**.
2. Set the asset variables above, open the RobotPath window, choose a layout,
   and press **Load Scene**.
3. For headless collection, run the Isaac Sim Python interpreter:

```bash
./python.sh headless_collect_dataset.py --layout <layout-name> --num-sequences 10
```

Waypoint execution supports two motion modes. `simple` (the default) rotates
in place until the target heading is reached, then translates. `hard` updates
translation and yaw in the same simulation step. Select the mode in the UI,
use `--motion-mode simple|hard` for headless collection, or call
`RobotController.set_motion_mode("hard")` / pass
`motion_mode=` to `start_waypoint_navigation`.

Use `--output-dir` to write generated results outside the repository. The
checked-in example in `data/examples/collision_sample.json` documents the
public data shape without containing third-party assets or a complete dataset.

## Repository contents

- `ExtCode/`: extension, planners, controllers, collision checking, and
  baseline implementations.
- `Experiment/`: layout JSON, waypoint examples, and experiment notes.
- `config/extension.toml`: Isaac Sim extension metadata.
- `headless_collect_dataset.py`: headless data-collection entry point.
- `docs/ASSET_MANIFEST.md`: asset provenance and redistribution checklist.

## Asset authorization and later releases

Before publishing an asset package, record the source page, exact version,
license, redistribution and modification permissions, and written permission
where required. Add the corresponding `NOTICE` and third-party license files,
verify USD references and texture/MDL completeness in a clean Isaac Sim
environment, and publish the package under a distinct version tag. The MIT
license in this repository covers project-authored code and documentation
only; it does not license excluded or third-party assets.

## Citation and support

Please cite the associated paper when using RobotPath in research. Issues and
pull requests are welcome for code and documentation; do not upload assets
unless their redistribution rights are documented.

---

# RobotPath（中文）

RobotPath 是一个基于 Isaac Sim 的移动机器人路径规划、碰撞检查、重规划
和碰撞数据采集扩展。仓库首版公开项目自有 Python 代码、布局与配置示例、
实验脚本，以及一个不含资产的最小数据格式示例。

## 首版发布范围

首版采用 **仅代码** 发布，明确不包含 `data/Assets/` 和完整的
`data/collision_dataset/`。Warehouse 场景虽然由项目作者自行拼装，但其中
的 USD 可能引用 NVIDIA、Fab 或其他第三方的网格、机器人、材质和纹理。
自行完成布局和组合并不会自动获得底层文件的再分发权。资产来源和授权状态
记录在 [`docs/ASSET_MANIFEST.md`](docs/ASSET_MANIFEST.md)。

扩展仍可能正常启用，因为 Python 模块和 UI 依赖存在；但加载 Warehouse
必须依赖外部 USD 文件。资产目录缺失时，启动检查会列出缺失路径，场景无法
完整加载。用户需要准备合法取得的兼容资产，或等待后续获得授权的资产版本。

## 环境与运行

需要 Isaac Sim 4.x、Isaac Sim 自带 Python，以及一套合法取得且兼容的
Warehouse、货架、机器人、道具、材质和纹理资产。资产不放在仓库中时设置：

```bash
export ROBOTPATH_ASSET_ROOT=/path/to/licensed/robotpath-assets
export ROBOTPATH_SCENE_PATH=/path/to/licensed/robotpath-assets/Warehouse.usd
```

两个入口会使用相同配置：图形扩展和 `headless_collect_dataset.py`。在 Isaac
Sim 中启用扩展后打开 RobotPath 窗口，选择布局并点击 **Load Scene**；无界面
采集示例：

```bash
./python.sh headless_collect_dataset.py --layout <layout-name> --num-sequences 10
```

导航执行支持两种模式：默认的 `simple` 先原地转向到目标方向再移动；`hard`
在同一个仿真步中同时更新转向和平移。图形界面可以直接选择，headless 采集使用
`--motion-mode simple|hard`，代码接口为 `set_motion_mode()` 或
`start_waypoint_navigation(..., motion_mode="hard")`。采集结果中的 motion 配置
会记录实际模式。

## 许可证

仓库中的 MIT 许可证仅覆盖项目自有代码和文档，不覆盖被排除的资产或任何
第三方内容。取得书面授权后，将通过独立资产包或新的版本标签追加资产，并在
`NOTICE` 中记录来源、版本、许可证和完整性验证结果。
