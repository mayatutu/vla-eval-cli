# 模型环境与权重

五个模型系列共用评测入口，但模型环境按权重格式隔离。仓库不分发权重、不保存下载 token，也不自动下载模型。权重应从你自己的训练导出或授权的模型发布地址取得；保留整个导出目录及其附属资产。

## 安装原则

每个 backend 使用独立 Python 环境，先按对应上游项目安装，再在该环境安装本仓库 `requirements/model-transport.txt`。把环境 Python 的绝对路径填入 `python`。不要在同一环境强行统一 Torch、Transformers、JAX 版本。

`requirements/observed-versions.json` 是原运行环境观察值，不是完整锁文件，也不是已验证的全新安装组合。下列源码版本用于定位兼容基线；未提供精确源码版本的组件须按自己的权重发布说明准备并验收。

| 系列 / backend | 权重格式、必要资产 | 环境与配置 |
|---|---|---|
| PI05 / pi05 | LeRobot `model.safetensors`、`config.json`、预/后处理 JSON 及其中引用的 state 文件 | LeRobot `7e241bd630a3719a56157a497ce5d08f244784f1`；填写本地 tokenizer |
| PI05 / openpi | JAX `params/`、`train_config.py`、`assets/<dataset_id>/norm_stats.json` | LAP `3958d1466d5b92445b67de7d4202c19608ad4d56` 的 `third_party/openpi`；填写 source root、tokenizer、dataset_id 和资产哈希 |
| G05 / g05 | `.hydra/config.yaml`、`dataset_stats.json`、`action_tokenizer.pt`、权重 PT | GalaxeaVLA `89f2322b4ad016e192437adc1a2c253b05bab246`，Python 3.10；填写 repo、overlay、processor、embodiment 和夹爪约定 |
| LingBot / lingbot | 全部分片、索引、config、norm_stats、tokenizer、`lingbotvla_cli.yaml`、机器人 YAML、preprocessor | 与权重匹配的 LingBot VLA v2 源码及环境；填写 lingbot_repo、lingbot_vlm_config、robot_config |
| MolmoAct2 / molmoact2_lerobot | LeRobot 权重、配置、预/后处理及状态；本地 base_assets | 与 PI05 表中版本对应的 LeRobot 包须包含 MolmoAct2 实现；base_assets 需与保存的处理器匹配 |
| MolmoAct2 / molmoact2 | 原生 HF 全部分片、norm_stats、tokenizer、processor 和模型 Python 文件 | 按该权重模型卡安装；保留原生连续动作、norm_tag 和 num_steps 配置 |
| GR00T / groot_lerobot | LeRobot 权重、配置、处理器及状态 | 对应 LeRobot、Cosmos tokenizer、`config/groot_base_architecture.json`；仅适用于该架构 |
| GR00T / groot_isaac | 全部分片、config、processor 目录中的统计/embodiment/处理器配置 | Isaac-GR00T `51d4c89f72fda44cbf77285c6a8114b52676b8a1`；填写 source root、Cosmos 和资产哈希 |

上游来源：[LeRobot](https://github.com/huggingface/lerobot)、[GalaxeaVLA](https://github.com/OpenGalaxea/GalaxeaVLA)、[LAP](https://github.com/lihzha/lap)、[Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)。LingBot 的精确 V2 来源和 revision 未从原部署中恢复，不能把任意同名仓库当成已核对版本。

安装命令框架（每个模型环境单独执行）：

```bash
python3 -m venv /absolute/path/to/model-env
/absolute/path/to/model-env/bin/python -m pip install -r /absolute/path/to/vla-eval-cli/requirements/model-transport.txt
# 按所选上游版本的安装说明安装模型包、CUDA/Torch 和其余依赖。
```

本版提供可迁移评测代码和配置入口，不包含一键重建所有 GPU 环境的安装器。新机器上需先完成独立模型加载/保存观测推理验证，再进行真机测试。

## G05 overlay

保留的 inferencer 改动通过独立目录生效，不覆盖上游仓库：

```bash
python tools/prepare_g05_overlay.py --repo /absolute/path/to/GalaxeaVLA --output /absolute/path/to/new-g05-overlay
```

输出目录必须不存在，填入 `g05_overlay`。默认权重为 `checkpoints/model_state_dict.pt`；其他导出布局可填相对路径 `g05_weights_file`。`g05_embodiment`、`g05_gripper_mode`、控制频率和对应 evidence 由训练约定决定，不能根据任务名称猜测。

## OpenPI / Isaac 资产指纹

```bash
python tools/checkpoint_hashes.py --backend openpi --checkpoint /absolute/path/to/checkpoint --dataset-id '<DATASET_ASSET_ID>'
python tools/checkpoint_hashes.py --backend groot_isaac --checkpoint /absolute/path/to/checkpoint
```

将输出的 `asset_sha256` 对象放进相应模型条目。它绑定的是配置/索引小文件，不能证明大权重的完整内容。`native_source_sha256` 绑定已核对的上游实现；如果不匹配，先确认版本及处理语义，不要为了绕过检查盲目换成当前文件的哈希。

同一系列在格式、架构、动作表示和处理器一致时可共用 backend；不同任务通常只改 checkpoint、revision、instruction 及对应统计资产。权重和归一化统计本身不通用，不能混用。

```bash
python model.py inspect --catalog config/models.local.json --model pi05
```

此命令只读小文件、检查布局，不启动 GPU、相机或机器人。运行入口详见 [README](../README.md)。
