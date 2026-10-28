# 配置填写

从 `config/models.example.json` 复制本地配置；字段按 defaults、backend、model 顺序覆盖，嵌套对象整体替换。只需保留要运行的 models 条目。`--task` 是会话标签，不用于寻找权重、推断指令或设置时长。

| 字段 | 来源与作用 |
|---|---|
| instruction | 自己的任务指令；同一任务可共享，也可按模型覆盖 |
| checkpoint / revision | 模型机权重目录及不可混淆的版本号或哈希 |
| python | 对应模型独立环境的 Python 绝对路径 |
| action_steps | 每次实际执行的预测块长度，不能超过加载器支持的输出长度 |
| control_hz | 自己的训练/控制频率，不能仅凭模型名称推断 |
| max_action_seconds | 首条动作开始后的时间上限；与 CLI 的总步数 `--steps` 共同生效 |
| bind | 两端可访问的模型机地址，例如 `tcp://MODEL_IP:9117` |
| nuc_ip | 已运行 DROID 控制服务的 NUC IP |
| camera_config | MiniPC 上相机 YAML 的绝对路径 |
| cameras | 模型输入的相机角色到数字字符串 ID 的映射 |
| recording_cameras | 左外部、腕部、右外部三路记录映射；输入角色必须与此一致 |

动作接口为 7 个绝对关节位置加 1 个夹爪值，共 8 维。不是任意机器人、任意动作表示的通用运行器。归一化/相对动作解码由各加载器和原生处理器完成，不能手动重复反归一化。

两端放置相同 JSON：profile 哈希包含路径、相机、提示词等完整字段。配置中的权重路径只需在模型机存在，相机路径只需在 MiniPC 存在。`revision` 不能替代文件内容检查。

## 相机

复制 `config/cameras.example.yaml`，使用稳定的 `/dev/v4l/by-path/` 路径，逐路核对物理位置、ID 和标定文件后将 `identity_confirmed` 设为 true。标定路径相对 YAML 所在目录，`calibration_sha256` 是该 NPZ 原始字节的 SHA256。

保留的 UVC 读取器要求 2560×720 YUYV 双目帧，取左半幅并校正，不适用于所有 USB 相机。每个 NPZ 包含 `serial`、`pixel_order='BGR'`、float32 的 `map_x`/`map_y`（720×1280）以及 3×3 的 `K`。不要使用其他相机的标定填空。

## 权重格式变体

`config/variants/*.json` 是单个模型条目的模板，不能直接作为完整 catalog。把条目复制到 `models` 中，并填写该 backend 对应配置。PI05 的 `pi05` 与 `openpi`、MolmoAct2 的两种格式、GR00T 的两种格式不能只改目录就互换。详见 [models.md](models.md)。

```bash
python tools/check_config.py --catalog config/models.local.json --models pi05
python evaluate.py --catalog config/models.local.json --task '<YOUR_TASK_ID>' --models pi05 --plan
```

以上只检查配置。检查成功不代表设备、GPU 和权重已经通过验收。
