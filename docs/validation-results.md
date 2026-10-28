# 发布验证结果

2026-10-28，Windows / Python 3.10，本地隔离测试环境：

```text
python -m pytest -q --tb=short
158 passed, 91 subtests passed in 34.03s
```

另外完成全部 Python 源码 AST 解析、Markdown 本地链接检查、五模型任意任务 `--plan`、模型 CLI `--help`、未填写占位配置拒绝执行，以及发布文件的常见凭据和现场地址扫描。

旧导出测试按现有最近邻采样/复用标记语义更新，保留源数据缺失、时钟异常、错误对齐声明、输出篡改等拒绝路径。历史现场 checkpoint 依赖改用隔离输入；离线测试不证明真实权重加载成功。

本轮没有连接机器人、相机或加载 GPU 模型。尚未验证全新 Linux/GPU 环境安装和整理版本的整机运行。模型兼容范围和缺失的上游来源信息见 models.md 与 THIRD_PARTY_NOTICES.md。
