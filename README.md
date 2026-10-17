# LingBot-VLA 2.0 / RoboTwin 实验工作区

当前使用工作区中的官方源码，接入共享模型、共享数据和 RoboTwin 仿真。启动时不再复制、导出源码到临时目录。修改这里的源码会直接影响下一次运行。

## 目录

| 路径 | 用途 |
|---|---|
| `lingbot-vla-v2/` | 官方模型、数据处理、训练及推理源码 |
| `RoboTwin/` | 完整仿真源码，包含 50 个任务、任务配置、描述、策略示例与 CuRobo 源码 |
| `evaluation/` | 官方模型加载、双卡放置、仿真启动与结果记录 |
| `configs/robotwin_tasks.json` | 50 个任务名称，不依赖旧训练数据索引 |
| `tools/` | 数据下载、素材准备、环境兼容与仿真检查工具 |
| `outputs/` | 当前有效验证和后续实验结果 |
| `reports/` | 目录清理与迁移检查记录 |

`evaluation/run.py` 管理运行，`policy.py` 调用官方模型推理，`rollout.py` 调用官方评测循环，`report.py` 校验并汇总结果，`verify_preprocessing.py` 检查预处理。

## 源码与素材

- RoboTwin：`13c3c47ff4312dd62484bcd51be034af55c062d1`，从已验证可运行的本机仓库迁入。仅将硬编码素材路径改为读取 `ROBOTWIN_ASSETS`；任务动力学与成功判定保持官方实现。
- LingBot-VLA：基于 `be969b8fd117fb70550c5d4bf4bc328211b5b1b6`。早期注意力及严格 FP32 实验补丁已移除，保留已有的训练代码注释。
- 默认素材：`/srv/common_dataset/RoboTwin2.0/assets`。通过 `ROBOTWIN_ASSETS` 可更换位置。
- `tools/prepare_robotwin_assets.py` 只生成小型机器人路径配置到 `RoboTwin/assets/embodiments/`。网格、URDF、纹理直接从共享素材读取，不复制素材或源码，也不依赖 NAS 符号链接。
- CuRobo 从当前 `RoboTwin/envs/curobo/src` 导入。已安装环境中的编译扩展仍是本机依赖。

## 模型和数据

| 内容 | 路径 |
|---|---|
| 官方 RoboTwin 后训练权重（step 50000） | `models/lingbot-vla-v2-6b-robotwin/checkpoints/global_step_50000/hf_ckpt` |
| LingBot-VLA 2.0 6B 预训练权重 | `/srv/common_model/huggingface/lingbot-vla-v2-6b` |
| Qwen processor / tokenizer | `/srv/common_model/huggingface/Qwen3-VL-4B-Instruct` |
| 原始 clean 数据（50 任务，各 50 条） | `/srv/common_dataset/RoboTwin2.0/dataset` |
| 官方 LeRobot v2.1 数据 | `/srv/common_dataset/RoboTwin2.0/lerobot_dataset/extracted_v21_3dc3b798/RoboTwin_lerobot_v21` |

共享模型、训练数据和素材未做清理。已删除工作区中早期基线的数据索引、输出头检查点、旧 BF16/FP32 评测与失败调试结果。按用户要求直接删除，不保留备份；清理清单见 `reports/workspace_cleanup.json`。

## 官方推理评测

已有推理环境：`/tmp/lingbo-baseline-venv`（沿用环境名称）；已有仿真环境：`/home/chenguanxu/miniconda3/envs/robotwin`。推理环境被清理后可执行 `bash setup_official_inference.sh` 重建；这不是官方完整训练环境的安装脚本。

从当前工作区运行，GPU 编号需按实际空闲情况调整：

```bash
bash run_official_eval.sh --tasks click_bell --episodes 1 \
  --model-gpus 3 5 --sim-gpu 6 --precision fp32 \
  --output outputs/click_bell_new
```

使用预训练模型运行全部 50 个任务，每任务一个 randomized 回合：

```bash
bash run_posttrained_eval.sh --model pretrained --setting demo_randomized --model-gpus 3 2 --sim-gpu 0
```

所有评测入口默认使用 `demo_randomized`；需要 clean 时显式传入 `--setting demo_clean`。历史 clean 结果保持原标注。默认使用官方光线追踪、ACT 部署配置中的 unseen 指令、50 步动作块、10 步采样。模型内部噪声及动作维度为 55，经官方后处理转换成 RoboTwin 的 14 维指令。FP32 指权重和官方推理模式；官方 FlashAttention 内部仍有 BF16 转换。双卡加载适配 24 GB 显卡，关闭 compile；本机软件版本并未完全复刻官方训练环境。

底层 `run_official_eval.sh` 默认加载的是预训练模型，不是官方 RoboTwin 后训练模型。其运行配置取自官方 RoboTwin 模板。正式微调应保存训练配置和符合比赛要求的 clean 归一化统计；当前示例官方统计仅用于实现对齐。

新微调模型须提供完整权重及其训练配置：

```bash
bash run_official_eval.sh --model-path /path/to/full/hf_ckpt \
  --training-config /path/to/lingbotvla_cli.yaml \
  --tasks click_bell --output outputs/finetuned_new
```

## 官方 RoboTwin 后训练权重

已下载 [robbyant/lingbot-vla-v2-6b-robotwin](https://huggingface.co/robbyant/lingbot-vla-v2-6b-robotwin)，固定版本 `0451855729ec904f970600e0aec8b84661423afe`，共 20 个文件、25.52 GB。所有文件均按 Hugging Face 提供的哈希完成校验，见 `models/lingbot-vla-v2-6b-robotwin/download_verification.json`。共享模型目录不可写，因此文件放在当前工作区的 `models/` 下。

重新下载或校验：

```bash
/tmp/lingbo-baseline-venv/bin/python tools/download_posttrained.py
```

全部 50 个 randomized 任务，每任务 1 回合（需要三张有足够空闲显存的 GPU）：

```bash
bash run_posttrained_eval.sh --model posttrained --setting demo_randomized --model-gpus 3 2 --sim-gpu 0
```

新入口自动创建带时间戳的 `outputs/posttrained_randomized50_video_*`，成功完成后只保留 `results.csv`、`videos/` 和 `index.html`。运行中的诊断文件暂存 `/tmp/lingbo-*-randomized50-*`，正常完成后删除；异常中断时保留并打印路径，不将环境异常当成模型失败。可用 `--output` 指定新的结果目录，GPU 编号请按空闲情况调整。

录像已修复为每个动作前刷新画面，不改变每个动作块 50 步的模型调用方式；此前录像缺失的中间帧无法恢复，需要重跑。录制开销会增加。

预训练与 RoboTwin 后训练模型各跑全部 50 个任务（共 100 回合，顺序复用同组 GPU）：

```bash
bash run_posttrained_eval.sh --model both --setting demo_randomized --model-gpus 3 2 --sim-gpu 0
```

两版均为 randomized / unseen、每任务 1 回合、场景种子参数 0、FP32 权重、无量化，并启用修复后的录像。预训练权重来自 `/srv/common_model/huggingface/lingbot-vla-v2-6b`，使用官方 RoboTwin 配置；后训练模型使用下载的 step 50000 检查点及附带配置。结果位于 `outputs/both_randomized50_video_时间戳/{pretrained,posttrained}/`，每个子目录只有 CSV、视频和 HTML，页面标明模型。某个模型出现环境异常会保留临时日志并继续另一模型，最终返回非零状态；不会将未完成任务当作模型失败。Ctrl+C 则停止整组运行。

使用检查点附带的训练配置和官方 RoboTwin 归一化统计，不使用早期输出头微调权重。`--continue-on-error` 只让环境异常后继续其他任务；异常会写入 `report.json` 的 `errors`，整组标为 `incomplete` 并返回非零退出码，不能当成模型失败。有效回合达到环境步数上限但未成功，才计作模型失败。

2026-10-09 实测已完成全部 **50 个 clean 任务，成功 44 个，成功率 88%**；每任务 1 回合、unseen 指令、官方 FP32 模式、无量化。这是 50 回合快速验证，不是官方完整基准复现。

结果直接打开 [HTML 展示页](outputs/posttrained_clean50_20261009/index.html)，可搜索任务、筛选失败项并播放视频。该结果目录只保留 `index.html`、`results.csv` 和 `videos/`；视频以任务名命名，CSV 中使用相对路径。其他结果文件已按要求删除，不保留备份。

HTML 可由 CSV 重新生成：

```bash
/tmp/lingbo-baseline-venv/bin/python -m evaluation.report outputs/posttrained_clean50_20261009
```

## 验证与训练状态

`outputs/official_alignment_verified_20261009/` 是迁移前已完成的官方对齐验证：click_bell 一回合，400 步、8 个有效动作块，任务失败。它证明链路跑通，不代表达到比赛成功率，也不属于迁移后的重新评测。

仿真环境单独检查（官方专家控制，不运行 VLA）：

```bash
ROBOTWIN_GPU=6 bash run_robotwin.sh python \
  "$PWD/tools/robotwin_smoke.py" --render-mode raytrace
```

`run_robotwin.sh` 不带参数时运行快速光栅渲染检查；正式模型评测入口保持官方光线追踪。

官方训练入口是 `lingbot-vla-v2/tasks/vla/train_lingbotvla.py`，配置在 `lingbot-vla-v2/configs/vla/robotwin/`。完整训练仍需配置依赖、教师权重、LeRobot 数据路径、批量大小和显存策略，尚未跑通新一版官方微调。早期根目录 `train.sh` 与 `eval.sh` 已移除，避免混淆。

迁移后验证：直接从当前源码运行 click_bell 官方专家，光线追踪模式，规划和任务均成功，视频 82 帧；官方图像、55 维状态与动作预处理检查通过。详见 `reports/workspace_validation.json`。这一条记录仅对应源码迁移时的检查；后训练模型评测见上节。
