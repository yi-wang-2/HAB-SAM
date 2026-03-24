# 太湖蓝藻分割 SAM3 微调项目说明 (LAKE_SH_FINETUNE_README.md)

本项目提供了一套完整的流程，用于在 **太湖蓝藻 (Lake Blue-Green Algal)** 数据集上微调 **SAM3 (Segment Anything Model 3)** 模型。使得该模型能够针对蓝藻爆发进行特定的目标检测和实例分割任务。

## 🌟 核心特性

*   **自定义数据集支持**：针对 `lake_sh` 数据集进行了完整配置。
*   **显存溢出 (OOM) 保护机制**：在 `sam3/eval/postprocessors.py` 中实现了极其稳健的故障转移机制。如果在验证阶段（掩码插值时）耗尽了 GPU 显存，程序会自动切换到 CPU 处理，并采用 **分块（Chunking）** 策略处理数据，防止程序崩溃。
*   **自动化训练脚本**：通过 Shell 脚本一键启动微调。
*   **集成评估系统**：在训练过程中自动计算目标检测和分割的 COCO 指标（AP, AR）。

## 📂 目录结构

请确保您的工作区结构如下所示：

```text
sam3-main/
├── data/
│   └── lake_sh/              # 数据集根目录
│       ├── train/            # 训练集图片
│       ├── train.json        # 训练集标注 (COCO 格式)
│       ├── val/              # 验证集图片
│       └── val.json          # 验证集标注 (COCO 格式)
├── output/
│   └── lake_sh_fine_tune/    # 训练日志和模型权重保存位置
├── scripts/
│   └── fine_tune_lake_sh.sh  # 启动脚本
└── sam3/
    └── train/
        └── configs/
            └── lake_sh_fine_tune.yaml  # 主配置文件
```

## 🚀 快速开始

1.  **激活环境**
    确保您处于正确的 conda 环境中（例如 `sam3`）：
    ```bash
    conda activate sam3
    ```

2.  **运行训练**
    在项目根目录下执行提供的 Shell 脚本：
    ```bash
    bash scripts/fine_tune_lake_sh.sh
    ```

## ⚙️ 配置说明

主配置文件位于：
`sam3/train/configs/lake_sh_fine_tune.yaml`

常见的可修改参数：
*   `paths.lake_sh_root`: 数据集的路径。
*   `scratch.resolution`: 输入图像分辨率（默认：1024）。
*   `scratch.base_lr`: 基础学习率。
*   `scratch.max_epochs`: 总训练轮数（Epochs）。

## 🛠️ 故障排除与常见问题 (FAQ)

**Q: 验证过程中我在日志里看到很多 `INFO ... Issue found, reverting to CPU mode!`，这是报错吗？**
**A: 这不是报错，而是正常的保护行为。**
这是 **OOM 保护机制** 正在生效。验证过程需要调整高分辨率掩码的大小，极易导致显存溢出。系统检测到显存不足时，会捕获错误，并将掩码分块转移到 CPU 上安全地进行处理。这确保了您的训练任务能够顺利完成，而不是直接崩溃。

**Q: 输出结果在哪里？**
**A:** 请检查 `output/lake_sh_fine_tune/` 目录。您会找到：
*   `checkpoints/`: 保存的模型权重 (`checkpoint.pt`)。
*   `dumps/`: 预测结果的 JSON 文件。
*   训练日志文件。

## 📊 性能监控

脚本会在每个 Epoch 结束时自动输出 COCO 评估指标：
*   `Average Precision (AP) @[ IoU=0.50:0.95 ]`: 综合性能指标。
*   `Average Precision (AP) @[ IoU=0.50 ]`: 标准 PASCAL 指标。

请关注日志中的 `val_lake_sh/detection` 和 `val_lake_sh/segmentation` 指标来跟踪模型进展。
