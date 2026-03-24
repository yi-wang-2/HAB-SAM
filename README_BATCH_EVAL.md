# SAM3 批量推理和评估脚本

## 功能特点

✅ **批量处理**: 自动处理目录下所有图像  
✅ **多种评估指标**: mIoU, Dice系数, 像素精度, Precision/Recall, F1分数  
✅ **结果可视化**: 自动生成对比图（原图、真实标注、预测结果）  
✅ **详细报告**: JSON格式保存整体和单张图像的评估结果  

## 评估指标说明

| 指标 | 说明 | 取值范围 |
|------|------|----------|
| **mIoU** | 平均交并比，衡量预测区域与真实区域的重叠程度 | 0-1，越高越好 |
| **IoU per class** | 每个类别的交并比 | 0-1，越高越好 |
| **Pixel Accuracy** | 像素级准确率 | 0-1，越高越好 |
| **Dice Score** | Dice系数，F1分数的等价形式 | 0-1，越高越好 |
| **Precision** | 精确率，预测为正例中真正为正例的比例 | 0-1，越高越好 |
| **Recall** | 召回率，真实正例中被正确预测的比例 | 0-1，越高越好 |
| **F1 Score** | 精确率和召回率的调和平均 | 0-1，越高越好 |

## 文件说明

```
batch_inference_eval.py      # 核心脚本（可作为库导入使用）
run_batch_eval_example.py    # 使用示例
README_BATCH_EVAL.md         # 本文档
```

## 使用方法

### 1. 准备数据

创建以下目录结构：

```
sam3-main/
├── assets/
│   ├── images/           # 原始图像
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── masks/            # 标注掩码（二值图像）
│       ├── img001_mask.png
│       ├── img002_mask.png
│       └── ...
├── sam3.pt               # 模型权重
└── batch_inference_eval.py
```

**标注掩码要求**:
- 二值图像（PNG 或 JPG 格式）
- 前景区域为白色（像素值 > 127）
- 背景区域为黑色（像素值 ≤ 127）
- 文件名与原图对应（支持后缀 `_mask`）

### 2. 方式一：使用示例脚本

修改 [`run_batch_eval_example.py`](run_batch_eval_example.py) 中的配置：

```python
CONFIG = {
    'image_dir': 'assets/images',      # 图像目录
    'mask_dir': 'assets/masks',        # 标注目录
    'output_dir': 'output/batch_eval', # 输出目录
    'text_prompt': 'person',           # 文本提示词
    'checkpoint_path': 'sam3.pt',      # 模型路径
    'mask_suffix': '_mask',            # 标注文件后缀
}
```

运行脚本：

```bash
python run_batch_eval_example.py
```

### 3. 方式二：使用命令行参数

```bash
python batch_inference_eval.py \
    --image_dir assets/images \
    --mask_dir assets/masks \
    --output_dir output/results \
    --prompt "person" \
    --checkpoint sam3.pt \
    --mask_suffix "_mask"
```

**参数说明**:
- `--image_dir`: 原始图像目录
- `--mask_dir`: 标注掩码目录
- `--output_dir`: 输出目录
- `--prompt`: 文本提示词（如 person, car, dog, cat 等）
- `--checkpoint`: 模型权重路径（默认 sam3.pt）
- `--mask_suffix`: 标注文件后缀（默认 _mask）

### 4. 方式三：在代码中调用

```python
from batch_inference_eval import batch_inference_and_eval

overall_metrics, results = batch_inference_and_eval(
    image_dir='assets/images',
    mask_dir='assets/masks',
    output_dir='output/results',
    text_prompt='person',
    checkpoint_path='sam3.pt',
    mask_suffix='_mask'
)

print(f"mIoU: {overall_metrics['mIoU']:.4f}")
print(f"Pixel Accuracy: {overall_metrics['pixel_accuracy']:.4f}")
```

## 输出结果

运行完成后，会在输出目录生成以下文件：

```
output/batch_eval/
├── evaluation_results.json    # 评估结果（JSON格式）
├── visualizations/            # 可视化对比图
│   ├── img001_comparison.png
│   ├── img002_comparison.png
│   └── ...
└── predictions/               # 预测掩码
    ├── img001_pred.png
    ├── img002_pred.png
    └── ...
```

### evaluation_results.json 结构

```json
{
  "config": {
    "image_dir": "assets/images",
    "mask_dir": "assets/masks",
    "text_prompt": "person",
    "num_images": 10,
    "timestamp": "2026-01-21 10:30:00"
  },
  "overall_metrics": {
    "mIoU": 0.8524,
    "IoU_per_class": [0.9234, 0.7814],
    "pixel_accuracy": 0.9456,
    "mean_dice": 0.8876,
    "dice_per_class": [0.9601, 0.8151],
    "precision": 0.8923,
    "recall": 0.8234,
    "f1_score": 0.8565
  },
  "per_image_results": [
    {
      "image_name": "img001.jpg",
      "mask_name": "img001_mask.png",
      "metrics": {
        "mIoU": 0.8734,
        ...
      }
    },
    ...
  ]
}
```

## 文件命名规则

脚本支持多种命名方式查找对应的标注文件：

如果原图为 `image.jpg`，脚本会按以下顺序查找：
1. `image_mask.png`（带后缀 + PNG）
2. `image_mask.jpg`（带后缀 + JPG）
3. `image.png`（同名 + PNG）
4. `image.jpg`（同名 + JPG）

## 常见问题

### Q1: 如何准备标注掩码？

**方法1**: 使用标注工具（如 LabelMe, CVAT）导出二值掩码  
**方法2**: 使用 Python 转换：

```python
from PIL import Image
import numpy as np

# 假设你有多边形标注
mask = np.zeros((height, width), dtype=np.uint8)
# ... 填充多边形区域为255 ...
mask_img = Image.fromarray(mask)
mask_img.save('img001_mask.png')
```

### Q2: mIoU 值很低怎么办？

1. **检查文本提示词**: 确保 `text_prompt` 与数据集类别匹配
2. **检查标注质量**: 确认标注掩码正确（白色为前景）
3. **尝试不同提示词**: 如 "person" vs "human" vs "people"

### Q3: 处理多类别分割怎么办？

当前脚本默认处理二分类（背景/前景）。对于多类别：

1. 为每个类别分别运行脚本
2. 使用不同的 `text_prompt` 和 `output_dir`
3. 汇总各类别的结果

### Q4: 如何调整预测阈值？

修改 [`batch_inference_eval.py`](batch_inference_eval.py) 中的阈值：

```python
# 在 load_ground_truth_mask 函数中
mask = (mask > 127).astype(np.uint8)  # 改为其他阈值

# 或在处理预测掩码时
pred_mask = (np.array(pred_mask_pil) > 127).astype(np.uint8)  # 调整阈值
```

## 扩展功能

### 添加更多评估指标

在 `SegmentationMetrics` 类中添加新方法：

```python
def compute_boundary_iou(self):
    """计算边界 IoU"""
    # 实现边界检测和IoU计算
    pass
```

### 支持多类别分割

修改 `num_classes` 参数：

```python
metrics_calculator = SegmentationMetrics(num_classes=10)  # 10个类别
```

### 批量处理优化

对于大规模数据集，可以添加多GPU支持或批量推理。

## 依赖环境

```bash
torch >= 2.0.0
numpy >= 1.26.0
pillow >= 10.0.0
matplotlib >= 3.0.0
tqdm
```

## 参考

- SAM3 模型: [facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- 语义分割评估指标: [mmsegmentation metrics](https://github.com/open-mmlab/mmsegmentation)

## License

与 SAM3 项目保持一致。
