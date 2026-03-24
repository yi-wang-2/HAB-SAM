"""
批量图像推理和评估脚本
支持文本提示的语义分割推理，并计算常用评估指标（包括 mIoU）
"""

import torch
import numpy as np
from PIL import Image
import os
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class SegmentationMetrics:
    """语义分割评估指标计算类"""
    
    def __init__(self, num_classes=2):
        """
        Args:
            num_classes: 类别数量（默认2：背景和前景）
        """
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """重置所有累计统计量"""
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        
    def update(self, pred_mask, gt_mask):
        """
        更新混淆矩阵
        Args:
            pred_mask: 预测掩码 (H, W)，值为0或1
            gt_mask: 真实标注掩码 (H, W)，值为0或1
        """
        pred_mask = pred_mask.flatten()
        gt_mask = gt_mask.flatten()
        
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                self.confusion_matrix[i, j] += np.sum((gt_mask == i) & (pred_mask == j))
    
    def compute_iou(self):
        """计算每个类别的 IoU"""
        iou_per_class = []
        for i in range(self.num_classes):
            intersection = self.confusion_matrix[i, i]
            union = self.confusion_matrix[i, :].sum() + self.confusion_matrix[:, i].sum() - intersection
            if union > 0:
                iou = intersection / union
            else:
                iou = float('nan')
            iou_per_class.append(iou)
        return np.array(iou_per_class)
    
    def compute_miou(self):
        """计算平均 IoU (mIoU)"""
        iou_per_class = self.compute_iou()
        # 忽略 NaN 值计算平均
        valid_ious = iou_per_class[~np.isnan(iou_per_class)]
        if len(valid_ious) > 0:
            return valid_ious.mean()
        return 0.0
    
    def compute_pixel_accuracy(self):
        """计算像素精度"""
        correct = np.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        if total > 0:
            return correct / total
        return 0.0
    
    def compute_dice_score(self):
        """计算 Dice 系数（F1 Score）"""
        dice_per_class = []
        for i in range(self.num_classes):
            intersection = self.confusion_matrix[i, i]
            pred_sum = self.confusion_matrix[:, i].sum()
            gt_sum = self.confusion_matrix[i, :].sum()
            if (pred_sum + gt_sum) > 0:
                dice = 2 * intersection / (pred_sum + gt_sum)
            else:
                dice = float('nan')
            dice_per_class.append(dice)
        return np.array(dice_per_class)
    
    def compute_mean_dice(self):
        """计算平均 Dice 系数"""
        dice_per_class = self.compute_dice_score()
        valid_dice = dice_per_class[~np.isnan(dice_per_class)]
        if len(valid_dice) > 0:
            return valid_dice.mean()
        return 0.0
    
    def compute_precision_recall(self):
        """计算精确率和召回率（针对前景类）"""
        if self.num_classes < 2:
            return 0.0, 0.0
        
        # 前景类（类别1）
        tp = self.confusion_matrix[1, 1]
        fp = self.confusion_matrix[0, 1]
        fn = self.confusion_matrix[1, 0]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return precision, recall
    
    def get_all_metrics(self):
        """获取所有评估指标"""
        iou_per_class = self.compute_iou()
        dice_per_class = self.compute_dice_score()
        precision, recall = self.compute_precision_recall()
        
        metrics = {
            'mIoU': float(self.compute_miou()),
            'IoU_per_class': [float(x) if not np.isnan(x) else None for x in iou_per_class],
            'pixel_accuracy': float(self.compute_pixel_accuracy()),
            'mean_dice': float(self.compute_mean_dice()),
            'dice_per_class': [float(x) if not np.isnan(x) else None for x in dice_per_class],
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        }
        
        return metrics


def load_ground_truth_mask(mask_path):
    """
    加载真实标注掩码
    支持多种格式：二值图像、灰度图像等
    
    Args:
        mask_path: 标注文件路径
    Returns:
        numpy array (H, W)，值为0（背景）或1（前景）
    """
    if not os.path.exists(mask_path):
        return None
    
    mask = Image.open(mask_path).convert('L')  # 转为灰度图
    mask = np.array(mask)
    
    # 二值化：非零值视为前景
    mask = (mask > 127).astype(np.uint8)
    
    return mask


def save_visualization(image, pred_mask, gt_mask, save_path, metrics_text=""):
    """
    保存可视化结果：原图、预测、真实标注的对比
    
    Args:
        image: PIL Image 原图
        pred_mask: 预测掩码
        gt_mask: 真实标注掩码
        save_path: 保存路径
        metrics_text: 指标文本
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原图
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 真实标注
    axes[1].imshow(image)
    if gt_mask is not None:
        axes[1].imshow(gt_mask, alpha=0.5, cmap='jet')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # 预测结果
    axes[2].imshow(image)
    axes[2].imshow(pred_mask, alpha=0.5, cmap='jet')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # 添加指标文本
    if metrics_text:
        plt.figtext(0.5, 0.02, metrics_text, ha='center', fontsize=10, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def batch_inference_and_eval(
    image_dir,
    mask_dir,
    output_dir,
    text_prompt,
    checkpoint_path="sam3.pt",
    image_extensions=['.jpg', '.jpeg', '.png', '.bmp'],
    mask_suffix='_mask'
):
    """
    批量推理和评估
    
    Args:
        image_dir: 图像目录
        mask_dir: 标注掩码目录
        output_dir: 输出目录
        text_prompt: 文本提示词（如 "person", "car" 等）
        checkpoint_path: 模型权重路径
        image_extensions: 支持的图像格式
        mask_suffix: 标注文件后缀（如 image.jpg 对应 image_mask.png）
    """
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    
    print(f"{'='*60}")
    print(f"批量推理和评估")
    print(f"{'='*60}")
    print(f"图像目录: {image_dir}")
    print(f"标注目录: {mask_dir}")
    print(f"输出目录: {output_dir}")
    print(f"文本提示: {text_prompt}")
    print(f"{'='*60}\n")
    
    # 加载模型
    print("加载 SAM3 模型...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    model = build_sam3_image_model(checkpoint_path=checkpoint_path)
    model = model.to(device)
    processor = Sam3Processor(model)
    print("模型加载完成\n")
    
    # 获取所有图像文件
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_dir.glob(f"*{ext}")))
    
    image_files = sorted(image_files)
    print(f"找到 {len(image_files)} 张图像\n")
    
    if len(image_files) == 0:
        print("错误：未找到图像文件！")
        return
    
    # 初始化评估指标
    metrics_calculator = SegmentationMetrics(num_classes=2)
    results = []
    
    # 批量处理
    print("开始推理和评估...\n")
    for img_path in tqdm(image_files, desc="处理进度"):
        try:
            # 加载图像
            image = Image.open(img_path).convert('RGB')
            
            # 查找对应的标注文件
            # 尝试多种命名方式
            stem = img_path.stem
            mask_candidates = [
                mask_dir / f"{stem}{mask_suffix}.png",
                mask_dir / f"{stem}{mask_suffix}.jpg",
                mask_dir / f"{stem}.png",
                mask_dir / f"{stem}.jpg",
            ]
            
            gt_mask = None
            mask_path = None
            for candidate in mask_candidates:
                if candidate.exists():
                    mask_path = candidate
                    gt_mask = load_ground_truth_mask(candidate)
                    break
            
            if gt_mask is None:
                print(f"警告：未找到 {img_path.name} 的标注文件，跳过")
                continue
            
            # 推理
            inference_state = processor.set_image(image)
            inference_state = processor.set_text_prompt(state=inference_state, prompt=text_prompt)
            
            # 获取预测掩码
            if 'masks' in inference_state and len(inference_state['masks']) > 0:
                # 取第一个掩码（通常是最高置信度的）
                pred_mask = inference_state['masks'][0].cpu().numpy()
                # 确保与 gt_mask 尺寸一致
                if pred_mask.shape != gt_mask.shape:
                    pred_mask_pil = Image.fromarray((pred_mask * 255).astype(np.uint8))
                    pred_mask_pil = pred_mask_pil.resize((gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)
                    pred_mask = (np.array(pred_mask_pil) > 127).astype(np.uint8)
            else:
                # 如果没有检测到任何对象，创建全零掩码
                pred_mask = np.zeros_like(gt_mask)
            
            # 更新评估指标
            metrics_calculator.update(pred_mask, gt_mask)
            
            # 计算单张图像的指标
            single_metrics = SegmentationMetrics(num_classes=2)
            single_metrics.update(pred_mask, gt_mask)
            img_metrics = single_metrics.get_all_metrics()
            
            # 保存预测掩码
            pred_mask_save = Image.fromarray((pred_mask * 255).astype(np.uint8))
            pred_mask_save.save(pred_dir / f"{stem}_pred.png")
            
            # 保存可视化结果
            metrics_text = f"IoU: {img_metrics['IoU_per_class'][1]:.4f} | " \
                          f"Dice: {img_metrics['dice_per_class'][1]:.4f} | " \
                          f"Pixel Acc: {img_metrics['pixel_accuracy']:.4f}"
            
            save_visualization(
                image, pred_mask, gt_mask,
                vis_dir / f"{stem}_comparison.png",
                metrics_text
            )
            
            # 记录结果
            result = {
                'image_name': img_path.name,
                'mask_name': mask_path.name if mask_path else None,
                'metrics': img_metrics
            }
            results.append(result)
            
        except Exception as e:
            print(f"处理 {img_path.name} 时出错: {str(e)}")
            continue
    
    # 计算总体指标
    print(f"\n{'='*60}")
    print("评估完成！")
    print(f"{'='*60}\n")
    
    overall_metrics = metrics_calculator.get_all_metrics()
    
    print("总体评估指标:")
    print(f"  mIoU (平均IoU):        {overall_metrics['mIoU']:.4f}")
    print(f"  背景 IoU:              {overall_metrics['IoU_per_class'][0]:.4f}")
    print(f"  前景 IoU:              {overall_metrics['IoU_per_class'][1]:.4f}")
    print(f"  像素精度:              {overall_metrics['pixel_accuracy']:.4f}")
    print(f"  平均 Dice 系数:        {overall_metrics['mean_dice']:.4f}")
    print(f"  背景 Dice:             {overall_metrics['dice_per_class'][0]:.4f}")
    print(f"  前景 Dice:             {overall_metrics['dice_per_class'][1]:.4f}")
    print(f"  精确率 (Precision):    {overall_metrics['precision']:.4f}")
    print(f"  召回率 (Recall):       {overall_metrics['recall']:.4f}")
    print(f"  F1 分数:               {overall_metrics['f1_score']:.4f}")
    
    # 保存结果
    results_json = {
        'config': {
            'image_dir': str(image_dir),
            'mask_dir': str(mask_dir),
            'text_prompt': text_prompt,
            'checkpoint_path': checkpoint_path,
            'num_images': len(results),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        'overall_metrics': overall_metrics,
        'per_image_results': results
    }
    
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_dir}")
    print(f"  - 评估报告: {results_path}")
    print(f"  - 可视化结果: {vis_dir}")
    print(f"  - 预测掩码: {pred_dir}")
    print(f"\n{'='*60}")
    
    return overall_metrics, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='批量图像推理和评估')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='图像目录路径')
    parser.add_argument('--mask_dir', type=str, required=True,
                       help='标注掩码目录路径')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录路径')
    parser.add_argument('--prompt', type=str, required=True,
                       help='文本提示词（如 person, car, dog 等）')
    parser.add_argument('--checkpoint', type=str, default='sam3.pt',
                       help='模型权重路径（默认: sam3.pt）')
    parser.add_argument('--mask_suffix', type=str, default='_mask',
                       help='标注文件后缀（默认: _mask）')
    
    args = parser.parse_args()
    
    batch_inference_and_eval(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        output_dir=args.output_dir,
        text_prompt=args.prompt,
        checkpoint_path=args.checkpoint,
        mask_suffix=args.mask_suffix
    )
