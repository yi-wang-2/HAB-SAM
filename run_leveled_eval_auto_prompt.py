"""
在 leveled 数据集上使用 SAM3 进行自动提示词推理与评估。

默认从：
  /home/ucas_yw/algorithm/unet_semantic-segmentation-main/lake/datasets/split/leveled
读取 level_1/2/3，并在每个 level 下优先使用：
  resplit_output/images/<split> 与 resplit_output/gtFine/<split>
若不存在，则回退到 level 目录下的 images/ 与 gtFine/。

评估结果将按 level 输出到 output_root 目录。
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from batch_inference_eval import SegmentationMetrics, save_visualization
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

try:
    from sam3.agent.client_llm import send_generate_request
except Exception:
    send_generate_request = None


IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]
DEFAULT_LEVELED_ROOT = "/home/ucas_yw/algorithm/unet_semantic-segmentation-main/lake/datasets/split/leveled"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3 leveled 自动提示词推理评估")
    parser.add_argument("--leveled_root", default=DEFAULT_LEVELED_ROOT, help="level_1/2/3 根目录")
    parser.add_argument("--levels", nargs="*", default=None, help="指定要评估的 level 文件夹名，如 level_1")
    parser.add_argument("--split", default="test", help="优先使用的 split 子目录名 (test/val/train)")
    parser.add_argument("--checkpoint", default="sam3.pt", help="SAM3 权重路径")
    parser.add_argument("--output_root", default="output/leveled_eval_sam3", help="输出目录")

    # prompt 相关
    parser.add_argument("--prompt_strategy", default="auto", choices=["auto", "fixed"], help="提示词策略")
    parser.add_argument("--fixed_prompt", default="water", help="固定提示词 (prompt_strategy=fixed)")
    parser.add_argument("--fallback_prompt", default="water", help="自动提示词失败时的回退提示词")
    parser.add_argument("--use_mask_prompt", action="store_true", help="是否使用 mask prompt")
    parser.add_argument("--use_5_chan", action="store_true", help="是否使用5通道(原图3通道+差异1通道+absi1通道)")
    parser.add_argument("--use_channel_attention", action="store_true", help="是否启用输入通道注意力")
    parser.add_argument("--use_vit_adapter", action="store_true", help="是否使用 ViT adapter")
    parser.add_argument("--adapter_ratio", type=float, default=8.0, help="ViT adapter ratio")
    parser.add_argument("--adapter_init_scale", type=float, default=1e-3, help="ViT adapter init scale")

    # LLM 自动提示词配置（可选）
    parser.add_argument("--llm_server_url", default=None, help="OpenAI 兼容服务地址")
    parser.add_argument("--llm_api_key", default=None, help="OpenAI 兼容服务 API Key")
    parser.add_argument("--llm_model", default=None, help="模型名称")

    # 阈值设置
    parser.add_argument("--confidence_threshold", type=float, default=0.5, help="检测置信度阈值")
    parser.add_argument("--mask_threshold", type=float, default=0.5, help="掩码二值化阈值")
    parser.add_argument("--score_threshold", type=float, default=None, help="预测分数筛选阈值")

    return parser.parse_args()


def list_level_dirs(leveled_root: str, levels: Optional[List[str]]) -> List[Path]:
    root = Path(leveled_root)
    if levels:
        return [root / level for level in levels]
    return sorted([p for p in root.iterdir() if p.is_dir()])


def resolve_data_dirs(level_dir: Path, split: str) -> Tuple[Path, Path]:
    resplit_root = level_dir / "resplit_output"
    resplit_image_dir = resplit_root / "images" / split
    resplit_mask_dir = resplit_root / "gtFine" / split
    if resplit_image_dir.is_dir() and resplit_mask_dir.is_dir():
        return resplit_image_dir, resplit_mask_dir

    image_dir = level_dir / "images"
    mask_dir = level_dir / "gtFine"
    return image_dir, mask_dir


def find_images(image_dir: Path) -> List[Path]:
    files: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(image_dir.glob(f"*{ext}"))
    return sorted(files)


def load_ground_truth_mask_any(mask_path: Path) -> Optional[np.ndarray]:
    if not mask_path.exists():
        return None
    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)
    # 兼容 labelIds（可能是 0/1 或多值）：非零即前景
    mask = (mask > 0).astype(np.uint8)
    return mask


def find_mask_path(mask_dir: Path, stem: str) -> Optional[Path]:
    candidates = [
        mask_dir / f"{stem}_gtFine_labelIds.png",
        mask_dir / f"{stem}_gtFine_labelIds.jpg",
        mask_dir / f"{stem}.png",
        mask_dir / f"{stem}.jpg",
        mask_dir / f"{stem}_mask.png",
        mask_dir / f"{stem}_mask.jpg",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def sanitize_prompt(text: str) -> str:
    if text is None:
        return ""
    text = text.strip().strip('"').strip("'")
    text = text.replace("\n", " ").strip()
    # 保留前 6 个词，避免过长
    words = [w for w in text.split(" ") if w]
    return " ".join(words[:6])


def generate_prompt_auto(
    image_path: Path,
    cache: Dict[str, str],
    fallback_prompt: str,
    llm_server_url: Optional[str],
    llm_api_key: Optional[str],
    llm_model: Optional[str],
) -> str:
    cache_key = str(image_path)
    if cache_key in cache:
        return cache[cache_key]

    prompt = ""
    if send_generate_request and (llm_server_url or llm_api_key or llm_model):
        system_prompt = (
            "You are a prompt generator for segmentation. "
            "Given an image, return a short, simple noun phrase that best describes "
            "the main foreground object. Return only the phrase, no punctuation or extra text."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": "Generate a short noun phrase for the main foreground object."},
                ],
            },
        ]
        try:
            raw = send_generate_request(
                messages,
                server_url=llm_server_url,
                model=llm_model or "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                api_key=llm_api_key,
            )
            prompt = sanitize_prompt(raw or "")
        except Exception:
            prompt = ""

    if not prompt:
        prompt = fallback_prompt

    cache[cache_key] = prompt
    return prompt


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def select_best_mask(
    masks: Optional[torch.Tensor],
    scores: Optional[torch.Tensor],
    score_threshold: Optional[float],
) -> Optional[np.ndarray]:
    if masks is None or len(masks) == 0:
        return None

    if scores is not None and score_threshold is not None:
        keep = scores >= score_threshold
        masks = masks[keep]
        scores = scores[keep]
        if masks is None or len(masks) == 0:
            return None

    if scores is not None and len(scores) == len(masks):
        best_idx = int(torch.argmax(scores).item())
    else:
        best_idx = 0

    best_mask = masks[best_idx]
    if best_mask.ndim == 3:
        best_mask = best_mask[0]
    return best_mask.cpu().numpy()


def evaluate_level(
    level_name: str,
    image_dir: Path,
    mask_dir: Path,
    output_dir: Path,
    processor: Sam3Processor,
    device: torch.device,
    prompt_strategy: str,
    fixed_prompt: str,
    fallback_prompt: str,
    llm_server_url: Optional[str],
    llm_api_key: Optional[str],
    llm_model: Optional[str],
    mask_threshold: float,
    score_threshold: Optional[float],
    use_mask_prompt: bool,
    use_5_chan: bool = False,
) -> Dict:
    ensure_dir(output_dir)
    vis_dir = output_dir / "visualizations"
    pred_dir = output_dir / "predictions"
    ensure_dir(vis_dir)
    ensure_dir(pred_dir)

    cache_path = output_dir / "prompt_cache.json"
    prompt_cache: Dict[str, str] = {}
    if cache_path.exists():
        try:
            prompt_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            prompt_cache = {}

    images = find_images(image_dir)
    if not images:
        print(f"[WARN] {level_name}: 未找到图像文件: {image_dir}")
        return {}

    metrics_calculator = SegmentationMetrics(num_classes=2)
    results = []

    print(f"\n{'='*60}")
    print(f"Level: {level_name}")
    print(f"图像目录: {image_dir}")
    print(f"标注目录: {mask_dir}")
    print(f"输出目录: {output_dir}")
    print(f"图像数量: {len(images)}")
    print(f"{'='*60}\n")

    for img_path in tqdm(images, desc=f"{level_name}"):
        try:
            stem = img_path.stem
            mask_path = find_mask_path(mask_dir, stem)
            if not mask_path:
                print(f"[WARN] 未找到标注: {img_path.name}")
                continue

            gt_mask = load_ground_truth_mask_any(mask_path)
            if gt_mask is None:
                print(f"[WARN] 读取标注失败: {mask_path.name}")
                continue

            if prompt_strategy == "fixed":
                text_prompt = fixed_prompt
            else:
                text_prompt = generate_prompt_auto(
                    img_path,
                    prompt_cache,
                    fallback_prompt,
                    llm_server_url,
                    llm_api_key,
                    llm_model,
                )

            image = Image.open(img_path).convert("RGB")
            
            if use_5_chan:
                absi_path = Path("data/lake_sh/raw_absi_gray") / level_name / "images" / f"{stem}_raw_absi_gray.png"
                sub_path = Path("data/lake_sh/subtraction_map_gray") / level_name / "images" / f"{stem}_subtraction_gray.png"
                if absi_path.exists() and sub_path.exists():
                    absi_img = Image.open(absi_path).convert("L")
                    sub_img = Image.open(sub_path).convert("L")
                    rgb_arr = np.array(image)
                    absi_arr = np.array(absi_img)[..., None]
                    sub_arr = np.array(sub_img)[..., None]
                    model_input_image = np.concatenate([rgb_arr, absi_arr, sub_arr], axis=-1)
                else:
                    print(f"[WARN] 找不到额外的五通道文件: {absi_path} / {sub_path}")
                    rgb_arr = np.array(image)
                    dummy_arr = np.zeros_like(rgb_arr[..., :2])
                    model_input_image = np.concatenate([rgb_arr, dummy_arr], axis=-1)
            else:
                model_input_image = image
                
            inference_state = processor.set_image(model_input_image)

            if use_mask_prompt:
                p_dir = Path("data/lake_sh/mask-prompt") / level_name / "images"
                mask_prompt_path = p_dir / f"{stem}_diff_mask.png"
                if not mask_prompt_path.exists():
                    mask_prompt_path = next(p_dir.glob(f"{stem}*.png"), None)
                if mask_prompt_path and mask_prompt_path.exists():
                    try:
                        mask_img = Image.open(mask_prompt_path).convert("L")
                        mask_np = np.array(mask_img)
                        mask_tensor = torch.from_numpy((mask_np > 127).astype(np.float32)).to(device)
                        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, H, W)
                        if "geometric_prompt" not in inference_state:
                            inference_state["geometric_prompt"] = processor.model._get_dummy_prompt()
                        inference_state["geometric_prompt"].append_masks(mask_tensor)
                    except Exception as e:
                        print(f"[WARN] 无法读取 mask prompt: {mask_prompt_path}, 错误: {e}")

            inference_state = processor.set_text_prompt(state=inference_state, prompt=text_prompt)

            masks = inference_state.get("masks")
            masks_logits = inference_state.get("masks_logits")
            scores = inference_state.get("scores")

            pred_mask_is_prob = False
            if masks_logits is not None:
                pred_mask = select_best_mask(masks_logits, scores, score_threshold)
                pred_mask_is_prob = True
            else:
                pred_mask = select_best_mask(masks, scores, score_threshold)

            if pred_mask is None:
                pred_mask = np.zeros_like(gt_mask)
            else:
                if pred_mask.shape != gt_mask.shape:
                    # Keep float precision for logits/prob maps when resizing.
                    pred_mask_t = torch.from_numpy(pred_mask).float().unsqueeze(0).unsqueeze(0)
                    if pred_mask_is_prob:
                        pred_mask_t = torch.nn.functional.interpolate(
                            pred_mask_t,
                            size=(gt_mask.shape[0], gt_mask.shape[1]),
                            mode="bilinear",
                            align_corners=False,
                        )
                    else:
                        pred_mask_t = torch.nn.functional.interpolate(
                            pred_mask_t,
                            size=(gt_mask.shape[0], gt_mask.shape[1]),
                            mode="nearest",
                        )
                    pred_mask = pred_mask_t.squeeze(0).squeeze(0).cpu().numpy()
                if pred_mask_is_prob:
                    pred_mask = (pred_mask > mask_threshold).astype(np.uint8)
                else:
                    pred_mask = (pred_mask > 0.5).astype(np.uint8)

            metrics_calculator.update(pred_mask, gt_mask)

            single_metrics = SegmentationMetrics(num_classes=2)
            single_metrics.update(pred_mask, gt_mask)
            img_metrics = single_metrics.get_all_metrics()

            pred_mask_save = Image.fromarray((pred_mask * 255).astype(np.uint8))
            pred_mask_save.save(pred_dir / f"{stem}_pred.png")

            metrics_text = (
                f"Prompt: {text_prompt} | "
                f"IoU: {img_metrics['IoU_per_class'][1]:.4f} | "
                f"Dice: {img_metrics['dice_per_class'][1]:.4f} | "
                f"Pixel Acc: {img_metrics['pixel_accuracy']:.4f}"
            )

            save_visualization(
                image,
                pred_mask,
                gt_mask,
                vis_dir / f"{stem}_comparison.png",
                metrics_text,
            )

            results.append(
                {
                    "image_name": img_path.name,
                    "mask_name": mask_path.name,
                    "prompt": text_prompt,
                    "metrics": img_metrics,
                }
            )

        except Exception as e:
            print(f"[ERROR] 处理 {img_path.name} 失败: {e}")
            continue

    overall_metrics = metrics_calculator.get_all_metrics()
    results_json = {
        "config": {
            "image_dir": str(image_dir),
            "mask_dir": str(mask_dir),
            "prompt_strategy": prompt_strategy,
            "fixed_prompt": fixed_prompt,
            "fallback_prompt": fallback_prompt,
            "num_images": len(results),
        },
        "overall_metrics": overall_metrics,
        "per_image_results": results,
    }

    (output_dir / "evaluation_results.json").write_text(
        json.dumps(results_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    cache_path.write_text(json.dumps(prompt_cache, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[{level_name}] Overall metrics:")
    print(f"  mIoU: {overall_metrics['mIoU']:.4f}")
    print(f"  Pixel Acc: {overall_metrics['pixel_accuracy']:.4f}")
    print(f"  Mean Dice: {overall_metrics['mean_dice']:.4f}")

    return overall_metrics


def main() -> None:
    args = parse_args()
    leveled_root = Path(args.leveled_root)
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    if not leveled_root.exists():
        print(f"[ERROR] leveled_root 不存在: {leveled_root}")
        return

    if not Path(args.checkpoint).exists():
        print(f"[ERROR] checkpoint 不存在: {args.checkpoint}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = build_sam3_image_model(
        checkpoint_path=args.checkpoint, 
        in_chans=5 if args.use_5_chan else 3,
        use_channel_attention=args.use_channel_attention,
        use_vit_adapter=args.use_vit_adapter,
        adapter_ratio=args.adapter_ratio,
        adapter_init_scale=args.adapter_init_scale
    )
    model = model.eval().to(device)
    processor = Sam3Processor(model, confidence_threshold=args.confidence_threshold)

    summary = {}
    for level_dir in list_level_dirs(str(leveled_root), args.levels):
        level_name = level_dir.name
        image_dir, mask_dir = resolve_data_dirs(level_dir, args.split)
        if not image_dir.exists() or not mask_dir.exists():
            print(f"[WARN] {level_name}: 数据目录不存在: {image_dir} | {mask_dir}")
            continue

        level_output_dir = output_root / level_name
        metrics = evaluate_level(
            level_name=level_name,
            image_dir=image_dir,
            mask_dir=mask_dir,
            output_dir=level_output_dir,
            processor=processor,
            device=device,
            prompt_strategy=args.prompt_strategy,
            fixed_prompt=args.fixed_prompt,
            fallback_prompt=args.fallback_prompt,
            llm_server_url=args.llm_server_url,
            llm_api_key=args.llm_api_key,
            llm_model=args.llm_model,
            mask_threshold=args.mask_threshold,
            score_threshold=args.score_threshold,
            use_mask_prompt=args.use_mask_prompt,
            use_5_chan=args.use_5_chan,
        )
        if metrics:
            summary[level_name] = metrics

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n✅ 完成。汇总结果: {summary_path}")


if __name__ == "__main__":
    main()
