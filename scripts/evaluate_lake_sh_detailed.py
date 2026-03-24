import argparse
import json
import os
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from pycocotools import mask as mask_utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SAM3 Lake SH Fine-tuning Results")
    parser.add_argument("--pred_json", type=str, default="output/lake_sh_fine_tune/dumps/lake_sh/segmentation/coco_predictions_segm.json", help="Path to prediction JSON")
    parser.add_argument("--gt_json", type=str, default="data/lake_sh/annotations/instances_val.json", help="Path to Ground Truth JSON")
    parser.add_argument("--img_dir", type=str, default="data/lake_sh/images", help="Path to image directory")
    parser.add_argument("--output_dir", type=str, default="output/lake_sh_fine_tune/detailed_eval", help="Directory to save results")
    parser.add_argument("--vis_count", type=int, default=10, help="Number of images to visualize")
    parser.add_argument("--score_thr", type=float, default=0.5, help="Score threshold for predictions")
    return parser.parse_args()

def rle_to_mask(rle):
    if isinstance(rle, list):
         # Polygon format not expected in prediction json, but strictly speaking possible. 
         # Preds are usually RLE.
         pass
    return mask_utils.decode(rle)

def polygons_to_mask(polygons, height, width):
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly).reshape((-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    print(f"Loading Ground Truth from {args.gt_json}...")
    with open(args.gt_json, 'r') as f:
        gt_data = json.load(f)

    print(f"Loading Predictions from {args.pred_json}...")
    with open(args.pred_json, 'r') as f:
        pred_data = json.load(f)

    # Index GT by image_id
    gt_by_img = defaultdict(list)
    for ann in gt_data['annotations']:
        gt_by_img[ann['image_id']].append(ann)
    
    img_info_map = {img['id']: img for img in gt_data['images']}

    # Index Preds by image_id
    pred_by_img = defaultdict(list)
    for pred in pred_data:
        pred_by_img[pred['image_id']].append(pred)

    # Metrics accumulators (Semantic Segmentation Metrics)
    total_inter = 0
    total_union = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    print("Evaluating...")
    
    # Iterate over all validation images
    processed_count = 0
    visualized_count = 0
    
    valid_image_ids = sorted(list(img_info_map.keys()))
    
    for img_id in tqdm(valid_image_ids):
        img_info = img_info_map[img_id]
        h, w = img_info['height'], img_info['width']
        
        # 1. Build GT Mask (H, W) - Semantic (Binary)
        gt_mask = np.zeros((h, w), dtype=np.uint8)
        if img_id in gt_by_img:
            for ann in gt_by_img[img_id]:
                if 'segmentation' in ann:
                    seg = ann['segmentation']
                    if isinstance(seg, list):
                        # Polygon
                        m = polygons_to_mask(seg, h, w)
                    elif isinstance(seg, dict) and 'counts' in seg:
                        # RLE
                        if isinstance(seg['counts'], list):
                             m = mask_utils.decode(mask_utils.frPyObjects(seg, h, w))
                        else:
                             m = mask_utils.decode(seg)
                    else:
                        continue
                    gt_mask = np.maximum(gt_mask, m)
        
        # 2. Build Pred Mask (H, W) - Semantic (Binary)
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        if img_id in pred_by_img:
            for pred in pred_by_img[img_id]:
                if pred['score'] < args.score_thr:
                    continue
                rle = pred['segmentation']
                m = mask_utils.decode(rle)
                pred_mask = np.maximum(pred_mask, m)
        
        # 3. Compute Stats
        intersection = (gt_mask & pred_mask).sum()
        union = (gt_mask | pred_mask).sum()
        tp = intersection
        fp = (pred_mask & (1 - gt_mask)).sum()
        fn = ((1 - pred_mask) & gt_mask).sum()
        
        total_inter += intersection
        total_union += union
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # 4. Visualization (only for first N images)
        if visualized_count < args.vis_count:
            img_path = os.path.join(args.img_dir, img_info['file_name'])
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Composition
                fig, ax = plt.subplots(1, 4, figsize=(20, 5))
                
                # Original
                ax[0].imshow(img)
                ax[0].set_title("Original Image")
                ax[0].axis('off')
                
                # Ground Truth
                ax[1].imshow(img)
                ax[1].imshow(gt_mask, alpha=0.5, cmap='Greens')
                ax[1].set_title("Ground Truth (Green)")
                ax[1].axis('off')
                
                # Prediction
                ax[2].imshow(img)
                ax[2].imshow(pred_mask, alpha=0.5, cmap='Reds')
                ax[2].set_title(f"Prediction (Red) thr={args.score_thr}")
                ax[2].axis('off')
                
                # Comparison
                # TP: Yellow (Red+Green), FP: Red, FN: Green, TN: None
                comp_map = np.zeros((h, w, 3), dtype=np.uint8)
                # Green channel for GT
                comp_map[gt_mask == 1, 1] = 255 
                # Red channel for Pred
                comp_map[pred_mask == 1, 0] = 255
                
                ax[3].imshow(img)
                ax[3].imshow(comp_map, alpha=0.6)
                ax[3].set_title("Overlay (Y=TP, R=FP, G=FN)")
                ax[3].axis('off')
                
                save_path = os.path.join(vis_dir, f"{os.path.basename(img_info['file_name'])[:-4]}_eval.jpg")
                plt.tight_layout()
                plt.savefig(save_path)
                plt.close()
                visualized_count += 1

    # 5. Global Metrics Calculation
    epsilon = 1e-7
    iou = total_inter / (total_union + epsilon)
    precision = total_tp / (total_tp + total_fp + epsilon)
    recall = total_tp / (total_tp + total_fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    dice = 2 * total_inter / (2 * total_inter + total_fp + total_fn + epsilon)

    results = {
        "IoU": float(iou),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1_Score": float(f1),
        "Dice": float(dice),
        "TP_pixels": int(total_tp),
        "FP_pixels": int(total_fp),
        "FN_pixels": int(total_fn)
    }

    print("\n" + "="*40)
    print("       Global Evaluation Results        ")
    print("="*40)
    for k, v in results.items():
        if "pixels" not in k:
            print(f"{k:<15}: {v:.4f}")
        else:
            print(f"{k:<15}: {v}")
    print("="*40)

    # Save metrics to JSON
    with open(os.path.join(args.output_dir, "metrics.json"), 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Metrics saved to {os.path.join(args.output_dir, 'metrics.json')}")
    print(f"Visualizations saved to {vis_dir}")

if __name__ == "__main__":
    main()
