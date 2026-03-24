
import os
import json
import shutil
import random
import argparse
import numpy as np
from PIL import Image
from pycocotools import mask as mask_util
from tqdm import tqdm

def rle_to_string(rle):
    """
    Convert RLE counts to string if it is binary (bytes).
    """
    if isinstance(rle['counts'], bytes):
        rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def create_coco_json(data_root, output_dir, levels=['level_1', 'level_2', 'level_3'], split_ratio=0.8):
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'annotations'), exist_ok=True)

    image_id_counter = 0
    annotation_id_counter = 0

    all_pairs = []

    print("Collecting images and masks...")
    for level in levels:
        level_img_dir = os.path.join(data_root, level, 'images')
        level_gt_dir = os.path.join(data_root, level, 'gtFine')

        if not os.path.exists(level_img_dir):
            print(f"Warning: {level_img_dir} does not exist. Skipping.")
            continue

        images = [f for f in os.listdir(level_img_dir) if f.endswith(('.jpg', '.png'))]
        
        for img_name in images:
            img_path = os.path.join(level_img_dir, img_name)
            # Find corresponding mask
            basename = os.path.splitext(img_name)[0]
            mask_name = f"{basename}_gtFine_labelIds.png"
            mask_path = os.path.join(level_gt_dir, mask_name)

            if os.path.exists(mask_path):
                all_pairs.append({'image': img_path, 'mask': mask_path})
            else:
                print(f"Warning: Mask not found for {img_name}")

    print(f"Found {len(all_pairs)} image-mask pairs.")
    
    # Shuffle
    random.seed(42)
    random.shuffle(all_pairs)

    # Split
    num_train = int(len(all_pairs) * split_ratio)
    train_pairs = all_pairs[:num_train]
    val_pairs = all_pairs[num_train:]

    splits = {'train': train_pairs, 'val': val_pairs}

    for split_name, pairs in splits.items():
        print(f"Processing {split_name} split ({len(pairs)} images)...")
        
        coco_output = {
            "info": {
                "description": "Lake Algae Dataset",
                "year": 2026,
                "version": "1.0",
                "contributor": "User",
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [
                {"id": 1, "name": "blue-green algal", "supercategory": "algae"}
            ]
        }

        for item in tqdm(pairs):
            img_src = item['image']
            mask_src = item['mask']
            
            # Copy image (or symlink?) Let's copy to be safe and simple
            img_filename = os.path.basename(img_src)
            img_dst = os.path.join(output_dir, 'images', img_filename)
            if not os.path.exists(img_dst):
                shutil.copy2(img_src, img_dst)
            
            # Read Image Info
            with Image.open(img_src) as img:
                width, height = img.size

            image_id = image_id_counter
            image_id_counter += 1
            
            image_info = {
                "id": image_id,
                "file_name": img_filename,
                "width": width,
                "height": height
            }
            coco_output["images"].append(image_info)
            
            # Process Mask
            mask_np = np.array(Image.open(mask_src))
            # Assume 1 is the class of interest
            binary_mask = (mask_np == 1).astype(np.uint8)
            
            # Use Fortran order for RLE encoding as per pycocotools convention for compatibility 
            # (though mask_util.encode expects F-order array usually?)
            # mask_util.encode expects column-major order (Fortran) if input is numpy array
            rle = mask_util.encode(np.asfortranarray(binary_mask))
            rle = rle_to_string(rle)

            # Calculate area and bbox
            area = float(mask_util.area(rle))
            bbox = mask_util.toBbox(rle).tolist() # [x, y, w, h]

            if area > 0:
                ann = {
                    "id": annotation_id_counter,
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": rle,
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                }
                coco_output["annotations"].append(ann)
                annotation_id_counter += 1
        
        output_json_path = os.path.join(output_dir, 'annotations', f'instances_{split_name}.json')
        with open(output_json_path, 'w') as f:
            json.dump(coco_output, f)
        print(f"Saved {output_json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Path closer to level_x folders')
    parser.add_argument('--output_dir', type=str, required=True, help='Output path')
    args = parser.parse_args()
    
    create_coco_json(args.data_root, args.output_dir)
