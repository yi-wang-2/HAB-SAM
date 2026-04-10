#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def _load_5ch(image_path: Path, absi_path: Path, sub_path: Path) -> np.ndarray:
    rgb = np.array(Image.open(image_path).convert("RGB"))
    absi = np.array(Image.open(absi_path).convert("L"))[..., None]
    sub = np.array(Image.open(sub_path).convert("L"))[..., None]
    return np.concatenate([rgb, absi, sub], axis=-1)


def _to_model_input_tensor(processor: Sam3Processor, image_5ch: np.ndarray, device: str) -> torch.Tensor:
    img = torch.from_numpy(image_5ch).to(device)
    # HWC -> CHW, uint8
    img = img.permute(2, 0, 1).contiguous()
    processor._normalize_transform_for_channels(int(img.shape[0]))
    x = processor.transform(img).unsqueeze(0)
    return x


def _compute_attention_maps(ca_module, x: torch.Tensor):
    with torch.no_grad():
        b, c, _, _ = x.size()
        y_avg = ca_module.fc(ca_module.avg_pool(x).view(b, c))
        y_max = ca_module.fc(ca_module.max_pool(x).view(b, c))
        y_c = ca_module.sigmoid_ca(y_avg + y_max).view(b, c, 1, 1)

        x_c = x * y_c.expand_as(x)

        avg_out = torch.mean(x_c, dim=1, keepdim=True)
        max_out, _ = torch.max(x_c, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        spatial_logits = ca_module.spatial_attention.conv1(y)
        y_s = ca_module.spatial_attention.sigmoid(spatial_logits)

    return y_c.squeeze(0).squeeze(-1).squeeze(-1).cpu().numpy(), y_s.squeeze(0).squeeze(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description="Visualize channel/spatial attention maps for SAM3 InputChannelAttention")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--absi", required=True)
    ap.add_argument("--sub", required=True)
    ap.add_argument("--output-dir", default="output/attention_vis")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--in-chans", type=int, default=5)
    ap.add_argument("--use-vit-adapter", action="store_true")
    ap.add_argument("--adapter-ratio", type=float, default=8.0)
    ap.add_argument("--adapter-init-scale", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_sam3_image_model(
        checkpoint_path=args.checkpoint,
        in_chans=args.in_chans,
        use_channel_attention=True,
        use_vit_adapter=args.use_vit_adapter,
        adapter_ratio=args.adapter_ratio,
        adapter_init_scale=args.adapter_init_scale,
        device=args.device,
        eval_mode=True,
        load_from_HF=False,
    )
    model = model.eval().to(args.device)

    processor = Sam3Processor(model, device=args.device)

    image_5ch = _load_5ch(Path(args.image), Path(args.absi), Path(args.sub))
    x = _to_model_input_tensor(processor, image_5ch, args.device)

    ca_module = model.backbone.vision_backbone.trunk.channel_attention
    if ca_module is None:
        raise RuntimeError("channel_attention is None. Please enable --use_channel_attention during training/inference model build.")

    channel_gate, spatial_gate = _compute_attention_maps(ca_module, x)

    # Save numeric channel gate values
    channel_names = ["R", "G", "B", "ABSI", "SUB"][: len(channel_gate)]
    txt_path = out_dir / "channel_gate_values.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        for n, v in zip(channel_names, channel_gate.tolist()):
            f.write(f"{n}: {v:.6f}\n")

    # Bar plot for channel gate
    plt.figure(figsize=(7, 4))
    plt.bar(channel_names, channel_gate)
    plt.ylim(0, 1.05)
    plt.title("Channel Attention Gate")
    plt.ylabel("Gate Value")
    plt.tight_layout()
    plt.savefig(out_dir / "channel_gate_bar.png", dpi=180)
    plt.close()

    # Spatial map heatmap
    plt.figure(figsize=(6, 6))
    plt.imshow(spatial_gate, cmap="jet")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Spatial Attention Map")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "spatial_gate_heatmap.png", dpi=180)
    plt.close()

    # Overlay on resized RGB (same size as spatial map)
    h, w = spatial_gate.shape
    rgb = np.array(Image.open(args.image).convert("RGB").resize((w, h), Image.BILINEAR))
    heat = plt.get_cmap("jet")(spatial_gate)[..., :3]
    overlay = (0.55 * rgb / 255.0 + 0.45 * heat)
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(out_dir / "spatial_gate_overlay.png")

    print(f"Saved: {txt_path}")
    print(f"Saved: {out_dir / 'channel_gate_bar.png'}")
    print(f"Saved: {out_dir / 'spatial_gate_heatmap.png'}")
    print(f"Saved: {out_dir / 'spatial_gate_overlay.png'}")


if __name__ == "__main__":
    main()
