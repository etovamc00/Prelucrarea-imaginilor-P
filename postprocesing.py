import re
import time
import json
import cv2
import torch
import lpips
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import warnings
warnings.filterwarnings("ignore")

IMG_SIZE = (224, 224)
lpips_model = lpips.LPIPS(net='alex')

def apply_postprocessing(img_rgb):
    """Corectie fina: Gamma exposure + Damped White Balance + CLAHE usor."""
    # Corectie Gamma
    img_float = img_rgb.astype(np.float32) / 255.0
    gamma_corrected = np.power(np.clip(img_float, 0, 1), 0.72) * 255.0

    # Balans de alb
    means = gamma_corrected.mean(axis=(0, 1))
    gray_mean = means.mean()
    scales = 1.0 + 0.35 * ((gray_mean / (means + 1e-6)) - 1.0)
    wb_img = np.clip(gamma_corrected * scales, 0, 255).astype(np.uint8)

    # Contrast local fin in spatiul LAB
    lab = cv2.cvtColor(wb_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    contrast_img = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2RGB)

    # Saturatie moderata (+15%)
    hsv = cv2.cvtColor(contrast_img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)
    final_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return final_img

def compute_metrics(pred_rgb, gt_rgb):
    """Calculeaza PSNR, SSIM si LPIPS intre imaginea finala si Ground Truth."""
    val_psnr = psnr(gt_rgb, pred_rgb, data_range=255)
    val_ssim = ssim(gt_rgb, pred_rgb, channel_axis=2, data_range=255)

    t_pred = torch.from_numpy(pred_rgb).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    t_gt = torch.from_numpy(gt_rgb).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0

    with torch.no_grad():
        val_lpips = lpips_model(t_pred, t_gt).item()

    return round(float(val_psnr), 2), round(float(val_ssim), 4), round(float(val_lpips), 4)

def save_comparison_collage(inp, raw_pred, post_pred, gt, save_path):
    """Salveaza o imagine comparativa cu cele 4 etape."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = ["1. Input (Cu reflexie)", "2. DSRNet (Brut)", "3. Final (+ Post-proc)", "4. Ground Truth"]
    images = [inp, raw_pred, post_pred, gt]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def run_evaluation_pipeline(input_dir, dsrnet_out_dir, gt_dir, output_dir="demo_outputs/final_results"):
    for nume, cale in [("input_dir", input_dir), ("dsrnet_out_dir", dsrnet_out_dir), ("gt_dir", gt_dir)]:
        if not Path(cale).exists():
            raise FileNotFoundError(f"Folderul {nume} NU exista la calea: {Path(cale).resolve()}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logs = []

    pred_files = sorted([f for f in Path(dsrnet_out_dir).iterdir() if f.is_file()])[:8]
    print("Fisiere detectate in folderul DSRNet:", [f.name for f in pred_files])

    for pred_path in pred_files:
        # Extrage doar numarul (ex: '0001' chiar daca fisierul e '0001.jpg.png' sau '0001_out.png')
        match = re.search(r"\d+", pred_path.stem)
        base_id = match.group(0) if match else pred_path.stem

        inp_matches = list(Path(input_dir).glob(f"{base_id}.*"))
        gt_matches = list(Path(gt_dir).glob(f"{base_id}.*"))

        if not inp_matches or not gt_matches:
            print(f"Sarit peste {pred_path.name}: nu am gasit ID-ul '{base_id}' in {input_dir}")
            continue

        inp_rgb = cv2.resize(cv2.cvtColor(cv2.imread(str(inp_matches[0])), cv2.COLOR_BGR2RGB), IMG_SIZE)
        raw_pred_rgb = cv2.resize(cv2.cvtColor(cv2.imread(str(pred_path)), cv2.COLOR_BGR2RGB), IMG_SIZE)
        gt_rgb = cv2.resize(cv2.cvtColor(cv2.imread(str(gt_matches[0])), cv2.COLOR_BGR2RGB), IMG_SIZE)

        start_time = time.time()
        final_rgb = apply_postprocessing(raw_pred_rgb)
        runtime = round(time.time() - start_time, 4)

        val_psnr, val_ssim, val_lpips = compute_metrics(final_rgb, gt_rgb)

        save_comparison_collage(inp_rgb, raw_pred_rgb, final_rgb, gt_rgb, f"{output_dir}/colaj_{base_id}.png")

        logs.append({
            "Imagine": pred_path.name,
            "PSNR": val_psnr,
            "SSIM": val_ssim,
            "LPIPS": val_lpips,
            "Runtime_sec": runtime
        })

    df = pd.DataFrame(logs)
    df.to_csv(f"{output_dir}/metrici_evaluare.csv", index=False)
    with open(f"{output_dir}/metrici_evaluare.json", "w") as f:
        json.dump(logs, f, indent=4)

    print("Evaluare finalizata! Rezultate salvate in:", output_dir)
    print(df)

if __name__ == "__main__":
    run_evaluation_pipeline(
        input_dir="dataset_sir2/test/SolidObjectDataset/blended",
        dsrnet_out_dir=r"rezultate_dsrnet",
        gt_dir="dataset_sir2/test/SolidObjectDataset/transmission_layer"
    )