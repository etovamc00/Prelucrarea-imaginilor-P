import cv2
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)

def preprocess_image(image_path):
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_resized = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)

    tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1) / 255.0
    return tensor.unsqueeze(0), img_resized

def synthetic_reflection_blending(clean_img, reflection_img, alpha=0.35):
    clean_res = cv2.resize(clean_img, IMG_SIZE)
    refl_res = cv2.resize(reflection_img, IMG_SIZE)

    blurred_refl = cv2.GaussianBlur(refl_res, (15, 15), 0)
    blended = cv2.addWeighted(clean_res, 1.0, blurred_refl, alpha, 0)
    return np.clip(blended, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    Path("demo_outputs").mkdir(exist_ok=True)

    sample_in = "dataset_sir2/test/SolidObjectDataset/blended/0001.jpg"
    sample_gt = "dataset_sir2/test/SolidObjectDataset/transmission_layer/0001.jpg"
    sample_rf = "dataset_sir2/test/SolidObjectDataset/reflection_layer/0001.jpg"

    tensor_out, img_resized = preprocess_image(sample_in)
    print(f"Tensor gata pentru model! Forma: {tensor_out.shape}, "
          f"Min: {tensor_out.min():.2f}, "
          f"Max: {tensor_out.max():.2f}")

    # generarea unei img sintetice
    gt = cv2.cvtColor(cv2.imread(sample_gt), cv2.COLOR_BGR2RGB)
    rf = cv2.cvtColor(cv2.imread(sample_rf), cv2.COLOR_BGR2RGB)
    gt_res = cv2.resize(gt, IMG_SIZE)
    rf_blur = cv2.GaussianBlur(cv2.resize(rf, IMG_SIZE), (15, 15), 0)
    syn = synthetic_reflection_blending(gt, rf, alpha=0.45)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, im, t in zip(ax, [gt_res, rf_blur, syn],
                        ["1.Scena curata (T)", "2.Reflexie blurata (R)", "3.Rezultat sintetic (T + αR)"]):
        a.imshow(im)
        a.set_title(t)
        a.axis("off")
    plt.tight_layout()
    plt.savefig("demo_outputs/exemplu_synthetic_blending.png", dpi=150)
    plt.close()

    print("Salvat in: demo_outputs/exemplu_synthetic_blending.png")