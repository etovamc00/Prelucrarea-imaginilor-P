import cv2
import torch
import importlib
from pathlib import Path
import xreflection.models
from xreflection.utils.registry import MODEL_REGISTRY

IMG_SIZE = (224, 224)

def get_dsrnet_class():
    """Gaseste automat clasa DSRNetModel din xreflection/models."""
    try:
        return MODEL_REGISTRY.get("DSRNetModel")
    except KeyError:
        for mod_file in Path("xreflection/models").glob("*_model.py"):
            mod = importlib.import_module(f"xreflection.models.{mod_file.stem}")
            if hasattr(mod, "DSRNetModel"):
                return getattr(mod, "DSRNetModel")
    raise ImportError("Clasa DSRNetModel nu a fost gasita in xreflection/models.")

def load_dsrnet_from_ckpt(ckpt_path):
    """Incarca arhitectura DSRNet si greutatile din last.ckpt pe CPU."""
    DSRNetModel = get_dsrnet_class()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    opt = ckpt.get("hyper_parameters", {}).get("opt", {})

    if "path" in opt:
        opt["path"]["pretrain_network_g"] = None
        opt["path"]["pretrain_network_g_ema"] = None
    opt["num_gpu"] = 0

    model = DSRNetModel(opt)
    if hasattr(model, "setup"):
        model.setup("test")

    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()
    return model

def run_inference(input_dir, ckpt_path="checkpoints/last.ckpt", out_clean_dir="rezultate_dsrnet", out_refl_dir="rezultate_reflexii"):
    Path(out_clean_dir).mkdir(parents=True, exist_ok=True)
    Path(out_refl_dir).mkdir(parents=True, exist_ok=True)

    print("Se incarca modelul DSRNet din:", ckpt_path)
    model = load_dsrnet_from_ckpt(ckpt_path)

    image_files = sorted(Path(input_dir).glob("*.*"))[:8]
    print(f"Rulare inferenta pe {len(image_files)} imagini...")

    for img_path in image_files:
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(cv2.resize(img_bgr, IMG_SIZE), cv2.COLOR_BGR2RGB)
        inp_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0

        with torch.no_grad():
            model.testing(inp_tensor)
            t_clean, r_refl = model.output

        # Modelul lucreaza direct in [0, 1]
        clean_np = (t_clean.clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        refl_np = (r_refl.clamp(0, 1).squeeze(0).permute(1, 2, 0).numpy() * 255.0).astype("uint8")

        cv2.imwrite(f"{out_clean_dir}/{img_path.stem}.png", cv2.cvtColor(clean_np, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{out_refl_dir}/{img_path.stem}_reflexie.png", cv2.cvtColor(refl_np, cv2.COLOR_RGB2BGR))
        print(f"Procesat: {img_path.name}")

    print("Inferenta finalizata! Imaginile curate sunt in:", out_clean_dir)

if __name__ == "__main__":
    run_inference(
        input_dir="dataset_sir2/test/SolidObjectDataset/blended",
        ckpt_path="checkpoints/last.ckpt"
    )