import shutil
import random
import re
from pathlib import Path
from collections import defaultdict

SOURCE_ROOT = Path(r"C:\Users\Elena\Documents\FACULTATE\PI\proiect\sir2")
TARGET_ROOT = Path("dataset_sir2")

CATEGORY_MAP = {
    "postcard": "PostcardDataset",
    "solidobject": "SolidObjectDataset",
    "wildscene": "WildSceneDataset",
}
LAYERS = ["blended", "transmission_layer", "reflection_layer"]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

if not SOURCE_ROOT.exists():
    raise FileNotFoundError(f"Calea SOURCE_ROOT nu există: {SOURCE_ROOT}")

if TARGET_ROOT.exists():
    shutil.rmtree(TARGET_ROOT)

for layer in LAYERS:
    (TARGET_ROOT / "train" / layer).mkdir(parents=True, exist_ok=True)

def classify_image(file_path, total_in_folder):
    stem = file_path.stem.lower()

    if stem in {"t", "g", "gt"}:
        return "gt", "default"
    if stem in {"r", "refl", "reflection"}:
        return "refl", "default"
    if stem in {"m", "original", "orig", "blended", "mix"}:
        return "inp", "default"

    tokens = [t for t in re.split(r"[_\-\s\(\)]+", stem) if t]
    for role, tags in [("gt", {"g", "t", "gt"}), ("refl", {"r", "refl"}), ("inp", {"m", "original", "orig"})]:
        for tag in tags:
            if tag in tokens:
                idx = tokens.index(tag)
                key = "_".join(tokens[:idx] + tokens[idx + 1:]) or "default"
                return role, key

    if total_in_folder == 3:
        if stem.startswith(("g", "t")) or stem.endswith(("g", "t")):
            return "gt", "default"
        if stem.startswith("r") or stem.endswith("r"):
            return "refl", "default"
        if stem.startswith("m") or stem.endswith("m") or "orig" in stem:
            return "inp", "default"

    return None, None

random.seed(42)
train_counter = 1
sample_unmatched = None

found_categories = {}
for folder in sorted([SOURCE_ROOT] + list(SOURCE_ROOT.rglob("*"))):
    if folder.is_dir():
        norm = folder.name.lower().replace(" ", "")
        for key, std_name in CATEGORY_MAP.items():
            if key in norm and std_name not in found_categories:
                found_categories[std_name] = folder

print("Foldere principale detectate:", {k: v.name for k, v in found_categories.items()})

for standard_cat, cat_dir in found_categories.items():
    triplets = []

    for folder in sorted([cat_dir] + list(cat_dir.rglob("*"))):
        if not folder.is_dir():
            continue
        imgs = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS]
        if len(imgs) < 3:
            continue

        groups = defaultdict(dict)
        for img in imgs:
            role, key = classify_image(img, len(imgs))
            if role:
                groups[key][role] = img

        matched_any = False
        for key, item in sorted(groups.items()):
            if {"inp", "gt", "refl"}.issubset(item.keys()):
                triplets.append((item["inp"], item["gt"], item["refl"]))
                matched_any = True

        if not matched_any and sample_unmatched is None:
            sample_unmatched = (folder, [f.name for f in imgs[:6]])

    print(f" -> {standard_cat}: {len(triplets)} seturi complete gasite.")
    random.shuffle(triplets)

    split_idx = int(len(triplets) * 0.8)
    train_sets = triplets[:split_idx]
    test_sets = triplets[split_idx:]

    for m_f, g_f, r_f in train_sets:
        fid = f"{train_counter:04d}"
        train_counter += 1
        shutil.copy(m_f, TARGET_ROOT / "train" / "blended" / f"{fid}{m_f.suffix.lower()}")
        shutil.copy(g_f, TARGET_ROOT / "train" / "transmission_layer" / f"{fid}{g_f.suffix.lower()}")
        shutil.copy(r_f, TARGET_ROOT / "train" / "reflection_layer" / f"{fid}{r_f.suffix.lower()}")

    for layer in LAYERS:
        (TARGET_ROOT / "test" / standard_cat / layer).mkdir(parents=True, exist_ok=True)

    for idx, (m_f, g_f, r_f) in enumerate(test_sets, start=1):
        fid = f"{idx:04d}"
        shutil.copy(m_f, TARGET_ROOT / "test" / standard_cat / "blended" / f"{fid}{m_f.suffix.lower()}")
        shutil.copy(g_f, TARGET_ROOT / "test" / standard_cat / "transmission_layer" / f"{fid}{g_f.suffix.lower()}")
        shutil.copy(r_f, TARGET_ROOT / "test" / standard_cat / "reflection_layer" / f"{fid}{r_f.suffix.lower()}")

print(f"Total seturi în train: {train_counter - 1}")
if train_counter == 1 and sample_unmatched:
    print("ATENȚIE - Exemplu de fișiere gasite intr-un folder:", sample_unmatched)