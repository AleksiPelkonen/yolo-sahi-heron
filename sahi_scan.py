#pip install ultralytics sahi pillow

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ------------ SETTINGS ------------

# Folder containing the Ultralytics run with weights/best.pt
RUN_DIR = Path("/path/to/yolo_run")   # e.g. "/Users/name/runs/heron_run"

# Folder containing the orthophoto tiles / images to scan
IMG_DIR = Path("/path/to/tiles_folder")

# Where to save SAHI outputs (visuals only for images with detections)
PROJECT_DIR = Path("/path/to/sahi_outputs")
RUN_NAME = "heron_sahi_scan"

# Device:
#   "cpu"
#   "mps"  Apple Silicon GPU (if PyTorch MPS is configured)
#   "cuda:0"  NVIDIA GPU
DEVICE = "cpu"

# Number of parallel worker processes
NUM_WORKERS = 2

# SAHI / model settings 
CONFIDENCE_THRESHOLD = 0.20
IMAGE_SIZE = 1536
SLICE_HEIGHT = 768
SLICE_WIDTH = 768
OVERLAP_HEIGHT_RATIO = 0.20
OVERLAP_WIDTH_RATIO = 0.20
POSTPROCESS_TYPE = "GREEDYNMM"
POSTPROCESS_MATCH_METRIC = "IOU"
POSTPROCESS_MATCH_THRESHOLD = 0.40

# Valid image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# ------------ END OF SETTINGS ------------

VISUALS_DIR = PROJECT_DIR / RUN_NAME / "visuals"
VISUALS_DIR.mkdir(parents=True, exist_ok=True)


def resolve_model_path() -> Path:
    weights_dir = RUN_DIR / "weights"
    candidates = [weights_dir / "best.pt"]
    model_pt = next((p for p in candidates if p.exists()), None)

    if model_pt is None:
        pts = sorted(weights_dir.glob("*.pt"))
        if not pts:
            raise FileNotFoundError(f"No .pt files found under {weights_dir}")
        model_pt = pts[0]

    return model_pt


def process_one_image(img_path_str: str) -> Tuple[str, bool]:
    from pathlib import Path  # re-import inside worker
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    img_path = Path(img_path_str)

    global _MODEL, _MODEL_PT

    if "_MODEL_PT" not in globals():
        _MODEL_PT = resolve_model_path()

    if "_MODEL" not in globals():
        try:
            _MODEL = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=str(_MODEL_PT),
                confidence_threshold=CONFIDENCE_THRESHOLD,
                image_size=IMAGE_SIZE,
                device=DEVICE,
            )
        except TypeError:
            _MODEL = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=str(_MODEL_PT),
                confidence_threshold=CONFIDENCE_THRESHOLD,
                imgsz=IMAGE_SIZE,
                device=DEVICE,
            )

    result = get_sliced_prediction(
        image=str(img_path),
        detection_model=_MODEL,
        slice_height=SLICE_HEIGHT,
        slice_width=SLICE_WIDTH,
        overlap_height_ratio=OVERLAP_HEIGHT_RATIO,
        overlap_width_ratio=OVERLAP_WIDTH_RATIO,
        postprocess_type=POSTPROCESS_TYPE,
        postprocess_match_metric=POSTPROCESS_MATCH_METRIC,
        postprocess_match_threshold=POSTPROCESS_MATCH_THRESHOLD,
    )

    if len(result.object_prediction_list) == 0:
        return img_path.name, False

    result.export_visuals(
        export_dir=str(VISUALS_DIR),
        file_name=img_path.stem,
    )
    return img_path.name, True


def main() -> None:
    assert RUN_DIR.exists(), f"RUN_DIR not found: {RUN_DIR}"
    assert IMG_DIR.exists(), f"IMG_DIR not found: {IMG_DIR}"

    model_pt = resolve_model_path()
    print(f"Using checkpoint: {model_pt}")
    print(f"Scanning images under: {IMG_DIR}")
    print(f"Saving hit visuals to: {VISUALS_DIR}")
    print(f"Device: {DEVICE}")

    img_paths = [
        p for p in sorted(IMG_DIR.iterdir())
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    ]

    num_total = len(img_paths)
    print(f"Found {num_total} images to process.")

    if num_total == 0:
        print("No images found. Check IMG_DIR and extensions.")
        return

    img_paths_str = [str(p) for p in img_paths]
    num_with_dets = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        for i, (fname, had_det) in enumerate(
            ex.map(process_one_image, img_paths_str),
            start=1,
        ):
            if had_det:
                num_with_dets += 1
                print(f"[{i}/{num_total}] {fname}: detections -> visual saved.")
            else:
                print(f"[{i}/{num_total}] {fname}: no detections -> skipped.")

    print(f"Done. {num_with_dets}/{num_total} images had at least one detection.")


if __name__ == "__main__":
    main()
