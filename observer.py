from ultralytics import YOLO
import supervision as sv
import numpy as np
from PIL import Image

# ── Model Loader ──────────────────────────────────────────────────────────────
def load_model(model_path: str) -> YOLO:
    """Load a YOLO model from the given path."""
    return YOLO(model_path)

# ── Core Detection Pipeline ───────────────────────────────────────────────────
def detect_vehicles(image_pil: Image.Image, model: YOLO) -> tuple[Image.Image, dict]:
    """
    Run vehicle detection on a PIL image.

    Steps:
        1. Convert PIL image → numpy array (RGB)
        2. Run YOLO inference
        3. Apply NMS to remove duplicate overlapping boxes
        4. Annotate image with bounding boxes + labels
        5. Count detections per class

    Args:
        image_pil : PIL.Image — the uploaded image
        model     : YOLO     — loaded YOLO model

    Returns:
        annotated_image : PIL.Image — image with bounding boxes drawn
        class_counts    : dict      — e.g. {"car": 4, "van": 2, "bus": 1}
    """

    # Step 1 — Convert to numpy RGB array for YOLO
    image_np = np.array(image_pil.convert("RGB"))

    # Step 2 — Run inference (verbose=False suppresses per-frame console output)
    results = model(image_np, verbose=False)[0]

    # Step 3 — Parse detections + apply NMS (removes overlapping duplicate boxes)
    detections = sv.Detections.from_ultralytics(results).with_nms()

    # Step 4 — Annotate: draw boxes and class + confidence labels
    box_annotator   = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated = image_pil.copy()
    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections)

    # Step 5 — Count detections per class
    class_counts = {}
    for class_name in detections.data.get("class_name", []):
        class_counts[str(class_name)] = class_counts.get(str(class_name), 0) + 1

    return annotated, class_counts