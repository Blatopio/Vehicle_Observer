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
        3. Filter low-confidence detections (threshold > 0.4)
        4. Apply NMS (threshold=0.3) to remove overlapping duplicate boxes
        5. Build instance labels e.g. "car 1", "car 2", "bus 1"
        6. Annotate image with bounding boxes + instance labels
        7. Count detections per class

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

    # Step 3 — Parse detections + filter low confidence + apply strict NMS
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.confidence > 0.4]  # keep confident detections only
    detections = detections.with_nms(threshold=0.3)       # stricter overlap tolerance

    # Step 4 — Build instance labels e.g. "car 1", "car 2", "bus 1"
    class_counters = {}
    labels = []
    for class_name in detections.data.get("class_name", []):
        class_counters[class_name] = class_counters.get(class_name, 0) + 1
        labels.append(f"{class_name} {class_counters[class_name]}")

    # Step 5 — Annotate: draw boxes and instance labels
    box_annotator   = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated = image_pil.copy()
    annotated = box_annotator.annotate(scene=annotated, detections=detections)
    annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)

    # Step 6 — class_counters already has final counts from Step 4
    class_counts = class_counters

    return annotated, class_counts