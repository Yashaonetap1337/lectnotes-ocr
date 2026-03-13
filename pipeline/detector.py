"""
pipeline/detector.py
Шаг 2: YOLO детектор регионов на странице.
"""
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from PIL import Image
from ultralytics import YOLO

from config import YOLO_MODEL, YOLO_IMG_SIZE, YOLO_CONF, CLASS_NAMES


@dataclass
class DetectedRegion:
    label:      str
    bbox:       tuple
    confidence: float
    page_idx:   int


@dataclass
class PageDetections:
    page_idx: int
    texts:    List[DetectedRegion] = field(default_factory=list)
    formulas: List[DetectedRegion] = field(default_factory=list)
    pictures: List[DetectedRegion] = field(default_factory=list)

    @property
    def all_regions(self) -> List[DetectedRegion]:
        """Все регионы отсортированные сверху вниз (по y0)."""
        all_r = self.texts + self.formulas + self.pictures
        return sorted(all_r, key=lambda r: r.bbox[1])


def load_detector() -> YOLO:
    """Загружает YOLO модель. Вызывается один раз при старте."""
    return YOLO(str(YOLO_MODEL))


def detect_page(model: YOLO, image: Image.Image, page_idx: int) -> PageDetections:
    """Запускает детекцию на одной странице."""
    detections = PageDetections(page_idx=page_idx)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name
    image.save(tmp_path)

    pred = model.predict(tmp_path, imgsz=YOLO_IMG_SIZE, conf=YOLO_CONF, verbose=False)[0]
    Path(tmp_path).unlink(missing_ok=True)

    for box in pred.boxes:
        cls   = int(box.cls[0])
        conf  = float(box.conf[0])
        bbox  = tuple(map(int, box.xyxy[0].tolist()))
        label = CLASS_NAMES[cls]

        region = DetectedRegion(
            label=label, bbox=bbox, confidence=conf, page_idx=page_idx
        )

        if label == "text":
            detections.texts.append(region)
        elif label == "formula":
            detections.formulas.append(region)
        elif label == "picture":
            detections.pictures.append(region)

    return detections
