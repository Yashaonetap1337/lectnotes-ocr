"""
pipeline/crop_regions.py
Шаг 3: вырезка bbox регионов из страницы → сохранение в папки по классам.

Структура на выходе:
    tmp/crops/page_001/text/region_00.png
    tmp/crops/page_001/formula/region_00.png
    tmp/crops/page_001/picture/region_00.png
"""
import shutil
from pathlib import Path
from typing import Dict, List

from PIL import Image

from config import CROPS_DIR
from pipeline.detector import PageDetections


def crop_page_regions(
    image: Image.Image,
    detections: PageDetections,
    padding: int = 6,
) -> Dict[str, List[Path]]:
    """
    Вырезает регионы из страницы и сохраняет по папкам.

    Returns:
        {"text": [path, ...], "formula": [...], "picture": [...]}
        Порядок путей совпадает с порядком регионов в каждом классе.
    """
    W, H = image.size
    page_dir = CROPS_DIR / f"page_{detections.page_idx:03d}"

    result: Dict[str, List[Path]] = {"text": [], "formula": [], "picture": []}

    groups = {
        "text":    detections.texts,
        "formula": detections.formulas,
        "picture": detections.pictures,
    }

    for label, regions in groups.items():
        label_dir = page_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        for idx, region in enumerate(regions):
            x0, y0, x1, y1 = region.bbox
            # Добавляем паддинг, не выходя за границы
            x0 = max(0, x0 - padding)
            y0 = max(0, y0 - padding)
            x1 = min(W, x1 + padding)
            y1 = min(H, y1 + padding)

            crop = image.crop((x0, y0, x1, y1))
            out_path = label_dir / f"region_{idx:02d}.png"
            crop.save(out_path, "PNG")
            result[label].append(out_path)

    return result


def cleanup_crops(page_idx: int) -> None:
    """Удаляет временные кропы страницы после обработки."""
    page_dir = CROPS_DIR / f"page_{page_idx:03d}"
    shutil.rmtree(page_dir, ignore_errors=True)
