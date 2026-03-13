"""
pipeline/formula_ocr.py
Шаг 4b: заглушка для распознавания формул.
Модель будет подключена позже (GOT-OCR 2.0 или другая).
"""
from pathlib import Path
from typing import List


def load_formula_model():
    """Заглушка. Вернёт None пока модель не подключена."""
    return None


def recognize_formula_regions(
    model,
    crop_paths: List[Path],
) -> List[str]:
    """
    Заглушка: возвращает placeholder LaTeX для каждого формульного региона.
    Заменить на реальный OCR когда модель будет выбрана.
    """
    return [f"$$[FORMULA_{i}]$$" for i in range(len(crop_paths))]
