"""
pipeline/text_ocr.py
Шаг 4a: заглушка для распознавания текста.
Модель будет выбрана и реализована позже.
"""
from pathlib import Path
from typing import List


def load_text_model():
    """Заглушка. Вернёт None пока модель не выбрана."""
    return None


def recognize_text_regions(
    model,
    crop_paths: List[Path],
) -> List[str]:
    """
    Заглушка: возвращает placeholder для каждого текстового региона.
    Заменить на реальный OCR когда модель будет выбрана.
    """
    return [f"[TEXT_REGION_{i}]" for i in range(len(crop_paths))]
