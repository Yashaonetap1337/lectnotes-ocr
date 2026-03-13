"""
config.py — все настройки проекта в одном месте.
"""
from pathlib import Path

# ── Пути ──────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "models"
YOLO_MODEL = MODELS_DIR / "best.pt"

TMP_DIR    = ROOT_DIR / "tmp"
CROPS_DIR  = TMP_DIR / "crops"

# ── PDF → Image ────────────────────────────────────────────────────────────────
PDF_DPI       = 300
TARGET_WIDTH  = 1240
TARGET_HEIGHT = 1754

# ── YOLO ──────────────────────────────────────────────────────────────────────
YOLO_IMG_SIZE = 1280
YOLO_CONF     = 0.25

# Должно совпадать с classes.txt из разметки
CLASS_NAMES   = ["formula", "picture", "text"]

# ── OCR (заглушки — заполнить когда модели будут выбраны) ─────────────────────
SURYA_LANGS      = ["ru"]
GOT_OCR_MODEL_ID = "ucaslcl/GOT-OCR2_0"
GOT_OCR_MODE     = "format"

# ── Агрегация / рендер ────────────────────────────────────────────────────────
PICTURE_EMBED = True    # True = base64 в Markdown, False = ссылка на файл