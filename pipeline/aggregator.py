"""
pipeline/aggregator.py
Шаг 5: сборка результатов страницы → рендер PNG + Markdown.

Логика рендера:
- Берём all_regions (отсортированы сверху вниз по y0)
- text    → параграф текста
- formula → блок $$ latex $$
- picture → вставка оригинального кропа
- Рендерим всё на белый холст через PIL → PNG страницы
- Параллельно собираем Markdown
"""
import base64
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

from config import TARGET_WIDTH, TARGET_HEIGHT, PICTURE_EMBED
from pipeline.detector import PageDetections


# ── Константы рендера ─────────────────────────────────────────────────────────
FONT_SIZE_TEXT    = 20
FONT_SIZE_FORMULA = 18
LINE_HEIGHT       = 28
MARGIN_X          = 60
MARGIN_Y          = 60
TEXT_COLOR        = (20, 20, 20)
FORMULA_COLOR     = (10, 60, 160)
MAX_TEXT_WIDTH    = TARGET_WIDTH - MARGIN_X * 2


def aggregate_page(
    page_idx: int,
    detections: PageDetections,
    text_results:    Dict[int, str],
    formula_results: Dict[int, str],
    crop_paths:      Dict[str, List[Path]],
) -> Tuple[Image.Image, str]:
    """
    Рендерит страницу и собирает Markdown.

    Args:
        page_idx:        номер страницы (0-based)
        detections:      PageDetections с регионами
        text_results:    {region_idx: текст}    для texts
        formula_results: {region_idx: LaTeX}    для formulas
        crop_paths:      {"picture": [path...]} для картинок

    Returns:
        (PIL.Image rendered page, markdown string)
    """
    canvas = Image.new("RGB", (TARGET_WIDTH, TARGET_HEIGHT), (255, 255, 255))
    draw   = ImageDraw.Draw(canvas)

    try:
        font_text    = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",    FONT_SIZE_TEXT)
        font_formula = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE_FORMULA)
    except Exception:
        font_text    = ImageFont.load_default()
        font_formula = ImageFont.load_default()

    md_lines: List[str] = [f"\n\n---\n\n## Страница {page_idx + 1}\n"]
    cursor_y = MARGIN_Y

    # Счётчики для индексации результатов
    text_idx    = 0
    formula_idx = 0
    picture_idx = 0

    for region in detections.all_regions:

        if region.label == "text":
            content = text_results.get(text_idx, "[TEXT]")
            text_idx += 1
            cursor_y = _render_text(draw, content, cursor_y, font_text)
            md_lines.append(content + "\n")

        elif region.label == "formula":
            content = formula_results.get(formula_idx, "$$[FORMULA]$$")
            formula_idx += 1
            cursor_y = _render_formula(draw, content, cursor_y, font_formula)
            md_lines.append(f"\n{content}\n")

        elif region.label == "picture":
            paths = crop_paths.get("picture", [])
            if picture_idx < len(paths):
                img_path = paths[picture_idx]
                cursor_y = _render_picture(canvas, img_path, cursor_y)
                md_lines.append(_md_image(img_path, picture_idx))
            picture_idx += 1

        cursor_y += 10  # отступ между блоками

        if cursor_y > TARGET_HEIGHT - MARGIN_Y:
            break  # страница заполнена

    return canvas, "\n".join(md_lines)


def aggregate_document(
    rendered_pages: List[Image.Image],
    page_markdowns: List[str],
) -> Tuple[List[Image.Image], str]:
    """
    Возвращает список страниц и полный Markdown документа.
    """
    full_md = "# Оцифрованный документ\n" + "\n".join(page_markdowns)
    return rendered_pages, full_md


# ── Вспомогательные функции рендера ──────────────────────────────────────────

def _render_text(draw: ImageDraw.Draw, text: str, y: int, font) -> int:
    """Рисует текст с переносом строк. Возвращает новый y."""
    chars_per_line = MAX_TEXT_WIDTH // (FONT_SIZE_TEXT // 2 + 2)
    lines = []
    for paragraph in text.split("\n"):
        lines += textwrap.wrap(paragraph, width=chars_per_line) or [""]

    for line in lines:
        draw.text((MARGIN_X, y), line, font=font, fill=TEXT_COLOR)
        y += LINE_HEIGHT
    return y


def _render_formula(draw: ImageDraw.Draw, latex: str, y: int, font) -> int:
    """Рисует формулу как текст (LaTeX строку). Возвращает новый y."""
    # Убираем обёртку $$
    clean = latex.strip().strip("$").strip()
    draw.text((MARGIN_X, y), f"  {clean}", font=font, fill=FORMULA_COLOR)
    return y + LINE_HEIGHT + 4


def _render_picture(canvas: Image.Image, img_path: Path, y: int) -> int:
    """Вставляет кроп картинки на холст. Возвращает новый y."""
    try:
        pic = Image.open(img_path).convert("RGB")
        max_w = TARGET_WIDTH - MARGIN_X * 2
        if pic.width > max_w:
            ratio = max_w / pic.width
            pic = pic.resize((max_w, int(pic.height * ratio)), Image.LANCZOS)
        canvas.paste(pic, (MARGIN_X, y))
        return y + pic.height
    except Exception:
        return y + 10


def _md_image(img_path: Path, idx: int) -> str:
    """Возвращает Markdown строку с изображением."""
    if PICTURE_EMBED:
        data = base64.b64encode(img_path.read_bytes()).decode()
        return f"\n![picture_{idx}](data:image/png;base64,{data})\n"
    return f"\n![picture_{idx}]({img_path})\n"
