from pathlib import Path
from textwrap import wrap
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont

from app.config import get_settings
from app.models import MeetingSummary

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except Exception:  # pragma: no cover - app still works without optional shaping deps
    arabic_reshaper = None
    get_display = None


WIDTH = 1080
HEIGHT = 1080
MARGIN = 90


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf' if bold else '/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf',
        '/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf' if bold else '/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf',
        '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
        '/System/Library/Fonts/Supplemental/Arial.ttf',
        'C:/Windows/Fonts/arial.ttf',
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _looks_arabic(text: str) -> bool:
    return any('\u0600' <= ch <= '\u06ff' or '\u0750' <= ch <= '\u077f' or '\u08a0' <= ch <= '\u08ff' for ch in text)


def _is_rtl_summary(summary: MeetingSummary) -> bool:
    lang = (summary.language or '').lower()
    return 'arabic' in lang or lang == 'ar' or _looks_arabic(summary.title + summary.short_summary)


def _shape_arabic(text: str) -> str:
    if not text:
        return text
    if arabic_reshaper and get_display and _looks_arabic(text):
        return get_display(arabic_reshaper.reshape(text))
    return text


def _text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def _draw_text(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, font, fill, rtl: bool = False, right_x: int | None = None) -> None:
    rendered = _shape_arabic(text) if rtl else text
    if rtl and right_x is not None:
        x = right_x - _text_width(draw, rendered, font)
    draw.text((x, y), rendered, font=font, fill=fill)


def _draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    font,
    fill,
    max_chars: int,
    line_gap: int,
    rtl: bool = False,
    right_x: int | None = None,
) -> int:
    for line in wrap(text, width=max_chars):
        _draw_text(draw, line, x, y, font, fill, rtl=rtl, right_x=right_x)
        y += font.size + line_gap if hasattr(font, 'size') else 32
    return y


def create_instagram_recap_image(summary: MeetingSummary) -> str:
    settings = get_settings()
    filename = f'recap-{uuid4().hex}.png'
    output_path = settings.generated_path / filename

    rtl = _is_rtl_summary(summary)

    bg = (248, 248, 248)
    ink = (25, 25, 25)
    muted = (90, 90, 90)
    accent = (30, 30, 30)

    img = Image.new('RGB', (WIDTH, HEIGHT), bg)
    draw = ImageDraw.Draw(img)

    title_font = _font(58, bold=True)
    heading_font = _font(36, bold=True)
    body_font = _font(34)
    small_font = _font(24)

    x = MARGIN
    right_x = WIDTH - MARGIN
    y = MARGIN

    eyebrow = 'ملخص الاجتماع' if rtl else 'Meeting Recap'
    heading = 'النقاط الرئيسية' if rtl else 'Key points'
    footer = 'مسودة مولدة — يرجى المراجعة قبل النشر' if rtl else 'Generated draft — review before publishing'

    _draw_text(draw, eyebrow, x, y, small_font, muted, rtl=rtl, right_x=right_x)
    y += 50
    y = _draw_wrapped(draw, summary.title[:120], x, y, title_font, ink, 24, 10, rtl=rtl, right_x=right_x)
    y += 35

    draw.line((MARGIN, y, WIDTH - MARGIN, y), fill=accent, width=4)
    y += 45
    _draw_text(draw, heading, x, y, heading_font, ink, rtl=rtl, right_x=right_x)
    y += 55

    points = summary.key_points[:4] or [summary.short_summary]
    for idx, point in enumerate(points, start=1):
        if rtl:
            y = _draw_wrapped(draw, f'{idx}. {point[:180]}', x, y, body_font, ink, 34, 8, rtl=True, right_x=right_x)
        else:
            _draw_text(draw, f'{idx}.', x, y, body_font, ink)
            y = _draw_wrapped(draw, point[:180], x + 58, y, body_font, ink, 34, 8)
        y += 20
        if y > HEIGHT - 170:
            break

    _draw_text(draw, footer, x, HEIGHT - 90, small_font, muted, rtl=rtl, right_x=right_x)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, quality=95)
    return str(output_path)


def public_url_for_generated_file(path: str | Path) -> str:
    settings = get_settings()
    if not settings.public_base_url:
        raise RuntimeError('PUBLIC_BASE_URL is required for Instagram publishing because Meta must fetch the image from a public HTTPS URL.')
    name = Path(path).name
    return f'{settings.public_base_url.rstrip("/")}/storage/generated/{name}'
