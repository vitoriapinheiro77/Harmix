from functools import lru_cache
from pathlib import Path

from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.reportLabPen import ReportLabPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.shapes import Path as RLPath
from reportlab.lib.colors import Color
from reportlab.lib.units import cm

BRAVURA_PATH = Path(__file__).resolve().parent / 'assets' / 'fonts' / 'Bravura.otf'
G_CLEF_GLYPH = 'uniE050'
TIME_SIG_4_GLYPH = 'uniE084'
NOTATION_COLOR = Color(0.08, 0.08, 0.08)


@lru_cache(maxsize=1)
def _bravura_font():
    if not BRAVURA_PATH.exists():
        raise FileNotFoundError(f'Fonte Bravura nao encontrada: {BRAVURA_PATH}')
    return TTFont(str(BRAVURA_PATH))


def _glyph_set():
    return _bravura_font().getGlyphSet()


def _glyph_bounds(glyph_name):
    glyph_set = _glyph_set()
    bounds_pen = BoundsPen(glyph_set)
    glyph_set[glyph_name].draw(bounds_pen)
    return bounds_pen.bounds


def _draw_bravura_glyph(pdf, glyph_name, origin_x, origin_y, scale):
    glyph_set = _glyph_set()
    rl_path = RLPath()
    pen = TransformPen(
        ReportLabPen(glyph_set, rl_path),
        (scale, 0, 0, scale, origin_x, origin_y),
    )
    glyph_set[glyph_name].draw(pen)
    rl_path.fillColor = NOTATION_COLOR
    rl_path.strokeColor = None

    drawing = Drawing(1, 1)
    drawing.add(rl_path)
    renderPDF.draw(drawing, pdf, 0, 0)

    xmin, ymin, xmax, ymax = _glyph_bounds(glyph_name)
    return origin_x + (xmax * scale), (xmax - xmin) * scale


def _draw_bravura_glyph_centered(pdf, glyph_name, center_x, center_y, scale):
    xmin, ymin, xmax, ymax = _glyph_bounds(glyph_name)
    center_glyph_x = (xmin + xmax) / 2
    center_glyph_y = (ymin + ymax) / 2
    origin_x = center_x - (center_glyph_x * scale)
    origin_y = center_y - (center_glyph_y * scale)
    _draw_bravura_glyph(pdf, glyph_name, origin_x, origin_y, scale)
    return origin_x + (xmax * scale)


def draw_treble_clef(pdf, x, staff_top, line_gap):
    staff_bottom = staff_top - (4 * line_gap)
    scale = line_gap / 250.0
    origin_x = x + 0.04 * cm
    return _draw_bravura_glyph(pdf, G_CLEF_GLYPH, origin_x, staff_bottom, scale)[0]


def draw_time_signature_44(pdf, x_after_clef, staff_top, line_gap):
    staff_bottom = staff_top - (4 * line_gap)
    scale = line_gap / 250.0
    sig_x = x_after_clef + (0.04 * cm)
    digit_center_x = sig_x + (430 * scale * 0.5)

    top_center_y = staff_bottom + (875 * scale)
    bottom_center_y = staff_bottom + (375 * scale)

    right_x = _draw_bravura_glyph_centered(
        pdf, TIME_SIG_4_GLYPH, digit_center_x, top_center_y, scale
    )
    right_x = max(
        right_x,
        _draw_bravura_glyph_centered(
            pdf, TIME_SIG_4_GLYPH, digit_center_x, bottom_center_y, scale
        ),
    )
    return right_x


def draw_system_signature(pdf, x, staff_top, line_gap, include_time_signature=False):
    clef_right = draw_treble_clef(pdf, x, staff_top, line_gap)
    content_start = clef_right + (0.06 * cm)
    if include_time_signature:
        content_start = draw_time_signature_44(pdf, clef_right, staff_top, line_gap) + (0.06 * cm)
    return content_start
