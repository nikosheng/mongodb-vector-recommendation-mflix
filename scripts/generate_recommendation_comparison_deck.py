"""
Generates an editable PPTX deck comparing three recommendation approaches:
1. YouTube's Gemini-based Large Recommender Model (LRM) + Semantic IDs
2. StreamFlix's real-time Rocchio vector approach (+ proposed region factor)
3. Traditional batch-trained ML recommenders
...and a closing comparison slide promoting the StreamFlix approach.

Run:  python3 scripts/generate_recommendation_comparison_deck.py
Output: docs/recommendation-approaches-comparison.pptx
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.oxml.ns import qn
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
import copy

# ---------------------------------------------------------------------------
# Palette \u2014 matched to the "Melco Capacity Review" / MongoDB brand theme
# (dk1 #023430, dk2 #00ED64, lt2 #00684A, accent2 #E3FCF7, accent3 #F9EBFF,
#  accent4 #E9FF99, accent5 #00D2FF, accent6 #006EFF)
# ---------------------------------------------------------------------------
DARK        = RGBColor(0x02, 0x34, 0x30)   # theme dk1 \u2014 primary text / headings
GRAY        = RGBColor(0x4B, 0x5D, 0x58)   # muted secondary text (dark-green tinted)
LIGHT_BG    = RGBColor(0xE3, 0xFC, 0xF7)   # theme accent2 \u2014 default light card bg
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)

MONGO_GREEN = RGBColor(0x00, 0xED, 0x64)   # theme dk2 \u2014 MongoDB signature green (accents/pills)
MONGO_TEAL  = RGBColor(0x00, 0x68, 0x4A)   # theme lt2 \u2014 solid banner fill (readable w/ white text)
MONGO_LILAC_BG = RGBColor(0xF9, 0xEB, 0xFF)  # theme accent3
MONGO_LIME_BG  = RGBColor(0xE9, 0xFF, 0x99)  # theme accent4
MONGO_CYAN  = RGBColor(0x00, 0xD2, 0xFF)   # theme accent5
MONGO_BLUE  = RGBColor(0x00, 0x6E, 0xFF)   # theme accent6

# "External" (YouTube) accent \u2014 mapped onto theme's blue accent6
YT_RED      = MONGO_BLUE
YT_RED_BG   = RGBColor(0xE3, 0xEF, 0xFF)

# "Our solution" (StreamFlix) accent \u2014 mapped onto theme teal/mint
SF_TEAL     = MONGO_TEAL
SF_TEAL_BG  = LIGHT_BG

# Semantic (kept distinct from brand palette on purpose \u2014 universal meaning)
PLAY_GREEN  = MONGO_TEAL
PLAY_BG     = LIGHT_BG
SKIP_RED    = RGBColor(0xC6, 0x28, 0x28)
SKIP_BG     = RGBColor(0xFD, 0xEC, 0xEA)
REGION_PURP = RGBColor(0x6A, 0x3F, 0xA0)
REGION_BG   = MONGO_LILAC_BG

ML_ORANGE   = RGBColor(0xE6, 0x7E, 0x22)
ML_ORANGE_BG= RGBColor(0xFC, 0xF0, 0xE3)

# Business controls (promoted/sponsored content) \u2014 distinct gold accent
PROMO_GOLD  = RGBColor(0xB8, 0x86, 0x00)
PROMO_BG    = RGBColor(0xFD, 0xF1, 0xD6)

FONT_TITLE  = "Lexend Deca Medium"
FONT_BODY   = "Lexend Deca"
FONT_LIGHT  = "Lexend Deca Light"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
BLANK = prs.slide_layouts[6]


def contrast_text_color(rgb):
    """Return DARK or WHITE depending on which reads better on the given fill color."""
    hexval = str(rgb)
    r, g, b = int(hexval[0:2], 16), int(hexval[2:4], 16), int(hexval[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b)
    return DARK if luminance > 150 else WHITE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def add_slide():
    return prs.slides.add_slide(BLANK)


def set_fill(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def add_title(slide, text, subtitle=None, color=DARK, accent=SF_TEAL, tag=None):
    tag_h = Inches(0.36)
    if tag:
        pill = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.4), Inches(0.25), Inches(0.28) + Inches(0.12) * len(tag), tag_h)
        pill.fill.solid()
        pill.fill.fore_color.rgb = accent
        pill.line.fill.background()
        pill.shadow.inherit = False
        tf = pill.text_frame
        tf.word_wrap = False
        tf.margin_left = Inches(0.12)
        tf.margin_right = Inches(0.12)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.text = tag.upper()
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.name = FONT_TITLE
        p.font.color.rgb = contrast_text_color(accent)
        title_top = Inches(0.72)
    else:
        title_top = Inches(0.3)

    tb = slide.shapes.add_textbox(Inches(0.4), title_top, Inches(12.6), Inches(0.7))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(28)
    p.font.bold = False
    p.font.color.rgb = color
    p.font.name = FONT_TITLE

    if subtitle:
        sub_top = title_top + Inches(0.56)
        tb2 = slide.shapes.add_textbox(Inches(0.42), sub_top, Inches(12.4), Inches(0.4))
        tf2 = tb2.text_frame
        tf2.word_wrap = True
        p2 = tf2.paragraphs[0]
        p2.text = subtitle
        p2.font.size = Pt(14)
        p2.font.name = FONT_LIGHT
        p2.font.color.rgb = GRAY

    line_y = title_top + Inches(0.98 if subtitle else 0.6)
    ln = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.4), line_y, Inches(12.5), Pt(2))
    set_fill(ln, RGBColor(0xDC, 0xEB, 0xE6))
    return line_y


def add_bullets(slide, x, y, w, h, items, font_size=16, color=DARK, bullet_color=None, line_spacing=1.15):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if isinstance(item, tuple):
            text, level = item
        else:
            text, level = item, 0
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.level = level
        p.font.size = Pt(font_size - (2 if level else 0))
        p.font.color.rgb = color
        p.font.name = FONT_BODY
        p.line_spacing = line_spacing
        p.space_after = Pt(8)
    return tb


def add_box(slide, x, y, w, h, fill=LIGHT_BG, line_color=None, radius=True):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    box = slide.shapes.add_shape(shape_type, x, y, w, h)
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    if line_color:
        box.line.color.rgb = line_color
        box.line.width = Pt(1.25)
    else:
        box.line.fill.background()
    box.shadow.inherit = False
    return box


def box_with_text(slide, x, y, w, h, title, body, fill, title_color, body_color=DARK,
                   title_size=14, body_size=12, dashed=False, line_color=None):
    box = add_box(slide, x, y, w, h, fill=fill, line_color=line_color)
    if dashed:
        ln = box.line
        ln.dash_style = MSO_LINE_DASH_STYLE.DASH
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.15)
    tf.margin_right = Inches(0.15)
    tf.margin_top = Inches(0.08)
    tf.margin_bottom = Inches(0.08)
    p0 = tf.paragraphs[0]
    p0.text = title
    p0.font.bold = True
    p0.font.size = Pt(title_size)
    p0.font.color.rgb = title_color
    p0.font.name = FONT_TITLE
    for line in body:
        p = tf.add_paragraph()
        p.text = line
        p.font.size = Pt(body_size)
        p.font.color.rgb = body_color
        p.font.name = FONT_BODY
        p.space_before = Pt(2)
    return box


def add_footnote(slide, text):
    tb = slide.shapes.add_textbox(Inches(0.4), Inches(7.05), Inches(12.5), Inches(0.35))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(11)
    p.font.italic = True
    p.font.name = FONT_LIGHT
    p.font.color.rgb = GRAY


def add_callout_banner(slide, x, y, w, h, text, fill, text_color=WHITE, size=14):
    box = add_box(slide, x, y, w, h, fill=fill)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = True
    p.font.name = FONT_TITLE
    p.font.color.rgb = text_color


def add_down_arrow(slide, x, y, w=Inches(0.4), h=Inches(0.3), color=GRAY):
    arr = slide.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, x, y, w, h)
    set_fill(arr, color)


def add_right_arrow(slide, x, y, w=Inches(0.5), h=Inches(0.35), color=GRAY):
    arr = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x, y, w, h)
    set_fill(arr, color)


def add_line(slide, x1, y1, x2, y2, color=GRAY, width=1.5):
    conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    conn.line.color.rgb = color
    conn.line.width = Pt(width)


def add_vector_arrow(slide, origin_x, origin_y, length, angle_deg, thickness, color,
                      dashed=False, outline_only=False):
    """Draws a RIGHT_ARROW shape pivoted around its own left-center point (origin),
    rotated to angle_deg (clockwise-positive, matching PowerPoint rotation convention),
    to approximate a 2D vector radiating from a shared origin point."""
    w = length
    h = thickness
    # Shape is created with its bounding box anchored at (origin_x, origin_y - h/2);
    # PowerPoint rotates shapes around their own center, so to pivot around the LEFT edge
    # (the origin point) we offset the center back after rotation using basic trig.
    import math
    rad = math.radians(angle_deg)
    cx = origin_x + (w / 2) * math.cos(rad)
    cy = origin_y + (w / 2) * math.sin(rad)
    left = Emu(int(cx - w / 2))
    top = Emu(int(cy - h / 2))
    arr = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, left, top, w, h)
    arr.rotation = angle_deg
    if outline_only:
        arr.fill.background()
        arr.line.color.rgb = color
        arr.line.width = Pt(1.75)
    else:
        arr.fill.solid()
        arr.fill.fore_color.rgb = color
        arr.line.fill.background()
    if dashed:
        arr.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    arr.shadow.inherit = False
    return arr


def rich_equation_box(slide, x, y, w, h, segments, size=34, fill=DARK, box_fill=DARK):
    """segments: list of (text, RGBColor or None-for-default) rendered on one centered line."""
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    box.fill.solid()
    box.fill.fore_color.rgb = box_fill
    box.line.color.rgb = MONGO_GREEN
    box.line.width = Pt(1.5)
    box.shadow.inherit = False
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    for text, color in segments:
        run = p.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = True
        run.font.name = "Consolas"
        run.font.color.rgb = color if color else WHITE
    return box


def label_under_equation(slide, x, y, w, h, label, color):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.text = label
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.name = FONT_TITLE
    p.font.color.rgb = color


# ===========================================================================
# SLIDE 0 — Cover
# ===========================================================================
s = add_slide()

# subtle decorative panel (behind everything)
panel = add_box(s, Inches(8.6), Inches(-0.8), Inches(6.0), Inches(9.0), fill=LIGHT_BG, radius=True)
panel.rotation = 8

pill = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.7), Inches(1.1), Inches(3.6), Inches(0.42))
pill.fill.solid()
pill.fill.fore_color.rgb = MONGO_GREEN
pill.line.fill.background()
pill.shadow.inherit = False
tf = pill.text_frame
tf.vertical_anchor = MSO_ANCHOR.MIDDLE
p = tf.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
p.text = "MONGODB ATLAS \u00b7 STREAMFLIX"
p.font.size = Pt(12)
p.font.bold = True
p.font.name = FONT_TITLE
p.font.color.rgb = contrast_text_color(MONGO_GREEN)

title_tb = s.shapes.add_textbox(Inches(0.7), Inches(1.75), Inches(9.5), Inches(2.2))
tf = title_tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Recommendation Intelligence"
p.font.size = Pt(42)
p.font.name = FONT_TITLE
p.font.color.rgb = DARK
p2 = tf.add_paragraph()
p2.text = "YouTube vs. Traditional ML vs. StreamFlix"
p2.font.size = Pt(42)
p2.font.name = FONT_TITLE
p2.font.color.rgb = MONGO_TEAL

sub_tb = s.shapes.add_textbox(Inches(0.72), Inches(3.95), Inches(9.2), Inches(0.8))
tf = sub_tb.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "A comparative deep-dive on personalization architectures \u2014 cost, speed, and real-time adaptability"
p.font.size = Pt(17)
p.font.name = FONT_LIGHT
p.font.color.rgb = GRAY

# byline pill row
bylines = ["Large Recommender Models", "Real-Time Rocchio Vectors", "Batch-Trained ML"]
bx = Inches(0.72)
for b in bylines:
    bw = Inches(0.3) + Inches(0.095) * len(b)
    bpill = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, bx, Inches(4.9), bw, Inches(0.4))
    bpill.fill.solid()
    bpill.fill.fore_color.rgb = MONGO_LILAC_BG
    bpill.line.color.rgb = REGION_PURP
    bpill.line.width = Pt(0.75)
    bpill.shadow.inherit = False
    tf = bpill.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.text = b
    p.font.size = Pt(11)
    p.font.name = FONT_BODY
    p.font.color.rgb = REGION_PURP
    bx += bw + Inches(0.2)

bottom_bar = add_box(s, Inches(0), Inches(7.38), Inches(13.333), Inches(0.12), fill=MONGO_GREEN, radius=False)


# ===========================================================================
# SLIDE 1 — YouTube: Teaching Gemini to Speak YouTube (end-to-end diagram)
# ===========================================================================
s = add_slide()
add_title(s, "YouTube's Approach: Teaching Gemini to Speak YouTube",
          "End-to-end pipeline: from raw video/user data to on-screen recommendations for 2B+ users",
          accent=YT_RED, tag="YouTube")

# --- Row 1: Build the model + vocabulary -----------------------------------
row1 = [
    ("Video Corpus +\nUser Signals", "Titles, transcripts, frames + demographics, watch history", RGBColor(0x42, 0xA5, 0xF5)),
    ("Gemini \u2192 LRM", "Fine-tuned on YouTube-scale video/creator/watch data", RGBColor(0x66, 0xBB, 0x6A)),
    ("Semantic ID (SID)\nVocabulary", "Feature Extraction \u2192 Embedding \u2192 Quantization (RQVE)", RGBColor(0xFF, 0xCA, 0x28)),
    ("Bilingual\nTraining", "Text \u2194 SID mapping + Watch-Sequence training", RGBColor(0xFF, 0x70, 0x43)),
]

# --- Row 2: Serve recommendations -------------------------------------------
row2 = [
    ("Generative\nPrompt", "Demographics + current watch + watch history", RGBColor(0xAB, 0x47, 0xBC)),
    ("LRM Decodes\nSIDs", "Generates a personalized candidate SID list", RGBColor(0xEC, 0x40, 0x7A)),
    ("Offline\nPrecompute", "Strips personal data \u2192 cached tables (95% cost saved)", RGBColor(0x5C, 0x6B, 0xC0)),
    ("Served to Users", "Home \u00b7 WatchNext \u00b7 Shorts \u00b7 Search \u2014 2B+ users", YT_RED),
]

row_y = [Inches(1.85), Inches(3.65)]
row_h = Inches(1.15)
start_x = Inches(0.5)
gap_w = Inches(0.22)
box_w = Inches(2.83)

for row_idx, row in enumerate([row1, row2]):
    y = row_y[row_idx]
    for i, (title, desc, color) in enumerate(row):
        x = start_x + i * (box_w + gap_w)
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, box_w, row_h)
        box.fill.solid()
        box.fill.fore_color.rgb = color
        box.line.fill.background()
        box.shadow.inherit = False
        tf = box.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = Inches(0.08)
        tf.margin_right = Inches(0.08)
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        p.text = title
        p.font.bold = True
        p.font.size = Pt(13.5)
        p.font.color.rgb = WHITE
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        p2.text = desc
        p2.font.size = Pt(9.5)
        p2.font.color.rgb = WHITE
        p2.space_before = Pt(3)
        if i < len(row) - 1:
            add_right_arrow(s, x + box_w + Inches(0.02), y + row_h/2 - Inches(0.11), w=Inches(0.18), h=Inches(0.22), color=GRAY)

# connector arrow from end of row 1 down into start of row 2
mid_x = start_x + 1.5 * (box_w + gap_w) + box_w / 2
add_down_arrow(s, mid_x - Inches(0.15), row_y[0] + row_h + Inches(0.05), w=Inches(0.3), h=Inches(0.5), color=YT_RED)
label_under_equation(s, mid_x - Inches(1.6), row_y[0] + row_h + Inches(0.02), Inches(3.4), Inches(0.3),
                      "at request / serving time", GRAY)

# --- Challenge tags ----------------------------------------------------------
tags = ["20B+ videos", "Freshness: minutes", "Retrain: hrs/days", "\u201cFlash\u201d for latency"]
tag_y = row_y[1] + row_h + Inches(0.22)
tag_w = Inches(2.83)
for i, tag in enumerate(tags):
    x = start_x + i * (box_w + gap_w)
    pill = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, tag_y, tag_w, Inches(0.4))
    pill.fill.solid()
    pill.fill.fore_color.rgb = YT_RED_BG
    pill.line.color.rgb = YT_RED
    pill.line.width = Pt(1)
    pill.shadow.inherit = False
    tf = pill.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.text = tag
    p.font.size = Pt(11.5)
    p.font.bold = True
    p.font.color.rgb = YT_RED

add_callout_banner(s, Inches(0.5), tag_y + Inches(0.6), Inches(12.4), Inches(0.6),
                    "Massive infrastructure investment \u2014 LLM fine-tuning + continuous retraining + huge serving infra "
                    "\u2014 only economical at YouTube's scale",
                    fill=YT_RED)
add_footnote(s, "Source: \u201cTeaching Gemini to Speak YouTube\u201d (AI Engineer, Devansh Tandon), distilled by Mohamed Elrefaey, Medium, Jul 2025")


# ===========================================================================
# SLIDE 2 — Semantic ID Funnel
# ===========================================================================
s = add_slide()
add_title(s, "Giving Videos a Language: Introducing Semantic ID (SID)",
           "LLMs understand text. Videos are not text \u2014 SID turns every video into a token the model can \u201cread\u201d",
           accent=YT_RED, tag="YouTube")

# Funnel: 4 trapezoid stages narrowing top-to-bottom
funnel_labels = [
    ("Video Data", "Titles, descriptions, transcripts, audio, frames", RGBColor(0x42, 0xA5, 0xF5)),
    ("① Feature Extraction", "Pulls relevant signal from every video element", RGBColor(0x66, 0xBB, 0x6A)),
    ("② Embedding", "Converts extracted features into multi-dimensional vectors", RGBColor(0xFF, 0xCA, 0x28)),
    ("③ Quantization (RQVE)", "Maps vectors to a discrete set of tokens", RGBColor(0xFF, 0x70, 0x43)),
    ("Semantic Tokens", "The \u201csemantic vocabulary\u201d of YouTube content", RGBColor(0xEC, 0x40, 0x7A)),
]

top_w = Inches(6.6)
bottom_w = Inches(3.2)
stage_h = Inches(0.72)
gap = Inches(0.14)
start_y = Inches(1.75)
center_x = Inches(3.55)

n = len(funnel_labels)
for i, (title, desc, color) in enumerate(funnel_labels):
    w = top_w - (top_w - bottom_w) * (i / (n - 1))
    x = center_x + (top_w - w) / 2
    y = start_y + i * (stage_h + gap)
    shp = s.shapes.add_shape(MSO_SHAPE.TRAPEZOID, x, y, w, stage_h)
    shp.rotation = 180
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    p.text = title
    p.font.bold = True
    p.font.size = Pt(15)
    p.font.color.rgb = WHITE

    # description to the right
    desc_tb = s.shapes.add_textbox(Inches(8.55), y - Inches(0.03), Inches(4.3), stage_h + Inches(0.15))
    dtf = desc_tb.text_frame
    dtf.word_wrap = True
    dtf.vertical_anchor = MSO_ANCHOR.MIDDLE
    dp = dtf.paragraphs[0]
    dp.text = desc
    dp.font.size = Pt(12.5)
    dp.font.color.rgb = GRAY

analogy_y = start_y + n * (stage_h + gap) + Inches(0.05)
add_callout_banner(
    s, Inches(0.5), analogy_y, Inches(12.4), Inches(0.72),
    "Just as words form the vocabulary of language, Semantic IDs form the vocabulary of YouTube video content \u2014 "
    "now LLMs can reason about videos just like they reason about text.",
    fill=DARK, size=14
)


# ===========================================================================
# SLIDE 2.5 — Semantic IDs, StreamFlix's Way (business-story version)
# ===========================================================================
s = add_slide()
add_title(s, "Semantic IDs: A Fingerprint for Every Movie \u2014 and Every You", accent=SF_TEAL, tag="StreamFlix")

col_w = Inches(4.6)
left_x = Inches(0.9)
right_x = Inches(7.85)
col_label_y = Inches(1.35)

label_under_equation(s, left_x, col_label_y, col_w, Inches(0.3), "MOVIES", GRAY)
label_under_equation(s, right_x, col_label_y, col_w, Inches(0.3), "YOU", GRAY)

step1_y = Inches(1.68)
step_h = Inches(0.55)
box_with_text(s, left_x, step1_y, col_w, step_h, "\U0001F3AC  Plot + Genres", [],
              fill=LIGHT_BG, title_color=DARK, title_size=15)
box_with_text(s, right_x, step1_y, col_w, step_h, "\U0001F464  What You Watch", [],
              fill=LIGHT_BG, title_color=DARK, title_size=15)

arrow_l_x = left_x + col_w//2 - Inches(0.15)
arrow_r_x = right_x + col_w//2 - Inches(0.15)
add_down_arrow(s, arrow_l_x, step1_y + step_h + Inches(0.06), w=Inches(0.28), h=Inches(0.22), color=SF_TEAL)
add_down_arrow(s, arrow_r_x, step1_y + step_h + Inches(0.06), w=Inches(0.28), h=Inches(0.22), color=SF_TEAL)

step2_y = step1_y + step_h + Inches(0.36)
step2_h = Inches(0.85)
box_with_text(s, left_x, step2_y, col_w, step2_h, "Semantic ID",
              ["A fingerprint of the story \u2014 created once per movie"],
              fill=SF_TEAL_BG, title_color=SF_TEAL, body_size=12, title_size=15.5, line_color=SF_TEAL)
box_with_text(s, right_x, step2_y, col_w, step2_h, "Your Semantic ID",
              ["A fingerprint of your taste \u2014 created the moment you start watching"],
              fill=SF_TEAL_BG, title_color=SF_TEAL, body_size=12, title_size=15.5, line_color=SF_TEAL)

# Right column: Play/Skip loop reshaping the user's Semantic ID
loop_y = step2_y + step2_h + Inches(0.14)
add_down_arrow(s, arrow_r_x, loop_y, w=Inches(0.28), h=Inches(0.2), color=SF_TEAL)
loop_label_y = loop_y + Inches(0.24)
loop_lines = [
    "\u25b6 Play \u2192 pulls it closer",
    "\u23ed Skip \u2192 pushes it away",
]
add_bullets(s, right_x + Inches(0.35), loop_label_y, col_w - Inches(0.5), Inches(0.5), loop_lines,
            font_size=12.5, color=DARK, line_spacing=1.0)
updates_live_y = loop_label_y + Inches(0.52)
label_under_equation(s, right_x, updates_live_y, col_w, Inches(0.22), "updates live", GRAY)

step3_y = updates_live_y + Inches(0.28)
step3_h = Inches(0.5)
box_with_text(s, right_x, step3_y, col_w, step3_h, "Always-Current Semantic ID", [],
              fill=SF_TEAL_BG, title_color=SF_TEAL, title_size=13.5, line_color=SF_TEAL)

# Left column: static Semantic ID sits until convergence point
left_hold_y = step3_y
box_with_text(s, left_x, left_hold_y, col_w, step3_h, "Stored in the Movie Catalog", [],
              fill=LIGHT_BG, title_color=DARK, title_size=13.5)

# Convergence lines into Atlas search box
conv_y = step3_y + step3_h + Inches(0.1)
search_w = Inches(7.6)
search_x = Emu(int((Inches(13.333) - search_w) / 2))
search_y = conv_y + Inches(0.3)
add_line(s, left_x + col_w//2, conv_y, search_x + search_w//2, search_y, color=SF_TEAL, width=1.5)
add_line(s, right_x + col_w//2, conv_y, search_x + search_w//2, search_y, color=SF_TEAL, width=1.5)

search_h = Inches(0.75)
box_with_text(s, search_x, search_y, search_w, search_h, "\U0001F50D  MongoDB Atlas Vector Search",
              ["Matches your Semantic ID to the closest movies \u2014 instantly, every time it changes"],
              fill=DARK, title_color=WHITE, body_color=RGBColor(0xCF, 0xEA, 0xE1), body_size=12.5, title_size=16)

add_down_arrow(s, Emu(int((Inches(13.333) - Inches(0.28)) / 2)), search_y + search_h + Inches(0.06),
               w=Inches(0.28), h=Inches(0.2), color=SF_TEAL)

add_callout_banner(
    s, Inches(1.5), search_y + search_h + Inches(0.32), Inches(10.3), Inches(0.5),
    "Your next recommendations",
    fill=SF_TEAL, size=15
)


# ===========================================================================
# SLIDE 3 — ELI5: Your Taste as a Compass (intuition before the math)
# ===========================================================================
s = add_slide()
add_title(s, "Understanding the Idea: Your Taste as a Compass",
          "Before the math \u2014 the intuition behind real-time personalization",
          accent=SF_TEAL, tag="StreamFlix")

analogy = [
    "Think of your taste profile as an arrow pointing in a direction. Every movie you WATCH pulls that "
    "arrow a little closer to it. Every movie you SKIP pushes it a little away. After enough pulls and "
    "pushes, the arrow points exactly where your next favorite movie is.",
]
add_bullets(s, Inches(0.6), Inches(1.85), Inches(12.1), Inches(0.85), analogy, font_size=15.5, color=DARK, line_spacing=1.2)

# --- Angled-arrow vector diagram --------------------------------------------
# Diagram is boxed between the analogy text (ends ~2.70) and the formula card (starts 5.55)
origin_x, origin_y = Inches(2.0), Inches(4.05)

# origin dot
dot = s.shapes.add_shape(MSO_SHAPE.OVAL, origin_x - Inches(0.06), origin_y - Inches(0.06), Inches(0.12), Inches(0.12))
set_fill(dot, DARK)

# "Before" vector — neutral gray, dashed, pointing right
add_vector_arrow(s, origin_x, origin_y, Inches(3.6), 0, Inches(0.16), RGBColor(0xAA, 0xAA, 0xAA), dashed=True)
label_under_equation(s, Inches(4.6), Inches(4.22), Inches(1.6), Inches(0.3), "Before", GRAY)

# "Play" vector — green, tilts up-right (tip \u2248 (3.88, 3.36))
add_vector_arrow(s, origin_x, origin_y, Inches(2.0), -20, Inches(0.14), PLAY_GREEN)
label_under_equation(s, Inches(2.85), Inches(2.92), Inches(2.6), Inches(0.3), "\u25b6 Play (attracts)", PLAY_GREEN)

# "Skip" vector — red, tilts down-right (tip \u2248 (3.88, 4.74))
add_vector_arrow(s, origin_x, origin_y, Inches(2.0), 20, Inches(0.14), SKIP_RED)
label_under_equation(s, Inches(2.85), Inches(4.85), Inches(2.6), Inches(0.3), "\u23ed Skip (repels)", SKIP_RED)

# "After" vector — bold brand teal, net result, tilted toward Play (tip \u2248 (7.77, 3.44))
add_vector_arrow(s, origin_x, origin_y, Inches(5.6), -6, Inches(0.22), MONGO_TEAL)
label_under_equation(s, Inches(7.7), Inches(2.92), Inches(2.9), Inches(0.3), "After (new taste)", MONGO_TEAL)

box_with_text(
    s, Inches(0.9), Inches(5.55), Inches(11.5), Inches(1.05),
    "New Taste  =  Old Taste",
    ["+ (a little bit toward what you just watched)", "\u2212 (a little bit away from what you just skipped)"],
    fill=LIGHT_BG, title_color=DARK, body_color=GRAY, body_size=13.5, title_size=16,
)

add_footnote(s, "This is exactly what the equation on the next slide calculates \u2014 just expressed as arrows instead of symbols.")


# ===========================================================================
# SLIDE 4 — StreamFlix Hero Equation (Play/Skip)
# ===========================================================================
s = add_slide()
add_title(s, "StreamFlix: The Real-Time User Vector Update", accent=SF_TEAL, tag="StreamFlix")

eq_box = rich_equation_box(
    s, Inches(1.0), Inches(1.7), Inches(11.3), Inches(2.0),
    segments=[
        ("f(U", WHITE), ("t", RGBColor(0x9E,0xC9,0xBB)), (") = normalize( ", WHITE),
        ("(1\u2212\u03b4)\u00b7U", WHITE), ("t", RGBColor(0x9E,0xC9,0xBB)), (" + \u03b4\u00b7P", MONGO_GREEN),
        ("t", RGBColor(0x9E,0xC9,0xBB)),
        ("  \u2212  \u03c1\u00b7(U", RGBColor(0xEF,0x53,0x50)), ("t", RGBColor(0x9E,0xC9,0xBB)), ("\u00b7N", RGBColor(0xEF,0x53,0x50)),
        ("t", RGBColor(0x9E,0xC9,0xBB)), (")N", RGBColor(0xEF,0x53,0x50)), ("t", RGBColor(0x9E,0xC9,0xBB)),
        (" )", WHITE),
    ],
    size=30,
)

# term labels under equation
label_under_equation(s, Inches(4.55), Inches(3.75), Inches(2.3), Inches(0.35), "\u25b6 PLAY term", PLAY_GREEN)
label_under_equation(s, Inches(7.1), Inches(3.75), Inches(2.6), Inches(0.35), "\u23ed SKIP term", SKIP_RED)

box_with_text(s, Inches(1.0), Inches(4.25), Inches(5.5), Inches(1.05),
              "\u25b6 PLAY term", ["Attracts the user vector toward the composite embedding of the played content."],
              fill=PLAY_BG, title_color=PLAY_GREEN, body_size=13)
box_with_text(s, Inches(6.83), Inches(4.25), Inches(5.5), Inches(1.05),
              "\u23ed SKIP term", ["Orthogonally rejects the direction of repeatedly skipped content."],
              fill=SKIP_BG, title_color=SKIP_RED, body_size=13)

legend = [
    "U\u209c = user's current vector      P\u209c = played (attractor) vector      \u03b4 = drift rate (0.35)",
    "N\u209c = normalized skip-direction      \u03c1 = rejection rate (0.5, triggered after repeat skips)",
]
add_bullets(s, Inches(1.0), Inches(5.4), Inches(11.3), Inches(0.9), legend, font_size=14, color=GRAY)

add_callout_banner(s, Inches(1.0), Inches(6.35), Inches(11.3), Inches(0.6),
                    "Recomputed after EVERY play/skip event \u00b7 async write \u00b7 SSE push to UI \u2014 "
                    "no retraining, no batch job, just linear algebra",
                    fill=SF_TEAL, size=13)


# ===========================================================================
# SLIDE 4 — Hero Equation + Region factor
# ===========================================================================
s = add_slide()
add_title(s, "StreamFlix: Real-Time Vector Update \u2014 with Region Factor", accent=SF_TEAL, tag="StreamFlix")

eq_box2 = rich_equation_box(
    s, Inches(0.55), Inches(1.65), Inches(12.25), Inches(2.0),
    segments=[
        ("f(U", WHITE), ("t", RGBColor(0x9E,0xC9,0xBB)), (") = normalize( ", WHITE),
        ("(1\u2212\u03b4)\u00b7U", WHITE), ("t", RGBColor(0x9E,0xC9,0xBB)), (" + \u03b4\u00b7P", MONGO_GREEN),
        ("t", RGBColor(0x9E,0xC9,0xBB)),
        ("  \u2212  \u03c1\u00b7(U", RGBColor(0xEF,0x53,0x50)), ("t", RGBColor(0x9E,0xC9,0xBB)), ("\u00b7N", RGBColor(0xEF,0x53,0x50)),
        ("t", RGBColor(0x9E,0xC9,0xBB)), (")N", RGBColor(0xEF,0x53,0x50)), ("t", RGBColor(0x9E,0xC9,0xBB)),
        ("  +  \u03b3\u00b7R", RGBColor(0xC9,0xA8,0xFF)), ("t", RGBColor(0x9E,0xC9,0xBB)),
        (" )", WHITE),
    ],
    size=26,
)

label_under_equation(s, Inches(3.6), Inches(3.7), Inches(2.0), Inches(0.35), "\u25b6 PLAY", PLAY_GREEN)
label_under_equation(s, Inches(5.85), Inches(3.7), Inches(2.0), Inches(0.35), "\u23ed SKIP", SKIP_RED)
label_under_equation(s, Inches(8.6), Inches(3.7), Inches(2.6), Inches(0.35), "\U0001F310 REGION (proposed)", REGION_PURP)

box_with_text(s, Inches(0.55), Inches(4.2), Inches(3.9), Inches(1.55),
              "\u25b6 PLAY", ["(1\u2212\u03b4)\u00b7U\u209c + \u03b4\u00b7P\u209c", "Attract toward watched content."],
              fill=PLAY_BG, title_color=PLAY_GREEN, body_size=12.5)
box_with_text(s, Inches(4.65), Inches(4.2), Inches(3.9), Inches(1.55),
              "\u23ed SKIP", ["\u2212\u03c1\u00b7(U\u209c\u00b7N\u209c)N\u209c", "Reject skipped direction."],
              fill=SKIP_BG, title_color=SKIP_RED, body_size=12.5)
box_with_text(s, Inches(8.75), Inches(4.2), Inches(4.05), Inches(1.55),
              "\U0001F310 REGION (NEW)", ["+\u03b3\u00b7R\u209c",
                            "R\u209c = viewer's current location context vector (session-detected)",
                            "\u03b3 = region-affinity weight (symbolic \u2014 not yet tuned/enabled)"],
              fill=REGION_BG, title_color=REGION_PURP, body_size=12, dashed=True, line_color=REGION_PURP)

add_callout_banner(s, Inches(0.55), Inches(5.95), Inches(12.25), Inches(0.6),
                    "Same core Play/Skip math, extended with one additive term \u2014 "
                    "no architecture change, no retraining, still pure linear algebra",
                    fill=SF_TEAL, size=13)
add_footnote(s, "Region factor is a proposed extension \u2014 not yet implemented in production.")


# ===========================================================================
# SLIDE 5 — Region Vector: The Formula (detailed)
# ===========================================================================
s = add_slide()
add_title(s, "Proposed Extension: Region-Aware Vector Dimension",
          "Region as a per-session viewer-location signal", accent=REGION_PURP, tag="Proposed")

intro = [
    "region_vec = embedding of the VIEWER\u2019S CURRENT location/locale (e.g. \u201cUS-West\u201d, \u201cKR\u201d, \u201cIN-South\u201d) "
    "\u2014 detected per session, NOT a fixed content attribute.",
    "Represents what's contextually relevant right now \u2014 region nudges the user's real-time vector alongside play/skip signals.",
]
add_bullets(s, Inches(0.5), Inches(1.85), Inches(12.4), Inches(1.3), intro, font_size=16)

box_with_text(
    s, Inches(0.5), Inches(3.3), Inches(6.0), Inches(2.15),
    "\u25b6 PLAY (existing + region term)",
    [
        "U\u209c \u2190 normalize(",
        "   (1\u2212\u03b4)\u00b7U\u209c + \u03b4\u00b7P\u209c + \u03b3\u00b7R\u209c )",
        "",
        "\u03b3 = region-affinity weight",
        "(new, symbolic \u2014 not yet tuned)",
    ],
    fill=PLAY_BG, title_color=PLAY_GREEN, body_size=14, title_size=15,
)

box_with_text(
    s, Inches(6.85), Inches(3.3), Inches(6.0), Inches(2.15),
    "Ephemeral per-request query (RFY)",
    [
        "V_rfy = normalize(",
        "   adjusted + \u03b3\u00b7R\u209c )",
        "",
        "Applied at query time, without",
        "touching the persisted profile",
    ],
    fill=REGION_BG, title_color=REGION_PURP, body_size=14, title_size=15,
)

add_callout_banner(
    s, Inches(0.5), Inches(5.7), Inches(12.4), Inches(0.6),
    "Skip/rejection math (\u03c1 term) unchanged \u2014 region is an additive contextual pull, not a rejection signal.",
    fill=DARK, size=14
)


# ===========================================================================
# SLIDE 6 — Region Vector Impact by Location
# ===========================================================================
s = add_slide()
add_title(s, "Same Taste Vector \u2192 Different Recs by Viewer Location", accent=REGION_PURP, tag="Proposed")

# top box: user_vec
uv = box_with_text(s, Inches(5.4), Inches(1.5), Inches(2.6), Inches(0.8),
                    "user_vec", ["Shared long-term taste profile", "(e.g. Sci-Fi lean) \u2014 same regardless of location"],
                    fill=SF_TEAL_BG, title_color=SF_TEAL, body_size=11, title_size=15)

add_line(s, Inches(6.0), Inches(2.3), Inches(3.1), Inches(2.9))
add_line(s, Inches(7.4), Inches(2.3), Inches(10.0), Inches(2.9))

box_with_text(s, Inches(1.5), Inches(2.95), Inches(3.6), Inches(1.0),
              "+ region_vec: \u201ccurrently in US\u201d", ["session-detected location", "(e.g. IP / device locale)"],
              fill=REGION_BG, title_color=REGION_PURP, body_size=11.5, title_size=13.5)
box_with_text(s, Inches(8.35), Inches(2.95), Inches(3.6), Inches(1.0),
              "+ region_vec: \u201ccurrently in KR\u201d", ["session-detected location", "(e.g. IP / device locale)"],
              fill=REGION_BG, title_color=REGION_PURP, body_size=11.5, title_size=13.5)

add_line(s, Inches(3.3), Inches(3.95), Inches(3.3), Inches(4.3))
add_line(s, Inches(10.15), Inches(3.95), Inches(10.15), Inches(4.3))

box_with_text(s, Inches(1.5), Inches(4.35), Inches(3.6), Inches(1.15),
              "\u201cInterstellar\u201d, \u201cThe Martian\u201d", ["Same taste, US-contextual pull"],
              fill=LIGHT_BG, title_color=DARK, body_size=12.5, title_size=14.5)
box_with_text(s, Inches(8.35), Inches(4.35), Inches(3.6), Inches(1.15),
              "\u201cSpace Sweepers\u201d, \u201cThe Wandering Earth\u201d", ["Same taste, KR-contextual pull"],
              fill=LIGHT_BG, title_color=DARK, body_size=12.5, title_size=14.5)

add_callout_banner(
    s, Inches(0.5), Inches(5.9), Inches(12.4), Inches(0.75),
    "Same person, same history \u2014 but recommendations shift with where they're watching from right now "
    "(e.g. traveler, relocated user)",
    fill=SF_TEAL, size=15
)


# ===========================================================================
# SLIDE 7 — Traditional ML Approach
# ===========================================================================
s = add_slide()
add_title(s, "Traditional ML Approach: Batch-Trained Recommenders", accent=ML_ORANGE, tag="Traditional ML")

methods_tb = add_bullets(s, Inches(0.5), Inches(1.5), Inches(12.4), Inches(0.5),
    ["Methods: Collaborative Filtering \u00b7 Matrix Factorization \u00b7 Gradient-Boosted Trees \u00b7 Two-Tower Neural Nets"],
    font_size=17)

# Pipeline: 4 boxes with right arrows
stages = [
    ("Feature\nEngineering", ""),
    ("Offline Training", "(hours\u2013days)"),
    ("Offline Eval / A-B", ""),
    ("Scheduled Redeploy", "(days\u2013weeks)"),
]
box_w = Inches(2.55)
box_h = Inches(1.1)
gap_w = Inches(0.55)
start_x = Inches(0.6)
py = Inches(2.35)
for i, (title, sub) in enumerate(stages):
    x = start_x + i * (box_w + gap_w)
    box_with_text(s, x, py, box_w, box_h, title, [sub] if sub else [],
                  fill=ML_ORANGE_BG, title_color=ML_ORANGE, body_size=12.5, title_size=14.5)
    if i < len(stages) - 1:
        add_right_arrow(s, x + box_w + Inches(0.03), py + box_h/2 - Inches(0.15), w=Inches(0.5), h=Inches(0.3), color=ML_ORANGE)

note = ["Served from cached / batch-scored tables \u2014 refreshed periodically, not from live behavior"]
add_bullets(s, Inches(0.6), Inches(3.75), Inches(12.0), Inches(0.5), note, font_size=15, color=GRAY)

pros_cons = [
    ("\u2714 Strengths: mature tooling, models complex nonlinear interactions well", 0),
    ("\u2718 Weaknesses: cold-start lag, stale until next retrain, heavy MLOps overhead", 0),
    ("\u2192 Slow time-to-market, NOT real-time", 0),
]
add_bullets(s, Inches(0.6), Inches(4.4), Inches(12.0), Inches(1.6), pros_cons, font_size=17)

add_callout_banner(s, Inches(0.5), Inches(6.35), Inches(12.4), Inches(0.65),
                    "Recommendations only reflect a user's latest watch/skip after the NEXT retrain cycle",
                    fill=ML_ORANGE, size=14)


# ===========================================================================
# SLIDE 8 — Comparison table
# ===========================================================================
s = add_slide()
add_title(s, "Comparison: Three Approaches to Content Recommendation", accent=SF_TEAL, tag="Summary")

rows_data = [
    ["Dimension", "YouTube (Gemini LRM)", "Traditional ML", "StreamFlix (Rocchio)"],
    ["Core Method", "Fine-tuned LLM + Semantic IDs", "CF / GBDT / Two-Tower", "Real-time vector math + $vectorSearch"],
    ["Real-Time Adaptation", "Near real-time (offline tables)", "Batch cycle (hrs\u2013weeks)", "Instant \u2014 same request"],
    ["Infra & Cost", "Very High (LLM fine-tune + continuous retrain)", "Moderate\u2013High (feature stores, training clusters)", "Low (lightweight math, existing vector DB)"],
    ["Time-to-Market", "Long (LLM + SID infra)", "Slow (weeks\u2013months)", "Fast (formula-based)"],
    ["Freshness", "Minutes", "Days\u2013weeks", "Immediate"],
    ["Explainability", "Low (black box)", "Low\u2013Medium", "High (transparent formulas)"],
    ["Scale Requirement", "YouTube-scale only", "Large labeled datasets", "Small \u2192 large catalogs OK"],
]

n_rows = len(rows_data)
n_cols = 4
tbl_x, tbl_y = Inches(0.4), Inches(1.5)
tbl_w, tbl_h = Inches(12.55), Inches(4.9)

gframe = s.shapes.add_table(n_rows, n_cols, tbl_x, tbl_y, tbl_w, tbl_h)
table = gframe.table

col_widths = [Inches(2.55), Inches(3.35), Inches(3.35), Inches(3.3)]
for i, w in enumerate(col_widths):
    table.columns[i].width = w

header_fill = DARK
sf_col_fill = SF_TEAL_BG
sf_header_fill = SF_TEAL

for r in range(n_rows):
    row_h = Inches(0.42) if r == 0 else Inches(0.67)
    table.rows[r].height = row_h
    for c in range(n_cols):
        cell = table.cell(r, c)
        cell.text = rows_data[r][c]
        cell.margin_left = Inches(0.1)
        cell.margin_right = Inches(0.1)
        cell.margin_top = Inches(0.04)
        cell.margin_bottom = Inches(0.04)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        para = cell.text_frame.paragraphs[0]
        para.font.size = Pt(10.5 if r else 13)
        para.font.name = FONT_TITLE if (r == 0 or c in (0, 3)) else FONT_BODY

        if r == 0:
            para.font.bold = True
            para.font.color.rgb = WHITE
            cell.fill.solid()
            cell.fill.fore_color.rgb = sf_header_fill if c == 3 else header_fill
            para.alignment = PP_ALIGN.CENTER
        else:
            if c == 0:
                para.font.bold = True
                para.font.color.rgb = DARK
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT_BG
            elif c == 3:
                para.font.bold = True
                para.font.color.rgb = RGBColor(0x00, 0x4D, 0x40)
                cell.fill.solid()
                cell.fill.fore_color.rgb = sf_col_fill
            else:
                para.font.color.rgb = GRAY
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE

add_callout_banner(
    s, Inches(0.4), Inches(6.8), Inches(12.55), Inches(0.6),
    "StreamFlix delivers true real-time personalization at a fraction of the cost and time-to-market \u2014 "
    "no LLM fine-tuning, no retraining pipeline.",
    fill=SF_TEAL, size=15
)


# ===========================================================================
# SLIDE 9 — Influencing Results: Promoted & Sponsored Content (IMPLEMENTED)
# ===========================================================================
s = add_slide()
add_title(s, "Influencing Results: Promoted & Sponsored Content",
          accent=PROMO_GOLD, tag="Implemented")

card_w = Inches(2.2)
card_gap = Inches(0.2)
card_h = Inches(0.75)
start_x = Inches(0.5)

label_under_equation(s, start_x, Inches(1.35), Inches(11.8), Inches(0.28), "BEFORE", GRAY)

before_cards = [
    ("#1", "Interstellar", "0.91"),
    ("#2", "Arrival", "0.88"),
    ("#3", "Dune", "0.86"),
    ("#4", "Midnight Sky", "0.81"),
    ("#5", "Ad Astra", "0.79"),
]
row1_y = Inches(1.65)
for i, (rank, title, score) in enumerate(before_cards):
    x = start_x + i * (card_w + card_gap)
    box_with_text(s, x, row1_y, card_w, card_h, f"{rank}  {title}", [score],
                  fill=LIGHT_BG, title_color=DARK, body_color=GRAY, body_size=11, title_size=12.5)

# arrow down from card #4 into the ranking-table box
card4_x = start_x + 3 * (card_w + card_gap)
add_down_arrow(s, card4_x + card_w/2 - Inches(0.15), row1_y + card_h + Inches(0.06), w=Inches(0.3), h=Inches(0.28), color=PROMO_GOLD)

# --- boost_score ranking table ------------------------------------------------
tbl_box_w = Inches(5.6)
tbl_box_x = Emu(int((Inches(13.333) - tbl_box_w) / 2))
tbl_box_y = row1_y + card_h + Inches(0.42)
tbl_box_h = Inches(1.9)

outer = add_box(s, tbl_box_x, tbl_box_y, tbl_box_w, tbl_box_h, fill=PROMO_BG, line_color=PROMO_GOLD)

cap = s.shapes.add_textbox(tbl_box_x, tbl_box_y + Inches(0.08), tbl_box_w, Inches(0.3))
p = cap.text_frame.paragraphs[0]
p.alignment = PP_ALIGN.CENTER
p.text = "boost_score decides order"
p.font.size = Pt(13)
p.font.bold = True
p.font.name = FONT_TITLE
p.font.color.rgb = PROMO_GOLD

rank_rows = [
    ["Movie", "boost_score", "Rank"],
    ["Midnight Sky", "2.0", "#1  \u2605"],
    ["Ocean's 9", "1.5", "#2"],
    ["Wild Hunt", "1.2", "#3"],
]
tw, th = Inches(5.0), Inches(1.35)
tx = tbl_box_x + Emu(int((tbl_box_w - tw) / 2))
ty = tbl_box_y + Inches(0.42)
gframe = s.shapes.add_table(4, 3, tx, ty, tw, th)
table = gframe.table
table.columns[0].width = Inches(2.2)
table.columns[1].width = Inches(1.5)
table.columns[2].width = Inches(1.3)
for r in range(4):
    table.rows[r].height = Inches(0.32) if r == 0 else Inches(0.34)
    for c in range(3):
        cell = table.cell(r, c)
        cell.text = rank_rows[r][c]
        cell.margin_top = Inches(0.02)
        cell.margin_bottom = Inches(0.02)
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        para = cell.text_frame.paragraphs[0]
        para.alignment = PP_ALIGN.CENTER
        para.font.name = FONT_BODY
        para.font.size = Pt(12 if r else 11)
        if r == 0:
            para.font.bold = True
            para.font.color.rgb = WHITE
            cell.fill.solid()
            cell.fill.fore_color.rgb = DARK
        elif r == 1:
            para.font.bold = True
            para.font.color.rgb = PROMO_GOLD
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE
        else:
            para.font.color.rgb = GRAY
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE

# arrow down from ranking table into AFTER row
add_down_arrow(s, card4_x + card_w/2 - Inches(0.15), tbl_box_y + tbl_box_h + Inches(0.05), w=Inches(0.3), h=Inches(0.28), color=PROMO_GOLD)

label_under_equation(s, start_x, tbl_box_y + tbl_box_h + Inches(0.4), Inches(11.8), Inches(0.28), "AFTER", GRAY)

after_cards = [
    ("#1", "Midnight Sky", "\u2605 SPONSORED"),
    ("#2", "Interstellar", "0.91"),
    ("#3", "Arrival", "0.88"),
    ("#4", "Dune", "0.86"),
    ("#5", "Ad Astra", "0.79"),
]
row2_y = tbl_box_y + tbl_box_h + Inches(0.7)
for i, (rank, title, score) in enumerate(after_cards):
    x = start_x + i * (card_w + card_gap)
    if i == 0:
        box_with_text(s, x, row2_y, card_w, card_h, f"{rank}  {title}", [score],
                      fill=PROMO_BG, title_color=PROMO_GOLD, body_color=PROMO_GOLD, body_size=10.5,
                      title_size=12.5, line_color=PROMO_GOLD)
    else:
        box_with_text(s, x, row2_y, card_w, card_h, f"{rank}  {title}", [score],
                      fill=LIGHT_BG, title_color=DARK, body_color=GRAY, body_size=11, title_size=12.5)

add_callout_banner(s, Inches(0.5), row2_y + card_h + Inches(0.3), Inches(12.4), Inches(0.5),
                    "Guaranteed slot \u2014 vector_score is never used to decide it.",
                    fill=PROMO_GOLD, size=13)


# ===========================================================================
# SLIDE 10 — Influencing Results: Trending (Click-Rate Driven) [PROPOSED]
# ===========================================================================
s = add_slide()
add_title(s, "Influencing Results: Trending Content (Click-Rate Driven)",
          accent=REGION_PURP, tag="Proposed")

# Pipeline: 4 boxes with right arrows
stages = [
    ("Impression", "(shown)"),
    ("Click / Play", ""),
    ("CTR score", ""),
    ("trending_score", ""),
]
box_w = Inches(2.85)
box_h = Inches(0.95)
gap_w = Inches(0.25)
start_x2 = Inches(0.55)
py = Inches(1.7)
box_xs = []
for i, (title, sub) in enumerate(stages):
    x = start_x2 + i * (box_w + gap_w)
    box_xs.append(x)
    box_with_text(s, x, py, box_w, box_h, title, [sub] if sub else [],
                  fill=REGION_BG, title_color=REGION_PURP, body_size=11, title_size=14)
    if i < len(stages) - 1:
        add_right_arrow(s, x + box_w + Inches(0.02), py + box_h/2 - Inches(0.13), w=Inches(0.21), h=Inches(0.26), color=REGION_PURP)

# CTR formula chip, centered under the "CTR score" box
ctr_x = box_xs[2]
rich_equation_box(
    s, ctr_x - Inches(0.35), py + box_h + Inches(0.25), Inches(3.55), Inches(0.55),
    segments=[("CTR = clicks \u00f7 impressions", MONGO_GREEN)], size=15, box_fill=DARK,
)

# down arrow from trending_score box into the blend formula box
tscore_x = box_xs[3]
add_down_arrow(s, tscore_x + box_w/2 - Inches(0.15), py + box_h + Inches(0.15), w=Inches(0.3), h=Inches(0.3), color=REGION_PURP)

blend_w, blend_h = Inches(4.9), Inches(1.35)
blend_x = tscore_x + box_w/2 - blend_w/2 + Inches(0.6)
blend_y = py + box_h + Inches(1.05)
if blend_x + blend_w > Inches(12.85):
    blend_x = Inches(12.85) - blend_w

rich_equation_box(
    s, blend_x, blend_y, blend_w, Inches(0.65),
    segments=[("score \u00d7= (1 + w \u00b7 trend)", MONGO_GREEN)], size=17, box_fill=DARK,
)
label_under_equation(s, blend_x, blend_y + Inches(0.7), blend_w, Inches(0.3),
                      "soft nudge \u2014 not a guaranteed slot", REGION_PURP)

add_callout_banner(
    s, Inches(0.5), Inches(6.35), Inches(12.4), Inches(0.65),
    "\u26a0 NOT BUILT YET \u2014 no impression logging exists in the system today",
    fill=SKIP_RED, size=15
)


# ===========================================================================
# SLIDE 11 — Tuning Rocchio Parameters Dynamically [PROPOSED]
# ===========================================================================
s = add_slide()
add_title(s, "Tuning Rocchio Parameters Dynamically", accent=REGION_PURP, tag="Proposed")

center_x = Inches(13.333)
box1_w = Inches(4.0)
box1_x = Emu(int((center_x - box1_w) / 2))

box1_y = Inches(1.55)
box_h = Inches(0.75)
box_with_text(s, box1_x, box1_y, box1_w, box_h, "algo_params (DB)", ["global defaults"],
              fill=REGION_BG, title_color=REGION_PURP, body_size=12, title_size=15, line_color=REGION_PURP)
label_under_equation(s, box1_x + box1_w + Inches(0.3), box1_y + Inches(0.22), Inches(3.6), Inches(0.3),
                      "\u25c0 admin PATCH \u2014 live update", GRAY)

arrow_x = Emu(int((center_x - Inches(0.3)) / 2))
add_down_arrow(s, arrow_x, box1_y + box_h + Inches(0.1), w=Inches(0.3), h=Inches(0.3), color=REGION_PURP)

box2_y = box1_y + box_h + Inches(0.7)
box_with_text(s, box1_x, box2_y, box1_w, box_h, "In-Memory Cache", ["fast per-request reads"],
              fill=REGION_BG, title_color=REGION_PURP, body_size=12, title_size=15, line_color=REGION_PURP)
label_under_equation(s, box1_x + box1_w + Inches(0.3), box2_y + Inches(0.22), Inches(3.7), Inches(0.3),
                      "\u25c0 30s TTL / change-stream push", GRAY)

add_down_arrow(s, arrow_x, box2_y + box_h + Inches(0.1), w=Inches(0.3), h=Inches(0.3), color=REGION_PURP)

box3_w = Inches(8.0)
box3_x = Emu(int((center_x - box3_w) / 2))
box3_y = box2_y + box_h + Inches(0.7)
box3_h = Inches(0.85)
box_with_text(s, box3_x, box3_y, box3_w, box3_h, "Per-Request Parameter Resolution",
              ["user override  \u2192  cohort override  \u2192  global default"],
              fill=LIGHT_BG, title_color=DARK, body_size=13, title_size=15)

add_down_arrow(s, arrow_x, box3_y + box3_h + Inches(0.1), w=Inches(0.3), h=Inches(0.3), color=REGION_PURP)

eq_w = Inches(6.4)
eq_x = Emu(int((center_x - eq_w) / 2))
eq_y = box3_y + box3_h + Inches(0.55)
rich_equation_box(
    s, eq_x, eq_y, eq_w, Inches(0.65),
    segments=[("\u03b4, \u03c1, \u03b1, \u03b2", MONGO_GREEN), ("  \u2192  used in Rocchio math", WHITE)],
    size=18, box_fill=DARK,
)

label_under_equation(s, Inches(0.5), eq_y + Inches(0.85), Inches(12.4), Inches(0.3),
                      "no redeploy  \u00b7  no restart  \u00b7  effective in seconds", REGION_PURP)


# ===========================================================================
# SLIDE 12 — A/B Testing the Rocchio Parameters [PROPOSED]
# ===========================================================================
s = add_slide()
add_title(s, "A/B Testing the Rocchio Parameters", accent=REGION_PURP, tag="Proposed")

hash_w = Inches(3.6)
hash_x = Emu(int((center_x - hash_w) / 2))
hash_y = Inches(1.45)
hash_h = Inches(0.6)
box_with_text(s, hash_x, hash_y, hash_w, hash_h, "hash(user_id) % 100", [],
              fill=DARK, title_color=WHITE, title_size=15)

ctrl_w = Inches(3.0)
ctrl_h = Inches(0.8)
ctrl_x = Inches(2.0)
var_x = Inches(8.33)
split_y = Inches(2.35)

add_line(s, hash_x + hash_w//2, hash_y + hash_h, ctrl_x + ctrl_w//2, split_y, color=REGION_PURP, width=1.5)
add_line(s, hash_x + hash_w//2, hash_y + hash_h, var_x + ctrl_w//2, split_y, color=REGION_PURP, width=1.5)

box_with_text(s, ctrl_x, split_y, ctrl_w, ctrl_h, "Control (A)", ["\u03b4 = 0.35"],
              fill=LIGHT_BG, title_color=DARK, body_color=GRAY, body_size=13, title_size=15)
box_with_text(s, var_x, split_y, ctrl_w, ctrl_h, "Variant (B)", ["\u03b4 = 0.20"],
              fill=REGION_BG, title_color=REGION_PURP, body_size=13, title_size=15, line_color=REGION_PURP)

label_under_equation(s, Inches(0.5), split_y + ctrl_h + Inches(0.12), Inches(12.4), Inches(0.28),
                      "events tagged by variant", GRAY)

arrow_x2 = Emu(int((center_x - Inches(0.3)) / 2))
add_down_arrow(s, arrow_x2, split_y + ctrl_h + Inches(0.45), w=Inches(0.3), h=Inches(0.28), color=REGION_PURP)

chart_cap_y = split_y + ctrl_h + Inches(0.85)
label_under_equation(s, Inches(0.5), chart_cap_y, Inches(12.4), Inches(0.28),
                      "Play-Rate %  \u2014  Control vs. Variant", DARK)

chart_y = chart_cap_y + Inches(0.32)
chart_w = Inches(10.3)
chart_x = Emu(int((center_x - chart_w) / 2))
chart_h = Inches(2.15)

chart_data = CategoryChartData()
chart_data.categories = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]
chart_data.add_series("Control (A)", (24, 23, 25, 24, 26, 25, 24))
chart_data.add_series("Variant (B)", (24, 27, 30, 33, 35, 37, 39))

gframe = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, chart_x, chart_y, chart_w, chart_h, chart_data)
chart = gframe.chart
chart.has_title = False
chart.font.name = FONT_BODY
chart.font.size = Pt(11)

chart.has_legend = True
chart.legend.position = XL_LEGEND_POSITION.BOTTOM
chart.legend.include_in_layout = False
chart.legend.font.size = Pt(11)
chart.legend.font.name = FONT_BODY

plot = chart.plots[0]
series_control = plot.series[0]
series_control.format.line.color.rgb = GRAY
series_control.format.line.width = Pt(2)
series_control.smooth = False
series_variant = plot.series[1]
series_variant.format.line.color.rgb = REGION_PURP
series_variant.format.line.width = Pt(2.75)
series_variant.smooth = False

cat_axis = chart.category_axis
cat_axis.tick_labels.font.size = Pt(10)
cat_axis.tick_labels.font.name = FONT_BODY
val_axis = chart.value_axis
val_axis.tick_labels.font.size = Pt(10)
val_axis.tick_labels.font.name = FONT_BODY
val_axis.has_major_gridlines = False

label_under_equation(s, Inches(0.5), chart_y + chart_h + Inches(0.15), Inches(12.4), Inches(0.3),
                      "roll out winning variant", REGION_PURP)


# ---------------------------------------------------------------------------
out_path = "/Users/nikofeng/Documents/Github/mongodb-mflix-behavior-analysis/docs/recommendation-approaches-comparison.pptx"
prs.save(out_path)
print(f"Saved: {out_path}")
print(f"Slides: {len(prs.slides)}")
