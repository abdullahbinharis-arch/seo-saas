"""
pdf_export.py — Generate a branded PDF report from a completed SEO audit.

Layout (v3):
  Page 1:   Cover — business name, URL, date, overall score
  Page 2:   Score Overview — overall score (large) + 4 pillar scores
  Page 3-4: Top 10 Quick Wins
  Page 5-8: One page per pillar with improvement steps
  Last:     CTA — upgrade prompt

Usage:
    from pdf_export import build_pdf
    pdf_bytes = build_pdf(audit_dict)
"""

from __future__ import annotations

from datetime import datetime
from fpdf import FPDF

# ---------------------------------------------------------------------------
# Latin-1 sanitiser — Helvetica only supports Latin-1 (no emoji / Unicode)
# ---------------------------------------------------------------------------
_REPLACEMENTS = {
    "\u2026": "...",   # ellipsis
    "\u2018": "'",     # left single quote
    "\u2019": "'",     # right single quote
    "\u201c": '"',     # left double quote
    "\u201d": '"',     # right double quote
    "\u2013": "-",     # en dash
    "\u2014": "--",    # em dash
    "\u2022": "*",     # bullet
    "\u00b7": ".",     # middle dot
    "\u00e9": "e",     # e-acute
    "\u2192": "->",    # arrow
}

def _s(text) -> str:
    """Latin-1-safe, single-line string for cell()."""
    t = str(text)
    for char, repl in _REPLACEMENTS.items():
        t = t.replace(char, repl)
    t = t.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return t.encode("latin-1", errors="replace").decode("latin-1")

def _ms(text) -> str:
    """Latin-1-safe string that preserves newlines for multi_cell()."""
    t = str(text)
    for char, repl in _REPLACEMENTS.items():
        t = t.replace(char, repl)
    t = t.replace("\r", "").replace("\t", " ")
    return t.encode("latin-1", errors="replace").decode("latin-1")

# ---------------------------------------------------------------------------
# Colour palette (dark-themed brand colours mapped to print)
# ---------------------------------------------------------------------------
NAVY       = (15,  23,  42)
DARK       = (24,  24,  27)   # card bg (#18181b)
EMERALD    = (16, 185, 129)
BLUE       = (59, 130, 246)
AMBER      = (245, 158,  11)
ROSE       = (244,  63,  62)
VIOLET     = (139,  92, 246)
LIGHT_BLUE = (219, 234, 254)
GRAY_BG    = (248, 250, 252)
GRAY_LINE  = (226, 232, 240)
GRAY_TEXT  = (100, 116, 139)
GREEN      = (22, 163,  74)
RED        = (220,  38,  38)
WHITE      = (255, 255, 255)

# Score colour helper
def _score_color(score: int) -> tuple:
    if score >= 70: return EMERALD
    if score >= 40: return AMBER
    return ROSE

def _score_label(score: int) -> str:
    if score >= 70: return "Good"
    if score >= 40: return "Needs Work"
    return "Critical"

def _priority_color(priority: str) -> tuple:
    if priority == "high": return ROSE
    if priority == "medium": return AMBER
    return EMERALD

# Pillar display names
_PILLAR_LABELS = {
    "website_seo": "Website SEO",
    "backlinks": "Backlinks",
    "local_seo": "Local SEO",
    "ai_seo": "AI SEO",
}


# ---------------------------------------------------------------------------
# PDF subclass
# ---------------------------------------------------------------------------

class SEOReport(FPDF):
    def __init__(self, audit: dict):
        super().__init__()
        self.audit = audit
        self.set_margins(18, 18, 18)
        self.set_auto_page_break(auto=True, margin=22)

    def header(self):
        if self.page_no() <= 1:
            return
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*GRAY_TEXT)
        biz = self.audit.get("business_name") or self.audit.get("target_url", "")
        self.cell(0, 6, _s(f"LocalRank SEO Report  |  {biz}"), align="L")
        self.ln(1)
        self.set_draw_color(*GRAY_LINE)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-16)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*GRAY_TEXT)
        self.cell(0, 6, f"Page {self.page_no()} of {{nb}}", align="C")

    # ── Drawing helpers ───────────────────────────────────────────────

    def rule(self):
        self.set_draw_color(*GRAY_LINE)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def section_title(self, title: str, color=EMERALD):
        self.ln(2)
        self.set_fill_color(*color)
        self.rect(self.l_margin, self.get_y(), 3, 9, "F")
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(*NAVY)
        self.set_x(self.l_margin + 6)
        self.cell(0, 9, _s(title), ln=True)
        self.ln(3)

    def sub_heading(self, text: str):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*NAVY)
        self.cell(0, 6, _s(text), ln=True)
        self.ln(1)

    def body(self, text: str):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*NAVY)
        self.multi_cell(0, 5, _ms(text))

    def small(self, text: str):
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*GRAY_TEXT)
        self.multi_cell(0, 5, _ms(text))

    def kv(self, key: str, value: str):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*GRAY_TEXT)
        self.cell(40, 5, _s(f"{key}:"))
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*NAVY)
        self.multi_cell(0, 5, _s(value))
        self.set_x(self.l_margin)

    def score_box(self, score: int, label: str, x: float, y: float, w: float, h: float):
        """Draw a score card with coloured score number."""
        color = _score_color(score)
        # Card background
        self.set_fill_color(*GRAY_BG)
        self.rect(x, y, w, h, "F")
        self.set_draw_color(*GRAY_LINE)
        self.rect(x, y, w, h, "D")
        # Score number
        self.set_font("Helvetica", "B", 22)
        self.set_text_color(*color)
        self.set_xy(x, y + 4)
        self.cell(w, 12, str(score), align="C")
        # "/100" suffix
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*GRAY_TEXT)
        self.set_xy(x, y + 15)
        self.cell(w, 5, "/100", align="C")
        # Label
        self.set_font("Helvetica", "B", 7)
        self.set_text_color(*NAVY)
        self.set_xy(x, y + 21)
        self.cell(w, 5, _s(label), align="C")
        # Status text
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*color)
        self.set_xy(x, y + 26)
        self.cell(w, 5, _score_label(score), align="C")

    def priority_badge(self, priority: str):
        color = _priority_color(priority)
        label = priority.capitalize() if priority != "low" else "Growth"
        self.set_font("Helvetica", "B", 7)
        self.set_text_color(*WHITE)
        self.set_fill_color(*color)
        self.cell(14, 4, _s(label), fill=True, align="C")
        self.set_text_color(*NAVY)
        self.set_fill_color(*WHITE)

    def pillar_badge(self, pillar: str):
        label = _PILLAR_LABELS.get(pillar, pillar)
        self.set_font("Helvetica", "", 7)
        self.set_text_color(*GRAY_TEXT)
        self.set_fill_color(*GRAY_BG)
        self.cell(len(label) * 3 + 6, 4, _s(label), fill=True, align="C")
        self.set_fill_color(*WHITE)


# ---------------------------------------------------------------------------
# Page renderers
# ---------------------------------------------------------------------------

def _cover(pdf: SEOReport):
    """Page 1: Cover with business name, URL, date, overall score."""
    pdf.add_page()
    a = pdf.audit

    # Dark banner
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, pdf.w, 75, "F")

    # Emerald accent bar
    pdf.set_fill_color(*EMERALD)
    pdf.rect(0, 75, pdf.w, 3, "F")

    # Brand
    pdf.set_xy(18, 14)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*EMERALD)
    pdf.cell(0, 6, "LOCALRANK", ln=True)

    # Title
    pdf.set_x(18)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 12, "SEO Audit Report", ln=True)

    # Business name
    biz = a.get("business_name", "")
    if biz:
        pdf.set_x(18)
        pdf.set_font("Helvetica", "", 13)
        pdf.set_text_color(148, 163, 184)
        pdf.cell(0, 8, _s(biz), ln=True)

    # URL
    pdf.set_x(18)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 7, _s(a.get("target_url", "")), ln=True)

    # Overall score in top-right of banner
    scores = a.get("scores", {})
    overall = scores.get("overall", 0)
    if overall:
        color = _score_color(overall)
        # Circle
        cx, cy = pdf.w - 45, 38
        pdf.set_fill_color(*color)
        pdf.ellipse(cx - 18, cy - 18, 36, 36, "F")
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(*WHITE)
        pdf.set_xy(cx - 18, cy - 7)
        pdf.cell(36, 14, str(overall), align="C")
        pdf.set_font("Helvetica", "", 7)
        pdf.set_xy(cx - 18, cy + 7)
        pdf.cell(36, 5, "/ 100", align="C")

    # Meta section
    pdf.set_xy(18, 84)

    ts = a.get("timestamp", "")
    try:
        dt = datetime.fromisoformat(ts)
        date_str = dt.strftime("%B %d, %Y")
    except Exception:
        date_str = ts[:10] if ts else "N/A"

    meta_items = [
        ("Keyword",   a.get("keyword", "")),
        ("Location",  a.get("location", "")),
        ("Date",      date_str),
        ("Agents",    str(a.get("agents_executed", 11))),
        ("Time",      f"{a.get('execution_time_seconds', 0):.0f}s"),
        ("Est. cost", f"${a.get('estimated_cost', a.get('summary', {}).get('estimated_api_cost', 0)):.3f}"),
    ]
    for key, val in meta_items:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*GRAY_TEXT)
        pdf.cell(32, 6, _s(f"{key}:"))
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*NAVY)
        pdf.cell(0, 6, _s(str(val)), ln=True)

    pdf.ln(4)
    pdf.rule()


def _score_overview(pdf: SEOReport):
    """Page 2: Overall score (large, centred) + 4 pillar score cards."""
    a = pdf.audit
    scores = a.get("scores", {})
    if not scores:
        return

    pdf.add_page()
    pdf.section_title("Score Overview", EMERALD)

    overall = scores.get("overall", 0)
    color = _score_color(overall)

    # Large centred overall score
    cx = pdf.w / 2
    cy = pdf.get_y() + 25
    # Outer ring
    pdf.set_draw_color(*color)
    pdf.set_line_width(2)
    pdf.ellipse(cx - 22, cy - 22, 44, 44, "D")
    pdf.set_line_width(0.2)
    # Score text
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(*color)
    pdf.set_xy(cx - 22, cy - 10)
    pdf.cell(44, 18, str(overall), align="C")
    # "/100" below
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*GRAY_TEXT)
    pdf.set_xy(cx - 22, cy + 6)
    pdf.cell(44, 6, "/100", align="C")
    # Label
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*NAVY)
    pdf.set_xy(cx - 50, cy + 16)
    pdf.cell(100, 7, "Overall LocalRank Score", align="C")
    # Status
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*color)
    pdf.set_xy(cx - 50, cy + 23)
    pdf.cell(100, 6, _score_label(overall), align="C")

    pdf.set_y(cy + 38)

    # 4 pillar cards in a row
    pillars = ["website_seo", "backlinks", "local_seo", "ai_seo"]
    card_w = 40
    gap = 5
    total_w = len(pillars) * card_w + (len(pillars) - 1) * gap
    start_x = (pdf.w - total_w) / 2
    card_y = pdf.get_y()

    for i, key in enumerate(pillars):
        score = scores.get(key, 0)
        label = _PILLAR_LABELS.get(key, key)
        x = start_x + i * (card_w + gap)
        pdf.score_box(score, label, x, card_y, card_w, 34)

    pdf.set_y(card_y + 40)

    # Score details
    details = a.get("score_details", {})
    if details:
        pdf.ln(4)
        pdf.sub_heading("Score Breakdown")

        ws = details.get("website_seo", {})
        if ws:
            pdf.kv("Page Speed", f"{ws.get('page_speed', 0)}/100")
            pdf.kv("On-Page", f"{ws.get('on_page', 0)}/100")
            pdf.kv("Technical", f"{ws.get('technical', 0)}/100")
            pdf.kv("Issues Found", str(ws.get("issues_count", 0)))
            pdf.ln(2)

        bl = details.get("backlinks", {})
        if bl:
            pdf.kv("Estimated DA", str(bl.get("estimated_da", 0)))
            pdf.kv("Est. Backlinks", str(bl.get("estimated_backlinks", 0)))
            pdf.kv("Competitors Avg DA", str(bl.get("competitors_avg_da", 0)))
            pdf.ln(2)

        ls = details.get("local_seo", {})
        if ls:
            pdf.kv("GBP Status", str(ls.get("gbp_status", "unknown")))
            pdf.kv("Citations", f"{ls.get('citations_found', 0)} / {ls.get('citations_needed', 0)}")
            pdf.kv("Reviews", str(ls.get("review_count", 0)))
            pdf.ln(2)

        ai = details.get("ai_seo", {})
        if ai:
            pdf.kv("FAQ Schema", "Yes" if ai.get("faq_schema") else "No")
            pdf.kv("E-E-A-T Signals", "Yes" if ai.get("eeat_signals") else "No")
            pdf.kv("Content Depth", str(ai.get("content_depth", "unknown")))


def _quick_wins(pdf: SEOReport):
    """Pages 3-4: Top 10 Quick Wins."""
    # Try new structured quick_wins first, fall back to summary
    wins = pdf.audit.get("quick_wins", [])
    if not wins:
        old_wins = pdf.audit.get("summary", {}).get("quick_wins", [])
        if not old_wins:
            return
        # Render old-style string wins
        pdf.add_page()
        pdf.section_title("Top Quick Wins", EMERALD)
        for i, w in enumerate(old_wins, 1):
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*NAVY)
            pdf.cell(8, 6, f"{i}.")
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 6, _s(w))
            pdf.ln(1)
        return

    pdf.add_page()
    pdf.section_title("Top 10 Quick Wins", EMERALD)
    pdf.small("Highest-impact actions sorted by expected ranking improvement. Do these first.")
    pdf.ln(3)

    for win in wins:
        rank = win.get("rank", 0)
        title = win.get("title", "")
        desc = win.get("description", "")
        pillar = win.get("pillar", "")
        priority = win.get("priority", "medium")
        impact = win.get("impact", "")
        time_est = win.get("time_estimate", "")

        # Check if we need a new page (leave room for at least 25mm)
        if pdf.get_y() > pdf.h - 40:
            pdf.add_page()

        # Rank number
        p_color = _priority_color(priority)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*p_color)
        y_start = pdf.get_y()
        pdf.cell(8, 6, str(rank))

        # Title
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*NAVY)
        x_body = pdf.l_margin + 10
        pdf.set_x(x_body)
        pdf.multi_cell(pdf.w - pdf.r_margin - x_body, 5, _s(title))

        # Description
        if desc:
            pdf.set_x(x_body)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*GRAY_TEXT)
            pdf.multi_cell(pdf.w - pdf.r_margin - x_body, 4.5, _s(desc[:200]))

        # Meta line: impact + time
        if impact or time_est:
            pdf.set_x(x_body)
            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(*GRAY_TEXT)
            meta_parts = []
            if impact:
                meta_parts.append(impact)
            if time_est:
                meta_parts.append(time_est)
            pdf.cell(0, 4, _s(" | ".join(meta_parts)), ln=True)

        # Priority + pillar badges
        pdf.set_x(x_body)
        pdf.priority_badge(priority)
        pdf.cell(2, 4, "")
        pdf.pillar_badge(pillar)
        pdf.ln(4)

        # Divider
        pdf.set_draw_color(*GRAY_LINE)
        pdf.line(pdf.l_margin + 10, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(3)


def _pillar_page(pdf: SEOReport, pillar_key: str, color: tuple):
    """One page per pillar with score header + improvement steps."""
    pillars = pdf.audit.get("pillars", {})
    pillar = pillars.get(pillar_key)
    if not pillar:
        return

    pdf.add_page()

    score = pillar.get("score", 0)
    title = pillar.get("title", _PILLAR_LABELS.get(pillar_key, pillar_key))
    subtitle = pillar.get("subtitle", "")
    steps = pillar.get("steps", [])

    # Pillar header with colour stripe
    pdf.set_fill_color(*color)
    pdf.rect(pdf.l_margin, pdf.get_y(), 3, 12, "F")
    pdf.set_x(pdf.l_margin + 6)

    # Title + score on same line
    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(*NAVY)
    pdf.cell(100, 7, _s(title))

    # Score right-aligned
    s_color = _score_color(score)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*s_color)
    pdf.cell(0, 7, f"{score}/100", align="R", ln=True)

    # Subtitle
    if subtitle:
        pdf.set_x(pdf.l_margin + 6)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*GRAY_TEXT)
        pdf.cell(0, 5, _s(subtitle), ln=True)

    pdf.ln(6)

    # Improvement steps
    pdf.sub_heading("Improvement Steps")

    for step in steps:
        rank = step.get("rank", 0)
        s_title = step.get("title", "")
        s_desc = step.get("description", "")
        s_cat = step.get("category", "")
        s_priority = step.get("priority", "medium")
        s_time = step.get("time_estimate", "")

        if pdf.get_y() > pdf.h - 35:
            pdf.add_page()

        # Step number
        p_color = _priority_color(s_priority)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*p_color)
        pdf.cell(8, 6, str(rank))

        # Step title
        x_body = pdf.l_margin + 10
        pdf.set_x(x_body)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*NAVY)
        pdf.multi_cell(pdf.w - pdf.r_margin - x_body, 5, _s(s_title))

        # Description
        if s_desc:
            pdf.set_x(x_body)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*GRAY_TEXT)
            pdf.multi_cell(pdf.w - pdf.r_margin - x_body, 4.5, _ms(s_desc[:300]))

        # Tags line
        pdf.set_x(x_body)
        if s_cat:
            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(*GRAY_TEXT)
            pdf.set_fill_color(*GRAY_BG)
            pdf.cell(len(s_cat) * 3 + 6, 4, _s(s_cat), fill=True, align="C")
            pdf.cell(2, 4, "")
        pdf.priority_badge(s_priority)
        if s_time:
            pdf.cell(2, 4, "")
            pdf.set_font("Helvetica", "", 7)
            pdf.set_text_color(*GRAY_TEXT)
            pdf.set_fill_color(*GRAY_BG)
            pdf.cell(len(s_time) * 3 + 6, 4, _s(s_time), fill=True, align="C")
            pdf.set_fill_color(*WHITE)
        pdf.ln(4)

        # Divider
        pdf.set_draw_color(*GRAY_LINE)
        pdf.line(x_body, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(3)


def _cta_page(pdf: SEOReport):
    """Last page: CTA to upgrade."""
    pdf.add_page()

    # Centred content
    pdf.ln(30)

    # Brand
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*EMERALD)
    pdf.cell(0, 7, "LOCALRANK", align="C", ln=True)

    # Headline
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 10, "Track your SEO progress", align="C", ln=True)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*GRAY_TEXT)
    pdf.cell(0, 7, "Turn this report into a living action plan", align="C", ln=True)

    pdf.ln(10)

    # Feature comparison table
    col_w = 58
    start_x = (pdf.w - col_w * 3) / 2
    headers = ["", "Free", "Pro"]
    features = [
        ("PDF Report", "Yes", "Yes"),
        ("Score Dashboard", "-", "Yes"),
        ("Task Tracking", "-", "Yes"),
        ("AI Content Tools", "-", "Yes"),
        ("Monthly Re-audits", "-", "Yes"),
        ("Priority Support", "-", "Yes"),
    ]

    y = pdf.get_y()
    # Header row
    pdf.set_fill_color(*NAVY)
    pdf.set_text_color(*WHITE)
    pdf.set_font("Helvetica", "B", 9)
    for i, h in enumerate(headers):
        pdf.set_xy(start_x + i * col_w, y)
        pdf.cell(col_w, 7, h, fill=True, align="C")
    pdf.ln(7)

    # Feature rows
    for j, (feat, free, pro) in enumerate(features):
        bg = GRAY_BG if j % 2 == 0 else WHITE
        pdf.set_fill_color(*bg)
        row_y = pdf.get_y()

        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*NAVY)
        pdf.set_xy(start_x, row_y)
        pdf.cell(col_w, 6, _s(f"  {feat}"), fill=True)

        pdf.set_text_color(*GRAY_TEXT)
        pdf.set_xy(start_x + col_w, row_y)
        pdf.cell(col_w, 6, free, fill=True, align="C")

        pdf.set_text_color(*EMERALD)
        pdf.set_xy(start_x + 2 * col_w, row_y)
        pdf.cell(col_w, 6, pro, fill=True, align="C")
        pdf.ln(6)

    pdf.ln(8)

    # CTA button
    btn_w = 60
    btn_x = (pdf.w - btn_w) / 2
    pdf.set_fill_color(*EMERALD)
    pdf.rect(btn_x, pdf.get_y(), btn_w, 10, "F")
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*WHITE)
    pdf.set_xy(btn_x, pdf.get_y())
    pdf.cell(btn_w, 10, "Upgrade to Pro", align="C")
    pdf.ln(14)

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(*GRAY_TEXT)
    pdf.cell(0, 5, "localrank.com", align="C")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_pdf(audit: dict) -> bytes:
    """
    Build a PDF report from a completed audit dict.
    Returns raw PDF bytes ready to send as an HTTP response.
    """
    pdf = SEOReport(audit)
    pdf.alias_nb_pages()

    # Page 1: Cover
    _cover(pdf)

    # Page 2: Score overview
    _score_overview(pdf)

    # Pages 3-4: Quick wins
    _quick_wins(pdf)

    # Pages 5-8: One page per pillar
    _pillar_page(pdf, "website_seo", BLUE)
    _pillar_page(pdf, "backlinks", ROSE)
    _pillar_page(pdf, "local_seo", AMBER)
    _pillar_page(pdf, "ai_seo", VIOLET)

    # Last page: CTA
    _cta_page(pdf)

    return bytes(pdf.output())
