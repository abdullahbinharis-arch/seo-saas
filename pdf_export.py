"""
pdf_export.py — Generate a branded PDF report from a completed SEO audit.

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
    "\u00b7": ".",     # middle dot (already Latin-1 but keep for safety)
    "\u00e9": "e",     # é
    "\u2192": "->",    # arrow
}

def _s(text) -> str:
    """Return a Latin-1-safe, single-line string for fpdf cell() calls."""
    t = str(text)
    for char, repl in _REPLACEMENTS.items():
        t = t.replace(char, repl)
    # Collapse newlines / tabs to a space so cell() doesn't choke
    t = t.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    # Final pass: drop anything still outside Latin-1
    return t.encode("latin-1", errors="replace").decode("latin-1")


def _ms(text) -> str:
    """Latin-1-safe string that preserves newlines for multi_cell()."""
    t = str(text)
    for char, repl in _REPLACEMENTS.items():
        t = t.replace(char, repl)
    t = t.replace("\r", "").replace("\t", " ")
    return t.encode("latin-1", errors="replace").decode("latin-1")

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
NAVY       = (15,  23,  42)   # headings / cover
BLUE       = (37,  99, 235)   # section titles
LIGHT_BLUE = (219, 234, 254)  # section title bg
GRAY_BG    = (248, 250, 252)  # table row bg
GRAY_LINE  = (226, 232, 240)  # dividers
GRAY_TEXT  = (100, 116, 139)  # secondary text
GREEN      = (22, 163,  74)   # pass
AMBER      = (217, 119,   6)  # warn
RED        = (220,  38,  38)  # fail
WHITE      = (255, 255, 255)


# ---------------------------------------------------------------------------
# PDF subclass with helpers
# ---------------------------------------------------------------------------

class SEOReport(FPDF):
    def __init__(self, audit: dict):
        super().__init__()
        self.audit = audit
        self.set_margins(18, 18, 18)
        self.set_auto_page_break(auto=True, margin=22)

    # ── Header / footer ────────────────────────────────────────────────────

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*GRAY_TEXT)
        self.cell(0, 6, _s(f"SEO Audit Report  |  {self.audit.get('target_url', '')}"),
                  align="L")
        self.ln(1)
        self.set_draw_color(*GRAY_LINE)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-16)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*GRAY_TEXT)
        self.cell(0, 6, f"Page {self.page_no()} of {{nb}}", align="C")

    # ── Low-level drawing primitives ────────────────────────────────────────

    def rule(self):
        self.set_draw_color(*GRAY_LINE)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def section_title(self, tag: str, title: str):
        self.ln(4)
        self.set_fill_color(*LIGHT_BLUE)
        self.set_text_color(*BLUE)
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 9, _s(f"  [{tag}]  {title}"), fill=True, ln=True)
        self.ln(3)

    def sub_heading(self, text: str):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*NAVY)
        self.cell(0, 6, _s(text), ln=True)
        self.ln(1)

    def body(self, text: str, indent: int = 0):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*NAVY)
        if indent:
            self.set_x(self.l_margin + indent)
        self.multi_cell(0, 5, _s(text))
        self.set_x(self.l_margin)

    def small(self, text: str):
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*GRAY_TEXT)
        self.multi_cell(0, 5, _s(text))
        self.set_x(self.l_margin)

    def bullet(self, text: str, color=NAVY, indent: int = 4):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*color)
        x = self.l_margin + indent
        self.set_x(x)
        self.set_font("Helvetica", "B", 9)
        self.cell(4, 5, "-")
        self.set_font("Helvetica", "", 9)
        self.multi_cell(self.w - self.r_margin - x - 4, 5, _s(text))
        self.set_x(self.l_margin)

    def numbered(self, n: int, text: str):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*NAVY)
        self.set_x(self.l_margin + 4)
        self.cell(6, 5, f"{n}.")
        self.multi_cell(self.w - self.r_margin - self.l_margin - 10, 5, _s(text))
        self.set_x(self.l_margin)

    def status_badge(self, status: str):
        """Inline coloured badge: pass / warn / fail."""
        s = (status or "").lower()
        if s == "pass":
            color, label = GREEN, "PASS"
        elif s == "warn":
            color, label = AMBER, "WARN"
        else:
            color, label = RED, "FAIL"
        self.set_font("Helvetica", "B", 7)
        self.set_text_color(*WHITE)
        self.set_fill_color(*color)
        self.cell(12, 4, label, fill=True, align="C")
        self.set_text_color(*NAVY)
        self.set_fill_color(*WHITE)

    def kv(self, key: str, value: str):
        """Key: value line."""
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*GRAY_TEXT)
        self.cell(40, 5, _s(f"{key}:"))
        self.set_font("Helvetica", "", 9)
        self.set_text_color(*NAVY)
        self.multi_cell(0, 5, _s(value))
        self.set_x(self.l_margin)

    def meta_box(self, label: str, value: str):
        """Left-stripe coloured box for meta recommendations."""
        self.set_font("Helvetica", "B", 7)
        self.set_text_color(*GRAY_TEXT)
        self.cell(0, 5, _s(label.upper()), ln=True)
        y = self.get_y()
        x = self.l_margin
        w = self.w - self.l_margin - self.r_margin
        self.set_fill_color(*BLUE)
        self.rect(x, y, 2, 10, "F")
        self.set_fill_color(*GRAY_BG)
        self.rect(x + 2, y, w - 2, 10, "F")
        self.set_xy(x + 5, y + 1)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*NAVY)
        self.multi_cell(w - 7, 4.5, _s(value))
        self.set_x(self.l_margin)
        self.ln(2)

    def score_circle(self, score: int, label: str = "Technical Score"):
        """Large score indicator."""
        cx = self.l_margin + 18
        cy = self.get_y() + 12
        c = GREEN if score >= 8 else AMBER if score >= 5 else RED
        self.set_fill_color(*c)
        self.ellipse(cx - 12, cy - 10, 24, 20, "F")
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(*WHITE)
        self.set_xy(cx - 12, cy - 5)
        self.cell(24, 10, str(score), align="C")
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*GRAY_TEXT)
        self.set_xy(cx + 15, cy - 3)
        self.cell(0, 5, _s(f"/ 10  {label}"))
        self.set_xy(self.l_margin, cy + 14)

    def tech_check_row(self, label: str, status: str, detail: str):
        y = self.get_y()
        if int(y) % 2 == 0:
            self.set_fill_color(*GRAY_BG)
            self.rect(self.l_margin, y, self.w - self.l_margin - self.r_margin, 7, "F")
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*NAVY)
        self.cell(45, 7, _s(label))
        self.status_badge(status)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*GRAY_TEXT)
        self.set_x(self.l_margin + 60)
        self.multi_cell(0, 7, _s(detail)[:120])
        self.set_x(self.l_margin)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _cover(pdf: SEOReport):
    pdf.add_page()

    # Dark banner
    pdf.set_fill_color(*NAVY)
    pdf.rect(0, 0, pdf.w, 68, "F")

    pdf.set_xy(18, 18)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(*WHITE)
    pdf.cell(0, 10, "SEO Audit Report", ln=True)

    pdf.set_x(18)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(148, 163, 184)
    url = _s(pdf.audit.get("target_url", ""))
    pdf.cell(0, 8, url, ln=True)

    pdf.set_xy(18, 72)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(*GRAY_TEXT)

    ts = pdf.audit.get("timestamp", "")
    try:
        dt = datetime.fromisoformat(ts)
        date_str = dt.strftime("%B %d, %Y")
    except Exception:
        date_str = ts[:10]

    meta_items = [
        ("Keyword",   pdf.audit.get("keyword", "")),
        ("Location",  pdf.audit.get("location", "")),
        ("Date",      date_str),
        ("Agents",    str(pdf.audit.get("agents_executed", 4))),
        ("Time",      f"{pdf.audit.get('execution_time_seconds', 0)}s"),
        ("Est. cost", f"${pdf.audit.get('summary', {}).get('estimated_api_cost', 0):.3f}"),
        ("Audit ID",  pdf.audit.get("audit_id", "")[:16] + "..."),
    ]
    for key, val in meta_items:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(*GRAY_TEXT)
        pdf.cell(32, 6, _s(f"{key}:"))
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*NAVY)
        pdf.cell(0, 6, _s(str(val)), ln=True)

    pdf.ln(6)
    pdf.rule()


def _quick_wins(pdf: SEOReport):
    wins = pdf.audit.get("summary", {}).get("quick_wins", [])
    if not wins:
        return
    pdf.section_title("!", "Quick Wins")
    for i, win in enumerate(wins, 1):
        pdf.numbered(i, win)
        pdf.ln(1)
    pdf.ln(2)


def _keyword_research(pdf: SEOReport):
    agent = pdf.audit.get("agents", {}).get("keyword_research", {})
    rec   = agent.get("recommendations", {})
    if not rec:
        return

    pdf.add_page()
    pdf.section_title("KW", "Keyword Research")
    pdf.kv("Primary keyword",       rec.get("primary_keyword", ""))
    pdf.kv("Competitors analysed",  str(agent.get("competitors_analyzed", "")))
    pdf.ln(3)

    # High-intent keywords table
    keywords = rec.get("high_intent_keywords", [])
    if keywords:
        pdf.sub_heading("High-Intent Keywords")
        col_w = [80, 30, 38, 26]
        headers = ["Keyword", "Intent", "Searches/mo", "Difficulty"]
        # header row
        pdf.set_fill_color(*NAVY)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", 8)
        for i, h in enumerate(headers):
            pdf.cell(col_w[i], 6, h, fill=True, border=0)
        pdf.ln()
        # data rows
        for j, kw in enumerate(keywords[:12]):
            bg = GRAY_BG if j % 2 == 0 else WHITE
            pdf.set_fill_color(*bg)
            pdf.set_text_color(*NAVY)
            pdf.set_font("Helvetica", "", 8)
            pdf.cell(col_w[0], 6, _s(str(kw.get("keyword", ""))[:40]), fill=True)
            pdf.cell(col_w[1], 6, _s(str(kw.get("intent", ""))), fill=True)
            pdf.cell(col_w[2], 6,
                     f"{kw.get('estimated_monthly_searches', 0):,}", fill=True)
            diff  = str(kw.get("difficulty", "")).lower()
            color = GREEN if diff == "low" else AMBER if diff == "medium" else RED
            pdf.set_text_color(*color)
            pdf.cell(col_w[3], 6, _s(diff.capitalize()), fill=True)
            pdf.set_text_color(*NAVY)
            pdf.ln()
        pdf.ln(3)

    # Long-tail
    long_tail = rec.get("long_tail_keywords", [])
    if long_tail:
        pdf.sub_heading("Long-Tail Opportunities")
        for lt in long_tail[:8]:
            pdf.bullet(lt, color=BLUE)
        pdf.ln(2)

    # Competitor gaps
    gaps = rec.get("competitor_keywords_we_miss", [])
    if gaps:
        pdf.sub_heading("Competitor Keyword Gaps")
        for g in gaps:
            pdf.bullet(g, color=AMBER)
        pdf.ln(2)

    # Clusters
    clusters = rec.get("keyword_clusters", [])
    if clusters:
        pdf.sub_heading("Keyword Clusters")
        for c in clusters[:5]:
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*NAVY)
            pdf.cell(0, 5, _s(c.get("theme", "")), ln=True)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*GRAY_TEXT)
            pdf.set_x(pdf.l_margin + 6)
            pdf.multi_cell(0, 4.5, _s(" / ".join(c.get("keywords", []))))
            pdf.set_x(pdf.l_margin)
            pdf.ln(1)
        pdf.ln(2)

    # Strategy note
    strategy = rec.get("recommendation", "")
    if strategy:
        pdf.sub_heading("Strategy")
        pdf.set_fill_color(*LIGHT_BLUE)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(*NAVY)
        pdf.multi_cell(0, 5, _ms(strategy), border=0)
        pdf.ln(2)


def _on_page_seo(pdf: SEOReport):
    agent = pdf.audit.get("agents", {}).get("on_page_seo", {})
    rec   = agent.get("recommendations", {})
    if not rec:
        return

    pdf.add_page()
    pdf.section_title("ON", "On-Page SEO")

    current = rec.get("current_analysis", {})
    recs    = rec.get("recommendations", {})

    # Score badge inline
    score = current.get("seo_score", 0)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*GRAY_TEXT)
    pdf.cell(30, 6, "SEO Score:")
    s_color = GREEN if score >= 7 else AMBER if score >= 4 else RED
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(*s_color)
    pdf.cell(0, 6, f"{score} / 10", ln=True)
    pdf.ln(1)

    # Current state
    pdf.sub_heading("Current Page State")
    pdf.kv("Title tag",    current.get("title", "N/A"))
    pdf.kv("Meta desc",    current.get("meta_description", "N/A"))
    pdf.kv("H1",           current.get("h1", "N/A"))
    pdf.kv("Word count",   str(current.get("word_count", 0)))
    pdf.ln(2)

    issues = current.get("issues_found", [])
    if issues:
        pdf.sub_heading("Issues Found")
        for issue in issues:
            pdf.bullet(issue, color=RED)
        pdf.ln(2)

    # Recommendations
    if recs:
        pdf.sub_heading("Recommended Tags")
        if recs.get("meta_title"):
            pdf.meta_box("Title Tag", recs["meta_title"])
        if recs.get("meta_description"):
            pdf.meta_box("Meta Description", recs["meta_description"])
        if recs.get("h1"):
            pdf.meta_box("H1 Heading", recs["h1"])
        if recs.get("target_word_count"):
            pdf.kv("Target word count", f"{recs['target_word_count']:,} words "
                   f"(currently {current.get('word_count', 0):,})")
        pdf.ln(2)

    # Priority actions
    actions = rec.get("priority_actions", [])
    if actions:
        pdf.sub_heading("Priority Actions")
        for i, action in enumerate(actions, 1):
            pdf.numbered(i, action)
            pdf.ln(1)
        pdf.ln(2)

    # Internal links
    links = rec.get("internal_links", [])
    if links:
        pdf.sub_heading("Internal Links to Add")
        for link in links[:5]:
            anchor = link.get("anchor_text", "")
            path   = link.get("target_path", "")
            reason = link.get("reason", "")
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*BLUE)
            pdf.cell(0, 5, _s(f'"{anchor}"  ->  {path}'), ln=True)
            pdf.small(reason)
            pdf.ln(1)
        pdf.ln(2)


def _local_seo(pdf: SEOReport):
    agent = pdf.audit.get("agents", {}).get("local_seo", {})
    rec   = agent.get("recommendations", {})
    if not rec:
        return

    pdf.add_page()
    pdf.section_title("LO", "Local SEO")

    # Quick wins
    qw = rec.get("quick_wins", [])
    if qw:
        pdf.sub_heading("Quick Wins")
        for w in qw:
            pdf.bullet(w, color=GREEN)
        pdf.ln(2)

    # GBP
    gbp = rec.get("gbp_optimization", {})
    if gbp:
        pdf.sub_heading("Google Business Profile")
        attrs = gbp.get("priority_attributes", [])
        cats  = gbp.get("categories", [])
        if attrs:
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*GRAY_TEXT)
            pdf.cell(0, 5, "Priority attributes:", ln=True)
            pdf.body(" / ".join(attrs), indent=4)
        if cats:
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*GRAY_TEXT)
            pdf.cell(0, 5, "Categories:", ln=True)
            pdf.body(" / ".join(cats), indent=4)
        if gbp.get("photo_strategy"):
            pdf.ln(1)
            pdf.small(f"Photo strategy: {gbp['photo_strategy']}")
        rs = gbp.get("review_strategy", {})
        if rs.get("target_reviews_per_month"):
            pdf.small(f"Review target: {rs['target_reviews_per_month']} reviews/month")
        pdf.ln(2)

    # Citations table
    citations = rec.get("citations", [])
    if citations:
        pdf.sub_heading("Citations to Build")
        col_w = [80, 52, 42]
        pdf.set_fill_color(*NAVY)
        pdf.set_text_color(*WHITE)
        pdf.set_font("Helvetica", "B", 8)
        for h, w in zip(["Directory", "Category", "Priority"], col_w):
            pdf.cell(w, 6, h, fill=True)
        pdf.ln()
        for j, c in enumerate(citations[:10]):
            bg = GRAY_BG if j % 2 == 0 else WHITE
            pdf.set_fill_color(*bg)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(*NAVY)
            pdf.cell(col_w[0], 6, str(c.get("site", ""))[:35], fill=True)
            pdf.cell(col_w[1], 6, str(c.get("category", "")), fill=True)
            pri = str(c.get("priority", "")).lower()
            p_color = RED if pri == "critical" else AMBER if pri == "high" else NAVY
            pdf.set_text_color(*p_color)
            pdf.cell(col_w[2], 6, pri.capitalize(), fill=True)
            pdf.set_text_color(*NAVY)
            pdf.ln()
        pdf.ln(3)

    # Link opportunities
    opps = rec.get("link_opportunities", [])
    if opps:
        pdf.sub_heading("Link Building Opportunities")
        for opp in opps[:5]:
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*NAVY)
            name = opp.get("name", "")
            ltype = opp.get("link_type", "")
            pdf.cell(0, 5, f"{name}  [{ltype}]", ln=True)
            pdf.small(opp.get("reason", ""))
            template = opp.get("outreach_template", "")
            if template:
                pdf.set_font("Helvetica", "I", 7)
                pdf.set_text_color(*GRAY_TEXT)
                pdf.set_x(pdf.l_margin + 4)
                pdf.multi_cell(0, 4, _s(f'"{template[:140]}"'))
                pdf.set_x(pdf.l_margin)
            pdf.ln(1)
        pdf.ln(2)

    # Content strategy
    cs = rec.get("local_content_strategy", {})
    topics = cs.get("blog_topics", [])
    areas  = cs.get("service_area_pages", [])
    if topics or areas:
        pdf.sub_heading("Content Strategy")
        if topics:
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*GRAY_TEXT)
            pdf.cell(0, 5, "Blog topics:", ln=True)
            for t in topics[:6]:
                pdf.bullet(t)
            pdf.ln(1)
        if areas:
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(*GRAY_TEXT)
            pdf.cell(0, 5, "Service area pages:", ln=True)
            pdf.body(" / ".join(areas), indent=4)
        pdf.ln(2)

    impact = rec.get("estimated_impact", "")
    if impact:
        pdf.set_fill_color(*LIGHT_BLUE)
        pdf.set_font("Helvetica", "BI", 8)
        pdf.set_text_color(*BLUE)
        pdf.multi_cell(0, 5, _ms(f"Estimated impact: {impact}"))
        pdf.ln(2)


def _technical_seo(pdf: SEOReport):
    agent = pdf.audit.get("agents", {}).get("technical_seo", {})
    rec   = agent.get("recommendations", {})
    if not rec:
        return

    pdf.add_page()
    pdf.section_title("TX", "Technical SEO")

    score = rec.get("technical_score", 0)
    pdf.score_circle(score)
    pdf.ln(2)

    # Check table
    checks = [
        ("HTTPS",           rec.get("https",           {}).get("status"), rec.get("https",           {}).get("detail", "")),
        ("Mobile viewport", rec.get("mobile",          {}).get("status"), rec.get("mobile",          {}).get("recommendation", "")),
        ("Canonical",       rec.get("canonical",       {}).get("status"), rec.get("canonical",       {}).get("recommendation", "")),
        ("Robots meta",     rec.get("robots",          {}).get("status"), rec.get("robots",          {}).get("recommendation", "")),
        ("Structured data", rec.get("structured_data", {}).get("status"), rec.get("structured_data", {}).get("recommendation", "")),
        ("Images / alt",    rec.get("images",          {}).get("status"), rec.get("images",          {}).get("recommendation", "")),
        ("Page speed",      rec.get("page_speed",      {}).get("status"), rec.get("page_speed",      {}).get("recommendation", "")),
    ]
    pdf.sub_heading("Technical Checks")
    for label, status, detail in checks:
        pdf.tech_check_row(label, status or "warn", detail)
    pdf.ln(3)

    # Structured data detail
    sd = rec.get("structured_data", {})
    found  = sd.get("schemas_found", [])
    to_add = sd.get("schemas_to_add", [])
    if found or to_add:
        pdf.sub_heading("Structured Data")
        if found:
            pdf.kv("Schemas found",  " / ".join(found))
        if to_add:
            pdf.kv("Schemas to add", " / ".join(to_add))
        pdf.ln(2)

    # Page speed detail
    ps = rec.get("page_speed", {})
    ps_issues = ps.get("issues", [])
    if ps_issues:
        pdf.sub_heading("Page Speed Issues")
        for issue in ps_issues:
            pdf.bullet(issue, color=RED)
        pdf.ln(2)

    # Priority fixes
    fixes = rec.get("priority_fixes", [])
    if fixes:
        pdf.sub_heading("Priority Fixes")
        for i, fix in enumerate(fixes, 1):
            pdf.numbered(i, fix)
            pdf.ln(1)
        pdf.ln(2)


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

    _cover(pdf)
    _quick_wins(pdf)
    _keyword_research(pdf)
    _on_page_seo(pdf)
    _local_seo(pdf)
    _technical_seo(pdf)

    return bytes(pdf.output())
