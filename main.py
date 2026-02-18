# =============================================================================
# SEO SaaS Platform — FastAPI Backend
# =============================================================================
# 3-Agent Local SEO Audit System
#
# Agents:
#   1. Keyword Research  — competitor gaps, intent mapping, clusters
#   2. On-Page SEO       — content audit, meta tags, internal links
#   3. Local SEO         — GBP optimization, citations, link building
#
# Run:  python main.py
# Test: curl http://localhost:8000/health
# =============================================================================

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()                       # ← CRITICAL: reads .env into os.environ

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("seo-saas")

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")

if not ANTHROPIC_API_KEY:
    logger.warning("⚠️  ANTHROPIC_API_KEY is not set — Claude calls will fail")
if not SERPAPI_KEY:
    logger.warning("⚠️  SERPAPI_KEY is not set — competitor research will fail")

# ---------------------------------------------------------------------------
# Anthropic client — ASYNC so we never block the event loop
# ---------------------------------------------------------------------------

from anthropic import AsyncAnthropic          # requires anthropic >= 0.39

anthropic_client = AsyncAnthropic()           # reads ANTHROPIC_API_KEY from env

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

from database import Audit, SessionLocal, init_db

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SEO SaaS API",
    version="1.0.0",
    description="3-Agent Local SEO Audit Platform",
)

# CORS — open for development, lock down for production
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,https://seo-frontend-six.vercel.app",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    init_db()
    logger.info("Database tables ready")


# ---------------------------------------------------------------------------
# Simple in-memory rate limiter (swap for Redis in production)
# ---------------------------------------------------------------------------

_rate_buckets: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "10"))


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path.startswith(("/health", "/info", "/docs", "/openapi")):
        return await call_next(request)

    ip = request.client.host if request.client else "unknown"
    now = time.time()
    _rate_buckets[ip] = [t for t in _rate_buckets[ip] if now - t < 60]

    if len(_rate_buckets[ip]) >= RATE_LIMIT:
        raise HTTPException(429, "Rate limit exceeded — try again in a minute")

    _rate_buckets[ip].append(now)
    return await call_next(request)


# =============================================================================
# Request / response models
# =============================================================================

class AuditRequest(BaseModel):
    keyword: str
    target_url: str
    location: str = "Toronto, Canada"
    user_id: Optional[str] = None

    @field_validator("keyword")
    @classmethod
    def keyword_valid(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("Keyword must be at least 2 characters")
        if len(v) > 200:
            raise ValueError("Keyword must be under 200 characters")
        return v

    @field_validator("target_url")
    @classmethod
    def url_valid(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v


# =============================================================================
# Utility functions
# =============================================================================

def extract_json(text: str) -> dict:
    """
    Robustly extract JSON from Claude responses.
    Handles markdown fences, preamble text, and trailing commentary.
    """
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0].strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the outermost balanced braces
    start = text.find("{")
    if start == -1:
        return {"raw_response": text}

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    break

    return {"raw_response": text}


# ---------------------------------------------------------------------------
# Web scraping helpers
# ---------------------------------------------------------------------------

import httpx
from bs4 import BeautifulSoup


async def scrape_page(url: str) -> dict:
    """Scrape a page and extract structured SEO data."""
    try:
        async with httpx.AsyncClient(
            timeout=12.0,
            follow_redirects=True,
            headers={"User-Agent": "SEOSaasBot/1.0"},
        ) as http:
            resp = await http.get(url)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "noscript", "iframe"]):
            tag.decompose()

        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        meta_desc_tag = soup.find("meta", attrs={"name": "description"})
        meta_desc = meta_desc_tag["content"].strip() if meta_desc_tag and meta_desc_tag.get("content") else ""

        h1 = soup.find("h1")
        h1_text = h1.get_text(strip=True) if h1 else ""

        headings = []
        for level in ["h1", "h2", "h3", "h4"]:
            for tag in soup.find_all(level):
                headings.append({"level": level, "text": tag.get_text(strip=True)})

        body_text = soup.get_text(separator=" ", strip=True)

        internal_links = []
        external_links = []
        from urllib.parse import urlparse
        base_domain = urlparse(url).netloc
        for a in soup.find_all("a", href=True):
            href = a["href"]
            anchor = a.get_text(strip=True)
            if href.startswith("/") or base_domain in href:
                internal_links.append({"href": href, "anchor": anchor})
            elif href.startswith("http"):
                external_links.append({"href": href, "anchor": anchor})

        return {
            "url": url,
            "success": True,
            "title": title,
            "meta_description": meta_desc,
            "h1": h1_text,
            "headings": headings[:30],
            "word_count": len(body_text.split()),
            "content": body_text[:4000],
            "internal_links_count": len(internal_links),
            "external_links_count": len(external_links),
            "internal_links": internal_links[:20],
        }

    except Exception as e:
        logger.warning(f"Scrape failed for {url}: {e}")
        return {"url": url, "success": False, "error": str(e)}


async def fetch_competitors(keyword: str, location: str) -> list[dict]:
    """Fetch top organic competitors from SerpApi."""
    if not SERPAPI_KEY:
        logger.error("SERPAPI_KEY not set — returning empty competitors")
        return []

    try:
        async with httpx.AsyncClient(timeout=15.0) as http:
            resp = await http.get(
                "https://serpapi.com/search",
                params={
                    "q": keyword,
                    "location": location,
                    "api_key": SERPAPI_KEY,
                    "num": 5,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        competitors = []
        for r in data.get("organic_results", [])[:5]:
            competitors.append({
                "url": r.get("link", ""),
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "position": r.get("position", 0),
            })
        return competitors

    except Exception as e:
        logger.error(f"SerpApi error: {e}")
        return []


async def scrape_competitors(competitors: list[dict]) -> str:
    """Scrape top 3 competitor pages and build a summary string for Claude."""
    tasks = [scrape_page(c["url"]) for c in competitors[:3]]
    pages = await asyncio.gather(*tasks)

    summaries = []
    for comp, page in zip(competitors[:3], pages):
        if page.get("success"):
            summaries.append(
                f"---\n"
                f"Competitor #{comp['position']}: {comp['title']}\n"
                f"URL: {comp['url']}\n"
                f"Title tag: {page.get('title', 'N/A')}\n"
                f"H1: {page.get('h1', 'N/A')}\n"
                f"Meta description: {page.get('meta_description', 'N/A')}\n"
                f"Word count: {page.get('word_count', 0)}\n"
                f"Headings: {', '.join(h['text'] for h in page.get('headings', [])[:8])}\n"
                f"Content preview: {page.get('content', '')[:600]}\n"
            )
    return "\n".join(summaries) if summaries else "No competitor data available."


# ---------------------------------------------------------------------------
# Claude helper — centralised, with retry
# ---------------------------------------------------------------------------

async def call_claude(
    system: str,
    prompt: str,
    max_tokens: int = 2000,
    retries: int = 2,
) -> dict:
    """
    Call Claude with a system prompt and user prompt.
    Retries on transient failures. Returns parsed JSON dict.
    """
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = await anthropic_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "{"},   # force JSON start
                ],
            )
            raw = "{" + response.content[0].text
            return extract_json(raw)

        except Exception as e:
            last_error = e
            logger.warning(f"Claude call attempt {attempt} failed: {e}")
            if attempt < retries:
                await asyncio.sleep(2 * attempt)

    logger.error(f"Claude call failed after {retries} attempts: {last_error}")
    return {"error": str(last_error)}


# =============================================================================
# AGENT 1 — Keyword Research
# =============================================================================

KEYWORD_SYSTEM = """You are an expert SEO keyword researcher specialising in local search.
You ALWAYS respond with valid JSON only — no markdown, no explanation, no preamble.
Your analysis is data-driven, specific, and actionable."""

KEYWORD_PROMPT = """Analyse keyword opportunities.

TARGET KEYWORD: {keyword}
LOCATION: {location}
TARGET URL: {target_url}

COMPETITOR ANALYSIS:
{competitor_data}

Return JSON with EXACTLY these keys:
{{
  "primary_keyword": "the main keyword",
  "high_intent_keywords": [
    {{"keyword": "...", "intent": "commercial|informational|navigational|transactional", "estimated_monthly_searches": 0, "difficulty": "low|medium|high", "local_modifier": "..."}}
  ],
  "long_tail_keywords": ["phrase 1", "phrase 2"],
  "competitor_keywords_we_miss": ["keyword 1", "keyword 2"],
  "keyword_clusters": [
    {{"theme": "...", "keywords": ["...", "..."]}}
  ],
  "content_gap_opportunities": ["topic 1", "topic 2"],
  "recommendation": "One paragraph summary of the keyword strategy"
}}

Provide at least 15 high-intent keywords, 8 long-tail, and 5 competitor gaps."""


@app.post("/agents/keyword-research")
async def keyword_research_agent(request: AuditRequest):
    """Keyword Research Agent — competitor gaps, intent mapping, clusters."""
    audit_id = str(uuid.uuid4())
    logger.info(f"[{audit_id}] Keyword Research starting for '{request.keyword}'")

    competitors = await fetch_competitors(request.keyword, request.location)
    competitor_data = await scrape_competitors(competitors)

    prompt = KEYWORD_PROMPT.format(
        keyword=request.keyword,
        location=request.location,
        target_url=request.target_url,
        competitor_data=competitor_data,
    )

    recommendations = await call_claude(KEYWORD_SYSTEM, prompt, max_tokens=2500)

    return {
        "agent": "keyword_research",
        "audit_id": audit_id,
        "status": "completed",
        "keyword": request.keyword,
        "competitors_analyzed": len(competitors),
        "recommendations": recommendations,
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# AGENT 2 — On-Page SEO
# =============================================================================

ONPAGE_SYSTEM = """You are an expert on-page SEO specialist with deep knowledge of Google ranking factors.
You ALWAYS respond with valid JSON only — no markdown, no explanation, no preamble.
Your recommendations are specific, measurable, and prioritised."""

ONPAGE_PROMPT = """Analyse and optimise this page for the target keyword.

TARGET KEYWORD: {keyword}
TARGET URL: {target_url}
LOCATION: {location}

CURRENT PAGE DATA:
- Title tag: {title}
- Meta description: {meta_description}
- H1: {h1}
- Word count: {word_count}
- Headings: {headings}
- Content preview: {content}

COMPETITOR PAGES (Top 3):
{competitor_data}

Return JSON with EXACTLY these keys:
{{
  "current_analysis": {{
    "title": "current title",
    "meta_description": "current meta desc",
    "h1": "current h1",
    "word_count": 0,
    "seo_score": 0,
    "issues_found": ["issue 1", "issue 2"]
  }},
  "recommendations": {{
    "meta_title": "New 60-char title with keyword",
    "meta_description": "New 155-char description with keyword and CTA",
    "h1": "New H1 with primary keyword",
    "target_word_count": 0,
    "heading_structure": ["H2: ...", "H3: ...", "H2: ..."],
    "keywords_to_add": ["keyword 1", "keyword 2"],
    "content_sections_to_add": ["section topic 1", "section topic 2"],
    "schema_markup": "Type of schema to add (LocalBusiness, FAQPage, etc.)"
  }},
  "internal_links": [
    {{"anchor_text": "...", "target_path": "/page", "reason": "why this link helps"}}
  ],
  "priority_actions": ["Most important fix first", "Second most important"],
  "priority_score": 0
}}

Be specific — write the actual meta title, not a template."""


@app.post("/agents/on-page-seo")
async def on_page_seo_agent(request: AuditRequest):
    """On-Page SEO Agent — content audit, meta tags, structure, internal links."""
    audit_id = str(uuid.uuid4())
    logger.info(f"[{audit_id}] On-Page SEO starting for '{request.target_url}'")

    # Scrape target + competitors concurrently
    page_task = scrape_page(request.target_url)
    comp_task = fetch_competitors(request.keyword, request.location)
    page_data, competitors = await asyncio.gather(page_task, comp_task)

    competitor_data = await scrape_competitors(competitors)

    prompt = ONPAGE_PROMPT.format(
        keyword=request.keyword,
        target_url=request.target_url,
        location=request.location,
        title=page_data.get("title", "N/A"),
        meta_description=page_data.get("meta_description", "N/A"),
        h1=page_data.get("h1", "N/A"),
        word_count=page_data.get("word_count", 0),
        headings=json.dumps(page_data.get("headings", [])[:15]),
        content=page_data.get("content", "")[:2000],
        competitor_data=competitor_data,
    )

    recommendations = await call_claude(ONPAGE_SYSTEM, prompt, max_tokens=2500)

    return {
        "agent": "on_page_seo",
        "audit_id": audit_id,
        "status": "completed",
        "keyword": request.keyword,
        "target_url": request.target_url,
        "page_scraped": page_data.get("success", False),
        "recommendations": recommendations,
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# AGENT 3 — Local SEO
# =============================================================================

LOCAL_SYSTEM = """You are a local SEO specialist focused on Google Map Pack rankings and local search dominance.
You ALWAYS respond with valid JSON only — no markdown, no explanation, no preamble.
Your strategies are specific to the business location and industry."""

LOCAL_PROMPT = """Create a local SEO strategy for this business.

TARGET KEYWORD: {keyword}
LOCATION: {location}
TARGET URL: {target_url}

TARGET PAGE INFO:
- Title: {title}
- H1: {h1}
- Content preview: {content}

TOP COMPETITORS:
{competitor_names}

Return JSON with EXACTLY these keys:
{{
  "gbp_optimization": {{
    "priority_attributes": ["attribute to complete 1", "attribute 2"],
    "categories": ["Primary category", "Secondary category"],
    "photo_strategy": "Specific photo recommendations with numbers",
    "post_strategy": "How often to post and what topics",
    "review_strategy": {{
      "target_reviews_per_month": 0,
      "review_request_template": "Short template text",
      "response_template": "Template for replying to reviews"
    }},
    "q_and_a": ["Question to seed 1", "Question 2", "Question 3"]
  }},
  "citations": [
    {{"site": "Site name", "url": "https://...", "priority": "critical|high|medium", "category": "general|industry|local"}}
  ],
  "link_opportunities": [
    {{"name": "Site or org name", "url": "https://...", "link_type": "directory|guest-post|sponsorship|resource", "reason": "Why this is valuable", "outreach_template": "Short outreach message"}}
  ],
  "local_content_strategy": {{
    "blog_topics": ["Topic 1", "Topic 2"],
    "service_area_pages": ["Area 1", "Area 2"],
    "faq_questions": ["Question 1", "Question 2"]
  }},
  "nap_checklist": ["Ensure Name/Address/Phone matches on Google", "Check Yelp listing", "Verify Facebook page"],
  "quick_wins": ["Fastest win 1", "Fastest win 2", "Fastest win 3"],
  "estimated_impact": "Summary of expected impact from these changes"
}}

Provide at least 8 citations, 5 link opportunities, and 5 blog topics."""


@app.post("/agents/local-seo")
async def local_seo_agent(request: AuditRequest):
    """Local SEO Agent — GBP, citations, link building, local content."""
    audit_id = str(uuid.uuid4())
    logger.info(f"[{audit_id}] Local SEO starting for '{request.keyword}' in {request.location}")

    page_task = scrape_page(request.target_url)
    comp_task = fetch_competitors(request.keyword, request.location)
    page_data, competitors = await asyncio.gather(page_task, comp_task)

    competitor_names = "\n".join(
        f"#{c['position']}: {c['title']} — {c['url']}"
        for c in competitors[:5]
    )

    prompt = LOCAL_PROMPT.format(
        keyword=request.keyword,
        location=request.location,
        target_url=request.target_url,
        title=page_data.get("title", "N/A"),
        h1=page_data.get("h1", "N/A"),
        content=page_data.get("content", "")[:1000],
        competitor_names=competitor_names or "No competitor data available.",
    )

    recommendations = await call_claude(LOCAL_SYSTEM, prompt, max_tokens=3000)

    return {
        "agent": "local_seo",
        "audit_id": audit_id,
        "status": "completed",
        "keyword": request.keyword,
        "location": request.location,
        "recommendations": recommendations,
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# ORCHESTRATOR — runs all 3 agents, builds combined report
# =============================================================================

def build_quick_wins(kw: dict, op: dict, local: dict) -> list[str]:
    """Extract real quick wins from each agent's output instead of hardcoding."""
    wins = []

    # From keyword research
    rec = kw.get("recommendations", {})
    if isinstance(rec, dict) and rec.get("recommendation"):
        wins.append(rec["recommendation"][:200])

    # From on-page
    op_rec = op.get("recommendations", {})
    if isinstance(op_rec, dict):
        recs = op_rec.get("recommendations", {})
        if isinstance(recs, dict) and recs.get("meta_title"):
            wins.append(f"Update title tag to: \"{recs['meta_title']}\"")
        priority = op_rec.get("priority_actions", [])
        if isinstance(priority, list):
            wins.extend(priority[:2])

    # From local SEO
    local_rec = local.get("recommendations", {})
    if isinstance(local_rec, dict):
        local_wins = local_rec.get("quick_wins", [])
        if isinstance(local_wins, list):
            wins.extend(local_wins[:3])

    # Fallback if agents didn't return structured data
    if not wins:
        wins = [
            "Complete Google Business Profile optimisation",
            "Update meta title and description with target keyword",
            "Build citations on top local directories",
            "Add internal links between service pages",
        ]

    return wins[:8]


def calculate_cost_estimate(agents_run: int = 3) -> float:
    """Rough cost per audit based on token usage."""
    # ~3K input + ~2K output per agent × 3 agents
    # Sonnet: $3/M input, $15/M output
    input_cost = (3000 * agents_run) / 1_000_000 * 3
    output_cost = (2000 * agents_run) / 1_000_000 * 15
    return round(input_cost + output_cost, 4)


@app.post("/workflow/seo-audit")
async def seo_audit_workflow(request: AuditRequest):
    """
    Full SEO Audit — runs keyword research first, then on-page + local concurrently.
    Total time: ~60-80 seconds instead of ~120 sequential.
    """
    audit_id = str(uuid.uuid4())
    start = time.time()
    logger.info(f"[{audit_id}] Full audit starting for '{request.keyword}' → {request.target_url}")

    try:
        # Phase 1 — keyword research (other agents benefit from this data)
        keyword_results = await keyword_research_agent(request)

        # Phase 2 — on-page + local run concurrently (they're independent)
        on_page_task = on_page_seo_agent(request)
        local_task = local_seo_agent(request)
        on_page_results, local_results = await asyncio.gather(on_page_task, local_task)

        elapsed = round(time.time() - start, 1)
        logger.info(f"[{audit_id}] Audit completed in {elapsed}s")

        report = {
            "audit_id": audit_id,
            "keyword": request.keyword,
            "target_url": request.target_url,
            "location": request.location,
            "status": "completed",
            "agents_executed": 3,
            "execution_time_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
            "agents": {
                "keyword_research": keyword_results,
                "on_page_seo": on_page_results,
                "local_seo": local_results,
            },
            "summary": {
                "estimated_api_cost": calculate_cost_estimate(),
                "quick_wins": build_quick_wins(keyword_results, on_page_results, local_results),
            },
        }

        # Persist to DB — run in thread pool so we don't block the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _save_audit, audit_id, request, report, elapsed)

        return report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{audit_id}] Audit failed: {e}", exc_info=True)
        raise HTTPException(500, "Audit failed — please try again")


def _save_audit(audit_id: str, request, report: dict, elapsed: float) -> None:
    """Synchronous DB write — called via run_in_executor."""
    try:
        db = SessionLocal()
        db.add(Audit(
            id=audit_id,
            keyword=request.keyword,
            target_url=request.target_url,
            location=request.location,
            status="completed",
            results_json=json.dumps(report),
            api_cost=report["summary"]["estimated_api_cost"],
            execution_time=elapsed,
        ))
        db.commit()
        logger.info(f"[{audit_id}] Saved to database")
    except Exception as e:
        logger.error(f"[{audit_id}] DB save failed: {e}")
    finally:
        db.close()


# =============================================================================
# Audit history
# =============================================================================

@app.get("/audits")
def list_audits(limit: int = 20, offset: int = 0):
    """Return the most recent audits (metadata only, no full results)."""
    db = SessionLocal()
    try:
        rows = (
            db.query(Audit)
            .order_by(Audit.created_at.desc())
            .offset(offset)
            .limit(min(limit, 100))
            .all()
        )
        return [
            {
                "id": r.id,
                "keyword": r.keyword,
                "target_url": r.target_url,
                "location": r.location,
                "status": r.status,
                "api_cost": r.api_cost,
                "execution_time": r.execution_time,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
    finally:
        db.close()


@app.get("/audits/{audit_id}")
def get_audit(audit_id: str):
    """Return the full result JSON for a single audit."""
    db = SessionLocal()
    try:
        row = db.query(Audit).filter(Audit.id == audit_id).first()
        if not row:
            raise HTTPException(status_code=404, detail="Audit not found")
        return json.loads(row.results_json)
    finally:
        db.close()


# =============================================================================
# Health & info
# =============================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "api_key_set": bool(ANTHROPIC_API_KEY),
        "serpapi_key_set": bool(SERPAPI_KEY),
    }


@app.get("/info")
async def info():
    return {
        "name": "SEO SaaS API",
        "version": "1.0.0",
        "model": CLAUDE_MODEL,
        "agents": ["keyword_research", "on_page_seo", "local_seo"],
        "endpoints": {
            "keyword_research": "POST /agents/keyword-research",
            "on_page_seo": "POST /agents/on-page-seo",
            "local_seo": "POST /agents/local-seo",
            "full_audit": "POST /workflow/seo-audit",
            "health": "GET /health",
        },
    }


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
