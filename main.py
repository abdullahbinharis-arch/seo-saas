# =============================================================================
# SEO SaaS Platform — FastAPI Backend
# =============================================================================
# Local SEO Audit System — 4 AI agents
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
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import bcrypt as _bcrypt_lib
from jose import JWTError, jwt as jose_jwt
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
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = 30
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "LocalRank <reports@localrank.io>")

if not ANTHROPIC_API_KEY:
    logger.warning("⚠️  ANTHROPIC_API_KEY is not set — Claude calls will fail")
if not SERPAPI_KEY:
    logger.warning("⚠️  SERPAPI_KEY is not set — competitor research will fail")
if not JWT_SECRET:
    logger.warning("⚠️  JWT_SECRET is not set — auth endpoints will fail")

# ---------------------------------------------------------------------------
# Anthropic client — ASYNC so we never block the event loop
# ---------------------------------------------------------------------------

from anthropic import AsyncAnthropic          # requires anthropic >= 0.39

anthropic_client = AsyncAnthropic()           # reads ANTHROPIC_API_KEY from env

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

from database import Audit, User, SessionLocal, init_db
from pdf_export import build_pdf

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
    "http://localhost:3000,http://localhost:5173,https://seo-frontend-six.vercel.app,https://seo-frontend-git-main-abdullas-projects-fb3622cf.vercel.app",
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
RATE_LIMIT_WHITELIST: set[str] = {
    e.strip().lower()
    for e in os.getenv("RATE_LIMIT_WHITELIST", "").split(",")
    if e.strip()
}


def _get_token_email(request: Request) -> str | None:
    """Extract email from Bearer JWT without raising — for whitelist check only."""
    try:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            return None
        from jose import jwt as _jose_jwt
        payload = _jose_jwt.decode(auth[7:], JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return (payload.get("email") or "").lower()
    except Exception:
        return None


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path.startswith(("/health", "/info", "/docs", "/openapi")):
        return await call_next(request)

    # Whitelisted users bypass rate limiting entirely
    if RATE_LIMIT_WHITELIST:
        email = _get_token_email(request)
        if email and email in RATE_LIMIT_WHITELIST:
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
    business_name: Optional[str] = None
    business_type: Optional[str] = None
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
# Auth — password hashing, JWT, request models, dependency
# =============================================================================

def _hash_password(password: str) -> str:
    return _bcrypt_lib.hashpw(password.encode(), _bcrypt_lib.gensalt()).decode()


def _verify_password(plain: str, hashed: str) -> bool:
    return _bcrypt_lib.checkpw(plain.encode(), hashed.encode())


def _create_token(user_id: str, email: str) -> str:
    from datetime import timedelta
    payload = {
        "sub": user_id,
        "email": email,
        "exp": datetime.utcnow() + timedelta(days=JWT_EXPIRE_DAYS),
    }
    return jose_jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


class CurrentUser(BaseModel):
    id: str
    email: str


def get_current_user(authorization: Optional[str] = Header(default=None)) -> CurrentUser:
    """FastAPI dependency — validates Bearer JWT and returns the current user."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization[len("Bearer "):]
    try:
        payload = jose_jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        if not user_id or not email:
            raise HTTPException(status_code=401, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    return CurrentUser(id=user_id, email=email)


class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class OAuthSyncRequest(BaseModel):
    email: str
    google_sub: str
    name: Optional[str] = None


# =============================================================================
# Auth endpoints
# =============================================================================

@app.post("/auth/register", status_code=201)
def auth_register(body: RegisterRequest):
    """Create a new email/password account. Called by the registration form."""
    email = body.email.lower().strip()
    db = SessionLocal()
    try:
        if db.query(User).filter(User.email == email).first():
            raise HTTPException(status_code=400, detail="Email already registered")
        user = User(
            id=str(uuid.uuid4()),
            email=email,
            hashed_password=_hash_password(body.password),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        token = _create_token(user.id, user.email)
        return {"id": user.id, "email": user.email, "access_token": token}
    finally:
        db.close()


@app.post("/auth/login")
def auth_login(body: LoginRequest):
    """
    Verify email/password credentials. Called by NextAuth CredentialsProvider.
    Returns access_token that NextAuth stores in the session for API calls.
    """
    email = body.email.lower().strip()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user or not user.hashed_password:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        if not _verify_password(body.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        token = _create_token(user.id, user.email)
        return {"id": user.id, "email": user.email, "access_token": token}
    finally:
        db.close()


@app.post("/auth/oauth-sync")
def auth_oauth_sync(body: OAuthSyncRequest):
    """
    Called by NextAuth JWT callback after Google OAuth. Creates the user on first
    sign-in, finds them on subsequent sign-ins, and returns an access_token so
    the frontend can make authenticated FastAPI calls.
    """
    email = body.email.lower().strip()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            user = User(
                id=str(uuid.uuid4()),
                email=email,
                google_sub=body.google_sub,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        elif not user.google_sub:
            # Link Google to an existing email/password account
            user.google_sub = body.google_sub
            db.commit()
        token = _create_token(user.id, user.email)
        return {"id": user.id, "email": user.email, "access_token": token}
    finally:
        db.close()


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


async def fetch_pagespeed(url: str) -> dict:
    """Call Google PageSpeed Insights API (free, no key) for mobile + desktop."""
    base = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
    result = {
        "mobile": {},
        "desktop": {},
        "success": False,
        "error": None,
    }

    def _parse_strategy(data: dict) -> dict:
        cats = data.get("lighthouseResult", {}).get("categories", {})
        audits = data.get("lighthouseResult", {}).get("audits", {})

        def ms(audit_id: str) -> str | None:
            a = audits.get(audit_id, {})
            disp = a.get("displayValue")
            return disp if disp else None

        def score(audit_id: str) -> float | None:
            s = audits.get(audit_id, {}).get("score")
            return round(s * 100) if s is not None else None

        # Top 5 opportunities by estimated savings
        opportunities = []
        for audit_id, audit in audits.items():
            if audit.get("details", {}).get("type") == "opportunity":
                savings_ms = audit.get("details", {}).get("overallSavingsMs", 0) or 0
                if savings_ms > 0:
                    opportunities.append({
                        "id": audit_id,
                        "title": audit.get("title", audit_id),
                        "savings_ms": round(savings_ms),
                        "description": audit.get("description", "")[:120],
                    })
        opportunities.sort(key=lambda x: x["savings_ms"], reverse=True)

        return {
            "performance_score": round((cats.get("performance", {}).get("score") or 0) * 100),
            "lcp": ms("largest-contentful-paint"),
            "inp": ms("interaction-to-next-paint") or ms("total-blocking-time"),
            "cls": ms("cumulative-layout-shift"),
            "fcp": ms("first-contentful-paint"),
            "ttfb": ms("server-response-time"),
            "opportunities": opportunities[:5],
        }

    try:
        psi_key = os.getenv("PAGESPEED_API_KEY")
        params_base = {"url": url}
        if psi_key:
            params_base["key"] = psi_key

        async with httpx.AsyncClient(timeout=30.0) as http:
            mobile_resp, desktop_resp = await asyncio.gather(
                http.get(base, params={**params_base, "strategy": "mobile"}),
                http.get(base, params={**params_base, "strategy": "desktop"}),
            )
        if mobile_resp.status_code == 200:
            result["mobile"] = _parse_strategy(mobile_resp.json())
        elif mobile_resp.status_code == 429:
            result["error"] = "quota_exceeded"
        if desktop_resp.status_code == 200:
            result["desktop"] = _parse_strategy(desktop_resp.json())
        result["success"] = bool(result["mobile"] or result["desktop"])
    except Exception as e:
        result["error"] = str(e)
        logger.warning(f"PageSpeed fetch failed for {url}: {e}")

    return result


async def scrape_technical_signals(url: str) -> dict:
    """Extract technical SEO signals from the raw HTML of a page."""
    from urllib.parse import urlparse

    signals: dict = {
        "url": url,
        "success": False,
        "https": urlparse(url).scheme == "https",
        "viewport": None,
        "canonical": None,
        "robots_meta": None,
        "schemas": [],
        "images_total": 0,
        "images_missing_alt": [],
        "render_blocking_scripts": 0,
        "lazy_loaded_images": 0,
        "inline_style_bytes": 0,
    }

    try:
        async with httpx.AsyncClient(
            timeout=12.0,
            follow_redirects=True,
            headers={"User-Agent": "SEOSaasBot/1.0"},
        ) as http:
            resp = await http.get(url)

        soup = BeautifulSoup(resp.text, "html.parser")
        signals["success"] = True

        # Viewport
        vp = soup.find("meta", attrs={"name": "viewport"})
        signals["viewport"] = vp.get("content", "") if vp else None

        # Canonical
        canon = soup.find("link", rel="canonical")
        signals["canonical"] = canon.get("href", "") if canon else None

        # Robots meta
        robots = soup.find("meta", attrs={"name": "robots"})
        signals["robots_meta"] = robots.get("content", "") if robots else None

        # Structured data
        for script in soup.find_all("script", type="application/ld+json"):
            try:
                schema = json.loads(script.string or "{}")
                schema_type = schema.get("@type") or schema.get("@graph", [{}])[0].get("@type", "Unknown")
                signals["schemas"].append(schema_type)
            except (json.JSONDecodeError, IndexError):
                signals["schemas"].append("Unknown")

        # Images
        imgs = soup.find_all("img")
        signals["images_total"] = len(imgs)
        signals["images_missing_alt"] = [
            img.get("src", "unknown")[:80]
            for img in imgs
            if not img.get("alt")
        ][:20]  # cap at 20 for prompt brevity
        signals["lazy_loaded_images"] = sum(
            1 for img in imgs if img.get("loading") == "lazy"
        )

        # Render-blocking scripts (in <head>, no defer/async)
        head = soup.find("head") or soup
        signals["render_blocking_scripts"] = sum(
            1 for s in head.find_all("script", src=True)
            if not s.get("defer") and not s.get("async")
        )

        # Inline styles size
        signals["inline_style_bytes"] = sum(
            len(s.string or "") for s in soup.find_all("style")
        )

    except Exception as e:
        logger.warning(f"Technical scrape failed for {url}: {e}")

    return signals


def calculate_keyword_density(text: str, keyword: str) -> dict:
    """Count keyword occurrences / total words. Returns density info."""
    if not text or not keyword:
        return {"occurrences": 0, "total_words": 0, "density_pct": 0.0, "status": "no_data"}

    words = text.lower().split()
    total_words = len(words)
    if total_words == 0:
        return {"occurrences": 0, "total_words": 0, "density_pct": 0.0, "status": "no_data"}

    # Count full keyword phrase occurrences
    kw_lower = keyword.lower()
    text_lower = text.lower()
    occurrences = text_lower.count(kw_lower)

    # Also count individual words if multi-word keyword
    kw_words = kw_lower.split()
    word_occurrences = sum(words.count(w) for w in kw_words if len(w) > 3)

    density_pct = round((occurrences / total_words) * 100, 2)

    if density_pct == 0:
        status = "missing"
    elif density_pct < 0.5:
        status = "too_low"
    elif density_pct <= 2.0:
        status = "optimal"
    else:
        status = "too_high"

    return {
        "occurrences": occurrences,
        "word_occurrences": word_occurrences,
        "total_words": total_words,
        "density_pct": density_pct,
        "recommended_range": "1–2%",
        "status": status,
    }


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
                    "num": 3,
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
    tasks = [scrape_page(c["url"]) for c in competitors[:2]]
    pages = await asyncio.gather(*tasks)

    summaries = []
    for comp, page in zip(competitors[:2], pages):
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
                ],
            )
            raw = response.content[0].text
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

KEYWORD_SYSTEM = """You are an expert local SEO keyword researcher helping local businesses rank in Google's Map Pack and local organic results.
You ALWAYS respond with valid JSON only — no markdown, no explanation, no preamble.
Focus on local intent: "near me" searches, city + service combinations, and keywords that drive phone calls, walk-ins, and appointment bookings.
Your analysis is specific to the business type and location provided — never generic."""

KEYWORD_PROMPT = """Analyse local keyword opportunities for this business.

BUSINESS NAME: {business_name}
BUSINESS TYPE: {business_type}
TARGET KEYWORD: {keyword}
LOCATION: {location}
TARGET URL: {target_url}

CLIENT PAGE DATA:
- Title: {client_title}
- H1: {client_h1}
- Word count: {client_word_count}
- Keyword density (Python-calculated): {keyword_density_pct}% ({keyword_density_status}) — {keyword_occurrences} occurrences in {total_words} words
- Content preview: {client_content}

COMPETITOR PAGES (Top 3 from SerpApi):
{competitor_data}

Focus on keywords that a {business_type} in {location} would need to rank for in the Google Map Pack and local organic results.
Prioritize "near me" searches, emergency/urgent service keywords, and location + service combinations.

Return JSON with EXACTLY these keys:
{{
  "primary_keyword": "the main keyword",
  "keyword_density": {{
    "current_pct": {keyword_density_pct},
    "occurrences": {keyword_occurrences},
    "total_words": {total_words},
    "status": "{keyword_density_status}",
    "recommended_range": "1-2%",
    "assessment": "<one sentence: is this optimal, too low, or over-stuffed? What should they change?>"
  }},
  "semantic_keywords": [
    {{"keyword": "...", "relevance": "why Google expects this term alongside the primary keyword"}}
  ],
  "keyword_gap": [
    {{
      "keyword": "...",
      "category": "missing|weak|strong|untapped",
      "category_explanation": "missing=competitor ranks, client doesn't | weak=client ranks but lower | strong=client outranks | untapped=nobody ranks well",
      "action": "create new page|optimize existing page|write blog post|protect with updates",
      "estimated_volume": "low|medium|high",
      "difficulty": "low|medium|high"
    }}
  ],
  "high_intent_keywords": [
    {{"keyword": "...", "intent": "commercial|informational|navigational|transactional", "estimated_monthly_searches": 0, "difficulty": "low|medium|high", "local_modifier": "..."}}
  ],
  "long_tail_keywords": ["phrase 1", "phrase 2"],
  "keyword_clusters": [
    {{"theme": "...", "keywords": ["...", "..."]}}
  ],
  "content_gap_opportunities": ["topic 1", "topic 2"],
  "recommendation": "One paragraph summary of the local keyword strategy for this {business_type}"
}}

Rules:
- semantic_keywords: at least 10 LSI terms Google expects alongside "{keyword}" for a {business_type}
- keyword_gap: at least 10 keywords, mix of all 4 categories. Base "missing" and "weak" on competitor page content provided above vs client content.
- high_intent_keywords: at least 15, specific to a {business_type} in {location}
- long_tail_keywords: at least 8 phrases (3-5 words each)"""


@app.post("/agents/keyword-research")
async def keyword_research_agent(request: AuditRequest):
    """Keyword Research Agent — density, semantic keywords, gap analysis, clusters."""
    audit_id = str(uuid.uuid4())
    logger.info(f"[{audit_id}] Keyword Research starting for '{request.keyword}'")

    # Scrape client page + fetch competitors concurrently
    page_data, competitors = await asyncio.gather(
        scrape_page(request.target_url),
        fetch_competitors(request.keyword, request.location),
    )
    competitor_data = await scrape_competitors(competitors)

    # Calculate keyword density from scraped page text
    client_content = page_data.get("content", "")
    density = calculate_keyword_density(client_content, request.keyword)

    prompt = KEYWORD_PROMPT.format(
        business_name=request.business_name or "this business",
        business_type=request.business_type or "local business",
        keyword=request.keyword,
        location=request.location,
        target_url=request.target_url,
        # Client page
        client_title=page_data.get("title", "N/A"),
        client_h1=page_data.get("h1", "N/A"),
        client_word_count=page_data.get("word_count", 0),
        client_content=client_content[:800],
        # Keyword density (Python-calculated)
        keyword_density_pct=density["density_pct"],
        keyword_density_status=density["status"],
        keyword_occurrences=density["occurrences"],
        total_words=density["total_words"],
        # Competitors
        competitor_data=competitor_data,
    )

    recommendations = await call_claude(KEYWORD_SYSTEM, prompt, max_tokens=2500)

    return {
        "agent": "keyword_research",
        "audit_id": audit_id,
        "status": "completed",
        "keyword": request.keyword,
        "competitors_analyzed": len(competitors),
        "keyword_density": density,
        "recommendations": recommendations,
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# AGENT 2 — On-Page SEO
# =============================================================================

ONPAGE_SYSTEM = """You are an expert on-page SEO specialist focused on local businesses and Google local ranking factors.
You ALWAYS respond with valid JSON only — no markdown, no explanation, no preamble.
Your recommendations prioritize local SEO signals: NAP consistency, local keywords in title/H1, LocalBusiness schema, service-area content, and mobile optimization.
Every recommendation must be actionable, specific, and tailored to the business type provided."""

ONPAGE_PROMPT = """Analyse and optimise this local business page for the target keyword.

BUSINESS NAME: {business_name}
BUSINESS TYPE: {business_type}
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
        business_name=request.business_name or "this business",
        business_type=request.business_type or "local business",
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

    recommendations = await call_claude(ONPAGE_SYSTEM, prompt, max_tokens=2000)

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

LOCAL_SYSTEM = """You are a local SEO specialist focused on Google Map Pack rankings, Google Business Profile optimization, and local search dominance for local businesses.
You ALWAYS respond with valid JSON only — no markdown, no explanation, no preamble.
Your recommendations are ALWAYS industry-specific — never generic. Every citation, GBP attribute, and content strategy must be tailored to the exact business type.
You understand that local SEO success = GBP completeness + NAP consistency + local citations + review velocity + local content signals."""

LOCAL_PROMPT = """Create a local SEO strategy for this business. Respond with valid JSON only.

BUSINESS: {business_name} ({business_type}) in {location}
URL: {target_url}
KEYWORD: {keyword}
PAGE: Title={title} | H1={h1}
COMPETITORS: {competitor_names}

Return this exact JSON structure:
{{
  "local_seo_score": <0-100 int>,
  "gbp_optimization": {{
    "priority_attributes": ["<3-5 specific GBP attributes to complete for a {business_type}>"],
    "categories": ["<primary GBP category>", "<secondary>"],
    "photo_strategy": "<specific photo types and quantities for a {business_type}>",
    "review_strategy": {{
      "target_reviews_per_month": <int>,
      "review_request_template": "<one sentence ask>"
    }}
  }},
  "citations": [
    {{"site": "<name>", "url": "<url>", "priority": "critical|high|medium", "category": "general|industry-specific|local"}}
  ],
  "link_opportunities": [
    {{"name": "<org>", "link_type": "directory|guest-post|sponsorship", "reason": "<why>"}}
  ],
  "local_content_strategy": {{
    "blog_topics": ["<topic 1>", "<topic 2>", "<topic 3>"],
    "service_area_pages": ["<area 1>", "<area 2>"]
  }},
  "quick_wins": ["<win 1>", "<win 2>", "<win 3>", "<win 4>", "<win 5>"],
  "estimated_impact": "<one sentence>"
}}

Rules:
- local_seo_score: +20 GBP visible, +15 NAP on page, +15 schema markup, +10 reviews on page, +15 local keyword in title/H1, +10 blog/content, +15 mobile-friendly. Conservative — most score 20-60.
- citations: exactly 8 total. First 4 = critical generals (GBP, Yelp, BBB, Bing Places). Next 4 = top industry-specific directories for {business_type}.
- All recommendations must be specific to a {business_type} in {location} — never generic."""


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
        for c in competitors[:3]
    )

    prompt = LOCAL_PROMPT.format(
        business_name=request.business_name or "this business",
        business_type=request.business_type or "local business",
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
# AGENT 4 — Technical SEO
# =============================================================================

TECHNICAL_SYSTEM = """You are a technical SEO engineer specialising in Core Web Vitals, crawlability, and structured data for local business websites.
You ALWAYS respond with valid JSON only — no markdown, no explanation, no preamble.
Your findings are specific, evidence-based, and prioritised by impact on local search rankings.
Pay special attention to LocalBusiness schema, mobile optimization, and page speed — all critical for Google Map Pack rankings."""

TECHNICAL_PROMPT = """Perform a technical SEO audit for this local business website.

BUSINESS NAME: {business_name}
BUSINESS TYPE: {business_type}
TARGET URL: {target_url}
TARGET KEYWORD: {keyword}

PAGESPEED INSIGHTS (real data from Google):
Mobile Performance Score: {psi_mobile_score}/100
Desktop Performance Score: {psi_desktop_score}/100
Mobile Metrics:
  - LCP (Largest Contentful Paint): {psi_mobile_lcp}
  - INP / TBT (Interaction to Next Paint / Total Blocking Time): {psi_mobile_inp}
  - CLS (Cumulative Layout Shift): {psi_mobile_cls}
  - FCP (First Contentful Paint): {psi_mobile_fcp}
  - TTFB (Time to First Byte): {psi_mobile_ttfb}
Desktop Metrics:
  - LCP: {psi_desktop_lcp}
  - FCP: {psi_desktop_fcp}
  - TTFB: {psi_desktop_ttfb}
Top PageSpeed Opportunities:
{psi_opportunities}

TECHNICAL SIGNALS (from page scrape):
- HTTPS: {https}
- Viewport meta tag: {viewport}
- Canonical tag: {canonical}
- Robots meta: {robots_meta}
- Structured data schemas found: {schemas}
- Total images: {images_total}
- Images missing alt text: {images_missing_alt}
- Lazy-loaded images: {lazy_loaded_images}
- Render-blocking scripts in <head>: {render_blocking_scripts}
- Inline CSS bytes: {inline_style_bytes}

Return JSON with EXACTLY these keys:
{{
  "technical_score": 0,
  "core_web_vitals": {{
    "mobile_performance_score": {psi_mobile_score},
    "desktop_performance_score": {psi_desktop_score},
    "lcp": {{"mobile": "{psi_mobile_lcp}", "desktop": "{psi_desktop_lcp}", "status": "pass|warn|fail"}},
    "cls": {{"mobile": "{psi_mobile_cls}", "status": "pass|warn|fail"}},
    "fcp": {{"mobile": "{psi_mobile_fcp}", "desktop": "{psi_desktop_fcp}", "status": "pass|warn|fail"}},
    "ttfb": {{"mobile": "{psi_mobile_ttfb}", "status": "pass|warn|fail"}},
    "top_opportunities": ["<opportunity 1 with estimated saving>", "<opportunity 2>", "<opportunity 3>"]
  }},
  "https": {{
    "status": "pass|fail",
    "detail": "One sentence explanation"
  }},
  "mobile": {{
    "viewport_present": true,
    "viewport_content": "exact viewport content or null",
    "status": "pass|fail|warn",
    "recommendation": "Specific fix or confirmation it is correct"
  }},
  "canonical": {{
    "tag_present": true,
    "canonical_url": "the canonical href or null",
    "status": "pass|fail|warn",
    "recommendation": "Specific fix or confirmation"
  }},
  "robots": {{
    "meta_content": "the robots content or null",
    "status": "pass|fail|warn",
    "recommendation": "Specific fix or confirmation"
  }},
  "structured_data": {{
    "schemas_found": ["type1", "type2"],
    "status": "pass|warn|fail",
    "schemas_to_add": ["FAQPage", "BreadcrumbList"],
    "recommendation": "Specific implementation advice"
  }},
  "images": {{
    "total": 0,
    "missing_alt_count": 0,
    "lazy_loaded_count": 0,
    "status": "pass|warn|fail",
    "recommendation": "Specific fix"
  }},
  "page_speed": {{
    "render_blocking_scripts": 0,
    "inline_css_bytes": 0,
    "status": "pass|warn|fail",
    "issues": ["specific issue 1", "specific issue 2"],
    "recommendation": "Specific fix"
  }},
  "priority_fixes": [
    "Most impactful fix first (be specific, reference real PSI numbers)",
    "Second fix",
    "Third fix"
  ],
  "quick_wins": [
    "Fastest technical fix 1",
    "Fastest technical fix 2"
  ]
}}

Score 0–100 (was 0-10, now 0-100). Scoring:
- HTTPS: 10pts
- Mobile performance score ≥ 90: 20pts, 50-89: 10pts, <50: 0pts
- Desktop performance score ≥ 90: 10pts, 50-89: 5pts, <50: 0pts
- LCP ≤ 2.5s: 15pts, ≤ 4s: 8pts, >4s: 0pts
- CLS ≤ 0.1: 10pts, ≤ 0.25: 5pts, >0.25: 0pts
- Viewport meta present: 5pts
- Canonical tag: 5pts
- Structured data: 10pts
- Images (all have alt text): 5pts
- PageSpeed data unavailable: use scrape signals only and score conservatively."""


@app.post("/agents/technical-seo")
async def technical_seo_agent(request: AuditRequest):
    """Technical SEO Agent — Core Web Vitals (PageSpeed API), HTTPS, mobile, canonical, schema, images."""
    audit_id = str(uuid.uuid4())
    logger.info(f"[{audit_id}] Technical SEO starting for '{request.target_url}'")

    # Fetch PageSpeed data and scrape signals concurrently
    signals, psi = await asyncio.gather(
        scrape_technical_signals(request.target_url),
        fetch_pagespeed(request.target_url),
    )

    mob = psi.get("mobile", {})
    desk = psi.get("desktop", {})

    # Format top opportunities as a bullet list
    opps = mob.get("opportunities") or desk.get("opportunities") or []
    opp_lines = "\n".join(
        f"  - {o['title']}: saves ~{o['savings_ms']}ms"
        for o in opps
    ) or "  - No opportunity data available"

    prompt = TECHNICAL_PROMPT.format(
        business_name=request.business_name or "this business",
        business_type=request.business_type or "local business",
        target_url=request.target_url,
        keyword=request.keyword,
        # PageSpeed
        psi_mobile_score=mob.get("performance_score", "N/A"),
        psi_desktop_score=desk.get("performance_score", "N/A"),
        psi_mobile_lcp=mob.get("lcp", "N/A"),
        psi_mobile_inp=mob.get("inp", "N/A"),
        psi_mobile_cls=mob.get("cls", "N/A"),
        psi_mobile_fcp=mob.get("fcp", "N/A"),
        psi_mobile_ttfb=mob.get("ttfb", "N/A"),
        psi_desktop_lcp=desk.get("lcp", "N/A"),
        psi_desktop_fcp=desk.get("fcp", "N/A"),
        psi_desktop_ttfb=desk.get("ttfb", "N/A"),
        psi_opportunities=opp_lines,
        # Scrape signals
        https=signals["https"],
        viewport=signals["viewport"] or "NOT FOUND",
        canonical=signals["canonical"] or "NOT FOUND",
        robots_meta=signals["robots_meta"] or "NOT FOUND",
        schemas=", ".join(signals["schemas"]) if signals["schemas"] else "None found",
        images_total=signals["images_total"],
        images_missing_alt=signals["images_missing_alt"] or "None",
        lazy_loaded_images=signals["lazy_loaded_images"],
        render_blocking_scripts=signals["render_blocking_scripts"],
        inline_style_bytes=signals["inline_style_bytes"],
    )

    recommendations = await call_claude(TECHNICAL_SYSTEM, prompt, max_tokens=2000)

    return {
        "agent": "technical_seo",
        "audit_id": audit_id,
        "status": "completed",
        "keyword": request.keyword,
        "target_url": request.target_url,
        "page_scraped": signals["success"],
        "pagespeed_fetched": psi["success"],
        "pagespeed": psi,
        "signals": signals,
        "recommendations": recommendations,
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# ORCHESTRATOR — runs all 4 agents, builds combined report
# =============================================================================

def calculate_local_seo_score(op: dict, local: dict, tech: dict) -> int:
    """Compute composite Local SEO Score (0-100) from agent outputs."""
    # On-page SEO: 30% weight (agent returns seo_score 0-100)
    op_score = op.get("recommendations", {}).get("current_analysis", {}).get("seo_score", 50)
    op_score = max(0, min(100, int(op_score)))

    # Technical: 20% weight (agent returns technical_score 0-10, scale to 0-100)
    tech_raw = tech.get("recommendations", {}).get("technical_score", 5)
    tech_score = max(0, min(10, int(tech_raw))) * 10

    # Local SEO: 50% weight (agent returns local_seo_score 0-100)
    local_score = local.get("recommendations", {}).get("local_seo_score", 35)
    local_score = max(0, min(100, int(local_score)))

    combined = int(op_score * 0.30 + tech_score * 0.20 + local_score * 0.50)
    return min(100, max(0, combined))


def build_quick_wins(kw: dict, op: dict, local: dict, tech: dict) -> list[str]:
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
            wins.extend(local_wins[:2])

    # From technical SEO
    tech_rec = tech.get("recommendations", {})
    if isinstance(tech_rec, dict):
        tech_wins = tech_rec.get("quick_wins", [])
        if isinstance(tech_wins, list):
            wins.extend(tech_wins[:2])

    # Fallback if agents didn't return structured data
    if not wins:
        wins = [
            "Complete Google Business Profile optimisation",
            "Update meta title and description with target keyword",
            "Build citations on top local directories",
            "Add internal links between service pages",
        ]

    return wins[:8]


def calculate_cost_estimate(agents_run: int = 4) -> float:
    """Rough cost per audit based on token usage."""
    # ~3K input + ~2K output per agent × 3 agents
    # Sonnet: $3/M input, $15/M output
    input_cost = (3000 * agents_run) / 1_000_000 * 3
    output_cost = (2000 * agents_run) / 1_000_000 * 15
    return round(input_cost + output_cost, 4)


def _send_audit_email(to_email: str, report: dict) -> None:
    """Send a branded HTML audit summary email via Resend."""
    if not RESEND_API_KEY:
        logger.warning("RESEND_API_KEY not set — skipping email")
        return

    import resend
    resend.api_key = RESEND_API_KEY

    business_name = report.get("business_name") or report.get("target_url", "Your Business")
    business_type = report.get("business_type", "")
    score = report.get("local_seo_score", 0)
    quick_wins = report.get("summary", {}).get("quick_wins", [])
    audit_id = report.get("audit_id", "")

    # Score colour
    if score >= 70:
        score_color = "#10b981"
        score_label = "Good"
    elif score >= 40:
        score_color = "#f59e0b"
        score_label = "Needs Work"
    else:
        score_color = "#ef4444"
        score_label = "Poor"

    wins_html = "".join(
        f'<li style="margin-bottom:8px;color:#d1d5db;">{w}</li>'
        for w in quick_wins[:5]
    )

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head>
<body style="margin:0;padding:0;background:#09090b;font-family:system-ui,-apple-system,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#09090b;padding:40px 16px;">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;">

        <!-- Header -->
        <tr><td style="background:#0f0f12;border:1px solid rgba(255,255,255,0.06);border-radius:16px 16px 0 0;padding:32px 40px;text-align:center;">
          <div style="display:inline-flex;align-items:center;gap:10px;margin-bottom:8px;">
            <div style="width:36px;height:36px;background:linear-gradient(135deg,#34d399,#059669);border-radius:10px;display:inline-block;vertical-align:middle;"></div>
            <span style="font-size:20px;font-weight:700;color:#ffffff;vertical-align:middle;">LocalRank</span>
          </div>
          <p style="margin:8px 0 0;color:#71717a;font-size:14px;">Your Local SEO Audit Report is ready</p>
        </td></tr>

        <!-- Score card -->
        <tr><td style="background:#18181b;border-left:1px solid rgba(255,255,255,0.06);border-right:1px solid rgba(255,255,255,0.06);padding:32px 40px;text-align:center;">
          <p style="margin:0 0 4px;color:#a1a1aa;font-size:13px;text-transform:uppercase;letter-spacing:0.05em;">Local SEO Score</p>
          <p style="margin:0;font-size:72px;font-weight:900;color:{score_color};line-height:1;">{score}</p>
          <p style="margin:4px 0 0;color:{score_color};font-size:16px;font-weight:600;">{score_label}</p>
          <div style="background:rgba(255,255,255,0.08);border-radius:999px;height:8px;margin:20px auto;max-width:300px;overflow:hidden;">
            <div style="background:{score_color};height:8px;width:{score}%;border-radius:999px;"></div>
          </div>
          <p style="margin:0;color:#71717a;font-size:13px;">{business_name}{f" &middot; {business_type}" if business_type else ""}</p>
        </td></tr>

        <!-- Quick wins -->
        {"" if not wins_html else f'''
        <tr><td style="background:#0f0f12;border-left:1px solid rgba(255,255,255,0.06);border-right:1px solid rgba(255,255,255,0.06);padding:28px 40px;">
          <p style="margin:0 0 16px;color:#ffffff;font-size:15px;font-weight:600;">⚡ Top Quick Wins</p>
          <ul style="margin:0;padding-left:20px;">{wins_html}</ul>
        </td></tr>'''}

        <!-- CTA -->
        <tr><td style="background:#18181b;border:1px solid rgba(255,255,255,0.06);border-radius:0 0 16px 16px;padding:32px 40px;text-align:center;">
          <p style="margin:0 0 20px;color:#a1a1aa;font-size:14px;">Your full report includes keyword research, on-page fixes, citation opportunities, and a complete local SEO strategy.</p>
          <p style="margin:0 0 28px;color:#52525b;font-size:12px;">Audit ID: {audit_id[:8]}</p>
          <p style="margin:24px 0 0;color:#52525b;font-size:11px;">LocalRank &mdash; AI-Powered Local SEO Platform</p>
        </td></tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""

    try:
        resend.Emails.send({
            "from": FROM_EMAIL,
            "to": [to_email],
            "subject": f"Your Local SEO Audit is Ready — Score: {score}/100",
            "html": html,
        })
        logger.info(f"Audit email sent to {to_email}")
    except Exception as e:
        logger.error(f"Failed to send audit email to {to_email}: {e}")


@app.post("/workflow/seo-audit")
async def seo_audit_workflow(request: AuditRequest, current_user: CurrentUser = Depends(get_current_user)):
    """
    Full SEO Audit — runs keyword research first, then on-page + local + technical concurrently.
    Total time: ~60-90 seconds instead of ~160 sequential.
    """
    audit_id = str(uuid.uuid4())
    start = time.time()
    logger.info(f"[{audit_id}] Full audit starting for '{request.keyword}' → {request.target_url}")

    try:
        # Phase 1 — keyword research (other agents benefit from this data)
        keyword_results = await keyword_research_agent(request)

        # Phase 2 — on-page, local, and technical run concurrently (all independent)
        on_page_results, local_results, technical_results = await asyncio.gather(
            on_page_seo_agent(request),
            local_seo_agent(request),
            technical_seo_agent(request),
        )

        elapsed = round(time.time() - start, 1)
        logger.info(f"[{audit_id}] Audit completed in {elapsed}s")

        report = {
            "audit_id": audit_id,
            "business_name": request.business_name or "",
            "business_type": request.business_type or "other",
            "keyword": request.keyword,
            "target_url": request.target_url,
            "location": request.location,
            "status": "completed",
            "agents_executed": 4,
            "execution_time_seconds": elapsed,
            "timestamp": datetime.now().isoformat(),
            "local_seo_score": calculate_local_seo_score(on_page_results, local_results, technical_results),
            "agents": {
                "keyword_research": keyword_results,
                "on_page_seo": on_page_results,
                "local_seo": local_results,
                "technical_seo": technical_results,
            },
            "summary": {
                "estimated_api_cost": calculate_cost_estimate(),
                "quick_wins": build_quick_wins(keyword_results, on_page_results, local_results, technical_results),
            },
        }

        # Persist to DB — run in thread pool so we don't block the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _save_audit, audit_id, request, report, elapsed, current_user.id)

        # Send audit summary email (non-blocking — failure won't affect response)
        await loop.run_in_executor(None, _send_audit_email, current_user.email, report)

        return report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{audit_id}] Audit failed: {e}", exc_info=True)
        raise HTTPException(500, "Audit failed — please try again")


def _save_audit(audit_id: str, request, report: dict, elapsed: float, user_id: str = None) -> None:
    """Synchronous DB write — called via run_in_executor."""
    try:
        db = SessionLocal()
        db.add(Audit(
            id=audit_id,
            user_id=user_id,
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
def list_audits(limit: int = 20, offset: int = 0, current_user: CurrentUser = Depends(get_current_user)):
    """Return the authenticated user's audits (metadata only, no full results)."""
    db = SessionLocal()
    try:
        rows = (
            db.query(Audit)
            .filter(Audit.user_id == current_user.id)
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
def get_audit(audit_id: str, current_user: CurrentUser = Depends(get_current_user)):
    """Return the full result JSON for a single audit (owner only)."""
    db = SessionLocal()
    try:
        row = db.query(Audit).filter(
            Audit.id == audit_id,
            Audit.user_id == current_user.id,
        ).first()
        if not row:
            raise HTTPException(status_code=404, detail="Audit not found")
        return json.loads(row.results_json)
    finally:
        db.close()


@app.post("/audits/{audit_id}/export")
def export_audit_pdf(audit_id: str, current_user: CurrentUser = Depends(get_current_user)):
    """Generate and return a PDF report for a completed audit (owner only)."""
    db = SessionLocal()
    try:
        row = db.query(Audit).filter(
            Audit.id == audit_id,
            Audit.user_id == current_user.id,
        ).first()
        if not row:
            raise HTTPException(status_code=404, detail="Audit not found")
        audit_data = json.loads(row.results_json)
    finally:
        db.close()

    try:
        pdf_bytes = build_pdf(audit_data)
    except Exception as e:
        logger.error(f"PDF generation failed for {audit_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="PDF generation failed")

    slug = audit_data.get("keyword", "audit").replace(" ", "-")[:30]
    filename = f"seo-audit-{slug}-{audit_id[:8]}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


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
