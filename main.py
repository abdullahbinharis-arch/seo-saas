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
from pydantic import BaseModel, field_validator, model_validator

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
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = 30
RESEND_API_KEY = os.getenv("RESEND_API_KEY", "")
FROM_EMAIL = os.getenv("FROM_EMAIL", "LocalRankr <reports@localrankr.io>")
MOZ_ACCESS_ID = os.getenv("MOZ_ACCESS_ID", "")
MOZ_SECRET_KEY = os.getenv("MOZ_SECRET_KEY", "")
MAX_CRAWL_PAGES = int(os.getenv("MAX_CRAWL_PAGES", "50"))

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

from database import Audit, Profile, User, SessionLocal, init_db
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

# In-memory store for background audit tasks: audit_id -> {"status": ..., "result": ...}
_pending_audits: dict[str, dict] = {}
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
    keyword: Optional[str] = None
    target_url: str = ""
    domain: Optional[str] = None  # Alternative to target_url — triggers site-wide crawl
    location: str = "Toronto, Canada"
    business_name: Optional[str] = None
    business_type: Optional[str] = None
    user_id: Optional[str] = None
    profile_id: Optional[str] = None
    include_blog: bool = False  # opt-in: adds ~40s but generates a full blog post

    @field_validator("keyword")
    @classmethod
    def keyword_valid(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        if len(v) < 2:
            raise ValueError("Keyword must be at least 2 characters")
        if len(v) > 200:
            raise ValueError("Keyword must be under 200 characters")
        return v

    @model_validator(mode="after")
    def resolve_url(self) -> "AuditRequest":
        # Normalise domain → target_url
        if self.domain:
            d = self.domain.strip().lower()
            d = d.removeprefix("http://").removeprefix("https://").rstrip("/")
            self.domain = d
            if not self.target_url:
                self.target_url = f"https://{d}"
        if not self.target_url:
            raise ValueError("Either target_url or domain must be provided")
        self.target_url = self.target_url.strip()
        if not self.target_url.startswith(("http://", "https://")):
            raise ValueError("target_url must start with http:// or https://")
        return self


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


def get_optional_user(authorization: Optional[str] = Header(default=None)) -> Optional[CurrentUser]:
    """Like get_current_user but returns None instead of raising for missing/invalid tokens.
    Use on endpoints that work for both authenticated and anonymous users."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization[len("Bearer "):]
    try:
        payload = jose_jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        if user_id and email:
            return CurrentUser(id=user_id, email=email)
    except JWTError:
        pass
    return None


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


class ProfileCreateRequest(BaseModel):
    business_name: str
    website_url: str
    business_category: Optional[str] = None
    services: Optional[list[str]] = None
    country: Optional[str] = None
    city: Optional[str] = None
    nap_data: Optional[dict] = None


class ProfileUpdateRequest(BaseModel):
    business_name: Optional[str] = None
    website_url: Optional[str] = None
    business_category: Optional[str] = None
    services: Optional[list[str]] = None
    country: Optional[str] = None
    city: Optional[str] = None
    nap_data: Optional[dict] = None


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
# Profile endpoints
# =============================================================================

def _profile_to_dict(p: Profile) -> dict:
    """Serialize a Profile ORM object to a JSON-safe dict."""
    services = None
    if p.services:
        try:
            services = json.loads(p.services)
        except Exception:
            services = []
    nap = None
    if p.nap_data:
        try:
            nap = json.loads(p.nap_data)
        except Exception:
            nap = {}
    return {
        "id": p.id,
        "user_id": p.user_id,
        "business_name": p.business_name,
        "website_url": p.website_url,
        "business_category": p.business_category,
        "services": services or [],
        "country": p.country,
        "city": p.city,
        "nap_data": nap,
        "is_active": p.is_active,
        "created_at": p.created_at.isoformat() if p.created_at else None,
        "updated_at": p.updated_at.isoformat() if p.updated_at else None,
    }


@app.post("/profiles", status_code=201)
def create_profile(body: ProfileCreateRequest, current_user: CurrentUser = Depends(get_current_user)):
    """Create a new business profile for the authenticated user."""
    db = SessionLocal()
    try:
        profile = Profile(
            id=str(uuid.uuid4()),
            user_id=current_user.id,
            business_name=body.business_name,
            website_url=body.website_url,
            business_category=body.business_category,
            services=json.dumps(body.services) if body.services else None,
            country=body.country,
            city=body.city,
            nap_data=json.dumps(body.nap_data) if body.nap_data else None,
        )
        db.add(profile)
        db.commit()
        db.refresh(profile)
        return _profile_to_dict(profile)
    finally:
        db.close()


@app.get("/profiles")
def list_profiles(current_user: CurrentUser = Depends(get_current_user)):
    """List the authenticated user's active profiles."""
    db = SessionLocal()
    try:
        rows = (
            db.query(Profile)
            .filter(Profile.user_id == current_user.id, Profile.is_active == True)
            .order_by(Profile.created_at.desc())
            .all()
        )
        result = []
        for p in rows:
            d = _profile_to_dict(p)
            # Add audit count + latest audit meta
            audit_count = db.query(Audit).filter(Audit.profile_id == p.id).count()
            d["audit_count"] = audit_count
            latest = (
                db.query(Audit)
                .filter(Audit.profile_id == p.id)
                .order_by(Audit.created_at.desc())
                .first()
            )
            d["latest_audit"] = {
                "id": latest.id,
                "version": latest.version,
                "keyword": latest.keyword,
                "status": latest.status,
                "created_at": latest.created_at.isoformat() if latest.created_at else None,
            } if latest else None
            result.append(d)
        return result
    finally:
        db.close()


@app.get("/profiles/{profile_id}")
def get_profile(profile_id: str, current_user: CurrentUser = Depends(get_current_user)):
    """Get a single profile with audit count and latest audit meta."""
    db = SessionLocal()
    try:
        p = db.query(Profile).filter(
            Profile.id == profile_id,
            Profile.user_id == current_user.id,
        ).first()
        if not p:
            raise HTTPException(status_code=404, detail="Profile not found")
        d = _profile_to_dict(p)
        audit_count = db.query(Audit).filter(Audit.profile_id == p.id).count()
        d["audit_count"] = audit_count
        latest = (
            db.query(Audit)
            .filter(Audit.profile_id == p.id)
            .order_by(Audit.created_at.desc())
            .first()
        )
        d["latest_audit"] = {
            "id": latest.id,
            "version": latest.version,
            "keyword": latest.keyword,
            "status": latest.status,
            "created_at": latest.created_at.isoformat() if latest.created_at else None,
        } if latest else None
        return d
    finally:
        db.close()


@app.patch("/profiles/{profile_id}")
def update_profile(profile_id: str, body: ProfileUpdateRequest, current_user: CurrentUser = Depends(get_current_user)):
    """Partial update of a profile."""
    db = SessionLocal()
    try:
        p = db.query(Profile).filter(
            Profile.id == profile_id,
            Profile.user_id == current_user.id,
        ).first()
        if not p:
            raise HTTPException(status_code=404, detail="Profile not found")
        if body.business_name is not None:
            p.business_name = body.business_name
        if body.website_url is not None:
            p.website_url = body.website_url
        if body.business_category is not None:
            p.business_category = body.business_category
        if body.services is not None:
            p.services = json.dumps(body.services)
        if body.country is not None:
            p.country = body.country
        if body.city is not None:
            p.city = body.city
        if body.nap_data is not None:
            p.nap_data = json.dumps(body.nap_data)
        p.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(p)
        return _profile_to_dict(p)
    finally:
        db.close()


@app.delete("/profiles/{profile_id}")
def delete_profile(profile_id: str, current_user: CurrentUser = Depends(get_current_user)):
    """Soft-delete a profile (sets is_active=False)."""
    db = SessionLocal()
    try:
        p = db.query(Profile).filter(
            Profile.id == profile_id,
            Profile.user_id == current_user.id,
        ).first()
        if not p:
            raise HTTPException(status_code=404, detail="Profile not found")
        p.is_active = False
        p.updated_at = datetime.utcnow()
        db.commit()
        return {"status": "deleted", "id": profile_id}
    finally:
        db.close()


@app.get("/profiles/{profile_id}/audits")
def list_profile_audits(profile_id: str, current_user: CurrentUser = Depends(get_current_user)):
    """List all audits for a profile (version descending, metadata only)."""
    db = SessionLocal()
    try:
        # Verify ownership
        p = db.query(Profile).filter(
            Profile.id == profile_id,
            Profile.user_id == current_user.id,
        ).first()
        if not p:
            raise HTTPException(status_code=404, detail="Profile not found")
        rows = (
            db.query(Audit)
            .filter(Audit.profile_id == profile_id)
            .order_by(Audit.version.desc())
            .all()
        )
        return [
            {
                "id": r.id,
                "version": r.version,
                "keyword": r.keyword,
                "target_url": r.target_url,
                "location": r.location,
                "status": r.status,
                "business_name": r.business_name,
                "business_type": r.business_type,
                "pages_crawled": r.pages_crawled,
                "api_cost": r.api_cost,
                "execution_time": r.execution_time,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
    finally:
        db.close()


@app.get("/profiles/{profile_id}/audits/latest")
def get_profile_latest_audit(profile_id: str, current_user: CurrentUser = Depends(get_current_user)):
    """Return full results of the most recent audit for a profile."""
    db = SessionLocal()
    try:
        p = db.query(Profile).filter(
            Profile.id == profile_id,
            Profile.user_id == current_user.id,
        ).first()
        if not p:
            raise HTTPException(status_code=404, detail="Profile not found")
        latest = (
            db.query(Audit)
            .filter(Audit.profile_id == profile_id)
            .order_by(Audit.created_at.desc())
            .first()
        )
        if not latest:
            raise HTTPException(status_code=404, detail="No audits found for this profile")
        return json.loads(latest.results_json)
    finally:
        db.close()


@app.get("/profiles/{profile_id}/audits/{audit_id}")
def get_profile_audit(profile_id: str, audit_id: str, current_user: CurrentUser = Depends(get_current_user)):
    """Return full results of a specific audit for a profile."""
    db = SessionLocal()
    try:
        p = db.query(Profile).filter(
            Profile.id == profile_id,
            Profile.user_id == current_user.id,
        ).first()
        if not p:
            raise HTTPException(status_code=404, detail="Profile not found")
        row = db.query(Audit).filter(
            Audit.id == audit_id,
            Audit.profile_id == profile_id,
        ).first()
        if not row:
            raise HTTPException(status_code=404, detail="Audit not found")
        return json.loads(row.results_json)
    finally:
        db.close()


# =============================================================================
# Utility functions
# =============================================================================

def extract_json(text: str) -> dict | list:
    """
    Robustly extract JSON from Claude responses.
    Handles markdown fences, preamble text, trailing commentary,
    and truncated JSON (from max_tokens cutoff).
    Returns a dict or list; wraps unparseable text in {"raw_response": text}.
    """
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        text = text.rsplit("```", 1)[0].strip()

    # Try direct parse (handles both objects and arrays)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the outermost JSON object or array
    obj_start = text.find("{")
    arr_start = text.find("[")

    if obj_start == -1 and arr_start == -1:
        return {"raw_response": text}

    if arr_start != -1 and (obj_start == -1 or arr_start < obj_start):
        start, open_c, close_c = arr_start, "[", "]"
    else:
        start, open_c, close_c = obj_start, "{", "}"

    # String-aware brace counting
    in_string = False
    escape = False
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_c:
            depth += 1
        elif ch == close_c:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    break

    # Truncated JSON repair: close unclosed strings, arrays, and objects
    fragment = text[start:]
    repaired = _repair_truncated_json(fragment)
    if repaired is not None:
        return repaired

    return {"raw_response": text}


def _is_claude_error(result) -> bool:
    """Check if a call_claude() result is an error dict rather than real data."""
    if isinstance(result, dict):
        if "error" in result and len(result) == 1:
            return True
        if "raw_response" in result and len(result) == 1:
            return True
    return False


def _repair_truncated_json(fragment: str) -> dict | list | None:
    """Attempt to repair JSON truncated by max_tokens cutoff."""
    # Trim trailing incomplete value (e.g. cut-off string or number)
    # Find the last complete key-value separator
    trimmed = fragment.rstrip()

    # Close unclosed string
    in_str = False
    escape = False
    for ch in trimmed:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str

    if in_str:
        trimmed += '"'

    # Remove trailing comma
    trimmed = trimmed.rstrip().rstrip(",")

    # Count unclosed braces/brackets (string-aware)
    stack = []
    in_str = False
    escape = False
    for ch in trimmed:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch in ('{', '['):
            stack.append(ch)
        elif ch == '}' and stack and stack[-1] == '{':
            stack.pop()
        elif ch == ']' and stack and stack[-1] == '[':
            stack.pop()

    # Close all unclosed brackets/braces
    closers = {'[': ']', '{': '}'}
    for opener in reversed(stack):
        trimmed += closers[opener]

    try:
        return json.loads(trimmed)
    except json.JSONDecodeError:
        return None


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


def _normalize_url(url: str) -> str:
    """Lowercase URL, strip fragment and trailing slash — used for crawl deduplication."""
    from urllib.parse import urlparse, urlunparse
    p = urlparse(url)
    path = p.path.rstrip("/") or "/"
    return urlunparse((p.scheme, p.netloc.lower(), path, "", "", ""))


async def crawl_site(start_url: str, max_pages: int = 50) -> list[dict]:
    """Crawl a site using sitemap URLs first, then BFS fallback for discovery.

    Strategy:
    1. Fetch sitemap.xml and extract URLs (same domain, HTML only)
    2. Seed the crawl queue with sitemap URLs
    3. If no sitemap or empty, fall back to BFS from start_url
    4. Always include the start_url (homepage) first

    Fetches pages in batches of 5 concurrently.
    Returns list of scrape_page() dicts where success=True.
    """
    from urllib.parse import urlparse

    parsed_start = urlparse(start_url)
    base_netloc = parsed_start.netloc
    base_scheme = parsed_start.scheme

    visited: set[str] = set()
    queue: list[str] = [start_url]
    pages: list[dict] = []

    # Try sitemap-first strategy: seed queue with sitemap URLs
    crawl_strategy = "bfs"
    try:
        robots_sitemap = await fetch_robots_and_sitemap(start_url)
        sitemap_urls = robots_sitemap.get("sitemap_xml", {}).get("urls", [])
        if sitemap_urls:
            crawl_strategy = "sitemap"
            # Add sitemap URLs to queue (after start_url to keep homepage first)
            for u in sitemap_urls:
                norm = _normalize_url(u)
                start_norm = _normalize_url(start_url)
                if norm != start_norm and norm not in visited:
                    queue.append(u)
            logger.info(f"Sitemap seeded queue with {len(sitemap_urls)} URLs for {start_url}")
    except Exception as e:
        logger.warning(f"Sitemap fetch failed during crawl, using BFS only: {e}")

    while queue and len(pages) < max_pages:
        # Build a batch of up to 5 unvisited URLs
        batch: list[str] = []
        while queue and len(batch) < 5:
            url = queue.pop(0)
            norm = _normalize_url(url)
            if norm not in visited:
                visited.add(norm)
                batch.append(url)

        if not batch:
            break

        results = await asyncio.gather(*[scrape_page(u) for u in batch], return_exceptions=True)

        for result in results:
            if isinstance(result, Exception) or not isinstance(result, dict):
                continue
            if not result.get("success"):
                continue
            pages.append(result)

            # Discover new internal links (BFS expansion — works for both strategies)
            for link in result.get("internal_links", []):
                href = link.get("href", "")
                if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
                    continue
                if href.startswith("/"):
                    full_url = f"{base_scheme}://{base_netloc}{href}"
                elif base_netloc in href:
                    full_url = href
                else:
                    continue

                # Skip non-HTML resources
                if any(href.endswith(ext) for ext in (".pdf", ".jpg", ".png", ".gif", ".svg", ".zip", ".xml")):
                    continue

                norm = _normalize_url(full_url)
                if norm not in visited:
                    queue.append(full_url)

    logger.info(f"Site crawl complete: {len(pages)} pages from {start_url} (strategy={crawl_strategy})")
    return pages


def detect_page_type(url: str, title: str = "", location: str = "") -> str:
    """Rule-based classification of a page by its URL path and title.

    Returns a human-readable page type string. No Claude calls.
    """
    from urllib.parse import urlparse
    path = urlparse(url).path.lower().rstrip("/")
    title_lower = title.lower()
    combined = f"{path} {title_lower}"

    # Homepage
    if path in ("", "/"):
        return "Homepage"

    # Blog / News / Article
    if any(kw in combined for kw in ("blog", "news", "article", "post", "/category/")):
        return "Blog Post"

    # About
    if any(kw in combined for kw in ("about", "our-story", "our-team", "who-we-are")):
        return "About"

    # Contact
    if any(kw in combined for kw in ("contact", "get-in-touch", "reach-us")):
        return "Contact"

    # FAQ
    if "faq" in combined or "frequently-asked" in combined:
        return "FAQ"

    # Pricing
    if "pricing" in combined or "price" in combined or "rates" in combined:
        return "Pricing"

    # Portfolio / Gallery
    if any(kw in combined for kw in ("portfolio", "gallery", "projects", "our-work", "case-stud")):
        return "Portfolio"

    # Testimonials / Reviews
    if any(kw in combined for kw in ("testimonial", "review", "feedback", "client-stories")):
        return "Testimonials"

    # Service area — check if location city name appears in path
    if location:
        city = location.split(",")[0].strip().lower()
        city_slug = city.replace(" ", "-")
        if city_slug in path or city in path:
            return "Service Area"

    # Services (check after more specific types)
    if any(kw in combined for kw in ("service", "what-we-do", "solutions", "offerings")):
        return "Services"

    # Careers
    if any(kw in combined for kw in ("career", "jobs", "hiring", "join-us")):
        return "Careers"

    # Privacy / Legal
    if any(kw in combined for kw in ("privacy", "terms", "legal", "policy", "disclaimer")):
        return "Legal"

    return "Page"


def aggregate_crawl_results(pages: list[dict]) -> dict:
    """Compute site-wide on-page aggregate stats from crawled pages."""
    total = len(pages)
    if not total:
        return {"pages_crawled": 0, "page_types": {}}

    missing_title = sum(1 for p in pages if not p.get("title"))
    missing_meta = sum(1 for p in pages if not p.get("meta_description"))
    missing_h1 = sum(1 for p in pages if not p.get("h1"))
    avg_word_count = round(sum(p.get("word_count", 0) for p in pages) / total)
    thin_pages = [p["url"] for p in pages if p.get("word_count", 0) < 300]
    coverage_score = round(
        100 - ((missing_title + missing_meta + missing_h1) / (total * 3)) * 100
    )

    # Page type breakdown
    page_types: dict[str, int] = {}
    for p in pages:
        pt = p.get("page_type", "Page")
        page_types[pt] = page_types.get(pt, 0) + 1

    return {
        "pages_crawled": total,
        "missing_title": missing_title,
        "missing_meta_description": missing_meta,
        "missing_h1": missing_h1,
        "avg_word_count": avg_word_count,
        "thin_content_count": len(thin_pages),
        "thin_content_pages": thin_pages[:10],
        "coverage_score": max(0, coverage_score),
        "page_types": page_types,
    }


async def fetch_moz_metrics(url: str) -> dict | None:
    """
    Fetch DA, PA, backlink count, and referring domains from Moz Links API.
    Returns None if Moz keys are not set, quota is exhausted, or request fails.
    When None is returned, the backlink agent falls back to Claude estimation.
    """
    if not MOZ_ACCESS_ID or not MOZ_SECRET_KEY:
        return None

    from urllib.parse import urlparse
    import base64

    # Normalise to root domain for DA lookup
    parsed = urlparse(url)
    root_domain = f"{parsed.scheme}://{parsed.netloc}/"

    try:
        credentials = base64.b64encode(
            f"{MOZ_ACCESS_ID}:{MOZ_SECRET_KEY}".encode()
        ).decode()

        async with httpx.AsyncClient(timeout=10.0) as http:
            resp = await http.post(
                "https://lsapi.seomoz.com/v2/url_metrics",
                headers={
                    "Authorization": f"Basic {credentials}",
                    "Content-Type": "application/json",
                },
                json={
                    "targets": [root_domain],
                },
            )

        if resp.status_code == 429:
            logger.warning("Moz API quota exhausted — falling back to Claude estimation")
            return None
        if resp.status_code != 200:
            logger.warning(f"Moz API returned {resp.status_code} — falling back to Claude estimation")
            return None

        data = resp.json()
        result = data.get("results", [{}])[0]

        return {
            "domain_authority": result.get("domain_authority"),
            "page_authority": result.get("page_authority"),
            "spam_score": result.get("spam_score"),
            "linking_domains": result.get("linking_domains"),
            "links": result.get("links"),
            "data_source": "verified",
        }

    except Exception as e:
        logger.warning(f"Moz API error for {url}: {e} — falling back to Claude estimation")
        return None


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


async def check_broken_links(base_url: str, soup: "BeautifulSoup", limit: int = 20) -> dict:
    """Check internal links for 4xx/5xx errors. Capped at `limit` links."""
    from urllib.parse import urlparse, urljoin

    parsed_base = urlparse(base_url)
    base_domain = f"{parsed_base.scheme}://{parsed_base.netloc}"

    # Collect unique internal hrefs only
    seen: set[str] = set()
    internal_links: list[str] = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        full = urljoin(base_url, href)
        parsed = urlparse(full)
        if parsed.netloc == parsed_base.netloc and full not in seen:
            seen.add(full)
            internal_links.append(full)
        if len(internal_links) >= limit:
            break

    broken: list[dict] = []
    if internal_links:
        try:
            async with httpx.AsyncClient(
                timeout=8.0,
                follow_redirects=True,
                headers={"User-Agent": "SEOSaasBot/1.0"},
            ) as http:
                tasks = [http.head(lnk) for lnk in internal_links]
                responses = await asyncio.gather(*tasks, return_exceptions=True)

            for lnk, resp in zip(internal_links, responses):
                if isinstance(resp, Exception):
                    broken.append({"url": lnk, "status": "error", "detail": str(resp)[:80]})
                elif resp.status_code in (404, 410, 500, 502, 503):
                    broken.append({"url": lnk, "status": resp.status_code})
        except Exception as e:
            logger.warning(f"Broken link check failed: {e}")

    return {
        "total_checked": len(internal_links),
        "broken_count": len(broken),
        "broken_links": broken[:10],  # cap for prompt
    }


async def check_redirect_chain(url: str) -> dict:
    """Follow redirects manually and report chain length."""
    chain: list[str] = [url]
    try:
        async with httpx.AsyncClient(
            timeout=10.0,
            follow_redirects=False,
            headers={"User-Agent": "SEOSaasBot/1.0"},
        ) as http:
            current = url
            for _ in range(6):  # max hops we'll follow
                resp = await http.get(current)
                if resp.is_redirect:
                    next_url = resp.headers.get("location", "")
                    if not next_url or next_url == current:
                        break
                    from urllib.parse import urljoin
                    next_url = urljoin(current, next_url)
                    chain.append(next_url)
                    current = next_url
                else:
                    break
    except Exception as e:
        logger.warning(f"Redirect chain check failed for {url}: {e}")

    hops = len(chain) - 1
    return {
        "hops": hops,
        "chain": chain,
        "status": "ok" if hops <= 1 else ("warn" if hops == 2 else "fail"),
    }


async def fetch_robots_and_sitemap(base_url: str) -> dict:
    """Fetch /robots.txt and /sitemap.xml, report existence and key details.

    Parses actual URLs from sitemap XML, filtering to same domain and HTML pages.
    Handles sitemap index files by fetching the first child sitemap.
    """
    from urllib.parse import urlparse
    parsed = urlparse(base_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    base_netloc = parsed.netloc.lower()

    NON_HTML_EXTS = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".zip", ".xml", ".gz",
                     ".mp4", ".mp3", ".avi", ".doc", ".docx", ".xls", ".xlsx", ".css", ".js")

    def _parse_sitemap_urls(xml_text: str) -> list[str]:
        """Extract <loc> URLs from sitemap XML, filter to same domain + HTML only."""
        raw_urls = re.findall(r"<loc>\s*(.*?)\s*</loc>", xml_text, re.IGNORECASE)
        urls = []
        for u in raw_urls:
            u = u.strip()
            try:
                p = urlparse(u)
            except Exception:
                continue
            if p.netloc.lower() != base_netloc:
                continue
            if any(p.path.lower().endswith(ext) for ext in NON_HTML_EXTS):
                continue
            urls.append(u)
        return urls

    result = {
        "robots_txt": {"exists": False, "blocks_important": False, "content_preview": None},
        "sitemap_xml": {"exists": False, "url_count": 0, "sitemap_url": None, "urls": []},
    }

    try:
        async with httpx.AsyncClient(
            timeout=10.0,
            follow_redirects=True,
            headers={"User-Agent": "SEOSaasBot/1.0"},
        ) as http:
            robots_resp, sitemap_resp = await asyncio.gather(
                http.get(f"{origin}/robots.txt"),
                http.get(f"{origin}/sitemap.xml"),
                return_exceptions=True,
            )

            # robots.txt
            if not isinstance(robots_resp, Exception) and robots_resp.status_code == 200:
                content = robots_resp.text
                result["robots_txt"]["exists"] = True
                result["robots_txt"]["content_preview"] = content[:300].replace("\r", "").replace("\t", " ")
                lines_lower = content.lower()
                result["robots_txt"]["blocks_important"] = (
                    "disallow: /" in lines_lower and "allow: /" not in lines_lower
                )

            # sitemap.xml
            if not isinstance(sitemap_resp, Exception) and sitemap_resp.status_code == 200:
                result["sitemap_xml"]["exists"] = True
                result["sitemap_xml"]["sitemap_url"] = f"{origin}/sitemap.xml"
                xml_text = sitemap_resp.text

                # Check if this is a sitemap index (<sitemapindex>)
                if "<sitemapindex" in xml_text.lower():
                    child_urls = re.findall(r"<loc>\s*(.*?)\s*</loc>", xml_text, re.IGNORECASE)
                    if child_urls:
                        # Fetch the first child sitemap
                        try:
                            child_resp = await http.get(child_urls[0].strip())
                            if child_resp.status_code == 200:
                                xml_text = child_resp.text
                        except Exception:
                            pass  # fall through with original xml_text

                sitemap_urls = _parse_sitemap_urls(xml_text)
                result["sitemap_xml"]["url_count"] = len(sitemap_urls)
                result["sitemap_xml"]["urls"] = sitemap_urls

    except Exception as e:
        logger.warning(f"Robots/sitemap fetch failed: {e}")

    return result


async def scrape_technical_signals(url: str) -> dict:
    """Extract technical SEO signals from the raw HTML of a page."""
    from urllib.parse import urlparse

    signals: dict = {
        "url": url,
        "success": False,
        "https": urlparse(url).scheme == "https",
        "redirect_chain": {},
        "viewport": None,
        "canonical": None,
        "robots_meta": None,
        "og_tags": {},
        "twitter_tags": {},
        "schemas": [],
        "images_total": 0,
        "images_missing_alt": [],
        "render_blocking_scripts": 0,
        "lazy_loaded_images": 0,
        "inline_style_bytes": 0,
        "html_text_ratio": None,
        "broken_links": {},
        "robots_txt": {},
        "sitemap_xml": {},
    }

    try:
        # Redirect chain + main page fetch run concurrently
        async with httpx.AsyncClient(
            timeout=12.0,
            follow_redirects=True,
            headers={"User-Agent": "SEOSaasBot/1.0"},
        ) as http:
            resp = await http.get(url)

        html = resp.text
        soup = BeautifulSoup(html, "html.parser")
        signals["success"] = True

        # Run redirect check + robots/sitemap concurrently while parsing HTML
        redirect_task = check_redirect_chain(url)
        robots_task = fetch_robots_and_sitemap(url)
        broken_task = check_broken_links(url, soup, limit=20)

        # --- Parse HTML signals ---

        # Viewport
        vp = soup.find("meta", attrs={"name": "viewport"})
        signals["viewport"] = vp.get("content", "") if vp else None

        # Canonical
        canon = soup.find("link", rel="canonical")
        signals["canonical"] = canon.get("href", "") if canon else None

        # Robots meta
        robots_meta = soup.find("meta", attrs={"name": "robots"})
        signals["robots_meta"] = robots_meta.get("content", "") if robots_meta else None

        # OG tags
        og = {}
        for tag in soup.find_all("meta", property=lambda p: p and p.startswith("og:")):
            og[tag.get("property")] = tag.get("content", "")[:100]
        signals["og_tags"] = og  # e.g. {"og:title": "...", "og:description": "..."}

        # Twitter tags
        tw = {}
        for tag in soup.find_all("meta", attrs={"name": lambda n: n and n.startswith("twitter:")}):
            tw[tag.get("name")] = tag.get("content", "")[:100]
        signals["twitter_tags"] = tw

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
        ][:20]
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

        # HTML/text ratio
        html_bytes = len(html.encode("utf-8"))
        visible_text = soup.get_text(separator=" ", strip=True)
        text_bytes = len(visible_text.encode("utf-8"))
        ratio_pct = round((text_bytes / html_bytes) * 100, 1) if html_bytes else 0
        signals["html_text_ratio"] = {
            "html_bytes": html_bytes,
            "text_bytes": text_bytes,
            "ratio_pct": ratio_pct,
            "status": "ok" if ratio_pct >= 10 else "warn",
        }

        # Await the concurrent tasks
        redirect_result, robots_result, broken_result = await asyncio.gather(
            redirect_task, robots_task, broken_task
        )
        signals["redirect_chain"] = redirect_result
        signals["robots_txt"] = robots_result["robots_txt"]
        signals["sitemap_xml"] = robots_result["sitemap_xml"]
        signals["broken_links"] = broken_result

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


async def fetch_competitors(keyword: str, location: str, num: int = 3) -> list[dict]:
    """Fetch top organic competitors from SerpApi. num controls how many results to return."""
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
                    "num": min(num, 10),  # one API call regardless of num
                },
            )
            resp.raise_for_status()
            data = resp.json()

        competitors = []
        for r in data.get("organic_results", [])[:num]:
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
    retries: int = 3,
    return_raw: bool = False,
) -> dict | str:
    """
    Call Claude with a system prompt and user prompt.
    Retries on transient failures with backoff.
    On rate-limit (429) errors, waits 30 s before retrying.
    Returns parsed JSON dict by default, or raw text string if return_raw=True.
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
            if return_raw:
                return raw
            return extract_json(raw)

        except Exception as e:
            last_error = e
            err_str = str(e)
            # Don't retry on billing/credit errors — they'll never succeed
            is_billing = "credit balance" in err_str.lower() or "billing" in err_str.lower()
            if is_billing:
                logger.error(f"Claude billing error (not retrying): {e}")
                break
            is_rate_limit = "rate_limit" in err_str.lower() or "429" in err_str
            wait = 30 if is_rate_limit else 2 * attempt
            logger.warning(
                f"Claude call attempt {attempt}/{retries} failed "
                f"({'rate limit — waiting 30 s' if is_rate_limit else f'retrying in {wait} s'}): {e}"
            )
            if attempt < retries:
                await asyncio.sleep(wait)

    logger.error(f"Claude call failed after {retries} attempts: {last_error}")
    return "" if return_raw else {"error": str(last_error)}


# =============================================================================
# Auto-detect keyword — scrapes site + asks Claude to infer business type/keyword
# =============================================================================

AUTODETECT_SYSTEM = """You are an expert SEO analyst. Given scraped website data, you detect the business type and the best primary keyword a potential customer would Google to find this business locally.
You ALWAYS respond with valid JSON only — no markdown, no explanation, no preamble."""

AUTODETECT_PROMPT = """Analyze this website and determine the business type and best local search keyword.

BUSINESS NAME (user-provided): {business_name}
LOCATION: {location}
WEBSITE URL: {target_url}

SCRAPED PAGE DATA:
- Page title: {title}
- Meta description: {meta_description}
- H1: {h1}
- Headings: {headings}
- Content preview (first 1500 chars): {content}

Based on the page content, return JSON with EXACTLY these keys:
{{
  "business_type": "the specific business category, e.g. 'Kitchen Remodeler', 'Family Dentist', 'Personal Injury Lawyer'",
  "primary_keyword": "the #1 keyword a local customer would Google to find this business, e.g. 'kitchen remodeler near me', 'dentist near me'",
  "secondary_keywords": ["3-5 additional high-intent local keywords"],
  "services": ["list of main services offered on the page"],
  "detected_name": "the business name as it appears on the website"
}}

Rules:
- primary_keyword MUST include a local modifier like 'near me' or the city name
- business_type should be specific (not just 'business' or 'company')
- secondary_keywords should be service-specific local searches
- If the page content is thin or unclear, make your best inference from the URL, title, and business name"""


async def auto_detect_keyword(request: AuditRequest) -> dict:
    """Scrape the target URL and ask Claude to detect business type + primary keyword."""
    logger.info(f"Auto-detecting keyword for {request.target_url}")

    try:
        page_data = await scrape_page(request.target_url)

        if not page_data.get("success"):
            logger.warning(f"Scrape failed for auto-detect — using fallback")
            return _fallback_detection(request)

        headings_text = ", ".join(
            f"{h['level']}: {h['text']}" for h in page_data.get("headings", [])[:15]
        )

        prompt = AUTODETECT_PROMPT.format(
            business_name=request.business_name or "Unknown",
            location=request.location,
            target_url=request.target_url,
            title=page_data.get("title", "N/A"),
            meta_description=page_data.get("meta_description", "N/A"),
            h1=page_data.get("h1", "N/A"),
            headings=headings_text or "N/A",
            content=page_data.get("content", "")[:1500],
        )

        result = await call_claude(AUTODETECT_SYSTEM, prompt, max_tokens=500)

        if isinstance(result, dict) and result.get("primary_keyword"):
            logger.info(f"Auto-detected: type='{result.get('business_type')}', keyword='{result['primary_keyword']}'")
            return result

        logger.warning("Claude auto-detect returned empty — using fallback")
        return _fallback_detection(request)

    except Exception as e:
        logger.error(f"Auto-detect failed: {e}")
        return _fallback_detection(request)


def _fallback_detection(request: AuditRequest) -> dict:
    """Construct a reasonable keyword when scrape or Claude fails."""
    from urllib.parse import urlparse

    if request.business_type:
        kw = f"{request.business_type} near me"
    elif request.business_name:
        kw = f"{request.business_name} near me"
    else:
        domain = urlparse(request.target_url).netloc.replace("www.", "")
        name = domain.split(".")[0].replace("-", " ").replace("_", " ").title()
        kw = f"{name} near me"

    return {
        "business_type": request.business_type or "local business",
        "primary_keyword": kw,
        "secondary_keywords": [],
        "services": [],
        "detected_name": request.business_name or "",
    }


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
async def keyword_research_agent(request: AuditRequest, secondary_keywords: list[str] | None = None):
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

    # Append auto-detected secondary keywords when available
    if secondary_keywords:
        prompt += f"\n\nAlso analyze these related keywords detected from the website: {', '.join(secondary_keywords)}"

    recommendations = await call_claude(KEYWORD_SYSTEM, prompt, max_tokens=4000)

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
async def on_page_seo_agent(request: AuditRequest, pre_scraped_pages: list[dict] | None = None):
    """On-Page SEO Agent — content audit, meta tags, structure, internal links.

    pre_scraped_pages: pass crawled pages (homepage first) to skip re-fetching.
    """
    audit_id = str(uuid.uuid4())
    logger.info(f"[{audit_id}] On-Page SEO starting for '{request.target_url}'")

    if pre_scraped_pages:
        # Use pre-crawled data; homepage is the first page
        page_data = pre_scraped_pages[0]
        competitors = await fetch_competitors(request.keyword, request.location)
    else:
        page_task = scrape_page(request.target_url)
        comp_task = fetch_competitors(request.keyword, request.location)
        page_data, competitors = await asyncio.gather(page_task, comp_task)

    competitor_data = await scrape_competitors(competitors)

    # Build optional extra context from other crawled pages
    extra_pages_section = ""
    if pre_scraped_pages and len(pre_scraped_pages) > 1:
        lines = ["\nOTHER SITE PAGES (crawled — top service pages by word count):"]
        for p in pre_scraped_pages[1:16]:
            lines.append(
                f"- {p['url']} | Title: {p.get('title','—')} | H1: {p.get('h1','—')} "
                f"| Words: {p.get('word_count', 0)}"
            )
        extra_pages_section = "\n".join(lines)

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
    ) + extra_pages_section

    recommendations = await call_claude(ONPAGE_SYSTEM, prompt, max_tokens=4000)

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
- Redirect chain hops: {redirect_hops} (status: {redirect_status}) — chain: {redirect_chain}
- Viewport meta tag: {viewport}
- Canonical tag: {canonical}
- Robots meta: {robots_meta}
- OG tags present: {og_tags_present} — keys: {og_tag_keys}
- Twitter/X card tags present: {twitter_tags_present} — keys: {twitter_tag_keys}
- Structured data schemas found: {schemas}
- Total images: {images_total}
- Images missing alt text: {images_missing_alt}
- Lazy-loaded images: {lazy_loaded_images}
- Render-blocking scripts in <head>: {render_blocking_scripts}
- Inline CSS bytes: {inline_style_bytes}
- HTML/text ratio: {html_ratio_pct}% ({html_ratio_status}) — {html_bytes} HTML bytes, {text_bytes} visible text bytes
- Broken internal links: {broken_link_count} broken out of {links_checked} checked — {broken_link_urls}
- robots.txt: exists={robots_txt_exists}, blocks important paths={robots_blocks}
- sitemap.xml: exists={sitemap_exists}, URL count={sitemap_url_count}

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
  "redirect_chain": {{
    "hops": 0,
    "status": "pass|warn|fail",
    "recommendation": "Specific fix or confirmation"
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
  "social_tags": {{
    "og_present": true,
    "og_missing": ["og:image", "og:description"],
    "twitter_present": true,
    "twitter_missing": ["twitter:card"],
    "status": "pass|warn|fail",
    "recommendation": "List exact tags to add"
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
  "html_text_ratio": {{
    "ratio_pct": 0,
    "status": "pass|warn",
    "recommendation": "Specific fix or confirmation"
  }},
  "broken_links": {{
    "broken_count": 0,
    "status": "pass|warn|fail",
    "broken_urls": [],
    "recommendation": "Specific fix or confirmation"
  }},
  "crawlability": {{
    "robots_txt_exists": true,
    "robots_blocks_important": false,
    "sitemap_exists": true,
    "sitemap_url_count": 0,
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
    "Most impactful fix first (be specific, reference real data)",
    "Second fix",
    "Third fix",
    "Fourth fix",
    "Fifth fix"
  ],
  "quick_wins": [
    "Fastest fix 1 (can be done today)",
    "Fastest fix 2",
    "Fastest fix 3"
  ]
}}

Score 0–100. Scoring:
- HTTPS: 10pts
- Mobile performance ≥90: 20pts, 50-89: 10pts, <50: 0pts
- Desktop performance ≥90: 10pts, 50-89: 5pts, <50: 0pts
- LCP ≤2.5s: 10pts, ≤4s: 5pts, >4s: 0pts
- CLS ≤0.1: 5pts, >0.25: 0pts
- Viewport present: 3pts | Canonical present: 3pts
- Structured data (LocalBusiness schema): 10pts, other schema: 5pts, none: 0pts
- All images have alt text: 5pts
- Robots.txt + sitemap both exist: 5pts
- No broken links: 5pts
- OG + Twitter tags: 5pts
- Redirect chain ≤1 hop: 4pts
- HTML/text ratio ≥10%: 5pts
- PageSpeed data unavailable: score conservatively, note data was unavailable."""


@app.post("/agents/technical-seo")
async def technical_seo_agent(request: AuditRequest, crawl_aggregate: dict | None = None):
    """Technical SEO Agent — Core Web Vitals (PageSpeed API), HTTPS, mobile, canonical, schema, images.

    crawl_aggregate: pass aggregate_crawl_results() output to include site-wide stats in the analysis.
    """
    audit_id = str(uuid.uuid4())
    logger.info(f"[{audit_id}] Technical SEO starting for '{request.target_url}'")

    try:
        # Fetch PageSpeed data and scrape signals concurrently
        signals, psi = await asyncio.gather(
            scrape_technical_signals(request.target_url),
            fetch_pagespeed(request.target_url),
        )

        mob = psi.get("mobile") or {}
        desk = psi.get("desktop") or {}

        # Format top opportunities as a bullet list
        opps = mob.get("opportunities") or desk.get("opportunities") or []
        opp_lines = "\n".join(
            f"  - {o['title']}: saves ~{o['savings_ms']}ms"
            for o in opps
        ) or "  - No opportunity data available"

        # New signal shorthands — use `or {}` to handle None values
        redir = signals.get("redirect_chain") or {}
        og = signals.get("og_tags") or {}
        tw = signals.get("twitter_tags") or {}
        html_ratio = signals.get("html_text_ratio") or {}
        broken = signals.get("broken_links") or {}
        robots_txt = signals.get("robots_txt") or {}
        sitemap = signals.get("sitemap_xml") or {}
        broken_urls = ", ".join(str(b.get("url", "")) for b in broken.get("broken_links", [])[:5]) or "None"

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
            # Existing scrape signals
            https=signals.get("https", False),
            viewport=signals.get("viewport") or "NOT FOUND",
            canonical=signals.get("canonical") or "NOT FOUND",
            robots_meta=signals.get("robots_meta") or "NOT FOUND",
            schemas=", ".join(signals.get("schemas", [])) or "None found",
            images_total=signals.get("images_total", 0),
            images_missing_alt=signals.get("images_missing_alt") or "None",
            lazy_loaded_images=signals.get("lazy_loaded_images", 0),
            render_blocking_scripts=signals.get("render_blocking_scripts", 0),
            inline_style_bytes=signals.get("inline_style_bytes", 0),
            # New signals
            redirect_hops=redir.get("hops", "N/A"),
            redirect_status=redir.get("status", "N/A"),
            redirect_chain=" → ".join(redir.get("chain", [])) or "N/A",
            og_tags_present=bool(og),
            og_tag_keys=", ".join(og.keys()) or "None",
            twitter_tags_present=bool(tw),
            twitter_tag_keys=", ".join(tw.keys()) or "None",
            html_ratio_pct=html_ratio.get("ratio_pct", "N/A"),
            html_ratio_status=html_ratio.get("status", "N/A"),
            html_bytes=html_ratio.get("html_bytes", "N/A"),
            text_bytes=html_ratio.get("text_bytes", "N/A"),
            broken_link_count=broken.get("broken_count", 0),
            links_checked=broken.get("total_checked", 0),
            broken_link_urls=broken_urls,
            robots_txt_exists=robots_txt.get("exists", False),
            robots_blocks=robots_txt.get("blocks_important", False),
            sitemap_exists=sitemap.get("exists", False),
            sitemap_url_count=sitemap.get("url_count", 0),
        )

        # Append site-wide crawl summary when available
        if crawl_aggregate and crawl_aggregate.get("pages_crawled", 0) > 1:
            agg = crawl_aggregate
            thin_list = ", ".join(agg.get("thin_content_pages", [])[:5]) or "None"
            prompt += f"""

SITE-WIDE CRAWL DATA ({agg['pages_crawled']} pages crawled):
- Pages missing title tag: {agg['missing_title']}
- Pages missing meta description: {agg['missing_meta_description']}
- Pages missing H1: {agg['missing_h1']}
- Average word count across site: {agg['avg_word_count']}
- Thin content pages (<300 words): {agg['thin_content_count']}
- Thin page URLs: {thin_list}
- Site coverage score (meta completeness): {agg['coverage_score']}/100

Factor these site-wide issues into your priority_actions — e.g. if 8/20 pages are missing meta descriptions, that's a site-wide fix."""

        recommendations = await call_claude(TECHNICAL_SYSTEM, prompt, max_tokens=4000)

        return {
            "agent": "technical_seo",
            "audit_id": audit_id,
            "status": "completed",
            "keyword": request.keyword,
            "target_url": request.target_url,
            "page_scraped": signals.get("success", False),
            "pagespeed_fetched": psi.get("success", False),
            "pagespeed": psi,
            "signals": signals,
            "crawl_aggregate": crawl_aggregate or {},
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"[{audit_id}] Technical SEO agent failed: {type(e).__name__}: {e}", exc_info=True)
        raise


# =============================================================================
# Per-Page SEO Analysis (batched Claude calls)
# =============================================================================

PAGE_ANALYSIS_SYSTEM = """You are an expert SEO auditor analysing individual web pages for on-page SEO quality.
You ALWAYS respond with valid JSON only — no markdown, no explanation, no preamble.
Be specific, actionable, and data-driven in your analysis."""

PAGE_ANALYSIS_PROMPT = """Analyse these {page_count} web pages for on-page SEO quality. For each page, evaluate title tag, meta description, H1 tags, word count, keyword usage, and content structure.

BUSINESS: {business_name} ({business_type}) in {location}
TARGET KEYWORD: {keyword}

PAGES TO ANALYSE:
{pages_data}

Respond with a JSON array. For each page return:
{{
  "url": "the page URL",
  "page_type": "detected page type",
  "page_score": 0-100,
  "title_analysis": {{
    "current": "current title",
    "length": number,
    "has_keyword": true/false,
    "has_location": true/false,
    "rating": "good|needs_improvement|poor"
  }},
  "meta_analysis": {{
    "current": "current meta description",
    "length": number,
    "has_keyword": true/false,
    "has_cta": true/false,
    "rating": "good|needs_improvement|poor"
  }},
  "h1_tags": ["list of H1 tags found"],
  "word_count": number,
  "issues": ["list of specific SEO issues found"],
  "recommended_title": "optimised title suggestion (under 60 chars)",
  "recommended_meta": "optimised meta description (under 155 chars)",
  "recommended_keywords": ["3-5 keywords this page should target"],
  "content_recommendation": "one sentence on how to improve this page's content"
}}"""


async def analyze_pages_batch(
    pages: list[dict],
    request,
    batch_size: int = 4,
) -> dict:
    """Run per-page SEO analysis in batched Claude calls.

    Splits pages into batches of `batch_size`, runs all batches concurrently,
    and returns a dict mapping URL path → page analysis.
    Individual batch failures are caught and logged, not propagated.
    """
    from urllib.parse import urlparse

    if not pages:
        return {}

    # Build batches
    batches: list[list[dict]] = []
    for i in range(0, len(pages), batch_size):
        batches.append(pages[i:i + batch_size])

    async def _analyze_batch(batch: list[dict]) -> list[dict]:
        """Analyze a single batch of pages via one Claude call."""
        pages_data_lines = []
        for idx, p in enumerate(batch, 1):
            pages_data_lines.append(
                f"Page {idx}:\n"
                f"  URL: {p.get('url', 'N/A')}\n"
                f"  Title: {p.get('title', 'N/A')}\n"
                f"  Meta Description: {p.get('meta_description', 'N/A')}\n"
                f"  H1: {p.get('h1', 'N/A')}\n"
                f"  Word Count: {p.get('word_count', 0)}\n"
                f"  Headings: {json.dumps(p.get('headings', [])[:8])}\n"
                f"  Content Preview: {p.get('content', '')[:400]}"
            )

        prompt = PAGE_ANALYSIS_PROMPT.format(
            page_count=len(batch),
            business_name=request.business_name or "this business",
            business_type=request.business_type or "local business",
            location=request.location,
            keyword=request.keyword,
            pages_data="\n\n".join(pages_data_lines),
        )

        result = await call_claude(PAGE_ANALYSIS_SYSTEM, prompt, max_tokens=3000)

        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "pages" in result:
            return result["pages"]
        return []

    # Run all batches concurrently
    batch_results = await asyncio.gather(
        *[_analyze_batch(b) for b in batches],
        return_exceptions=True,
    )

    # Merge results into a dict keyed by URL path
    page_analysis: dict = {}
    for i, result in enumerate(batch_results):
        if isinstance(result, Exception):
            logger.warning(f"Page analysis batch {i+1}/{len(batches)} failed: {result}")
            continue
        if not isinstance(result, list):
            continue
        for entry in result:
            if isinstance(entry, dict) and entry.get("url"):
                path = urlparse(entry["url"]).path.rstrip("/") or "/"
                page_analysis[path] = entry

    logger.info(f"Per-page analysis complete: {len(page_analysis)}/{len(pages)} pages analyzed")
    return page_analysis


# =============================================================================
# AGENT 5 — Content Rewriter
# =============================================================================

REWRITER_SYSTEM = """You are an expert SEO content strategist and copywriter specialising in local business websites.
You reverse-engineer why top-ranking pages rank, then write better content that outranks them.
Every rewrite is: data-driven from competitor analysis, locally targeted, keyword-optimised at 1-2% density, and written to convert visitors into customers."""

# Call 1: compact JSON — benchmark analysis + SEO template (no long prose)
REWRITER_ANALYSIS_PROMPT = """Analyse top-ranking competitor pages for this keyword. Respond with valid JSON only.

BUSINESS: {business_name} ({business_type}) in {location}
TARGET KEYWORD: {keyword}
CLIENT PAGE: Title={client_title} | H1={client_h1} | Words={client_word_count}

COMPETITOR BENCHMARK ({competitor_count} pages, avg {avg_word_count} words):
Top headings found: {heading_patterns}
{competitor_data}

Return this JSON:
{{
  "benchmark": {{
    "avg_competitor_word_count": {avg_word_count},
    "recommended_word_count": <avg + 200, min 1200>,
    "content_gaps": ["<topic competitors cover that client page is missing>"],
    "client_vs_benchmark": "<one paragraph: specific gaps between client and top competitors>"
  }},
  "seo_template": {{
    "title_tag": "<60 chars max, contains '{keyword}', for {business_name}>",
    "meta_description": "<155 chars max, contains '{keyword}' and a CTA>",
    "h1": "<H1 that contains '{keyword}' naturally>",
    "heading_structure": [
      {{"tag": "H2", "text": "<section heading>"}},
      {{"tag": "H3", "text": "<sub-heading>"}}
    ],
    "target_word_count": 0,
    "keywords_to_include": ["{keyword}", "<secondary>", "<location modifier>"],
    "schema_to_implement": "LocalBusiness + FAQPage"
  }},
  "internal_links": [
    {{"anchor_text": "<text>", "target_path": "/slug", "suggested_placement": "<section>"}}
  ],
  "quick_wins": ["<immediate fix 1>", "<fix 2>", "<fix 3>"]
}}"""

# Call 2: plain text — full page rewrite (no JSON wrapper = no truncation risk)
REWRITER_CONTENT_PROMPT = """Write a complete, SEO-optimised page for a {business_type} targeting the keyword "{keyword}" in {location}.

BRIEF:
- Business: {business_name}
- Target keyword: {keyword} (use 8-12 times naturally, ~1-2% density)
- Location: {location} (mention at least 5 times)
- Target length: {target_word_count} words minimum
- Heading structure to follow: {heading_structure}
- Key topics to cover: {content_gaps}

Write the full page content now. Rules:
- Output ONLY the article text — no JSON, no preamble, no explanation
- Use [H2: heading text] and [H3: heading text] markers for all headings
- End with a 5-question FAQ section: [Q: question] [A: answer]
- Write naturally for humans first — no keyword stuffing
- Include a strong call-to-action paragraph near the end"""


@app.post("/agents/content-rewriter")
async def content_rewriter_agent(request: AuditRequest):
    """Content Rewriter — builds competitor benchmark then generates full page rewrite (2 Claude calls)."""
    audit_id = str(uuid.uuid4())
    logger.info(f"[{audit_id}] Content Rewriter starting for '{request.keyword}'")

    # Fetch top 10 competitors + client page concurrently (1 SerpApi call)
    competitors, page_data = await asyncio.gather(
        fetch_competitors(request.keyword, request.location, num=10),
        scrape_page(request.target_url),
    )

    # Scrape up to 5 competitor pages concurrently
    top_comps = competitors[:5]
    comp_pages = await asyncio.gather(*[scrape_page(c["url"]) for c in top_comps])

    # Build benchmark
    word_counts: list[int] = []
    all_headings: list[str] = []
    comp_summaries: list[str] = []

    for comp, page in zip(top_comps, comp_pages):
        if not page.get("success"):
            continue
        wc = page.get("word_count", 0)
        if wc > 0:
            word_counts.append(wc)
        headings = [h["text"] for h in page.get("headings", [])[:8]]
        all_headings.extend(headings)
        comp_summaries.append(
            f"#{comp['position']}: {comp['title']}\n"
            f"  URL: {comp['url']}\n"
            f"  Word count: {wc}\n"
            f"  H1: {page.get('h1', 'N/A')}\n"
            f"  Headings: {', '.join(headings[:6])}\n"
            f"  Content preview: {page.get('content', '')[:400]}"
        )

    avg_wc = round(sum(word_counts) / len(word_counts)) if word_counts else 1200
    heading_patterns = ", ".join(all_headings[:12]) or "No heading data"
    comp_data_str = "\n\n".join(comp_summaries) or "No competitor data — write based on business type and location."

    # Call 1: analysis + template (compact JSON, fast)
    analysis_prompt = REWRITER_ANALYSIS_PROMPT.format(
        business_name=request.business_name or "this business",
        business_type=request.business_type or "local business",
        location=request.location,
        keyword=request.keyword,
        client_title=page_data.get("title", "N/A"),
        client_h1=page_data.get("h1", "N/A"),
        client_word_count=page_data.get("word_count", 0),
        competitor_count=len(comp_summaries),
        avg_word_count=avg_wc,
        heading_patterns=heading_patterns,
        competitor_data=comp_data_str,
    )

    analysis = await call_claude(REWRITER_SYSTEM, analysis_prompt, max_tokens=1200)

    # Extract template details to inform the content write
    tmpl = analysis.get("seo_template", {})
    target_wc = tmpl.get("target_word_count") or (avg_wc + 200)
    heading_structure = ", ".join(
        f"{h['tag']}: {h['text']}"
        for h in (tmpl.get("heading_structure") or [])[:8]
    ) or "Use logical H2/H3 structure for the business type"
    content_gaps = "; ".join(
        (analysis.get("benchmark", {}).get("content_gaps") or [])[:5]
    ) or "Cover all main services, local expertise, trust signals, and FAQ"

    # Call 2: full page content as plain text (no JSON = no truncation risk)
    content_prompt = REWRITER_CONTENT_PROMPT.format(
        business_name=request.business_name or "this business",
        business_type=request.business_type or "local business",
        location=request.location,
        keyword=request.keyword,
        target_word_count=max(target_wc, 1200),
        heading_structure=heading_structure,
        content_gaps=content_gaps,
    )

    rewritten_content = await call_claude(
        REWRITER_SYSTEM, content_prompt, max_tokens=4000, return_raw=True
    )

    word_count = len(rewritten_content.split()) if rewritten_content else 0
    logger.info(f"[{audit_id}] Content rewriter done — {word_count} words written")

    return {
        "agent": "content_rewriter",
        "audit_id": audit_id,
        "status": "completed",
        "keyword": request.keyword,
        "target_url": request.target_url,
        "competitors_analyzed": len(comp_summaries),
        "benchmark": {
            "avg_competitor_word_count": avg_wc,
            "client_word_count": page_data.get("word_count", 0),
        },
        "recommendations": {
            **analysis,
            "rewritten_content": rewritten_content,
            "rewritten_word_count": word_count,
        },
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# AGENT 6 — Backlink Analysis
# =============================================================================

BACKLINK_SYSTEM = """You are an expert SEO analyst specialising in backlink profiles and domain authority for local businesses.
You ALWAYS respond with valid JSON only — no markdown, no explanation, no preamble.
When real Moz data is unavailable you estimate domain strength from scraped signals — be conservative and accurate.
Every metric you return includes a data_source field: "verified" (from Moz API) or "estimated" (Claude analysis)."""

BACKLINK_PROMPT = """Analyse the backlink profile and domain authority for this local business website.

BUSINESS: {business_name} ({business_type}) in {location}
TARGET URL: {target_url}
TARGET KEYWORD: {keyword}

CLIENT PAGE SIGNALS (scraped):
- Title: {client_title}
- Word count: {client_word_count}
- Schemas found: {schemas}
- Outbound links: {outbound_links}
- HTTPS: {https}
- Canonical present: {canonical_present}

MOZ DATA (if available):
{moz_data}

COMPETITOR COMPARISON:
{competitor_moz_data}

SERP PRESENCE (how often this domain appears in search results):
{serp_presence}

Estimation methodology when Moz data is unavailable:
- DA 0-20: new site, thin content, few citations, no major directory presence
- DA 20-40: established local site, some directories, moderate content depth
- DA 40-60: strong local presence, many citations, regular content, good schema
- DA 60-80: authority site, featured snippets, press mentions, industry links
- High DA signals: appears in top 3 SERP positions, has schema, BBB/Yelp listed, 1000+ words per page

Return JSON with EXACTLY these keys:
{{
  "domain_authority": {{
    "score": <0-100 int>,
    "data_source": "verified|estimated",
    "assessment": "<one sentence interpretation for a local business owner>",
    "vs_competitors_avg": <competitor avg DA int or null>
  }},
  "page_authority": {{
    "score": <0-100 int>,
    "data_source": "verified|estimated",
    "assessment": "<one sentence>"
  }},
  "backlink_profile": {{
    "total_backlinks": <int or estimated range string like "50-200">,
    "referring_domains": <int or estimated range string>,
    "data_source": "verified|estimated",
    "quality_assessment": "<description of likely backlink quality for this business type>",
    "dofollow_estimate_pct": <estimated % of dofollow links>
  }},
  "competitor_comparison": [
    {{
      "url": "<competitor url>",
      "da": <int>,
      "pa": <int>,
      "data_source": "verified|estimated",
      "gap": "<how client compares — ahead/behind/similar>"
    }}
  ],
  "link_gap_summary": "<one paragraph: where the client's backlink profile is weak vs competitors>",
  "top_issues": [
    "<specific backlink issue 1>",
    "<specific issue 2>",
    "<specific issue 3>"
  ],
  "quick_wins": [
    "<fast backlink win 1 — e.g. claim BBB listing>",
    "<fast win 2>",
    "<fast win 3>"
  ]
}}"""


@app.post("/agents/backlink-analysis")
async def backlink_analysis_agent(request: AuditRequest):
    """Backlink Analysis — Moz API when available, Claude estimation as fallback."""
    audit_id = str(uuid.uuid4())
    logger.info(f"[{audit_id}] Backlink Analysis starting for '{request.target_url}'")

    # Gather: client page scrape + Moz for client + competitors + SERP data concurrently
    competitors = await fetch_competitors(request.keyword, request.location, num=3)

    client_page_task = scrape_page(request.target_url)
    client_moz_task = fetch_moz_metrics(request.target_url)
    comp_moz_tasks = [fetch_moz_metrics(c["url"]) for c in competitors]

    client_page, client_moz, *comp_moz_results = await asyncio.gather(
        client_page_task,
        client_moz_task,
        *comp_moz_tasks,
    )

    # Format Moz data for the prompt
    if client_moz:
        moz_data = (
            f"DA: {client_moz.get('domain_authority')} | "
            f"PA: {client_moz.get('page_authority')} | "
            f"Backlinks: {client_moz.get('links')} | "
            f"Referring domains: {client_moz.get('linking_domains')} | "
            f"Spam score: {client_moz.get('spam_score')}"
        )
    else:
        moz_data = "NOT AVAILABLE — Moz keys not set or quota exceeded. Use Claude estimation from page signals."

    # Format competitor Moz data
    comp_moz_lines = []
    for comp, moz in zip(competitors, comp_moz_results):
        if moz:
            comp_moz_lines.append(
                f"  {comp['url']}: DA={moz.get('domain_authority')} PA={moz.get('page_authority')} "
                f"Links={moz.get('links')} RDs={moz.get('linking_domains')} (verified)"
            )
        else:
            comp_moz_lines.append(
                f"  {comp['url']}: Moz unavailable — estimate from SERP position #{comp['position']} "
                f"and snippet: {comp.get('snippet', '')[:100]}"
            )

    competitor_moz_data = "\n".join(comp_moz_lines) or "No competitor data available."

    # SERP presence: use competitor list as proxy (client's position not fetched separately)
    serp_presence = (
        f"Client domain appeared in SerpApi results: {'yes' if any(request.target_url.split('/')[2] in c['url'] for c in competitors) else 'not in top 3'}. "
        f"Top 3 organic competitors: {', '.join(c['url'] for c in competitors[:3])}"
    )

    # Outbound links from scrape
    ext_links = client_page.get("external_links_count", 0)
    int_links = client_page.get("internal_links_count", 0)
    outbound_summary = f"{int_links} internal, {ext_links} external links"

    prompt = BACKLINK_PROMPT.format(
        business_name=request.business_name or "this business",
        business_type=request.business_type or "local business",
        location=request.location,
        target_url=request.target_url,
        keyword=request.keyword,
        client_title=client_page.get("title", "N/A"),
        client_word_count=client_page.get("word_count", 0),
        schemas="Available via technical agent",
        outbound_links=outbound_summary,
        https=str(request.target_url.startswith("https")),
        canonical_present="Available via technical agent",
        moz_data=moz_data,
        competitor_moz_data=competitor_moz_data,
        serp_presence=serp_presence,
    )

    recommendations = await call_claude(BACKLINK_SYSTEM, prompt, max_tokens=1500)

    return {
        "agent": "backlink_analysis",
        "audit_id": audit_id,
        "status": "completed",
        "keyword": request.keyword,
        "target_url": request.target_url,
        "moz_available": bool(client_moz),
        "moz_data": client_moz,
        "recommendations": recommendations,
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# AGENT 7 — Link Building Strategy
# =============================================================================

LINKBUILDING_SYSTEM = """You are an expert local SEO link building strategist for small and medium local businesses.
You ALWAYS respond with valid JSON only — no markdown, no explanation, no preamble.
You prioritise high-probability wins over ambitious targets: easy directory submissions before guest posts."""

# Call 1: compact opportunity list — no email bodies, fits in ~1500 tokens
LINKBUILDING_OPPS_PROMPT = """Generate a prioritised link building strategy for this local business. Respond with valid JSON only.

BUSINESS: {business_name} ({business_type}) in {location}
TARGET URL: {target_url}
TARGET KEYWORD: {keyword}
DOMAIN AUTHORITY ESTIMATE: {da_estimate} (data_source: {da_source})

TOP ORGANIC COMPETITORS:
{competitor_data}

Return JSON with EXACTLY this structure (no outreach_template fields yet — those come separately):
{{
  "quick_wins": [
    {{"name": "<directory name>", "url": "<registration URL>", "link_type": "directory|citation|profile", "difficulty": "easy", "expected_da": <int>, "reason": "<why this link matters for a {business_type}>"}}
  ],
  "guest_posting": [
    {{"name": "<publication name>", "url": "<site URL>", "link_type": "guest-post", "difficulty": "medium|hard", "expected_da": <int>, "topic_idea": "<specific article title>", "contact_method": "email|contact-form|social"}}
  ],
  "resource_pages": [
    {{"name": "<org name>", "url": "<resource page URL>", "link_type": "resource", "difficulty": "medium", "expected_da": <int>, "reason": "<why {business_name} belongs here>"}}
  ],
  "local_opportunities": [
    {{"name": "<local org or outlet>", "url": "<URL>", "link_type": "local|sponsorship|press", "difficulty": "easy|medium", "expected_da": <int>, "opportunity_type": "chamber|news|sponsorship|community|event", "reason": "<specific local angle in {location}>"}}
  ],
  "competitor_gaps": [
    {{"name": "<site linking to competitors>", "url": "<linking page URL>", "competitor_linked": "<competitor URL>", "link_type": "competitor-gap", "difficulty": "medium", "expected_da": <int>, "angle": "<why {business_name} deserves a link too>"}}
  ],
  "summary": {{
    "total_opportunities": <int>,
    "estimated_da_gain_3mo": "<realistic DA improvement in 3 months>",
    "priority_order": ["<category 1>", "<category 2>", "<category 3>", "<category 4>", "<category 5>"],
    "monthly_link_target": <int>
  }}
}}

Rules:
- quick_wins: 4 real directories a {business_type} should be on (Google Business Profile, Yelp, BBB, industry-specific)
- guest_posting: 3 real publications covering {business_type} topics or {location} local news
- resource_pages: 3 pages (local government, community resource lists, industry associations)
- local_opportunities: 4 (chamber of commerce, local newspaper, sponsor, neighbourhood group in {location})
- competitor_gaps: 3 (infer from competitor URLs what sites likely link to them)"""

# Call 2: outreach email templates for all opportunities
LINKBUILDING_TEMPLATES_PROMPT = """Write ready-to-send outreach email templates for these link building opportunities. Respond with valid JSON only.

BUSINESS: {business_name} ({business_type}) in {location}
WEBSITE: {target_url}

OPPORTUNITIES (numbered — return templates in the SAME ORDER, one per item):
{opportunities_list}

Return JSON as an object with a "templates" array — one entry per opportunity, same order:
{{
  "templates": [
    {{
      "n": 1,
      "outreach_template": {{
        "subject": "<specific subject line for opportunity 1>",
        "body": "<2-3 paragraphs, 60-80 words. Uses {business_name} and mentions their org/site by name. Ends with {target_url}.>"
      }}
    }}
  ]
}}

Rules:
- Return exactly {count} templates, numbered 1 to {count} in the same order as the list above
- Use {business_name} and {target_url} in every template — no generic placeholders
- Every subject line is specific to that opportunity (not generic "Link request")
- Body is professional, concise, and ready to send with zero editing
- For directories/citations: focus on getting listed
- For guest posts: pitch the specific article idea from the opportunity
- For local organisations: emphasise the local community angle in {location}"""


@app.post("/agents/link-building")
async def link_building_agent(request: AuditRequest):
    """Link Building Strategy — 5 categories, full outreach templates, ready to send."""
    audit_id = str(uuid.uuid4())
    logger.info(f"[{audit_id}] Link Building starting for '{request.target_url}'")

    # Fetch competitors + client page concurrently
    competitors, page_data = await asyncio.gather(
        fetch_competitors(request.keyword, request.location, num=5),
        scrape_page(request.target_url),
    )

    # Build competitor summary for context
    comp_lines = []
    for c in competitors[:5]:
        comp_lines.append(
            f"  #{c['position']}: {c['title']}\n"
            f"    URL: {c['url']}\n"
            f"    Snippet: {c.get('snippet', '')[:120]}"
        )
    competitor_data = "\n".join(comp_lines) or "No competitor data — generate recommendations based on business type and location."

    # Try to get DA estimate from Moz; fall back to a rough heuristic
    moz = await fetch_moz_metrics(request.target_url)
    if moz:
        da_estimate = moz.get("domain_authority", 30)
        da_source = "verified"
    else:
        wc = page_data.get("word_count", 0)
        da_estimate = 35 if wc > 800 else 20
        da_source = "estimated"

    biz_name = request.business_name or "this business"
    biz_type = request.business_type or "local business"

    # ── Call 1: compact opportunity list (no email bodies) ────────────────────
    opps_prompt = LINKBUILDING_OPPS_PROMPT.format(
        business_name=biz_name,
        business_type=biz_type,
        location=request.location,
        target_url=request.target_url,
        keyword=request.keyword,
        da_estimate=da_estimate,
        da_source=da_source,
        competitor_data=competitor_data,
    )
    recommendations = await call_claude(LINKBUILDING_SYSTEM, opps_prompt, max_tokens=1800)

    # ── Call 2: outreach templates for every opportunity ──────────────────────
    cats = ["quick_wins", "guest_posting", "resource_pages", "local_opportunities", "competitor_gaps"]

    # Build flat ordered list: (cat, item_ref) so we can match back by index
    all_opps_flat: list[tuple[str, dict]] = []
    numbered_lines: list[str] = []
    for cat in cats:
        for item in recommendations.get(cat, []):
            if isinstance(item, dict):
                n = len(all_opps_flat) + 1
                numbered_lines.append(
                    f"{n}. [{cat}] {item.get('name', 'Unknown')} — {item.get('url', '')}"
                )
                all_opps_flat.append((cat, item))

    if all_opps_flat:
        templates_prompt = LINKBUILDING_TEMPLATES_PROMPT.format(
            business_name=biz_name,
            business_type=biz_type,
            location=request.location,
            target_url=request.target_url,
            opportunities_list="\n".join(numbered_lines),
            count=len(all_opps_flat),
        )
        templates_raw = await call_claude(LINKBUILDING_SYSTEM, templates_prompt, max_tokens=3000)

        # Extract the templates array — may be under "templates" key or be a bare list
        if isinstance(templates_raw, list):
            templates_list = templates_raw
        elif isinstance(templates_raw, dict):
            templates_list = templates_raw.get("templates", [])
            if not templates_list:
                # fall back: first list value
                templates_list = next(
                    (v for v in templates_raw.values() if isinstance(v, list)), []
                )
        else:
            templates_list = []

        # Attach by position (index-based — immune to name mismatch)
        for idx, (_cat, item) in enumerate(all_opps_flat):
            if idx < len(templates_list):
                t = templates_list[idx]
                if isinstance(t, dict):
                    item["outreach_template"] = t.get("outreach_template", {})

    # Count total opportunities
    total = sum(len(recommendations.get(c, [])) for c in cats if isinstance(recommendations.get(c), list))

    return {
        "agent": "link_building",
        "audit_id": audit_id,
        "status": "completed",
        "keyword": request.keyword,
        "target_url": request.target_url,
        "total_opportunities": total,
        "recommendations": recommendations,
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# AGENT 6 — AI SEO Visibility
# =============================================================================

AISEO_SYSTEM = """You are an expert in Answer Engine Optimisation (AEO) — how local businesses appear in AI-generated answers from ChatGPT, Perplexity, Claude, and Google AI Overviews.
You ALWAYS respond with valid JSON only — no markdown, no explanation, no preamble.
Your recommendations are specific and implementable: exact schema templates, exact FAQ copy, exact E-E-A-T improvements."""

AISEO_PROMPT = """Analyse this local business's AI search visibility and return a complete AEO strategy. Respond with valid JSON only.

BUSINESS: {business_name} ({business_type}) in {location}
WEBSITE: {target_url}
KEYWORD: {keyword}

=== SCHEMA MARKUP FOUND ON PAGE ===
Types present: {schema_types_found}
Types missing (common for this business type): {schema_types_missing}
FAQPage schema present: {has_faq_schema}

=== E-E-A-T SIGNALS ===
Author bio visible: {has_author_bio}
Professional credentials visible: {has_credentials}
Reviews/testimonials on page: {reviews_on_page}
About page linked: {has_about_page}
Word count: {word_count}

=== PEOPLE ALSO ASK (from Google) ===
{paa_questions}

=== TOP COMPETITORS (who outrank this business) ===
{competitor_data}

Return JSON with this exact structure:
{{
  "ai_visibility_score": <int 0-100>,
  "score_breakdown": {{
    "schema_markup": <int 0-25, points for JSON-LD schemas>,
    "faq_content": <int 0-20, points for FAQ/Q&A optimisation>,
    "eeat_signals": <int 0-25, points for author, credentials, reviews>,
    "content_depth": <int 0-20, points for word count and topical authority>,
    "local_signals": <int 0-10, points for location mentions, NAP consistency>
  }},
  "ai_mention_likelihood": "low|medium|high",
  "ai_answer_preview": "<2-3 sentence simulation of what an AI assistant would say if asked about this business type in {location}. Write it as if the AI is responding to a user query.>",
  "current_gaps": [
    "<specific gap 1 — why AI tools currently overlook this business>",
    "<specific gap 2>",
    "<specific gap 3>"
  ],
  "priority_actions": [
    {{
      "action": "<specific action>",
      "impact": "high|medium|low",
      "effort": "easy|medium|hard",
      "why": "<why this helps AI tools surface the business>",
      "how": "<specific implementation steps>"
    }}
  ],
  "schema_templates": [
    {{
      "type": "<schema type e.g. FAQPage, LocalBusiness, Review>",
      "priority": "high|medium|low",
      "description": "<what this schema does for AI visibility>",
      "json_ld": "<complete valid JSON-LD string, escaped for JSON, ready to paste into a <script> tag>"
    }}
  ],
  "faq_content": [
    {{
      "question": "<natural language question a customer would ask>",
      "answer": "<direct 2-3 sentence answer mentioning {business_name} and {location}. Written to be featured in AI answers.>",
      "ai_intent": "service-discovery|comparison|how-to|location|pricing"
    }}
  ],
  "summary": {{
    "top_priority": "<the single most impactful thing to do this week>",
    "estimated_score_after_fixes": <realistic score after implementing all actions>,
    "time_to_implement": "<realistic estimate e.g. 2-3 hours>"
  }}
}}

Rules:
- ai_visibility_score: score honestly based on signals above (most local SMBs score 20-50)
- priority_actions: exactly 5 actions, ordered by impact descending
- schema_templates: provide 2-3 schemas most critical for {business_type} (always include FAQPage if missing)
- json_ld in schema_templates must be valid JSON string with {business_name} and {target_url} populated
- faq_content: exactly 8 Q&As targeting the PAA questions above plus common {business_type} questions
- Every answer in faq_content mentions {location} at least once"""


def _extract_schema_and_eeat(soup) -> dict:
    """Extract JSON-LD schema types and E-E-A-T signals from a BeautifulSoup page."""
    import re as _re

    # Schema types from JSON-LD
    schema_types: list[str] = []
    has_faq_schema = False
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            raw = tag.string or ""
            obj = json.loads(raw)
            if isinstance(obj, dict):
                t = obj.get("@type", "")
                if isinstance(t, list):
                    schema_types.extend(t)
                elif t:
                    schema_types.append(t)
                if "FAQPage" in str(t):
                    has_faq_schema = True
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        t = item.get("@type", "")
                        if t:
                            schema_types.append(t if isinstance(t, str) else str(t))
                        if "FAQPage" in str(t):
                            has_faq_schema = True
        except (json.JSONDecodeError, TypeError):
            pass

    # FAQ content on page (non-schema)
    faq_count = len(soup.find_all("details")) + len(
        soup.find_all(class_=_re.compile(r"faq|accordion|question", _re.I))
    )

    # E-E-A-T signals
    text_lower = (soup.get_text() or "").lower()
    has_author_bio = any(w in text_lower for w in ["author", "written by", "about the author"])
    has_credentials = any(
        c in text_lower
        for c in ["dds", "dmd", "bds", "md ", "phd", "rdn", "certified", "licensed", "dr.", "dr "]
    )
    reviews_on_page = (
        text_lower.count("★") + text_lower.count("⭐") > 0
        or bool(soup.find_all(attrs={"itemprop": "review"}))
        or bool(soup.find_all(attrs={"itemprop": "ratingValue"}))
        or "testimonial" in text_lower
    )
    has_about_page = bool(soup.find("a", href=_re.compile(r"about", _re.I)))

    return {
        "schema_types_found": schema_types,
        "has_faq_schema": has_faq_schema,
        "faq_count": faq_count,
        "has_author_bio": has_author_bio,
        "has_credentials": has_credentials,
        "reviews_on_page": reviews_on_page,
        "has_about_page": has_about_page,
    }


async def _fetch_paa(keyword: str, location: str) -> list[dict]:
    """Fetch People Also Ask questions from SerpAPI (1 credit)."""
    if not SERPAPI_KEY:
        return []
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://serpapi.com/search",
                params={
                    "engine": "google",
                    "q": keyword,
                    "location": location,
                    "num": "10",
                    "api_key": SERPAPI_KEY,
                },
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            questions = []
            for item in data.get("related_questions", [])[:8]:
                questions.append({
                    "question": item.get("question", ""),
                    "snippet": item.get("snippet", "")[:200],
                })
            return questions
    except Exception:
        return []


async def _scrape_for_ai_seo(url: str) -> dict:
    """Fetch page HTML and extract schema + E-E-A-T signals for AI SEO analysis."""
    try:
        async with httpx.AsyncClient(
            timeout=12.0,
            follow_redirects=True,
            headers={"User-Agent": "SEOSaasBot/1.0"},
        ) as http:
            resp = await http.get(url)
        soup = BeautifulSoup(resp.text, "html.parser")
        signals = _extract_schema_and_eeat(soup)
        text = soup.get_text(separator=" ", strip=True)
        signals["word_count"] = len(text.split())
        return signals
    except Exception:
        return {
            "schema_types_found": [],
            "has_faq_schema": False,
            "faq_count": 0,
            "has_author_bio": False,
            "has_credentials": False,
            "reviews_on_page": False,
            "has_about_page": False,
            "word_count": 0,
        }


@app.post("/agents/ai-seo")
async def ai_seo_agent(request: AuditRequest):
    """AI SEO Visibility — schema gaps, E-E-A-T signals, PAA optimisation, AEO action plan."""
    audit_id = str(uuid.uuid4())
    logger.info(f"[{audit_id}] AI SEO starting for '{request.target_url}'")

    biz_name = request.business_name or "this business"
    biz_type = request.business_type or "local business"

    # Run all data collection concurrently (no redundant sequential waits)
    soup_signals, competitors, paa_questions = await asyncio.gather(
        _scrape_for_ai_seo(request.target_url),
        fetch_competitors(request.keyword, request.location, num=5),
        _fetch_paa(request.keyword, request.location),
    )

    # Common schema types for this business type that are often missing
    COMMON_SCHEMAS = {
        "dental": ["Dentist", "LocalBusiness", "FAQPage", "Review", "BreadcrumbList", "MedicalOrganization"],
        "restaurant": ["Restaurant", "LocalBusiness", "FAQPage", "Menu", "Review", "BreadcrumbList"],
        "contractor": ["HomeAndConstructionBusiness", "LocalBusiness", "FAQPage", "Review", "BreadcrumbList"],
        "lawyer": ["LegalService", "LocalBusiness", "FAQPage", "Review", "BreadcrumbList"],
        "default": ["LocalBusiness", "FAQPage", "Review", "BreadcrumbList", "WebSite"],
    }
    biz_key = next(
        (k for k in COMMON_SCHEMAS if k in biz_type.lower()), "default"
    )
    expected = COMMON_SCHEMAS[biz_key]
    found = soup_signals.get("schema_types_found", [])
    missing = [s for s in expected if s not in found]

    # Build competitor summary
    comp_lines = []
    for c in competitors[:5]:
        comp_lines.append(f"  #{c['position']}: {c['title']} — {c['url']}")
    competitor_data = "\n".join(comp_lines) or "No competitor data available."

    # Format PAA questions
    paa_str = "\n".join(
        f"  Q: {q['question']}\n  A: {q.get('snippet', 'No snippet')}" for q in paa_questions
    ) or "No PAA data — infer common questions from business type."

    prompt = AISEO_PROMPT.format(
        business_name=biz_name,
        business_type=biz_type,
        location=request.location,
        target_url=request.target_url,
        keyword=request.keyword,
        schema_types_found=", ".join(found) if found else "None detected",
        schema_types_missing=", ".join(missing) if missing else "None — good coverage",
        has_faq_schema=soup_signals.get("has_faq_schema", False),
        has_author_bio=soup_signals.get("has_author_bio", False),
        has_credentials=soup_signals.get("has_credentials", False),
        reviews_on_page=soup_signals.get("reviews_on_page", False),
        has_about_page=soup_signals.get("has_about_page", False),
        word_count=soup_signals.get("word_count", 0),
        paa_questions=paa_str,
        competitor_data=competitor_data,
    )

    analysis = await call_claude(AISEO_SYSTEM, prompt, max_tokens=4000)

    return {
        "agent": "ai_seo",
        "audit_id": audit_id,
        "status": "completed",
        "keyword": request.keyword,
        "target_url": request.target_url,
        "signals_collected": {
            "schema_types_found": found,
            "schema_types_missing": missing,
            "paa_questions_found": len(paa_questions),
            **{k: soup_signals.get(k) for k in ["has_faq_schema", "has_author_bio", "has_credentials", "reviews_on_page", "has_about_page"]},
        },
        "analysis": analysis,
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# AGENT 7 — Blog Writer
# =============================================================================

BLOG_SYSTEM = """You are an expert SEO content writer specialising in blog posts for local businesses.
You research what's already ranking, build a data-driven brief, then write a complete post that outranks it.
Every post is: locally targeted, keyword-optimised at 0.5-1.5% density, structured for featured snippets, and written to build topical authority."""

# Call 1: content brief as compact JSON
BLOG_BRIEF_PROMPT = """Research top-ranking content for this keyword and build a content brief. Respond with valid JSON only.

BUSINESS: {business_name} ({business_type}) in {location}
BLOG KEYWORD: {keyword}

TOP-RANKING CONTENT ANALYSED:
{competitor_data}

Return this JSON:
{{
  "content_brief": {{
    "seo_title": "<60 chars, contains '{keyword}', compelling for click-through>",
    "meta_description": "<155 chars, contains '{keyword}', includes a benefit + CTA>",
    "h1": "<H1 that contains '{keyword}' naturally>",
    "target_word_count": <1500-2500 based on competitor avg>,
    "avg_competitor_length": <avg words from competitor data>,
    "recommended_sections": [
      {{"heading": "<H2 text>", "purpose": "<what this section should cover>"}},
      {{"heading": "<H2 text>", "purpose": "<what this section should cover>"}}
    ],
    "primary_keyword": "{keyword}",
    "semantic_keywords": ["<LSI term 1>", "<LSI term 2>", "<LSI term 3>"],
    "questions_to_answer": ["<question 1>", "<question 2>", "<question 3>", "<question 4>", "<question 5>"],
    "featured_snippet_opportunity": "<one specific question this post can own as a featured snippet>",
    "internal_links": [
      {{"anchor_text": "<text>", "target_path": "/service-page", "placement": "<which section>"}}
    ],
    "external_authority_links": [
      {{"anchor_text": "<text>", "url": "<real authority URL — .gov, .edu, or major industry site>", "reason": "<why cite this>"}}
    ]
  }}
}}"""

# Call 2: full blog post as plain text
BLOG_WRITE_PROMPT = """Write a complete, SEO-optimised blog post for a {business_type} in {location}.

CONTENT BRIEF:
- Title (H1): {h1}
- Target keyword: {keyword} (use {keyword_density_target} times naturally, ~0.5-1.5% density)
- Location: {location} (mention at least 3 times)
- Target length: {target_word_count} words
- Sections to include: {sections}
- Questions to answer: {questions}
- Featured snippet target: {featured_snippet}

Write the complete blog post now. Rules:
- Output ONLY the blog content — no JSON, no preamble, no "Here is the blog post:" intro
- Start directly with the opening paragraph (do NOT repeat the H1 — it comes before the content)
- Use [H2: heading text] and [H3: heading text] markers for all subheadings
- Include a "Frequently Asked Questions" section near the end with 5 Q&As: [Q: question] [A: direct 2-3 sentence answer]
- After the FAQ, include a brief conclusion with a call-to-action
- Write naturally and authoritatively — the goal is to become the definitive local resource"""


@app.post("/agents/blog-writer")
async def blog_writer_agent(request: AuditRequest):
    """Blog Writer — researches top-ranking posts, builds brief, writes complete 1500-2500 word blog."""
    audit_id = str(uuid.uuid4())
    logger.info(f"[{audit_id}] Blog Writer starting for '{request.keyword}'")

    # Fetch top 5 results + scrape concurrently (1 SerpApi call)
    competitors = await fetch_competitors(request.keyword, request.location, num=5)
    comp_pages = await asyncio.gather(*[scrape_page(c["url"]) for c in competitors])

    # Build competitor content summary
    word_counts: list[int] = []
    comp_summaries: list[str] = []
    for comp, page in zip(competitors, comp_pages):
        if not page.get("success"):
            continue
        wc = page.get("word_count", 0)
        if wc > 0:
            word_counts.append(wc)
        headings = [h["text"] for h in page.get("headings", [])[:6]]
        comp_summaries.append(
            f"#{comp['position']}: {comp['title']}\n"
            f"  URL: {comp['url']}\n"
            f"  Word count: {wc}\n"
            f"  Headings: {', '.join(headings)}\n"
            f"  Content preview: {page.get('content', '')[:400]}"
        )

    avg_wc = round(sum(word_counts) / len(word_counts)) if word_counts else 1500
    comp_data_str = "\n\n".join(comp_summaries) or "No competitor data — write authoritatively based on business type and keyword."

    # Call 1: content brief (compact JSON)
    brief_prompt = BLOG_BRIEF_PROMPT.format(
        business_name=request.business_name or "this business",
        business_type=request.business_type or "local business",
        location=request.location,
        keyword=request.keyword,
        competitor_data=comp_data_str,
    )
    brief = await call_claude(BLOG_SYSTEM, brief_prompt, max_tokens=1000)
    cb = brief.get("content_brief", {})

    # Pull values from brief to inform writing
    h1 = cb.get("h1") or f"Complete Guide to {request.keyword} in {request.location}"
    target_wc = cb.get("target_word_count") or max(avg_wc + 200, 1500)
    sections = "; ".join(
        f"{s['heading']} ({s['purpose']})"
        for s in (cb.get("recommended_sections") or [])[:6]
    ) or "Introduction, Main Services, Why Choose Us, Local Expertise, FAQ, Conclusion"
    questions = "; ".join(cb.get("questions_to_answer") or [])[:400] or "What, Why, How, When, Cost questions"
    featured_snippet = cb.get("featured_snippet_opportunity") or f"What is the best {request.keyword}?"
    # Aim for 0.5-1.5% density: ~10 uses per 1500 words
    kw_target = max(8, round(target_wc * 0.008))

    # Call 2: full blog post as plain text
    write_prompt = BLOG_WRITE_PROMPT.format(
        business_name=request.business_name or "this business",
        business_type=request.business_type or "local business",
        location=request.location,
        keyword=request.keyword,
        h1=h1,
        keyword_density_target=kw_target,
        target_word_count=target_wc,
        sections=sections,
        questions=questions,
        featured_snippet=featured_snippet,
    )
    blog_content = await call_claude(
        BLOG_SYSTEM, write_prompt, max_tokens=4000, return_raw=True
    )

    word_count = len(blog_content.split()) if blog_content else 0
    logger.info(f"[{audit_id}] Blog Writer done — {word_count} words written")

    return {
        "agent": "blog_writer",
        "audit_id": audit_id,
        "status": "completed",
        "keyword": request.keyword,
        "competitors_analyzed": len(comp_summaries),
        "content_brief": cb,
        "blog_post": {
            "title": cb.get("seo_title", h1),
            "meta_description": cb.get("meta_description", ""),
            "h1": h1,
            "word_count": word_count,
            "content": blog_content,
        },
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# ORCHESTRATOR — runs all 4 agents, builds combined report
# =============================================================================

def calculate_ai_seo_score_from_signals(signals: dict) -> int:
    """Estimate AI SEO score from collected page signals when agent score is missing."""
    score = 0
    if signals.get("has_faq_schema"):
        score += 20
    if signals.get("has_author_bio"):
        score += 15
    if signals.get("has_credentials"):
        score += 10
    schema_found = signals.get("schema_types_found") or []
    score += min(15, len(schema_found) * 5)
    if signals.get("reviews_on_page"):
        score += 10
    if signals.get("has_about_page"):
        score += 10
    paa = signals.get("paa_questions_found", 0)
    score += min(15, int(paa) * 3)
    return min(100, score)


def calculate_pillar_scores(agents: dict) -> dict:
    """Calculate 0-100 scores for each pillar plus overall weighted average."""
    op = agents.get("on_page_seo", {})
    tech = agents.get("technical_seo", {})
    local = agents.get("local_seo", {})
    backlink = agents.get("backlink_analysis", {})
    ai_seo = agents.get("ai_seo", {})
    gbp = agents.get("gbp_audit", {})
    citation = agents.get("citation_builder", {})

    # Website SEO: on-page 60% + technical 40%
    on_page_score = op.get("recommendations", {}).get("current_analysis", {}).get("seo_score", 50)
    on_page_score = max(0, min(100, int(on_page_score)))
    tech_raw = tech.get("recommendations", {}).get("technical_score", 5)
    tech_score = max(0, min(10, int(tech_raw))) * 10
    website_seo = max(0, min(100, int(on_page_score * 0.60 + tech_score * 0.40)))

    # Backlinks: map DA range to a 0-100 pillar score
    da_raw = backlink.get("recommendations", {}).get("domain_authority", {}).get("score", 0)
    da_val = max(0, min(100, int(da_raw)))
    if da_val < 20:
        backlinks_score = 10 + int((da_val / 20) * 15)        # 0-20 DA → 10-25
    elif da_val < 40:
        backlinks_score = 25 + int(((da_val - 20) / 20) * 25)  # 20-40 DA → 25-50
    elif da_val < 60:
        backlinks_score = 50 + int(((da_val - 40) / 20) * 25)  # 40-60 DA → 50-75
    else:
        backlinks_score = 75 + int(((da_val - 60) / 40) * 25)  # 60-100 DA → 75-100
    backlinks_score = max(0, min(100, backlinks_score))

    # Local SEO: local_seo 40% + gbp 35% + citation 25%
    local_score = local.get("recommendations", {}).get("local_seo_score", 35)
    local_score = max(0, min(100, int(local_score)))
    gbp_score = gbp.get("analysis", {}).get("gbp_score", 30)
    gbp_score = max(0, min(100, int(gbp_score)))
    citation_score = citation.get("plan", {}).get("citation_score", 20)
    citation_score = max(0, min(100, int(citation_score)))
    local_seo = max(0, min(100, int(local_score * 0.40 + gbp_score * 0.35 + citation_score * 0.25)))

    # AI SEO: use agent score if present, else calculate from signals
    ai_visibility = ai_seo.get("analysis", {}).get("ai_visibility_score", 0)
    if ai_visibility:
        ai_seo_score = max(0, min(100, int(ai_visibility)))
    else:
        signals = ai_seo.get("signals_collected", {})
        ai_seo_score = calculate_ai_seo_score_from_signals(signals)

    # Overall: website 30% + local 30% + backlinks 20% + ai 20%
    overall = max(0, min(100, int(
        website_seo * 0.30 + local_seo * 0.30 + backlinks_score * 0.20 + ai_seo_score * 0.20
    )))

    return {
        "overall": overall,
        "website_seo": website_seo,
        "backlinks": backlinks_score,
        "local_seo": local_seo,
        "ai_seo": ai_seo_score,
    }


def build_score_details(agents: dict) -> dict:
    """Build sub-score details for each pillar, extracted from raw agent data."""
    op = agents.get("on_page_seo", {})
    tech = agents.get("technical_seo", {})
    backlink = agents.get("backlink_analysis", {})
    local = agents.get("local_seo", {})
    gbp = agents.get("gbp_audit", {})
    citation = agents.get("citation_builder", {})
    ai = agents.get("ai_seo", {})

    # Website SEO sub-scores
    op_rec = op.get("recommendations", {})
    tech_rec = tech.get("recommendations", {})
    current = op_rec.get("current_analysis", {})
    on_page_raw = current.get("seo_score", 0)
    on_page_val = max(0, min(100, int(on_page_raw * 10) if on_page_raw <= 10 else int(on_page_raw)))
    tech_raw = tech_rec.get("technical_score", 5)
    tech_val = max(0, min(100, int(tech_raw) * 10 if int(tech_raw) <= 10 else int(tech_raw)))
    page_speed = tech_rec.get("page_speed_score", tech_val)
    page_speed = max(0, min(100, int(page_speed)))
    issues = current.get("issues_found", [])
    issues_count = len(issues) if isinstance(issues, list) else 0

    # Backlink sub-scores
    bl_rec = backlink.get("recommendations", {})
    da_obj = bl_rec.get("domain_authority", {})
    estimated_da = int(da_obj.get("score", 0))
    linking_domains = da_obj.get("linking_domains", 0) or bl_rec.get("total_backlinks", 0)
    estimated_backlinks = int(linking_domains) if linking_domains else 0
    competitors = bl_rec.get("competitors", []) or backlink.get("analysis", {}).get("competitors", [])
    if competitors and isinstance(competitors, list):
        comp_das = [int(c.get("domain_authority", 0)) for c in competitors if isinstance(c, dict)]
        competitors_avg_da = int(sum(comp_das) / len(comp_das)) if comp_das else 0
    else:
        competitors_avg_da = 0

    # Local SEO sub-scores
    gbp_analysis = gbp.get("analysis", {})
    gbp_score = gbp_analysis.get("gbp_score", 0)
    if gbp_score and int(gbp_score) >= 70:
        gbp_status = "optimized"
    elif gbp_score and int(gbp_score) >= 40:
        gbp_status = "needs optimization"
    else:
        gbp_status = "not optimized"
    cit_plan = citation.get("plan", {})
    cit_recs = cit_plan.get("recommendations", {})
    citations_found = cit_plan.get("existing_citations", 0)
    if not citations_found:
        citations_found = cit_plan.get("summary", {}).get("existing_count", 0)
    t1 = len(cit_recs.get("tier_1_critical", []))
    t2 = len(cit_recs.get("tier_2_important", []))
    t3 = len(cit_recs.get("tier_3_supplemental", []))
    citations_needed = t1 + t2 + t3
    review_count = gbp_analysis.get("review_count", 0)
    if not review_count:
        review_count = gbp_analysis.get("map_pack_status", {}).get("reviews", 0)

    # AI SEO sub-scores
    signals = ai.get("signals_collected", {})
    analysis = ai.get("analysis", {})
    has_faq = bool(signals.get("has_faq_schema") or analysis.get("score_breakdown", {}).get("faq_content", 0) > 5)
    has_eeat = bool(signals.get("has_author_bio") or analysis.get("score_breakdown", {}).get("eeat_signals", 0) > 5)
    word_count = op_rec.get("current_analysis", {}).get("word_count", 0)
    if word_count >= 1500:
        content_depth = "comprehensive"
    elif word_count >= 800:
        content_depth = "adequate"
    elif word_count >= 300:
        content_depth = "thin"
    else:
        content_depth = "very thin"

    return {
        "website_seo": {
            "page_speed": page_speed,
            "on_page": on_page_val,
            "technical": tech_val,
            "issues_count": issues_count,
        },
        "backlinks": {
            "estimated_da": estimated_da,
            "estimated_backlinks": estimated_backlinks,
            "competitors_avg_da": competitors_avg_da,
        },
        "local_seo": {
            "gbp_status": gbp_status,
            "citations_found": int(citations_found),
            "citations_needed": citations_needed,
            "review_count": int(review_count),
        },
        "ai_seo": {
            "faq_schema": has_faq,
            "eeat_signals": has_eeat,
            "content_depth": content_depth,
        },
    }


def _ensure_list(val) -> list:
    """Normalise an agent output value to a list.

    Agents sometimes return strings instead of lists (e.g. a paragraph
    describing priority actions). This splits strings into items by
    newlines or numbered bullets so downstream code can iterate safely.
    """
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, dict):
        return [val]
    if isinstance(val, str):
        # Split by newlines, numbered bullets (1. 2.), or dashes/bullets
        items = []
        for line in re.split(r'\n|(?<=\. )(?=\d+\.)', val):
            line = line.strip().lstrip("0123456789.-•) ").strip()
            if line and len(line) > 15:  # skip trivially short fragments
                items.append(line)
        return items if items else ([val.strip()] if val.strip() and len(val.strip()) > 15 else [])
    return []


def build_structured_quick_wins(agents: dict) -> list[dict]:
    """Build top 10 structured quick-win objects from all agent outputs."""
    wins: list[dict] = []

    def _add(title: str, description: str, pillar: str, priority: str = "high",
             impact: str = "", time_estimate: str = "") -> None:
        if not title or len(wins) >= 10:
            return
        # Deduplicate by title (case-insensitive first 60 chars)
        norm = title[:60].lower()
        if any(w["title"][:60].lower() == norm for w in wins):
            return
        wins.append({
            "rank": len(wins) + 1,
            "title": title[:120],
            "description": description or title,
            "pillar": pillar,
            "priority": priority,
            "impact": impact,
            "time_estimate": time_estimate,
        })

    # On-Page priority actions — check top-level and nested path
    op_full = agents.get("on_page_seo", {}).get("recommendations", {})
    op_actions = _ensure_list(
        op_full.get("priority_actions")
        or op_full.get("recommendations", {}).get("priority_actions") if isinstance(op_full.get("recommendations"), dict) else None
    )
    # Fallback: issues_found from current_analysis
    if not op_actions:
        op_actions = _ensure_list(op_full.get("current_analysis", {}).get("issues_found"))
    for action in op_actions[:3]:
        if isinstance(action, str) and action:
            _add(action, action, "website_seo", "high", "Improve search rankings", "1-2 hours")

    # Technical priority fixes — check top-level and nested
    tech_full = agents.get("technical_seo", {}).get("recommendations", {})
    tech_fixes = _ensure_list(
        tech_full.get("priority_fixes")
        or tech_full.get("recommendations", {}).get("priority_fixes") if isinstance(tech_full.get("recommendations"), dict) else None
    )
    if not tech_fixes:
        tech_fixes = _ensure_list(tech_full.get("quick_wins"))
    for fix in tech_fixes[:2]:
        if isinstance(fix, str) and fix:
            _add(fix, fix, "website_seo", "high", "Fix critical technical issues", "30-60 min")

    # Local SEO quick wins — increased limit
    local_rec = agents.get("local_seo", {}).get("recommendations", {})
    for win in _ensure_list(local_rec.get("quick_wins"))[:3]:
        if isinstance(win, str) and win:
            _add(win, win, "local_seo", "high", "Improve local rankings", "1-2 hours")

    # GBP priority actions — handle string values
    gbp_analysis = agents.get("gbp_audit", {}).get("analysis", {})
    for action in _ensure_list(gbp_analysis.get("priority_actions"))[:2]:
        if isinstance(action, dict):
            _add(action.get("action", ""), action.get("reason", "") or action.get("how_to", ""),
                 "local_seo", action.get("impact", "medium"), action.get("how_to", ""), "30-60 min")
        elif isinstance(action, str) and action:
            _add(action, action, "local_seo", "medium", "Improve GBP visibility", "30 min")

    # AI SEO priority actions — handle string values
    ai_analysis = agents.get("ai_seo", {}).get("analysis", {})
    for action in _ensure_list(ai_analysis.get("priority_actions"))[:2]:
        if isinstance(action, dict):
            _add(action.get("action", ""), action.get("why", "") or action.get("how", ""),
                 "ai_seo", action.get("impact", "medium"), action.get("how", ""),
                 action.get("effort", "1-2 hours"))
        elif isinstance(action, str) and action:
            _add(action, action, "ai_seo", "medium", "Improve AI visibility", "1-2 hours")

    # Citation tier 1 — handle string values
    citation_plan = agents.get("citation_builder", {}).get("plan", {})
    tier1 = _ensure_list(citation_plan.get("recommendations", {}).get("tier_1_critical") if isinstance(citation_plan.get("recommendations"), dict) else None)
    for cite in tier1[:1]:
        if isinstance(cite, dict) and cite.get("name"):
            _add(f"Build citation on {cite['name']}", cite.get("reason", "Build local authority"),
                 "local_seo", "medium", f"DA: {cite.get('da', '?')}", cite.get("time_to_list", "1 hour"))
        elif isinstance(cite, str) and cite:
            _add(cite, cite, "local_seo", "medium", "Build local authority", "1 hour")

    # Backlink quick wins
    backlink_rec = agents.get("backlink_analysis", {}).get("recommendations", {})
    for win in _ensure_list(backlink_rec.get("quick_wins"))[:2]:
        if isinstance(win, str) and win:
            _add(win, win, "backlinks", "low", "Improve domain authority", "1-2 weeks")

    # Fallback — provide meaningful default quick wins when agents return errors
    if len(wins) < 5:
        defaults = [
            {"title": "Complete Google Business Profile optimisation", "description": "Fill all GBP fields, add 40+ photos, select the right primary category.", "pillar": "local_seo", "priority": "high", "impact": "Map Pack visibility", "time_estimate": "2 hours"},
            {"title": "Update meta title and description with target keyword", "description": "Write a compelling 60-char title and 155-char description with your primary keyword.", "pillar": "website_seo", "priority": "high", "impact": "+20% CTR from search", "time_estimate": "10 min"},
            {"title": "Build citations on top local directories", "description": "Submit your business to Google, Yelp, BBB, and industry-specific directories.", "pillar": "local_seo", "priority": "high", "impact": "+10-15 local authority", "time_estimate": "1 hour each"},
            {"title": "Add internal links between service pages", "description": "Link related service pages together to distribute link equity.", "pillar": "website_seo", "priority": "medium", "impact": "Better site structure", "time_estimate": "30 min"},
            {"title": "Add FAQ section with structured data markup", "description": "Create a FAQ page with FAQPage schema to appear in AI answers and People Also Ask.", "pillar": "ai_seo", "priority": "high", "impact": "AI overview visibility", "time_estimate": "1-2 hours"},
            {"title": "Get 5 new Google reviews this month", "description": "Send review request emails to recent customers using a simple template.", "pillar": "local_seo", "priority": "high", "impact": "+15% local trust signals", "time_estimate": "30 min setup"},
            {"title": "Fix missing alt text on images", "description": "Add descriptive alt attributes to all images for accessibility and SEO.", "pillar": "website_seo", "priority": "medium", "impact": "Image search + accessibility", "time_estimate": "30-60 min"},
            {"title": "Build 3 guest post backlinks", "description": "Write guest articles for local blogs or industry publications to earn quality backlinks.", "pillar": "backlinks", "priority": "medium", "impact": "+5-10 DA over 3 months", "time_estimate": "2-3 hours each"},
            {"title": "Create service area pages for each location", "description": "Build dedicated pages for each service area to rank for local + service keywords.", "pillar": "local_seo", "priority": "medium", "impact": "Rank in nearby cities", "time_estimate": "1-2 hours each"},
            {"title": "Add LocalBusiness schema markup", "description": "Implement JSON-LD structured data for your business name, address, phone, hours, and services.", "pillar": "ai_seo", "priority": "medium", "impact": "Rich snippets + AI visibility", "time_estimate": "30 min"},
        ]
        existing_titles = {w["title"][:60].lower() for w in wins}
        for d in defaults:
            if len(wins) >= 10:
                break
            if d["title"][:60].lower() not in existing_titles:
                d["rank"] = len(wins) + 1
                wins.append(d)
                existing_titles.add(d["title"][:60].lower())

    for i, win in enumerate(wins[:10]):
        win["rank"] = i + 1
    return wins[:10]


def _make_step(rank: int, title: str, description: str, category: str, priority: str,
               impact: str = "", time_estimate: str = "") -> dict:
    return {
        "rank": rank,
        "title": title[:120] if title else "",
        "description": description or title,
        "category": category,
        "priority": priority,
        "impact": impact,
        "time_estimate": time_estimate,
    }


def build_pillar_steps(agents: dict, scores: dict) -> dict:
    """Build 5 improvement steps per pillar from agent outputs."""

    # === Website SEO ===
    website_steps: list[dict] = []
    op_full = agents.get("on_page_seo", {}).get("recommendations", {})
    tech_full = agents.get("technical_seo", {}).get("recommendations", {})
    # On-page: check top-level and nested, fallback to issues_found
    op_actions = _ensure_list(
        op_full.get("priority_actions")
        or (op_full.get("recommendations", {}).get("priority_actions") if isinstance(op_full.get("recommendations"), dict) else None)
    )
    if not op_actions:
        op_actions = _ensure_list(op_full.get("current_analysis", {}).get("issues_found"))
    for action in op_actions[:3]:
        if isinstance(action, str) and action:
            website_steps.append(_make_step(len(website_steps) + 1, action, action, "On-Page", "high", "", "1-2 hours"))
    # Technical: check top-level and nested, fallback to quick_wins
    tech_fixes = _ensure_list(
        tech_full.get("priority_fixes")
        or (tech_full.get("recommendations", {}).get("priority_fixes") if isinstance(tech_full.get("recommendations"), dict) else None)
    )
    for fix in tech_fixes[:3]:
        if isinstance(fix, str) and fix and len(website_steps) < 5:
            website_steps.append(_make_step(len(website_steps) + 1, fix, fix, "Technical", "medium", "", "30-60 min"))
    for win in _ensure_list(tech_full.get("quick_wins"))[:2]:
        if isinstance(win, str) and win and len(website_steps) < 5:
            website_steps.append(_make_step(len(website_steps) + 1, win, win, "Technical", "low", "", "15 min"))
    if not website_steps:
        website_steps = [
            _make_step(1, "Update meta title and description", "Write a compelling title (60 chars) and description (155 chars) with your primary keyword.", "On-Page", "high", "+20% CTR from search", "10 min"),
            _make_step(2, "Improve content depth to 1,500+ words", "Add service descriptions, FAQs, and testimonials to match top-ranking competitors.", "Content", "high", "+20-30 organic positions", "3 hours"),
            _make_step(3, "Optimize page speed — images, lazy loading, defer scripts", "Convert images to WebP, add loading='lazy', defer non-critical JS. Target PageSpeed 70+ on mobile.", "Technical", "medium", "Better Core Web Vitals", "2 hours"),
            _make_step(4, "Add alt text to all images with local keywords", "Use descriptive, keyword-rich alt text on every image.", "Technical", "medium", "Image search visibility", "30 min"),
            _make_step(5, "Add canonical tags + submit XML sitemap", "Prevents duplicate content and speeds up indexing.", "Technical", "low", "Cleaner crawl budget", "15 min"),
        ]
    for i, s in enumerate(website_steps[:5]):
        s["rank"] = i + 1

    # === Backlinks ===
    backlink_steps: list[dict] = []
    backlink_rec = agents.get("backlink_analysis", {}).get("recommendations", {})
    lb_rec = agents.get("link_building", {}).get("recommendations", {})
    for issue in (backlink_rec.get("top_issues") or [])[:2]:
        if isinstance(issue, str) and issue:
            backlink_steps.append(_make_step(len(backlink_steps) + 1, issue, issue, "Profile", "high", "", "1-2 weeks"))
    for opp in (lb_rec.get("quick_wins") or [])[:3]:
        if isinstance(opp, dict) and len(backlink_steps) < 5:
            name = opp.get("name", "")
            reason = opp.get("reason", "")
            backlink_steps.append(_make_step(
                len(backlink_steps) + 1,
                f"Submit to {name}" if name else reason,
                reason or f"Get a backlink from {name}",
                (opp.get("link_type") or "Directory").capitalize(),
                opp.get("difficulty", "medium"),
                f"Est. DA: {opp.get('expected_da', '?')}",
                "1-2 hours",
            ))
    for opp in (lb_rec.get("local_opportunities") or [])[:2]:
        if isinstance(opp, dict) and len(backlink_steps) < 5:
            name = opp.get("name", "")
            reason = opp.get("reason", "")
            backlink_steps.append(_make_step(
                len(backlink_steps) + 1,
                f"Partner with {name}" if name else reason,
                reason or f"Local link opportunity: {name}",
                "Local",
                opp.get("difficulty", "medium"),
                f"Est. DA: {opp.get('expected_da', '?')}",
                "1-2 weeks",
            ))
    if not backlink_steps:
        backlink_steps = [
            _make_step(1, "Submit to industry directories", "Find and submit to the top 5 industry-specific directories in your niche.", "Directory", "high", "DA increase", "2 hours"),
            _make_step(2, "Guest post on local blogs and news sites", "Pitch relevant articles to local publications. DA 50+ links have the highest impact.", "Guest Post", "high", "High-DA backlink", "2 weeks"),
            _make_step(3, "Create a linkable content asset", "Build a resource (guide, calculator, data report) that naturally attracts links over time.", "Content", "medium", "Passive link acquisition", "1 week"),
            _make_step(4, "Sponsor a local event or association", "Get citations from event websites, local press, and partner sites.", "Sponsorship", "medium", "Local authority signal", "2 weeks"),
            _make_step(5, "Analyse competitor backlinks for gaps", "Find sites linking to competitors but not you — and pitch them your content.", "Strategy", "low", "Link gap closure", "2 hours"),
        ]
    for i, s in enumerate(backlink_steps[:5]):
        s["rank"] = i + 1

    # === Local SEO ===
    local_steps: list[dict] = []
    local_rec = agents.get("local_seo", {}).get("recommendations", {})
    gbp_analysis = agents.get("gbp_audit", {}).get("analysis", {})
    citation_plan = agents.get("citation_builder", {}).get("plan", {})
    for action in _ensure_list(gbp_analysis.get("priority_actions"))[:2]:
        if isinstance(action, dict) and action.get("action"):
            local_steps.append(_make_step(
                len(local_steps) + 1,
                action.get("action", ""),
                action.get("reason", "") or action.get("how_to", ""),
                "GBP",
                action.get("impact", "high"),
                action.get("how_to", ""),
                "30-60 min",
            ))
        elif isinstance(action, str) and action:
            local_steps.append(_make_step(len(local_steps) + 1, action, action, "GBP", "high", "", "30-60 min"))
    for win in _ensure_list(local_rec.get("quick_wins"))[:3]:
        if isinstance(win, str) and win and len(local_steps) < 5:
            local_steps.append(_make_step(len(local_steps) + 1, win, win, "Local SEO", "high", "", "1-2 hours"))
    tier1_local = _ensure_list(citation_plan.get("recommendations", {}).get("tier_1_critical") if isinstance(citation_plan.get("recommendations"), dict) else None)
    for cite in tier1_local[:2]:
        if isinstance(cite, dict) and cite.get("name") and len(local_steps) < 5:
            local_steps.append(_make_step(
                len(local_steps) + 1,
                f"Build citation on {cite['name']}",
                cite.get("reason", ""),
                "Citations",
                "medium",
                f"DA: {cite.get('da', '?')}",
                cite.get("time_to_list", "1 hour"),
            ))
        elif isinstance(cite, str) and cite and len(local_steps) < 5:
            local_steps.append(_make_step(len(local_steps) + 1, cite, cite, "Citations", "medium", "", "1 hour"))
    if not local_steps:
        local_steps = [
            _make_step(1, "Complete Google Business Profile", "Fill all fields, upload 40+ photos, add services with descriptions, seed 10 Q&As.", "GBP", "high", "Map Pack visibility", "2 hours"),
            _make_step(2, "Build top directory citations", "Submit to Google, Yelp, BBB, and industry-specific directories with consistent NAP.", "Citations", "high", "+10-15 local authority", "1 hour each"),
            _make_step(3, "Start review acquisition campaign", "Send review requests after every job. Target 4+ reviews per month.", "Reviews", "medium", "Improved local rankings", "Ongoing"),
            _make_step(4, "Create service area pages", "Build individual pages for each city/neighbourhood you serve with unique content.", "Content", "medium", "Rank for local terms", "2 hours each"),
            _make_step(5, "Post weekly on GBP", "Share project photos, tips, and seasonal offers. Consistency signals active business.", "GBP Posts", "low", "Engagement signal", "Ongoing"),
        ]
    for i, s in enumerate(local_steps[:5]):
        s["rank"] = i + 1

    # === AI SEO ===
    ai_steps: list[dict] = []
    ai_analysis = agents.get("ai_seo", {}).get("analysis", {})
    for action in _ensure_list(ai_analysis.get("priority_actions"))[:5]:
        if isinstance(action, dict) and action.get("action"):
            ai_steps.append(_make_step(
                len(ai_steps) + 1,
                action.get("action", ""),
                action.get("why", "") or action.get("how", ""),
                "Content Structure",
                action.get("impact", "medium"),
                action.get("how", ""),
                "1-3 hours",
            ))
        elif isinstance(action, str) and action:
            ai_steps.append(_make_step(len(ai_steps) + 1, action, action, "Content Structure", "medium", "", "1-3 hours"))
    if not ai_steps:
        ai_steps = [
            _make_step(1, "Add FAQ section with FAQPage schema", "Create 8-10 Q&A pairs answering common customer questions. Add JSON-LD FAQPage schema markup.", "Content Structure", "high", "Featured snippets + AI citations", "1 hour"),
            _make_step(2, "Build E-E-A-T signals (author bios + About page)", "Create team page with photos, experience, licenses, certifications. Link from every page.", "E-E-A-T", "high", "Trusted source for AI engines", "1-2 hours"),
            _make_step(3, "Write a definitive industry guide (3,000+ words)", "Create the most comprehensive resource on your topic — the #1 content AI engines cite.", "Topical Authority", "medium", "AI citation likelihood", "1 day"),
            _make_step(4, "Add original data and local statistics", "Include pricing tables, project data, and local market stats unique to your business.", "Citation Potential", "medium", "Citable source", "2-3 hours"),
            _make_step(5, "Rewrite key content in direct-answer format", "Lead each section with a clear answer, then supporting detail. Avoid burying key info.", "Content Format", "low", "AI extraction friendly", "Ongoing"),
        ]
    for i, s in enumerate(ai_steps[:5]):
        s["rank"] = i + 1

    return {
        "website_seo": {
            "score": scores.get("website_seo", 0),
            "title": "Website SEO",
            "subtitle": "Page speed, on-page optimization, technical health, keyword positioning",
            "color": "blue",
            "steps": website_steps[:5],
        },
        "backlinks": {
            "score": scores.get("backlinks", 0),
            "title": "Backlink & Link Building",
            "subtitle": "Domain authority, referring domains, link acquisition strategy",
            "color": "rose",
            "steps": backlink_steps[:5],
        },
        "local_seo": {
            "score": scores.get("local_seo", 0),
            "title": "Local SEO",
            "subtitle": "Google Business Profile, citations, NAP consistency, local rankings",
            "color": "amber",
            "steps": local_steps[:5],
        },
        "ai_seo": {
            "score": scores.get("ai_seo", 0),
            "title": "AI SEO Visibility",
            "subtitle": "Optimization for ChatGPT, Perplexity, Google AI Overviews, Gemini",
            "color": "violet",
            "steps": ai_steps[:5],
        },
    }


def build_seo_tasks(quick_wins: list[dict], pillars: dict) -> list[dict]:
    """Build a flat task list from quick wins + pillar steps for the SEO Tasks view."""
    seen_titles: set[str] = set()
    tasks: list[dict] = []
    task_counter = 0

    def _add_task(title: str, pillar: str, priority: str, time_est: str = "",
                  impact: str = "") -> None:
        nonlocal task_counter
        norm = title.strip().lower()
        if norm in seen_titles or not title:
            return
        seen_titles.add(norm)
        task_counter += 1
        tasks.append({
            "id": f"task_{task_counter}",
            "title": title[:150],
            "pillar": pillar,
            "priority": priority,
            "time_estimate": time_est,
            "impact": impact,
            "status": "pending",
        })

    # First: add all quick wins (these are the highest priority)
    for win in quick_wins:
        _add_task(
            win.get("title", ""),
            win.get("pillar", ""),
            win.get("priority", "medium"),
            win.get("time_estimate", ""),
            win.get("impact", ""),
        )

    # Then: add pillar steps that weren't already covered by quick wins
    for pillar_key, pillar_data in pillars.items():
        if not isinstance(pillar_data, dict):
            continue
        for step in pillar_data.get("steps", []):
            _add_task(
                step.get("title", ""),
                pillar_key,
                step.get("priority", "medium"),
                step.get("time_estimate", ""),
                step.get("impact", ""),
            )

    return tasks


# =============================================================================
# TOOL-PAGE BUILDERS — flatten agent outputs into the frontend-ready format
# =============================================================================


def _resolve_agent_recs(agent_data: dict, key: str = "recommendations") -> dict:
    """Get an agent's recommendations, repairing raw_response fallback if needed."""
    recs = agent_data.get(key, {})
    if not isinstance(recs, dict):
        return {}
    if "raw_response" in recs and len(recs) == 1:
        raw = recs["raw_response"]
        if isinstance(raw, str):
            repaired = _repair_truncated_json(raw)
            if isinstance(repaired, dict):
                return repaired
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                pass
    return recs


def build_gmb_data(agents: dict) -> dict:
    """Build gmb_data section from gbp_audit + citation_builder + local_seo agents."""
    gbp = agents.get("gbp_audit", {})
    cit = agents.get("citation_builder", {})
    local = agents.get("local_seo", {})
    gbp_a = _resolve_agent_recs(gbp, "analysis")
    cit_plan = _resolve_agent_recs(cit, "plan")
    local_rec = _resolve_agent_recs(local)

    # --- GBP Score ---
    gbp_score = int(gbp_a.get("gbp_score", 0))

    # --- Reviews from map pack or GBP analysis ---
    pack = gbp_a.get("map_pack_status", {})
    review_count = 0
    avg_rating = 0.0
    # Try to get from local_pack data on the gbp agent result
    local_pack = gbp.get("local_pack", [])
    # Find the client's own entry in the pack
    for entry in local_pack:
        if isinstance(entry, dict):
            reviews = entry.get("reviews")
            rating = entry.get("rating")
            if reviews is not None and rating is not None:
                # Use first entry as proxy if client is not in pack
                if review_count == 0:
                    review_count = int(reviews) if reviews else 0
                    avg_rating = float(rating) if rating else 0.0
    # Override with review_strategy if available
    rs = gbp_a.get("review_strategy", {})
    vis = rs.get("current_visibility", "")
    if "review" in vis.lower():
        # Try to extract a number from the visibility text
        import re as _re
        nums = _re.findall(r'(\d+)', vis)
        if nums:
            review_count = int(nums[0])

    # --- Response rate (estimated) ---
    review_responses = gbp_a.get("completeness_audit", {}).get("review_responses", {})
    rr_status = review_responses.get("status", "unknown") if isinstance(review_responses, dict) else "unknown"
    response_rate = "100%" if rr_status == "pass" else "50%" if rr_status == "warn" else "0%"

    # --- Photos ---
    photos_cover = gbp_a.get("completeness_audit", {}).get("photos_cover", {})
    photos_interior = gbp_a.get("completeness_audit", {}).get("photos_interior", {})
    photos_count = 0
    for pf in [photos_cover, photos_interior]:
        if isinstance(pf, dict) and pf.get("status") == "pass":
            photos_count += 5  # estimate
    photos_needed = max(0, 40 - photos_count)

    # --- NAP ---
    nap_raw = gbp_a.get("nap_consistency", {})
    gbp_opt = local_rec.get("gbp_optimization", {})
    categories = gbp_opt.get("categories", [])
    primary_cat = categories[0] if categories else "Local Business"
    completeness = gbp_a.get("completeness_audit", {})
    cat_status = completeness.get("primary_category", {})
    cat_optimal = (cat_status.get("status") == "pass") if isinstance(cat_status, dict) else False

    nap = {
        "name": nap_raw.get("name_on_website", ""),
        "address": nap_raw.get("address_on_website", "not detected"),
        "phone": nap_raw.get("phone_on_website", "not detected"),
        "website": gbp.get("target_url", ""),
        "category": primary_cat,
        "category_optimal": cat_optimal,
    }

    # --- Checklist from completeness_audit ---
    checklist = []
    field_map = {
        "business_name_consistent": ("Business name claimed", "high"),
        "phone_number": ("Phone number added", "high"),
        "website_linked": ("Website linked", "high"),
        "business_hours": ("Business hours set", "high"),
        "primary_category": ("Primary category set", "medium"),
        "description": ("Business description", "medium"),
        "photos_cover": ("40+ photos uploaded", "high"),
        "products_services": ("Services with descriptions", "high"),
        "qa_section": ("10 seeded Q&As", "medium"),
        "attributes": ("Attributes set", "low"),
        "posts_active": ("GBP posts active", "medium"),
        "review_responses": ("Review responses", "medium"),
    }
    for field_key, (label, priority) in field_map.items():
        item = completeness.get(field_key, {})
        if isinstance(item, dict):
            status = item.get("status", "unknown")
            done = status == "pass"
            # "warn" means partially done — mark as done for checklist but note the issue
            if status == "warn":
                done = True
        else:
            done = False
        checklist.append({"item": label, "done": done, "priority": priority})

    # --- Citations from citation_builder ---
    citations = []
    cit_recs = cit_plan.get("recommendations", {})
    if isinstance(cit_recs, dict):
        for tier_key in ("tier_1_critical", "tier_2_important", "tier_3_supplemental"):
            for c in (cit_recs.get(tier_key) or []):
                if isinstance(c, dict):
                    citations.append({
                        "directory": c.get("name", "Unknown"),
                        "da": int(c.get("da", 0)),
                        "found": c.get("status", "likely_missing") != "likely_missing",
                        "nap_match": "exact" if c.get("status") != "likely_missing" else None,
                    })

    return {
        "gbp_score": gbp_score,
        "review_count": review_count,
        "avg_rating": avg_rating,
        "response_rate": response_rate,
        "photos_count": photos_count,
        "photos_needed": photos_needed,
        "nap": nap,
        "checklist": checklist,
        "citations": citations,
    }


def _difficulty_to_int(diff_str) -> int:
    """Convert difficulty string (low/medium/high) to a numeric value."""
    if isinstance(diff_str, (int, float)):
        return int(diff_str)
    d = str(diff_str).lower()
    if "low" in d or "easy" in d:
        return 22
    if "med" in d:
        return 45
    if "high" in d or "hard" in d:
        return 68
    return 40


def _volume_to_int(vol) -> int:
    """Convert volume label or int to a numeric estimate."""
    if isinstance(vol, (int, float)):
        return int(vol)
    v = str(vol).lower()
    if "high" in v:
        return 1200
    if "med" in v:
        return 480
    if "low" in v:
        return 140
    # Try parsing directly
    try:
        return int(vol)
    except (ValueError, TypeError):
        return 200


def build_keyword_data(agents: dict, auto_detected: dict | None, profile_services: list[str] | None = None) -> dict:
    """Build keyword_data section from keyword_research agent."""
    kw = agents.get("keyword_research", {})
    recs = _resolve_agent_recs(kw)

    primary = recs.get("primary_keyword", kw.get("keyword", ""))

    # --- Keywords from high_intent_keywords ---
    keywords = []
    for i, hik in enumerate(recs.get("high_intent_keywords", [])):
        if not isinstance(hik, dict):
            continue
        kw_text = hik.get("keyword", "")
        vol = hik.get("estimated_monthly_searches", 0)
        diff = hik.get("difficulty", "medium")
        intent = hik.get("intent", "commercial")
        is_primary = (kw_text.lower() == primary.lower())
        # Map difficulty to action
        diff_val = _difficulty_to_int(diff)
        if diff_val < 30:
            action = "Easy win"
        elif is_primary:
            action = "Optimize"
        elif diff_val < 50:
            action = "Improve"
        else:
            action = "Create page"

        keywords.append({
            "keyword": kw_text,
            "volume": _volume_to_int(vol),
            "difficulty": diff_val,
            "intent": intent.capitalize() if isinstance(intent, str) else "Commercial",
            "position": None,
            "action": action,
            "is_primary": is_primary,
        })

    # Ensure primary keyword is first
    primary_items = [k for k in keywords if k.get("is_primary")]
    other_items = [k for k in keywords if not k.get("is_primary")]
    keywords = primary_items + other_items

    # --- Keyword gaps from keyword_gap OR competitor_keywords_we_miss ---
    keyword_gaps = []
    gap_list = recs.get("keyword_gap", [])
    if not gap_list:
        # Fallback: competitor_keywords_we_miss (may be list of strings or dicts)
        gap_list = recs.get("competitor_keywords_we_miss", [])
    for gap in gap_list:
        if isinstance(gap, str):
            keyword_gaps.append({
                "keyword": gap,
                "volume": _volume_to_int("medium"),
                "difficulty": _difficulty_to_int("medium"),
                "competitor": "",
                "competitor_position": None,
                "opportunity": "Create page",
            })
            continue
        if not isinstance(gap, dict):
            continue
        kw_text = gap.get("keyword", "")
        vol = gap.get("estimated_volume", gap.get("estimated_monthly_searches", "medium"))
        diff = gap.get("difficulty", "medium")
        action = gap.get("action", "Create page")
        keyword_gaps.append({
            "keyword": kw_text,
            "volume": _volume_to_int(vol),
            "difficulty": _difficulty_to_int(diff),
            "competitor": "",
            "competitor_position": None,
            "opportunity": action,
        })

    # --- Build keyword_groups by service ---
    keyword_groups = []
    if profile_services:
        city = ""
        # Try to extract city from primary keyword or auto_detected
        if auto_detected and isinstance(auto_detected, dict):
            city = auto_detected.get("city", "")
        assigned = set()
        for service in profile_services:
            service_lower = service.lower().strip()
            if not service_lower:
                continue
            group_primary = f"{service} {city}".strip() if city else service
            group_keywords = []
            for kw in keywords:
                kw_text = kw.get("keyword", "").lower()
                if service_lower in kw_text and kw_text not in assigned:
                    group_keywords.append(kw)
                    assigned.add(kw_text)
            keyword_groups.append({
                "service": service,
                "primary": group_primary,
                "keywords": group_keywords,
            })
        # Collect unmatched keywords into a "General" group
        general_kws = [kw for kw in keywords if kw.get("keyword", "").lower() not in assigned]
        if general_kws:
            keyword_groups.append({
                "service": "General",
                "primary": primary,
                "keywords": general_kws,
            })

    return {
        "primary_keyword": primary,
        "keywords": keywords[:20],
        "keyword_gaps": keyword_gaps[:15],
        "keyword_groups": keyword_groups,
    }


def build_backlink_data(agents: dict) -> dict:
    """Build backlink_data section from backlink_analysis + link_building agents."""
    bl = agents.get("backlink_analysis", {})
    lb = agents.get("link_building", {})
    bl_rec = _resolve_agent_recs(bl)
    lb_rec = _resolve_agent_recs(lb)

    da_obj = bl_rec.get("domain_authority", {})
    bp = bl_rec.get("backlink_profile", {})

    # Parse total backlinks (may be int or string like "40-150")
    total_raw = bp.get("total_backlinks", 0)
    if isinstance(total_raw, str):
        # Take the higher end of a range
        nums = [int(n) for n in re.findall(r'\d+', total_raw)]
        total_backlinks = nums[-1] if nums else 0
    else:
        total_backlinks = int(total_raw)

    referring_raw = bp.get("referring_domains", 0)
    if isinstance(referring_raw, str):
        nums = [int(n) for n in re.findall(r'\d+', referring_raw)]
        referring_domains = nums[-1] if nums else 0
    else:
        referring_domains = int(referring_raw)

    dofollow_pct = bp.get("dofollow_estimate_pct", 70)
    dofollow = int(total_backlinks * (int(dofollow_pct) / 100))
    nofollow = total_backlinks - dofollow

    # Competitor comparison
    comp_das = []
    comp_domains = []
    for comp in bl_rec.get("competitor_comparison", []):
        if isinstance(comp, dict):
            comp_das.append(int(comp.get("da", 0)))
    competitor_avg_da = int(sum(comp_das) / len(comp_das)) if comp_das else da_obj.get("vs_competitors_avg", 0) or 0

    # Current backlinks (limited data — build from quick_wins and competitor info)
    current_backlinks = []
    for win in bl_rec.get("quick_wins", [])[:5]:
        if isinstance(win, str) and win:
            current_backlinks.append({
                "source": win[:60],
                "da": 0,
                "type": "follow",
                "anchor": "",
            })

    # Opportunities from link_building agent
    opportunities = []
    for cat in ("quick_wins", "guest_posting", "resource_pages", "local_opportunities"):
        for opp in (lb_rec.get(cat) or []):
            if isinstance(opp, dict):
                opportunities.append({
                    "target": opp.get("name", ""),
                    "da": int(opp.get("expected_da", 0)),
                    "type": (opp.get("link_type") or cat.replace("_", " ")).capitalize(),
                    "effort": (opp.get("difficulty") or "Medium").capitalize(),
                    "url": opp.get("url", ""),
                })
    # Also add competitor gaps
    for opp in (lb_rec.get("competitor_gaps") or []):
        if isinstance(opp, dict):
            opportunities.append({
                "target": opp.get("name", ""),
                "da": int(opp.get("expected_da", 0)),
                "type": "Competitor Gap",
                "effort": "Medium",
                "url": opp.get("url", ""),
            })

    return {
        "domain_authority": int(da_obj.get("score", 0)),
        "total_backlinks": total_backlinks,
        "dofollow": dofollow,
        "nofollow": nofollow,
        "referring_domains": referring_domains,
        "competitor_avg_da": int(competitor_avg_da),
        "competitor_avg_domains": 0,
        "current_backlinks": current_backlinks,
        "opportunities": opportunities[:15],
    }


def build_content_data(agents: dict) -> dict:
    """Build content_data section from on_page_seo, keyword_research, local_seo, ai_seo agents."""
    op = agents.get("on_page_seo", {})
    kw = agents.get("keyword_research", {})
    local = agents.get("local_seo", {})
    ai = agents.get("ai_seo", {})
    rewriter = agents.get("content_rewriter", {})

    op_rec = _resolve_agent_recs(op)
    kw_rec = _resolve_agent_recs(kw)
    local_rec = _resolve_agent_recs(local)
    ai_analysis = _resolve_agent_recs(ai, "analysis")
    rewriter_rec = _resolve_agent_recs(rewriter)

    # --- Homepage words ---
    current = op_rec.get("current_analysis", {})
    homepage_words = int(current.get("word_count", 0))

    # Competitor avg from content rewriter benchmark
    bench = rewriter_rec.get("benchmark", rewriter.get("benchmark", {}))
    competitor_avg_words = int(bench.get("avg_competitor_word_count", 0)) or 1800

    # --- Pages to rewrite (from on_page issues) ---
    pages_to_rewrite = []
    issues = current.get("issues_found", [])
    if isinstance(issues, list):
        for issue_text in issues:
            if isinstance(issue_text, str):
                # Classify as thin content, weak targeting, etc.
                is_thin = any(w in issue_text.lower() for w in ["word count", "thin", "content depth", "short"])
                if is_thin or len(pages_to_rewrite) < 3:
                    pages_to_rewrite.append({
                        "page": "Homepage",
                        "url": "/",
                        "words": homepage_words,
                        "issue": issue_text[:120],
                        "priority": "Critical" if is_thin else "Important",
                    })

    # --- Service areas from local_seo ---
    local_content = local_rec.get("local_content_strategy", {})
    area_pages_raw = local_content.get("service_area_pages", [])
    service_areas = []
    primary_kw = kw_rec.get("primary_keyword", op.get("keyword", ""))
    # Build service prefix by removing location words from keyword
    # e.g. "kitchen cabinets toronto" → location words {"toronto"} → prefix "kitchen cabinets"
    location_str = local.get("location", "")
    loc_words = set(w.lower() for w in location_str.split(",")[0].split() if len(w) > 2)
    kw_words = primary_kw.lower().split()
    service_prefix = " ".join(w for w in kw_words if w not in loc_words)

    for area in area_pages_raw:
        if isinstance(area, str):
            # Parse "Kitchen Cabinets North York" → city="North York"
            area_lower = area.lower().strip()
            city = area
            if service_prefix and area_lower.startswith(service_prefix):
                city = area[len(service_prefix):].strip()
            elif service_prefix and area_lower.endswith(service_prefix):
                city = area[:len(area) - len(service_prefix)].strip()
            service_areas.append({
                "city": city if city else area,
                "keyword": area.lower(),
                "volume": 120,
                "difficulty": 18,
            })

    # --- FAQ suggestions from ai_seo (faq_content) + priority_actions questions ---
    faq_suggestions = []
    seen_qs = set()
    for faq in ai_analysis.get("faq_content", []):
        if isinstance(faq, dict):
            q = faq.get("question", "")
            if q and q.lower() not in seen_qs:
                seen_qs.add(q.lower())
                faq_suggestions.append({
                    "question": q,
                    "volume": 200,
                    "source": faq.get("ai_intent", "PAA"),
                })
    # Also pull questions from schema_templates FAQ answers
    for tmpl in ai_analysis.get("schema_templates", []):
        if isinstance(tmpl, dict) and tmpl.get("type") == "FAQPage":
            # The json_ld might have questions embedded
            pass
    # Pull from current_gaps as implicit questions
    for gap in ai_analysis.get("current_gaps", []):
        if isinstance(gap, str) and "?" in gap and gap.lower() not in seen_qs:
            seen_qs.add(gap.lower())
            faq_suggestions.append({"question": gap, "volume": 150, "source": "AI Gap"})
    # If still empty, generate from priority_actions
    for action in ai_analysis.get("priority_actions", []):
        if isinstance(action, dict) and len(faq_suggestions) < 5:
            why = action.get("why", "")
            if why and "?" not in why:
                # Convert action to a question format
                q = f"How to {action.get('action', '')[:80]}?"
                if q.lower() not in seen_qs:
                    seen_qs.add(q.lower())
                    faq_suggestions.append({"question": q, "volume": 100, "source": "AI Strategy"})

    # --- Blog topics from keyword gap + content gaps ---
    blog_topics = []
    for gap in kw_rec.get("content_gap_opportunities", []):
        if isinstance(gap, str) and gap:
            blog_topics.append({
                "title": gap,
                "keyword": gap.lower(),
                "volume": 300,
                "difficulty": 25,
                "word_target": 1800,
            })
    # Also pull from keyword_gap items with action "write blog post"
    for gap in kw_rec.get("keyword_gap", []):
        if isinstance(gap, dict) and "blog" in str(gap.get("action", "")).lower():
            blog_topics.append({
                "title": gap.get("keyword", ""),
                "keyword": gap.get("keyword", "").lower(),
                "volume": _volume_to_int(gap.get("estimated_volume", "medium")),
                "difficulty": _difficulty_to_int(gap.get("difficulty", "medium")),
                "word_target": 2000,
            })
    # Add blog topics from local_seo
    for topic in local_content.get("blog_topics", []):
        if isinstance(topic, str) and topic:
            blog_topics.append({
                "title": topic,
                "keyword": topic.lower(),
                "volume": 250,
                "difficulty": 22,
                "word_target": 1800,
            })

    return {
        "homepage_words": homepage_words,
        "competitor_avg_words": competitor_avg_words,
        "pages_to_rewrite": pages_to_rewrite[:10],
        "service_areas": service_areas[:10],
        "faq_suggestions": faq_suggestions[:10],
        "blog_topics": blog_topics[:10],
    }


def build_post_calendar(request_data: dict, keyword_data: dict) -> list[dict]:
    """Build a 4-week post calendar structure (topics only, not generated content)."""
    from datetime import timedelta

    keyword = request_data.get("keyword", "")
    biz_name = request_data.get("business_name", "")
    biz_type = request_data.get("business_type", "local business")
    location = request_data.get("location", "")
    # Extract city from location (e.g. "Toronto" from "Toronto, Canada")
    city = location.split(",")[0].strip() if location else "your area"
    # Use keyword for more natural post titles
    kw_short = keyword.rsplit(" ", 1)[0] if " " in keyword else keyword  # e.g. "kitchen cabinets"

    # Find next Monday
    today = datetime.now()
    days_ahead = 0 - today.weekday()  # Monday is 0
    if days_ahead <= 0:
        days_ahead += 7
    next_monday = today + timedelta(days=days_ahead)

    # Build 4 weeks of content
    weeks = []
    post_templates = [
        [
            {"type": "GBP", "title": f"Before & after: {kw_short} transformation"},
            {"type": "Social", "title": f"5 signs it's time to upgrade your {kw_short}"},
            {"type": "Blog", "title": f"{kw_short.title()} cost guide for {city} homeowners"},
        ],
        [
            {"type": "GBP", "title": f"Meet the {biz_name} team — behind the scenes"},
            {"type": "Social", "title": f"Top {kw_short} trends for {today.year}"},
            {"type": "Blog", "title": f"How to choose the right {kw_short} in {city}"},
        ],
        [
            {"type": "GBP", "title": f"Customer spotlight: happy {biz_name} client"},
            {"type": "Social", "title": f"Common {kw_short} mistakes to avoid"},
            {"type": "Blog", "title": f"{kw_short.title()} ideas {city} homeowners love"},
        ],
        [
            {"type": "GBP", "title": f"Seasonal {kw_short} tips for {city} homes"},
            {"type": "Social", "title": f"FAQ: questions our {kw_short} clients ask most"},
            {"type": "Blog", "title": f"DIY vs professional {kw_short}: {city} guide"},
        ],
    ]

    for i in range(4):
        week_start = next_monday + timedelta(weeks=i)
        weeks.append({
            "week": i + 1,
            "date": week_start.strftime("%Y-%m-%d"),
            "posts": post_templates[i],
        })

    return weeks


def build_auto_detected(request, auto_detected: dict | None, agents: dict) -> dict | None:
    """Build the auto_detected section, filling from agent data when needed."""
    if auto_detected:
        return auto_detected

    # Build from agent data if keyword was user-provided
    local_rec = agents.get("local_seo", {}).get("recommendations", {})
    gbp_opt = local_rec.get("gbp_optimization", {})
    categories = gbp_opt.get("categories", [])
    service_areas = local_rec.get("local_content_strategy", {}).get("service_area_pages", [])

    return {
        "business_type": request.business_type or "local business",
        "primary_keyword": request.keyword or "",
        "secondary_keywords": [],
        "services": service_areas[:5] if service_areas else [],
    }


def calculate_cost_estimate(agents_run: int = 4) -> float:
    """Rough cost per audit based on token usage."""
    # ~3K input + ~2K output per agent × 3 agents
    # Sonnet: $3/M input, $15/M output
    input_cost = (3000 * agents_run) / 1_000_000 * 3
    output_cost = (2000 * agents_run) / 1_000_000 * 15
    return round(input_cost + output_cost, 4)


def build_service_keywords(
    profile_services: list[str],
    location: str,
    keyword_data: dict,
    crawled_pages: list[dict],
) -> dict:
    """Build per-service keyword targets + page mapping. No Claude calls.

    Returns dict[service_name → {primary, related, current_page, has_dedicated_page, recommendation}].
    """
    from urllib.parse import urlparse

    if not profile_services:
        return {}

    city = location.split(",")[0].strip() if location else ""

    # Build a lookup: URL path → page dict
    page_paths: dict[str, dict] = {}
    for p in crawled_pages:
        path = urlparse(p.get("url", "")).path.lower().rstrip("/") or "/"
        page_paths[path] = p

    # Extract all keywords from keyword_data for matching
    all_kw_items = []
    for kw in keyword_data.get("keywords", []):
        if isinstance(kw, dict):
            all_kw_items.append(kw.get("keyword", "").lower())
        elif isinstance(kw, str):
            all_kw_items.append(kw.lower())

    result: dict = {}
    for service in profile_services:
        service_lower = service.lower().strip()
        if not service_lower:
            continue

        slug = service_lower.replace(" ", "-")
        primary = f"{service} {city}".strip() if city else service

        # Find related keywords from keyword_data
        related = [
            kw for kw in all_kw_items
            if service_lower in kw and kw != primary.lower()
        ][:5]

        # Check if a dedicated page exists for this service
        current_page = None
        has_dedicated = False
        for path, page in page_paths.items():
            if slug in path or service_lower.replace(" ", "") in path.replace("-", ""):
                current_page = page.get("url", "")
                has_dedicated = True
                break

        if has_dedicated:
            recommendation = f"Optimise existing page for '{primary}' — ensure keyword in title, H1, and first paragraph."
        else:
            recommendation = f"Create a dedicated service page targeting '{primary}' with 800+ words of locally-optimised content."

        result[service] = {
            "primary": primary,
            "related": related,
            "current_page": current_page,
            "has_dedicated_page": has_dedicated,
            "recommendation": recommendation,
        }

    return result


def build_schema_recommendations(crawled_pages: list[dict]) -> dict:
    """Rule-based schema type recommendations per page. No Claude calls.

    Returns dict[url_path → [schema_type, ...]].
    """
    from urllib.parse import urlparse

    SCHEMA_MAP = {
        "Homepage": ["LocalBusiness", "BreadcrumbList", "WebSite"],
        "Services": ["Service", "BreadcrumbList", "FAQPage"],
        "About": ["Organization", "BreadcrumbList", "AboutPage"],
        "Contact": ["LocalBusiness", "BreadcrumbList", "ContactPage"],
        "Blog Post": ["Article", "BreadcrumbList"],
        "FAQ": ["FAQPage", "BreadcrumbList"],
        "Pricing": ["Product", "BreadcrumbList", "AggregateOffer"],
        "Portfolio": ["CreativeWork", "BreadcrumbList", "ImageGallery"],
        "Testimonials": ["Review", "BreadcrumbList", "AggregateRating"],
        "Service Area": ["Service", "BreadcrumbList", "LocalBusiness"],
        "Careers": ["JobPosting", "BreadcrumbList"],
        "Legal": ["WebPage", "BreadcrumbList"],
        "Page": ["WebPage", "BreadcrumbList"],
    }

    result: dict = {}
    for p in crawled_pages:
        path = urlparse(p.get("url", "")).path.rstrip("/") or "/"
        page_type = p.get("page_type", "Page")
        result[path] = SCHEMA_MAP.get(page_type, ["WebPage", "BreadcrumbList"])

    return result


def _send_audit_email(to_email: str, report: dict) -> None:
    """Send a branded HTML audit summary email via Resend."""
    if not RESEND_API_KEY:
        logger.warning("RESEND_API_KEY not set — skipping email")
        return

    import resend
    resend.api_key = RESEND_API_KEY

    business_name = report.get("business_name") or report.get("target_url", "Your Business")
    business_type = report.get("business_type", "")
    score = report.get("scores", {}).get("overall", report.get("local_seo_score", 0))
    quick_wins_raw = report.get("quick_wins", report.get("summary", {}).get("quick_wins", []))
    quick_wins = [w.get("title", str(w)) if isinstance(w, dict) else str(w) for w in quick_wins_raw[:5]]
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
            <span style="font-size:20px;font-weight:700;color:#ffffff;vertical-align:middle;">LocalRankr</span>
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
          <p style="margin:24px 0 0;color:#52525b;font-size:11px;">LocalRankr &mdash; AI-Powered Local SEO Platform</p>
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


# =============================================================================
# PHASE 4 — LOCAL SEO SUB-AGENTS
# Agents: GBP Audit | Citation Builder | Rank Tracker
# =============================================================================

# ---------------------------------------------------------------------------
# Shared helper — rich SERP fetch (local pack + organic rankings)
# ---------------------------------------------------------------------------

async def _fetch_serp_rich(keyword: str, location: str, target_url: str) -> dict:
    """
    Single SerpAPI call that returns:
      - local_pack: up to 3 map-pack entries
      - client_map_rank: 1-3 or None
      - organic_results: top 20
      - client_organic_rank: 1-20 or None
      - serp_features: list of SERP feature names present
    """
    result: dict = {
        "local_pack": [],
        "client_map_rank": None,
        "organic_results": [],
        "client_organic_rank": None,
        "serp_features": [],
    }
    if not SERPAPI_KEY:
        return result

    from urllib.parse import urlparse
    client_domain = urlparse(target_url).netloc.lstrip("www.")

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://serpapi.com/search",
                params={
                    "engine": "google",
                    "q": keyword,
                    "location": location,
                    "num": "20",
                    "api_key": SERPAPI_KEY,
                },
            )
            if resp.status_code != 200:
                return result
            data = resp.json()
    except Exception:
        return result

    # Local pack — ensure list (SerpAPI sometimes returns a dict)
    local_results = data.get("local_results", [])
    if not isinstance(local_results, list):
        local_results = list(local_results.values()) if isinstance(local_results, dict) else []
    for i, place in enumerate(local_results[:3], start=1):
        if not isinstance(place, dict):
            continue
        entry = {
            "rank": i,
            "title": place.get("title", ""),
            "rating": place.get("rating"),
            "reviews": place.get("reviews"),
            "address": place.get("address", ""),
            "phone": place.get("phone", ""),
            "website": place.get("website", ""),
            "place_id": place.get("place_id", ""),
        }
        result["local_pack"].append(entry)
        if client_domain and client_domain in (entry["website"] or ""):
            result["client_map_rank"] = i

    # Organic results — ensure list
    organic_results = data.get("organic_results", [])
    if not isinstance(organic_results, list):
        organic_results = list(organic_results.values()) if isinstance(organic_results, dict) else []
    for i, org in enumerate(organic_results[:20], start=1):
        if not isinstance(org, dict):
            continue
        link = org.get("link", "")
        entry = {
            "rank": i,
            "title": org.get("title", ""),
            "url": link,
            "snippet": (org.get("snippet") or "")[:150],
        }
        result["organic_results"].append(entry)
        if client_domain and client_domain in link:
            result["client_organic_rank"] = i

    # SERP features
    features = []
    if data.get("local_results"):
        features.append("local_pack")
    if data.get("related_questions"):
        features.append("people_also_ask")
    if data.get("knowledge_graph"):
        features.append("knowledge_panel")
    if data.get("answer_box"):
        features.append("featured_snippet")
    if data.get("shopping_results"):
        features.append("shopping")
    result["serp_features"] = features

    return result


# ---------------------------------------------------------------------------
# 4a — GBP Audit Agent
# ---------------------------------------------------------------------------

GBP_SYSTEM = """You are a Google Business Profile (GBP) optimisation specialist for local businesses.
You ALWAYS respond with valid JSON only — no markdown, no explanation, no preamble.
Your audit is evidence-based: you score what you can verify and clearly flag what is unknown.
You focus on the 20% of GBP optimisations that drive 80% of Map Pack ranking improvements."""

GBP_PROMPT = """Audit the Google Business Profile optimisation for this local business. Respond with valid JSON only.

BUSINESS: {business_name} ({business_type}) in {location}
WEBSITE: {target_url}
KEYWORD: {keyword}

=== SERP DATA (live from Google) ===
Map Pack Position: {map_pack_rank}
Client appears in Local Pack: {in_pack}
Map Pack Competitors:
{map_pack_competitors}

=== ON-PAGE NAP SIGNALS (from website scrape) ===
Title tag: {title}
H1: {h1}
NAP on page: {nap_on_page}
Content excerpt (first 600 chars): {content_excerpt}

=== ORGANIC RANKINGS ===
Client organic rank for keyword: {organic_rank}
Top 5 organic competitors: {organic_competitors}

Return JSON with EXACTLY this structure:
{{
  "gbp_score": <int 0-100>,
  "map_pack_status": {{
    "in_pack": {in_pack},
    "current_rank": {map_pack_rank_raw},
    "pack_competitors": [<list of competitor names from pack>]
  }},
  "completeness_audit": {{
    "business_name_consistent": {{"status": "pass|warn|fail|unknown", "note": "<specific finding>"}},
    "primary_category": {{"status": "pass|warn|fail|unknown", "note": "<finding>"}},
    "service_areas": {{"status": "pass|warn|fail|unknown", "note": "<finding>"}},
    "business_hours": {{"status": "pass|warn|fail|unknown", "note": "<finding>"}},
    "phone_number": {{"status": "pass|warn|fail|unknown", "note": "<finding>"}},
    "website_linked": {{"status": "pass|warn|fail|unknown", "note": "<finding>"}},
    "photos_cover": {{"status": "pass|warn|fail|unknown", "note": "<finding>"}},
    "photos_interior": {{"status": "pass|warn|fail|unknown", "note": "<finding>"}},
    "posts_active": {{"status": "pass|warn|fail|unknown", "note": "<finding>"}},
    "qa_section": {{"status": "pass|warn|fail|unknown", "note": "<finding>"}},
    "products_services": {{"status": "pass|warn|fail|unknown", "note": "<finding>"}},
    "review_responses": {{"status": "pass|warn|fail|unknown", "note": "<finding>"}},
    "attributes": {{"status": "pass|warn|fail|unknown", "note": "<finding>"}},
    "description": {{"status": "pass|warn|fail|unknown", "note": "<finding>"}}
  }},
  "nap_consistency": {{
    "name_on_website": "<business name found on page>",
    "address_on_website": "<address found or 'not detected'>",
    "phone_on_website": "<phone found or 'not detected'>",
    "consistent": true|false,
    "issues": ["<any NAP inconsistency found>"]
  }},
  "review_strategy": {{
    "current_visibility": "<what reviews are visible in SERP>",
    "recommended_target": "<realistic target review count and rating>",
    "acquisition_tactics": ["<tactic 1>", "<tactic 2>", "<tactic 3>"]
  }},
  "priority_actions": [
    {{
      "action": "<specific GBP action>",
      "impact": "high|medium|low",
      "effort": "easy|medium|hard",
      "reason": "<why this improves Map Pack ranking>",
      "how_to": "<step-by-step implementation>"
    }}
  ],
  "competitor_insights": {{
    "what_competitors_do_better": ["<insight 1>", "<insight 2>"],
    "gaps_to_exploit": ["<opportunity 1>", "<opportunity 2>"]
  }},
  "summary": {{
    "top_priority": "<single most impactful action this week>",
    "estimated_pack_entry_timeline": "<realistic months to enter Local Pack>",
    "score_after_fixes": <realistic score after implementing all actions>
  }}
}}

Rules:
- gbp_score: score conservatively (most unoptimised businesses score 15-40)
- completeness_audit: use "unknown" when you genuinely cannot determine from SERP/page data
- priority_actions: exactly 5, ordered by impact descending
- Use actual competitor names from the map pack data above, not generic placeholders
- nap_consistency: scan the content excerpt for business name, address, and phone"""


@app.post("/agents/gbp-audit")
async def gbp_audit_agent(request: AuditRequest):
    """GBP Audit — map pack position, completeness checklist, NAP consistency, review strategy."""
    audit_id = str(uuid.uuid4())
    logger.info(f"[{audit_id}] GBP Audit starting for '{request.target_url}'")

    try:
        biz_name = request.business_name or "this business"
        biz_type = request.business_type or "local business"

        # Fetch SERP data + page data concurrently
        serp_data, page_data = await asyncio.gather(
            _fetch_serp_rich(request.keyword, request.location, request.target_url),
            scrape_page(request.target_url),
        )

        # Format map pack competitors
        pack_lines = []
        for entry in serp_data.get("local_pack", []):
            rating = f"{entry.get('rating', '?')}★ ({entry.get('reviews', '?')} reviews)" if entry.get("rating") else "no rating data"
            pack_lines.append(
                f"  #{entry.get('rank', '?')}: {entry.get('title', '?')} — {entry.get('address', '?')} — {rating}"
            )
        map_pack_str = "\n".join(pack_lines) or "No local pack found for this keyword."

        # Format top organic competitors
        organic_comp_lines = [
            f"  #{r.get('rank', '?')}: {r.get('title', '?')} — {r.get('url', '?')}"
            for r in serp_data.get("organic_results", [])[:5]
        ]
        organic_comps_str = "\n".join(organic_comp_lines) or "No organic results."

        map_rank = serp_data.get("client_map_rank")
        org_rank = serp_data.get("client_organic_rank")

        # Escape curly braces in user-supplied content to prevent format() crash
        content_excerpt = (page_data.get("content", "") or "")[:600].replace("{", "{{").replace("}", "}}")

        prompt = GBP_PROMPT.format(
            business_name=biz_name,
            business_type=biz_type,
            location=request.location,
            target_url=request.target_url,
            keyword=request.keyword,
            map_pack_rank=f"#{map_rank}" if map_rank else "Not in top 3",
            in_pack=bool(map_rank),
            map_pack_rank_raw=map_rank,
            map_pack_competitors=map_pack_str,
            organic_rank=f"#{org_rank}" if org_rank else "Not in top 20",
            organic_competitors=organic_comps_str,
            title=(page_data.get("title") or "N/A").replace("{", "{{").replace("}", "}}"),
            h1=(page_data.get("h1") or "N/A").replace("{", "{{").replace("}", "}}"),
            nap_on_page="Check content below",
            content_excerpt=content_excerpt,
        )

        analysis = await call_claude(GBP_SYSTEM, prompt, max_tokens=4000)

        return {
            "agent": "gbp_audit",
            "audit_id": audit_id,
            "status": "completed",
            "keyword": request.keyword,
            "target_url": request.target_url,
            "map_pack_rank": map_rank,
            "organic_rank": org_rank,
            "serp_features": serp_data.get("serp_features", []),
            "local_pack": serp_data.get("local_pack", []),
            "analysis": analysis,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"[{audit_id}] GBP Audit agent failed: {type(e).__name__}: {e}", exc_info=True)
        raise


# ---------------------------------------------------------------------------
# 4b — Citation Builder Agent
# ---------------------------------------------------------------------------

CITATION_SYSTEM = """You are a local SEO citations expert for small and medium businesses.
You ALWAYS respond with valid JSON only — no markdown, no explanation, no preamble.
You prioritise citations by their DA, relevance to the business type, and ease of submission.
You focus on free citations first, paid only when the ROI is clear."""

CITATION_PROMPT = """Build a prioritised citation plan for this local business. Respond with valid JSON only.

BUSINESS: {business_name} ({business_type}) in {location}
WEBSITE: {target_url}
KEYWORD: {keyword}
REGION: {region}

CITATION DATABASE (pre-filtered for this business):
The following {total_citations} citations are available. Select and prioritise the best ones:
{citation_list}

Return JSON with EXACTLY this structure:
{{
  "citation_score": <int 0-100, estimate of current citation health>,
  "total_in_database": {total_citations},
  "recommendations": {{
    "tier_1_critical": [
      {{
        "name": "<citation name>",
        "submit_url": "<submission URL>",
        "da": <int>,
        "free": true|false,
        "reason": "<why this specific citation matters for a {business_type}>",
        "time_to_list": "<e.g. 1-2 days, instant, 2-4 weeks>",
        "status": "likely_missing"
      }}
    ],
    "tier_2_important": [
      {{
        "name": "<citation name>",
        "submit_url": "<submission URL>",
        "da": <int>,
        "free": true|false,
        "reason": "<industry-specific reason>",
        "time_to_list": "<time estimate>",
        "status": "likely_missing"
      }}
    ],
    "tier_3_supplemental": [
      {{
        "name": "<citation name>",
        "submit_url": "<submission URL>",
        "da": <int>,
        "free": true|false,
        "reason": "<why worth adding>",
        "status": "likely_missing"
      }}
    ]
  }},
  "nap_template": {{
    "business_name": "{business_name}",
    "address": "<format: Street, City, Province/State, Postal/Zip>",
    "phone": "<format: (416) 555-0100 — use correct format for {location}>",
    "website": "{target_url}",
    "categories": ["<primary category>", "<secondary category>"],
    "description": "<150-200 word business description optimised for citations. Mentions {location} 2-3 times. Includes primary keyword naturally.>"
  }},
  "consistency_rules": [
    "<NAP consistency rule 1 — e.g. always use exact same name format>",
    "<rule 2>",
    "<rule 3>"
  ],
  "monthly_plan": {{
    "month_1": ["<citation 1>", "<citation 2>", "<citation 3>"],
    "month_2": ["<citation 4>", "<citation 5>"],
    "month_3": ["<remaining citations>"]
  }},
  "summary": {{
    "tier_1_count": <int>,
    "tier_2_count": <int>,
    "tier_3_count": <int>,
    "total_recommended": <int>,
    "estimated_da_impact": "<realistic DA improvement from citations alone>",
    "time_to_complete": "<total time estimate to submit all citations>"
  }}
}}

Rules:
- tier_1_critical: 5-8 citations — the non-negotiables for any {business_type} in {location}
- tier_2_important: 5-8 industry-specific or region-specific citations
- tier_3_supplemental: 3-5 nice-to-have general directories
- For Canadian businesses: prioritise .ca directories and include YellowPages.ca, Canada411
- nap_template.description must be unique, locally targeted, and genuinely useful for citations
- Do NOT include citations from the database that are clearly irrelevant to the business type"""


@app.post("/agents/citation-builder")
async def citation_builder_agent(request: AuditRequest):
    """Citation Builder — loads citations.json, filters by business type/region, generates NAP template."""
    audit_id = str(uuid.uuid4())
    logger.info(f"[{audit_id}] Citation Builder starting for '{request.business_name}'")

    biz_name = request.business_name or "this business"
    biz_type = (request.business_type or "local business").lower()
    location_lower = request.location.lower()

    # Load citations database
    citations_path = os.path.join(os.path.dirname(__file__), "citations.json")
    try:
        with open(citations_path) as f:
            all_citations: list[dict] = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_citations = []

    # Determine region
    region = "ca" if any(w in location_lower for w in ["canada", " on", " bc", " ab", " qc", "ontario", "toronto", "vancouver", "montreal", "calgary"]) else "us"

    # Filter: include if regions is global, matches region, or industries matches all/business type
    def _matches(c: dict) -> bool:
        regions = c.get("regions", ["global"])
        inds = c.get("industries", ["all"])
        region_ok = "global" in regions or region in regions
        ind_ok = "all" in inds or any(
            ind in biz_type or biz_type in ind for ind in inds
        )
        return region_ok and ind_ok

    filtered = [c for c in all_citations if _matches(c)]

    # Build compact citation list for the prompt
    citation_lines = []
    for c in sorted(filtered, key=lambda x: (-x.get("tier", 3), -x.get("da", 0))):
        free_label = "FREE" if c.get("free") else "PAID"
        citation_lines.append(
            f"[Tier {c.get('tier',3)} | DA:{c.get('da',0)} | {free_label}] "
            f"{c['name']} — submit: {c.get('submit_url', c['url'])}"
        )

    prompt = CITATION_PROMPT.format(
        business_name=biz_name,
        business_type=request.business_type or "local business",
        location=request.location,
        target_url=request.target_url,
        keyword=request.keyword,
        region=region.upper(),
        total_citations=len(filtered),
        citation_list="\n".join(citation_lines),
    )

    plan = await call_claude(CITATION_SYSTEM, prompt, max_tokens=4000)

    return {
        "agent": "citation_builder",
        "audit_id": audit_id,
        "status": "completed",
        "keyword": request.keyword,
        "target_url": request.target_url,
        "region": region,
        "citations_in_database": len(filtered),
        "plan": plan,
        "timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# 4c — Rank Tracker Agent (pure data, no Claude call needed)
# ---------------------------------------------------------------------------

@app.post("/agents/rank-tracker")
async def rank_tracker_agent(request: AuditRequest):
    """Rank Tracker — checks live Google rankings for keyword (organic + map pack)."""
    audit_id = str(uuid.uuid4())
    logger.info(f"[{audit_id}] Rank Tracker starting for '{request.keyword}' → {request.target_url}")

    serp_data = await _fetch_serp_rich(request.keyword, request.location, request.target_url)

    organic_rank = serp_data["client_organic_rank"]
    map_rank = serp_data["client_map_rank"]

    # Determine ranking health
    def _health(rank):
        if rank is None:
            return "not_ranking"
        if rank <= 3:
            return "excellent"
        if rank <= 10:
            return "good"
        if rank <= 20:
            return "improving"
        return "needs_work"

    # Estimate positions to close (to reach page 1 / map pack)
    positions_to_p1 = max(0, (organic_rank or 21) - 10)
    positions_to_pack = 0 if map_rank else 3  # need to enter pack

    top_organic = [
        {"rank": r["rank"], "title": r["title"], "url": r["url"]}
        for r in serp_data["organic_results"][:10]
    ]

    return {
        "agent": "rank_tracker",
        "audit_id": audit_id,
        "status": "completed",
        "keyword": request.keyword,
        "location": request.location,
        "target_url": request.target_url,
        "rankings": {
            "organic_rank": organic_rank,
            "organic_health": _health(organic_rank),
            "map_pack_rank": map_rank,
            "map_pack_health": _health(map_rank),
            "in_top_10": organic_rank is not None and organic_rank <= 10,
            "in_map_pack": map_rank is not None,
            "positions_to_page_1": positions_to_p1,
            "positions_to_map_pack": positions_to_pack,
        },
        "serp_features": serp_data["serp_features"],
        "local_pack": serp_data["local_pack"],
        "top_10_organic": top_organic,
        "snapshot_date": datetime.now().isoformat(),
    }


@app.post("/workflow/seo-audit")
async def seo_audit_workflow(request: AuditRequest, current_user: Optional[CurrentUser] = Depends(get_optional_user)):
    """
    Full SEO Audit — returns immediately with audit_id, runs in background.
    Poll GET /audits/{audit_id}/status for results.
    """
    audit_id = str(uuid.uuid4())
    _pending_audits[audit_id] = {"status": "processing"}
    asyncio.create_task(_run_audit_background(audit_id, request, current_user))
    return {"audit_id": audit_id, "status": "processing"}


async def _run_audit_background(audit_id: str, request: AuditRequest, current_user) -> None:
    """Run the full audit in the background and store result in _pending_audits."""
    try:
        result = await _do_audit_core(audit_id, request, current_user)
        _pending_audits[audit_id] = {"status": "completed", "result": result}
    except Exception as e:
        logger.error(f"[{audit_id}] Background audit failed: {e}", exc_info=True)
        _pending_audits[audit_id] = {"status": "failed"}
    # Auto-clean from memory after 2 hours
    await asyncio.sleep(7200)
    _pending_audits.pop(audit_id, None)


async def _do_audit_core(audit_id: str, request: AuditRequest, current_user) -> dict:
    """
    Full SEO Audit — always crawls the full site (sitemap-first, BFS fallback),
    runs keyword research first, then all agents + per-page analysis concurrently.
    Total time: ~90-120 seconds.
    """
    start = time.time()
    logger.info(f"[{audit_id}] Full audit starting for '{request.keyword}' → {request.target_url}")

    # Profile resolution — populate request fields from profile if provided
    profile_id = request.profile_id
    version = None
    if profile_id and current_user:
        db = SessionLocal()
        try:
            profile = db.query(Profile).filter(
                Profile.id == profile_id,
                Profile.user_id == current_user.id,
            ).first()
            if profile:
                # Fill empty request fields from profile
                if not request.target_url or request.target_url == "":
                    request.target_url = profile.website_url
                if not request.business_name:
                    request.business_name = profile.business_name
                if not request.business_type and profile.business_category:
                    request.business_type = profile.business_category
                if request.location == "Toronto, Canada" and profile.city:
                    loc_parts = [profile.city]
                    if profile.country:
                        loc_parts.append(profile.country)
                    request.location = ", ".join(loc_parts)
                # Compute next version number
                from sqlalchemy import func
                max_ver = db.query(func.max(Audit.version)).filter(
                    Audit.profile_id == profile_id
                ).scalar()
                version = (max_ver or 0) + 1
                logger.info(f"[{audit_id}] Profile {profile_id} → version {version}")
            else:
                logger.warning(f"[{audit_id}] Profile {profile_id} not found or not owned by user")
                profile_id = None
        finally:
            db.close()

    auto_detected: dict | None = None
    secondary_keywords: list[str] | None = None
    profile_services: list[str] = []

    # Parse services from profile if available
    if profile_id and current_user:
        db = SessionLocal()
        try:
            profile = db.query(Profile).filter(Profile.id == profile_id).first()
            if profile and profile.services:
                try:
                    parsed = json.loads(profile.services)
                    if isinstance(parsed, list):
                        profile_services = [s for s in parsed if isinstance(s, str) and s.strip()]
                except (json.JSONDecodeError, TypeError):
                    pass
        finally:
            db.close()

    try:
        # Phase 0 — always crawl the site (sitemap-first, then BFS fallback)
        logger.info(f"[{audit_id}] Crawling site: {request.target_url}")
        crawled_pages = await crawl_site(request.target_url, max_pages=MAX_CRAWL_PAGES)
        # Enrich pages with detected page type
        for p in crawled_pages:
            p["page_type"] = detect_page_type(p["url"], p.get("title", ""), request.location)
        crawl_aggregate = aggregate_crawl_results(crawled_pages)
        logger.info(
            f"[{audit_id}] Crawl done: {crawl_aggregate.get('pages_crawled', 0)} pages, "
            f"coverage {crawl_aggregate.get('coverage_score', 0)}/100"
        )

        # Phase 0.5 — auto-detect keyword (only when keyword is not provided)
        if not request.keyword:
            logger.info(f"[{audit_id}] No keyword provided — running auto-detection")
            auto_detected = await auto_detect_keyword(request)
            request.keyword = auto_detected["primary_keyword"]
            secondary_keywords = auto_detected.get("secondary_keywords") or None
            # Fill in business_type if user didn't provide one
            if not request.business_type and auto_detected.get("business_type"):
                request.business_type = auto_detected["business_type"]
            logger.info(f"[{audit_id}] Auto-detected keyword='{request.keyword}', type='{request.business_type}'")
            # Enrich profile_services from auto-detected if profile had none
            if not profile_services and auto_detected.get("services"):
                profile_services = [s for s in auto_detected["services"] if isinstance(s, str)]

        # Phase 1 — keyword research (other agents benefit from this data)
        keyword_results = await keyword_research_agent(request, secondary_keywords=secondary_keywords)

        # Phase 2, 3 + 4 — all independent agents run concurrently
        # On-page gets pre-crawled pages sorted by word count (homepage first),
        # technical gets the site-wide aggregate for site-level recommendations.
        sorted_pages = sorted(crawled_pages, key=lambda p: p.get("word_count", 0), reverse=True)
        # Ensure homepage is always first
        homepage_norm = _normalize_url(request.target_url)
        homepage_first = [p for p in sorted_pages if _normalize_url(p["url"]) == homepage_norm]
        other_pages = [p for p in sorted_pages if _normalize_url(p["url"]) != homepage_norm]
        pages_for_onpage = (homepage_first + other_pages) if crawled_pages else []

        concurrent_tasks = [
            on_page_seo_agent(request, pre_scraped_pages=pages_for_onpage or None),
            local_seo_agent(request),
            technical_seo_agent(request, crawl_aggregate=crawl_aggregate or None),
            content_rewriter_agent(request),
            backlink_analysis_agent(request),
            link_building_agent(request),
            ai_seo_agent(request),
            gbp_audit_agent(request),
            citation_builder_agent(request),
            rank_tracker_agent(request),
            analyze_pages_batch(crawled_pages, request),
        ]
        if request.include_blog:
            concurrent_tasks.append(blog_writer_agent(request))

        results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)

        # Log any per-agent exceptions so we can diagnose without hiding them
        agent_names = ["on_page_seo", "local_seo", "technical_seo", "content_rewriter",
                       "backlink_analysis", "link_building", "ai_seo", "gbp_audit",
                       "citation_builder", "rank_tracker", "page_analysis"]
        for name, res in zip(agent_names, results[:11]):
            if isinstance(res, Exception):
                logger.error(f"[{audit_id}] Agent '{name}' raised: {type(res).__name__}: {res}")

        # Replace any exceptions with empty dicts so the rest of the report still builds
        results = [r if not isinstance(r, Exception) else {} for r in results]

        # Mark agents whose Claude calls failed (error dicts in recommendations/analysis)
        # so downstream score builders and the frontend can distinguish partial from real data
        for res in [keyword_results] + list(results):
            if not isinstance(res, dict):
                continue
            for key in ("recommendations", "analysis", "plan"):
                if key in res and _is_claude_error(res[key]):
                    res["status"] = "error"
                    res["_claude_error"] = res[key].get("error", "Unknown error")
                    break

        (
            on_page_results,
            local_results,
            technical_results,
            rewriter_results,
            backlink_results,
            linkbuilding_results,
            aiseo_results,
            gbp_results,
            citation_results,
            rank_results,
            page_analysis_results,
        ) = results[:11]
        blog_results = results[11] if request.include_blog else None

        elapsed = round(time.time() - start, 1)
        agents_run = 12 + (1 if request.include_blog else 0)
        logger.info(f"[{audit_id}] Audit completed in {elapsed}s ({agents_run} agents)")

        agents_dict = {
            "keyword_research": keyword_results,
            "on_page_seo": on_page_results,
            "local_seo": local_results,
            "technical_seo": technical_results,
            "content_rewriter": rewriter_results,
            "backlink_analysis": backlink_results,
            "link_building": linkbuilding_results,
            "ai_seo": aiseo_results,
            "gbp_audit": gbp_results,
            "citation_builder": citation_results,
            "rank_tracker": rank_results,
        }
        if blog_results:
            agents_dict["blog_writer"] = blog_results

        # Ensure page_analysis_results is a dict (could be {} from exception fallback)
        if not isinstance(page_analysis_results, dict):
            page_analysis_results = {}

        # Build site_crawl summary with per-page breakdown
        from urllib.parse import urlparse as _urlparse
        site_crawl_pages = []
        for p in crawled_pages:
            p_url = p.get("url", "")
            p_path = _urlparse(p_url).path.rstrip("/") or "/"
            pa = page_analysis_results.get(p_path, {})
            issues = [
                issue for issue in [
                    "Missing title" if not p.get("title") else None,
                    "Missing meta description" if not p.get("meta_description") else None,
                    "Missing H1" if not p.get("h1") else None,
                    "Thin content (<300 words)" if p.get("word_count", 0) < 300 else None,
                ]
                if issue is not None
            ]
            # Merge issues from per-page analysis
            if pa.get("issues"):
                issues = list(set(issues + pa["issues"]))
            site_crawl_pages.append({
                "url": p_url,
                "title": p.get("title", ""),
                "type": p.get("page_type", "Page"),
                "score": pa.get("page_score", 0),
                "word_count": p.get("word_count", 0),
                "issues_count": len(issues),
                "top_issue": issues[0] if issues else None,
            })

        site_crawl = {
            "pages_found": len(crawled_pages),
            "pages_analyzed": len(page_analysis_results),
            "pages": site_crawl_pages,
        }

        # Build service keywords + schema recommendations (pure functions)
        # Calculate structured scores and step data
        scores = calculate_pillar_scores(agents_dict)
        score_details = build_score_details(agents_dict)
        structured_wins = build_structured_quick_wins(agents_dict)
        pillars = build_pillar_steps(agents_dict, scores)
        seo_tasks = build_seo_tasks(structured_wins, pillars)
        est_cost = calculate_cost_estimate()

        # Build tool-page data sections
        request_data = {
            "keyword": request.keyword,
            "business_name": request.business_name or "",
            "business_type": request.business_type or "local business",
            "location": request.location,
            "target_url": request.target_url,
        }
        gmb_data = build_gmb_data(agents_dict)
        keyword_data = build_keyword_data(agents_dict, auto_detected, profile_services)
        backlink_data = build_backlink_data(agents_dict)
        content_data = build_content_data(agents_dict)
        post_calendar = build_post_calendar(request_data, keyword_data)
        auto_detected_section = build_auto_detected(request, auto_detected, agents_dict)

        # Pure function outputs: per-service keyword targets + schema recommendations
        service_keywords = build_service_keywords(profile_services, request.location, keyword_data, crawled_pages)
        schema_recs = build_schema_recommendations(crawled_pages)

        report = {
            "audit_id": audit_id,
            "profile_id": profile_id,
            "version": version,
            "business_name": request.business_name or "",
            "business_type": request.business_type or "other",
            "keyword": request.keyword,
            "target_url": request.target_url,
            "location": request.location,
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "agents_executed": agents_run,
            "execution_time_seconds": elapsed,
            "estimated_cost": est_cost,
            # Auto-detected data
            "auto_detected": auto_detected_section,
            # Scores
            "scores": scores,
            "score_details": score_details,
            # Tool-page data (flat sections for frontend)
            "gmb_data": gmb_data,
            "keyword_data": keyword_data,
            "backlink_data": backlink_data,
            "content_data": content_data,
            "post_calendar": post_calendar,
            # Quick wins, pillars, tasks (for overview + tasks pages)
            "quick_wins": structured_wins,
            "pillars": pillars,
            "seo_tasks": seo_tasks,
            # Full-site crawl data
            "site_crawl": site_crawl,
            "per_page_analysis": page_analysis_results,
            "service_keywords": service_keywords,
            "schema_recommendations": schema_recs,
            # Backward compat
            "local_seo_score": scores["overall"],
            "domain": request.domain or "",
            "site_aggregate": crawl_aggregate,
            "pages_crawled": site_crawl["pages"],
            # Raw agent data (for debugging / advanced views)
            "agents": agents_dict,
            "summary": {
                "estimated_api_cost": est_cost,
                "quick_wins": [w["title"] for w in structured_wins],
            },
        }

        # Persist to DB — run in thread pool so we don't block the event loop
        loop = asyncio.get_event_loop()
        user_id_for_db = current_user.id if current_user else None
        await loop.run_in_executor(None, _save_audit, audit_id, request, report, elapsed, user_id_for_db, profile_id, version)

        # Send audit summary email (non-blocking — failure won't affect response)
        if current_user:
            await loop.run_in_executor(None, _send_audit_email, current_user.email, report)

        return report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{audit_id}] Audit failed: {e}", exc_info=True)
        raise


def _save_audit(audit_id: str, request, report: dict, elapsed: float, user_id: str = None, profile_id: str = None, version: int = None) -> None:
    """Synchronous DB write — called via run_in_executor."""
    try:
        # Sanitize JSON: remove null bytes (PostgreSQL rejects \x00 in text columns)
        results_str = json.dumps(report, default=str).replace("\x00", "")
        db = SessionLocal()
        db.add(Audit(
            id=audit_id,
            user_id=user_id,
            keyword=request.keyword,
            target_url=request.target_url,
            location=request.location,
            status="completed",
            results_json=results_str,
            api_cost=report.get("summary", {}).get("estimated_api_cost", 0.0),
            execution_time=elapsed,
            profile_id=profile_id,
            version=version,
            business_name=request.business_name,
            business_type=request.business_type,
            pages_crawled=len(report.get("pages_crawled", [])) or None,
        ))
        db.commit()
        logger.info(f"[{audit_id}] Saved to database (user={user_id}, profile={profile_id}, v={version})")
    except Exception as e:
        logger.error(f"[{audit_id}] DB save failed: {type(e).__name__}: {e}", exc_info=True)
    finally:
        db.close()


# =============================================================================
# Audit status polling (no auth — audit_id is the token)
# =============================================================================

@app.get("/audits/{audit_id}/status")
def get_audit_status(audit_id: str):
    """
    Poll for background audit completion.
    Returns {"status": "processing"}, {"status": "failed"}, or the full report with {"status": "completed", ...}.
    No authentication required — the audit_id itself acts as the access token.
    """
    entry = _pending_audits.get(audit_id)
    if entry is not None:
        if entry["status"] == "completed":
            return entry["result"]
        return {"status": entry["status"]}

    # Not in memory — check DB (handles server restart edge case)
    db = SessionLocal()
    try:
        row = db.query(Audit).filter(Audit.id == audit_id).first()
        if row and row.results_json:
            return json.loads(row.results_json)
        raise HTTPException(status_code=404, detail="Audit not found")
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
                "profile_id": r.profile_id,
                "version": r.version,
                "business_name": r.business_name,
                "business_type": r.business_type,
                "pages_crawled": r.pages_crawled,
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


@app.patch("/audits/{audit_id}/tasks/{task_id}")
def update_task_status(
    audit_id: str,
    task_id: str,
    body: dict,
    current_user: CurrentUser = Depends(get_optional_user),
):
    """Toggle a single SEO task status (completed/pending)."""
    new_status = body.get("status", "completed")
    if new_status not in ("completed", "pending"):
        raise HTTPException(status_code=400, detail="status must be 'completed' or 'pending'")

    db = SessionLocal()
    try:
        filters = [Audit.id == audit_id]
        if current_user:
            filters.append(Audit.user_id == current_user.id)
        row = db.query(Audit).filter(*filters).first()
        if not row:
            raise HTTPException(status_code=404, detail="Audit not found")

        audit_data = json.loads(row.results_json)
        tasks = audit_data.get("seo_tasks", [])
        found = False
        for t in tasks:
            if t.get("id") == task_id:
                t["status"] = new_status
                t["completed_at"] = datetime.utcnow().isoformat() if new_status == "completed" else None
                found = True
                break
        if not found:
            raise HTTPException(status_code=404, detail="Task not found")

        audit_data["seo_tasks"] = tasks
        row.results_json = json.dumps(audit_data, default=str)
        db.commit()
        return {"ok": True, "task_id": task_id, "status": new_status}
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
# Content Generation Endpoints (Post Creator & Content Writer)
# =============================================================================

# ── System prompts for content generation ─────────────────────────────────────

CALENDAR_SYSTEM = """You are a local SEO content strategist. Generate a 4-week content calendar for a local business.
Each week has 3 posts: a Google Business Profile (GBP) post, a social media post, and a blog intro.
All content should be optimized for the business's primary keyword and location.
Make each topic unique, specific, and locally relevant — reference the city, nearby areas, and seasonal trends.
You ALWAYS respond with valid JSON only — no markdown, no explanation, no preamble.
Return format: {"weeks": [{"label": "Week 1", "posts": [{"type": "GBP Post", "title": "..."}, {"type": "Social Post", "title": "..."}, {"type": "Blog Intro", "title": "..."}]}]}"""

POST_GENERATE_SYSTEM = """You are a local SEO content writer. Generate content for the specified post type.
GBP post: 1500 chars max, include a call-to-action, mention the city/location prominently, use the primary keyword naturally.
Social post: Instagram-style caption, engaging and shareable, include relevant hashtags.
Blog intro: 300 words with an SEO-optimized opening paragraph + a 5-section outline for the full article.
All content must reference the business name, location, and primary keyword naturally.
You ALWAYS respond with valid JSON only — no markdown, no explanation, no preamble.
Return format: {"content": "the post content here", "hashtags": ["#relevant", "#hashtags"], "word_count": 250, "seo_score": 85}
seo_score: rate 0-100 based on keyword usage, local relevance, CTA presence, and readability."""

REVIEW_RESPONSE_SYSTEM = """You are a reputation management expert for local businesses.
Draft a professional, empathetic response to a customer review.
For positive reviews (4-5 stars): Thank them warmly, mention specific details from their review, invite them back.
For neutral reviews (3 stars): Acknowledge their feedback, address concerns, offer to improve.
For negative reviews (1-2 stars): Apologize sincerely, address specifics, offer resolution, take conversation offline.
Never be defensive. Keep responses 50-150 words.
You ALWAYS respond with valid JSON only — no markdown, no explanation, no preamble.
Return format: {"response": "the response text here"}"""

CONTENT_GENERATE_SYSTEM = """You are an expert local SEO content writer.
Generate high-quality, SEO-optimized content for the specified page type.
Include the target keyword naturally throughout. Reference the business name, city, and services.
For page_rewrite: Full rewrite of the page, 1500+ words, proper H2/H3 heading structure, strong local SEO signals.
For service_area_page: Location-specific service page (800 words) with local references, landmarks, and service details.
For faq_answer: Schema-ready FAQ answer, 100-150 words, authoritative and concise.
For blog_article: Full article, 1200-2000 words, SEO-optimized with H2/H3 structure, internal linking suggestions.
You ALWAYS respond with valid JSON only — no markdown, no explanation, no preamble.
Return format: {"content": "the content here", "word_count": 850, "meta_title": "SEO title under 60 chars", "meta_description": "155-char description with keyword", "suggested_title": "Page title suggestion", "keyword_density": 1.5}"""


class CalendarRequest(BaseModel):
    keyword: str = ""
    business_name: Optional[str] = None
    business_type: Optional[str] = None
    location: str = "Toronto, Canada"


class PostGenerateRequest(BaseModel):
    topic: str
    post_type: str = "GBP Post"
    keyword: str = ""
    business_name: Optional[str] = None
    business_type: Optional[str] = None
    location: str = "Toronto, Canada"


class ReviewResponseRequest(BaseModel):
    review_text: str
    rating: int = 5
    business_name: Optional[str] = None
    business_type: Optional[str] = None


class ContentGenerateRequest(BaseModel):
    page_type: str  # page_rewrite, service_area_page, faq_answer, blog_article
    keyword: str = ""
    target_url: str = ""
    business_name: Optional[str] = None
    business_type: Optional[str] = None
    location: str = "Toronto, Canada"
    context: Optional[str] = None  # extra context (area name, question, topic, etc.)


@app.post("/api/generate-calendar")
async def generate_calendar(
    request: CalendarRequest,
    current_user: Optional[CurrentUser] = Depends(get_optional_user),
):
    prompt = (
        f"Create a 4-week content calendar for {request.business_name or 'a local business'} "
        f"({request.business_type or 'local service'}) in {request.location}. "
        f"Primary keyword: \"{request.keyword}\". "
        f"Each week should have 3 posts targeting different aspects of the business."
    )
    result = await call_claude(CALENDAR_SYSTEM, prompt, max_tokens=2000)
    return result if isinstance(result, dict) else {"weeks": []}


@app.post("/api/generate-post")
async def generate_post(
    request: PostGenerateRequest,
    current_user: Optional[CurrentUser] = Depends(get_optional_user),
):
    prompt = (
        f"Write a {request.post_type} about: \"{request.topic}\"\n"
        f"Business: {request.business_name or 'Local Business'} ({request.business_type or 'service'})\n"
        f"Location: {request.location}\n"
        f"Primary keyword: \"{request.keyword}\"\n"
        f"Make it engaging, locally relevant, and SEO-friendly."
    )
    result = await call_claude(POST_GENERATE_SYSTEM, prompt, max_tokens=1500)
    return result if isinstance(result, dict) else {"content": str(result)}


@app.post("/api/generate-review-response")
async def generate_review_response(
    request: ReviewResponseRequest,
    current_user: Optional[CurrentUser] = Depends(get_optional_user),
):
    star_label = (
        "positive" if request.rating >= 4
        else "neutral" if request.rating == 3
        else "negative"
    )
    prompt = (
        f"Write a {star_label} review response for {request.business_name or 'our business'} "
        f"({request.business_type or 'local service'}).\n"
        f"Rating: {request.rating}/5 stars\n"
        f"Customer review: \"{request.review_text}\""
    )
    result = await call_claude(REVIEW_RESPONSE_SYSTEM, prompt, max_tokens=800)
    return result if isinstance(result, dict) else {"response": str(result)}


@app.post("/api/generate-content")
async def generate_content(
    request: ContentGenerateRequest,
    current_user: Optional[CurrentUser] = Depends(get_optional_user),
):
    page_label = request.page_type.replace("_", " ").title()
    prompt = (
        f"Generate a {page_label} for {request.business_name or 'a local business'} "
        f"({request.business_type or 'service'}) in {request.location}.\n"
        f"Target keyword: \"{request.keyword}\"\n"
        f"Target URL: {request.target_url}\n"
    )
    if request.context:
        prompt += f"Additional context: {request.context}\n"
    prompt += "Write comprehensive, SEO-optimized content."
    result = await call_claude(CONTENT_GENERATE_SYSTEM, prompt, max_tokens=3000)
    return result if isinstance(result, dict) else {"content": str(result)}


MANUAL_KW_RESEARCH_SYSTEM = (
    "You are a local SEO keyword analyst. You analyse keywords for local businesses "
    "and return actionable data. Respond with valid JSON only — no markdown fences, "
    "no extra text."
)


@app.post("/api/keyword-research")
async def manual_keyword_research(
    body: dict,
    current_user: Optional[CurrentUser] = Depends(get_optional_user),
):
    keyword = body.get("keyword", "").strip()
    location = body.get("location", "").strip()
    business_type = body.get("business_type", "local business").strip()

    if not keyword:
        raise HTTPException(status_code=400, detail="keyword is required")

    prompt = (
        f"Analyse this keyword for a {business_type} in {location}: \"{keyword}\".\n\n"
        f"Return a JSON object with these exact keys:\n"
        f"- keyword (string): the keyword analysed\n"
        f"- volume_estimate (integer): estimated monthly search volume\n"
        f"- difficulty_estimate (integer 0-100): SEO difficulty score\n"
        f"- intent (string): one of commercial, informational, navigational, transactional\n"
        f"- related_keywords (array of 8-10 strings): closely related keywords\n"
        f"- serp_analysis (string): what type of content currently ranks for this keyword\n"
        f"- recommendation (string): how a {business_type} should target this keyword\n"
        f"- content_type (string): one of blog, service page, landing page"
    )

    result = await call_claude(MANUAL_KW_RESEARCH_SYSTEM, prompt, max_tokens=2000)

    if isinstance(result, dict):
        # Ensure the keyword field is set
        result.setdefault("keyword", keyword)
        return result
    return {"keyword": keyword, "error": "Could not parse result"}


# =============================================================================
# Keyword Gap Analysis (on-demand competitor comparison)
# =============================================================================

KEYWORD_GAP_SYSTEM = (
    "You are a local SEO keyword gap analyst. You compare a competitor's website content "
    "against a target business to identify keyword opportunities the target is missing. "
    "Respond with valid JSON only — no markdown fences, no extra text."
)


@app.post("/api/keyword-gap")
async def keyword_gap_analysis(
    body: dict,
    current_user: Optional[CurrentUser] = Depends(get_optional_user),
):
    """Scrape a competitor URL and identify keyword gaps vs the target business."""
    competitor_url = (body.get("competitor_url") or "").strip()
    target_url = (body.get("target_url") or "").strip()
    keyword = (body.get("keyword") or "").strip()
    location = (body.get("location") or "").strip()
    business_type = (body.get("business_type") or "local business").strip()

    if not competitor_url:
        raise HTTPException(status_code=400, detail="competitor_url is required")

    # Scrape competitor page
    comp_data = await scrape_page(competitor_url)
    if not comp_data.get("success"):
        raise HTTPException(
            status_code=422,
            detail=f"Could not scrape competitor URL: {comp_data.get('error', 'unknown error')}",
        )

    prompt = (
        f"Compare this competitor website against a {business_type} "
        f"{'in ' + location if location else ''} "
        f"{'targeting \"' + keyword + '\"' if keyword else ''}.\n\n"
        f"COMPETITOR PAGE DATA:\n"
        f"- URL: {competitor_url}\n"
        f"- Title: {comp_data.get('title', 'N/A')}\n"
        f"- H1: {comp_data.get('h1', 'N/A')}\n"
        f"- Meta description: {comp_data.get('meta_description', 'N/A')}\n"
        f"- Headings: {json.dumps(comp_data.get('headings', [])[:15])}\n"
        f"- Word count: {comp_data.get('word_count', 0)}\n"
        f"- Content preview: {comp_data.get('content', '')[:2000]}\n\n"
        f"TARGET BUSINESS:\n"
        f"- URL: {target_url or 'not provided'}\n"
        f"- Type: {business_type}\n"
        f"- Location: {location or 'not provided'}\n"
        f"- Primary keyword: {keyword or 'not provided'}\n\n"
        f"Identify keywords the competitor targets that the target business likely misses.\n\n"
        f"Return JSON with EXACTLY these keys:\n"
        f'{{\n'
        f'  "competitor_url": "{competitor_url}",\n'
        f'  "competitor_title": "<title from scraped data>",\n'
        f'  "estimated_da": <integer 0-100, estimate domain authority>,\n'
        f'  "keywords_found": <integer, total keywords identified on competitor page>,\n'
        f'  "gaps": [\n'
        f'    {{\n'
        f'      "keyword": "...",\n'
        f'      "volume_estimate": <integer>,\n'
        f'      "difficulty_estimate": <integer 0-100>,\n'
        f'      "competitor_position": "strong|moderate|weak",\n'
        f'      "opportunity": "Create page|Optimize existing|Write blog post|Add to service page",\n'
        f'      "intent": "commercial|informational|transactional|navigational"\n'
        f'    }}\n'
        f'  ],\n'
        f'  "overlap_keywords": ["keywords both sites likely target"],\n'
        f'  "summary": "One paragraph analysis of the competitive gap"\n'
        f'}}\n\n'
        f"Rules:\n"
        f"- gaps: provide 10-15 keyword gaps, ranked by opportunity value\n"
        f"- estimated_da: realistic estimate based on content quality and site signals\n"
        f"- overlap_keywords: 5-10 keywords both sites likely compete on\n"
        f"- Be specific to {business_type} keywords, not generic terms"
    )

    result = await call_claude(KEYWORD_GAP_SYSTEM, prompt, max_tokens=2500)

    if isinstance(result, dict) and "error" not in result:
        result.setdefault("competitor_url", competitor_url)
        result.setdefault("competitor_title", comp_data.get("title", ""))
        return result

    return {
        "competitor_url": competitor_url,
        "competitor_title": comp_data.get("title", ""),
        "estimated_da": 0,
        "keywords_found": 0,
        "gaps": [],
        "overlap_keywords": [],
        "summary": "Analysis could not be completed.",
        "error": result.get("error") if isinstance(result, dict) else "Could not parse result",
    }


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
