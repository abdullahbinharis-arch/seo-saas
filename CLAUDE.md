# CLAUDE.md — Project Instructions for Claude Code

## Project Overview

**LocalRank — AI Local SEO Platform**

A FastAPI backend with 4 AI agents (Keyword Research, On-Page SEO, Local SEO, Technical SEO) that analyse local business websites and return actionable SEO recommendations. Includes JWT-based auth (email/password + Google OAuth), PDF report export, and a dark-themed Next.js frontend.

## Tech Stack

- **Backend:** Python 3.10+, FastAPI, AsyncAnthropic, httpx, BeautifulSoup4, fpdf2
- **Frontend:** Next.js 14+ (App Router), TypeScript, Tailwind CSS v4, Outfit + Geist fonts
- **AI:** Claude API via `anthropic` SDK (async client only) — model: `claude-sonnet-4-6`
- **Auth:** JWT (python-jose) + bcrypt passwords + Google OAuth (synced via FastAPI)
- **Data:** SerpApi for Google results, httpx for scraping
- **Email:** Resend (`resend` package) — triggered after audit completes
- **Deployment:** Railway (backend), Vercel (frontend)

## Architecture

```
POST /workflow/seo-audit  (requires JWT)
  → keyword_research_agent        (runs first — other agents benefit from this context)
  → on_page_seo_agent             ┐
  → local_seo_agent               ├─ run concurrently via asyncio.gather
  → technical_seo_agent           ┘
  → combined report with local_seo_score + dynamic quick_wins
  → _save_audit() → PostgreSQL
  → _send_audit_email() → Resend (non-blocking, skipped if RESEND_API_KEY not set)
```

## Auth Flow

```
Registration:  POST /auth/register  → bcrypt hash → User row → JWT
Login:         POST /auth/login     → verify hash → JWT
Google OAuth:  POST /auth/oauth-sync → upsert User → JWT
All JWTs validated via get_current_user() FastAPI dependency (Bearer token)
NextAuth.js on frontend stores JWT in session, sends as Authorization: Bearer <token>
```

## Key Rules

1. **Always use AsyncAnthropic** — never the sync client. All Claude calls must be `await`ed.
2. **All Claude calls go through `call_claude()`** — the centralised helper in main.py. Never call `anthropic_client.messages.create()` directly from endpoints.
3. **JSON extraction uses `extract_json()`** — never raw regex. This handles markdown fences, preamble text, and unbalanced braces.
4. **System prompts are separate constants** — defined at module level (e.g., `KEYWORD_SYSTEM`), not inline in functions.
5. **No assistant prefill** — `claude-sonnet-4-6` does not support it. Force JSON via the system prompt wording ("respond with valid JSON only") and `extract_json()` handles parsing.
6. **Environment variables load via python-dotenv** — `load_dotenv()` is called at startup.
7. **Error messages to users are generic** — never expose stack traces, API keys, or internal details in HTTP responses.
8. **Rate limiting is enabled** — in-memory, configurable via `RATE_LIMIT_PER_MIN`.
9. **Scraping uses BeautifulSoup** — not stdlib html.parser. Extract title, meta desc, H1, headings, word count, and links.
10. **CORS origins are configurable** — via `ALLOWED_ORIGINS` env var, not hardcoded `["*"]`.
11. **PDF export uses fpdf2** — via `pdf_export.py`. Never use reportlab or weasyprint.
12. **Email is non-blocking** — `_send_audit_email()` runs in `run_in_executor`. A failure must never affect the audit response.

## File Structure

```
seo-saas/
├── main.py              ← FastAPI backend — all 4 agents, auth, orchestrator, email
├── database.py          ← SQLAlchemy models: User, Audit
├── pdf_export.py        ← fpdf2 PDF report builder (build_pdf(audit_dict) → bytes)
├── requirements.txt     ← Python deps
├── .env                 ← Actual keys (never commit)
├── .env.example         ← Template for API keys
├── .gitignore
├── Procfile             ← Railway deployment
├── CLAUDE.md            ← This file
└── README.md            ← Quick start

seo-frontend/
├── src/app/
│   ├── page.tsx                     ← Dark landing page (public)
│   ├── audit/page.tsx               ← Audit form page (protected)
│   ├── (auth)/login/page.tsx        ← Login (dark theme)
│   ├── (auth)/register/page.tsx     ← Register (dark theme)
│   ├── api/auth/[...nextauth]/      ← NextAuth route handler
│   └── globals.css                  ← Tailwind v4 theme + custom classes
├── src/components/
│   ├── AuditForm.tsx                ← Business fields, runs audit, shows results
│   ├── AuditResults.tsx             ← Full results display + Download PDF button
│   ├── ProgressIndicator.tsx        ← Animated audit progress
│   └── RevealObserver.tsx           ← IntersectionObserver for scroll animations
├── src/auth.ts                      ← NextAuth v5 config (Credentials + Google)
├── src/proxy.ts                     ← Middleware: protects /audit, public / /login /register
└── src/types/index.ts               ← Shared TypeScript types
```

## Environment Variables

### Backend (Railway)
```
ANTHROPIC_API_KEY=
SERPAPI_KEY=
JWT_SECRET=              # 32+ char random hex — must match NEXTAUTH_SECRET on frontend
ALLOWED_ORIGINS=         # Comma-separated: https://yourdomain.vercel.app,...
DATABASE_URL=            # PostgreSQL (Railway provides this automatically)
CLAUDE_MODEL=claude-sonnet-4-6
RESEND_API_KEY=          # Optional — email skipped if not set
FROM_EMAIL=              # e.g. LocalRank <noreply@yourdomain.com>
RATE_LIMIT_PER_MIN=10
```

### Frontend (Vercel)
```
NEXT_PUBLIC_API_URL=     # Railway backend URL
NEXTAUTH_URL=            # Vercel frontend URL
NEXTAUTH_SECRET=         # Must match JWT_SECRET above
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
```

## Common Tasks

### Add a new agent
1. Define `NEWAGENT_SYSTEM` and `NEWAGENT_PROMPT` constants at module level
2. Create `async def newagent(request: AuditRequest)` following existing agent pattern
3. Call `call_claude(NEWAGENT_SYSTEM, prompt, max_tokens=N)`
4. Add to `seo_audit_workflow` inside `asyncio.gather()`
5. Update `build_quick_wins()` to pull from new agent's output

### Update a prompt
- Edit the module-level constant (e.g., `KEYWORD_PROMPT`)
- Test individually before running the full workflow:
```bash
curl -X POST http://localhost:8000/agents/keyword-research \
  -H "Content-Type: application/json" \
  -d '{"keyword": "dentist near me", "target_url": "https://example.com", "location": "Toronto, Canada", "business_name": "Test Dental", "business_type": "dentist"}'
```

### Run locally
```bash
source venv/bin/activate
python main.py
# Test: curl http://localhost:8000/health
```

### Generate a PDF locally
```python
from pdf_export import build_pdf
pdf_bytes = build_pdf(audit_dict)
open("report.pdf", "wb").write(pdf_bytes)
```

## Testing

Test each agent individually, then the full workflow:
```bash
# Health check
curl http://localhost:8000/health

# Register + get token
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "test@example.com", "password": "password123"}'

# Full audit (use token from above)
curl -X POST http://localhost:8000/workflow/seo-audit \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"keyword": "dentist near me", "target_url": "https://example.com", "location": "Toronto, Canada", "business_name": "Test Dental", "business_type": "dentist"}'
```

## Dependencies

- `anthropic>=0.39.0` — AsyncAnthropic + current model names
- `claude-sonnet-4-6` — correct model ID (not `claude-sonnet-4-20250514`)
- `fpdf2>=2.7.9` — PDF generation (not reportlab)
- `resend>=2.0.0` — transactional email

## Roadmap

### Completed
- [x] 4 AI agents: Keyword Research, On-Page SEO, Local SEO, Technical SEO
- [x] Full auth: JWT, email/password, Google OAuth
- [x] PostgreSQL audit history per user
- [x] PDF report export (`/audits/{id}/export`)
- [x] Email on audit complete (Resend, non-blocking)
- [x] Dark glassmorphism UI (LocalRank brand)
- [x] Local SEO Score (0–100 composite)

### Planned — Rate Limiting Tiers
Replace the flat `RATE_LIMIT_PER_MIN` with per-user tier enforcement:

| Tier | Audits/month | Rate limit | Price |
|------|-------------|------------|-------|
| Free | 3 | 1/hour | $0 |
| Starter | 25 | 5/day | $29/mo |
| Pro | 100 | 20/day | $79/mo |
| Agency | Unlimited | 50/day | $199/mo |

Implementation plan:
1. Add `plan` column (`free|starter|pro|agency`) to `User` model
2. Add `monthly_audit_count` + `reset_date` to `User`
3. Replace in-memory rate limiter with `check_user_quota()` DB dependency
4. Integrate Stripe webhooks to update `plan` on subscription events
5. Return `402 Payment Required` with upgrade prompt when quota exceeded
6. Surface quota usage in the frontend dashboard

### Planned — Other
- [ ] Audit history dashboard (`/dashboard`)
- [ ] Stripe billing integration
- [ ] White-label PDF (custom logo/colours)
- [ ] Webhook on audit complete (agency integrations)
