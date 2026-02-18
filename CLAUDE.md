# CLAUDE.md — Project Instructions for Claude Code

## Project Overview

This is a Local SEO SaaS platform — a FastAPI backend with 3 AI agents (Keyword Research, On-Page SEO, Local SEO) that analyse websites and return actionable SEO recommendations. The frontend is Next.js + Tailwind.

## Tech Stack

- **Backend:** Python 3.10+, FastAPI, AsyncAnthropic, httpx, BeautifulSoup4
- **Frontend:** Next.js 14+ (App Router), TypeScript, Tailwind CSS
- **AI:** Claude API via `anthropic` SDK (async client only)
- **Data:** SerpApi for Google results, httpx for scraping
- **Deployment:** Railway (backend), Vercel (frontend)

## Architecture

```
POST /workflow/seo-audit
  → keyword_research_agent (runs first)
  → on_page_seo_agent + local_seo_agent (run concurrently via asyncio.gather)
  → combined report with dynamic quick_wins
```

## Key Rules

1. **Always use AsyncAnthropic** — never the sync client. All Claude calls must be `await`ed.
2. **All Claude calls go through `call_claude()`** — the centralised helper in main.py. Never call `anthropic_client.messages.create()` directly from endpoints.
3. **JSON extraction uses `extract_json()`** — never raw regex. This handles markdown fences, preamble text, and unbalanced braces.
4. **System prompts are separate constants** — defined at module level (e.g., `KEYWORD_SYSTEM`), not inline in functions.
5. **Prompts use assistant prefill** — `{"role": "assistant", "content": "{"}` to force JSON output.
6. **Environment variables load via python-dotenv** — `load_dotenv()` is called at startup.
7. **Error messages to users are generic** — never expose stack traces, API keys, or internal details in HTTP responses.
8. **Rate limiting is enabled** — in-memory, configurable via `RATE_LIMIT_PER_MIN`.
9. **Scraping uses BeautifulSoup** — not stdlib html.parser. Extract title, meta desc, H1, headings, word count, and links.
10. **CORS origins are configurable** — via `ALLOWED_ORIGINS` env var, not hardcoded `["*"]`.

## File Structure

```
seo-saas/
├── main.py              ← FastAPI backend, all 3 agents + orchestrator
├── requirements.txt     ← Python deps
├── .env.example         ← Template for API keys
├── .env                 ← Actual keys (never commit)
├── .gitignore
├── Procfile             ← Railway/Heroku deployment
├── CLAUDE.md            ← This file
├── README.md            ← Quick start
├── BUILD_GUIDE.md       ← Full 4-week build instructions
└── database.py          ← (Week 3) SQLAlchemy models
```

## Common Tasks

### Add a new agent
1. Define `NEWAGENT_SYSTEM` and `NEWAGENT_PROMPT` constants
2. Create endpoint function following the pattern of existing agents
3. Call `call_claude(NEWAGENT_SYSTEM, prompt, max_tokens=N)`
4. Add to the orchestrator's `seo_audit_workflow`
5. Update `build_quick_wins()` to pull from new agent's output

### Update a prompt
- Edit the module-level constant (e.g., `KEYWORD_PROMPT`)
- Test with: `curl -X POST http://localhost:8000/agents/keyword-research -H "Content-Type: application/json" -d '{"keyword": "test", "target_url": "https://example.com"}'`

### Run locally
```bash
source venv/bin/activate
python main.py
# Test: curl http://localhost:8000/health
```

## Testing

Test each agent individually before running the full workflow:
```bash
curl -X POST http://localhost:8000/agents/keyword-research -H "Content-Type: application/json" -d '{"keyword": "pizza near me", "target_url": "https://example.com", "location": "Toronto, Canada"}'
```

## Dependencies

Keep `anthropic>=0.39.0` — older versions don't support AsyncAnthropic or current model names.
