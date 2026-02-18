# Local SEO Audit Tool — Complete Build Guide

## What You're Building

A web app where users enter a keyword and URL, and 3 AI agents analyse their SEO and return specific, actionable recommendations. Users pay $49–$499/month. Your cost per audit is ~$0.12.

**The 3 Agents:**

| Agent | What It Does |
|-------|-------------|
| Keyword Research | Finds competitor keywords you're missing, maps search intent, builds keyword clusters |
| On-Page SEO | Audits your page vs. competitors — title tags, content gaps, heading structure, internal links |
| Local SEO | Google Business Profile strategy, citation building, link opportunities, local content plan |

**Your Timeline:**

| Week | Goal | Hours |
|------|------|-------|
| 1 | Backend API running + all 3 agents tested | 8–10 |
| 2 | Frontend built + deployed to live URLs | 10–12 |
| 3 | Database + 5–10 beta users testing | 8–10 |
| 4 | Payments + launch + first customers | 8–10 |

---

## Your Files

```
seo-saas/
├── main.py              ← Your backend (FastAPI + 3 agents)
├── requirements.txt     ← Python dependencies
├── .env.example         ← Template for API keys
├── .gitignore           ← Keeps secrets out of git
├── Procfile             ← For Railway/Heroku deployment
├── BUILD_GUIDE.md       ← Detailed week-by-week instructions
└── README.md            ← This file
```

---

## Quick Start (15 minutes)

### 1. Get API Keys

**Anthropic Claude** — the AI brain:
1. Go to https://console.anthropic.com
2. Sign up → Settings → API Keys → Create Key
3. Copy the key (starts with `sk-ant-`)

**SerpApi** — for Google search results:
1. Go to https://serpapi.com
2. Sign up → Account → copy API Key
3. Free tier gives you 100 searches/month

### 2. Set Up Project

```bash
# Create and enter project
mkdir seo-saas && cd seo-saas

# Copy all project files into this folder, then:

# Create virtual environment
python3 -m venv venv
source venv/bin/activate          # Mac/Linux
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Open .env in your editor and paste your real API keys
```

### 3. Run It

```bash
python main.py
```

You'll see:
```
INFO  Uvicorn running on http://0.0.0.0:8000
```

### 4. Test It

Open a second terminal:

```bash
# Health check
curl http://localhost:8000/health

# Run a full audit (takes 60-90 seconds)
curl -X POST http://localhost:8000/workflow/seo-audit \
  -H "Content-Type: application/json" \
  -d '{
    "keyword": "best pizza near me",
    "target_url": "https://example.com",
    "location": "Toronto, Canada"
  }'
```

If you get JSON back with keyword recommendations, on-page analysis, and local SEO strategy — you're live.

---

## What's Next

Open **BUILD_GUIDE.md** for the detailed day-by-day walkthrough covering:
- Week 1: Testing each agent individually, debugging, understanding responses
- Week 2: Deploying backend to Railway, building the Next.js frontend, deploying to Vercel
- Week 3: Adding PostgreSQL, getting beta users, refining prompts
- Week 4: Stripe payments, pricing page, auth, launch

---

## Business Model

| Tier | Price | You Keep | What's Included |
|------|-------|----------|----------------|
| Basic | $49/mo | ~$48.80 | 5 audits/month, 3 agents, email reports |
| Pro | $149/mo | ~$146 | Unlimited audits, priority processing, export |
| Agency | $499/mo | ~$492 | Multi-client, white-label, API access |

**100 customers at typical mix = ~$20,000/month revenue, ~$250/month in API costs.**
