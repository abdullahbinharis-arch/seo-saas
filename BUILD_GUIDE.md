# Build Guide — Week by Week

This is your day-by-day playbook. Follow it in order.

---

# WEEK 1: Backend + Agent Testing

**Goal:** All 3 agents returning quality JSON from your local machine.

---

## Monday — Setup (1–2 hours)

### Get API Keys

**Anthropic Claude:**
1. https://console.anthropic.com → sign up
2. Settings → API Keys → Create Key
3. Copy it (starts with `sk-ant-`)

**SerpApi:**
1. https://serpapi.com → sign up (free tier = 100 searches/month)
2. Account → copy your API Key

### Set Up the Project

```bash
mkdir seo-saas && cd seo-saas
# Copy all project files into this folder

python3 -m venv venv
source venv/bin/activate       # Mac/Linux
# venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

### Configure Environment

```bash
cp .env.example .env
```

Open `.env` in any editor and replace the placeholder values with your real keys:

```
ANTHROPIC_API_KEY=sk-ant-abc123...your-real-key
SERPAPI_KEY=your-real-serpapi-key
```

### Start the Server

```bash
python main.py
```

Expected output:
```
INFO  Uvicorn running on http://0.0.0.0:8000
```

### Verify It Works

In a **second terminal** (keep the server running in the first):

```bash
curl http://localhost:8000/health
```

Expected:
```json
{"status":"ok","timestamp":"...","api_key_set":true,"serpapi_key_set":true}
```

Both `api_key_set` and `serpapi_key_set` should be `true`. If not, check your `.env` file.

**Monday done.** ✓

---

## Tuesday — Test Keyword Research Agent (30 minutes)

With the server still running, open your second terminal:

```bash
curl -X POST http://localhost:8000/agents/keyword-research \
  -H "Content-Type: application/json" \
  -d '{
    "keyword": "best pizza near me",
    "target_url": "https://example.com/pizza",
    "location": "Toronto, Canada"
  }'
```

Wait 20–40 seconds. You'll get JSON back.

### What to Check

- `"status": "completed"` — the agent finished
- `recommendations.high_intent_keywords` — should have 10+ keywords
- `recommendations.long_tail_keywords` — should have 5+ phrases
- `recommendations.keyword_clusters` — grouped keyword themes
- `recommendations.recommendation` — a plain-English strategy summary

### Try a Different Keyword

```bash
curl -X POST http://localhost:8000/agents/keyword-research \
  -H "Content-Type: application/json" \
  -d '{
    "keyword": "plumber near me",
    "target_url": "https://example.com",
    "location": "Vancouver, Canada"
  }'
```

**Tuesday done.** ✓

---

## Wednesday — Test On-Page SEO Agent (30 minutes)

```bash
curl -X POST http://localhost:8000/agents/on-page-seo \
  -H "Content-Type: application/json" \
  -d '{
    "keyword": "best pizza near me",
    "target_url": "https://example.com/pizza",
    "location": "Toronto, Canada"
  }'
```

Wait 30–40 seconds.

### What to Check

- `recommendations.current_analysis` — shows what your page has now
- `recommendations.recommendations.meta_title` — specific new title (not a template)
- `recommendations.recommendations.meta_description` — specific new description
- `recommendations.recommendations.target_word_count` — how long your content should be
- `recommendations.internal_links` — specific pages to link to
- `recommendations.priority_actions` — ranked list of what to fix first

**Wednesday done.** ✓

---

## Thursday — Test Local SEO Agent (30 minutes)

```bash
curl -X POST http://localhost:8000/agents/local-seo \
  -H "Content-Type: application/json" \
  -d '{
    "keyword": "best pizza near me",
    "target_url": "https://example.com/pizza",
    "location": "Toronto, Canada"
  }'
```

Wait 30–40 seconds.

### What to Check

- `recommendations.gbp_optimization` — Google Business Profile action items
- `recommendations.citations` — 8+ directory sites to list on
- `recommendations.link_opportunities` — 5+ sites to get links from (with outreach templates)
- `recommendations.local_content_strategy.blog_topics` — content ideas
- `recommendations.quick_wins` — fastest impact actions

**Thursday done.** ✓

---

## Friday — Full Workflow Test (1 hour)

This runs all 3 agents in one call:

```bash
curl -X POST http://localhost:8000/workflow/seo-audit \
  -H "Content-Type: application/json" \
  -d '{
    "keyword": "best pizza near me",
    "target_url": "https://example.com/pizza",
    "location": "Toronto, Canada"
  }'
```

Wait 60–90 seconds (keyword research runs first, then on-page + local run concurrently).

### What to Check

- `agents_executed: 3`
- `agents.keyword_research` — full keyword data
- `agents.on_page_seo` — full on-page audit
- `agents.local_seo` — full local strategy
- `summary.quick_wins` — real recommendations pulled from each agent (not hardcoded)
- `execution_time_seconds` — should be 60–90 seconds

### Test with 2–3 More Keywords

Try real businesses you know:

```bash
# A dentist
curl -X POST http://localhost:8000/workflow/seo-audit \
  -H "Content-Type: application/json" \
  -d '{"keyword": "dentist near me", "target_url": "https://example.com", "location": "Calgary, Canada"}'

# A gym
curl -X POST http://localhost:8000/workflow/seo-audit \
  -H "Content-Type: application/json" \
  -d '{"keyword": "gym near me", "target_url": "https://example.com", "location": "Toronto, Canada"}'
```

### Week 1 Checklist

- [ ] Server runs without errors
- [ ] Health endpoint returns `api_key_set: true`
- [ ] Keyword Research Agent returns 10+ keywords
- [ ] On-Page SEO Agent returns specific meta title/description
- [ ] Local SEO Agent returns citations + link opportunities
- [ ] Full workflow runs all 3 agents in <90 seconds
- [ ] Tested with at least 3 different keywords
- [ ] Cost per audit < $0.15

**Week 1 complete!** You have a working multi-agent SEO analysis API. 🎉

---

# WEEK 2: Deploy + Build Frontend

**Goal:** Live web app anyone can visit and use.

---

## Monday — Deploy Backend to Railway (30 minutes)

Railway hosts your API for free (hobby tier).

### Push Code to GitHub

If you don't have a GitHub account, create one at https://github.com (free, 2 minutes).

```bash
cd seo-saas
git init
git add .
git commit -m "Initial commit — 3-agent SEO backend"
```

Go to GitHub → New Repository → name it `seo-saas` → Create.

```bash
git remote add origin https://github.com/YOUR_USERNAME/seo-saas.git
git branch -M main
git push -u origin main
```

### Deploy to Railway

1. Go to https://railway.app → Start Now → sign in with GitHub
2. New Project → Deploy from GitHub Repo → select `seo-saas`
3. Click Variables (left sidebar) → add:
   - `ANTHROPIC_API_KEY` = your key
   - `SERPAPI_KEY` = your key
4. Click Settings → Networking → Generate Domain
5. Copy your URL (looks like: `https://seo-saas-production.up.railway.app`)

### Test It

```bash
curl https://YOUR_RAILWAY_URL/health
```

Should return `{"status": "ok", ...}`. If so, your backend is live on the internet.

**Save this URL — your frontend will call it.**

**Monday done.** ✓

---

## Tuesday — Create Next.js Frontend (1 hour)

In a **separate folder** (not inside seo-saas):

```bash
npx create-next-app@latest seo-frontend --typescript --tailwind --app --src-dir
```

Accept all defaults. Then:

```bash
cd seo-frontend
```

Create the environment file:

```bash
echo "NEXT_PUBLIC_API_URL=https://YOUR_RAILWAY_URL" > .env.local
```

Replace `YOUR_RAILWAY_URL` with your actual Railway URL.

Create the types folder:

```bash
mkdir -p src/types
```

Create `src/types/index.ts`:

```typescript
export interface AuditRequest {
  keyword: string;
  target_url: string;
  location?: string;
}

export interface AuditResult {
  audit_id: string;
  keyword: string;
  target_url: string;
  location: string;
  status: string;
  execution_time_seconds: number;
  agents: {
    keyword_research: AgentResult;
    on_page_seo: AgentResult;
    local_seo: AgentResult;
  };
  summary: {
    quick_wins: string[];
    estimated_api_cost: number;
  };
}

export interface AgentResult {
  agent: string;
  audit_id: string;
  status: string;
  recommendations: Record<string, any>;
}
```

**Tuesday done.** ✓

---

## Wednesday — Build the UI (2 hours)

Replace the contents of `src/app/page.tsx` with:

```tsx
import AuditTool from "./AuditTool";

export default function Home() {
  return <AuditTool />;
}
```

Create `src/app/AuditTool.tsx` — this is your entire frontend in one file:

```tsx
"use client";

import { useState } from "react";
import type { AuditResult } from "@/types";

// ── Main Page ──────────────────────────────────────────────────────

export default function AuditTool() {
  const [keyword, setKeyword] = useState("");
  const [url, setUrl] = useState("");
  const [location, setLocation] = useState("Toronto, Canada");
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState("");
  const [error, setError] = useState("");
  const [results, setResults] = useState<AuditResult | null>(null);

  async function runAudit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResults(null);

    // Progress stages so users know it's working
    setStage("Finding competitors on Google...");
    const t1 = setTimeout(() => setStage("Analysing keywords and content gaps..."), 15000);
    const t2 = setTimeout(() => setStage("Auditing on-page SEO factors..."), 35000);
    const t3 = setTimeout(() => setStage("Building local SEO strategy..."), 55000);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const res = await fetch(`${apiUrl}/workflow/seo-audit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ keyword, target_url: url, location }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => null);
        throw new Error(errData?.detail || `API error ${res.status}`);
      }

      setResults(await res.json());
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
      setLoading(false);
      setStage("");
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b">
        <div className="max-w-5xl mx-auto px-6 py-8">
          <h1 className="text-3xl font-bold text-gray-900">Local SEO Audit</h1>
          <p className="text-gray-500 mt-1">
            3 AI agents analyse your keywords, content, and local presence
          </p>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-10">
        {/* Input Form */}
        <form onSubmit={runAudit} className="bg-white rounded-xl shadow-sm border p-8 mb-10">
          <div className="grid md:grid-cols-3 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Target Keyword
              </label>
              <input
                type="text"
                value={keyword}
                onChange={(e) => setKeyword(e.target.value)}
                placeholder="e.g. best pizza near me"
                className="w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Your Website URL
              </label>
              <input
                type="url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="https://example.com"
                className="w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Location
              </label>
              <input
                type="text"
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                className="w-full px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="mt-6 w-full md:w-auto bg-blue-600 text-white px-8 py-3 rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
          >
            {loading ? "Running Audit..." : "Run SEO Audit"}
          </button>

          {/* Loading indicator */}
          {loading && (
            <div className="mt-4 flex items-center gap-3 text-blue-600">
              <div className="animate-spin h-5 w-5 border-2 border-blue-600 border-t-transparent rounded-full" />
              <span>{stage || "Starting audit..."}</span>
            </div>
          )}

          {/* Error */}
          {error && (
            <div className="mt-4 bg-red-50 border border-red-200 text-red-700 rounded-lg p-4">
              {error}
            </div>
          )}
        </form>

        {/* Results */}
        {results && <Results data={results} />}
      </main>
    </div>
  );
}


// ── Results Display ────────────────────────────────────────────────

function Results({ data }: { data: AuditResult }) {
  const kw = data.agents?.keyword_research?.recommendations ?? {};
  const op = data.agents?.on_page_seo?.recommendations ?? {};
  const local = data.agents?.local_seo?.recommendations ?? {};

  return (
    <div className="space-y-8">
      {/* Success header */}
      <div className="bg-green-50 border border-green-200 rounded-xl p-6 flex items-start gap-4">
        <span className="text-2xl">✓</span>
        <div>
          <p className="font-semibold text-green-900">Audit complete</p>
          <p className="text-sm text-green-700">
            Analysed in {data.execution_time_seconds}s
            {" · "}Cost: ${data.summary?.estimated_api_cost?.toFixed(2) ?? "0.12"}
            {" · "}ID: {data.audit_id?.slice(0, 8)}
          </p>
        </div>
      </div>

      {/* Quick Wins */}
      {data.summary?.quick_wins?.length > 0 && (
        <Section title="⚡ Quick Wins" color="blue">
          <ul className="space-y-2">
            {data.summary.quick_wins.map((win, i) => (
              <li key={i} className="flex gap-2 text-gray-700">
                <span className="text-blue-500 font-bold">→</span>
                <span>{win}</span>
              </li>
            ))}
          </ul>
        </Section>
      )}

      {/* Keyword Research */}
      <Section title="🔍 Keyword Research" color="white">
        {kw.high_intent_keywords?.length > 0 && (
          <div className="mb-6">
            <h4 className="font-medium text-gray-700 mb-3">High-Intent Keywords</h4>
            <div className="grid md:grid-cols-2 gap-2">
              {kw.high_intent_keywords.slice(0, 8).map((k: any, i: number) => (
                <div key={i} className="bg-gray-50 p-3 rounded-lg border">
                  <p className="font-medium text-gray-900">{k.keyword}</p>
                  <p className="text-sm text-gray-500">
                    {k.intent} · ~{k.estimated_monthly_searches ?? k.searches ?? "?"} searches/mo
                    {k.difficulty && ` · ${k.difficulty} difficulty`}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}

        {kw.long_tail_keywords?.length > 0 && (
          <div className="mb-6">
            <h4 className="font-medium text-gray-700 mb-2">Long-Tail Opportunities</h4>
            <div className="flex flex-wrap gap-2">
              {kw.long_tail_keywords.slice(0, 10).map((lt: string, i: number) => (
                <span key={i} className="bg-blue-50 text-blue-700 px-3 py-1 rounded-full text-sm">
                  {lt}
                </span>
              ))}
            </div>
          </div>
        )}

        {kw.keyword_clusters?.length > 0 && (
          <div>
            <h4 className="font-medium text-gray-700 mb-2">Keyword Clusters</h4>
            <div className="grid md:grid-cols-2 gap-2">
              {kw.keyword_clusters.slice(0, 6).map((c: any, i: number) => (
                <div key={i} className="bg-gray-50 p-3 rounded-lg border">
                  <p className="font-medium text-gray-800">{c.theme ?? c}</p>
                  {c.keywords && (
                    <p className="text-sm text-gray-500 mt-1">{c.keywords.join(", ")}</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {kw.recommendation && (
          <p className="mt-4 text-gray-600 bg-blue-50 p-4 rounded-lg">{kw.recommendation}</p>
        )}
      </Section>

      {/* On-Page SEO */}
      <Section title="📄 On-Page SEO" color="white">
        {op.recommendations && (
          <div className="space-y-4">
            <MetaField label="Recommended Title" value={op.recommendations.meta_title} />
            <MetaField label="Recommended Description" value={op.recommendations.meta_description} />
            <MetaField label="Recommended H1" value={op.recommendations.h1} />

            {op.recommendations.target_word_count && (
              <div>
                <h4 className="font-medium text-gray-700">Target Word Count</h4>
                <p className="text-gray-900 text-lg font-semibold">
                  {op.recommendations.target_word_count} words
                </p>
              </div>
            )}
          </div>
        )}

        {op.priority_actions?.length > 0 && (
          <div className="mt-6">
            <h4 className="font-medium text-gray-700 mb-2">Priority Actions</h4>
            <ol className="space-y-2">
              {op.priority_actions.map((action: string, i: number) => (
                <li key={i} className="flex gap-3 text-gray-700">
                  <span className="bg-orange-100 text-orange-700 rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold shrink-0">
                    {i + 1}
                  </span>
                  <span>{action}</span>
                </li>
              ))}
            </ol>
          </div>
        )}

        {op.internal_links?.length > 0 && (
          <div className="mt-6">
            <h4 className="font-medium text-gray-700 mb-2">Internal Links to Add</h4>
            {op.internal_links.slice(0, 5).map((link: any, i: number) => (
              <div key={i} className="bg-gray-50 p-3 rounded-lg border mb-2">
                <p className="font-medium text-gray-800">"{link.anchor_text}" → {link.target_path}</p>
                <p className="text-sm text-gray-500">{link.reason}</p>
              </div>
            ))}
          </div>
        )}
      </Section>

      {/* Local SEO */}
      <Section title="📍 Local SEO" color="white">
        {local.gbp_optimization && (
          <div className="mb-6">
            <h4 className="font-medium text-gray-700 mb-2">Google Business Profile</h4>
            {local.gbp_optimization.priority_attributes?.map((attr: string, i: number) => (
              <p key={i} className="text-gray-700 ml-2">• {attr}</p>
            ))}
            {local.gbp_optimization.photo_strategy && (
              <p className="text-sm text-gray-500 mt-2 bg-gray-50 p-3 rounded">
                📷 {local.gbp_optimization.photo_strategy}
              </p>
            )}
          </div>
        )}

        {local.citations?.length > 0 && (
          <div className="mb-6">
            <h4 className="font-medium text-gray-700 mb-2">Citations to Build</h4>
            <div className="grid md:grid-cols-2 gap-2">
              {local.citations.slice(0, 8).map((c: any, i: number) => (
                <div key={i} className="bg-gray-50 p-3 rounded-lg border flex justify-between items-center">
                  <span className="font-medium text-gray-800">{c.site}</span>
                  <span className={`text-xs font-medium px-2 py-1 rounded ${
                    c.priority === "critical" ? "bg-red-100 text-red-700" :
                    c.priority === "high" ? "bg-orange-100 text-orange-700" :
                    "bg-gray-100 text-gray-600"
                  }`}>
                    {c.priority}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {local.link_opportunities?.length > 0 && (
          <div>
            <h4 className="font-medium text-gray-700 mb-2">Link Opportunities</h4>
            {local.link_opportunities.slice(0, 5).map((opp: any, i: number) => (
              <div key={i} className="bg-gray-50 p-4 rounded-lg border mb-2">
                <p className="font-medium text-gray-800">{opp.name}</p>
                <p className="text-sm text-gray-500">{opp.reason}</p>
                {opp.outreach_template && (
                  <p className="text-sm text-blue-600 mt-2 italic">"{opp.outreach_template}"</p>
                )}
              </div>
            ))}
          </div>
        )}
      </Section>

      {/* Raw JSON toggle */}
      <details className="bg-gray-50 border rounded-xl p-6">
        <summary className="font-medium text-gray-700 cursor-pointer">View raw JSON</summary>
        <pre className="mt-4 bg-gray-900 text-gray-100 p-4 rounded-lg overflow-auto text-xs max-h-96">
          {JSON.stringify(data, null, 2)}
        </pre>
      </details>
    </div>
  );
}


// ── Helper Components ──────────────────────────────────────────────

function Section({
  title,
  color,
  children,
}: {
  title: string;
  color: "blue" | "white";
  children: React.ReactNode;
}) {
  const bg = color === "blue" ? "bg-blue-50 border-blue-200" : "bg-white border-gray-200";
  return (
    <div className={`${bg} border rounded-xl p-6`}>
      <h3 className="text-lg font-semibold text-gray-900 mb-4">{title}</h3>
      {children}
    </div>
  );
}

function MetaField({ label, value }: { label: string; value?: string }) {
  if (!value) return null;
  return (
    <div>
      <h4 className="font-medium text-gray-700 text-sm">{label}</h4>
      <div className="bg-gray-50 p-3 rounded-lg border-l-4 border-blue-500 mt-1">
        <p className="text-gray-900 font-mono text-sm">{value}</p>
      </div>
    </div>
  );
}
```

Run it:

```bash
npm run dev
```

Open http://localhost:3000. You should see the form. Try an audit — it should take 60–90 seconds and display all three agent results.

**Wednesday done.** ✓

---

## Thursday — Test End-to-End (1 hour)

1. Make sure your Railway backend is running (check the dashboard)
2. Make sure `.env.local` has your Railway URL (not localhost)
3. Run `npm run dev`
4. Open http://localhost:3000
5. Enter a keyword, URL, and location
6. Click "Run SEO Audit"
7. Wait 60–90 seconds — you should see progress messages updating

### Troubleshooting

**"Failed to fetch" error:**
- Check Railway dashboard — is deployment "Active"?
- Check `.env.local` has `https://` (not `http://`)
- Check for typos in the URL

**CORS error in browser console:**
- In your Railway Variables, add: `ALLOWED_ORIGINS=http://localhost:3000`
- Redeploy

**Results display but some sections are empty:**
- This is normal if Claude's JSON didn't match the expected shape for some fields
- Check the "View raw JSON" section at the bottom to see what came back
- The frontend handles missing data gracefully with optional chaining

**Thursday done.** ✓

---

## Friday — Deploy Frontend to Vercel (1 hour)

### Build Check

```bash
cd seo-frontend
npm run build
```

Should say "✓ Compiled successfully". If there are errors, fix them before deploying.

### Deploy

1. Push frontend to GitHub:
   ```bash
   cd seo-frontend
   git init
   git add .
   git commit -m "SEO audit frontend"
   git remote add origin https://github.com/YOUR_USERNAME/seo-frontend.git
   git push -u origin main
   ```

2. Go to https://vercel.com → sign in with GitHub
3. Import Project → select `seo-frontend`
4. Add environment variable: `NEXT_PUBLIC_API_URL` = your Railway URL
5. Click Deploy (takes 2–3 minutes)
6. Copy your live URL (e.g., `https://seo-frontend.vercel.app`)

### Update CORS on Backend

In Railway Variables, update:
```
ALLOWED_ORIGINS=http://localhost:3000,https://seo-frontend.vercel.app
```

Redeploy Railway.

### Test Live

Open your Vercel URL → run an audit → confirm results display.

**You now have a live product.** Share the link with anyone.

**Week 2 complete!** 🎉

---

# WEEK 3: Database + Beta Testing

**Goal:** Save audit results, get 5–10 real users testing it.

---

## Monday–Tuesday — Add PostgreSQL

### Option A: Railway PostgreSQL (Easiest)

In Railway dashboard → New → Database → PostgreSQL. Copy the `DATABASE_URL`.

Add to Railway variables:
```
DATABASE_URL=postgresql://...your-railway-postgres-url
```

### Add Database Code

Uncomment `sqlalchemy` and `psycopg2-binary` in `requirements.txt`, then:

Create `database.py` in your `seo-saas` folder:

```python
from sqlalchemy import create_engine, Column, String, JSON, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./seo_audits.db")

# Fix Railway's postgres:// URL (SQLAlchemy needs postgresql://)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Audit(Base):
    __tablename__ = "audits"
    id = Column(String, primary_key=True)
    keyword = Column(String, nullable=False)
    target_url = Column(String, nullable=False)
    location = Column(String)
    status = Column(String, default="completed")
    results = Column(JSON)
    api_cost = Column(Float)
    execution_time = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables
Base.metadata.create_all(engine)
```

Then add saving to `main.py` after the audit completes:

```python
# At the end of seo_audit_workflow, before return:
from database import SessionLocal, Audit

db = SessionLocal()
try:
    audit = Audit(
        id=audit_id,
        keyword=request.keyword,
        target_url=request.target_url,
        location=request.location,
        results=combined_report,
        api_cost=calculate_cost_estimate(),
        execution_time=elapsed,
    )
    db.add(audit)
    db.commit()
except Exception as e:
    logger.error(f"DB save failed: {e}")
finally:
    db.close()
```

Add a GET endpoint to retrieve past audits:

```python
@app.get("/audits")
async def list_audits(limit: int = 20):
    from database import SessionLocal, Audit
    db = SessionLocal()
    audits = db.query(Audit).order_by(Audit.created_at.desc()).limit(limit).all()
    db.close()
    return [{"id": a.id, "keyword": a.keyword, "target_url": a.target_url, "created_at": str(a.created_at)} for a in audits]

@app.get("/audits/{audit_id}")
async def get_audit(audit_id: str):
    from database import SessionLocal, Audit
    db = SessionLocal()
    audit = db.query(Audit).filter(Audit.id == audit_id).first()
    db.close()
    if not audit:
        raise HTTPException(404, "Audit not found")
    return audit.results
```

### Test

```bash
# Run an audit
curl -X POST http://localhost:8000/workflow/seo-audit ...

# Check it was saved
curl http://localhost:8000/audits
```

---

## Wednesday–Thursday — Get Beta Users

### Who to Ask

1. Friends who own local businesses
2. Anyone you know doing SEO or marketing
3. People in online communities (Reddit r/SEO, r/smallbusiness, Indie Hackers)

### What to Say

> "I built an AI-powered SEO audit tool. Enter your website + a keyword and it gives you specific recommendations for ranking higher on Google. Can you test it and tell me what you think? Takes 2 minutes."

Send them your Vercel URL.

### What to Ask Them

1. Did the results load? How long did it wait?
2. Were the keyword suggestions relevant?
3. Were the recommendations actionable — could you actually do them?
4. What's confusing or unclear?
5. Would you pay for this? How much?

---

## Friday — Refine Based on Feedback

Common feedback and fixes:

**"The results are too generic"** → Improve agent prompts. Add more specific instructions about the industry/location.

**"It's slow"** → Already optimised (concurrent agents), but consider caching competitor data for repeat keywords.

**"I don't understand the recommendations"** → Add plain-English explanations in the frontend. Group by difficulty (easy/medium/hard).

**Week 3 complete!** ✓

---

# WEEK 4: Payments + Launch

**Goal:** First paying customer.

---

## Monday–Tuesday — Add Stripe

### Set Up Stripe

1. https://stripe.com → sign up
2. Developers → API Keys → copy Publishable Key + Secret Key
3. Create 3 Products in Stripe Dashboard:
   - Basic: $49/month
   - Pro: $149/month
   - Agency: $499/month

### Simplest Payment Integration

For your MVP, don't build a full subscription system. Use **Stripe Payment Links**:

1. In Stripe Dashboard → Payment Links → create one for each tier
2. Add the links to a pricing page on your frontend
3. After someone pays, manually give them access (or use Stripe webhooks later)

This gets you from $0 to revenue in an hour, without building auth/subscription logic.

### Create a Pricing Page

Add a `/pricing` route to your frontend with 3 cards linking to Stripe Payment Links. You can build the full auth + subscription system in Month 2.

---

## Wednesday — Marketing Prep

### Launch Checklist

- [ ] Landing page with clear value proposition
- [ ] 3 pricing tiers with Stripe links
- [ ] At least 2 testimonials/screenshots from beta users
- [ ] Your own audit results as demo content

### Where to Launch

1. **Product Hunt** — submit your tool
2. **Reddit** — r/SEO, r/smallbusiness, r/SaaS, r/Entrepreneur
3. **Twitter/X** — share screenshots of audit results
4. **LinkedIn** — post about building it
5. **Indie Hackers** — share your story
6. **Facebook Groups** — local business groups

---

## Thursday–Friday — Launch

Post everywhere. Respond to every comment. Get your first paying customer.

**Week 4 complete. You have a live, revenue-generating SaaS.** 🚀

---

# Troubleshooting Reference

### "ANTHROPIC_API_KEY not found"

Your `.env` file is missing or the key isn't set:
```bash
cat .env | grep ANTHROPIC
```
If empty, add your key. Restart the server.

### "ModuleNotFoundError"

Virtual environment isn't activated:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### "Connection refused" on localhost:8000

Server isn't running. Start it:
```bash
python main.py
```

### Agent returns `raw_response` instead of structured JSON

Claude didn't return valid JSON. This happens occasionally. The system will use fallback data. If it happens consistently, check the server logs for the raw response.

### CORS error in browser

Add your frontend URL to `ALLOWED_ORIGINS` in Railway variables:
```
ALLOWED_ORIGINS=http://localhost:3000,https://your-app.vercel.app
```

### Audit takes more than 2 minutes

Normal for the first request (cold start). If it's consistently slow, check your internet connection and Railway logs.

---

# Architecture Overview

```
User Browser → Next.js Frontend (Vercel)
                    ↓ POST /workflow/seo-audit
              FastAPI Backend (Railway)
                    ↓
        ┌───────────┼───────────┐
        ↓           ↓           ↓
   Keyword      On-Page      Local SEO
   Research       SEO         Agent
   Agent         Agent           ↓
     ↓             ↓        Claude API
  SerpApi      Scraper      (Anthropic)
     ↓             ↓
  Claude API   Claude API
     ↓             ↓
     └─────────────┼───────────┘
                   ↓
            Combined Report
                   ↓
              User sees results
```

**How it works:**
1. User submits keyword + URL
2. Backend fetches top 5 Google results via SerpApi
3. Backend scrapes competitor pages
4. Keyword Research Agent runs first (other agents benefit from this data)
5. On-Page + Local SEO agents run concurrently (saves ~30 seconds)
6. All results combined into one report with quick wins
7. Frontend displays everything in a clean UI

**Cost per audit:** ~$0.10–0.15 (Claude API + SerpApi)
**Revenue per customer:** $49–499/month
**Margin:** 97%+
