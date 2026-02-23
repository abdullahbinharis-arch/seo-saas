# =============================================================================
# SEO Content Engine — 15-Rule AI Content Generation + Scoring
# =============================================================================
#
# Standalone module for SEO-optimized content generation.
# - analyze_competitors(): scrape top SERP pages, calculate targets
# - generate_seo_content(): full 15-rule content generation with auto-fix
# - calculate_seo_score(): pure Python scorer against 15 rules
# - Content type helpers: homepage, service page, area page
#
# Dependencies are injected (claude_caller, fetch_competitors_fn, scrape_page_fn)
# to avoid circular imports with main.py.
# =============================================================================

import asyncio
import logging
import re
import time
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger("seo-engine")

# ---------------------------------------------------------------------------
# Constants — Transition words, CTA phrases, E-E-A-T signals
# ---------------------------------------------------------------------------

TRANSITION_WORDS = [
    "however", "therefore", "additionally", "meanwhile", "as a result",
    "in fact", "for example", "because", "furthermore", "moreover",
    "consequently", "specifically", "on the other hand", "in particular",
    "notably", "similarly", "first", "second", "finally", "next",
    "also", "besides", "indeed", "certainly", "of course",
    "in addition", "above all", "in contrast", "nevertheless",
]

CTA_PHRASES = [
    "contact us", "get a quote", "call us", "book now", "schedule",
    "free estimate", "get started", "request", "reach out", "speak with",
    "book a", "call today", "get in touch", "learn more", "find out",
]

EEAT_SIGNALS = [
    "years", "experience", "certified", "licensed", "projects",
    "testimonial", "guarantee", "warranty", "rated", "trusted",
    "qualified", "professional", "expert", "award", "accredited",
]

# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------

SEO_CONTENT_SYSTEM_PROMPT = """You are an expert SEO content writer for local businesses. You write content that ranks #1 on Google by following these exact rules. NEVER deviate from these rules.

CONTEXT:
- Business: {business_name}
- Category: {business_category}
- Location: {city}, {country}
- Services: {services}
- Target Keyword: {primary_keyword}
- Secondary Keywords: {secondary_keywords}
- Competitor Average Word Count: {competitor_avg_words}
- Target Word Count: {target_words} (20-35% longer than competitors)

THE 15 RULES YOU MUST FOLLOW:

RULE 1 — CONTENT LENGTH
Write exactly {target_words} words (20-35% longer than the competitor average of {competitor_avg_words}).
Be MORE COMPLETE than competitors, not more bloated. Every sentence must add value.

RULE 2 — KEYWORD PLACEMENT (MANDATORY POSITIONS)
The primary keyword "{primary_keyword}" MUST appear in ALL of these:
- H1 tag (exact match)
- First 100 words of the page
- Meta title
- Meta description
- URL slug suggestion
- At least one H2 heading
- At least one image alt text suggestion
- Last 100 words of the page
If you miss ANY of these positions, the content fails.

RULE 3 — KEYWORD FREQUENCY
- Primary keyword: 6-10 times per 1000 words (naturally placed)
- Secondary keywords: 3-5 times each
- Semantic variations: use throughout (NOT exact repetitions)
Generate semantic variations like:
- "{primary_keyword}" -> also use: {semantic_variations}
Google ranks CONTEXT, not repetition. Vary your phrasing.

RULE 4 — HEADING STRUCTURE
Use this exact hierarchy:
- 1x H1 (contains primary keyword + location)
- 4-6x H2 (subtopics, at least one as a question)
- 6-12x H3 (supporting details under H2s)
Include: question headings, benefit headings, problem-solving headings.

RULE 5 — FIRST PARAGRAPH
The opening paragraph MUST contain:
- Primary keyword (in first sentence if possible)
- Location name
- Problem + solution angle
Formula: "[Service] in [Location] helps [audience] solve [problem] by providing [solution]."

RULE 6 — NLP/TOPICAL COMPLETENESS
Cover ALL of these topic facets:
- Materials/tools used
- Process/how it works
- Cost/pricing ranges
- Timeline/duration
- Benefits/advantages
- Comparisons (vs alternatives)
- Common mistakes to avoid
- FAQs
Google checks: does this page FULLY cover the topic?

RULE 7 — READABILITY
- Passive voice: under 10% of sentences
- Transition words: over 30% of sentences start with one
- Average sentence length: under 20 words
- Paragraph length: 2-4 lines max
- Never start 3+ consecutive sentences with the same word

RULE 8 — TRANSITION WORDS
Use these throughout: However, Therefore, Additionally, Meanwhile, As a result, In fact, On the other hand, For example, Because, Furthermore, Moreover, Consequently, Similarly, Specifically, In particular, Notably.

RULE 9 — INTERNAL LINKING
Include 3-5 internal link suggestions: [INTERNAL LINK: anchor text -> /suggested-page-url]
Include 1-2 external authority link suggestions: [EXTERNAL LINK: anchor text -> authority-source]

RULE 10 — IMAGE OPTIMIZATION
Suggest 3-5 images with:
- Descriptive alt text containing keyword variation
- SEO-friendly filename suggestion
- Placement location in the content
Format: [IMAGE: alt="keyword variation description" file="keyword-description.webp" placement="after H2 about X"]

RULE 11 — E-E-A-T SIGNALS
Weave in trust signals naturally:
- Years of experience / projects completed
- Certifications or qualifications
- Process explanation (shows expertise)
- Customer testimonial placeholder: [TESTIMONIAL PLACEHOLDER]
- Specific numbers and data points

RULE 12 — FAQ SECTION
End with 3-5 FAQs using these question types:
- "How much does [service] cost in [city]?"
- "How long does [service] take?"
- "What is the best [related topic]?"
- "Do I need [requirement] for [service]?"
- "Why should I choose [type of business] in [city]?"
Format as proper FAQ with Q: and A: for easy schema extraction.

RULE 13 — CTA PLACEMENT
Include calls-to-action in exactly 3 positions:
1. After the introduction (soft CTA)
2. Middle of content (value-driven CTA)
3. End of content (strong CTA)
CTAs should feel natural, not salesy. Mention the business name.

RULE 14 — LOCAL SEO SIGNALS
Naturally include:
- City name (6+ times)
- Service area / nearby neighborhoods
- Nearby landmarks or known areas
- Province/state name
- "serving [city] and surrounding areas"

RULE 15 — CONVERSION ELEMENTS
Include these persuasion elements:
- Clear benefits (not just features)
- Social proof placeholder: [SOCIAL PROOF: X+ projects completed]
- Guarantee or assurance statement
- Timeline clarity ("completed in X days")
- Pricing transparency ("starting from $X")

OUTPUT FORMAT:

Return your response as JSON:
{{
  "meta_title": "Primary Keyword | Business Name (under 60 chars)",
  "meta_description": "Compelling description with primary keyword and CTA (under 160 chars)",
  "url_slug": "/suggested-url-slug",
  "content": "Full HTML content with proper H1-H3 tags, paragraphs, lists, CTAs, FAQs",
  "word_count": 0,
  "primary_keyword_count": 0,
  "images": [{{"alt": "...", "filename": "...", "placement": "..."}}],
  "internal_links": [{{"anchor": "...", "url": "..."}}],
  "external_links": [{{"anchor": "...", "url": "..."}}],
  "faqs": [{{"question": "...", "answer": "..."}}],
  "semantic_keywords_used": ["variation1", "variation2"]
}}

Write the content NOW. Follow ALL 15 rules exactly."""


COMPETITOR_ANALYSIS_PROMPT = """Analyze the top-ranking pages for the keyword "{keyword}" in {city}, {country}.

I'll provide you with the content from the top ranking pages.

{competitor_data}

For each page, extract:
1. Word count
2. Number of H2, H3 tags
3. How many times the primary keyword appears
4. What topics/sections they cover
5. Whether they have FAQs
6. CTA count and placement
7. Internal link count
8. Image count

Then calculate:
- Average word count across all pages
- Target word count (avg x 1.25, rounded to nearest 100)
- Common topics ALL competitors cover (must-have sections)
- Topics only 1-2 competitors cover (differentiation opportunities)
- Topics NO competitor covers (gap = your advantage)

Return JSON:
{{
  "competitors": [
    {{"url": "...", "word_count": 0, "h2_count": 0, "keyword_count": 0, "topics": [...], "has_faq": false}}
  ],
  "averages": {{
    "word_count": 0,
    "h2_count": 0,
    "keyword_count": 0,
    "faq_percentage": 0
  }},
  "targets": {{
    "word_count": 0,
    "h2_count": 0,
    "min_keyword_count": 0
  }},
  "must_have_topics": [...],
  "differentiation_topics": [...],
  "gap_topics": [...],
  "semantic_keywords": [...]
}}"""


COMPETITOR_ESTIMATION_PROMPT = """For the keyword "{keyword}" in {city}, {country}, estimate what the top 5 ranking pages would typically contain.

Business type: {business_type}

Estimate:
- Average word count for top ranking pages
- Common H2 topics/sections
- Whether they typically have FAQ sections
- Semantic keywords they would use
- Content gaps competitors likely miss

Return JSON:
{{
  "competitors": [],
  "averages": {{
    "word_count": 1200,
    "h2_count": 5,
    "keyword_count": 8,
    "faq_percentage": 40
  }},
  "targets": {{
    "word_count": 1500,
    "h2_count": 6,
    "min_keyword_count": 8
  }},
  "must_have_topics": ["services offered", "pricing", "process", "benefits", "testimonials"],
  "differentiation_topics": [],
  "gap_topics": ["common mistakes", "comparison guides", "cost breakdown"],
  "semantic_keywords": []
}}"""


SEMANTIC_VARIATIONS_PROMPT = """Generate 8 semantic variations of the keyword "{keyword}" for a {business_type} in {city}.

These should be natural ways people search for the same thing — not exact synonyms but related phrasing.

Return JSON array only:
["variation 1", "variation 2", "variation 3", "variation 4", "variation 5", "variation 6", "variation 7", "variation 8"]"""


FIX_CONTENT_PROMPT = """The following SEO content scored {score}% — below the 80% target.

Fix ONLY these specific issues:
{failed_rules}

ORIGINAL CONTENT (as JSON):
{original_content}

Return the COMPLETE corrected content as the same JSON format. Fix the issues while keeping all the good parts unchanged. Return valid JSON only."""


# ---------------------------------------------------------------------------
# Type aliases for dependency injection
# ---------------------------------------------------------------------------

ClaudeCaller = Callable[..., Coroutine[Any, Any, Any]]
FetchCompetitorsFn = Callable[..., Coroutine[Any, Any, list]]
ScrapePageFn = Callable[..., Coroutine[Any, Any, dict]]


# ---------------------------------------------------------------------------
# Competitor Analysis
# ---------------------------------------------------------------------------

async def analyze_competitors(
    keyword: str,
    city: str,
    country: str,
    *,
    claude_caller: ClaudeCaller,
    fetch_competitors_fn: FetchCompetitorsFn,
    scrape_page_fn: ScrapePageFn,
    business_type: str = "local business",
    top_n: int = 5,
) -> dict:
    """
    Analyze top-ranking pages for a keyword.
    Scrapes top SERP results, extracts data, then uses Claude to analyze.
    Falls back to Claude estimation if scraping fails.
    """
    start = time.time()
    location = f"{city}, {country}"

    # Fetch top competitors from SerpApi
    competitors = await fetch_competitors_fn(keyword, location, num=top_n)

    if not competitors:
        # Fallback: Claude estimation
        logger.info("No competitors found — using Claude estimation")
        prompt = COMPETITOR_ESTIMATION_PROMPT.format(
            keyword=keyword, city=city, country=country,
            business_type=business_type,
        )
        result = await claude_caller(
            "You are an SEO analyst. Respond with valid JSON only.",
            prompt, max_tokens=1500,
        )
        if isinstance(result, dict) and "averages" in result:
            target_words = result.get("targets", {}).get("word_count", 1500)
            target_words = max(1100, min(3000, round(target_words / 100) * 100))
            result["targets"]["word_count"] = target_words
            result["analysis_time_seconds"] = round(time.time() - start, 1)
            result["competitors_analyzed"] = 0
            result["source"] = "estimated"
            return result
        return _default_competitor_analysis()

    # Scrape competitor pages concurrently
    scrape_tasks = [scrape_page_fn(c["url"]) for c in competitors[:top_n]]
    pages = await asyncio.gather(*scrape_tasks, return_exceptions=True)

    # Build competitor data string for Claude analysis
    comp_data_parts = []
    scraped_pages = []
    for comp, page in zip(competitors[:top_n], pages):
        if isinstance(page, Exception) or not page.get("success"):
            continue
        scraped_pages.append(page)
        headings = [h["text"] for h in page.get("headings", [])[:10]]
        comp_data_parts.append(
            f"URL: {comp['url']}\n"
            f"Title: {page.get('title', 'N/A')}\n"
            f"Word Count: {page.get('word_count', 0)}\n"
            f"H1: {page.get('h1', 'N/A')}\n"
            f"Headings: {', '.join(headings)}\n"
            f"Content preview: {page.get('content', '')[:600]}"
        )

    if not scraped_pages:
        # All scrapes failed — fall back to estimation
        logger.warning("All competitor scrapes failed — using Claude estimation")
        prompt = COMPETITOR_ESTIMATION_PROMPT.format(
            keyword=keyword, city=city, country=country,
            business_type=business_type,
        )
        result = await claude_caller(
            "You are an SEO analyst. Respond with valid JSON only.",
            prompt, max_tokens=1500,
        )
        if isinstance(result, dict) and "averages" in result:
            result["analysis_time_seconds"] = round(time.time() - start, 1)
            result["competitors_analyzed"] = 0
            result["source"] = "estimated"
            return result
        return _default_competitor_analysis()

    # Calculate basic averages from scraped data
    word_counts = [p.get("word_count", 0) for p in scraped_pages if p.get("word_count", 0) > 0]
    avg_words = round(sum(word_counts) / len(word_counts)) if word_counts else 1200

    # Calculate targets
    target_words = max(1100, min(3000, round(avg_words * 1.25 / 100) * 100))

    # Use Claude to analyze competitor content in detail
    comp_data_str = "\n\n---\n\n".join(comp_data_parts)
    prompt = COMPETITOR_ANALYSIS_PROMPT.format(
        keyword=keyword, city=city, country=country,
        competitor_data=comp_data_str,
    )

    analysis = await claude_caller(
        "You are an SEO analyst. Respond with valid JSON only.",
        prompt, max_tokens=2000,
    )

    if not isinstance(analysis, dict) or "averages" not in analysis:
        # Claude analysis failed — return basic analysis from scraped data
        analysis = {
            "competitors": [
                {"url": p.get("url", ""), "word_count": p.get("word_count", 0)}
                for p in scraped_pages
            ],
            "averages": {"word_count": avg_words, "h2_count": 5, "keyword_count": 8, "faq_percentage": 30},
            "targets": {"word_count": target_words, "h2_count": 6, "min_keyword_count": 8},
            "must_have_topics": [],
            "differentiation_topics": [],
            "gap_topics": [],
            "semantic_keywords": [],
        }

    # Ensure targets are within bounds
    analysis.setdefault("targets", {})
    analysis["targets"]["word_count"] = target_words
    analysis.setdefault("averages", {})
    analysis["averages"]["word_count"] = avg_words
    analysis["analysis_time_seconds"] = round(time.time() - start, 1)
    analysis["competitors_analyzed"] = len(scraped_pages)
    analysis["source"] = "scraped"

    return analysis


def _default_competitor_analysis() -> dict:
    """Fallback competitor analysis with reasonable defaults."""
    return {
        "competitors": [],
        "averages": {"word_count": 1200, "h2_count": 5, "keyword_count": 8, "faq_percentage": 30},
        "targets": {"word_count": 1500, "h2_count": 6, "min_keyword_count": 8},
        "must_have_topics": ["services", "pricing", "process", "benefits", "FAQ"],
        "differentiation_topics": [],
        "gap_topics": ["cost breakdown", "common mistakes", "comparisons"],
        "semantic_keywords": [],
        "competitors_analyzed": 0,
        "source": "default",
    }


# ---------------------------------------------------------------------------
# SEO Content Generator
# ---------------------------------------------------------------------------

async def generate_seo_content(
    *,
    business_name: str,
    business_type: str,
    city: str,
    country: str,
    services: str,
    target_keyword: str,
    secondary_keywords: list[str],
    page_type: str,
    competitor_data: dict,
    claude_caller: ClaudeCaller,
    existing_content: Optional[str] = None,
) -> dict:
    """
    Generate SEO-optimized content using the 15-rule system prompt.
    Scores output and auto-fixes if score < 80%.
    """
    start = time.time()

    # Step 1: Generate semantic keyword variations
    sem_prompt = SEMANTIC_VARIATIONS_PROMPT.format(
        keyword=target_keyword, business_type=business_type, city=city,
    )
    sem_result = await claude_caller(
        "You are an SEO keyword expert. Respond with a JSON array only.",
        sem_prompt, max_tokens=300,
    )
    if isinstance(sem_result, list):
        semantic_variations = sem_result
    elif isinstance(sem_result, dict) and "raw_response" not in sem_result:
        semantic_variations = list(sem_result.values())[:8] if sem_result else []
    else:
        # Generate basic variations
        semantic_variations = [
            f"best {target_keyword}",
            f"{target_keyword} services",
            f"affordable {target_keyword}",
            f"professional {target_keyword}",
            f"top {target_keyword}",
            f"{target_keyword} near me",
            f"local {target_keyword}",
            f"trusted {target_keyword}",
        ]

    # Step 2: Get targets from competitor data
    targets = competitor_data.get("targets", {})
    target_words = targets.get("word_count", 1500)
    avg_words = competitor_data.get("averages", {}).get("word_count", 1200)

    # Step 3: Build the system prompt
    system = SEO_CONTENT_SYSTEM_PROMPT.format(
        business_name=business_name,
        business_category=business_type,
        city=city,
        country=country,
        services=services,
        primary_keyword=target_keyword,
        secondary_keywords=", ".join(secondary_keywords) if secondary_keywords else target_keyword,
        competitor_avg_words=avg_words,
        target_words=target_words,
        semantic_variations=", ".join(semantic_variations[:8]),
    )

    # Step 4: Build the user prompt
    gap_topics = competitor_data.get("gap_topics", [])
    must_have = competitor_data.get("must_have_topics", [])

    user_prompt = f"Generate a {page_type} for {business_name} in {city}, {country}.\n"
    user_prompt += f"Primary keyword: \"{target_keyword}\"\n"
    if secondary_keywords:
        user_prompt += f"Secondary keywords: {', '.join(secondary_keywords)}\n"
    if must_have:
        user_prompt += f"Must-cover topics (competitors all cover these): {', '.join(must_have[:8])}\n"
    if gap_topics:
        user_prompt += f"Content gaps to exploit (competitors miss these): {', '.join(gap_topics[:5])}\n"
    if existing_content:
        user_prompt += f"\nExisting content to improve upon:\n{existing_content[:2000]}\n"
    user_prompt += "\nGenerate the content now following ALL 15 rules. Return valid JSON only."

    # Step 5: Generate content
    content_json = await claude_caller(system, user_prompt, max_tokens=8192)

    if not isinstance(content_json, dict) or "content" not in content_json:
        logger.warning("Content generation returned unexpected format")
        return {
            "content": content_json if isinstance(content_json, dict) else {"raw_response": str(content_json)},
            "seo_score": {"total_score": 0, "max_score": 100, "percentage": 0, "grade": "F", "rules": {}},
            "competitor_analysis": {
                "avg_words": avg_words,
                "target_words": target_words,
                "competitors_analyzed": competitor_data.get("competitors_analyzed", 0),
                "gap_topics": gap_topics,
            },
            "meta": {
                "generation_time_seconds": round(time.time() - start, 1),
                "model": "claude",
                "auto_fixed": False,
            },
        }

    # Step 6: Score the content
    score_result = calculate_seo_score(content_json, targets, target_keyword, city)

    auto_fixed = False

    # Step 7: Auto-fix if score < 80%
    if score_result["percentage"] < 80:
        logger.info(f"Score {score_result['percentage']}% < 80% — attempting auto-fix")
        failed_rules = []
        for rule_name, rule_data in score_result["rules"].items():
            if rule_data["status"] in ("fail", "warn"):
                failed_rules.append(f"- {rule_name}: {rule_data['detail']} (scored {rule_data['score']}/{rule_data['max']})")

        import json
        fix_prompt = FIX_CONTENT_PROMPT.format(
            score=score_result["percentage"],
            failed_rules="\n".join(failed_rules),
            original_content=json.dumps(content_json, indent=2)[:6000],
        )

        fixed_json = await claude_caller(
            "You are an SEO content editor. Fix the issues and return the complete corrected content as valid JSON only.",
            fix_prompt,
            max_tokens=8192,
        )

        if isinstance(fixed_json, dict) and "content" in fixed_json:
            fixed_score = calculate_seo_score(fixed_json, targets, target_keyword, city)
            if fixed_score["percentage"] > score_result["percentage"]:
                content_json = fixed_json
                score_result = fixed_score
                auto_fixed = True
                logger.info(f"Auto-fix improved score to {score_result['percentage']}%")
            else:
                logger.info("Auto-fix did not improve score — keeping original")

    return {
        "content": content_json,
        "seo_score": score_result,
        "competitor_analysis": {
            "avg_words": avg_words,
            "target_words": target_words,
            "competitors_analyzed": competitor_data.get("competitors_analyzed", 0),
            "gap_topics": gap_topics[:10],
        },
        "meta": {
            "generation_time_seconds": round(time.time() - start, 1),
            "model": "claude",
            "auto_fixed": auto_fixed,
        },
    }


# ---------------------------------------------------------------------------
# SEO Score Calculator — Pure Python, no Claude needed
# ---------------------------------------------------------------------------

def calculate_seo_score(
    content_json: dict,
    targets: dict,
    primary_keyword: str,
    city: str,
) -> dict:
    """
    Score generated content against 15 SEO rules.
    Returns: total score (0-100) + per-rule breakdown.
    """
    scores: dict[str, dict] = {}
    content = content_json.get("content", "")
    words = content.split()
    word_count = len(words)
    # Split on sentence-ending punctuation
    sentences = [s.strip() for s in re.split(r'[.!?]+', content) if s.strip()]

    # ── RULE 1: Content Length (10 points) ────────────────────────
    target = targets.get("word_count", 1500)
    ratio = word_count / target if target > 0 else 0
    if 0.95 <= ratio <= 1.10:
        scores["content_length"] = {
            "score": 10, "max": 10, "status": "pass",
            "detail": f"{word_count} words (target: {target})",
        }
    elif 0.80 <= ratio < 0.95 or 1.10 < ratio <= 1.25:
        scores["content_length"] = {
            "score": 6, "max": 10, "status": "warn",
            "detail": f"{word_count} words — slightly {'under' if ratio < 1 else 'over'} target {target}",
        }
    else:
        scores["content_length"] = {
            "score": 3, "max": 10, "status": "fail",
            "detail": f"{word_count} words — target was {target}",
        }

    # ── RULE 2: Keyword Placement (15 points — most important) ───
    kw = primary_keyword.lower()
    content_lower = content.lower()

    # Extract H1 from content HTML
    h1_match = re.search(r'<h1[^>]*>(.*?)</h1>', content, re.I | re.S)
    h1_text = h1_match.group(1).lower() if h1_match else ""

    # Extract H2 tags from content
    h2_matches = re.findall(r'<h2[^>]*>(.*?)</h2>', content, re.I | re.S)
    h2_texts = [h.lower() for h in h2_matches]

    placements = {
        "h1": kw in h1_text,
        "first_100_words": kw in " ".join(words[:100]).lower(),
        "meta_title": kw in content_json.get("meta_title", "").lower(),
        "meta_description": kw in content_json.get("meta_description", "").lower(),
        "h2": any(kw in h2 for h2 in h2_texts),
        "image_alt": any(kw in img.get("alt", "").lower() for img in content_json.get("images", [])),
        "last_100_words": kw in " ".join(words[-100:]).lower(),
    }
    placement_score = sum(placements.values())
    total_placements = len(placements)
    placement_pts = round((placement_score / total_placements) * 15) if total_placements > 0 else 0
    scores["keyword_placement"] = {
        "score": placement_pts, "max": 15,
        "status": "pass" if placement_score >= 6 else ("warn" if placement_score >= 4 else "fail"),
        "detail": f"{placement_score}/{total_placements} positions filled",
        "positions": placements,
    }

    # ── RULE 3: Keyword Frequency (8 points) ─────────────────────
    kw_count = content_lower.count(kw)
    per_1000 = (kw_count / word_count) * 1000 if word_count > 0 else 0
    if 6 <= per_1000 <= 10:
        scores["keyword_frequency"] = {
            "score": 8, "max": 8, "status": "pass",
            "detail": f"{kw_count} times ({per_1000:.1f} per 1000 words)",
        }
    elif 4 <= per_1000 < 6 or 10 < per_1000 <= 13:
        scores["keyword_frequency"] = {
            "score": 5, "max": 8, "status": "warn",
            "detail": f"{kw_count} times ({per_1000:.1f}/1000) — adjust slightly",
        }
    else:
        scores["keyword_frequency"] = {
            "score": 2, "max": 8, "status": "fail",
            "detail": f"{kw_count} times ({per_1000:.1f}/1000) — {'too few' if per_1000 < 4 else 'too many'}",
        }

    # ── RULE 4: Heading Structure (7 points) ─────────────────────
    h1s = len(re.findall(r'<h1[^>]*>', content, re.I))
    h2s = len(re.findall(r'<h2[^>]*>', content, re.I))
    h3s = len(re.findall(r'<h3[^>]*>', content, re.I))
    h_score = 0
    if h1s == 1:
        h_score += 2
    if 4 <= h2s <= 6:
        h_score += 3
    elif 3 <= h2s <= 8:
        h_score += 2
    if 6 <= h3s <= 12:
        h_score += 2
    elif 4 <= h3s <= 15:
        h_score += 1
    scores["heading_structure"] = {
        "score": h_score, "max": 7,
        "status": "pass" if h_score >= 5 else ("warn" if h_score >= 3 else "fail"),
        "detail": f"H1:{h1s} H2:{h2s} H3:{h3s}",
    }

    # ── RULE 5: First Paragraph (5 points) ───────────────────────
    first_para = content.split('</p>')[0] if '</p>' in content else " ".join(words[:80])
    first_para_lower = first_para.lower()
    has_kw = kw in first_para_lower
    has_city = city.lower() in first_para_lower
    fp_score = 0
    if has_kw:
        fp_score += 3
    if has_city:
        fp_score += 2
    scores["first_paragraph"] = {
        "score": fp_score, "max": 5,
        "status": "pass" if fp_score >= 4 else ("warn" if fp_score >= 2 else "fail"),
        "detail": f"Keyword: {'pass' if has_kw else 'missing'}, Location: {'pass' if has_city else 'missing'}",
    }

    # ── RULE 6: Topical Completeness (7 points) ──────────────────
    nlp_topics = [
        "cost", "price", "pricing", "process", "timeline", "benefit",
        "material", "mistake", "comparison", "faq", "how",
    ]
    found = sum(1 for t in nlp_topics if t in content_lower)
    nlp_score = min(7, round((found / len(nlp_topics)) * 7))
    scores["topical_completeness"] = {
        "score": nlp_score, "max": 7,
        "status": "pass" if nlp_score >= 5 else ("warn" if nlp_score >= 3 else "fail"),
        "detail": f"{found}/{len(nlp_topics)} topic facets covered",
    }

    # ── RULE 7: Readability (7 points) ───────────────────────────
    passive_indicators = [
        "is done", "was made", "are built", "were created",
        "is provided", "was completed", "are offered", "is known",
        "is located", "was established", "are designed", "is recommended",
    ]
    passive_count = sum(1 for p in passive_indicators if p in content_lower)
    passive_pct = (passive_count / max(len(sentences), 1)) * 100

    transition_count = 0
    for s in sentences:
        s_lower = s.lower().strip()
        if any(s_lower.startswith(tw) for tw in TRANSITION_WORDS):
            transition_count += 1
    transition_pct = (transition_count / max(len(sentences), 1)) * 100

    read_score = 0
    if passive_pct < 10:
        read_score += 3
    elif passive_pct < 15:
        read_score += 2
    if transition_pct > 25:
        read_score += 4
    elif transition_pct > 15:
        read_score += 2
    scores["readability"] = {
        "score": read_score, "max": 7,
        "status": "pass" if read_score >= 5 else ("warn" if read_score >= 3 else "fail"),
        "detail": f"Passive: {passive_pct:.0f}%, Transitions: {transition_pct:.0f}%",
    }

    # ── RULE 8: Transition Words (included in readability, separate display) ─
    tw_found = sum(1 for tw in TRANSITION_WORDS if tw in content_lower)
    scores["transition_words"] = {
        "score": min(5, round((tw_found / 10) * 5)), "max": 5,
        "status": "pass" if tw_found >= 8 else ("warn" if tw_found >= 4 else "fail"),
        "detail": f"{tw_found} transition words found",
    }

    # ── RULE 9: Internal Linking (5 points) ──────────────────────
    internal_links = len(content_json.get("internal_links", []))
    external_links = len(content_json.get("external_links", []))
    link_score = 0
    if 3 <= internal_links <= 5:
        link_score += 3
    elif internal_links >= 2:
        link_score += 2
    elif internal_links >= 1:
        link_score += 1
    if 1 <= external_links <= 2:
        link_score += 2
    elif external_links >= 1:
        link_score += 1
    scores["internal_linking"] = {
        "score": link_score, "max": 5,
        "status": "pass" if link_score >= 4 else ("warn" if link_score >= 2 else "fail"),
        "detail": f"{internal_links} internal, {external_links} external links",
    }

    # ── RULE 10: Image Optimization (5 points) ───────────────────
    images = content_json.get("images", [])
    img_score = 0
    if len(images) >= 3:
        img_score += 3
    elif len(images) >= 1:
        img_score += 1
    kw_parts = kw.split()
    imgs_with_kw_alt = sum(
        1 for img in images
        if any(k in img.get("alt", "").lower() for k in kw_parts)
    )
    if imgs_with_kw_alt >= 2:
        img_score += 2
    elif imgs_with_kw_alt >= 1:
        img_score += 1
    scores["image_optimization"] = {
        "score": img_score, "max": 5,
        "status": "pass" if img_score >= 4 else ("warn" if img_score >= 2 else "fail"),
        "detail": f"{len(images)} images, {imgs_with_kw_alt} with keyword alt text",
    }

    # ── RULE 11: E-E-A-T Signals (6 points) ─────────────────────
    eeat_found = sum(1 for s in EEAT_SIGNALS if s in content_lower)
    eeat_score = min(6, round((eeat_found / 5) * 6))
    scores["eeat_signals"] = {
        "score": eeat_score, "max": 6,
        "status": "pass" if eeat_score >= 4 else ("warn" if eeat_score >= 2 else "fail"),
        "detail": f"{eeat_found} trust signals found",
    }

    # ── RULE 12: FAQ Section (5 points) ──────────────────────────
    faqs = content_json.get("faqs", [])
    if 3 <= len(faqs) <= 5:
        faq_score = 5
    elif len(faqs) >= 2:
        faq_score = 3
    elif len(faqs) >= 1:
        faq_score = 1
    else:
        faq_score = 0
    scores["faq_section"] = {
        "score": faq_score, "max": 5,
        "status": "pass" if faq_score >= 4 else ("warn" if faq_score >= 2 else "fail"),
        "detail": f"{len(faqs)} FAQs included",
    }

    # ── RULE 13: CTA Placement (5 points) ────────────────────────
    cta_count = sum(1 for c in CTA_PHRASES if c in content_lower)
    cta_score = min(5, cta_count * 2)
    scores["cta_placement"] = {
        "score": cta_score, "max": 5,
        "status": "pass" if cta_score >= 4 else ("warn" if cta_score >= 2 else "fail"),
        "detail": f"{cta_count} CTAs detected",
    }

    # ── RULE 14: Local SEO Signals (5 points) ────────────────────
    city_lower = city.lower()
    city_count = content_lower.count(city_lower)
    local_score = 0
    if city_count >= 6:
        local_score += 3
    elif city_count >= 3:
        local_score += 2
    elif city_count >= 1:
        local_score += 1
    if "serving" in content_lower or "surrounding" in content_lower:
        local_score += 1
    if any(n in content_lower for n in ["nearby", "neighborhood", "area"]):
        local_score += 1
    local_score = min(5, local_score)
    scores["local_seo"] = {
        "score": local_score, "max": 5,
        "status": "pass" if local_score >= 4 else ("warn" if local_score >= 2 else "fail"),
        "detail": f"City mentioned {city_count} times",
    }

    # ── RULE 15: Conversion Elements (5 points) ──────────────────
    conversion_signals = [
        "benefit", "guarantee", "free", "starting from",
        "completed in", "within", "save", "affordable",
        "satisfaction", "no obligation", "risk-free",
    ]
    conv_found = sum(1 for c in conversion_signals if c in content_lower)
    conv_score = min(5, round((conv_found / 4) * 5))
    scores["conversion"] = {
        "score": conv_score, "max": 5,
        "status": "pass" if conv_score >= 4 else ("warn" if conv_score >= 2 else "fail"),
        "detail": f"{conv_found} conversion elements found",
    }

    # ── TOTAL ────────────────────────────────────────────────────
    total = sum(r["score"] for r in scores.values())
    max_total = sum(r["max"] for r in scores.values())
    percentage = round((total / max_total) * 100) if max_total > 0 else 0

    if percentage >= 90:
        grade = "A+"
    elif percentage >= 80:
        grade = "A"
    elif percentage >= 70:
        grade = "B"
    elif percentage >= 60:
        grade = "C"
    else:
        grade = "D"

    return {
        "total_score": total,
        "max_score": max_total,
        "percentage": percentage,
        "grade": grade,
        "rules": scores,
    }


# ---------------------------------------------------------------------------
# Score existing content (no generation)
# ---------------------------------------------------------------------------

def score_existing_content(
    content_html: str,
    keyword: str,
    city: str,
) -> dict:
    """Score existing content without generating new content."""
    words = content_html.split()
    word_count = len(words)

    # Build a minimal content_json from raw HTML
    content_json = {
        "content": content_html,
        "meta_title": "",
        "meta_description": "",
        "images": [],
        "internal_links": [],
        "external_links": [],
        "faqs": [],
    }

    # Try to extract meta title from content
    title_match = re.search(r'<title[^>]*>(.*?)</title>', content_html, re.I | re.S)
    if title_match:
        content_json["meta_title"] = title_match.group(1)

    # Count images
    img_matches = re.findall(r'<img[^>]*alt="([^"]*)"[^>]*>', content_html, re.I)
    content_json["images"] = [{"alt": alt, "filename": "", "placement": ""} for alt in img_matches]

    # Count internal/external links
    link_matches = re.findall(r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', content_html, re.I | re.S)
    for href, anchor in link_matches:
        if href.startswith("/") or href.startswith("#"):
            content_json["internal_links"].append({"anchor": anchor, "url": href})
        elif href.startswith("http"):
            content_json["external_links"].append({"anchor": anchor, "url": href})

    # Check for FAQ section
    faq_matches = re.findall(r'(?:Q:|question)[:\s]*(.*?)(?:A:|answer)[:\s]*(.*?)(?=(?:Q:|question)|\Z)', content_html, re.I | re.S)
    content_json["faqs"] = [{"question": q.strip(), "answer": a.strip()} for q, a in faq_matches]

    targets = {"word_count": max(1200, round(word_count * 1.1 / 100) * 100)}

    return calculate_seo_score(content_json, targets, keyword, city)


# ---------------------------------------------------------------------------
# Content Type Helpers
# ---------------------------------------------------------------------------

async def generate_homepage_content(
    *,
    business_name: str,
    business_type: str,
    city: str,
    country: str,
    services: str,
    target_url: str,
    competitor_data: dict,
    claude_caller: ClaudeCaller,
) -> dict:
    """Generate homepage content with LocalBusiness focus."""
    primary_kw = f"{business_type} {city}".lower()
    secondary = [
        f"best {business_type} in {city}",
        f"{business_type} near me",
        f"{business_type} services {city}",
    ]

    return await generate_seo_content(
        business_name=business_name,
        business_type=business_type,
        city=city,
        country=country,
        services=services,
        target_keyword=primary_kw,
        secondary_keywords=secondary,
        page_type="homepage",
        competitor_data=competitor_data,
        claude_caller=claude_caller,
    )


async def generate_service_page(
    *,
    business_name: str,
    business_type: str,
    city: str,
    country: str,
    services: str,
    service_name: str,
    competitor_data: dict,
    claude_caller: ClaudeCaller,
) -> dict:
    """Generate a service-specific page."""
    primary_kw = f"{service_name} {city}".lower()
    secondary = [
        f"best {service_name} in {city}",
        f"affordable {service_name} {city}",
        f"professional {service_name} near me",
    ]

    return await generate_seo_content(
        business_name=business_name,
        business_type=business_type,
        city=city,
        country=country,
        services=services,
        target_keyword=primary_kw,
        secondary_keywords=secondary,
        page_type="service_page",
        competitor_data=competitor_data,
        claude_caller=claude_caller,
    )


async def generate_area_page(
    *,
    business_name: str,
    business_type: str,
    city: str,
    country: str,
    services: str,
    target_city: str,
    service_name: str,
    competitor_data: dict,
    claude_caller: ClaudeCaller,
) -> dict:
    """Generate a service area page: '{service} in {target_city}'."""
    primary_kw = f"{service_name} in {target_city}".lower()
    secondary = [
        f"{service_name} {target_city}",
        f"best {service_name} {target_city}",
        f"{business_type} {target_city}",
    ]

    return await generate_seo_content(
        business_name=business_name,
        business_type=business_type,
        city=target_city,
        country=country,
        services=services,
        target_keyword=primary_kw,
        secondary_keywords=secondary,
        page_type="area_page",
        competitor_data=competitor_data,
        claude_caller=claude_caller,
    )
