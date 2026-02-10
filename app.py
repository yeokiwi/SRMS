import os
import asyncio
import json
import re
from datetime import date, datetime, timedelta
from flask import Flask, render_template, request, jsonify
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai import LLMConfig
from crawl4ai.content_filter_strategy import LLMContentFilter
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

today = date.today()
formatted_date = today.strftime("%d %b, %Y")


def build_analysis_instruction():
    """Build the LLM instruction for website change analysis."""
    return f"""You are a website change analyst. Today's date is {formatted_date}.

Analyze the provided web page content and produce a structured report with the following sections.
For each section, only include items you can find evidence of in the content. If a section has no relevant findings, write "No relevant content found."

**IMPORTANT**: Distinguish between:
- CONFIRMED recent: Items with explicit dates within the last 30 days (since {(today - timedelta(days=30)).strftime('%d %b, %Y')})
- LIKELY recent: Items that appear recent based on context clues (e.g., "new", "updated", "coming soon") but lack explicit dates
- NOT recent: Items with dates older than 30 days

## 1. What's New or Changed (Last 30 Days)
Identify any content that appears to have been added or modified in the last 30 days.
Include the date if available, and note whether the dating is confirmed or estimated.

## 2. Announcements, Blog Posts & News
List any announcements, press releases, blog posts, or news items from the past month.
Include titles, dates, and brief summaries.

## 3. Product, Service & Feature Updates
Note any updates to products, services, or features mentioned on the page.
Include version numbers, release dates, or changelogs if present.

## 4. Pricing, Terms of Service & Policy Changes
Flag any changes to pricing, terms of service, privacy policies, or other policies.
Note effective dates if mentioned.

## 5. Page Metadata
- Last updated date (if found on page):
- Copyright year:
- Any "last modified" headers or timestamps:

At the top of your response, provide a one-line **Recency Verdict**:
State whether this page has confirmed recent updates, likely recent updates, or no evidence of recent changes.
"""


async def crawl_and_analyze(url, max_depth=1):
    """Crawl a URL and analyze it for recent changes."""
    browser_config = BrowserConfig(
        headless=True,
        enable_stealth=True,
        user_agent_mode="random",
        verbose=False
    )

    deep_crawl_strategy = BFSDeepCrawlStrategy(
        max_depth=max_depth,
        include_external=False
    )

    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.ENABLED,
        deep_crawl_strategy=deep_crawl_strategy,
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=False
    )

    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        return {"error": "DEEPSEEK_API_KEY not found in environment variables. Create a .env file with your API key."}

    llm_filter = LLMContentFilter(
        llm_config=LLMConfig(
            provider="deepseek/deepseek-chat",
            api_token=api_key
        ),
        instruction=build_analysis_instruction(),
        verbose=False
    )

    pages = []

    try:
        async with AsyncWebCrawler(config=browser_config) as crawler:
            results = await crawler.arun(url, config=run_config)

            for i, result in enumerate(results):
                depth = result.metadata.get('depth', 0)
                page_url = result.url

                # Apply LLM analysis
                filtered_content = llm_filter.filter_content(result.cleaned_html)

                analysis_text = ""
                if filtered_content:
                    analysis_text = "\n".join(filtered_content)

                # Get raw markdown for reference
                raw_markdown = ""
                if result.markdown:
                    if isinstance(result.markdown, str):
                        raw_markdown = result.markdown
                    elif hasattr(result.markdown, 'raw_markdown'):
                        raw_markdown = result.markdown.raw_markdown
                    else:
                        raw_markdown = str(result.markdown)

                pages.append({
                    "index": i + 1,
                    "url": page_url,
                    "depth": depth,
                    "analysis": analysis_text,
                    "content_length": len(raw_markdown),
                    "has_content": bool(analysis_text.strip())
                })

        return {
            "success": True,
            "url": url,
            "query_date": formatted_date,
            "lookback_date": (today - timedelta(days=30)).strftime("%d %b, %Y"),
            "total_pages_crawled": len(pages),
            "max_depth": max_depth,
            "pages": pages
        }

    except Exception as e:
        return {"error": f"Crawl failed: {str(e)}"}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query_website():
    data = request.get_json()
    url = data.get("url", "").strip()
    max_depth = int(data.get("max_depth", 1))

    if not url:
        return jsonify({"error": "URL is required"}), 400

    # Basic URL validation
    if not re.match(r'^https?://', url):
        url = "https://" + url

    # Cap max_depth to prevent excessive crawling
    max_depth = min(max(max_depth, 0), 2)

    result = asyncio.run(crawl_and_analyze(url, max_depth))

    if "error" in result:
        return jsonify(result), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
