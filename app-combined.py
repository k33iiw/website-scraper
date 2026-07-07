import asyncio
import csv
import io
import json
import os
import time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

import streamlit as st
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright


# ── .env loading ───────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
ENV_PATH = APP_DIR / ".env"

try:
    from dotenv import load_dotenv
    # Force-load the .env file beside this app file.
    # This helps prevent old terminal environment variables from overriding your project .env.
    load_dotenv(dotenv_path=ENV_PATH, override=True)
except ImportError:
    pass


# Streamlit page config must be the first Streamlit UI command.
st.set_page_config(page_title="AI Website Scraper", layout="wide")

st.title("AI Website Scraper")
st.write(
    "Detect same-domain links, manually select two or more pages, scrape them, "
    "and optionally use AI to extract structured JSON."
)


def get_secret(name: str, default: str = "") -> str:
    """Read secrets from .env/environment first, then Streamlit secrets for cloud deployment."""
    env_value = os.getenv(name, "")
    if env_value:
        return env_value
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


# ── OpenAI pricing (USD per 1M tokens) ────────────────────────────────────────
OPENAI_PRICING = {
    "gpt-4o": {"input": 2.50, "cached_input": 1.25, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "cached_input": 0.075, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "cached_input": 0.50, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "cached_input": 0.10, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "cached_input": 0.025, "output": 0.40},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
}
DEFAULT_PRICING_MODEL = "gpt-4o-mini"

GEMINI_PRICING = {
    "gemini-2.5-flash-lite": {"input": 0.10, "cached_input": 0.01, "output": 0.40},
    "gemini-2.5-flash": {"input": 0.30, "cached_input": 0.03, "output": 2.50},
    "gemini-2.0-flash": {"input": 0.10, "cached_input": 0.01, "output": 0.40},
}

DEFAULT_GEMINI_PRICING_MODEL = "gemini-2.5-flash-lite"

GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]


SKIP_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".css", ".js", ".zip", ".rar", ".7z", ".mp4", ".mp3", ".avi",
    ".mov", ".wmv", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
)
DOWNLOAD_EXTENSIONS = (".pdf", ".csv", ".txt")


# ── URL helpers ────────────────────────────────────────────────────────────────
def normalize_url(base_url: str, href: str | None) -> str | None:
    if not href:
        return None
    href = href.strip()
    if not href or href.startswith(("mailto:", "tel:", "javascript:", "#")):
        return None

    normalized = urljoin(base_url, href)
    parsed = urlparse(normalized)
    if parsed.scheme not in ("http", "https"):
        return None

    # Drop fragments so /page#one and /page#two are treated as the same page.
    return parsed._replace(fragment="").geturl()


def same_domain(base_url: str, target_url: str) -> bool:
    return urlparse(base_url).netloc.lower() == urlparse(target_url).netloc.lower()


def is_download_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(DOWNLOAD_EXTENSIONS)


def is_scrapable_page_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    if path.endswith(SKIP_EXTENSIONS):
        return False
    if is_download_url(url):
        return False
    return True


def dedupe_links(links: list[dict]) -> list[dict]:
    seen = set()
    output = []
    for link in links:
        url = link.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        output.append(link)
    return output


def dedupe_urls(urls: list[str]) -> list[str]:
    seen = set()
    output = []
    for url in urls:
        if not url or url in seen:
            continue
        seen.add(url)
        output.append(url)
    return output


# ── Text helpers ───────────────────────────────────────────────────────────────
def clean_lines(text: str, keep_duplicates: bool = False) -> str:
    lines = []
    seen = set()
    for line in text.splitlines():
        line = " ".join(line.strip().split())
        if not line:
            continue
        if keep_duplicates:
            lines.append(line)
            continue
        key = line.lower()
        if key not in seen:
            seen.add(key)
            lines.append(line)
    return "\n".join(lines)


def html_to_text(html: str, include_header_footer: bool = True, keep_duplicates: bool = True) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "canvas"]):
        tag.decompose()

    if not include_header_footer:
        for selector in ["header", "nav", "footer"]:
            for tag in soup.select(selector):
                tag.decompose()

    return clean_lines(soup.get_text("\n", strip=True), keep_duplicates=keep_duplicates)


def extract_links_from_html(base_url: str, html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a"):
        href = normalize_url(base_url, a.get("href"))
        if not href:
            continue
        label = a.get_text(" ", strip=True) or "Untitled link"
        links.append({
            "text": label,
            "url": href,
            "same_domain": same_domain(base_url, href),
            "is_download": is_download_url(href),
        })
    return dedupe_links(links)


def extract_sections_from_html(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "canvas"]):
        tag.decompose()

    sections = []
    current = None
    for tag in soup.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = tag.get_text(" ", strip=True)
        if not text:
            continue
        if tag.name in ["h1", "h2", "h3", "h4"]:
            current = {"heading": text, "content": []}
            sections.append(current)
        elif current:
            current["content"].append(text)
    return sections


def is_possible_login_page(url: str, title: str, text: str, forms: list[dict]) -> bool:
    login_keywords = [
        "login", "log in", "sign in", "signin", "member login", "customer login",
        "portal", "username", "password", "forgot password",
    ]
    combined = f"{url} {title} {text}".lower()
    has_keyword = any(keyword in combined for keyword in login_keywords)
    has_password_field = any(
        any(field.get("type") == "password" for field in form.get("inputs", []))
        for form in forms
    )
    return has_keyword or has_password_field


async def extract_form_details(page) -> list[dict]:
    # Only reads public form structure. It does not read typed values, cookies, storage, or submit forms.
    return await page.evaluate("""
    () => {
        const forms = Array.from(document.querySelectorAll("form"));
        return forms.map((form, formIndex) => {
            const inputs = Array.from(form.querySelectorAll("input, textarea, select"))
                .map(input => {
                    let labelText = "";
                    if (input.id) {
                        const safeId = CSS.escape(input.id);
                        const label = document.querySelector(`label[for="${safeId}"]`);
                        if (label) labelText = label.innerText.trim();
                    }
                    const parentLabel = input.closest("label");
                    if (!labelText && parentLabel) {
                        labelText = parentLabel.innerText.trim();
                    }
                    return {
                        tag: input.tagName.toLowerCase(),
                        type: input.getAttribute("type") || "",
                        name: input.getAttribute("name") || "",
                        id: input.getAttribute("id") || "",
                        label: labelText,
                        placeholder: input.getAttribute("placeholder") || "",
                        autocomplete: input.getAttribute("autocomplete") || "",
                        required: input.required || false
                    };
                });
            const buttons = Array.from(form.querySelectorAll("button, input[type='submit'], input[type='button']"))
                .map(btn => ({
                    text: btn.innerText || btn.value || "",
                    type: btn.getAttribute("type") || ""
                }));
            return {
                form_index: formIndex + 1,
                action: form.getAttribute("action") || "",
                method: form.getAttribute("method") || "GET",
                inputs: inputs,
                buttons: buttons
            };
        });
    }
    """)


# ── Playwright scraping ───────────────────────────────────────────────────────
async def scrape_page(page_url: str, include_header_footer: bool = True, keep_duplicates: bool = True) -> dict:
    started = time.perf_counter()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={"width": 1366, "height": 1400})
        try:
            try:
                response = await page.goto(page_url, wait_until="networkidle", timeout=60000)
            except Exception:
                response = await page.goto(page_url, wait_until="domcontentloaded", timeout=60000)

            # Hover common nav/dropdown labels so hidden menu links can appear.
            for text in [
                "OUR EXPERTISE", "Our Expertise", "SERVICES", "Services", "SOLUTIONS", "Solutions",
                "PRODUCTS", "Products", "ABOUT", "About", "CONTACT", "Contact", "BLOG", "Blog",
                "CAREERS", "Careers", "LOGIN", "Login", "SIGN IN", "Sign In", "Portal",
            ]:
                try:
                    await page.get_by_text(text, exact=False).first.hover(timeout=1000)
                    await page.wait_for_timeout(250)
                except Exception:
                    pass

            # Scroll so lazy-loaded content has a chance to appear.
            for _ in range(12):
                await page.mouse.wheel(0, 1000)
                await page.wait_for_timeout(250)

            title = await page.title()
            final_url = page.url
            html = await page.content()
            try:
                visible_text = await page.locator("body").inner_text(timeout=10000)
            except Exception:
                visible_text = ""
            forms = await extract_form_details(page)
            status_code = response.status if response else None
            error = ""

        except Exception as exc:
            title = ""
            final_url = page_url
            html = ""
            visible_text = ""
            forms = []
            status_code = None
            error = str(exc)
        finally:
            try:
                await browser.close()
            except Exception as close_exc:
                if "Connection closed" not in str(close_exc) and "Target page, context or browser has been closed" not in str(close_exc):
                    raise

    all_links = extract_links_from_html(final_url, html) if html else []
    internal_links = [
        link for link in all_links
        if link.get("same_domain") and is_scrapable_page_url(link.get("url", ""))
    ]
    download_links = [link for link in all_links if link.get("is_download")]
    cleaned_text = html_to_text(html, include_header_footer, keep_duplicates) if html else ""

    return {
        "input_url": page_url,
        "final_url": final_url,
        "title": title,
        "status_code": status_code,
        "error": error,
        "text": cleaned_text,
        "visible_text": clean_lines(visible_text, keep_duplicates=keep_duplicates),
        "sections": extract_sections_from_html(html) if html else [],
        "links": all_links,
        "internal_links": internal_links,
        "download_links": download_links,
        "forms": forms,
        "possible_login_page": is_possible_login_page(final_url, title, visible_text, forms),
        "scrape_seconds": round(time.perf_counter() - started, 3),
    }


async def detect_homepage_links(start_url: str, include_header_footer: bool = True, keep_duplicates: bool = True) -> list[dict]:
    page_data = await scrape_page(start_url, include_header_footer, keep_duplicates)
    current = {
        "text": f"Current page: {page_data.get('final_url') or start_url}",
        "url": page_data.get("final_url") or start_url,
        "same_domain": True,
        "is_download": False,
    }
    return dedupe_links([current] + page_data.get("internal_links", []))


# ── Export helpers ─────────────────────────────────────────────────────────────
def build_markdown(results: list[dict]) -> str:
    parts = []
    for r in results:
        parts.append(f"# {r.get('title') or 'Untitled page'}")
        parts.append(f"Input URL: {r.get('input_url', '')}")
        parts.append(f"Final URL: {r.get('final_url', '')}")
        if r.get("status_code"):
            parts.append(f"Status: {r.get('status_code')}")
        parts.append("")
        parts.append(r.get("text", ""))
        parts.append("")
    return "\n".join(parts)


def build_csv(results: list[dict]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["title", "input_url", "final_url", "status_code", "text"])
    writer.writeheader()
    for r in results:
        writer.writerow({
            "title": r.get("title", ""),
            "input_url": r.get("input_url", ""),
            "final_url": r.get("final_url", ""),
            "status_code": r.get("status_code", ""),
            "text": r.get("text", ""),
        })
    return output.getvalue()


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours}h {minutes}m {remaining_seconds:.2f}s"
    if minutes:
        return f"{minutes}m {remaining_seconds:.2f}s"
    return f"{seconds:.2f}s"


# ── Cost estimator ─────────────────────────────────────────────────────────────
def estimate_openai_cost(model: str, usage) -> dict:
    pricing = OPENAI_PRICING.get(model)
    pricing_model = model
    if pricing is None:
        pricing = OPENAI_PRICING[DEFAULT_PRICING_MODEL]
        pricing_model = f"{DEFAULT_PRICING_MODEL} (fallback)"

    input_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", 0) or 0
    details = getattr(usage, "prompt_tokens_details", None)
    cached_input = getattr(details, "cached_tokens", 0) or 0
    normal_input = max(input_tokens - cached_input, 0)

    input_cost = (normal_input / 1_000_000) * pricing["input"]
    cached_cost = (cached_input / 1_000_000) * pricing["cached_input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + cached_cost + output_cost

    return {
        "model_used": model,
        "pricing_model": pricing_model,
        "input_tokens": input_tokens,
        "normal_input_tokens": normal_input,
        "cached_input_tokens": cached_input,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_cost_usd": round(input_cost, 8),
        "cached_input_cost_usd": round(cached_cost, 8),
        "output_cost_usd": round(output_cost, 8),
        "estimated_total_cost_usd": round(total_cost, 8),
        "pricing_usd_per_1m_tokens": pricing,
    }


def _sum_numeric(values: list) -> float | None:
    numeric_values = [v for v in values if isinstance(v, (int, float))]
    if not numeric_values:
        return None
    return round(sum(numeric_values), 8)


def combine_usage(usages: list[dict | None]) -> dict | None:
    valid = [u for u in usages if isinstance(u, dict)]
    if not valid:
        return None

    cost_total = _sum_numeric([u.get("estimated_total_cost_usd") for u in valid])

    return {
        "provider": valid[-1].get("provider", ""),
        "model_used": valid[-1].get("model_used", ""),
        "pricing_model": valid[-1].get("pricing_model", ""),
        "input_tokens": sum(u.get("input_tokens", 0) or 0 for u in valid),
        "normal_input_tokens": sum(u.get("normal_input_tokens", 0) or 0 for u in valid),
        "cached_input_tokens": sum(u.get("cached_input_tokens", 0) or 0 for u in valid),
        "output_tokens": sum(u.get("output_tokens", 0) or 0 for u in valid),
        "total_tokens": sum(u.get("total_tokens", 0) or 0 for u in valid),
        "input_cost_usd": _sum_numeric([u.get("input_cost_usd") for u in valid]),
        "cached_input_cost_usd": _sum_numeric([u.get("cached_input_cost_usd") for u in valid]),
        "output_cost_usd": _sum_numeric([u.get("output_cost_usd") for u in valid]),
        "estimated_total_cost_usd": cost_total,
        "pricing_usd_per_1m_tokens": valid[-1].get("pricing_usd_per_1m_tokens", {}),
    }


# ── OpenAI schemas/helpers ────────────────────────────────────────────────────
LINK_SELECTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "selected_links": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "url": {"type": "string"},
                    "label": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["url", "label", "reason"],
            },
        },
        "rationale": {"type": "string"},
    },
    "required": ["selected_links", "rationale"],
}

EXTRACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "task": {"type": "string"},
        "overall_summary": {"type": "string"},
        "extracted_records": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "category": {"type": "string"},
                    "name": {"type": "string"},
                    "value": {"type": "string"},
                    "details": {"type": "array", "items": {"type": "string"}},
                    "source_url": {"type": "string"},
                    "source_quote": {"type": "string"},
                },
                "required": ["category", "name", "value", "details", "source_url", "source_quote"],
            },
        },
        "pages": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "summary": {"type": "string"},
                    "important_points": {"type": "array", "items": {"type": "string"}},
                    "important_links": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "text": {"type": "string"},
                                "url": {"type": "string"},
                            },
                            "required": ["text", "url"],
                        },
                    },
                },
                "required": ["title", "url", "summary", "important_points", "important_links"],
            },
        },
        "download_links": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "text": {"type": "string"},
                    "url": {"type": "string"},
                    "source_url": {"type": "string"},
                },
                "required": ["text", "url", "source_url"],
            },
        },
        "forms_detected": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "page_url": {"type": "string"},
                    "purpose": {"type": "string"},
                    "fields": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["page_url", "purpose", "fields"],
            },
        },
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "task", "overall_summary", "extracted_records", "pages",
        "download_links", "forms_detected", "warnings",
    ],
}


def extract_json_from_text(output_text: str) -> dict:
    """Parse JSON from a model response. Handles plain JSON and simple fenced output."""
    text = (output_text or "").strip()
    if not text:
        return {"raw_output": ""}

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback for responses that accidentally include markdown fences or extra text.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return {"raw_output": output_text}


def openai_json_call(api_key: str, model: str, messages: list[dict], schema_name: str, schema: dict) -> dict:
    try:
        from openai import OpenAI
    except ImportError:
        return {"error": "openai package not installed. Run: pip install openai"}

    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                },
            },
        )
    except Exception as exc:
        return {"error": str(exc)}

    output_text = response.choices[0].message.content if response.choices else ""
    usage = getattr(response, "usage", None)
    cost = estimate_openai_cost(model, usage) if usage else None
    if cost:
        cost["provider"] = "OpenAI"

    if not output_text:
        return {"error": "OpenAI returned an empty response.", "usage": cost}

    return {"data": extract_json_from_text(output_text), "usage": cost}

def estimate_gemini_usage(model: str, usage_metadata) -> dict:
    pricing = GEMINI_PRICING.get(model)
    pricing_model = model

    if pricing is None:
        pricing = GEMINI_PRICING[DEFAULT_GEMINI_PRICING_MODEL]
        pricing_model = f"{DEFAULT_GEMINI_PRICING_MODEL} (fallback)"

    input_tokens = getattr(usage_metadata, "prompt_token_count", 0) or 0
    cached_input = getattr(usage_metadata, "cached_content_token_count", 0) or 0
    output_tokens = getattr(usage_metadata, "candidates_token_count", 0) or 0
    total_tokens = getattr(usage_metadata, "total_token_count", input_tokens + output_tokens) or (input_tokens + output_tokens)

    # Gemini output pricing includes thinking tokens.
    # If thoughts_token_count is not returned, estimate hidden thinking tokens from total_tokens.
    thinking_tokens = getattr(usage_metadata, "thoughts_token_count", 0) or 0
    if thinking_tokens == 0:
        thinking_tokens = max(total_tokens - input_tokens - output_tokens, 0)

    normal_input = max(input_tokens - cached_input, 0)
    billable_output_tokens = output_tokens + thinking_tokens

    input_cost = (normal_input / 1_000_000) * pricing["input"]
    cached_cost = (cached_input / 1_000_000) * pricing["cached_input"]
    output_cost = (billable_output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + cached_cost + output_cost

    return {
        "provider": "Gemini",
        "model_used": model,
        "pricing_model": pricing_model,
        "input_tokens": input_tokens,
        "normal_input_tokens": normal_input,
        "cached_input_tokens": cached_input,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "billable_output_tokens": billable_output_tokens,
        "total_tokens": total_tokens,
        "input_cost_usd": round(input_cost, 8),
        "cached_input_cost_usd": round(cached_cost, 8),
        "output_cost_usd": round(output_cost, 8),
        "estimated_total_cost_usd": round(total_cost, 8),
        "pricing_usd_per_1m_tokens": pricing,
    }


def gemini_json_call(api_key: str, model: str, messages: list[dict], schema_name: str, schema: dict) -> dict:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return {"error": "google-genai package not installed. Run: pip install google-genai"}

    client = genai.Client(api_key=api_key)

    prompt_parts = [
        "Return valid JSON only. Do not include markdown, comments, or explanation.",
        f"The JSON must match this schema named {schema_name}:",
        json.dumps(schema, ensure_ascii=False),
        "Conversation:",
    ]
    for message in messages:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        prompt_parts.append(f"{role}:\n{content}")
    prompt = "\n\n".join(prompt_parts)

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
    except Exception as exc:
        return {"error": str(exc)}

    output_text = getattr(response, "text", "") or ""
    usage_metadata = getattr(response, "usage_metadata", None)
    usage = estimate_gemini_usage(model, usage_metadata) if usage_metadata else {
    "provider": "Gemini",
    "model_used": model,
    "pricing_model": f"{DEFAULT_GEMINI_PRICING_MODEL} (fallback)",
    "input_tokens": 0,
    "normal_input_tokens": 0,
    "cached_input_tokens": 0,
    "output_tokens": 0,
    "thinking_tokens": 0,
    "billable_output_tokens": 0,
    "total_tokens": 0,
    "input_cost_usd": 0,
    "cached_input_cost_usd": 0,
    "output_cost_usd": 0,
    "estimated_total_cost_usd": 0,
    "pricing_usd_per_1m_tokens": GEMINI_PRICING[DEFAULT_GEMINI_PRICING_MODEL],
}
    

    if not output_text:
        return {"error": "Gemini returned an empty response.", "usage": usage}

    return {"data": extract_json_from_text(output_text), "usage": usage}


def ai_json_call(provider: str, api_key: str, model: str, messages: list[dict], schema_name: str, schema: dict) -> dict:
    if provider == "Gemini":
        return gemini_json_call(api_key, model, messages, schema_name, schema)
    return openai_json_call(api_key, model, messages, schema_name, schema)


def ai_choose_next_links(
    start_url: str,
    task: str,
    visited_urls: set[str],
    candidate_links: list[dict],
    api_key: str,
    model: str,
    max_new_links: int,
    provider: str = "OpenAI",
) -> dict:
    cleaned_candidates = []
    seen = set()
    for link in candidate_links:
        link_url = link.get("url", "")
        if not link_url or link_url in visited_urls or link_url in seen:
            continue
        if not same_domain(start_url, link_url) or not is_scrapable_page_url(link_url):
            continue
        cleaned_candidates.append({"text": link.get("text", "Untitled link"), "url": link_url})
        seen.add(link_url)
        if len(cleaned_candidates) >= 120:
            break

    if not cleaned_candidates:
        return {"data": {"selected_links": [], "rationale": "No valid unvisited same-domain links."}, "usage": None}

    messages = [
        {
            "role": "system",
            "content": (
                "You are the link-selection brain for a public website scraper. "
                "Choose same-domain public pages that should be visited next for the user's scraping goal. "
                "Select only URLs from the candidate list. Do not invent URLs. "
                "Prefer pages likely to contain useful public content. Avoid pages whose only purpose is credentials, "
                "private account access, tokens, cookies, or secrets."
            ),
        },
        {
            "role": "user",
            "content": json.dumps({
                "start_url": start_url,
                "user_task": task,
                "already_visited_urls": sorted(visited_urls),
                "max_new_links": max_new_links,
                "candidate_links": cleaned_candidates,
            }, ensure_ascii=False),
        },
    ]

    result = ai_json_call(provider, api_key, model, messages, "next_link_selection", LINK_SELECTION_SCHEMA)
    if "error" in result:
        return result

    allowed = {link["url"] for link in cleaned_candidates}
    selected = []
    selected_seen = set()
    for item in result["data"].get("selected_links", []):
        link_url = item.get("url", "")
        if link_url in allowed and link_url not in selected_seen:
            selected.append(item)
            selected_seen.add(link_url)
        if len(selected) >= max_new_links:
            break

    result["data"]["selected_links"] = selected
    return result


def ai_extract_from_scraped_pages(results: list[dict], task: str, api_key: str, model: str, provider: str = "OpenAI") -> dict:
    compact_pages = []
    all_downloads = []

    for r in results:
        compact_pages.append({
            "title": r.get("title", ""),
            "input_url": r.get("input_url", ""),
            "final_url": r.get("final_url", ""),
            "status_code": r.get("status_code", ""),
            "text": r.get("text", "")[:25_000],
            "sections": r.get("sections", [])[:80],
            "links": r.get("links", [])[:100],
            "forms": r.get("forms", []),
            "possible_login_page": r.get("possible_login_page", False),
        })
        for link in r.get("download_links", []):
            all_downloads.append({
                "text": link.get("text", ""),
                "url": link.get("url", ""),
                "source_url": r.get("final_url", ""),
            })

    messages = [
        {
            "role": "system",
            "content": (
                "You are the AI extraction layer for a public website scraper. "
                "Extract only information present in the provided scraped pages, sections, links, and public form metadata. "
                "Follow the user's extraction goal. Do not invent missing content. "
                "Never extract, infer, request, or expose actual usernames, passwords, API keys, tokens, "
                "session cookies, or private account data. For forms, describe only public form structure and purpose. "
                "Do not suggest submitting forms or bypassing login."
            ),
        },
        {
            "role": "user",
            "content": json.dumps({
                "user_task": task,
                "scraped_pages": compact_pages,
                "download_links_seen_by_browser": all_downloads[:150],
            }, ensure_ascii=False),
        },
    ]

    return ai_json_call(provider, api_key, model, messages, "ai_agent_extraction", EXTRACTION_SCHEMA)


# ── Crawl modes ────────────────────────────────────────────────────────────────
def collect_unvisited_candidates(start_url: str, results: list[dict], visited_urls: set[str]) -> list[dict]:
    candidates = []
    seen = set()
    for r in results:
        for link in r.get("internal_links", []):
            link_url = link.get("url", "")
            if not link_url or link_url in visited_urls or link_url in seen:
                continue
            if same_domain(start_url, link_url) and is_scrapable_page_url(link_url):
                candidates.append({"text": link.get("text", "Untitled link"), "url": link_url})
                seen.add(link_url)
    return candidates


async def scrape_selected_urls(
    urls: list[str],
    include_header_footer: bool,
    keep_duplicates: bool,
    progress=None,
    status=None,
) -> list[dict]:
    results = []
    total = len(urls)
    for idx, page_url in enumerate(urls, start=1):
        if status:
            status.write(f"Scraping {idx}/{total}: {page_url}")
        page_data = await scrape_page(page_url, include_header_footer, keep_duplicates)
        page_data["crawl_depth"] = 0
        page_data["manual_selected"] = True
        results.append(page_data)
        if progress:
            progress.progress(idx / total)
    return results


async def ai_agent_crawl(
    start_url: str,
    task: str,
    api_key: str,
    model: str,
    provider: str,
    max_pages: int,
    max_depth: int,
    max_ai_links_per_round: int,
    include_header_footer: bool,
    keep_duplicates: bool,
) -> dict:
    results = []
    selections = []
    usages = []
    visited_urls = set()

    first = await scrape_page(start_url, include_header_footer, keep_duplicates)
    first_url = first.get("final_url") or start_url
    visited_urls.add(first_url)
    visited_urls.add(start_url)
    first["crawl_depth"] = 0
    first["ai_selection_reason"] = "Starting URL supplied by user."
    results.append(first)

    for depth in range(1, max_depth + 1):
        if len(results) >= max_pages:
            break
        candidates = collect_unvisited_candidates(start_url, results, visited_urls)
        if not candidates:
            break

        max_new = min(max_ai_links_per_round, max_pages - len(results))
        selection = ai_choose_next_links(start_url, task, visited_urls, candidates, api_key, model, max_new, provider)    
        selections.append(selection)
        if selection.get("usage"):
            usages.append(selection["usage"])
        if selection.get("error"):
            return {"results": results, "selections": selections, "usages": usages, "error": selection["error"]}

        selected_links = selection.get("data", {}).get("selected_links", [])

        # Fallback: if AI returns no valid URL, crawl the first few same-domain candidates.
        if not selected_links:
            selected_links = [
                {
                    "url": link["url"],
                    "label": link.get("text", "Untitled link"),
                    "reason": "Fallback because AI selected no valid link.",
                }
                for link in candidates[:max_new]
            ]

        for selected in selected_links:
            if len(results) >= max_pages:
                break
            page_url = selected.get("url", "")
            if not page_url or page_url in visited_urls:
                continue
            visited_urls.add(page_url)
            page_data = await scrape_page(page_url, include_header_footer, keep_duplicates)
            visited_urls.add(page_data.get("final_url") or page_url)
            page_data["crawl_depth"] = depth
            page_data["ai_selection_reason"] = selected.get("reason", "")
            results.append(page_data)

    extraction = ai_extract_from_scraped_pages(results, task, api_key, model, provider)
    if extraction.get("usage"):
        usages.append(extraction["usage"])

    return {
        "results": results,
        "selections": selections,
        "extraction": extraction,
        "usage_total": combine_usage(usages),
        "error": extraction.get("error"),
    }


async def whole_site_crawl(
    start_url: str,
    max_pages: int,
    max_depth: int,
    include_header_footer: bool,
    keep_duplicates: bool,
) -> list[dict]:
    results = []
    visited = set()
    queue = deque([(start_url, 0)])

    while queue and len(results) < max_pages:
        page_url, depth = queue.popleft()
        if page_url in visited or not same_domain(start_url, page_url) or not is_scrapable_page_url(page_url):
            continue
        visited.add(page_url)

        page_data = await scrape_page(page_url, include_header_footer, keep_duplicates)
        final_url = page_data.get("final_url") or page_url
        visited.add(final_url)
        page_data["crawl_depth"] = depth
        results.append(page_data)

        if depth < max_depth:
            for link in page_data.get("internal_links", []):
                next_url = link.get("url", "")
                if next_url and next_url not in visited and same_domain(start_url, next_url) and is_scrapable_page_url(next_url):
                    queue.append((next_url, depth + 1))

    return results


# ── UI helpers ────────────────────────────────────────────────────────────────
def render_usage_panel(usage: dict):
    st.subheader("📊 API Usage & Estimated Cost")
    cost = usage.get("estimated_total_cost_usd")
    cost_display = f"${cost:.8f}" if isinstance(cost, (int, float)) else "N/A"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("AI Provider", usage.get("provider") or "N/A")
    c2.metric("💰 Est. Cost (USD)", cost_display)
    c3.metric("📥 Input Tokens", f"{usage.get('input_tokens', 0):,}")
    c4.metric("📤 Output Tokens", f"{usage.get('output_tokens', 0):,}")
    c5.metric("🔢 Total Tokens", f"{usage.get('total_tokens', 0):,}")

    

    with st.expander("🔍 Full cost breakdown"):
        st.json(usage)


def parse_extra_urls(base_url: str, pasted_text: str) -> list[str]:
    urls = []
    for raw in pasted_text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        normalized = normalize_url(base_url, raw)
        if normalized and same_domain(base_url, normalized) and is_scrapable_page_url(normalized):
            urls.append(normalized)
    return dedupe_urls(urls)


# ── Session state ──────────────────────────────────────────────────────────────
for _k, _v in [
    ("detected_links", []),
    ("scraped_results", []),
    ("ai_selections", []),
    ("ai_output", None),
    ("ai_usage_total", None),
    ("elapsed_seconds", None),
    ("run_mode_used", ""),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    crawl_mode = st.selectbox(
        "Scraping mode",
        [
            "Manual selected links + AI extraction",
            "Manual selected links only",
            "AI follows links step-by-step",
            "Whole website crawl + AI extraction",
        ],
    )
    max_pages = st.number_input("Maximum pages to scrape", 1, 50, 15)
    max_depth = st.number_input("Maximum crawl depth", 0, 5, 2)
    max_ai_links_per_round = st.number_input("AI links per round", 1, 10, 4)
    include_header_footer = st.checkbox("Include header and footer", value=True)
    keep_duplicates = st.checkbox("Keep repeated text", value=True)

    st.divider()
    st.subheader("AI Provider")
    ai_provider = st.selectbox("Provider", ["OpenAI", "Gemini"])

    openai_api_key = get_secret("OPENAI_API_KEY")
    gemini_api_key = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")

    if ai_provider == "OpenAI":
        ai_api_key = openai_api_key
        ai_model = st.selectbox(
            "OpenAI model",
            list(OPENAI_PRICING.keys()) + ["other"],
            index=list(OPENAI_PRICING.keys()).index("gpt-4o-mini"),
        )
        if ai_model == "other":
            ai_model = st.text_input("Custom OpenAI model name", placeholder="gpt-5")
    else:
        ai_api_key = gemini_api_key
        ai_model = st.selectbox(
            "Gemini model",
            GEMINI_MODELS + ["other"],
            index=0,
        )
        if ai_model == "other":
            ai_model = st.text_input("Custom Gemini model name", placeholder="gemini-2.5-flash")

    st.caption("API keys are loaded silently from environment variables or .env and are not displayed on the page.")

    st.divider()
    output_mode = st.selectbox(
        "Display output",
        [
            "AI Result",
            "Full text",
            "Visible browser text",
            "Sections JSON",
            "Forms / Login Detection",
            "Links JSON",
        ],
    )


# ── Main inputs ────────────────────────────────────────────────────────────────
url = st.text_input("Enter website URL", placeholder="https://example.com")
ai_task = st.text_area(
    "Prompt AI: what do you want to scrape?",
    value=(
        "Scrape the selected pages and extract company overview, services/products, important pages, "
        "contact details, downloadable PDFs, public form structure, and useful links."
    ),
    height=110,
)

col1, col2, col3 = st.columns(3)
with col1:
    detect_clicked = st.button("🔍 Detect Links", use_container_width=True)
with col2:
    run_clicked = st.button("🚀 Run Scraper", use_container_width=True)
with col3:
    clear_clicked = st.button("🗑 Clear Results", use_container_width=True)

if clear_clicked:
    st.session_state.detected_links = []
    st.session_state.scraped_results = []
    st.session_state.ai_selections = []
    st.session_state.ai_output = None
    st.session_state.ai_usage_total = None
    st.session_state.elapsed_seconds = None
    st.session_state.run_mode_used = ""

if detect_clicked:
    if not url:
        st.error("Please enter a website URL first.")
    else:
        with st.spinner("Detecting same-domain homepage links..."):
            st.session_state.detected_links = asyncio.run(
                detect_homepage_links(url, include_header_footer, keep_duplicates)
            )
            st.success(f"Detected {len(st.session_state.detected_links)} same-domain links.")


# ── Manual link selection UI ──────────────────────────────────────────────────
manual_modes = {
    "Manual selected links + AI extraction",
    "Manual selected links only",
}

selected_detected_labels = []
extra_urls_text = ""
selected_urls_for_preview = []

if crawl_mode in manual_modes:
    st.subheader("Manual Link Selection")

    if st.session_state.detected_links:
        link_options = {
            f"{item.get('text', 'Untitled link')} — {item.get('url', '')}": item.get("url", "")
            for item in st.session_state.detected_links
        }

        selected_detected_labels = st.multiselect(
            "Choose one or more detected links to scrape",
            list(link_options.keys()),
            max_selections=int(max_pages),
            help="You can select two or more pages here. The app will scrape all selected pages.",
        )

        selected_urls_for_preview.extend([link_options[label] for label in selected_detected_labels])
    else:
        st.info("Click Detect Links first, or paste extra URLs below. If you run now, the entered website URL will be scraped.")

    extra_urls_text = st.text_area(
        "Optional: add extra same-domain URLs manually, one per line",
        height=90,
        placeholder="https://example.com/about\nhttps://example.com/contact",
    )

    if url:
        selected_urls_for_preview.extend(parse_extra_urls(url, extra_urls_text))
        selected_urls_for_preview = dedupe_urls(selected_urls_for_preview)[: int(max_pages)]

    if selected_urls_for_preview:
        with st.expander(f"Selected URLs preview ({len(selected_urls_for_preview)})", expanded=False):
            st.json(selected_urls_for_preview)


# ── Run logic ─────────────────────────────────────────────────────────────────
if run_clicked:
    if not url:
        st.error("Please enter a website URL.")
    elif crawl_mode != "Manual selected links only" and not ai_api_key:
        st.error(f"{ai_provider} API key not found. Add the correct key to your .env file, Streamlit secrets, or deployment environment variables.")
    elif crawl_mode != "Manual selected links only" and not ai_task.strip():
        st.error("Please describe what you want AI to scrape/extract.")
    else:
        progress = st.progress(0)
        status = st.empty()
        started = time.perf_counter()

        st.session_state.scraped_results = []
        st.session_state.ai_selections = []
        st.session_state.ai_output = None
        st.session_state.ai_usage_total = None
        st.session_state.run_mode_used = crawl_mode

        if crawl_mode in manual_modes:
            selected_urls = []

            if st.session_state.detected_links:
                link_options = {
                    f"{item.get('text', 'Untitled link')} — {item.get('url', '')}": item.get("url", "")
                    for item in st.session_state.detected_links
                }
                selected_urls.extend([link_options[label] for label in selected_detected_labels if label in link_options])

            selected_urls.extend(parse_extra_urls(url, extra_urls_text))

            # If user does not select anything, scrape the main URL as a fallback.
            if not selected_urls:
                selected_urls = [url]

            selected_urls = dedupe_urls(selected_urls)[: int(max_pages)]

            status.write(f"Manual mode: scraping {len(selected_urls)} selected URL(s)...")
            results = asyncio.run(scrape_selected_urls(
                urls=selected_urls,
                include_header_footer=include_header_footer,
                keep_duplicates=keep_duplicates,
                progress=progress,
                status=status,
            ))
            st.session_state.scraped_results = results

            if crawl_mode == "Manual selected links + AI extraction":
                progress.progress(0.85)
                status.write(f"Asking {ai_provider} to extract structured JSON from your manually selected pages...")
                extraction = ai_extract_from_scraped_pages(results, ai_task, ai_api_key, ai_model, ai_provider)
                if extraction.get("error"):
                    st.error(extraction["error"])
                st.session_state.ai_output = extraction
                st.session_state.ai_usage_total = combine_usage([extraction.get("usage")])
                progress.progress(1.0)

        elif crawl_mode == "AI follows links step-by-step":
            status.write(f"{ai_provider} mode: scraping start page, then AI chooses same-domain links step-by-step...")
            agent_result = asyncio.run(ai_agent_crawl(
                start_url=url,
                task=ai_task,
                api_key=ai_api_key,
                model=ai_model,
                provider=ai_provider,
                max_pages=int(max_pages),
                max_depth=int(max_depth),
                max_ai_links_per_round=int(max_ai_links_per_round),
                include_header_footer=include_header_footer,
                keep_duplicates=keep_duplicates,
            ))
            progress.progress(1.0)
            if agent_result.get("error"):
                st.error(agent_result["error"])
            st.session_state.scraped_results = agent_result.get("results", [])
            st.session_state.ai_selections = agent_result.get("selections", [])
            st.session_state.ai_output = agent_result.get("extraction")
            st.session_state.ai_usage_total = agent_result.get("usage_total")

        elif crawl_mode == "Whole website crawl + AI extraction":
            status.write("Whole website mode: crawling same-domain pages up to your page/depth limit...")
            results = asyncio.run(whole_site_crawl(
                start_url=url,
                max_pages=int(max_pages),
                max_depth=int(max_depth),
                include_header_footer=include_header_footer,
                keep_duplicates=keep_duplicates,
            ))
            progress.progress(0.70)
            status.write(f"Asking {ai_provider} to extract structured data from crawled pages...")
            extraction = ai_extract_from_scraped_pages(results, ai_task, ai_api_key, ai_model, ai_provider)
            progress.progress(1.0)
            if extraction.get("error"):
                st.error(extraction["error"])
            st.session_state.scraped_results = results
            st.session_state.ai_selections = []
            st.session_state.ai_output = extraction
            st.session_state.ai_usage_total = combine_usage([extraction.get("usage")])

        st.session_state.elapsed_seconds = time.perf_counter() - started
        status.write(f"Complete. Time used: {format_duration(st.session_state.elapsed_seconds)}")


# ── Detected links display ─────────────────────────────────────────────────────
if st.session_state.detected_links:
    with st.expander(f"Detected homepage links ({len(st.session_state.detected_links)})"):
        st.json(st.session_state.detected_links)


# ── Results ────────────────────────────────────────────────────────────────────
if st.session_state.scraped_results:
    st.subheader("📄 Results")
    c1, c2, c3 = st.columns(3)
    c1.metric("⏱ Time Used", format_duration(st.session_state.elapsed_seconds))
    c2.metric("Pages Scraped", len(st.session_state.scraped_results))
    c3.metric("Mode", st.session_state.run_mode_used or "N/A")

    if st.session_state.ai_usage_total:
        render_usage_panel(st.session_state.ai_usage_total)

    if st.session_state.ai_selections:
        with st.expander("🤖 AI link-selection rounds"):
            st.json(st.session_state.ai_selections)

    if output_mode == "AI Result":
        if st.session_state.ai_output and st.session_state.ai_output.get("data"):
            formatted = json.dumps(st.session_state.ai_output["data"], indent=2, ensure_ascii=False)
            st.subheader("🗂 AI Structured Output")
            st.code(formatted, language="json")
            with st.expander("Interactive JSON viewer"):
                st.json(st.session_state.ai_output["data"])
            st.download_button(
                "⬇ Download AI extracted JSON",
                data=formatted,
                file_name="ai_extracted_output.json",
                mime="application/json",
            )
        elif st.session_state.ai_output and st.session_state.ai_output.get("error"):
            st.error(st.session_state.ai_output["error"])
        else:
            st.info("No AI output for this run. Choose another display output to view raw scraped data.")
    else:
        for i, result in enumerate(st.session_state.scraped_results):
            st.markdown(f"### {i + 1}. {result.get('title') or 'Untitled page'}")
            if result.get("manual_selected"):
                st.info("This page was manually selected.")
            if result.get("ai_selection_reason"):
                st.info(f"AI selected this page because: {result['ai_selection_reason']}")
            if result.get("possible_login_page"):
                st.warning("Possible login page detected. Only public form structure is shown; no credentials are collected.")

            st.write("**Input URL:**", result.get("input_url", ""))
            st.write("**Final URL:**", result.get("final_url", ""))
            st.write("**Status:**", result.get("status_code", ""))
            st.write("**Depth:**", result.get("crawl_depth", 0))
            st.write("**Scrape seconds:**", result.get("scrape_seconds", ""))
            if result.get("error"):
                st.error(result["error"])

            if output_mode == "Full text":
                st.text_area("Full Page Text", value=result.get("text", ""), height=400, key=f"full_{i}")
            elif output_mode == "Visible browser text":
                st.text_area("Visible Browser Text", value=result.get("visible_text", ""), height=400, key=f"vis_{i}")
            elif output_mode == "Sections JSON":
                st.json(result.get("sections", []))
            elif output_mode == "Forms / Login Detection":
                st.json({
                    "possible_login_page": result.get("possible_login_page", False),
                    "forms": result.get("forms", []),
                })
            elif output_mode == "Links JSON":
                st.json({
                    "all_links": result.get("links", []),
                    "internal_links": result.get("internal_links", []),
                    "download_links": result.get("download_links", []),
                })
            st.divider()

    raw_json = json.dumps({
        "scraped_results": st.session_state.scraped_results,
        "ai_selections": st.session_state.ai_selections,
        "ai_output": st.session_state.ai_output,
        "ai_usage_total": st.session_state.ai_usage_total,
    }, indent=2, ensure_ascii=False)
    markdown_output = build_markdown(st.session_state.scraped_results)
    csv_output = build_csv(st.session_state.scraped_results)

    ca, cb, cc = st.columns(3)
    with ca:
        st.download_button("⬇ Markdown", data=markdown_output, file_name="scraped_output.md", mime="text/markdown")
    with cb:
        st.download_button("⬇ Full raw JSON", data=raw_json, file_name="full_scrape_report.json", mime="application/json")
    with cc:
        st.download_button("⬇ CSV", data=csv_output, file_name="scraped_output.csv", mime="text/csv")
