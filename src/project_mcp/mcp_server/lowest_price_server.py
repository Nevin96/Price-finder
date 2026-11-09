import os
import sys
import time
import re
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from mcp.server.fastmcp import FastMCP
import asyncio
from recommendation_engine import predict_buying_decision

# Ensure .env is loaded
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path) if env_path.exists() else load_dotenv()

mcp = FastMCP("lowest-price")

HISTORY_CACHE_TTL = 3600
_HISTORY_CACHE: dict[str, tuple[float, str]] = {}

PRICE_CACHE_TTL = 5 * 60
_PRICE_CACHE: dict[str, tuple[float, str]] = {}


# --- Logging Setup (Safe for MCP stdio) ---

def _build_logger() -> logging.Logger:
    logger = logging.getLogger("project_mcp.lowest_price")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", 
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Log to stderr so it does NOT corrupt MCP's stdout JSON-RPC stream
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    # Local file log for persistent inspection
    log_file = Path(__file__).resolve().parent / "price_finder.log"
    try:
        file_handler = RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as exc:
        logger.warning(f"Failed to initialize file logger: {exc}")

    return logger


LOGGER = _build_logger()


# --- Retailer & Parsing Filters ---

def detect_retailer(url: str) -> str | None:
    """Matches only approved retailers (Croma and Vijay Sales explicitly removed)."""
    u = url.lower()
    if "amazon.in" in u:
        return "Amazon"
    if "flipkart.com" in u:
        return "Flipkart"
    if "reliancedigital.in" in u:
        return "Reliance Digital"
    if "jiomart.com" in u:
        return "JioMart"
    return None


def is_retail_product_page(url: str) -> bool:
    """Filters out resource center, blog posts, and buying guides."""
    path = url.lower()
    non_product_paths = (
        "resource-center", "/unboxed/", "/blog", "/news", "/article", 
        "/reviews/", "/buying-guide/", "/category/", "/sections/"
    )
    return not any(part in path for part in non_product_paths)


def get_minimum_price_threshold(product_query: str) -> float:
    q = product_query.lower()
    if any(k in q for k in ["buds", "earbuds", "headphones", "headset", "earphones"]):
        return 500.0
    if "fold" in q:
        return 40000.0
    if "iphone" in q:
        return 20000.0
    if "galaxy s" in q or "pixel" in q:
        return 12000.0
    if any(k in q for k in ["macbook", "laptop"]):
        return 10000.0
    if "ipad" in q or "tablet" in q:
        return 5000.0
    return 1000.0


def extract_price(text: str, min_price: float) -> float | None:
    if not text:
        return None

    # Matches explicit currency amounts (₹82,900 or Rs. 82,900)
    explicit_matches = re.findall(r"(?:₹|rs\.?|inr)\s*([\d,]{4,8})", text, re.IGNORECASE)
    for m in explicit_matches:
        clean = m.replace(",", "").strip()
        try:
            val = float(clean)
            if min_price <= val <= 350000:
                return val
        except ValueError:
            continue

    # Matches comma-separated numbers (82,900 or 1,24,900)
    comma_matches = re.findall(r"\b([1-9]\d{0,2},\d{2},\d{3}|[1-9]\d{1,2},\d{3})\b", text)
    for m in comma_matches:
        clean = m.replace(",", "").strip()
        try:
            val = float(clean)
            if min_price <= val <= 350000:
                return val
        except ValueError:
            continue

    return None


def _normalize_variant_text(text: str) -> str:
    text = text.lower()
    return re.sub(r"(?<=\d)\s+(?=(?:gb|tb|mb)\b)", "", text)


def is_correct_product(title: str, url: str, query: str, description: str = "") -> bool:
    title_lower = title.lower()
    url_lower = url.lower()
    query_lower = query.lower()
    searchable_text = _normalize_variant_text(f"{title_lower} {description.lower()} {url_lower}")

    # 1. Block accessories
    bad_keywords = [
        "case", "cover", "accessory", "accessories", "screen-protector",
        "tempered-glass", "cable", "charger", "back-cover", "skin", "guard",
        "pouch", "strap", "stand", "mount", "adapter", "warranty"
    ]
    if any(k in url_lower for k in bad_keywords) or any(re.search(rf"\b{re.escape(k)}\b", title_lower) for k in bad_keywords):
        return False

    # 2. Token match: Ignore generic shopping terms
    ignore = {"buy", "price", "in", "india", "online", "deal", "lowest", "phone", "mobile", "gb", "tb"}
    tokens = [w for w in re.findall(r"[a-z0-9]+", _normalize_variant_text(query_lower)) if w not in ignore]

    for t in tokens:
        if t not in searchable_text:
            return False

    # 3. Model tier validation for iPhones
    if "iphone" in query_lower:
        tiers = ["pro max", "pro", "max", "plus"]
        for tier in tiers:
            if tier in title_lower and tier not in query_lower:
                return False

    return True


def _search_items(search_res) -> list:
    items = getattr(search_res, "web", None) or getattr(search_res, "data", None) or []
    if isinstance(search_res, dict):
        items = search_res.get("web") or search_res.get("data") or items
    return items if isinstance(items, list) else []


# --- MCP Tools ---

@mcp.tool()
async def lowest_price(product: str) -> str:
    """Find the lowest trusted-store price for a product in India across Amazon, Flipkart, Reliance Digital, and JioMart.
    
    Returns a JSON string matching the fallback protocol:
      - status: 'found' | 'variant_unavailable' | 'no_match' | 'search_error'
    """
    started = time.perf_counter()
    LOGGER.info(f"==> Tool called: lowest_price for query: '{product}'")

    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        LOGGER.error("FIRECRAWL_API_KEY is not set.")
        return json.dumps({
            "status": "search_error",
            "message": "FIRECRAWL_API_KEY is missing."
        })

    cache_key = re.sub(r"[^a-z0-9]+", " ", product.lower()).strip()
    hit = _PRICE_CACHE.get(cache_key)
    if hit and time.time() - hit[0] < PRICE_CACHE_TTL:
        LOGGER.info(f"Cache hit for key: '{cache_key}'")
        return hit[1]

    firecrawl = FirecrawlApp(api_key=api_key)
    min_price = get_minimum_price_threshold(product)
    
    # Exclude both Croma and Vijay Sales from search
    query = f"{product} buy online price India Amazon Flipkart Reliance Digital JioMart"
    LOGGER.info(f"Executing Firecrawl search with query: '{query}'")

    try:
        search_res = await asyncio.wait_for(
            asyncio.to_thread(lambda: firecrawl.search(query=query, limit=10)),
            timeout=8.0
        )
        items = _search_items(search_res)
        LOGGER.info(f"Firecrawl returned {len(items)} items in {time.perf_counter() - started:.2f}s")
    except asyncio.TimeoutError:
        LOGGER.warning("Firecrawl search exceeded timeout (8.0s).")
        return json.dumps({
            "status": "search_error",
            "message": "Search timed out. Retail servers did not respond in time."
        })
    except Exception as exc:
        LOGGER.error(f"Firecrawl search error: {exc}")
        return json.dumps({
            "status": "search_error",
            "message": str(exc)
        })

    matches = []
    other_variants_found = set()
    general_alternatives = set()

    base_product = re.sub(r"\b\d+\s*(?:gb|tb)\b", "", product, flags=re.I).strip()
    has_specific_variant = base_product.lower() != product.strip().lower()

    for idx, item in enumerate(items):
        title = (item.get("title") if isinstance(item, dict) else getattr(item, "title", "")) or ""
        url = (item.get("url") if isinstance(item, dict) else getattr(item, "url", "")) or ""
        desc = (item.get("description") if isinstance(item, dict) else getattr(item, "description", "")) or ""

        retailer = detect_retailer(url)
        if not retailer:
            LOGGER.debug(f"Item #{idx} skipped: retailer excluded or unverified ({url})")
            continue

        if not is_retail_product_page(url):
            LOGGER.info(f"Item #{idx} skipped: non-retail page ({url})")
            continue

        price = extract_price(desc, min_price) or extract_price(title, min_price)

        if is_correct_product(title, url, product, desc):
            if price:
                LOGGER.info(f"Match found: [{retailer}] ₹{price:,.0f} -> '{title}'")
                matches.append({
                    "retailer": retailer,
                    "price": price,
                    "display_price": f"₹{price:,.0f}",
                    "link": url,
                    "title": title
                })
        else:
            if is_correct_product(title, url, base_product, desc):
                var_match = re.findall(r"\b(128\s*gb|256\s*gb|512\s*gb|1\s*tb)\b", f"{title} {desc}", re.I)
                for v in var_match:
                    other_variants_found.add(v.upper().replace(" ", ""))
                if price:
                    clean_title = title.split("|")[0].split("-")[0].strip()
                    general_alternatives.add(f"{clean_title} ({retailer} at ₹{price:,.0f})")
            else:
                if price:
                    clean_title = title.split("|")[0].split("-")[0].strip()
                    if clean_title:
                        general_alternatives.add(clean_title)

    # Status: found
    if matches:
        best_per_store = {}
        for m in matches:
            cur = best_per_store.get(m["retailer"])
            if cur is None or m["price"] < cur["price"]:
                best_per_store[m["retailer"]] = m

        ranked = sorted(best_per_store.values(), key=lambda x: x["price"])
        best = ranked[0]

        result_payload = {
            "status": "found",
            "best_store": best["retailer"],
            "lowest_price": best["display_price"],
            "link": best["link"],
            "other_stores": [
                {"store": m["retailer"], "price": m["display_price"], "link": m["link"]}
                for m in ranked[1:]
            ]
        }
        res_str = json.dumps(result_payload)
        _PRICE_CACHE[cache_key] = (time.time(), res_str)
        LOGGER.info(f"Tool finished: status=found, best={best['retailer']} {best['display_price']}")
        return res_str

    # Status: variant_unavailable
    if has_specific_variant and other_variants_found:
        result_payload = {
            "status": "variant_unavailable",
            "requested_product": product,
            "base_product": base_product,
            "available_variants": sorted(list(other_variants_found)),
            "closest_variant": sorted(list(other_variants_found))[0]
        }
        LOGGER.info(f"Tool finished: status=variant_unavailable, variants={list(other_variants_found)}")
        return json.dumps(result_payload)

    # Status: no_match
    result_payload = {
        "status": "no_match",
        "product": product,
        "alternatives": list(general_alternatives)[:3]
    }
    LOGGER.info(f"Tool finished: status=no_match, alternatives={len(general_alternatives)}")
    return json.dumps(result_payload)


@mcp.tool()
async def get_price_history(product: str) -> str:
    """Get historical all-time low, high, and drop trends for a gadget in India."""
    started = time.perf_counter()
    LOGGER.info(f"==> Tool called: get_price_history for query: '{product}'")

    cache_key = product.lower().strip()
    if cache_key in _HISTORY_CACHE:
        timestamp, cached_data = _HISTORY_CACHE[cache_key]
        if time.time() - timestamp < HISTORY_CACHE_TTL:
            LOGGER.info(f"Cache hit for price history: '{cache_key}'")
            return cached_data

    api_key = os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        return "Error: FIRECRAWL_API_KEY is missing."

    firecrawl = FirecrawlApp(api_key=api_key)
    query = f"{product} price history all time low site:pricehistoryapp.com OR site:pricebefore.com"

    try:
        search_res = await asyncio.wait_for(
            asyncio.to_thread(lambda: firecrawl.search(query=query, limit=3)),
            timeout=8.0
        )
        items = _search_items(search_res)
    except Exception as exc:
        LOGGER.error(f"Firecrawl history search failed: {exc}")
        return f"Firecrawl history search error: {exc}"

    history_records = []
    low_val = None
    avg_val = None

    for item in items:
        title = (item.get("title") if isinstance(item, dict) else getattr(item, "title", "")) or ""
        desc = (item.get("description") if isinstance(item, dict) else getattr(item, "description", "")) or ""
        url = (item.get("url") if isinstance(item, dict) else getattr(item, "url", "")) or ""

        if not is_correct_product(title, url, product, desc):
            continue

        low_match = re.search(r"(?:lowest|all[\s-]time low|drop to)\s*(?:is|was|of)?\s*(?:₹|rs\.?|inr)?\s*([\d,]{4,8})", f"{title} {desc}", re.I)
        if low_match and not low_val:
            try:
                low_val = float(low_match.group(1).replace(",", "").strip())
            except ValueError:
                pass

        avg_match = re.search(r"(?:average|typical)\s*(?:is|was|of)?\s*(?:₹|rs\.?|inr)?\s*([\d,]{4,8})", f"{title} {desc}", re.I)
        if avg_match and not avg_val:
            try:
                avg_val = float(avg_match.group(1).replace(",", "").strip())
            except ValueError:
                pass

        history_records.append(f"Source ({url}):\n{desc.strip()}")

    if not history_records:
        LOGGER.warning(f"No price history records found for '{product}'")
        return f"No verified price history found for '{product}'."

    lowest_so_far = low_val or avg_val or 0.0
    current_ref = avg_val or (lowest_so_far * 1.15 if lowest_so_far > 0 else 50000.0)

    prod_lower = product.lower()
    cat = "smartphone" if any(w in prod_lower for w in ["phone", "iphone", "galaxy", "pixel"]) else "electronics"

    try:
        rec = predict_buying_decision(
            current_price=current_ref,
            lowest_price_so_far=lowest_so_far if lowest_so_far > 0 else current_ref,
            category=cat,
            store="Amazon India",
            avg_price=avg_val,
        )
        model_block = (
            f"--- ML Model Recommendation ---\n"
            f"Verdict: {rec['verdict']}\n"
            f"Drop Probability: {rec['wait_probability']}%\n"
            f"Advice: {rec['action_advice']}"
        )
        LOGGER.info(f"Generated ML recommendation: {rec['verdict']} ({rec['wait_probability']}%)")
    except Exception as exc:
        LOGGER.error(f"Prediction model execution failed: {exc}")
        model_block = "ML recommendation could not be generated."

    result = "\n\n".join(history_records[:2]) + f"\n\n{model_block}"
    _HISTORY_CACHE[cache_key] = (time.time(), result)
    return result


if __name__ == "__main__":
    mcp.run()