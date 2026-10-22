#!/usr/bin/env python3
"""
Research collector for GamingLoreAI.

Collects source text for one game + topic and writes it in the format that
prepare_evidence.py reads (SOURCE / RETRIEVAL STATUS / [PASSAGE nnn] lines).

Two kinds of source are supported:

* Fandom wikis (anything on *.fandom.com / *.wikia.com): fetched through the
  wiki's own MediaWiki API (api.php), which returns the clean article HTML.
  Passages keep their section name, e.g. "(Walkthrough > Aftermath) ...", and the
  infobox is captured as a key/value passage. If the API is unavailable the
  rendered article (?action=render) is used instead.
* Any other web page: generic paragraph extraction, as before.

You can paste URLs, and/or search a Fandom wiki by topic and pick pages.

Fandom community text is licensed CC BY-SA. The output records an ATTRIBUTION
line per page; credit the wiki when you publish.

Interactive:   python research_sources.py
Scripted:      python research_sources.py --game "Some Game" --topic "Some topic" \
                   --wiki somegame.fandom.com --search "boss lore" --pick top3 \
                   --url https://somegame.fandom.com/wiki/Some_Page
"""
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, unquote, quote, parse_qs
import argparse
import re
import sys
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup

PROJECT = Path.home() / "Desktop" / "GamingLoreAI"
SOURCES = PROJECT / "sources"

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )
}
API_HEADERS = {
    "User-Agent": "GamingLoreAI-research/1.0 (personal fact-checking tool for short videos)",
    "Accept": "application/json",
}

TIMEOUT_SECONDS = 25
REQUEST_DELAY_SECONDS = 1.0          # be polite to wikis: at most ~1 request per second
MAX_PARAGRAPHS_PER_SOURCE = 100      # generic pages
MAX_PARAGRAPHS_PER_WIKI_PAGE = 150   # Fandom articles are structured, so allow a little more
MAX_CHARS_PER_PARAGRAPH = 1500
MIN_PARAGRAPH_LENGTH = 40
SEARCH_RESULTS = 8

# Sections of wiki articles that hold navigation or link lists, not facts.
SKIP_SECTIONS = {
    "references", "gallery", "videos", "images", "see also", "external links", "navigation",
    "sources", "translations", "in other languages", "appearances", "related pages", "contents",
}

# Elements that are page chrome rather than article text.
REMOVE_SELECTORS = (
    "script, style, noscript, .mw-editsection, sup.reference, .reference, .toc, #toc, "
    ".navbox, table.navbox, .noprint, .printfooter, .ambox, .mbox, .message-box, .gallery, "
    ".wikia-gallery, .wds-tabs, .mw-empty-elt, figure, figcaption, .thumb, .thumbcaption, "
    ".pi-image, .licensebox, .stub, .hatnote"
)
GENERIC_REMOVE_SELECTORS = "script, style, nav, footer, header, aside, form, noscript, iframe, svg, button"

FANDOM_HOST_RE = re.compile(r"(^|\.)(fandom\.com|wikia\.com|wikia\.org)$", re.I)
NOT_A_WIKI_HOSTS = {"www.fandom.com", "fandom.com"}   # editorial site, not a wiki

_last_request_at = 0.0


class FetchError(Exception):
    pass


# --------------------------------------------------------------------------- helpers

def slugify(text):
    text = re.sub(r"[^a-z0-9]+", "_", text.lower().strip())
    return text.strip("_")[:60] or "research"


def clean_text(text):
    """Normalise whitespace and strip footnote/edit markers without changing meaning."""
    text = re.sub(r"\[(?:\d+|edit|citation needed|source needed|note \d+)\]", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text, limit=MAX_CHARS_PER_PARAGRAPH):
    """Split over-long text on sentence boundaries instead of cutting facts off."""
    if len(text) <= limit:
        return [text]
    chunks, current = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if current and len(current) + len(sentence) + 1 > limit:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return [c[:limit] for c in chunks]


def make_session():
    session = requests.Session()
    retry = Retry(
        total=3, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",), respect_retry_after_header=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session


def polite_get(session, url, **kwargs):
    global _last_request_at
    wait = REQUEST_DELAY_SECONDS - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    kwargs.setdefault("timeout", TIMEOUT_SECONDS)
    try:
        return session.get(url, **kwargs)
    finally:
        _last_request_at = time.monotonic()


# --------------------------------------------------------------------------- Fandom

def is_fandom_url(url):
    host = urlparse(url).netloc.lower().split(":")[0]
    return bool(FANDOM_HOST_RE.search(host)) and host not in NOT_A_WIKI_HOSTS


def normalise_wiki(value):
    """'witcher', 'witcher.fandom.com' or a full URL -> ('https://witcher.fandom.com', language prefix)."""
    value = value.strip()
    if not value:
        return None
    if "://" not in value:
        value = "https://" + (value if "." in value else f"{value}.fandom.com")
    parsed = urlparse(value)
    host = parsed.netloc.lower()
    if not FANDOM_HOST_RE.search(host.split(":")[0]):
        return None
    lang = re.match(r"^(/[a-z]{2,3}(?:-[a-z]{2,4})?)(?:/|$)", parsed.path, re.I)
    lang = lang.group(1) if lang and lang.group(1).lower() not in ("/wiki", "/api") else ""
    return f"https://{host}{lang}"


def parse_fandom_url(url):
    """Return {'base', 'title'} for a Fandom article URL, or None if it is not an article URL."""
    parsed = urlparse(url)
    path = unquote(parsed.path)
    match = re.match(r"^(?P<lang>/[a-z]{2,3}(?:-[a-z]{2,4})?)?/wiki/(?P<title>.+)$", path, re.I)
    if match:
        lang, title = match.group("lang") or "", match.group("title")
    else:
        query = parse_qs(parsed.query)
        if "title" not in query:
            return None
        title = query["title"][0]
        lang_match = re.match(r"^(/[a-z]{2,3}(?:-[a-z]{2,4})?)/", path, re.I)
        lang = lang_match.group(1) if lang_match else ""
    title = title.replace("_", " ").strip()
    if not title:
        return None
    return {"base": f"https://{parsed.netloc.lower()}{lang}", "title": title}


URL_SAFE = ":/()!,'"


def article_url(base, title):
    return f"{base}/wiki/{quote(title.replace(' ', '_'), safe=URL_SAFE)}"


def wiki_api(session, base, params):
    """Call a wiki's api.php and return parsed JSON; raise FetchError with a readable reason."""
    query = {**params, "format": "json", "formatversion": 2}
    response = polite_get(session, f"{base}/api.php", params=query, headers=API_HEADERS)
    if response.status_code in (403, 406):   # some CDNs dislike non-browser agents
        response = polite_get(session, f"{base}/api.php", params=query, headers=BROWSER_HEADERS)
    if response.status_code >= 400:
        raise FetchError(f"HTTP {response.status_code} from {base}/api.php")
    try:
        data = response.json()
    except ValueError:
        raise FetchError("API did not return JSON (wrong wiki address?)")
    if "error" in data:
        raise FetchError(f"{data['error'].get('code', 'error')}: {data['error'].get('info', '')}")
    return data


def search_wiki(session, base, query, limit=SEARCH_RESULTS):
    data = wiki_api(session, base, {
        "action": "query", "list": "search", "srsearch": query,
        "srlimit": limit, "srnamespace": 0, "srprop": "snippet",
    })
    results = []
    for item in data.get("query", {}).get("search", []):
        snippet = BeautifulSoup(item.get("snippet", ""), "html.parser").get_text(" ", strip=True)
        results.append({"title": item["title"], "snippet": clean_text(snippet), "url": article_url(base, item["title"])})
    return results


def wiki_site_name(session, base):
    try:
        data = wiki_api(session, base, {"action": "query", "meta": "siteinfo", "siprop": "general"})
        return data["query"]["general"].get("sitename") or ""
    except (FetchError, KeyError, requests.RequestException):
        return ""


def _own_text(element):
    """Text of a list item without the text of its nested sub-lists."""
    parts = []
    for child in element.children:
        if getattr(child, "name", None) in ("ul", "ol", "dl"):
            continue
        parts.append(child.get_text(" ", strip=True) if hasattr(child, "get_text") else str(child))
    return clean_text(" ".join(parts))


def _infobox_passage(root):
    box = root.select_one("aside.portable-infobox") or root.select_one("table.infobox")
    if box is None:
        return None
    pairs = []
    for item in box.select(".pi-data"):
        label, value = item.select_one(".pi-data-label"), item.select_one(".pi-data-value")
        if label and value:
            pairs.append(f"{clean_text(label.get_text(' ', strip=True))}: {clean_text(value.get_text(' ', strip=True))}")
    if not pairs:   # classic table infobox
        for row in box.find_all("tr"):
            head, cell = row.find("th"), row.find("td")
            if head and cell:
                pairs.append(f"{clean_text(head.get_text(' ', strip=True))}: {clean_text(cell.get_text(' ', strip=True))}")
    box.extract()
    text = "; ".join(pair for pair in pairs if not pair.endswith(": "))
    return f"(Infobox) {text}"[:MAX_CHARS_PER_PARAGRAPH] if len(text) >= MIN_PARAGRAPH_LENGTH else None


def _table_passages(table, context):
    """Small data tables become 'Header: value | Header: value' rows; big/layout tables are skipped."""
    rows = table.find_all("tr")
    if not 2 <= len(rows) <= 15:
        return []
    headers = [clean_text(th.get_text(" ", strip=True)) for th in rows[0].find_all(["th", "td"])]
    if not headers or len(headers) > 6:
        return []
    out = []
    for row in rows[1:]:
        cells = [clean_text(c.get_text(" ", strip=True)) for c in row.find_all(["th", "td"])]
        if len(cells) == len(headers) and any(cells):
            out.append(context + " | ".join(f"{h}: {c}" for h, c in zip(headers, cells) if c))
    return out


def wiki_html_to_passages(html_text, skip_sections=SKIP_SECTIONS, limit=MAX_PARAGRAPHS_PER_WIKI_PAGE):
    """Turn a rendered MediaWiki article into section-labelled passages."""
    soup = BeautifulSoup(html_text, "html.parser")
    root = soup.find("div", class_="mw-parser-output") or soup
    passages = []

    infobox = _infobox_passage(root)
    if infobox:
        passages.append(infobox)
    for element in root.select(REMOVE_SELECTORS):
        element.extract()

    stack, skipping = [], None
    description = ""
    for element in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "dd", "blockquote", "table"]):
        if element.name[0] == "h" and len(element.name) == 2:
            level, title = int(element.name[1]), clean_text(element.get_text(" ", strip=True))
            if not title:
                continue
            stack = stack[: max(level - 2, 0)] + [title]
            if title.casefold() in skip_sections:
                skipping = level
            elif skipping is not None and level <= skipping:
                skipping = None
            continue
        if skipping is not None:
            continue
        if element.name != "table" and element.find_parent(["table", "aside"]):
            continue
        context = f"({' > '.join(stack)}) " if stack else ""
        if element.name == "table":
            if element.find_parent("table"):
                continue
            passages.extend(_table_passages(element, context))
            continue
        text = _own_text(element) if element.name in ("li", "dd") else clean_text(element.get_text(" ", strip=True))
        if len(text) < MIN_PARAGRAPH_LENGTH:
            continue
        if not description and element.name == "p":
            description = text[:300]
        passages.extend(f"{context}{chunk}" for chunk in chunk_text(text))

    unique = []
    for passage in passages:
        if passage not in unique:
            unique.append(passage)
    return unique[:limit], description


def fetch_fandom_article(session, info, skip_sections=SKIP_SECTIONS, limit=MAX_PARAGRAPHS_PER_WIKI_PAGE):
    base, title = info["base"], info["title"]
    via, page_html, final_title, revid, categories = "api", None, title, "", []
    try:
        data = wiki_api(session, base, {
            "action": "parse", "page": title, "prop": "text|revid|categories|displaytitle",
            "redirects": 1, "disableeditsection": 1, "disabletoc": 1,
        })
        parsed = data["parse"]
        text = parsed["text"]
        page_html = text["*"] if isinstance(text, dict) else text
        final_title = BeautifulSoup(parsed.get("displaytitle") or parsed.get("title", title), "html.parser").get_text(strip=True) or title
        revid = str(parsed.get("revid", ""))
        categories = [c.get("category", "").replace("_", " ") for c in parsed.get("categories", []) if not c.get("hidden")]
    except (FetchError, KeyError, requests.RequestException) as api_error:
        # Fallback: the rendered article body (same HTML structure, no JSON wrapper).
        response = polite_get(session, article_url(base, title), params={"action": "render"}, headers=BROWSER_HEADERS)
        if response.status_code >= 400 or "html" not in response.headers.get("Content-Type", "").lower():
            raise FetchError(f"API failed ({api_error}); render fallback gave HTTP {response.status_code}")
        page_html, via = response.text, "render"

    passages, description = wiki_html_to_passages(page_html, skip_sections, limit)
    site = wiki_site_name(session, base) or urlparse(base).netloc.split(".")[0].replace("-", " ").title() + " Wiki"
    return {
        "title": final_title, "site": site, "paragraphs": passages, "description": description,
        "final_url": article_url(base, final_title), "revid": revid, "categories": categories[:10], "via": via,
        "status_code": 200, "content_type": "text/html (MediaWiki " + via + ")",
    }


# --------------------------------------------------------------------------- generic pages

def extract_page(session, url):
    """Download an accessible HTML page and extract its main text (non-wiki sources)."""
    response = polite_get(session, url, headers=BROWSER_HEADERS)
    response.raise_for_status()
    content_type = response.headers.get("Content-Type", "")
    if "html" not in content_type.lower():
        raise ValueError(f"Expected an HTML page, received: {content_type or 'unknown'}")

    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup.select(GENERIC_REMOVE_SELECTORS):
        element.extract()   # extract(), not decompose(): deep pages can exceed the recursion limit

    title_element = soup.find("title")
    title = clean_text(title_element.get_text(" ", strip=True)) if title_element else ""
    description = ""
    description_element = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    if description_element:
        description = clean_text(description_element.get("content", ""))

    main = soup.find("article") or soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup
    paragraphs = []
    for element in main.find_all(["h1", "h2", "h3", "p", "li"]):
        text = clean_text(element.get_text(" ", strip=True))
        if len(text) < MIN_PARAGRAPH_LENGTH:
            continue
        for chunk in chunk_text(text):
            if chunk not in paragraphs:
                paragraphs.append(chunk)
        if len(paragraphs) >= MAX_PARAGRAPHS_PER_SOURCE:
            break
    return {
        "title": title or url, "description": description, "paragraphs": paragraphs[:MAX_PARAGRAPHS_PER_SOURCE],
        "final_url": response.url, "status_code": response.status_code, "content_type": content_type,
    }


# --------------------------------------------------------------------------- output

def source_block(url, page, fandom):
    """Lines describing one successfully retrieved source, in the format prepare_evidence.py reads."""
    if fandom:
        lines = [
            f"SOURCE: {page['title']} | {page['site']} (Fandom)",
            "RETRIEVAL STATUS: SUCCESS",
            "SOURCE TYPE: Fandom wiki article (community-edited, CC BY-SA; verify against the game itself)",
            f"URL: {url}",
            f"FINAL URL: {page['final_url']}",
            f"HOST: {urlparse(page['final_url']).netloc}",
            f"CONTENT TYPE: {page['content_type']}",
            f"REVISION: {page['revid'] or 'unknown'}",
            f"CATEGORIES: {', '.join(page['categories']) or 'none listed'}",
            f"ATTRIBUTION: \"{page['title']}\" from {page['site']} on Fandom, licensed CC BY-SA ({page['final_url']})",
            f"PAGE DESCRIPTION: {page['description']}",
            f"EXTRACTED PARAGRAPHS: {len(page['paragraphs'])}",
            "",
        ]
    else:
        lines = [
            f"SOURCE: {page['title']}",
            "RETRIEVAL STATUS: SUCCESS",
            "SOURCE TYPE: Webpage; classify and verify separately",
            f"URL: {url}",
            f"FINAL URL: {page['final_url']}",
            f"HOST: {urlparse(page['final_url']).netloc}",
            f"HTTP STATUS: {page['status_code']}",
            f"CONTENT TYPE: {page['content_type']}",
            f"PAGE DESCRIPTION: {page['description']}",
            f"EXTRACTED PARAGRAPHS: {len(page['paragraphs'])}",
            "",
        ]
    lines += [f"[PASSAGE {n:03d}] {paragraph}" for n, paragraph in enumerate(page["paragraphs"], start=1)]
    lines.append("")
    return lines


def failure_block(url, status, detail):
    return [
        "SOURCE: Download failed", f"RETRIEVAL STATUS: {status}", f"URL: {url}",
        f"HOST: {urlparse(url).netloc}", f"ERROR: {detail}",
        "NOTE: This source was not retrieved successfully. No attempt was made to bypass access restrictions.", "",
    ]


# --------------------------------------------------------------------------- interaction

def ask(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        return ""


def valid_url(url):
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def choose_results(results, pick):
    """pick: 'topN', 'all', '1,3' or '' (prompt)."""
    if not results:
        return []
    if not pick:
        for number, item in enumerate(results, 1):
            print(f"  {number}. {item['title']}  -  {item['snippet'][:110]}")
        pick = ask("Pages to add (numbers like 1,3 | 'all' | Enter = top 3): ") or "top3"
    pick = pick.strip().lower()
    if pick == "all":
        return results
    top = re.fullmatch(r"top\s*(\d+)", pick)
    if top:
        return results[: int(top.group(1))]
    chosen = []
    for part in re.split(r"[,\s]+", pick):
        if part.isdigit() and 1 <= int(part) <= len(results):
            chosen.append(results[int(part) - 1])
    return chosen


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Collect source text (Fandom wikis and web pages) for one game/topic.")
    parser.add_argument("--game")
    parser.add_argument("--topic")
    parser.add_argument("--url", action="append", default=[], help="source URL (repeatable)")
    parser.add_argument("--wiki", help="Fandom wiki to search, e.g. witcher.fandom.com or just 'witcher'")
    parser.add_argument("--search", help="search words for --wiki (default: the topic)")
    parser.add_argument("--pick", default="", help="which search results to add: top3, all, or 1,3")
    parser.add_argument("--skip-section", action="append", default=[], help="extra wiki section names to ignore")
    parser.add_argument("--no-prompt", action="store_true", help="never ask questions; use only the arguments")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    SOURCES.mkdir(parents=True, exist_ok=True)
    interactive = not args.no_prompt and sys.stdin.isatty()

    game = (args.game or (ask("Game title: ") if interactive else "")).strip()
    topic = (args.topic or (ask("Video topic: ") if interactive else "")).strip()
    if not game or not topic:
        print("Game and topic are required.")
        return 1

    session = make_session()
    urls = []
    for url in args.url:
        if valid_url(url) and url not in urls:
            urls.append(url)
        else:
            print(f"Skipping invalid or duplicate URL: {url}")

    if not urls and interactive and not args.wiki:
        print("\nEnter source URLs, one per line (Fandom wiki pages work best). Press Enter on an empty line when finished.")
        while True:
            url = ask("URL: ")
            if not url:
                break
            if not valid_url(url):
                print("Please enter a complete http:// or https:// URL.")
            elif url in urls:
                print("That URL is already on the list.")
            else:
                urls.append(url)

    wiki_value = args.wiki
    if wiki_value is None and interactive:
        guess = urlparse(urls[0]).netloc if urls and is_fandom_url(urls[0]) else ""
        wiki_value = ask(f"Fandom wiki to search for more pages (e.g. witcher.fandom.com{'; Enter = ' + guess if guess else ''}; blank to skip): ") or guess
    if wiki_value:
        base = normalise_wiki(wiki_value)
        if not base:
            print(f"'{wiki_value}' is not a Fandom wiki address; skipping search.")
        else:
            query = args.search or (ask(f"Search words [{topic}]: ") if interactive else "") or topic
            try:
                results = search_wiki(session, base, query)
                print(f"\nFound {len(results)} page(s) on {base} for '{query}'.")
                for item in choose_results(results, args.pick or ("" if interactive else "top3")):
                    if item["url"] not in urls:
                        urls.append(item["url"])
                        print(f"  + {item['title']}")
            except (FetchError, requests.RequestException) as error:
                print(f"Wiki search failed: {error}")

    if not urls:
        print("No URLs supplied; nothing to download.")
        return 1

    skip_sections = SKIP_SECTIONS | {s.casefold() for s in args.skip_section}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = SOURCES / f"{slugify(game)}_{slugify(topic)}_research_{timestamp}.txt"
    sections = [
        f"GAME: {game}",
        f"TOPIC: {topic}",
        f"RETRIEVED: {datetime.now().isoformat(timespec='seconds')}",
        "COLLECTION METHOD: Fandom wikis via the MediaWiki API (api.php); other pages by direct download",
        "IMPORTANT: Extracted text is unverified. Collection does not prove that a claim is true.",
        "",
    ]
    successful = failed = total_paragraphs = 0

    for index, url in enumerate(urls, start=1):
        print(f"\n[{index}/{len(urls)}] Downloading: {url}")
        fandom = is_fandom_url(url)
        try:
            if fandom:
                info = parse_fandom_url(url)
                if info is None:
                    raise FetchError("not a wiki article URL (expected .../wiki/Page_Name)")
                page = fetch_fandom_article(session, info, skip_sections)
            else:
                page = extract_page(session, url)
            successful += 1
            total_paragraphs += len(page["paragraphs"])
            sections.extend(source_block(url, page, fandom))
            print(f"Collected {len(page['paragraphs'])} passages from: {page['title']}")
            if len(page["paragraphs"]) < 3:
                print("Warning: very little text was extracted (stub page, or the page needs a browser to render).")
        except requests.HTTPError as error:
            failed += 1
            code = getattr(error.response, "status_code", "?")
            sections.extend(failure_block(url, "HTTP ERROR", f"HTTP {code}: {error}"))
            print(f"HTTP error: {error}")
        except FetchError as error:
            failed += 1
            sections.extend(failure_block(url, "REQUEST ERROR", str(error)))
            print(f"Could not fetch: {error}")
        except requests.RequestException as error:
            failed += 1
            sections.extend(failure_block(url, "REQUEST ERROR", f"{type(error).__name__}: {error}"))
            print(f"Request failed: {error}")
        except Exception as error:
            failed += 1
            sections.extend(failure_block(url, "EXTRACTION ERROR", f"{type(error).__name__}: {error}"))
            print(f"Could not extract this page: {error}")

    sections.extend([
        "COLLECTION SUMMARY",
        f"URLS SUBMITTED: {len(urls)}",
        f"SUCCESSFUL DOWNLOADS: {successful}",
        f"FAILED DOWNLOADS: {failed}",
        f"TOTAL EXTRACTED PASSAGES: {total_paragraphs}",
        "",
        "NEXT STEP: Run prepare_evidence.py, then generate the script. Wiki text is community-edited; "
        "check important claims against the game itself or an official source.",
    ])
    output.write_text("\n".join(sections), encoding="utf-8")

    print(f"\nCollection complete. Successful: {successful} | Failed: {failed} | Passages: {total_paragraphs}")
    print(f"Saved research to:\n{output}")
    return 0 if successful else 1


if __name__ == "__main__":
    sys.exit(main())
