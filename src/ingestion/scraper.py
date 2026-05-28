import requests
from bs4 import BeautifulSoup
import markdownify
import json
import time
import os
from datetime import datetime
from tqdm import tqdm

# We only scrape these HuggingFace doc sections — focused and relevant
SECTIONS = [
    "https://huggingface.co/docs/transformers",
    "https://huggingface.co/docs/datasets",
    "https://huggingface.co/docs/peft",
    "https://huggingface.co/docs/tokenizers",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (educational RAG project)"
}

OUTPUT_DIR = "data/raw"


def get_doc_links(section_url: str) -> list[str]:
    """
    Fetches all internal doc page links from a HuggingFace docs section.
    HuggingFace docs have a sidebar with nav links — we extract those.
    """
    try:
        response = requests.get(section_url, headers=HEADERS, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to fetch {section_url}: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    # HuggingFace docs sidebar links all share the same base path
    base = section_url.rstrip("/")
    base_path = "/".join(base.split("/")[3:])  # e.g. "docs/transformers"

    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Only keep links that belong to this section
        if href.startswith("/" + base_path) or href.startswith(base):
            full = "https://huggingface.co" + href if href.startswith("/") else href
            # Strip fragment identifiers (#section-name)
            full = full.split("#")[0]
            links.add(full)

    return list(links)


def scrape_page(url: str) -> dict | None:
    """
    Downloads a single doc page and returns a structured dict with:
    - url: source URL (used for citations later)
    - title: page title
    - content: clean markdown text
    - scraped_at: timestamp
    """
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  Skipping {url}: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # HuggingFace docs put the main content inside this element
    main = soup.find("main") or soup.find("article") or soup.find("div", class_="prose")
    if not main:
        return None

    # Remove nav, footer, and script noise from the content area
    for tag in main.find_all(["nav", "footer", "script", "style", "button"]):
        tag.decompose()

    # Convert HTML to markdown — cleaner than raw text for chunking later
    content = markdownify.markdownify(str(main), heading_style="ATX")

    # Clean up excessive blank lines
    lines = [line for line in content.splitlines() if line.strip()]
    content = "\n\n".join(lines)

    if len(content) < 100:  # skip empty or near-empty pages
        return None

    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url.split("/")[-1]

    return {
        "url": url,
        "title": title,
        "content": content,
        "scraped_at": datetime.utcnow().isoformat(),
        "section": url.split("/docs/")[1].split("/")[0] if "/docs/" in url else "unknown"
    }


def save_doc(doc: dict, output_dir: str):
    """Saves a single document as a JSON file. Filename derived from URL."""
    # Turn the URL path into a safe filename
    slug = doc["url"].replace("https://huggingface.co/", "").replace("/", "_")
    slug = "".join(c for c in slug if c.isalnum() or c in "_-")
    filepath = os.path.join(output_dir, f"{slug}.json")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_links = []
    print("Discovering doc pages...")
    for section in SECTIONS:
        links = get_doc_links(section)
        print(f"  {section.split('/')[-1]}: {len(links)} pages found")
        all_links.extend(links)

    # Deduplicate
    all_links = list(set(all_links))
    print(f"\nTotal unique pages: {len(all_links)}")
    print("Starting scrape...\n")

    success = 0
    for url in tqdm(all_links):
        doc = scrape_page(url)
        if doc:
            save_doc(doc, OUTPUT_DIR)
            success += 1
        time.sleep(0.5)  # be polite to HuggingFace servers

    print(f"\nDone. Saved {success}/{len(all_links)} documents to {OUTPUT_DIR}/")


if __name__ == "__main__":
    run()