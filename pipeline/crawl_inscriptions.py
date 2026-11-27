import os
import re
import csv
import time
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://db.inscriptions.lk"
SEARCH_URL = f"{BASE_URL}/search"
IMAGE_DIR = "inscription_images"
CSV_FILE = "inscriptions_dataset.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

CSV_COLUMNS = [
    "inscriptionNo",
    "locallink",
    "realInscriptionUrl",
    "Transliteration",
    "Translation",
    "Footnotes",
    "Publication",
    "Site",
    "Period",
    "Language",
    "Sinhala Prakrit",
    "Script",
]


def get_soup(session: requests.Session, url: str) -> BeautifulSoup:
    resp = session.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def collect_all_inscription_urls(session: requests.Session) -> list[str]:
    """Paginates through /search?per_page=100&page=N&lang=en to collect all detail URLs."""
    urls = []
    seen = set()
    page = 1

    while True:
        page_url = f"{SEARCH_URL}?per_page=100&page={page}&lang=en"
        print(f"Fetching search page {page}: {page_url}")
        soup = get_soup(session, page_url)

        found_on_page = 0
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/inscription/" in href:
                full_url = urljoin(BASE_URL, href.split("?")[0])
                if full_url not in seen:
                    seen.add(full_url)
                    urls.append(full_url)
                    found_on_page += 1

        print(f"  -> Found {found_on_page} new inscriptions (Total: {len(urls)})")
        if found_on_page == 0:
            break

        page += 1
        time.sleep(0.5)

    return urls


def extract_tab_text(soup: BeautifulSoup, keywords: list[str]) -> str:
    """Extracts text from tab panes or sections matching Transliteration, Translation, or Footnotes."""
    # 1. Check Bootstrap/Tailwind tab panels via button/link target (data-bs-target, aria-controls, href)
    for trigger in soup.find_all(["button", "a"]):
        label = trigger.get_text(" ", strip=True).lower()
        if any(kw in label for kw in keywords):
            target_id = (
                trigger.get("data-bs-target")
                or trigger.get("data-target")
                or trigger.get("aria-controls")
                or trigger.get("href", "")
            )
            if target_id:
                clean_id = target_id.lstrip("#")
                panel = soup.find(id=clean_id)
                if panel:
                    return clean_block_text(panel)

    # 2. Fallback: look for headings or labels followed by content containers
    for tag in soup.find_all(["h2", "h3", "h4", "h5", "div", "span", "p"]):
        text = tag.get_text(" ", strip=True).lower()
        if text in keywords:
            sibling = tag.find_next_sibling()
            if sibling:
                return clean_block_text(sibling)

    return ""


def clean_block_text(element) -> str:
    """Joins list items or lines cleanly for ML dataset CSV storage."""
    items = element.find_all("li")
    if items:
        return ", ".join(li.get_text(" ", strip=True) for li in items if li.get_text(strip=True))
    lines = [line.strip() for line in element.get_text("\n").splitlines() if line.strip()]
    return " | ".join(lines)


def extract_sidebar_metadata(soup: BeautifulSoup) -> dict:
    """Parses the right-hand INFORMATION card fields."""
    meta = {
        "Publication": "",
        "Site": "",
        "Period": "",
        "Language": "",
        "Sinhala Prakrit": "",
        "Script": "",
    }

    target_labels = {
        "publication": "Publication",
        "site": "Site",
        "period": "Period",
        "language": "Language",
        "sinhala prakrit": "Sinhala Prakrit",
        "script": "Script",
    }

    for el in soup.find_all(["div", "dt", "span", "p", "h6", "small", "label"]):
        label_text = el.get_text(" ", strip=True).rstrip(":").lower()
        if label_text in target_labels:
            key = target_labels[label_text]
            if meta[key]:
                continue

            # Look at next sibling or parent's next element
            val_el = el.find_next_sibling()
            if not val_el and el.parent:
                val_el = el.parent.find_next_sibling()

            if val_el:
                li_tags = val_el.find_all("li")
                if li_tags:
                    meta[key] = ", ".join(li.get_text(" ", strip=True) for li in li_tags)
                else:
                    # Handle multiple badges/spans or comma-separated lines
                    lines = [
                        line.strip()
                        for line in val_el.get_text("\n").splitlines()
                        if line.strip() and line.strip().lower() != label_text
                    ]
                    meta[key] = ", ".join(lines)

    # Convenience mirror if Language is Sinhala Prakrit and dedicated column is blank
    if not meta["Sinhala Prakrit"] and "sinhala prakrit" in meta["Language"].lower():
        meta["Sinhala Prakrit"] = meta["Language"]

    return meta


def find_and_download_image(session: requests.Session, soup: BeautifulSoup, url_id: str) -> str:
    """Finds the inscription rubbing image, downloads it as <url_id>.<ext>, and returns relative path."""
    img_url = None

    # Prefer high-res modal/enlarge link if present, otherwise main plate <img>
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if any(ext in href.lower() for ext in [".jpg", ".jpeg", ".png", ".webp"]):
            img_url = urljoin(BASE_URL, href)
            break

    if not img_url:
        for img in soup.find_all("img", src=True):
            src = img["src"]
            # Skip site logos/icons
            if any(skip in src.lower() for skip in ["logo", "icon", "avatar", "flag"]):
                continue
            if any(ext in src.lower() for ext in [".jpg", ".jpeg", ".png", ".webp", "storage", "uploads", "inscription"]):
                img_url = urljoin(BASE_URL, src)
                break

    if not img_url:
        return ""

    parsed_path = urlparse(img_url).path
    ext = os.path.splitext(parsed_path)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp", ".gif"]:
        ext = ".jpg"

    local_path = os.path.join(IMAGE_DIR, f"{url_id}{ext}")

    if not os.path.exists(local_path):
        img_resp = session.get(img_url, headers=HEADERS, timeout=30)
        img_resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(img_resp.content)

    return local_path.replace("\\", "/")


def scrape_inscription(session: requests.Session, url: str) -> dict:
    url_id = url.rstrip("/").split("/")[-1]
    detail_url = f"{url}?lang=en"
    soup = get_soup(session, detail_url)

    local_img_path = find_and_download_image(session, soup, url_id)
    meta = extract_sidebar_metadata(soup)

    transliteration = extract_tab_text(soup, ["transliteration"])
    translation = extract_tab_text(soup, ["translation"])
    footnotes = extract_tab_text(soup, ["footnote", "footnotes", "notes"])

    return {
        "inscriptionNo": url_id,
        "locallink": local_img_path,
        "realInscriptionUrl": url,
        "Transliteration": transliteration,
        "Translation": translation,
        "Footnotes": footnotes,
        "Publication": meta["Publication"],
        "Site": meta["Site"],
        "Period": meta["Period"],
        "Language": meta["Language"],
        "Sinhala Prakrit": meta["Sinhala Prakrit"],
        "Script": meta["Script"],
    }


def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    session = requests.Session()

    # Load already-scraped URLs so script can resume safely if interrupted
    completed_urls = set()
    file_exists = os.path.exists(CSV_FILE)
    if file_exists:
        with open(CSV_FILE, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("realInscriptionUrl"):
                    completed_urls.add(row["realInscriptionUrl"])
        print(f"Resuming: {len(completed_urls)} entries already in {CSV_FILE}")

    all_urls = collect_all_inscription_urls(session)
    remaining_urls = [u for u in all_urls if u not in completed_urls]
    print(f"Total to scrape: {len(remaining_urls)}")

    with open(CSV_FILE, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()

        for idx, url in enumerate(remaining_urls, start=1):
            try:
                row = scrape_inscription(session, url)
                writer.writerow(row)
                f.flush()
                print(f"[{idx}/{len(remaining_urls)}] Saved #{row['inscriptionNo']} -> {row['locallink']}")
            except Exception as e:
                print(f"[{idx}/{len(remaining_urls)}] ERROR on {url}: {e}")

            time.sleep(0.35)  # Polite rate limit


if __name__ == "__main__":
    main()