import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import time
import re

# Base wiki sections to start from
BASE_URLS = [
    "https://eldenring.wiki.fextralife.com/Weapons",
    "https://eldenring.wiki.fextralife.com/Armor",
    "https://eldenring.wiki.fextralife.com/Talismans",
    "https://eldenring.wiki.fextralife.com/Bosses",
    "https://eldenring.wiki.fextralife.com/Spells",
    "https://eldenring.wiki.fextralife.com/Quests"
]

# --- Configuration ---
MAX_SUBPAGES_PER_BASE = 15   # limit per category (safe for Fextralife)
REQUEST_DELAY = 2.0          # seconds between requests
USER_AGENT = "Mozilla/5.0 (compatible; EldenRingChatbot/1.0)"

# --- Functions ---

def get_soup(url):
    """Fetches and parses a webpage."""
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")

def clean_text(text):
    """Clean up whitespace and special characters."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def scrape_page(url):
    """Scrapes text from a single wiki page."""
    soup = get_soup(url)
    content = soup.get_text(separator="\n", strip=True)
    return clean_text(content)

def find_internal_links(base_url):
    """Find internal Elden Ring wiki links on a given page."""
    soup = get_soup(base_url)
    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]

        # Normalize internal links
        if href.startswith("/"):
            href = "https://eldenring.wiki.fextralife.com" + href

        # Only keep valid Elden Ring wiki pages
        if "eldenring.wiki.fextralife.com" in href and "http" in href:
            # Filter out irrelevant links (like categories or nav links)
            if not any(skip in href.lower() for skip in ["#","edit","file","image","special","user"]):
                links.add(href)

    return list(links)

# --- Main Scraper ---

def main():
    scraped_data = []
    visited = set()

    print("🕸️ Starting Elden Ring Wiki Scraper (with subpages)...\n")

    for base_url in BASE_URLS:
        print(f"📖 Scraping base category: {base_url}")
        try:
            base_content = scrape_page(base_url)
            scraped_data.append({"url": base_url, "content": base_content})
            visited.add(base_url)
            time.sleep(REQUEST_DELAY)

            # --- Find and scrape subpages ---
            sub_links = find_internal_links(base_url)
            sub_links = [link for link in sub_links if link not in visited][:MAX_SUBPAGES_PER_BASE]

            print(f"🔗 Found {len(sub_links)} subpages under {base_url}")

            for link in tqdm(sub_links, desc="Scraping subpages"):
                try:
                    text = scrape_page(link)
                    scraped_data.append({"url": link, "content": text})
                    visited.add(link)
                    time.sleep(REQUEST_DELAY)
                except Exception as e:
                    print(f"⚠️ Error scraping {link}: {e}")
                    continue

        except Exception as e:
            print(f"❌ Failed to scrape {base_url}: {e}")

    # Save to CSV
    df = pd.DataFrame(scraped_data)
    df.to_csv("eldenring_wiki_full.csv", index=False, encoding="utf-8")

    print("\n✅ Scraping complete! Saved to eldenring_wiki_full.csv")
    print(f"📄 Total pages scraped: {len(scraped_data)}")

if __name__ == "__main__":
    main()