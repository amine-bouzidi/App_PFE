# press_scraper.py
"""
Scraper Presse officielle haute capacité — jusqu'à 1000 articles sur "Uber"
Sources : Google News + Bing News + sources directes
Extraction propre via Trafilatura (élimine menus, pubs, footers)
Output : press_uber_TIMESTAMP.json

Chaque article contient :
  - source      : nom du journal/site (ex: "BBC", "Le Monde")
  - source_url  : domaine de la source
  - date        : date de publication
  - title       : titre de l'article
  - text        : contenu éditorial propre
  - url         : lien direct vers l'article
  - origin      : "press"
"""

import json
import os
import sys
import re
import time
import random
import requests
import trafilatura
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, quote_plus
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from base_scraper import BaseScraper

# =============================================================
# CONFIG
# =============================================================
CONFIG = {
    "SEARCH_QUERY":    "Uber",
    "MAX_ARTICLES":    1000,
    "OUTPUT_DIR":      "datasets",
    "MIN_TEXT_LENGTH": 200,   # ignorer les articles trop courts (en caractères)

    # Variantes de recherche pour maximiser la diversité
    "SEARCH_VARIANTS": [],

    # Sources de presse directes à scraper (pour diversifier)
    "DIRECT_SOURCES": [
        "https://www.bbc.com/news",
        "https://techcrunch.com/",
        "https://www.theguardian.com/technology",
        "https://www.reuters.com/technology/",
        "https://www.bloomberg.com/",
    ],
}


def build_search_variants(query):
    query = query.strip()
    if not query:
        query = "Uber"
    base = query
    return [
        base,
        f"{base} news",
        f"{base} company",
        f"{base} CEO",
        f"{base} driver",
        f"{base} driver strike",
        f"{base} lawsuit",
        f"{base} regulation",
        f"{base} earnings",
        f"{base} acquisition",
        f"{base} price",
        f"{base} safety",
        f"{base} market",
    ]


def sanitize_filename(text):
    return re.sub(r"[^\w]+", "_", text.strip()).strip("_").lower() or "search"


# Allow the search query to be overridden by environment variables or command-line args.
if os.getenv("SEARCH_QUERY"):
    CONFIG["SEARCH_QUERY"] = os.getenv("SEARCH_QUERY")
elif len(sys.argv) > 1:
    CONFIG["SEARCH_QUERY"] = sys.argv[1]

CONFIG["SEARCH_VARIANTS"] = build_search_variants(CONFIG["SEARCH_QUERY"])

# =============================================================
# SETUP
# =============================================================
def setup_output():
    output_dir = Path(CONFIG["OUTPUT_DIR"])
    output_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_slug = sanitize_filename(CONFIG["SEARCH_QUERY"])
    return str(output_dir / f"press_{query_slug}_{ts}.json")

def get_domain_name(url):
    """Extrait le nom de domaine propre depuis une URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        # Nettoyage : on garde juste le nom principal
        parts = domain.split(".")
        if len(parts) >= 2:
            return parts[-2].capitalize()
        return domain
    except Exception:
        return "Unknown"


def matches_search_query(text, title):
    query = CONFIG["SEARCH_QUERY"].strip().lower()
    if not query:
        return True

    haystack = f"{title} {text}".lower()
    terms = [term for term in re.split(r"\s+", query) if len(term) > 2]
    if not terms:
        return query in haystack
    return any(term in haystack for term in terms)


class NewsScraper(BaseScraper):
    def scrape(self):
        if self.use_mock:
            return self.generate_mock_data()

        previous_query = CONFIG["SEARCH_QUERY"]
        previous_target = CONFIG["MAX_ARTICLES"]
        previous_variants = CONFIG["SEARCH_VARIANTS"]

        try:
            CONFIG["SEARCH_QUERY"] = self.keyword
            CONFIG["MAX_ARTICLES"] = self.limit
            CONFIG["SEARCH_VARIANTS"] = build_search_variants(self.keyword)
            articles = scrape_press(target=self.limit)
            return [self.normalize_article(article) for article in articles]
        except Exception as e:
            self.log(f"Error during News scraping: {str(e)}. Using fallback mock data.")
            return self.generate_mock_data()
        finally:
            CONFIG["SEARCH_QUERY"] = previous_query
            CONFIG["MAX_ARTICLES"] = previous_target
            CONFIG["SEARCH_VARIANTS"] = previous_variants

    def normalize_article(self, article):
        title = article.get("title") or f"News about {self.keyword}"
        content = article.get("text") or title
        raw_date = article.get("date") or datetime.now().isoformat()

        return {
            "title": title,
            "content": content,
            "source": "news",
            "author": article.get("author") or article.get("source") or "Press",
            "timestamp": raw_date,
            "url": article.get("url"),
            "sentiment": "NEUTRAL",
        }

    def generate_mock_data(self):
        self.log(f"Generating mock News data for '{self.keyword}' in topic '{self.topic}'")
        templates = [
            {
                "title": "{keyword} accelere son adoption dans {topic}",
                "content": "Les analystes observent une progression nette de {keyword} dans le secteur {topic}, avec des usages plus matures et une demande croissante.",
                "author": "Insight Daily",
            },
            {
                "title": "Les entreprises de {topic} evaluent {keyword}",
                "content": "Plusieurs organisations testent {keyword} pour ameliorer leurs operations, tout en surveillant les couts, la securite et la qualite des resultats.",
                "author": "Tech Market News",
            },
            {
                "title": "{keyword}: opportunites et limites pour {topic}",
                "content": "Le marche reste optimiste, mais les decideurs demandent davantage de preuves avant de generaliser {keyword} dans les processus critiques.",
                "author": "Business Watch",
            },
        ]

        results = []
        count = min(max(self.limit, 0), 50)
        for i in range(count):
            tpl = random.choice(templates)
            results.append({
                "title": tpl["title"].format(keyword=self.keyword, topic=self.topic),
                "content": tpl["content"].format(keyword=self.keyword, topic=self.topic),
                "source": "news",
                "author": tpl["author"],
                "timestamp": datetime.now().isoformat(),
                "url": f"https://news.example.com/{sanitize_filename(self.keyword)}-{i}",
                "sentiment": "NEUTRAL",
            })
        return results

# =============================================================
# DRIVER SELENIUM (pour collecter les URLs)
# =============================================================
def init_driver():
    options = uc.ChromeOptions()
    options.add_argument("--window-size=1400,900")
    options.add_argument("--disable-blink-features=AutomationControlled")
    # options.add_argument("--headless=new")
    return uc.Chrome(version_main=147, options=options)

# =============================================================
# COLLECTE DES URLS — Google News
# =============================================================
def collect_urls_google_news(driver, query, max_urls=100):
    """
    Ouvre Google News, scrolle pour charger plus de résultats,
    et collecte les URLs vers les articles.
    """
    urls = set()
    encoded = quote_plus(query)
    url = f"https://www.google.com/search?q={encoded}&tbm=nws&num=100"
    driver.get(url)
    time.sleep(4)

    # Scroll pour charger plus de résultats
    for _ in range(5):
        links = driver.find_elements(By.XPATH, '//a')
        for link in links:
            href = link.get_attribute("href")
            if href and "google.com" not in href and href.startswith("http"):
                urls.add(href)
        driver.execute_script("window.scrollBy(0, 1000)")
        time.sleep(2)

    # Cliquer sur "Page suivante" si disponible
    try:
        next_btn = driver.find_element(By.XPATH, '//a[@id="pnnext"]')
        next_btn.click()
        time.sleep(3)
        links = driver.find_elements(By.XPATH, '//a')
        for link in links:
            href = link.get_attribute("href")
            if href and "google.com" not in href and href.startswith("http"):
                urls.add(href)
    except Exception:
        pass

    return list(urls)[:max_urls]

# =============================================================
# COLLECTE DES URLS — Bing News
# =============================================================
def collect_urls_bing_news(driver, query, max_urls=100):
    """
    Bing News comme source complémentaire à Google News.
    """
    urls = set()
    encoded = quote_plus(query)
    url = f"https://www.bing.com/news/search?q={encoded}&count=100"
    driver.get(url)
    time.sleep(4)

    for _ in range(5):
        links = driver.find_elements(By.XPATH, '//a[contains(@class,"title")]')
        for link in links:
            href = link.get_attribute("href")
            if href and "bing.com" not in href and "microsoft.com" not in href and href.startswith("http"):
                urls.add(href)
        driver.execute_script("window.scrollBy(0, 1000)")
        time.sleep(2)

    return list(urls)[:max_urls]

# =============================================================
# COLLECTE DES URLS — Sources directes
# =============================================================
def collect_urls_direct_sources(driver, max_urls_per_source=50):
    """
    Visite directement les pages de tag/topic des grands médias.
    """
    all_urls = set()

    for source_url in CONFIG["DIRECT_SOURCES"]:
        try:
            print(f"    LOG: Source URL: {source_url}")
            driver.get(source_url)
            time.sleep(4)

            for _ in range(3):
                links = driver.find_elements(By.TAG_NAME, "a")
                for link in links:
                    href = link.get_attribute("href")
                    if href and href.startswith("http"):
                        domain = urlparse(source_url).netloc
                        if domain.replace("www.", "") in href:
                            all_urls.add(href)
                driver.execute_script("window.scrollBy(0, 1000)")
                time.sleep(2)

        except Exception as e:
            print(f"    ERROR: Erreur sur {source_url}: {e}")
            continue

    return list(all_urls)

# =============================================================
# EXTRACTION D'UN ARTICLE — Trafilatura
# =============================================================
def extract_article(url):
    """
    Télécharge et extrait le contenu propre d'un article via Trafilatura.
    Retourne None si le contenu est trop court ou inutilisable.
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None

        # Extraction avec métadonnées
        meta_json = trafilatura.extract(
            downloaded,
            output_format="json",
            with_metadata=True,
            include_comments=False,
            include_tables=False,
        )

        if not meta_json:
            return None

        import json as _json
        meta = _json.loads(meta_json)

        text    = meta.get("text", "") or ""
        title   = meta.get("title", "") or ""
        date    = meta.get("date", "") or ""
        author  = meta.get("author", "") or ""

        # Filtrer les articles trop courts
        if len(text) < CONFIG["MIN_TEXT_LENGTH"]:
            return None

        # Filtrer les articles sans rapport avec la requête
        if not matches_search_query(text, title):
            return None

        source_name = get_domain_name(url)

        return {
            "source":      source_name,
            "source_url":  urlparse(url).netloc.replace("www.", ""),
            "date":        date,
            "title":       title,
            "author":      author,
            "text":        text,
            "url":         url,
            "origin":      "press",
        }

    except Exception:
        return None

# =============================================================
# PIPELINE PRINCIPAL
# =============================================================
def scrape_press(target=1000):
    articles  = []
    seen_urls = set()
    all_urls  = []

    print(f"\nSTEP: Start press scraping - target: {target} articles")

    driver = init_driver()

    try:
        # ── 1. Collecte des URLs via Google News (multi-variantes)
        print("\nSTEP: 1/3 Google News")
        for variant in CONFIG["SEARCH_VARIANTS"]:
            if len(all_urls) >= target * 2:
                break
            print(f"LOG: Request: {variant}")
            urls = collect_urls_google_news(driver, variant, max_urls=80)
            for u in urls:
                if u not in seen_urls:
                    all_urls.append(u)
                    seen_urls.add(u)
            time.sleep(random.uniform(2, 4))

        print(f"LOG: {len(all_urls)} URLs collected via Google News")

        # ── 2. Collecte via Bing News
        print("\nSTEP: 2/3 Bing News")
        for variant in CONFIG["SEARCH_VARIANTS"][:8]:
            if len(all_urls) >= target * 2:
                break
            print(f"LOG: Request: {variant}")
            urls = collect_urls_bing_news(driver, variant, max_urls=80)
            for u in urls:
                if u not in seen_urls:
                    all_urls.append(u)
                    seen_urls.add(u)
            time.sleep(random.uniform(2, 4))

        print(f"LOG: {len(all_urls)} total URLs after Bing News")

        # ── 3. Sources directes
        print("\n  [3/3] Sources directes (BBC, TechCrunch, Guardian...)...")
        direct_urls = collect_urls_direct_sources(driver)
        for u in direct_urls:
            if u not in seen_urls:
                all_urls.append(u)
                seen_urls.add(u)

        print(f"\nLOG: Total URLs to process: {len(all_urls)}")

    finally:
        try:
            driver.quit()
        except Exception:
            pass

    # ── Extraction du contenu avec Trafilatura (sans Selenium)
    print(f"\nSTEP: Extraction du contenu des articles")
    seen_content = set()

    for i, url in enumerate(all_urls):
        if len(articles) >= target:
            break

        article = extract_article(url)

        if article:
            # Déduplication par titre
            title_key = article["title"].lower().strip()
            if title_key and title_key in seen_content:
                continue
            seen_content.add(title_key)

            articles.append(article)
            if len(articles) % 50 == 0 or len(articles) == target:
                print(f"PROGRESS: {len(articles)}/{target}")

        # Pause aléatoire pour éviter le blocage
        if i % 10 == 0:
            time.sleep(random.uniform(1, 2))

        # Pause aléatoire pour éviter le blocage
        if i % 10 == 0:
            time.sleep(random.uniform(1, 2))

    return articles[:target]

# =============================================================
# SAUVEGARDE JSON
# =============================================================
def save_json(articles, filepath):
    # Distribution par source
    sources_dist = {}
    for a in articles:
        src = a["source"]
        sources_dist[src] = sources_dist.get(src, 0) + 1
    top_sources = sorted(sources_dist.items(), key=lambda x: -x[1])[:10]

    output = {
        "meta": {
            "query":        CONFIG["SEARCH_QUERY"],
            "total":        len(articles),
            "scraped_at":   datetime.now().isoformat(),
            "origin":       "press",
            "top_sources":  [{"source": s, "count": c} for s, c in top_sources],
        },
        "articles": articles,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"OUTPUT: {filepath}")
    print(f"PROGRESS: {len(articles)}/{CONFIG['MAX_ARTICLES']}")
    print(f"\nLOG: {len(articles)} articles saved, file: {filepath}")

    print("\n-- Top 10 sources --")
    for source, count in top_sources:
        print(f"  {source:30} : {count} articles")

# =============================================================
# MAIN
# =============================================================
def main():
    print("=" * 60)
    print("   PRESS SCRAPER - Uber  (target: 1000 articles)")
    print("=" * 60)

    output_file = setup_output()
    articles    = scrape_press(target=CONFIG["MAX_ARTICLES"])
    save_json(articles, output_file)

    # Aperçu
    print("\n-- Preview of first 3 articles --")
    for a in articles[:3]:
        print(f"  [{a['source']}] {a['date']}")
        print(f"  {a['title'][:80]}...")
        print(f"  {a['text'][:120]}...")
        print()

    print("=" * 60)
    print("   FINISHED")
    print("=" * 60)

if __name__ == "__main__":
    main()
