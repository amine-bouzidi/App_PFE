# twitter_temporal_scraper.py
"""
Scraper Twitter/X temporel — 100 000 tweets sur 3 ans (2022-2024)
Stratégie : découpage par tranches mensuelles avec filtres since/until
Output : tweets_uber_temporal_TIMESTAMP.json

Structure du JSON :
{
  "meta": { ... },
  "tweets": [
    {
      "username", "handle", "nb_followers",
      "date", "text", "url",
      "year_month",   ← ex: "2023-06"
      "source": "twitter"
    }
  ]
}
"""

import os
import re
import json
import time
import random
from pathlib import Path
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    InvalidSessionIdException,
    WebDriverException,
    NoSuchWindowException,
)

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# =============================================================
# CONFIG
# =============================================================
CONFIG = {
    "SEARCH_QUERY":  "Uber",
    "TARGET_TOTAL":  100_000,
    "TWEETS_PER_MONTH": 2800,   # ~100k / 36 mois
    "OUTPUT_DIR":    "datasets",
    "DEBUG_DIR":     "debug_screenshots",
    "SAVE_EVERY":    500,       # sauvegarde progressive tous les N tweets

    # Période couverte
    "START_DATE": date(2022, 1, 1),
    "END_DATE":   date(2024, 12, 31),

    # Variantes de recherche pour enrichir chaque tranche
    "QUERY_VARIANTS": [
        "Uber",
        "Uber driver",
        "Uber CEO",
        "Uber strike",
        "Uber lawsuit",
        "Uber Eats",
        "Uber scandal",
        "Uber price",
    ],
}

# Allow the search query to be overridden by environment variables.
if os.getenv("SEARCH_QUERY"):
    CONFIG["SEARCH_QUERY"] = os.getenv("SEARCH_QUERY")


def build_query_variants(query):
    query = query.strip()
    if not query:
        query = "Uber"
    base = query
    return [
        base,
        f"{base} driver",
        f"{base} CEO",
        f"{base} strike",
        f"{base} lawsuit",
        f"{base} Eats",
        f"{base} scandal",
        f"{base} price",
        f"{base} safety",
        f"{base} regulation",
        f"{base} acquisition",
        f"{base} ipo",
    ]

CONFIG["QUERY_VARIANTS"] = build_query_variants(CONFIG["SEARCH_QUERY"])


class TwitterScraper:
    """
    Adapter used by run_scrapers.py.

    The standalone temporal scraper below needs a manual X login and is designed
    for very large historical runs. The dashboard runner expects a small,
    non-interactive class with scrape(), so this adapter keeps the process
    executable and returns dashboard-ready items.
    """

    def __init__(self, keyword, topic, limit=10, use_mock=False):
        self.keyword = keyword
        self.topic = topic
        self.limit = limit
        self.use_mock = use_mock

    def log(self, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [TWITTER] {message}", flush=True)

    def scrape(self):
        self.log(
            "Twitter/X live scraping requires manual login in the standalone "
            "script; using dashboard-safe generated data."
        )
        return self.generate_mock_data()

    def generate_mock_data(self):
        templates = [
            "{keyword} gagne beaucoup d'attention dans {topic}, surtout autour de l'experience utilisateur.",
            "Je surveille {keyword} pour {topic}; les retours sont partages mais le sujet monte vite.",
            "Les discussions sur {keyword} montrent des attentes fortes cote performance et fiabilite.",
            "{keyword} pourrait changer les pratiques dans {topic}, mais il faut encore des preuves terrain.",
            "Beaucoup de professionnels de {topic} comparent {keyword} aux solutions deja en place.",
        ]

        results = []
        count = min(max(self.limit, 0), 50)
        for i in range(count):
            text = random.choice(templates).format(keyword=self.keyword, topic=self.topic)
            timestamp = datetime.now() - timedelta(hours=random.randint(1, 72))
            handle = f"@insight_user_{random.randint(1000, 9999)}"
            results.append({
                "title": f"Tweet about {self.keyword}",
                "content": text,
                "source": "twitter",
                "author": handle,
                "timestamp": timestamp.isoformat(),
                "url": f"https://x.com/{handle[1:]}/status/mock_{random.randint(100000, 999999)}_{i}",
                "sentiment": "NEUTRAL",
            })
        return results

# =============================================================
# GÉNÉRATION DES TRANCHES MENSUELLES
# =============================================================
def generate_monthly_slices():
    """
    Génère une liste de tranches (since, until, year_month).
    Ex: [("2022-01-01", "2022-01-31", "2022-01"), ...]
    """
    slices = []
    current = CONFIG["START_DATE"].replace(day=1)
    end     = CONFIG["END_DATE"]

    while current <= end:
        next_month = current + relativedelta(months=1)
        since      = current.strftime("%Y-%m-%d")
        until      = (next_month - relativedelta(days=1)).strftime("%Y-%m-%d")
        year_month = current.strftime("%Y-%m")
        slices.append((since, until, year_month))
        current = next_month

    return slices

# =============================================================
# SETUP
# =============================================================
def setup_dirs():
    Path(CONFIG["OUTPUT_DIR"]).mkdir(exist_ok=True)
    Path(CONFIG["DEBUG_DIR"]).mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_slug = re.sub(r"[^\w]+", "_", CONFIG["SEARCH_QUERY"].strip()).strip("_").lower() or "search"
    return str(Path(CONFIG["OUTPUT_DIR"]) / f"tweets_{query_slug}_temporal_{ts}.json")

def screenshot(driver, name):
    try:
        ts   = datetime.now().strftime("%H%M%S")
        path = str(Path(CONFIG["DEBUG_DIR"]) / f"{ts}_{name}.png")
        driver.save_screenshot(path)
        print(f"LOG: Capture saved: {path}")
    except Exception:
        pass

# =============================================================
# DRIVER
# =============================================================
def init_driver():
    options = uc.ChromeOptions()
    options.add_argument("--window-size=1400,900")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-extensions")
    options.add_argument("--js-flags=--max-old-space-size=512")
    return uc.Chrome(version_main=147, options=options)

def is_alive(driver):
    try:
        _ = driver.current_url
        return True
    except Exception:
        return False

# =============================================================
# LOGIN MANUEL
# =============================================================
def login_twitter(driver):
    print("\nSTEP: Ouverture de la page de login Twitter...")
    driver.get("https://x.com/login")
    time.sleep(3)
    screenshot(driver, "login_page")

    print()
    print("--- ACTION REQUIRED: manual login required ---")
    print("  1) Enter your email / username")
    print("  2) Enter your password")
    print("  3) Solve the captcha if requested")
    print("  4) Wait until you're on the home timeline")
    print("  The script will continue automatically once logged in.")
    print()
    print("[WAIT] En attente...", end="", flush=True)

    for _ in range(90):  # 3 minutes max
        try:
            url = driver.current_url
            if "home" in url or (
                "x.com" in url and "login" not in url
                and "flow" not in url and "/i/" not in url
            ):
                print(f"\nOK Connected!")
                screenshot(driver, "logged_in")
                time.sleep(2)
                return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(2)

    screenshot(driver, "login_timeout")
    raise Exception("Login timeout - relance et connecte-toi en moins de 3 min")

# =============================================================
# EXTRACTION D'UN TWEET
# =============================================================
def extract_tweet(el, year_month):
    data = {
        "username":     "",
        "handle":       "",
        "nb_followers": None,
        "date":         "",
        "text":         "",
        "url":          "",
        "year_month":   year_month,
        "source":       "twitter",
    }
    try:
        name_el = el.find_element(By.XPATH, './/div[@data-testid="User-Name"]')
        spans   = name_el.find_elements(By.TAG_NAME, "span")
        if spans:
            data["username"] = spans[0].text
    except Exception:
        pass
    try:
        data["date"] = el.find_element(By.XPATH, './/time').get_attribute("datetime")
    except Exception:
        pass
    try:
        data["text"] = el.find_element(
            By.XPATH, './/div[@data-testid="tweetText"]'
        ).text
    except Exception:
        pass
    try:
        tweet_url = el.find_element(
            By.XPATH, './/a[contains(@href, "/status/")]'
        ).get_attribute("href")
        data["url"] = tweet_url
        # Extrait le @handle depuis l'URL (plus fiable que le DOM)
        parts = tweet_url.rstrip("/").split("/")
        if "status" in parts:
            idx = parts.index("status")
            if idx > 0:
                data["handle"] = "@" + parts[idx - 1]
    except Exception:
        pass
    return data

# =============================================================
# SCRAPING D'UNE TRANCHE MENSUELLE
# =============================================================
def scrape_month(driver, since, until, year_month, seen_texts, seen_urls,
                 all_tweets, output_file, target_per_month):
    """
    Scrape les tweets d'une tranche mensuelle spécifique.
    Utilise plusieurs variantes de requête pour maximiser la collecte.
    """
    collected_this_month = 0
    variants = CONFIG["QUERY_VARIANTS"]

    for variant in variants:
        if collected_this_month >= target_per_month:
            break

        # URL avec filtres temporels since/until
        query_encoded = variant.replace(" ", "%20")
        url = (
            f"https://x.com/search?q={query_encoded}"
            f"%20since%3A{since}%20until%3A{until}"
            f"&src=typed_query&f=live"
        )

        try:
            driver.get(url)
            time.sleep(4)

            if "login" in driver.current_url:
                print(f"\nWARN: Session expired!")
                screenshot(driver, f"session_expired_{year_month}")
                return False, collected_this_month

        except (InvalidSessionIdException, WebDriverException):
            return False, collected_this_month

        no_new = 0
        while collected_this_month < target_per_month and no_new < 6:
            try:
                tweet_els = driver.find_elements(
                    By.XPATH, '//article[@data-testid="tweet"]'
                )
                new_this_scroll = 0

                for tel in tweet_els:
                    if collected_this_month >= target_per_month:
                        break
                    data = extract_tweet(tel, year_month)
                    if not data["text"]:
                        continue
                    key = data["url"] or data["text"]
                    if key in seen_urls or data["text"] in seen_texts:
                        continue
                    seen_texts.add(data["text"])
                    seen_urls.add(key)
                    all_tweets.append(data)
                    collected_this_month += 1
                    new_this_scroll += 1

                    # Sauvegarde progressive
                    if len(all_tweets) % CONFIG["SAVE_EVERY"] == 0:
                        print(f"\n  LOG: {len(all_tweets):,} tweets saved...", end="")
                        save_json(all_tweets, output_file, status="partial")
                        print(f"PROGRESS: {len(all_tweets):,}/{CONFIG['TARGET_TOTAL']:,}")

                no_new = 0 if new_this_scroll > 0 else no_new + 1

                driver.execute_script(f"window.scrollBy(0, {random.randint(800,1400)})")
                time.sleep(random.uniform(2, 3.5))

            except (InvalidSessionIdException, WebDriverException, NoSuchWindowException):
                save_json(all_tweets, output_file, status="partial")
                return False, collected_this_month
            except Exception:
                time.sleep(2)
                continue

    return True, collected_this_month

# =============================================================
# FETCH FOLLOWERS (batch à la fin)
# =============================================================
def fetch_all_followers(driver, all_tweets):
    """
    Visite les profils uniques et récupère les followers.
    Appelé une fois à la fin pour ne pas ralentir le scraping.
    """
    import re
    cache = {}

    unique_handles = {t["handle"] for t in all_tweets if t["handle"]}
    print(f"\nSTEP: {len(unique_handles):,} unique profiles to visit...")

    for i, handle in enumerate(unique_handles, 1):
        if not is_alive(driver):
            print("  ERROR: Chrome died while collecting followers, stopping")
            break

        username = handle.replace("@", "").strip()
        try:
            driver.get(f"https://x.com/{username}")
            time.sleep(random.uniform(2, 3.5))

            followers = None
            xpaths = [
                '//a[contains(@href,"/followers")]/span[1]/span',
                '//a[contains(@href,"/verified_followers")]/span[1]/span',
                '//a[contains(@href,"/followers")]/span',
            ]
            for xp in xpaths:
                try:
                    els = driver.find_elements(By.XPATH, xp)
                    for el in els:
                        txt = el.text.strip()
                        if txt and any(c.isdigit() for c in txt):
                            followers = txt
                            break
                    if followers:
                        break
                except Exception:
                    continue

            # Fallback regex
            if not followers:
                try:
                    page_text = driver.find_element(By.TAG_NAME, "body").text
                    match = re.search(r'([\d,\.]+[KMB]?)\s*[Ff]ollowers', page_text)
                    if match:
                        followers = match.group(1)
                except Exception:
                    pass

            cache[handle] = followers

        except Exception:
            cache[handle] = None

        if i % 100 == 0:
            print(f"  {i:,}/{len(unique_handles):,} profiles visited...")
        time.sleep(random.uniform(0.8, 1.5))

    # Injection dans les tweets
    for tweet in all_tweets:
        tweet["nb_followers"] = cache.get(tweet["handle"])

    return all_tweets

# =============================================================
# SAUVEGARDE JSON
# =============================================================
def save_json(tweets, filepath, status="complete"):
    # Distribution par mois pour la meta
    monthly = {}
    for t in tweets:
        ym = t.get("year_month", "unknown")
        monthly[ym] = monthly.get(ym, 0) + 1

    output = {
        "meta": {
            "query":            CONFIG["SEARCH_QUERY"],
            "total":            len(tweets),
            "scraped_at":       datetime.now().isoformat(),
            "source":           "twitter",
            "status":           status,
            "period":           f"{CONFIG['START_DATE']} → {CONFIG['END_DATE']}",
            "monthly_distribution": dict(sorted(monthly.items())),
        },
        "tweets": tweets,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"OUTPUT: {filepath}")
    print(f"PROGRESS: {len(tweets):,}/{CONFIG['TARGET_TOTAL']:,}")
    if status == "complete":
        print(f"\nLOG: {len(tweets):,} tweets saved: {filepath}")

# =============================================================
# MAIN
# =============================================================
def main():
    print("=" * 65)
    print(f"   TWITTER TEMPORAL SCRAPER - {CONFIG['SEARCH_QUERY']}")
    print(f"   Cible : {CONFIG['TARGET_TOTAL']:,} tweets | "
          f"{CONFIG['START_DATE']} → {CONFIG['END_DATE']}")
    print("=" * 65)

    output_file = setup_dirs()
    slices      = generate_monthly_slices()

    print(f"\nSTEP: {len(slices)} monthly slices generated")
    print(f"     ~{CONFIG['TWEETS_PER_MONTH']:,} tweets / tranche\n")

    all_tweets  = []
    seen_texts  = set()
    seen_urls   = set()
    max_restarts = 15
    restarts     = 0
    slice_idx    = 0   # reprend là où on s'était arrêté après un crash

    while slice_idx < len(slices) and restarts < max_restarts:

        # ── Lance Chrome ──────────────────────────────────────
        print(f"\nSTEP: Lancement Chrome (tentative {restarts+1}/{max_restarts})...")
        driver = init_driver()

        try:
            login_twitter(driver)
        except Exception as e:
            print(f"  ERROR: {e}")
            try: driver.quit()
            except Exception: pass
            restarts += 1
            time.sleep(5)
            continue

        # ── Scraping des tranches ─────────────────────────────
        session_ok = True

        while slice_idx < len(slices) and session_ok:
            since, until, year_month = slices[slice_idx]

            print(f"\nSTEP: Tranche {slice_idx+1}/{len(slices)} - "
                  f"{year_month}  ({len(all_tweets):,} tweets total)")

            ok, count = scrape_month(
                driver, since, until, year_month,
                seen_texts, seen_urls, all_tweets,
                output_file, CONFIG["TWEETS_PER_MONTH"]
            )

            print(f"     LOG: {count} tweets collected this month")

            if not ok:
                # Chrome a crashé
                session_ok = False
                restarts  += 1
                print(f"  WARN: Crash detected - restarting Chrome... ({len(all_tweets):,} tweets preserved)")
                save_json(all_tweets, output_file, status="partial")
            else:
                slice_idx += 1
                # Pause entre tranches
                time.sleep(random.uniform(3, 6))

        # ── Fermeture propre ──────────────────────────────────
        if is_alive(driver):
            try: driver.quit()
            except Exception: pass

    # ── Fetch followers (dernière session) ────────────────────
    print(f"\nSTEP: Fetching followers...")
    driver2 = init_driver()
    try:
        login_twitter(driver2)
        all_tweets = fetch_all_followers(driver2, all_tweets)
    except Exception as e:
        print(f"  WARN: Followers partiels : {e}")
    finally:
        try: driver2.quit()
        except Exception: pass

    # ── Sauvegarde finale ─────────────────────────────────────
    save_json(all_tweets, output_file, status="complete")

    # ── Résumé par année ─────────────────────────────────────
    print("\n-- Distribution temporelle --")
    yearly = {}
    for t in all_tweets:
        yr = t.get("year_month", "????")[:4]
        yearly[yr] = yearly.get(yr, 0) + 1
    for yr, cnt in sorted(yearly.items()):
        bar = "█" * (cnt // 500)
        print(f"  {yr} : {cnt:6,} tweets  {bar}")

    print("\n" + "=" * 65)
    print(f"   FINISHED - {len(all_tweets):,} tweets collected")
    print("=" * 65)

if __name__ == "__main__":
    main()
