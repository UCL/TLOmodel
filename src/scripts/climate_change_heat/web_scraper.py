import re
import io
import logging
import pdfplumber
import pandas as pd
from datetime import datetime
from typing import Optional
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("scraper.log")],
)
logger = logging.getLogger(__name__)

# ── Indicators ─────────────────────────────────────────────────────────────────
COUNT_INDICATORS = [
    "opd_attendance", "ipd_total_admissions", "fp_total_clients",
    "fp_subsequent_clients_total", "bcg_under1", "penta3_under1",
    "measles1_under1", "fully_immunised_under1", "live_births_total",
    "htc_results_new_negative", "htc_results_new_positive",
    "anc_total_visits", "cervical_screening_total",
]

DATE_FORMATS = [
    "%d %B %Y", "%Y-%m-%d", "%d/%m/%Y",
    "%B %Y",        # ← NEW: "January 2023"
    "%d-%b-%Y",     # ← NEW: "03-Jan-2023"
    "%b %d, %Y",    # ← NEW: "Jan 03, 2023"
    "%d %b %Y",     # ← NEW: "03 Jan 2023"
]

DATE_PATTERNS = [
    r"\d{1,2}\s+(?:January|February|March|April|May|June|July|"
    r"August|September|October|November|December)\s+\d{4}",
    r"(?:January|February|March|April|May|June|July|"
    r"August|September|October|November|December)\s+\d{4}",   # ← NEW: month-year only
    r"\d{4}-\d{2}-\d{2}",
    r"\d{1,2}/\d{1,2}/\d{4}",
    r"\d{1,2}-(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4}",  # ← NEW
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}",  # ← NEW
]

# ── How far back to scrape (set None for no limit) ─────────────────────────────
EARLIEST_DATE = datetime(2019, 1, 1)   # ← CHANGE THIS to go further back

# ── WordPress year/month archive seeds ────────────────────────────────────────
BASE_URL = "https://phim.health.gov.mw"
CURRENT_YEAR = datetime.now().year


def _build_archive_urls(start_year: int = 2019) -> list[str]:
    """
    Generate WordPress year-archive URLs for every year from
    start_year up to the current year.  These act as extra entry
    points so the scraper doesn't rely solely on paginating the
    category listing.
    """
    return [
        f"{BASE_URL}/{year}/"
        for year in range(start_year, CURRENT_YEAR + 1)
    ]


class MalawiHealthScraper:
    def __init__(self, earliest_date: Optional[datetime] = EARLIEST_DATE):
        self.bulletins_url  = f"{BASE_URL}/category/latest-news-and-events/"
        self.earliest_date  = earliest_date
        self._visited_urls: set[str] = set()   # ← dedup across all entry points
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }

    # ── Playwright helpers ─────────────────────────────────────────────────────

    def _get_rendered_soup(self, page, url: str):
        from bs4 import BeautifulSoup
        try:
            page.goto(url, wait_until="networkidle", timeout=30_000)
            return BeautifulSoup(page.content(), "html.parser")
        except PlaywrightTimeout:
            logger.warning(f"Timeout rendering {url}")
        except Exception as exc:
            logger.error(f"Playwright error for {url}: {exc}")
        return None

    # ── PDF helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _is_pdf(url: str) -> bool:
        return url.lower().endswith(".pdf") or "pdf" in url.lower()

    def _extract_pdf_text(self, page, url: str) -> str:
        try:
            api_response = page.request.get(url, timeout=30_000)
            raw = api_response.body()
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                pages_text = [p.extract_text() or "" for p in pdf.pages]
                text = "\n".join(pages_text)
            logger.info(f"  PDF extracted — {len(pdf.pages)} pages, {len(text):,} chars")
            return text
        except Exception as exc:
            logger.error(f"  PDF extraction failed for {url}: {exc}")
            return ""

    # ── Date helpers ───────────────────────────────────────────────────────────

    def _extract_date(self, text: str) -> Optional[datetime]:
        for pattern in DATE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                raw = match.group()
                for fmt in DATE_FORMATS:
                    try:
                        return datetime.strptime(raw, fmt)
                    except ValueError:
                        continue
        return None

    def _is_too_old(self, date: Optional[datetime]) -> bool:
        """
        Returns True only when we KNOW the date is before the cutoff.
        If date is None (unknown) we keep the article — better to
        include than silently drop.
        """
        if self.earliest_date is None or date is None:
            return False
        return date < self.earliest_date

    # ── Indicator helpers ──────────────────────────────────────────────────────

    def _map_indicator_to_terms(self, indicator: str) -> list[str]:
        mapping = {
            "opd_attendance":               ["opd", "outpatient", "attendance", "consultation"],
            "ipd_total_admissions":         ["ipd", "inpatient", "admission", "hospitalization"],
            "fp_total_clients":             ["family planning", "fp", "contraception", "reproductive health"],
            "fp_subsequent_clients_total":  ["family planning", "subsequent", "follow-up"],
            "bcg_under1":                   ["bcg", "tuberculosis vaccine", "immunization"],
            "penta3_under1":                ["penta3", "pentavalent", "dpt-hepb-hib"],
            "measles1_under1":              ["measles", "mr", "measles-rubella"],
            "fully_immunised_under1":       ["fully immunized", "fully immunised", "complete vaccination"],
            "live_births_total":            ["live birth", "birth", "delivery"],
            "htc_results_new_negative":     ["hiv test", "htc", "negative", "hiv testing"],
            "htc_results_new_positive":     ["hiv positive", "htc", "positive result"],
            "anc_total_visits":             ["anc", "antenatal", "pregnancy", "maternal"],
            "cervical_screening_total":     ["cervical", "pap smear", "hpv", "cancer screening"],
        }
        return mapping.get(indicator, [indicator])

    def _find_indicators(self, text: str) -> list[str]:
        lower = text.lower()
        return [
            ind for ind in COUNT_INDICATORS
            if any(term in lower for term in self._map_indicator_to_terms(ind))
        ]

    # ── Article-level scrape ───────────────────────────────────────────────────

    def _scrape_article(
        self, page, url: str, title: str, listing_date: Optional[datetime]
    ) -> Optional[dict]:

        # ── Dedup ──────────────────────────────────────────────────────────────
        if url in self._visited_urls:
            logger.debug(f"  Already visited: {url}")
            return None
        self._visited_urls.add(url)

        # ── Early date-gate on listing date ───────────────────────────────────
        if self._is_too_old(listing_date):
            logger.info(f"  Skipping (too old per listing date): {url}")
            return None

        logger.info(f"  → Fetching article: {url}")

        if self._is_pdf(url):
            full_text   = self._extract_pdf_text(page, url)
            source_type = "pdf"
        else:
            soup = self._get_rendered_soup(page, url)
            if soup is None:
                return None
            full_text   = soup.get_text(separator=" ")
            source_type = "html"

        if not full_text.strip():
            logger.warning(f"  No text extracted from {url}")
            return None

        # Refine date from article body; re-check cutoff
        article_date = self._extract_date(full_text) or listing_date
        if self._is_too_old(article_date):
            logger.info(f"  Skipping (too old per article date): {url}")
            return None

        found_indicators = self._find_indicators(full_text)
        if not found_indicators:
            return None

        return {
            "title":            title,
            "date":             article_date.isoformat() if article_date else "Unknown",
            "link":             url,
            "source_type":      source_type,
            "indicators_found": found_indicators,
            "snippet":          full_text[:800].replace("\n", " "),
        }

    # ── Listing-page helpers ───────────────────────────────────────────────────

    def _collect_article_links(self, soup) -> list[dict]:
        """
        Robustly collect every article link on a listing page.

        Strategy (applied in order, results merged & deduped):
          1. <article> tags
          2. Common WordPress div class patterns
          3. Any <a href> whose text looks like a bulletin title
             (last-resort fallback)
        """
        seen_hrefs: set[str] = set()
        articles_meta: list[dict] = []

        def _add(href: str, title: str, date):
            if href.startswith("/"):
                href = BASE_URL + href
            if not href.startswith("http"):
                return
            if href in seen_hrefs:
                return
            seen_hrefs.add(href)
            articles_meta.append({"title": title, "url": href, "date": date})

        # ── Strategy 1: <article> tags ─────────────────────────────────────
        for article in soup.find_all("article"):
            link_tag = article.find("a", href=True)
            if not link_tag:
                continue
            heading = article.find(["h1", "h2", "h3", "h4"])
            title   = (heading.get_text(strip=True) if heading
                       else link_tag.get_text(strip=True) or "No title")
            _add(link_tag["href"], title, self._extract_date(article.get_text()))

        # ── Strategy 2: div class patterns ────────────────────────────────
        for div in soup.find_all("div", class_=re.compile(r"post|entry|article|bulletin|item", re.I)):
            link_tag = div.find("a", href=True)
            if not link_tag:
                continue
            heading = div.find(["h1", "h2", "h3", "h4"])
            title   = (heading.get_text(strip=True) if heading
                       else link_tag.get_text(strip=True) or "No title")
            _add(link_tag["href"], title, self._extract_date(div.get_text()))

        # ── Strategy 3: fallback — any internal link with meaningful text ──
        if not articles_meta:
            logger.warning("  Primary selectors found nothing — using link fallback")
            for a in soup.find_all("a", href=True):
                href  = a["href"]
                label = a.get_text(strip=True)
                # Only keep links that look like article permalinks
                if (
                    len(label) > 20
                    and BASE_URL in href
                    and not any(skip in href for skip in ["/category/", "/tag/", "/page/", "#"])
                ):
                    _add(href, label, None)

        return articles_meta

    def _next_page_url(self, soup, current_url: str) -> Optional[str]:
        """
        Robustly find the next-page URL using four different strategies.
        """
        # ── Strategy 1: explicit next link text ───────────────────────────
        for a in soup.find_all("a", href=True):
            txt = a.get_text(strip=True)
            if re.fullmatch(r"next|›|»|next\s*page|older\s*posts?", txt, re.I):
                return a["href"]

        # ── Strategy 2: WordPress nav-links / pagination block ────────────
        nav = soup.find(
            ["div", "nav"],
            class_=re.compile(r"nav-links|pagination|wp-pagenavi", re.I),
        )
        if nav:
            current_span = nav.find("span", class_=re.compile(r"current|active", re.I))
            if current_span:
                nxt = current_span.find_next_sibling("a")
                if nxt and nxt.get("href"):
                    return nxt["href"]

        # ── Strategy 3: rel="next" link element ───────────────────────────
        rel_next = soup.find("a", rel=re.compile(r"next", re.I))
        if rel_next and rel_next.get("href"):
            return rel_next["href"]

        # ── Strategy 4: increment /page/N/ in current URL ─────────────────
        page_match = re.search(r"/page/(\d+)/", current_url)
        if page_match:
            next_n    = int(page_match.group(1)) + 1
            candidate = re.sub(r"/page/\d+/", f"/page/{next_n}/", current_url)
        else:
            candidate = current_url.rstrip("/") + "/page/2/"

        # Probe: only follow if the candidate actually exists
        # (We can't probe here without a page object; caller handles 404s)
        # Return candidate and let the caller detect a dead end via empty article list.
        if candidate != current_url:
            return candidate

        return None

    # ── Paginated listing scrape ───────────────────────────────────────────────

    def _scrape_listing(self, page, start_url: str) -> list[dict]:
        """
        Paginate through a listing URL and scrape every article found.
        Stops when:
          - no more pages are found, OR
          - every article on a page is older than earliest_date
            (avoids crawling the entire archive when a cutoff is set)
        """
        results      = []
        listing_url  = start_url
        page_num     = 0
        empty_pages  = 0          # consecutive pages with no new articles

        while listing_url:
            page_num += 1
            logger.info(f"── Listing page {page_num}: {listing_url}")

            soup = self._get_rendered_soup(page, listing_url)
            if soup is None:
                logger.error("Could not render listing page — stopping pagination.")
                break

            articles_meta = self._collect_article_links(soup)
            logger.info(f"  Found {len(articles_meta)} article link(s) on this page")

            if not articles_meta:
                empty_pages += 1
                if empty_pages >= 2:
                    logger.info("  Two consecutive empty pages — ending pagination.")
                    break
            else:
                empty_pages = 0

            # ── Date-based early-stop ──────────────────────────────────────
            # If ALL articles on this page are older than the cutoff,
            # there is no point paginating further (posts are newest-first).
            if self.earliest_date and articles_meta:
                all_too_old = all(
                    self._is_too_old(m["date"]) for m in articles_meta if m["date"]
                )
                if all_too_old:
                    logger.info("  All articles on this page pre-date cutoff — stopping.")
                    break

            for meta in articles_meta:
                result = self._scrape_article(
                    page,
                    url=meta["url"],
                    title=meta["title"],
                    listing_date=meta["date"],
                )
                if result:
                    logger.info(f"  ✓ Indicators: {result['indicators_found']}")
                    results.append(result)
                else:
                    logger.debug(f"  – No indicators: {meta['url']}")

            prev_url    = listing_url
            listing_url = self._next_page_url(soup, listing_url)

            # Guard: if next URL is same as current, stop to avoid infinite loop
            if listing_url == prev_url:
                logger.warning("  Next-page URL unchanged — stopping to avoid loop.")
                break

        return results

    # ── Public entry-point ─────────────────────────────────────────────────────

    def scrape_phim_bulletins(self) -> list[dict]:
        """
        Main scrape routine.

        Entry points (all deduped via self._visited_urls):
          1. Category listing  → paginated
          2. WordPress year archives (e.g. /2019/, /2020/ …) → paginated
             These guarantee we reach older content even if pagination
             on the category page breaks.
        """
        results: list[dict] = []

        # Build all entry-point URLs
        start_year   = self.earliest_date.year if self.earliest_date else 2019
        entry_points = [self.bulletins_url] + _build_archive_urls(start_year)

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=self.headers["User-Agent"],
                locale="en-GB",
            )
            page = context.new_page()

            for entry_url in entry_points:
                logger.info(f"\n{'═'*60}")
                logger.info(f"Entry point: {entry_url}")
                logger.info(f"{'═'*60}")
                batch = self._scrape_listing(page, entry_url)
                results.extend(batch)

            browser.close()

        # Final dedup by link (in case year archives overlap with category)
        seen: set[str] = set()
        unique_results  = []
        for r in results:
            if r["link"] not in seen:
                seen.add(r["link"])
                unique_results.append(r)

        logger.info(
            f"\nScrape complete — {len(unique_results)} unique article(s) with indicators."
        )
        return unique_results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    scraper = MalawiHealthScraper(earliest_date=EARLIEST_DATE)

    logger.info("Starting PHIM bulletin scrape …")
    bulletins = scraper.scrape_phim_bulletins()
    logger.info(f"Total bulletins with matching indicators: {len(bulletins)}")

    summary_rows = [
        {
            "indicator":    indicator,
            "source_title": b["title"],
            "date":         b["date"],
            "source_type":  b["source_type"],
            "link":         b["link"],
            "snippet":      b["snippet"],
        }
        for b in bulletins
        for indicator in b["indicators_found"]
    ]

    if summary_rows:
        df        = pd.DataFrame(summary_rows)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv   = f"malawi_health_indicators_{timestamp}.csv"
        out_json  = f"malawi_health_indicators_{timestamp}.json"
        df.to_csv(out_csv, index=False)
        df.to_json(out_json, orient="records", indent=2)
        logger.info(f"Saved → {out_csv}")
        logger.info(f"Saved → {out_json}")
        print(df.to_string(index=False))
    else:
        logger.warning("No indicator data found — nothing saved.")


if __name__ == "__main__":
    main()
