"""
crawl_letterboxd_movie_centric.py

Crawler Letterboxd dạng HYBRID cho hệ thống gợi ý phim.

Điểm sửa chính so với file movie-centric cũ:
1. KHÔNG crawl /films/, /films/popular/, /films/by/rating/ để lấy seed films nữa.
   Các trang browse này dễ bị rỗng hoặc đổi layout/render nên hay làm Phase 1 = 0 film.
2. Dùng danh sách film slug cứng để crawl /film/{slug}/members/.
   Members page vẫn là HTML server-rendered và lấy được username.
3. Sau khi có seed users theo movie-overlap, quay lại luồng user-centric ổn định:
   - /{username}/rss/
   - /{username}/films/page/N/
   - /{username}/likes/films/page/N/
   - /{username}/reviews/page/N/
   - /{username}/diary/page/N/
   - /{username}/following/page/N/
4. Vẫn có resume, checkpoint, chống trùng user/movie/interaction, và k-core filtering.
5. Bản v4 lọc mạnh cache seed users cũ để tránh nhầm slug hệ thống như api-beta/gift-guide thành username.

Cách chạy khuyến nghị:
    python crawl_letterboxd_movie_centric.py --no-resume

Chạy tiếp:
    python crawl_letterboxd_movie_centric.py --resume

Nếu bị rate-limit:
    python crawl_letterboxd_movie_centric.py --resume --sleep 4 --jitter 2

Nếu CF bị rỗng:
    python crawl_letterboxd_movie_centric.py --resume --cf-min-user-interactions 5 --cf-min-movie-interactions 3
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import sys
import time
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


# ============================================================
# Config mặc định
# ============================================================

BASE_URL = "https://letterboxd.com"

SEED_USERNAMES = [
    # Fallback nếu phase movie-overlap không lấy được user.
    "dave", "karsten", "brat", "lucy", "maria", "jack", "michael",
    "sarah", "josh", "emma", "daniel", "alex", "sam", "max", "anna",
    "matthew", "laura", "ben", "nathan", "george", "helen", "chris",
    "james", "will", "tom", "oliver", "nick", "kate", "amy", "paul",
]

# Seed films dùng để khởi động movie-overlap.
# Không crawl /films/ nữa; chỉ dùng slug cứng để vào /film/{slug}/members/.
# Nên chọn phim phổ biến, nhiều người xem, đa dạng thể loại/thời kỳ để overlap cao.
SEED_FILM_SLUGS = [
    "parasite-2019",
    "everything-everywhere-all-at-once",
    "interstellar",
    "fight-club",
    "the-dark-knight",
    "pulp-fiction",
    "inception",
    "the-godfather",
    "spirited-away",
    "la-la-land",
    "whiplash-2014",
    "the-social-network",
    "get-out-2017",
    "barbie",
    "oppenheimer-2023",
    "dune-2021",
    "poor-things-2023",
    "past-lives",
    "anatomy-of-a-fall",
    "the-substance",
    "the-shawshank-redemption",
    "goodfellas",
    "the-matrix",
    "se7en",
    "portrait-of-a-lady-on-fire",
    "lady-bird",
    "call-me-by-your-name",
    "aftersun",
    "the-grand-budapest-hotel",
    "mad-max-fury-road",
]

SEED_MEMBERS_PAGES = 10
MAX_SEED_USER_CANDIDATES = 20000

# Target mở rộng: raw phải lớn hơn MovieLens để sau k-core vẫn còn ~100K.
# MovieLens-small tham chiếu: ~600 users, ~9K movies, ~100K ratings.
MIN_USERS = 600
TARGET_USERS = 1500
MAX_USERS = 3000

MIN_MOVIES = 9000
TARGET_MOVIES = 15000
MAX_MOVIES = 30000

# Đây là ngưỡng RAW/hard-cap, không phải mục tiêu CF-ready.
# Raw cần lớn hơn 100K vì k-core sẽ loại bớt user/movie thưa.
MIN_INTERACTIONS = 100000
TARGET_INTERACTIONS = 250000
MAX_INTERACTIONS = 400000

# Mục tiêu thật sau k-core: dùng để quyết định dừng crawl.
TARGET_CF_USERS = 600
TARGET_CF_MOVIES = 9000
TARGET_CF_INTERACTIONS = 100000

# Default đã chỉnh để bớt loãng: films thấp, likes/reviews/diary cao hơn.
PAGES_PER_FILMS = 2
PAGES_PER_LIKES = 8
PAGES_PER_REVIEWS = 5
PAGES_PER_DIARY = 6
PAGES_PER_FOLLOWING = 5

# K-core cho CF.
CF_MIN_USER_INTERACTIONS = 10
CF_MIN_MOVIE_INTERACTIONS = 5

# Crawl lịch sự.
SLEEP_SECONDS = 2.5
JITTER_SECONDS = 1.5
MAX_RETRIES = 4
BACKOFF_SECONDS = 8
COOLDOWN_ON_429_503 = 45
CHECKPOINT_EVERY_USERS = 5

# Tự nhận diện thư mục gốc để chạy được cả khi file nằm trong /crawler hoặc ngay project root.
_THIS_FILE = Path(__file__).resolve()
if _THIS_FILE.parent.name.lower() in {"crawler", "crawlers", "scraper", "scrapers", "scripts"}:
    ROOT_DIR = _THIS_FILE.parents[1]
else:
    ROOT_DIR = _THIS_FILE.parent

RAW_DIR = ROOT_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

USERS_CSV = RAW_DIR / "users.csv"
MOVIES_CSV = RAW_DIR / "movies_seed.csv"
INTERACTIONS_CSV = RAW_DIR / "interactions.csv"
RATINGS_CSV = RAW_DIR / "ratings.csv"
INTERACTIONS_CF_CSV = RAW_DIR / "interactions_cf.csv"
RATINGS_CF_CSV = RAW_DIR / "ratings_cf.csv"
MOVIES_CF_CSV = RAW_DIR / "movies_cf.csv"
CRAWL_REPORT_TXT = RAW_DIR / "crawl_report.txt"
CRAWL_STATE_CSV = RAW_DIR / "crawl_state.csv"
SEED_USERS_CSV = RAW_DIR / "seed_user_candidates.csv"
BACKUP_ROOT_DIR = RAW_DIR / "backups"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
}


@dataclass
class CrawlConfig:
    min_users: int = MIN_USERS
    target_users: int = TARGET_USERS
    max_users: int = MAX_USERS

    min_movies: int = MIN_MOVIES
    target_movies: int = TARGET_MOVIES
    max_movies: int = MAX_MOVIES

    min_interactions: int = MIN_INTERACTIONS
    target_interactions: int = TARGET_INTERACTIONS
    max_interactions: int = MAX_INTERACTIONS

    # Đích sau k-core. Không dùng raw interactions để dừng sớm nữa.
    target_cf_users: int = TARGET_CF_USERS
    target_cf_movies: int = TARGET_CF_MOVIES
    target_cf_interactions: int = TARGET_CF_INTERACTIONS

    pages_per_films: int = PAGES_PER_FILMS
    pages_per_likes: int = PAGES_PER_LIKES
    pages_per_reviews: int = PAGES_PER_REVIEWS
    pages_per_diary: int = PAGES_PER_DIARY
    pages_per_following: int = PAGES_PER_FOLLOWING

    cf_min_user_interactions: int = CF_MIN_USER_INTERACTIONS
    cf_min_movie_interactions: int = CF_MIN_MOVIE_INTERACTIONS

    sleep_seconds: float = SLEEP_SECONDS
    jitter_seconds: float = JITTER_SECONDS
    max_retries: int = MAX_RETRIES
    backoff_seconds: float = BACKOFF_SECONDS
    cooldown_on_429_503: float = COOLDOWN_ON_429_503
    checkpoint_every_users: int = CHECKPOINT_EVERY_USERS

    seed_members_pages: int = SEED_MEMBERS_PAGES
    max_seed_user_candidates: int = MAX_SEED_USER_CANDIDATES
    skip_movie_overlap_seed: bool = False
    rebuild_seed_users: bool = False

    resume: bool = True
    backup_before_run: bool = True
    debug_urls: bool = False


DEFAULT_CONFIG = CrawlConfig()


# ============================================================
# HTTP
# ============================================================

def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(HEADERS)
    retry_strategy = Retry(
        total=2,
        connect=2,
        read=2,
        status=2,
        status_forcelist=[429, 500, 502, 504],
        allowed_methods=["GET"],
        backoff_factor=1.5,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = build_session()


def fetch_url(url: str, config: CrawlConfig, accept_xml: bool = False) -> Optional[str]:
    headers = dict(HEADERS)
    if accept_xml:
        headers["Accept"] = "application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.8"

    last_status = None
    for attempt in range(1, config.max_retries + 1):
        try:
            if config.debug_urls:
                print(f"[GET] {url}", flush=True)
            response = SESSION.get(url, headers=headers, timeout=30)
            last_status = response.status_code

            if response.status_code == 200:
                return response.text

            if response.status_code in {401, 403, 404, 410}:
                print(f"[WARN] Skip {url}. Status={response.status_code}", flush=True)
                return None

            if response.status_code in {429, 503}:
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = int(retry_after)
                else:
                    wait = config.cooldown_on_429_503 + config.backoff_seconds * attempt + random.random() * 5
                print(f"[WARN] Status={response.status_code}. Cooldown {wait:.1f}s retry {attempt}/{config.max_retries}: {url}", flush=True)
                time.sleep(wait)
                continue

            wait = config.backoff_seconds * attempt + random.random() * 3
            print(f"[WARN] Status={response.status_code}. Retry after {wait:.1f}s: {url}", flush=True)
            time.sleep(wait)

        except requests.RequestException as exc:
            wait = config.backoff_seconds * attempt + random.random() * 3
            print(f"[ERROR] Request failed retry {attempt}/{config.max_retries}: {url} - {exc}", flush=True)
            time.sleep(wait)

    print(f"[WARN] Give up {url}. Last status={last_status}", flush=True)
    return None


def sleep_polite(config: CrawlConfig) -> None:
    time.sleep(config.sleep_seconds + random.random() * config.jitter_seconds)


# ============================================================
# Utils
# ============================================================

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_text(text: Optional[object]) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_username(username: str) -> str:
    return clean_text(username).strip("/").lower()


# Các slug hệ thống/điều hướng của Letterboxd, không phải username thật.
# Nếu không lọc kỹ, crawler sẽ crawl nhầm /pro/, /apps/, /year-in-review/... và sinh 404 hàng loạt.
LETTERBOXD_NON_USER_SLUGS = {
    "about", "activity", "apps", "boxd", "changes", "crew", "create-account",
    "film", "films", "journal", "jobs", "legal", "list", "lists",
    "members", "popular", "press", "pro", "search", "signin", "sign-in",
    "stats", "tag", "tags", "watchlist", "welcome", "year-in-review",
    "actor", "actors", "director", "directors", "writer", "writers",
    "cinemas", "countries", "genres", "languages", "newsletter", "patrons",
    "settings", "shop", "tmdb", "trailer", "trailers", "username",
    # Footer/static/API paths — nếu không loại, members page có thể lọt vào queue như user giả.
    "api", "api-beta", "help", "terms", "contact", "news", "video-store",
    "gifts", "gift-guide", "gift-card", "gift-cards", "shop", "store", "merch",
    "upgrade", "features", "mobile", "forgotten", "reset-password",
    "password", "email", "invite", "import", "export", "films-you-own",
    "beta", "ajax", "static", "sitemap", "rss", "robots", "security",
    "privacy", "privacy-policy", "cookie-policy", "cookies", "guidelines",
    "community", "community-policy", "brand", "advertise", "advertising",
    "media", "podcast", "festivals", "festival", "awards", "studios",
    "giftguide", "lists-popular", "crew-picks", "hq", "labs",
}


def is_valid_letterboxd_username(candidate: str) -> bool:
    """Trả về True nếu candidate có khả năng là username Letterboxd thật."""
    candidate = normalize_username(candidate)
    if not candidate:
        return False

    if candidate in LETTERBOXD_NON_USER_SLUGS:
        return False

    # Loại các URL path nhiều tầng hoặc có ký tự không thuộc username.
    if "/" in candidate or "." in candidate:
        return False

    # Loại thêm các slug trông giống endpoint/trang tĩnh, không phải profile người dùng.
    if candidate.startswith((
        "api-", "ajax-", "static-", "film-", "list-", "tag-", "year-",
        "video-", "mobile-", "help-", "news-", "admin-", "gift-",
    )):
        return False
    if candidate.endswith(("-guide", "-policy", "-store", "-beta", "-api")):
        return False

    # Username Letterboxd thường là chữ/số, có thể có gạch dưới hoặc gạch ngang.
    if not re.fullmatch(r"[a-z0-9_-]{2,30}", candidate):
        return False

    # Loại vài dạng slug điều hướng phổ biến.
    if candidate.startswith(("film-", "list-", "tag-", "year-")):
        return False

    return True


def make_id(text: str, prefix: str) -> str:
    normalized = clean_text(text).lower()
    digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def parse_int(text: object) -> str:
    raw = clean_text(text)
    match = re.search(r"[\d,\.]+", raw)
    if not match:
        return ""
    value = match.group(0).replace(",", "").replace(".", "")
    return value if value.isdigit() else ""


def parse_year_from_text(text: str) -> str:
    match = re.search(r"\b(19\d{2}|20\d{2})\b", text or "")
    return match.group(1) if match else ""




def is_probably_user_profile(username: str, html: str) -> bool:
    """Kiểm tra HTML có thật sự là profile user Letterboxd không.

    Bản v3 còn quá lỏng vì chỉ cần thấy link /{slug}/ là cho qua, nên các trang tĩnh
    như /gift-guide/ vẫn có thể bị crawl tiếp thành /gift-guide/rss/ và sinh 404/403.
    Bản v4 chỉ chấp nhận khi trang có các đường dẫn đặc trưng của profile thật:
    /{username}/films/, /{username}/following/, /{username}/followers/, /{username}/rss/...
    """
    username = normalize_username(username)
    if not html or not username or not is_valid_letterboxd_username(username):
        return False

    # Profile thật gần như luôn có ít nhất một trong các link nội bộ này.
    strong_signals = [
        f'href="/{username}/films/"',
        f"href='/{username}/films/'",
        f'href="/{username}/following/"',
        f"href='/{username}/following/'",
        f'href="/{username}/followers/"',
        f"href='/{username}/followers/'",
        f'href="/{username}/diary/"',
        f"href='/{username}/diary/'",
        f'href="/{username}/rss/"',
        f"href='/{username}/rss/'",
    ]
    if any(sig in html for sig in strong_signals):
        return True

    soup = BeautifulSoup(html, "html.parser")

    # Fallback có kiểm soát: tìm nav/profile stats có link /username/films hoặc /username/following.
    # Không chấp nhận chỉ vì có self-link /username/, vì trang tĩnh cũng có thể có self-link.
    for link in soup.select("a[href]"):
        href = str(link.get("href", ""))
        if href in {f"/{username}/films/", f"/{username}/following/", f"/{username}/followers/", f"/{username}/diary/"}:
            return True

    return False

def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def csv_read(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def csv_write_atomic(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp_path.replace(path)


def backup_existing_raw_files() -> Optional[Path]:
    """Tạo bản sao CSV cũ trước khi resume để tránh mất dữ liệu nếu chạy lỗi giữa chừng."""
    existing_files = [
        USERS_CSV, MOVIES_CSV, INTERACTIONS_CSV, RATINGS_CSV,
        INTERACTIONS_CF_CSV, RATINGS_CF_CSV, MOVIES_CF_CSV,
        CRAWL_STATE_CSV, SEED_USERS_CSV, CRAWL_REPORT_TXT,
    ]
    existing_files = [path for path in existing_files if path.exists()]
    if not existing_files:
        return None

    backup_dir = BACKUP_ROOT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in existing_files:
        shutil.copy2(path, backup_dir / path.name)
    return backup_dir


def deduplicate_by_key(rows: Iterable[Dict], key_fields: Iterable[str]) -> List[Dict]:
    seen: Set[Tuple] = set()
    result: List[Dict] = []
    for row in rows:
        key = tuple(row.get(field, "") for field in key_fields)
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


# ============================================================
# Row builders
# ============================================================

def build_user_row(username: str, profile: Optional[Dict] = None) -> Dict:
    username = normalize_username(username)
    profile = profile or {}
    return {
        "user_id": make_id(username, "user"),
        "username": username,
        "display_name": clean_text(profile.get("display_name", "")),
        "location": clean_text(profile.get("location", "")),
        "profile_url": f"{BASE_URL}/{username}/",
        "films_count": clean_text(profile.get("films_count", "")),
        "this_year_count": clean_text(profile.get("this_year_count", "")),
        "lists_count": clean_text(profile.get("lists_count", "")),
        "following_count": clean_text(profile.get("following_count", "")),
        "followers_count": clean_text(profile.get("followers_count", "")),
        "member_since": clean_text(profile.get("member_since", "")),
        "bio": clean_text(profile.get("bio", "")),
        "created_at": now_iso(),
    }


def build_movie_row(movie_id: str, title: str, year: str, movie_url: str) -> Dict:
    return {
        "movie_id": movie_id,
        "title": clean_text(title),
        "year": clean_text(year),
        "movie_url": clean_text(movie_url),
        "created_at": now_iso(),
    }


def build_interaction_row(
    user_id: str,
    movie_id: str,
    interaction_type: str,
    rating: Optional[float],
    implicit_score: float,
    source: str,
    watched_date: str = "",
) -> Dict:
    return {
        "user_id": user_id,
        "movie_id": movie_id,
        "interaction_type": clean_text(interaction_type),
        "rating": rating if rating is not None else "",
        "implicit_score": implicit_score,
        "source": clean_text(source),
        "watched_date": clean_text(watched_date),
        "created_at": now_iso(),
    }


# ============================================================
# Rating parsing
# ============================================================

def parse_rating_from_text(text: str) -> Optional[float]:
    if not text:
        return None
    matches = re.findall(r"[★½]+", text)
    if not matches:
        return None
    rating_text = max(matches, key=len)
    stars = rating_text.count("★")
    half = 0.5 if "½" in rating_text else 0.0
    rating = stars + half
    if 0 < rating <= 5:
        return rating
    return None


def parse_rating_from_classes(tag: Tag) -> Optional[float]:
    candidates: List[Tag] = []
    if isinstance(tag, Tag):
        candidates.append(tag)
        candidates.extend([t for t in tag.select("span.rating, p.rating, .rating") if isinstance(t, Tag)])
    for candidate in candidates:
        for cls in candidate.get("class", []):
            match = re.match(r"rated-(\d+)", str(cls))
            if match:
                value = int(match.group(1)) / 2
                if 0 < value <= 5:
                    return value
    text_rating = parse_rating_from_text(tag.get_text(" ", strip=True)) if isinstance(tag, Tag) else None
    return text_rating


# ============================================================
# Movie extraction from Letterboxd cards/posters
# ============================================================

def normalize_film_url(value: str) -> str:
    value = clean_text(value)
    if not value:
        return ""
    if value.startswith("http"):
        return value
    if value.startswith("/film/"):
        return urljoin(BASE_URL, value)
    if value.startswith("/"):
        return urljoin(BASE_URL, value)
    if "/" not in value:
        return f"{BASE_URL}/film/{value}/"
    return urljoin(BASE_URL, value)


def extract_movie_from_card(card: Tag) -> Optional[Tuple[str, str, str, str]]:
    if not isinstance(card, Tag):
        return None

    # Letterboxd poster thường có data-film-slug/name/target-link ở chính nó hoặc node con.
    film_node = card
    if not any(film_node.get(a) for a in ["data-film-slug", "data-film-name", "data-target-link", "data-film-title"]):
        found = card.select_one("[data-film-slug], [data-film-name], [data-target-link], [data-film-title]")
        if isinstance(found, Tag):
            film_node = found

    title = clean_text(
        film_node.get("data-film-name")
        or film_node.get("data-film-title")
        or film_node.get("data-item-name")
        or ""
    )
    year = clean_text(film_node.get("data-film-year") or "")

    link = ""
    for attr in ["data-target-link", "data-film-link", "data-film-slug"]:
        raw = film_node.get(attr)
        if raw:
            link = normalize_film_url(str(raw))
            break

    a_tag = card.select_one("a[href*='/film/']")
    if isinstance(a_tag, Tag) and a_tag.get("href"):
        link = normalize_film_url(str(a_tag.get("href")))

    if not title:
        img = card.select_one("img[alt]")
        if isinstance(img, Tag):
            title = clean_text(img.get("alt"))

    if not title:
        for attr in ["title", "aria-label"]:
            value = clean_text(card.get(attr))
            if value:
                title = value
                break

    if not title and link:
        slug = link.rstrip("/").split("/")[-1]
        title = slug.replace("-", " ").title()

    if not year:
        year = parse_year_from_text(card.get_text(" ", strip=True))

    if not title:
        return None

    movie_id = make_id(link or f"{title}_{year}", "movie")
    return movie_id, title, year, link


def movie_cards_from_soup(soup: BeautifulSoup) -> List[Tag]:
    selectors = [
        "li.poster-container",
        "div.poster-container",
        "li.film-detail",
        "li.film-poster-container",
        "div.film-poster",
        "li.griditem",
        "li.js-listitem",
        "article.film-detail",
    ]
    cards: List[Tag] = []
    seen_keys: Set[str] = set()

    for selector in selectors:
        for tag in soup.select(selector):
            if not isinstance(tag, Tag):
                continue
            extracted = extract_movie_from_card(tag)
            key = extracted[3] if extracted else str(id(tag))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            cards.append(tag)
    return cards


# ============================================================
# Profile parsing based on current Letterboxd profile layout
# ============================================================

def parse_profile_stats_from_links(username: str, soup: BeautifulSoup) -> Dict[str, str]:
    stats = {
        "films_count": "",
        "this_year_count": "",
        "lists_count": "",
        "following_count": "",
        "followers_count": "",
    }

    uname = normalize_username(username)
    for a in soup.select("a[href]"):
        if not isinstance(a, Tag):
            continue
        href = str(a.get("href", ""))
        text = a.get_text(" ", strip=True)
        number = parse_int(text)
        if not number:
            continue

        # Các link trên profile thường có dạng /dave/films/, /dave/following/...
        if re.fullmatch(rf"/{re.escape(uname)}/films/?", href):
            stats["films_count"] = number
        elif "/films/this/year" in href or "this-year" in href:
            stats["this_year_count"] = number
        elif re.fullmatch(rf"/{re.escape(uname)}/lists/?", href):
            stats["lists_count"] = number
        elif re.fullmatch(rf"/{re.escape(uname)}/following/?", href):
            stats["following_count"] = number
        elif re.fullmatch(rf"/{re.escape(uname)}/followers/?", href):
            stats["followers_count"] = number

    # Fallback: đọc text ở cụm profile stats nếu selector phía trên không bắt được.
    if not any(stats.values()):
        text = soup.get_text(" ", strip=True)
        patterns = {
            "films_count": r"([\d,\.]+)\s+FILMS",
            "this_year_count": r"([\d,\.]+)\s+THIS YEAR",
            "lists_count": r"([\d,\.]+)\s+LISTS",
            "following_count": r"([\d,\.]+)\s+FOLLOWING",
            "followers_count": r"([\d,\.]+)\s+FOLLOWERS",
        }
        for key, pattern in patterns.items():
            m = re.search(pattern, text, flags=re.I)
            if m:
                stats[key] = parse_int(m.group(1))

    return stats


def parse_profile_info(username: str, html: str) -> Tuple[Dict, List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Trả về:
    - profile_info
    - favorite interactions
    - favorite movies
    - recent interactions
    - recent movies
    """
    soup = BeautifulSoup(html, "html.parser")
    user_id = make_id(username, "user")

    display_name = ""
    for selector in ["h1.title-1", "h1.profile-name", "h1", ".displayname", ".person-summary h1"]:
        node = soup.select_one(selector)
        if isinstance(node, Tag):
            display_name = clean_text(node.get_text(" ", strip=True))
            display_name = re.sub(r"\bPATRON\b.*$", "", display_name).strip()
            if display_name:
                break

    location = ""
    for selector in [".profile-location", ".person-location", "p.location", ".metadata"]:
        node = soup.select_one(selector)
        if isinstance(node, Tag):
            text = clean_text(node.get_text(" ", strip=True))
            if text and len(text) <= 80:
                location = text
                break

    bio = ""
    for selector in [".profile-about", ".bio", ".body-text", ".profile-text", ".person-bio"]:
        node = soup.select_one(selector)
        if isinstance(node, Tag):
            text = clean_text(node.get_text(" ", strip=True))
            if text and len(text) > 10:
                bio = text[:1000]
                break

    member_since = ""
    page_text = soup.get_text(" ", strip=True)
    m = re.search(r"Member since\s+([^\s]+)", page_text, flags=re.I)
    if m:
        member_since = m.group(1)

    profile = parse_profile_stats_from_links(username, soup)
    profile.update({
        "display_name": display_name,
        "location": location,
        "bio": bio,
        "member_since": member_since,
    })

    fav_interactions, fav_movies = parse_profile_movie_section(
        soup=soup,
        username=username,
        user_id=user_id,
        heading_keywords=["favorite films", "favourite films"],
        interaction_type="favorite",
        rating=5.0,
        implicit_score=5.0,
        source="profile_favorites",
        limit=4,
    )

    recent_interactions, recent_movies = parse_profile_movie_section(
        soup=soup,
        username=username,
        user_id=user_id,
        heading_keywords=["recent activity"],
        interaction_type="watched",
        rating=None,
        implicit_score=2.5,
        source="profile_recent_activity",
        limit=12,
    )

    return profile, fav_interactions, fav_movies, recent_interactions, recent_movies


def parse_profile_movie_section(
    soup: BeautifulSoup,
    username: str,
    user_id: str,
    heading_keywords: List[str],
    interaction_type: str,
    rating: Optional[float],
    implicit_score: float,
    source: str,
    limit: int,
) -> Tuple[List[Dict], List[Dict]]:
    heading = None
    for h in soup.find_all(["h2", "h3", "h4"]):
        if not isinstance(h, Tag):
            continue
        text = h.get_text(" ", strip=True).lower()
        if any(keyword in text for keyword in heading_keywords):
            heading = h
            break

    scope: Optional[Tag] = None
    if heading is not None:
        # Đi lần lượt qua các sibling sau heading, lấy block đầu tiên có poster.
        for sibling in heading.find_all_next(["ul", "div", "section"], limit=8):
            if not isinstance(sibling, Tag):
                continue
            test_soup = BeautifulSoup(str(sibling), "html.parser")
            if movie_cards_from_soup(test_soup):
                scope = sibling
                break

    if scope is None:
        return [], []

    scoped_soup = BeautifulSoup(str(scope), "html.parser")
    cards = movie_cards_from_soup(scoped_soup)[:limit]

    movies: List[Dict] = []
    interactions: List[Dict] = []
    seen_movies: Set[str] = set()
    for card in cards:
        extracted = extract_movie_from_card(card)
        if not extracted:
            continue
        movie_id, title, year, movie_url = extracted
        if movie_id in seen_movies:
            continue
        seen_movies.add(movie_id)
        movies.append(build_movie_row(movie_id, title, year, movie_url))
        interactions.append(
            build_interaction_row(
                user_id=user_id,
                movie_id=movie_id,
                interaction_type=interaction_type,
                rating=rating,
                implicit_score=implicit_score,
                source=source,
            )
        )

    return interactions, movies


# ============================================================
# RSS parsing
# ============================================================

def get_child_text(item: ET.Element, tag_name: str) -> str:
    for child in item:
        pure_tag = child.tag.split("}")[-1]
        if pure_tag == tag_name:
            return clean_text(child.text)
    return ""


def fallback_parse_title(title_raw: str) -> str:
    title = title_raw or ""
    title = re.sub(r"\s*[★½]+.*$", "", title).strip()
    title = re.sub(r"^.* watched ", "", title).strip()
    title = re.sub(r"^.* reviewed ", "", title).strip()
    title = re.sub(r"^.* liked ", "", title).strip()
    title = re.sub(r",\s*(19\d{2}|20\d{2}).*$", "", title).strip()
    return title


def parse_rss_items(username: str, xml_text: str) -> Tuple[List[Dict], List[Dict]]:
    user_id = make_id(username, "user")
    interactions: List[Dict] = []
    movies: List[Dict] = []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        print(f"[WARN] Cannot parse RSS for {username}: {exc}", flush=True)
        return interactions, movies

    for item in root.findall(".//item"):
        title_raw = get_child_text(item, "title")
        link = get_child_text(item, "link")
        description = get_child_text(item, "description")
        pub_date = get_child_text(item, "pubDate")

        film_title = get_child_text(item, "filmTitle") or fallback_parse_title(title_raw)
        film_year = get_child_text(item, "filmYear") or parse_year_from_text(title_raw)
        member_rating = get_child_text(item, "memberRating")

        if not film_title:
            continue

        movie_id = make_id(link or f"{film_title}_{film_year}", "movie")
        movies.append(build_movie_row(movie_id, film_title, film_year, link))

        rating: Optional[float] = None
        if member_rating:
            try:
                rating = float(member_rating)
            except ValueError:
                rating = None
        rating = rating or parse_rating_from_text(description) or parse_rating_from_text(title_raw)

        lowered = title_raw.lower()
        if rating is not None:
            interaction_type = "rating"
            score = rating
        elif "liked" in lowered:
            interaction_type = "liked"
            score = 4.0
        elif "reviewed" in lowered:
            interaction_type = "review"
            score = 3.5
        else:
            interaction_type = "watched"
            score = 2.5

        interactions.append(
            build_interaction_row(user_id, movie_id, interaction_type, rating, score, "rss", pub_date)
        )

    return interactions, movies


# ============================================================
# HTML paginated sources
# ============================================================

def crawl_paginated_movie_page(
    username: str,
    path_template: str,
    pages: int,
    interaction_type: str,
    default_score: float,
    source: str,
    use_rating_if_found: bool = True,
    config: CrawlConfig = DEFAULT_CONFIG,
) -> Tuple[List[Dict], List[Dict]]:
    user_id = make_id(username, "user")
    interactions: List[Dict] = []
    movies: List[Dict] = []
    seen_page_movie_ids: Set[str] = set()

    empty_pages = 0
    for page in range(1, pages + 1):
        url = f"{BASE_URL}/{username}/{path_template.format(page=page)}"
        html = fetch_url(url, config=config)
        sleep_polite(config)
        if not html:
            empty_pages += 1
            if empty_pages >= 2:
                break
            continue

        soup = BeautifulSoup(html, "html.parser")
        cards = movie_cards_from_soup(soup)
        if not cards:
            empty_pages += 1
            if empty_pages >= 2:
                break
            continue

        page_added = 0
        for card in cards:
            extracted = extract_movie_from_card(card)
            if not extracted:
                continue
            movie_id, title, year, movie_url = extracted
            if movie_id in seen_page_movie_ids:
                continue
            seen_page_movie_ids.add(movie_id)

            movies.append(build_movie_row(movie_id, title, year, movie_url))

            rating = parse_rating_from_classes(card) if use_rating_if_found else None
            final_type = "rating" if rating is not None and use_rating_if_found else interaction_type
            score = rating if rating is not None and use_rating_if_found else default_score

            interactions.append(
                build_interaction_row(user_id, movie_id, final_type, rating, score, source)
            )
            page_added += 1

        if page_added == 0:
            empty_pages += 1
        else:
            empty_pages = 0

    return interactions, movies


def crawl_user_interactions(username: str, config: CrawlConfig) -> Tuple[Dict, List[Dict], List[Dict]]:
    """Crawl toàn bộ data của một user."""
    all_interactions: List[Dict] = []
    all_movies: List[Dict] = []
    profile_info: Dict = {}

    # Profile: stats + favorite films + recent activity.
    # Nếu profile không tồn tại hoặc là trang hệ thống, dừng luôn để tránh sinh hàng loạt 404
    # như /api-beta/rss/, /api-beta/films/page/1/, ...
    profile_html = fetch_url(f"{BASE_URL}/{username}/", config=config)
    sleep_polite(config)
    if not profile_html or not is_probably_user_profile(username, profile_html):
        return {"_invalid_user": "1"}, [], []

    profile_info, fav_i, fav_m, recent_i, recent_m = parse_profile_info(username, profile_html)
    all_interactions.extend(fav_i)
    all_movies.extend(fav_m)
    all_interactions.extend(recent_i)
    all_movies.extend(recent_m)

    # RSS: rating/review/activity gần đây.
    rss_xml = fetch_url(f"{BASE_URL}/{username}/rss/", config=config, accept_xml=True)
    sleep_polite(config)
    if rss_xml:
        rss_i, rss_m = parse_rss_items(username, rss_xml)
        all_interactions.extend(rss_i)
        all_movies.extend(rss_m)

    # Multi-source HTML. Films để thấp để tránh movies explode.
    sources = [
        ("films/page/{page}/", config.pages_per_films, "watched", 2.5, "films_page", True),
        ("likes/films/page/{page}/", config.pages_per_likes, "liked", 4.0, "likes_page", True),
        ("reviews/page/{page}/", config.pages_per_reviews, "review", 3.5, "reviews_page", True),
        ("diary/page/{page}/", config.pages_per_diary, "diary", 3.0, "diary_page", True),
    ]
    for args in sources:
        interactions, movies = crawl_paginated_movie_page(username, *args, config=config)
        all_interactions.extend(interactions)
        all_movies.extend(movies)

    return profile_info, all_interactions, all_movies


def get_following_users(username: str, config: CrawlConfig) -> List[str]:
    following_users: List[str] = []
    seen: Set[str] = set()

    for page in range(1, config.pages_per_following + 1):
        if page == 1:
            url = f"{BASE_URL}/{username}/following/"
        else:
            url = f"{BASE_URL}/{username}/following/page/{page}/"

        html = fetch_url(url, config=config)
        sleep_polite(config)
        if not html:
            continue

        soup = BeautifulSoup(html, "html.parser")
        page_users: List[str] = []

        for link in soup.find_all("a", href=True):
            if not isinstance(link, Tag):
                continue
            href = str(link.get("href", ""))
            classes = " ".join(link.get("class", []))
            text = clean_text(link.get_text(" ", strip=True))

            looks_like_user_link = (
                href.startswith("/")
                and href.count("/") <= 2
                and not href.startswith((
                    "/film/", "/films/", "/list/", "/lists/", "/journal/",
                    "/crew/", "/actor/", "/director/", "/search/", "/about/",
                ))
            )
            has_user_signal = "name" in classes or "avatar" in classes or bool(text)
            if looks_like_user_link and has_user_signal:
                candidate = normalize_username(href.strip("/").split("/")[0])
                if is_valid_letterboxd_username(candidate):
                    page_users.append(candidate)

        for user in page_users:
            if user not in seen:
                seen.add(user)
                following_users.append(user)

        if not page_users:
            break

    return following_users


# ============================================================
# Movie-overlap seed phase: /film/{slug}/members/
# ============================================================

def extract_usernames_from_members_soup(soup: BeautifulSoup) -> List[str]:
    """Lấy username từ trang /film/{slug}/members/.

    Bản v4 không quét toàn bộ link /abc/ một cách mù nữa, vì footer của Letterboxd
    cũng có các link dạng /api-beta/, /help/, /terms/... dễ bị nhầm thành user.
    """
    usernames: List[str] = []
    seen: Set[str] = set()

    def add_candidate(raw: str) -> None:
        username = normalize_username(raw)
        if is_valid_letterboxd_username(username) and username not in seen:
            seen.add(username)
            usernames.append(username)

    # Ưu tiên vùng bảng/list member nếu selector bắt được.
    member_scopes = soup.select(
        "table.person-table tr, table.member-table tr, .person-table tr, "
        ".member-list li, .person-list li, .table-person, .person-summary, "
        "section.section table tr"
    )

    scopes = member_scopes if member_scopes else []
    if not scopes:
        # Fallback: chỉ dùng phần nội dung chính, tránh footer/nav.
        main = soup.select_one("main#content, main, #content")
        scopes = [main] if isinstance(main, Tag) else [soup]

    for scope in scopes:
        if not isinstance(scope, Tag):
            continue
        for link in scope.find_all("a", href=True):
            if not isinstance(link, Tag):
                continue
            href = str(link.get("href", ""))
            match = re.fullmatch(r"/([A-Za-z0-9_-]{2,30})/", href)
            if not match:
                continue

            classes = " ".join(str(c).lower() for c in link.get("class", []))
            text = clean_text(link.get_text(" ", strip=True))
            title = clean_text(link.get("title", ""))
            aria = clean_text(link.get("aria-label", ""))

            # Link user trong members page thường nằm ở avatar/name/person row.
            # Nếu đang ở fallback main content thì KHÔNG cho qua chỉ vì có text,
            # vì footer/menu có nhiều link dạng /gift-guide/, /api-beta/...
            parent_classes = " ".join(
                str(c).lower()
                for parent in link.parents
                if isinstance(parent, Tag)
                for c in parent.get("class", [])
            )
            class_blob = f"{classes} {parent_classes}"
            scoped_by_member_selector = bool(member_scopes)
            has_member_signal = (
                "avatar" in class_blob
                or "name" in class_blob
                or "person" in class_blob
                or "member" in class_blob
                or "table-person" in class_blob
                or bool(title)
                or bool(aria)
                or (scoped_by_member_selector and bool(text))
            )
            if has_member_signal:
                add_candidate(match.group(1))

    return usernames


def crawl_film_members(slug: str, config: CrawlConfig) -> List[str]:
    """Crawl /film/{slug}/members/page/N/ và trả về username list."""
    slug = clean_text(slug).strip("/")
    if not slug:
        return []

    found: List[str] = []
    seen: Set[str] = set()

    for page in range(1, max(1, config.seed_members_pages) + 1):
        if page == 1:
            url = f"{BASE_URL}/film/{slug}/members/"
        else:
            url = f"{BASE_URL}/film/{slug}/members/page/{page}/"

        if config.debug_urls:
            print(f"[SEED members] {url}", flush=True)

        html = fetch_url(url, config=config)
        sleep_polite(config)
        if not html:
            break

        soup = BeautifulSoup(html, "html.parser")
        page_users = extract_usernames_from_members_soup(soup)
        if not page_users:
            break

        for username in page_users:
            if username not in seen:
                seen.add(username)
                found.append(username)

    return found


def save_seed_user_candidates(user_counter: Counter) -> None:
    rows = [
        {
            "username": username,
            "overlap_count": count,
            "created_at": now_iso(),
        }
        for username, count in user_counter.most_common()
        if is_valid_letterboxd_username(username)
    ]
    csv_write_atomic(SEED_USERS_CSV, rows, ["username", "overlap_count", "created_at"])


def load_seed_user_candidates(store: "DatasetStore", config: CrawlConfig) -> List[str]:
    if config.rebuild_seed_users:
        return []
    if not config.resume or not SEED_USERS_CSV.exists():
        return []

    rows = csv_read(SEED_USERS_CSV)
    scored: List[Tuple[int, str]] = []
    cleaned_rows: List[Dict] = []
    dropped = 0

    for row in rows:
        username = normalize_username(row.get("username", ""))
        if not is_valid_letterboxd_username(username):
            dropped += 1
            continue
        if username in store.crawled_usernames:
            continue
        try:
            count = int(row.get("overlap_count", "1") or 1)
        except ValueError:
            count = 1
        count = max(1, count)
        scored.append((-count, username))
        cleaned_rows.append({
            "username": username,
            "overlap_count": count,
            "created_at": clean_text(row.get("created_at", "")) or now_iso(),
        })

    if dropped:
        # Tự làm sạch cache để lần chạy sau không bị dính lại slug hệ thống cũ.
        csv_write_atomic(SEED_USERS_CSV, cleaned_rows, ["username", "overlap_count", "created_at"])
        print(f"[SEED] Dropped {dropped} invalid cached seed users and rewrote {SEED_USERS_CSV}", flush=True)

    scored.sort()
    return [username for _, username in scored[: config.max_seed_user_candidates]]

def build_movie_overlap_seed_users(store: "DatasetStore", config: CrawlConfig) -> List[str]:
    """
    Tạo seed user list từ các phim phổ biến.
    User xuất hiện trong nhiều members pages sẽ được ưu tiên crawl trước.
    """
    if config.skip_movie_overlap_seed:
        return []

    cached = load_seed_user_candidates(store, config)
    if cached:
        print(f"[SEED] Loaded {len(cached)} movie-overlap users from {SEED_USERS_CSV}", flush=True)
        return cached

    print("[SEED] Building movie-overlap users from hard-coded film slugs...", flush=True)
    user_counter: Counter = Counter()

    with tqdm(total=len(SEED_FILM_SLUGS), desc="Seed films → members") as pbar:
        for slug in SEED_FILM_SLUGS:
            members = crawl_film_members(slug, config=config)
            added = 0
            for username in members:
                if not is_valid_letterboxd_username(username):
                    continue
                if username in store.crawled_usernames:
                    continue
                user_counter[username] += 1
                added += 1

            pbar.set_postfix(slug=slug[:18], members=len(members), unique=len(user_counter))
            pbar.update(1)
            if len(user_counter) >= config.max_seed_user_candidates:
                # Đủ rộng rồi thì dừng sớm để tránh crawl seed quá lâu.
                break

    save_seed_user_candidates(user_counter)
    seed_users = [
        username
        for username, _count in user_counter.most_common(config.max_seed_user_candidates)
        if username not in store.crawled_usernames
    ]
    print(f"[SEED] Built {len(seed_users)} candidate users -> {SEED_USERS_CSV}", flush=True)
    return seed_users


# ============================================================
# Dataset store
# ============================================================

def interaction_priority(row: Dict) -> Tuple[float, int]:
    type_priority = {
        "favorite": 6,
        "rating": 5,
        "liked": 4,
        "review": 3,
        "diary": 2,
        "watched": 1,
    }
    return (
        safe_float(row.get("implicit_score"), 0.0),
        type_priority.get(clean_text(row.get("interaction_type")), 0),
    )


def merge_interactions(rows: Iterable[Dict]) -> List[Dict]:
    best: Dict[Tuple[str, str], Dict] = {}
    for row in rows:
        user_id = row.get("user_id", "")
        movie_id = row.get("movie_id", "")
        if not user_id or not movie_id:
            continue
        key = (user_id, movie_id)
        cur = best.get(key)
        if cur is None or interaction_priority(row) > interaction_priority(cur):
            best[key] = dict(row)
    return list(best.values())


def build_ratings_rows(interactions: Iterable[Dict]) -> List[Dict]:
    rows: List[Dict] = []
    for row in interactions:
        rating_value = row.get("rating")
        if rating_value in (None, ""):
            rating_value = row.get("implicit_score", "")
        rows.append(
            {
                "user_id": row.get("user_id", ""),
                "movie_id": row.get("movie_id", ""),
                "rating": rating_value,
                "liked": 1 if row.get("interaction_type") in {"liked", "favorite"} else "",
                "watched_date": row.get("watched_date", ""),
            }
        )
    return rows


def k_core_filter_interactions(
    interactions: List[Dict],
    min_user_interactions: int,
    min_movie_interactions: int,
) -> List[Dict]:
    current = list(interactions)
    while True:
        user_counts = Counter(row.get("user_id") for row in current)
        movie_counts = Counter(row.get("movie_id") for row in current)
        filtered = [
            row for row in current
            if user_counts[row.get("user_id")] >= min_user_interactions
            and movie_counts[row.get("movie_id")] >= min_movie_interactions
        ]
        if len(filtered) == len(current):
            return filtered
        current = filtered
        if not current:
            return []


class DatasetStore:
    def __init__(self, config: CrawlConfig) -> None:
        self.config = config
        self.users: List[Dict] = csv_read(USERS_CSV) if config.resume else []
        self.movies: List[Dict] = csv_read(MOVIES_CSV) if config.resume else []
        self.interactions: List[Dict] = csv_read(INTERACTIONS_CSV) if config.resume else []
        self.crawl_state: List[Dict] = csv_read(CRAWL_STATE_CSV) if config.resume else []

        # Nếu file cũ đã lỡ lưu nhầm các slug hệ thống như pro/apps/year-in-review,
        # loại chúng khi resume để không tiếp tục tính/crawl các dòng rác này.
        self.users = [
            r for r in self.users
            if is_valid_letterboxd_username(r.get("username", ""))
        ]
        self.crawl_state = [
            r for r in self.crawl_state
            if is_valid_letterboxd_username(r.get("username", ""))
        ]

        self.users = deduplicate_by_key(self.users, ["user_id"])
        self.movies = deduplicate_by_key(self.movies, ["movie_id"])
        valid_user_ids = {r.get("user_id", "") for r in self.users if r.get("user_id")}
        if valid_user_ids:
            self.interactions = [
                r for r in self.interactions
                if r.get("user_id", "") in valid_user_ids
            ]
        self.interactions = merge_interactions(self.interactions)

        self.user_by_id: Dict[str, Dict] = {r.get("user_id", ""): r for r in self.users if r.get("user_id")}
        self.username_to_id: Dict[str, str] = {
            normalize_username(r.get("username", "")): r.get("user_id", "")
            for r in self.users
            if r.get("username") and r.get("user_id")
        }
        self.movie_ids: Set[str] = {r.get("movie_id", "") for r in self.movies if r.get("movie_id")}
        self.interaction_keys: Set[Tuple[str, str]] = {
            (r.get("user_id", ""), r.get("movie_id", ""))
            for r in self.interactions
            if r.get("user_id") and r.get("movie_id")
        }
        # Các username đã xử lý rồi: done = crawl thành công, invalid = 404/không phải profile thật.
        # Bản cũ chỉ bỏ qua status="done", nên mỗi lần --resume lại thử lại các user 404 trong seed cache.
        self.crawled_usernames: Set[str] = {
            normalize_username(r.get("username", ""))
            for r in self.crawl_state
            if r.get("username") and r.get("status") in {"done", "invalid"}
        }

    def counts(self) -> Tuple[int, int, int]:
        return len(self.user_by_id), len(self.movie_ids), len(self.interaction_keys)

    def upsert_user(self, username: str, profile: Optional[Dict] = None) -> Optional[Dict]:
        username = normalize_username(username)
        if not is_valid_letterboxd_username(username):
            return None
        user_id = make_id(username, "user")
        new_row = build_user_row(username, profile=profile)

        if user_id in self.user_by_id:
            old = self.user_by_id[user_id]
            # Update profile fields nếu có dữ liệu mới.
            for key, value in new_row.items():
                if key == "created_at":
                    continue
                if value not in (None, ""):
                    old[key] = value
            return old

        if len(self.user_by_id) >= self.config.max_users:
            return None

        self.users.append(new_row)
        self.user_by_id[user_id] = new_row
        self.username_to_id[username] = user_id
        return new_row

    def add_movie(self, movie: Dict) -> bool:
        movie_id = movie.get("movie_id", "")
        if not movie_id:
            return False
        if movie_id in self.movie_ids:
            return True
        if len(self.movie_ids) >= self.config.max_movies:
            return False
        movie.setdefault("created_at", now_iso())
        self.movies.append(movie)
        self.movie_ids.add(movie_id)
        return True

    def add_interaction(self, interaction: Dict) -> bool:
        user_id = interaction.get("user_id", "")
        movie_id = interaction.get("movie_id", "")
        if not user_id or not movie_id:
            return False
        if movie_id not in self.movie_ids:
            return False
        if len(self.interaction_keys) >= self.config.max_interactions:
            return False

        key = (user_id, movie_id)
        if key in self.interaction_keys:
            self._maybe_upgrade_interaction(interaction)
            return False

        interaction.setdefault("created_at", now_iso())
        self.interactions.append(interaction)
        self.interaction_keys.add(key)
        return True

    def _maybe_upgrade_interaction(self, new_row: Dict) -> None:
        key = (new_row.get("user_id", ""), new_row.get("movie_id", ""))
        for idx, old_row in enumerate(self.interactions):
            if (old_row.get("user_id", ""), old_row.get("movie_id", "")) != key:
                continue
            if interaction_priority(new_row) > interaction_priority(old_row):
                updated = dict(new_row)
                updated["created_at"] = old_row.get("created_at", "") or now_iso()
                self.interactions[idx] = updated
            return

    def add_rows(self, movies: List[Dict], interactions: List[Dict]) -> Tuple[int, int]:
        added_movies = 0
        added_interactions = 0

        for movie in movies:
            before = len(self.movie_ids)
            self.add_movie(movie)
            if len(self.movie_ids) > before:
                added_movies += 1

        for interaction in interactions:
            if self.add_interaction(interaction):
                added_interactions += 1

        return added_movies, added_interactions

    def mark_crawled(self, username: str, status: str = "done") -> None:
        username = normalize_username(username)
        if not is_valid_letterboxd_username(username):
            return
        if status not in {"done", "invalid"}:
            status = "done"
        self.crawled_usernames.add(username)
        self.crawl_state.append({"username": username, "status": status, "crawled_at": now_iso()})
        self.crawl_state = deduplicate_by_key(reversed(self.crawl_state), ["username"])
        self.crawl_state = list(reversed(self.crawl_state))

    def cf_interactions(self) -> List[Dict]:
        return k_core_filter_interactions(
            self.interactions,
            min_user_interactions=self.config.cf_min_user_interactions,
            min_movie_interactions=self.config.cf_min_movie_interactions,
        )

    def cf_stats(self) -> Dict[str, object]:
        return quality_stats(self.cf_interactions())

    def should_stop(self) -> bool:
        users, movies, interactions = self.counts()

        # Hard cap để crawler không chạy vô hạn. Đây không phải điều kiện thành công.
        if users >= self.config.max_users:
            return True
        if interactions >= self.config.max_interactions:
            return True

        # Chưa đủ raw thì chắc chắn CF-ready chưa thể đạt mục tiêu.
        if (
            users < self.config.target_cf_users
            or movies < self.config.target_cf_movies
            or interactions < self.config.target_cf_interactions
        ):
            return False

        cf_stats = self.cf_stats()
        return (
            int(cf_stats["unique_users"]) >= self.config.target_cf_users
            and int(cf_stats["unique_movies"]) >= self.config.target_cf_movies
            and int(cf_stats["unique_interactions"]) >= self.config.target_cf_interactions
        )

    def save_all(self, reason: str = "") -> None:
        self.interactions = merge_interactions(self.interactions)
        self.interaction_keys = {
            (r.get("user_id", ""), r.get("movie_id", ""))
            for r in self.interactions
            if r.get("user_id") and r.get("movie_id")
        }

        interactions_cf = self.cf_interactions()
        ratings_rows = build_ratings_rows(self.interactions)
        ratings_cf_rows = build_ratings_rows(interactions_cf)
        cf_movie_ids = {r.get("movie_id") for r in interactions_cf if r.get("movie_id")}
        movies_cf_rows = [m for m in self.movies if m.get("movie_id") in cf_movie_ids]

        user_fields = [
            "user_id", "username", "display_name", "location", "profile_url", "films_count",
            "this_year_count", "lists_count", "following_count", "followers_count",
            "member_since", "bio", "created_at",
        ]
        movie_fields = ["movie_id", "title", "year", "movie_url", "created_at"]
        interaction_fields = [
            "user_id", "movie_id", "interaction_type", "rating", "implicit_score",
            "source", "watched_date", "created_at",
        ]
        ratings_fields = ["user_id", "movie_id", "rating", "liked", "watched_date"]

        csv_write_atomic(USERS_CSV, self.users, user_fields)
        csv_write_atomic(MOVIES_CSV, self.movies, movie_fields)
        csv_write_atomic(INTERACTIONS_CSV, self.interactions, interaction_fields)
        csv_write_atomic(RATINGS_CSV, ratings_rows, ratings_fields)
        csv_write_atomic(INTERACTIONS_CF_CSV, interactions_cf, interaction_fields)
        csv_write_atomic(RATINGS_CF_CSV, ratings_cf_rows, ratings_fields)
        csv_write_atomic(MOVIES_CF_CSV, movies_cf_rows, movie_fields)
        csv_write_atomic(CRAWL_STATE_CSV, self.crawl_state, ["username", "status", "crawled_at"])

        write_report(self, interactions_cf)
        if reason:
            users, movies, interactions = self.counts()
            cf_stats = quality_stats(interactions_cf)
            print(
                f"\n[CHECKPOINT] {reason}: "
                f"raw users={users}, movies={movies}, interactions={interactions} | "
                f"CF users={cf_stats['unique_users']}, movies={cf_stats['unique_movies']}, "
                f"interactions={cf_stats['unique_interactions']}",
                flush=True,
            )


# ============================================================
# Report
# ============================================================

def quality_stats(interactions: List[Dict]) -> Dict[str, object]:
    user_counts = Counter(row.get("user_id") for row in interactions)
    movie_counts = Counter(row.get("movie_id") for row in interactions)
    type_counts = Counter(row.get("interaction_type") for row in interactions)
    rating_count = sum(1 for row in interactions if row.get("rating") not in (None, ""))

    def avg(counter: Counter) -> float:
        return sum(counter.values()) / len(counter) if counter else 0.0

    return {
        "unique_users": len(user_counts),
        "unique_movies": len(movie_counts),
        "unique_interactions": len(interactions),
        "avg_interactions_per_user": round(avg(user_counts), 2),
        "avg_interactions_per_movie": round(avg(movie_counts), 2),
        "non_null_ratings": rating_count,
        "interaction_types": dict(type_counts),
        "users_lt_10_interactions": sum(1 for count in user_counts.values() if count < 10),
        "movies_lt_5_interactions": sum(1 for count in movie_counts.values() if count < 5),
    }


def write_report(store: DatasetStore, interactions_cf: List[Dict]) -> None:
    raw_stats = quality_stats(store.interactions)
    cf_stats = quality_stats(interactions_cf)

    lines = [
        "Crawl report",
        f"Generated at: {now_iso()}",
        "",
        "Raw dataset:",
        f"- Users: {len(store.user_by_id)}",
        f"- Movies: {len(store.movie_ids)}",
        f"- Interactions: {len(store.interaction_keys)}",
        f"- Non-null ratings: {raw_stats['non_null_ratings']}",
        f"- Avg interactions/user: {raw_stats['avg_interactions_per_user']}",
        f"- Avg interactions/movie: {raw_stats['avg_interactions_per_movie']}",
        f"- Interaction types: {raw_stats['interaction_types']}",
        f"- Users with <10 interactions: {raw_stats['users_lt_10_interactions']}",
        f"- Movies with <5 interactions: {raw_stats['movies_lt_5_interactions']}",
        "",
        "CF-ready dataset after k-core filtering:",
        f"- Target users/movies/interactions: {store.config.target_cf_users}/{store.config.target_cf_movies}/{store.config.target_cf_interactions}",
        f"- Users: {cf_stats['unique_users']}",
        f"- Movies: {cf_stats['unique_movies']}",
        f"- Interactions: {cf_stats['unique_interactions']}",
        f"- Non-null ratings: {cf_stats['non_null_ratings']}",
        f"- Avg interactions/user: {cf_stats['avg_interactions_per_user']}",
        f"- Avg interactions/movie: {cf_stats['avg_interactions_per_movie']}",
        "",
        "Files:",
        f"- {USERS_CSV}",
        f"- {MOVIES_CSV}",
        f"- {INTERACTIONS_CSV}",
        f"- {RATINGS_CSV}",
        f"- {INTERACTIONS_CF_CSV}",
        f"- {RATINGS_CF_CSV}",
        f"- {MOVIES_CF_CSV}",
    ]
    CRAWL_REPORT_TXT.write_text("\n".join(lines), encoding="utf-8")


def print_final_report(store: DatasetStore) -> None:
    interactions_cf = csv_read(INTERACTIONS_CF_CSV)
    raw_stats = quality_stats(store.interactions)
    cf_stats = quality_stats(interactions_cf)

    print("\nDone.")
    print("Raw dataset:")
    print(f"  Users        : {len(store.user_by_id)}")
    print(f"  Movies       : {len(store.movie_ids)}")
    print(f"  Interactions : {len(store.interaction_keys)}")
    print(f"  Ratings      : {raw_stats['non_null_ratings']}")
    print(f"  Avg I/User   : {raw_stats['avg_interactions_per_user']}")
    print(f"  Avg I/Movie  : {raw_stats['avg_interactions_per_movie']}")
    print("CF-ready dataset:")
    print(f"  Users        : {cf_stats['unique_users']}")
    print(f"  Movies       : {cf_stats['unique_movies']}")
    print(f"  Interactions : {cf_stats['unique_interactions']}")
    print(f"Report         : {CRAWL_REPORT_TXT}")

    if cf_stats["unique_interactions"] == 0:
        print("\n[WARN] interactions_cf.csv rỗng sau k-core filtering.")
        print("Gợi ý: chạy tiếp --resume hoặc giảm --cf-min-movie-interactions 3.")
    elif int(cf_stats["unique_interactions"]) < TARGET_CF_INTERACTIONS:
        print(f"\n[NOTE] CF-ready interactions vẫn thấp hơn mục tiêu {TARGET_CF_INTERACTIONS:,}.")
        print("Gợi ý: chạy tiếp --resume, tăng --max-users/--max-interactions, hoặc tăng --pages-likes/--pages-diary.")


# ============================================================
# Main
# ============================================================

def main_crawl(config: CrawlConfig) -> DatasetStore:
    store = DatasetStore(config)
    existing_users, existing_movies, existing_interactions = store.counts()

    if config.resume and config.backup_before_run:
        backup_dir = backup_existing_raw_files()
        if backup_dir:
            print(f"[BACKUP] Existing raw files copied to: {backup_dir}", flush=True)

    print("Start Letterboxd hybrid crawling: movie-overlap seed + user-centric crawl...")
    print(f"Data dir: {RAW_DIR}")
    print(f"Resume: {config.resume}")
    print(f"Existing: users={existing_users}, movies={existing_movies}, interactions={existing_interactions}")
    print(
        "Raw hard caps: "
        f"users={config.min_users}-{config.max_users}, "
        f"movies={config.min_movies}-{config.max_movies}, "
        f"interactions={config.min_interactions}-{config.max_interactions}"
    )
    print(
        "CF-ready target: "
        f"users={config.target_cf_users}, "
        f"movies={config.target_cf_movies}, "
        f"interactions={config.target_cf_interactions}"
    )
    print(
        "Pages/user: "
        f"films={config.pages_per_films}, likes={config.pages_per_likes}, "
        f"reviews={config.pages_per_reviews}, diary={config.pages_per_diary}, "
        f"following={config.pages_per_following}"
    )
    print(
        "CF filter: "
        f"min_user_interactions={config.cf_min_user_interactions}, "
        f"min_movie_interactions={config.cf_min_movie_interactions}"
    )
    print(f"Seed members pages: {config.seed_members_pages}, max seed candidates: {config.max_seed_user_candidates}")
    print(f"Checkpoint every {config.checkpoint_every_users} crawled users.\n")

    initial_usernames: List[str] = []

    # 1) Ưu tiên seed users lấy từ /film/{slug}/members/ để tăng movie-overlap.
    for username in build_movie_overlap_seed_users(store, config):
        username = normalize_username(username)
        if is_valid_letterboxd_username(username) and username not in initial_usernames:
            initial_usernames.append(username)

    # 2) Fallback static seed users nếu members phase bị chặn/rỗng.
    for u in SEED_USERNAMES:
        nu = normalize_username(u)
        if is_valid_letterboxd_username(nu) and nu not in initial_usernames:
            initial_usernames.append(nu)

    # 3) Nếu resume, đưa user cũ vào queue để following expansion vẫn tiếp tục nếu cần.
    for username in store.username_to_id.keys():
        if is_valid_letterboxd_username(username) and username not in initial_usernames:
            initial_usernames.append(username)

    print(f"Initial queue: {len(initial_usernames)} users", flush=True)

    queue: deque[str] = deque(initial_usernames)
    discovered: Set[str] = set(initial_usernames)
    crawled_this_run: Set[str] = set()

    with tqdm(total=config.max_users, initial=min(existing_users, config.max_users), desc="Unique users") as pbar:
        while queue:
            if store.should_stop():
                break

            username = normalize_username(queue.popleft())
            if not is_valid_letterboxd_username(username) or username in crawled_this_run:
                continue

            # Nếu resume và user đã crawl xong trước đó, bỏ qua tương tác nhưng vẫn không cần crawl lại.
            if config.resume and username in store.crawled_usernames:
                continue

            before_users = len(store.user_by_id)

            profile_info, interactions, movies = crawl_user_interactions(username, config=config)
            if profile_info.get("_invalid_user"):
                # Lưu lại user 404/không hợp lệ để các lần --resume sau không thử lại nữa.
                store.mark_crawled(username, status="invalid")
                crawled_this_run.add(username)
                print(f"[SKIP] Not a valid Letterboxd user profile: {username}", flush=True)

                if len(crawled_this_run) % max(1, config.checkpoint_every_users) == 0:
                    store.save_all(reason=f"saved after {len(crawled_this_run)} users this run")
                continue

            user = store.upsert_user(username, profile=profile_info)
            if user is None:
                continue
            if len(store.user_by_id) > before_users:
                pbar.update(1)

            added_movies, added_interactions = store.add_rows(movies, interactions)

            following_users = get_following_users(username, config=config)
            for following in following_users:
                if (
                    is_valid_letterboxd_username(following)
                    and following not in discovered
                    and len(discovered) < config.max_users * 4
                ):
                    discovered.add(following)
                    queue.append(following)

            store.mark_crawled(username)
            crawled_this_run.add(username)

            users, movies_count, interactions_count = store.counts()
            pbar.set_postfix(
                users=users,
                movies=movies_count,
                interactions=interactions_count,
                added_i=added_interactions,
                queue=len(queue),
            )

            if len(crawled_this_run) % max(1, config.checkpoint_every_users) == 0:
                store.save_all(reason=f"saved after {len(crawled_this_run)} users this run")

    store.save_all(reason="final save")
    print_final_report(store)
    return store


# ============================================================
# CLI
# ============================================================

def parse_args(argv: Optional[List[str]] = None) -> CrawlConfig:
    parser = argparse.ArgumentParser(description="Standard Letterboxd crawler for movie recommender dataset")

    parser.add_argument("--resume", action="store_true", help="Đọc CSV cũ và crawl tiếp. Mặc định đã bật nếu không dùng --no-resume")
    parser.add_argument("--no-resume", action="store_true", help="Không đọc CSV cũ, crawl lại từ đầu")

    parser.add_argument("--min-users", type=int, default=MIN_USERS)
    parser.add_argument("--target-users", type=int, default=TARGET_USERS)
    parser.add_argument("--max-users", type=int, default=MAX_USERS)

    parser.add_argument("--min-movies", type=int, default=MIN_MOVIES)
    parser.add_argument("--target-movies", type=int, default=TARGET_MOVIES)
    parser.add_argument("--max-movies", type=int, default=MAX_MOVIES)

    parser.add_argument("--min-interactions", type=int, default=MIN_INTERACTIONS)
    parser.add_argument("--target-interactions", type=int, default=TARGET_INTERACTIONS)
    parser.add_argument("--max-interactions", type=int, default=MAX_INTERACTIONS)

    parser.add_argument("--target-cf-users", type=int, default=TARGET_CF_USERS)
    parser.add_argument("--target-cf-movies", type=int, default=TARGET_CF_MOVIES)
    parser.add_argument("--target-cf-interactions", type=int, default=TARGET_CF_INTERACTIONS)

    parser.add_argument("--pages-films", type=int, default=PAGES_PER_FILMS)
    parser.add_argument("--pages-likes", type=int, default=PAGES_PER_LIKES)
    parser.add_argument("--pages-reviews", type=int, default=PAGES_PER_REVIEWS)
    parser.add_argument("--pages-diary", type=int, default=PAGES_PER_DIARY)
    parser.add_argument("--pages-following", type=int, default=PAGES_PER_FOLLOWING)

    parser.add_argument("--cf-min-user-interactions", type=int, default=CF_MIN_USER_INTERACTIONS)
    parser.add_argument("--cf-min-movie-interactions", type=int, default=CF_MIN_MOVIE_INTERACTIONS)

    parser.add_argument("--sleep", type=float, default=SLEEP_SECONDS)
    parser.add_argument("--jitter", type=float, default=JITTER_SECONDS)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--checkpoint-every-users", type=int, default=CHECKPOINT_EVERY_USERS)
    parser.add_argument("--seed-members-pages", type=int, default=SEED_MEMBERS_PAGES,
                        help="Số trang /film/{slug}/members/ crawl cho mỗi seed film")
    parser.add_argument("--max-seed-user-candidates", type=int, default=MAX_SEED_USER_CANDIDATES,
                        help="Giới hạn số seed users lấy từ movie-overlap phase")
    parser.add_argument("--skip-movie-overlap-seed", action="store_true",
                        help="Bỏ phase /film/{slug}/members/, chỉ dùng SEED_USERNAMES fallback")
    parser.add_argument("--rebuild-seed-users", action="store_true",
                        help="Bỏ cache seed_user_candidates.csv và crawl lại seed users từ members pages")
    parser.add_argument("--no-backup", action="store_true",
                        help="Không tạo bản backup CSV cũ trước khi resume")
    parser.add_argument("--debug-urls", action="store_true", help="In URL đang request để dễ biết crawler có đang chạy không")

    args = parser.parse_args(argv)

    return CrawlConfig(
        min_users=args.min_users,
        target_users=args.target_users,
        max_users=args.max_users,
        min_movies=args.min_movies,
        target_movies=args.target_movies,
        max_movies=args.max_movies,
        min_interactions=args.min_interactions,
        target_interactions=args.target_interactions,
        max_interactions=args.max_interactions,
        target_cf_users=args.target_cf_users,
        target_cf_movies=args.target_cf_movies,
        target_cf_interactions=args.target_cf_interactions,
        pages_per_films=args.pages_films,
        pages_per_likes=args.pages_likes,
        pages_per_reviews=args.pages_reviews,
        pages_per_diary=args.pages_diary,
        pages_per_following=args.pages_following,
        cf_min_user_interactions=args.cf_min_user_interactions,
        cf_min_movie_interactions=args.cf_min_movie_interactions,
        sleep_seconds=args.sleep,
        jitter_seconds=args.jitter,
        max_retries=args.max_retries,
        checkpoint_every_users=args.checkpoint_every_users,
        seed_members_pages=args.seed_members_pages,
        max_seed_user_candidates=args.max_seed_user_candidates,
        skip_movie_overlap_seed=args.skip_movie_overlap_seed,
        rebuild_seed_users=args.rebuild_seed_users,
        resume=not args.no_resume,
        backup_before_run=not args.no_backup,
        debug_urls=args.debug_urls,
    )


if __name__ == "__main__":
    try:
        cfg = parse_args()
        main_crawl(cfg)
    except KeyboardInterrupt:
        print("\n[STOP] Người dùng dừng chương trình bằng Ctrl+C.")
        sys.exit(130)
