"""
enrich_tmdb.py — Enrich movies_seed.csv với metadata từ TMDB API

Chạy SAU khi crawler (Strategy A hoặc gốc) đã xong.

Luồng:
  1. Đọc movies_seed.csv (output của crawler).
  2. Với mỗi phim: gọi TMDB /search/movie?query=title&year=year.
  3. Lấy tmdb_id, rồi gọi /movie/{tmdb_id} để lấy full metadata.
  4. Ghi ra movies_enriched.csv (giữ nguyên movie_id cũ để interactions không đổi).
  5. Ghi enrich_report.txt tóm tắt kết quả.

Cài đặt:
    pip install requests tqdm

Lấy API key miễn phí tại: https://www.themoviedb.org/settings/api
    (dùng "API Read Access Token" — dài hơn, bắt đầu bằng eyJ...)
    hoặc "API Key" — ngắn hơn, dạng hex 32 ký tự.

Cách chạy:
    # Dùng API key (v3):
    python enrich_tmdb.py --api-key YOUR_API_KEY

    # Dùng Bearer token (v4 / Read Access Token):
    python enrich_tmdb.py --bearer-token YOUR_BEARER_TOKEN

    # Chỉ định thư mục data khác (mặc định: data/raw/):
    python enrich_tmdb.py --api-key YOUR_KEY --data-dir data/raw

    # Resume nếu bị gián đoạn (bỏ qua phim đã enrich):
    python enrich_tmdb.py --api-key YOUR_KEY --resume

    # Giới hạn số phim để test:
    python enrich_tmdb.py --api-key YOUR_KEY --limit 100

    # Tăng delay nếu bị rate limit (TMDB free: 50 req/s):
    python enrich_tmdb.py --api-key YOUR_KEY --sleep 0.3
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


# ============================================================
# Config
# ============================================================

TMDB_BASE = "https://api.themoviedb.org/3"

# Các field metadata muốn lấy từ TMDB
# Có thể thêm: "production_companies", "spoken_languages", "keywords"...
TMDB_APPEND = "credits,keywords,release_dates"

ROOT_DIR = Path(__file__).resolve().parent

# Field names cho output CSV
ENRICHED_FIELDS = [
    # --- Giữ nguyên từ crawler ---
    "movie_id",          # internal ID của crawler (md5), KHÔNG thay đổi
    "title",             # title từ Letterboxd
    "year",              # year từ Letterboxd
    "movie_url",         # URL Letterboxd
    # --- Thêm từ TMDB ---
    "tmdb_id",           # ID trên TMDB
    "tmdb_title",        # title chính thức từ TMDB (có thể khác Letterboxd)
    "original_title",    # tên gốc (tiếng nước ngoài)
    "original_language", # ngôn ngữ gốc (en, fr, ja, ko...)
    "release_date",      # YYYY-MM-DD
    "runtime",           # phút
    "status",            # Released, Post Production...
    "genres",            # pipe-separated: Drama|Thriller|Romance
    "overview",          # mô tả phim (tiếng Anh)
    "tagline",
    "vote_average",      # điểm TMDB (0-10)
    "vote_count",        # số lượt vote TMDB
    "popularity",        # điểm popularity TMDB
    "budget",            # USD
    "revenue",           # USD
    "tmdb_poster_path",  # /abc123.jpg — ghép với https://image.tmdb.org/t/p/w500/
    "tmdb_backdrop_path",
    "directors",         # pipe-separated names
    "top_cast",          # pipe-separated, tối đa 10 diễn viên chính
    "keywords",          # pipe-separated keyword names
    "production_companies", # pipe-separated production company names
    "production_countries", # pipe-separated ISO codes: US|FR|JP
    "collection_id",     # nếu thuộc franchise (Marvel, HP...)
    "collection_name",
    "certification",     # US movie certification from release_dates
    "adult",             # true/false
    "imdb_id",           # tt1234567
    # --- Meta của enrich ---
    "enrich_status",     # matched / not_found / error
    "enrich_match_score",# 0-100, độ tin cậy của match
    "created_at",        # timestamp crawler
    "enriched_at",       # timestamp enrich
]


@dataclass
class EnrichConfig:
    api_key: str = ""
    bearer_token: str = ""
    data_dir: Path = ROOT_DIR / "data" / "raw"
    sleep_seconds: float = 0.25      # TMDB free tier: ~50 req/s → 0.25s an toàn
    max_retries: int = 4
    resume: bool = True              # bỏ qua phim đã enrich
    limit: int = 0                   # 0 = không giới hạn
    min_match_score: int = 60        # bỏ qua match nếu score thấp hơn ngưỡng này
    language: str = "en-US"          # ngôn ngữ response TMDB
    debug: bool = False


# ============================================================
# HTTP session
# ============================================================

def build_session(config: EnrichConfig) -> requests.Session:
    session = requests.Session()
    if config.bearer_token:
        session.headers["Authorization"] = f"Bearer {config.bearer_token}"
        session.headers["Accept"] = "application/json"
    retry = Retry(
        total=config.max_retries,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


SESSION: Optional[requests.Session] = None


def tmdb_get(endpoint: str, config: EnrichConfig, params: Optional[Dict] = None) -> Optional[Dict]:
    """Gọi TMDB API, tự động xử lý rate limit và retry."""
    global SESSION
    if SESSION is None:
        SESSION = build_session(config)

    url = f"{TMDB_BASE}{endpoint}"
    p = {"language": config.language}
    if params:
        p.update(params)
    if config.api_key and not config.bearer_token:
        p["api_key"] = config.api_key

    for attempt in range(1, config.max_retries + 2):
        try:
            if config.debug:
                print(f"  [GET] {url}  params={p}", flush=True)
            resp = SESSION.get(url, params=p, timeout=20)

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 10))
                print(f"\n[WARN] Rate limit — chờ {retry_after}s...", flush=True)
                time.sleep(retry_after + 1)
                continue

            if resp.status_code == 404:
                return None

            if resp.status_code in {401, 403}:
                print(f"\n[ERROR] Auth failed ({resp.status_code}). Kiểm tra lại API key/token.", flush=True)
                sys.exit(1)

            wait = 2 ** attempt
            time.sleep(wait)

        except requests.RequestException as exc:
            wait = 2 ** attempt
            print(f"\n[ERROR] Request failed attempt {attempt}: {exc}", flush=True)
            time.sleep(wait)

    return None


# ============================================================
# Matching logic
# ============================================================

def normalize_title(title: str) -> str:
    """Chuẩn hóa title để so sánh: lowercase, bỏ dấu câu, bỏ khoảng trắng thừa."""
    t = title.lower().strip()
    # Bỏ article ở đầu (The, A, An) vì TMDB và Letterboxd xử lý khác nhau
    t = re.sub(r"^(the|a|an)\s+", "", t)
    # Bỏ ký tự đặc biệt
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def title_similarity(a: str, b: str) -> int:
    """Trả về score 0-100 dựa trên độ giống nhau của 2 title sau normalize."""
    na, nb = normalize_title(a), normalize_title(b)
    if na == nb:
        return 100
    # Partial match: một cái là substring của cái kia
    if na in nb or nb in na:
        return 85
    # Word overlap
    words_a = set(na.split())
    words_b = set(nb.split())
    if not words_a or not words_b:
        return 0
    overlap = len(words_a & words_b)
    union = len(words_a | words_b)
    return int(overlap / union * 80)


def year_match_score(year_lb: str, release_date: str) -> int:
    """So khớp năm: exact +20, lệch 1 năm +10, lệch nhiều hơn 0."""
    if not year_lb or not release_date:
        return 5  # không rõ năm → không phạt mạnh
    try:
        y_lb = int(year_lb)
        y_tmdb = int(release_date[:4])
        diff = abs(y_lb - y_tmdb)
        if diff == 0:
            return 20
        if diff == 1:
            return 10  # Letterboxd đôi khi dùng năm ra mắt festival
        return 0
    except (ValueError, IndexError):
        return 5


def pick_best_result(
    results: List[Dict],
    lb_title: str,
    lb_year: str,
    min_score: int,
) -> Tuple[Optional[Dict], int]:
    """
    Chọn kết quả TMDB tốt nhất từ search results.
    Trả về (best_result, match_score) hoặc (None, 0) nếu không đạt ngưỡng.
    """
    best: Optional[Dict] = None
    best_score = 0

    for r in results[:5]:  # chỉ xét top 5
        title_score = title_similarity(lb_title, r.get("title", ""))
        orig_score = title_similarity(lb_title, r.get("original_title", ""))
        t_score = max(title_score, orig_score)
        y_score = year_match_score(lb_year, r.get("release_date", ""))
        total = t_score + y_score

        if total > best_score:
            best_score = total
            best = r

    if best_score < min_score:
        return None, best_score
    return best, best_score


# ============================================================
# TMDB fetch + parse
# ============================================================

def search_movie(title: str, year: str, config: EnrichConfig) -> Tuple[Optional[Dict], int]:
    """Tìm phim trên TMDB. Trả về (search_result, match_score)."""
    params: Dict = {"query": title, "include_adult": "false"}
    if year:
        params["year"] = year

    data = tmdb_get("/search/movie", config, params)
    if not data or not data.get("results"):
        # Thử lại không có year (đôi khi year lệch làm TMDB không tìm được)
        if year:
            data = tmdb_get("/search/movie", config, {"query": title, "include_adult": "false"})

    if not data or not data.get("results"):
        return None, 0

    return pick_best_result(data["results"], title, year, config.min_match_score)


def fetch_movie_details(tmdb_id: int, config: EnrichConfig) -> Optional[Dict]:
    """Lấy full detail + credits + keywords của 1 phim."""
    return tmdb_get(
        f"/movie/{tmdb_id}",
        config,
        {"append_to_response": TMDB_APPEND},
    )


def pipe(items: List[str]) -> str:
    """Join list thành chuỗi pipe-separated."""
    return "|".join(i for i in items if i)


def certification_from_release_dates(raw: Dict, country: str = "US") -> str:
    results = raw.get("release_dates", {}).get("results", [])
    country_release = next((item for item in results if item.get("iso_3166_1") == country), None)
    if not country_release:
        return ""
    for release in country_release.get("release_dates", []):
        certification = str(release.get("certification", "")).strip()
        if certification:
            return certification
    return ""


def parse_details(raw: Dict) -> Dict:
    """Extract các field cần thiết từ TMDB movie detail response."""
    # Genres
    genres = pipe([g.get("name", "") for g in raw.get("genres", [])])

    # Directors từ credits.crew
    crew = raw.get("credits", {}).get("crew", [])
    directors = pipe([p["name"] for p in crew if p.get("job") == "Director" and p.get("name")])

    # Top cast (tối đa 10)
    cast = raw.get("credits", {}).get("cast", [])
    top_cast = pipe([p["name"] for p in cast[:10] if p.get("name")])

    # Keywords
    kw_list = raw.get("keywords", {}).get("keywords", [])
    keywords = pipe([k.get("name", "") for k in kw_list[:30]])  # tối đa 30

    # Production countries
    companies = pipe([c.get("name", "") for c in raw.get("production_companies", [])])
    countries = pipe([c.get("iso_3166_1", "") for c in raw.get("production_countries", [])])

    # Collection
    coll = raw.get("belongs_to_collection") or {}
    collection_id = str(coll.get("id", "")) if coll else ""
    collection_name = coll.get("name", "") if coll else ""

    return {
        "tmdb_id": str(raw.get("id", "")),
        "tmdb_title": raw.get("title", ""),
        "original_title": raw.get("original_title", ""),
        "original_language": raw.get("original_language", ""),
        "release_date": raw.get("release_date", ""),
        "runtime": str(raw.get("runtime") or ""),
        "status": raw.get("status", ""),
        "genres": genres,
        "overview": (raw.get("overview") or "")[:500],
        "tagline": raw.get("tagline", ""),
        "vote_average": str(raw.get("vote_average") or ""),
        "vote_count": str(raw.get("vote_count") or ""),
        "popularity": str(raw.get("popularity") or ""),
        "budget": str(raw.get("budget") or ""),
        "revenue": str(raw.get("revenue") or ""),
        "tmdb_poster_path": raw.get("poster_path") or "",
        "tmdb_backdrop_path": raw.get("backdrop_path") or "",
        "directors": directors,
        "top_cast": top_cast,
        "keywords": keywords,
        "production_companies": companies,
        "production_countries": countries,
        "collection_id": collection_id,
        "collection_name": collection_name,
        "certification": certification_from_release_dates(raw),
        "adult": str(raw.get("adult", False)).lower(),
        "imdb_id": raw.get("imdb_id") or "",
    }


# ============================================================
# CSV helpers
# ============================================================

def csv_read(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def csv_write_atomic(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ============================================================
# Main enrich loop
# ============================================================

def enrich(config: EnrichConfig) -> None:
    movies_path    = config.data_dir / "movies_seed.csv"
    enriched_path  = config.data_dir / "movies_enriched.csv"
    report_path    = config.data_dir / "enrich_report.txt"

    if not movies_path.exists():
        print(f"[ERROR] Không tìm thấy {movies_path}", flush=True)
        print("  Hãy chạy crawler trước, hoặc chỉ định --data-dir đúng.", flush=True)
        sys.exit(1)

    movies = csv_read(movies_path)
    if not movies:
        print("[ERROR] movies_seed.csv rỗng.", flush=True)
        sys.exit(1)

    # Load enriched cũ nếu resume
    already_enriched: Dict[str, Dict] = {}
    if config.resume and enriched_path.exists():
        for row in csv_read(enriched_path):
            mid = row.get("movie_id", "")
            if mid and row.get("enrich_status") in {"matched", "not_found"}:
                already_enriched[mid] = row
        print(f"[RESUME] {len(already_enriched)} phim đã enrich trước đó — bỏ qua.", flush=True)

    # Áp limit nếu có
    to_process = movies
    if config.limit > 0:
        to_process = movies[: config.limit]

    print(f"\nEnriching {len(to_process)} movies từ {movies_path.name}...")
    print(f"Output   : {enriched_path}")
    print(f"Resume   : {config.resume}  |  Min match score: {config.min_match_score}\n")

    results: List[Dict] = []
    stats = {"matched": 0, "not_found": 0, "error": 0, "skipped": 0}

    with tqdm(total=len(to_process), desc="Enriching") as pbar:
        for movie in to_process:
            mid = movie.get("movie_id", "")
            title = movie.get("title", "").strip()
            year = movie.get("year", "").strip()

            # Resume: đã xử lý rồi thì bỏ qua
            if mid in already_enriched:
                results.append(already_enriched[mid])
                stats["skipped"] += 1
                pbar.update(1)
                continue

            # Base row giữ nguyên data từ crawler
            row: Dict = {f: "" for f in ENRICHED_FIELDS}
            row.update({
                "movie_id":  mid,
                "title":     title,
                "year":      year,
                "movie_url": movie.get("movie_url", ""),
                "created_at": movie.get("created_at", ""),
                "enriched_at": now_iso(),
            })

            if not title:
                row["enrich_status"] = "error"
                row["enrich_match_score"] = "0"
                results.append(row)
                stats["error"] += 1
                pbar.update(1)
                continue

            # ── Bước 1: Search ───────────────────────────────
            search_result, match_score = search_movie(title, year, config)
            time.sleep(config.sleep_seconds)

            if search_result is None:
                row["enrich_status"] = "not_found"
                row["enrich_match_score"] = str(match_score)
                results.append(row)
                stats["not_found"] += 1
                pbar.set_postfix(**stats)
                pbar.update(1)
                continue

            # ── Bước 2: Fetch full details ───────────────────
            tmdb_id = search_result.get("id")
            details = fetch_movie_details(tmdb_id, config)
            time.sleep(config.sleep_seconds)

            if details is None:
                row["enrich_status"] = "error"
                row["enrich_match_score"] = str(match_score)
                results.append(row)
                stats["error"] += 1
                pbar.set_postfix(**stats)
                pbar.update(1)
                continue

            # ── Bước 3: Parse và merge ───────────────────────
            parsed = parse_details(details)
            row.update(parsed)
            row["enrich_status"] = "matched"
            row["enrich_match_score"] = str(match_score)
            results.append(row)
            stats["matched"] += 1

            pbar.set_postfix(**stats)
            pbar.update(1)

            # Checkpoint mỗi 200 phim
            if len(results) % 200 == 0:
                csv_write_atomic(enriched_path, results, ENRICHED_FIELDS)

    # Final save
    # Thêm các phim không nằm trong to_process (nếu limit) — giữ data cũ
    processed_ids = {r["movie_id"] for r in results}
    for mid, row in already_enriched.items():
        if mid not in processed_ids:
            results.append(row)

    csv_write_atomic(enriched_path, results, ENRICHED_FIELDS)

    # ── Report ───────────────────────────────────────────────
    total = len(to_process)
    match_rate = stats["matched"] / (total - stats["skipped"]) * 100 if (total - stats["skipped"]) > 0 else 0
    lines = [
        "TMDB Enrich Report",
        f"Generated   : {now_iso()}",
        f"Source      : {movies_path}",
        f"Output      : {enriched_path}",
        "",
        f"Total movies in seed  : {len(movies)}",
        f"Processed this run    : {total - stats['skipped']}",
        f"Skipped (resume)      : {stats['skipped']}",
        "",
        f"Matched               : {stats['matched']}",
        f"Not found             : {stats['not_found']}",
        f"Error                 : {stats['error']}",
        f"Match rate            : {match_rate:.1f}%",
        "",
        "Tips nếu match rate thấp:",
        "  - Giảm --min-match-score 50 (chấp nhận match ít chính xác hơn)",
        "  - Kiểm tra movies_enriched.csv cột enrich_match_score",
        "  - Một số phim obscure/short film TMDB không có — bình thường",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 55)
    print("TMDB Enrich — Done")
    print("=" * 55)
    print(f"  Matched     : {stats['matched']:>6}")
    print(f"  Not found   : {stats['not_found']:>6}")
    print(f"  Error       : {stats['error']:>6}")
    print(f"  Skipped     : {stats['skipped']:>6}")
    print(f"  Match rate  : {match_rate:.1f}%")
    print(f"\n  Output      : {enriched_path}")
    print(f"  Report      : {report_path}")

    if match_rate < 70:
        print("\n[NOTE] Match rate < 70% — thử --min-match-score 50 để recover thêm phim.")


# ============================================================
# CLI
# ============================================================

def parse_args(argv: Optional[List[str]] = None) -> EnrichConfig:
    p = argparse.ArgumentParser(
        description="Enrich movies_seed.csv với metadata TMDB"
    )

    auth = p.add_mutually_exclusive_group(required=True)
    auth.add_argument(
        "--api-key",
        help="TMDB API key v3 (hex 32 ký tự). Lấy tại themoviedb.org/settings/api",
    )
    auth.add_argument(
        "--bearer-token",
        help="TMDB Read Access Token v4 (bắt đầu bằng eyJ...)",
    )

    p.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT_DIR / "data" / "raw",
        help="Thư mục chứa movies_seed.csv (mặc định: data/raw/)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Bỏ qua phim đã enrich trong lần chạy trước (mặc định: bật)",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Enrich lại toàn bộ từ đầu",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Giới hạn số phim xử lý (0 = không giới hạn, dùng để test)",
    )
    p.add_argument(
        "--min-match-score",
        type=int,
        default=60,
        help="Ngưỡng score match title (0-100). Thấp hơn = chấp nhận match mơ hồ hơn. Mặc định: 60",
    )
    p.add_argument(
        "--sleep",
        type=float,
        default=0.25,
        help="Delay giữa các request (giây). TMDB free: ~50 req/s → 0.25s. Mặc định: 0.25",
    )
    p.add_argument(
        "--language",
        default="en-US",
        help="Ngôn ngữ response TMDB. Mặc định: en-US",
    )
    p.add_argument("--debug", action="store_true", help="In URL đang gọi")

    args = p.parse_args(argv)

    return EnrichConfig(
        api_key=args.api_key or "",
        bearer_token=args.bearer_token or "",
        data_dir=args.data_dir,
        sleep_seconds=args.sleep,
        resume=not args.no_resume,
        limit=args.limit,
        min_match_score=args.min_match_score,
        language=args.language,
        debug=args.debug,
    )


if __name__ == "__main__":
    try:
        cfg = parse_args()
        enrich(cfg)
    except KeyboardInterrupt:
        print("\n[STOP] Ctrl+C — data đã checkpoint tự động.", flush=True)
        sys.exit(130)
