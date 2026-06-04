# Letterboxd Crawl Data

This folder contains the usable parts merged from the remote `crawl` branch:

- `crawl_letterboxd_movie_centric.py`: resumable Letterboxd crawler.
- `enrich_tmdb.py`: TMDb metadata enrichment for crawler output.
- `data/raw/`: crawled CSV outputs from that branch.

The crawler writes Letterboxd-style data under `crawl/data/raw/`. This dataset is not a drop-in MovieLens replacement because it uses crawler-specific IDs and schemas; convert it before using the existing MovieLens training scripts.

Typical commands:

```bash
python crawl/crawl_letterboxd_movie_centric.py --resume
python crawl/enrich_tmdb.py --api-key "$TMDB_API_KEY" --data-dir crawl/data/raw
```
