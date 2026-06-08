# Data Quality Report

- Data dir: `data/ml-latest-small`
- Generated at: `2026-06-02T03:38:44.585376+00:00`

## Core Counts

- Users: 610
- Movies: 9742
- Ratings: 100836
- Rated movies: 9724
- Sparsity: 98.3032%

## Main Issues

- rating_movieIds_not_in_movies: 0
- tag_movieIds_not_in_movies: 0
- link_movieIds_not_in_movies: 0
- enriched_movieIds_not_in_movies: 0
- movies_without_ratings: 18
- movies_without_links: 0
- movies_without_enriched: 121
- any_core_metadata_missing: 156
- all_core_metadata_missing: 121
- no_genres_listed: 34
- missing_year: 13
- tagged_movies: 1572
- tagged_movies_pct: 16.1363
- no_budget_no_revenue: 2285

## Recommendations

- Run scripts/audit_data_quality.py --fix-enriched to add placeholder rows, then rerun TMDb enrichment for true metadata.
- Keep unrated movies for content/cold-start demos; filter them out for pure collaborative training if needed.
- Report warm-item and cold-item ranking metrics separately; pure CF cannot learn unseen train items.
- Use minimum-interaction filters or segment metrics by item popularity for stable offline comparisons.
- Treat zero budget/revenue as missing values; do not use finance metadata as a primary feature.
