# Data Quality Report

- Data dir: `data/letterboxd-full`
- Generated at: `2026-06-03T04:31:48.336144+00:00`

## Core Counts

- Users: 551
- Movies: 6720
- Ratings: 33946
- Rated movies: 6720
- Sparsity: 99.0832%

## Main Issues

- rating_movieIds_not_in_movies: 0
- tag_movieIds_not_in_movies: 0
- link_movieIds_not_in_movies: 0
- enriched_movieIds_not_in_movies: 0
- movies_without_ratings: 0
- movies_without_links: 0
- movies_without_enriched: 0
- any_core_metadata_missing: 6720
- all_core_metadata_missing: 6720
- no_genres_listed: 6720
- missing_year: 787
- tagged_movies: 0
- tagged_movies_pct: 0.0
- no_budget_no_revenue: 6720

## Split Diagnostics

- train_size: 27142
- val_size: 3402
- test_size: 3402
- train_pct: 79.9564
- val_pct: 10.0218
- test_pct: 10.0218
- train_users: 551
- val_users: 547
- test_users: 547
- test_users_not_in_train: 0
- val_movies_not_in_train: 389
- test_movies_not_in_train: 613
- val_warm_interactions: 2968
- val_cold_interactions: 434
- test_warm_interactions: 2657
- test_cold_interactions: 745
- test_cold_positive_interactions: 333

## Recommendations

- Use warm-item ranking as the primary pure-CF metric, and report cold-item metrics separately for hybrid/content fallback.
- Use minimum-interaction filters or segment metrics by item popularity for stable offline comparisons.
- Treat zero budget/revenue as missing values; do not use finance metadata as a primary feature.
