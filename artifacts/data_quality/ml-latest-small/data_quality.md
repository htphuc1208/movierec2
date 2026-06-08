# Data Quality Report

- Data dir: `data/ml-latest-small`
- Generated at: `2026-06-03T08:29:40.035984+00:00`

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
- movies_without_enriched: 0
- any_core_metadata_missing: 156
- all_core_metadata_missing: 121
- no_genres_listed: 34
- missing_year: 12
- tagged_movies: 1572
- tagged_movies_pct: 16.1363
- no_budget_no_revenue: 2285

## Split Diagnostics

- train_size: 80650
- val_size: 10093
- test_size: 10093
- train_pct: 79.9814
- val_pct: 10.0093
- test_pct: 10.0093
- train_users: 610
- val_users: 610
- test_users: 610
- test_users_not_in_train: 0
- val_movies_not_in_train: 694
- test_movies_not_in_train: 884
- val_warm_interactions: 9344
- val_cold_interactions: 749
- test_warm_interactions: 9146
- test_cold_interactions: 947
- test_cold_positive_interactions: 359

## Recommendations

- Keep unrated movies for content/cold-start demos; filter them out for pure collaborative training if needed.
- Use warm-item ranking as the primary pure-CF metric, and report cold-item metrics separately for hybrid/content fallback.
- Use minimum-interaction filters or segment metrics by item popularity for stable offline comparisons.
- Treat zero budget/revenue as missing values; do not use finance metadata as a primary feature.
