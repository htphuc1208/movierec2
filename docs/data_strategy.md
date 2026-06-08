# Data Strategy

## Decisions

- Keep the full MovieLens catalog for API/UI and hybrid/content recommendation.
- Use train-rated items as the candidate universe for pure collaborative models such as SVD and LightGCN.
- Use warm-item top-k metrics as the primary pure-CF benchmark. Keep all-item and cold-item metrics in the reports for transparency.
- Keep `budget` and `revenue` as optional analysis fields only. They are not primary recommendation features.
- Prefer content fields in this order: title, genres, overview, tagline, director, cast, TMDb keywords, MovieLens Tag Genome tags, production companies/country, release/language/franchise/certification.

## Enrichment

TMDb enrichment uses `links.csv` and each row's `tmdbId`, so it does not need title search for MovieLens data.

Refresh rows with placeholder/empty core metadata:

```bash
export TMDB_API_KEY=your_tmdb_v3_api_key  # or set TMDB_API_KEY in .env

.venv/bin/python scripts/enrich_tmdb.py \
  --data-dir data/ml-latest-small \
  --retry-empty \
  --sleep 0.3
```

Refresh missing keywords and the new content metadata for the whole catalog:

```bash
export TMDB_API_KEY=your_tmdb_v3_api_key  # or set TMDB_API_KEY in .env

.venv/bin/python scripts/enrich_tmdb.py \
  --data-dir data/ml-latest-small \
  --refresh-empty-columns keywords,release_date,runtime,original_language,production_companies,production_countries,collection_name,certification \
  --sleep 0.3
```

For the full MovieLens dataset in `data/ml-latest`, create Tag Genome locally from the bundled genome files:

```bash
.venv/bin/python scripts/prepare_tag_genome.py \
  --data-dir data/ml-latest \
  --genome-dir data/ml-latest \
  --top-n 20 \
  --min-relevance 0.35
```

Then enrich missing TMDb metadata. This uses existing `links.csv` `tmdbId` values and skips rows that already have complete selected columns:

```bash
.venv/bin/python scripts/enrich_tmdb.py \
  --data-dir data/ml-latest \
  --refresh-empty-columns overview,director,cast,keywords,release_date,runtime,production_companies,production_countries,certification,vote_average,vote_count \
  --sleep 0.3
```

For Letterboxd, resolve missing `tmdbId` by title/year before fetching TMDb fields:

```bash
.venv/bin/python scripts/enrich_tmdb.py \
  --data-dir data/letterboxd-full \
  --retry-empty \
  --search-missing-tmdb \
  --sleep 0.3
```

The Tag Genome script writes `tag_genome.csv`, which the dataloader automatically joins into `tag_genome_tags`.

## Benchmark Claims

`ml-latest-small` is for development and smoke testing. Do not claim research-grade benchmark results from it.

Use MovieLens 25M when the project needs a movie-domain benchmark claim. It is stable, much larger, and includes Tag Genome features. Use RecBole configs with fixed temporal/warm-item protocol and multiple seeds.

Amazon Reviews'23, Yelp, Gowalla, and LastFM are useful external recommendation benchmarks, but they are not drop-in replacements for a movie recommender:

- Amazon Reviews'23: best for product recommendation and text/item metadata experiments.
- Yelp Open Dataset: best for local business/restaurant recommendation, geo/context, and review text.
- Gowalla: best for location/check-in and graph recommendation.
- LastFM/HetRec: best for music/listening and social/tag recommendation.

For this project, the default serious benchmark path is MovieLens 25M plus Tag Genome. Use the other datasets only if the product scope expands beyond movies or if the paper/report needs cross-domain robustness.
