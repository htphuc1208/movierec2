#!/usr/bin/env python3
"""Comprehensive EDA report generator for movierec3.

Generates static plots (PNG) and a markdown report covering the full
Data Science pipeline story:
  1. Data Overview & Quality
  2. Rating Distribution Analysis
  3. User Activity Analysis (power-law, cold-start)
  4. Item Popularity Analysis (long-tail)
  5. Temporal Patterns
  6. Genre/Content Analysis
  7. Sparsity & Interaction Matrix
  8. User Segmentation
  9. Cross-dataset Comparison

Output: reports/eda_{dataset}/ with PNG charts and eda_report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from recommender.analysis.eda import (
    genre_counts,
    infer_interaction_columns,
    load_catalog,
    load_interactions,
    overview_stats,
    rating_distribution,
    top_interacted_movies,
    user_segmentation,
)

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate comprehensive EDA report")
    parser.add_argument("--dataset", choices=["movielens", "letterboxd", "both"], default="both")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    return parser.parse_args()


DATASETS = {
    "movielens": {
        "name": "MovieLens (ml-latest-small)",
        "ratings": "data/raw/ml-latest-small/ratings.csv",
        "catalog": "data/processed/movie_catalog_enriched.parquet",
        "movies_raw": "data/raw/ml-latest-small/movies.csv",
    },
    "letterboxd": {
        "name": "Letterboxd (crawled)",
        "ratings": "data/processed/letterboxd/ratings.csv",
        "catalog": "data/processed/letterboxd/movie_catalog_enriched.parquet",
        "movies_raw": "data/processed/letterboxd/movies.csv",
    },
}


def generate_eda(dataset_key: str, output_dir: Path) -> str:
    """Generate full EDA for one dataset. Returns markdown text."""
    info = DATASETS[dataset_key]
    root = Path(__file__).resolve().parent.parent
    ratings_path = root / info["ratings"]
    catalog_path = root / info["catalog"]
    movies_raw_path = root / info["movies_raw"]

    output_dir.mkdir(parents=True, exist_ok=True)
    md_lines: list[str] = []

    def save_fig(name: str) -> str:
        path = output_dir / f"{name}.png"
        plt.savefig(path)
        plt.close()
        return str(path)

    # Load data
    interactions = load_interactions(str(ratings_path))
    catalog = load_catalog(str(catalog_path))
    cols = infer_interaction_columns(interactions)

    if movies_raw_path.exists():
        movies_raw = pd.read_csv(movies_raw_path)
    else:
        movies_raw = None

    # ===== 1. DATA OVERVIEW =====
    md_lines.append(f"# Phân tích Khám phá Dữ liệu (EDA) — {info['name']}\n")
    md_lines.append("## 1. Tổng quan dữ liệu (Data Overview)\n")

    stats = overview_stats(interactions, cols)
    md_lines.append(f"| Thuộc tính | Giá trị |")
    md_lines.append(f"|---|---:|")
    md_lines.append(f"| Số người dùng | {stats['users']:,} |")
    md_lines.append(f"| Số phim có tương tác | {stats['items']:,} |")
    md_lines.append(f"| Tổng interactions | {stats['interactions']:,} |")
    md_lines.append(f"| Sparsity | {stats['sparsity']*100:.2f}% |")
    avg_rating = interactions[cols.rating].mean()
    md_lines.append(f"| Rating trung bình | {avg_rating:.2f} |")
    md_lines.append(f"| Rating trung vị | {interactions[cols.rating].median():.1f} |")
    md_lines.append(f"| Rating min | {interactions[cols.rating].min():.1f} |")
    md_lines.append(f"| Rating max | {interactions[cols.rating].max():.1f} |")
    md_lines.append("")

    # Data quality
    md_lines.append("### 1.1. Chất lượng dữ liệu\n")
    missing = interactions.isnull().sum()
    md_lines.append("**Missing values trong bảng interactions:**\n")
    for c in interactions.columns:
        md_lines.append(f"- `{c}`: {missing[c]} ({missing[c]/len(interactions)*100:.2f}%)")
    md_lines.append("")

    if catalog is not None:
        cat_missing = catalog.isnull().sum()
        key_cols = [c for c in ["title", "genres", "tmdb_genres", "overview", "release_year", "popularity", "vote_average", "poster_path", "director", "cast_names"] if c in catalog.columns]
        md_lines.append("**Missing values trong catalog (các cột quan trọng):**\n")
        for c in key_cols:
            pct = cat_missing[c] / len(catalog) * 100
            md_lines.append(f"- `{c}`: {cat_missing[c]} ({pct:.1f}%)")
        md_lines.append("")

    # ===== 2. RATING DISTRIBUTION =====
    md_lines.append("## 2. Phân phối Rating\n")
    ratings = rating_distribution(interactions, cols)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(ratings["rating"].astype(str), ratings["count"], color=sns.color_palette("Blues_d", len(ratings)))
    axes[0].set_title(f"Phân phối rating — {info['name']}")
    axes[0].set_xlabel("Rating")
    axes[0].set_ylabel("Số lượng")
    for i, (_, row) in enumerate(ratings.iterrows()):
        axes[0].text(i, row["count"] + max(ratings["count"])*0.01, f"{int(row['count']):,}", ha="center", fontsize=8)

    # Cumulative
    sorted_r = ratings.sort_values("rating")
    sorted_r["cumulative"] = sorted_r["count"].cumsum() / sorted_r["count"].sum() * 100
    axes[1].plot(sorted_r["rating"].astype(str), sorted_r["cumulative"], "o-", color="coral", linewidth=2)
    axes[1].axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_title("Phân phối tích lũy rating")
    axes[1].set_xlabel("Rating")
    axes[1].set_ylabel("% tích lũy")
    axes[1].set_ylim(0, 105)
    fig.tight_layout()
    path = save_fig("01_rating_distribution")
    md_lines.append(f"![Phân phối rating]({path})\n")

    # Positive ratio
    pos_count = (interactions[cols.rating] >= 4.0).sum()
    md_lines.append(f"- Tỷ lệ positive (rating ≥ 4.0): **{pos_count:,}** / {len(interactions):,} = **{pos_count/len(interactions)*100:.1f}%**")
    md_lines.append(f"- Rating trung bình: **{avg_rating:.2f}** — cho thấy xu hướng rating {'tích cực' if avg_rating >= 3.5 else 'trung bình' if avg_rating >= 3.0 else 'thấp'}")
    md_lines.append("")

    # ===== 3. USER ACTIVITY ANALYSIS =====
    md_lines.append("## 3. Phân tích hoạt động User\n")

    user_counts = interactions.groupby(cols.user).size()
    md_lines.append(f"| Thuộc tính | Giá trị |")
    md_lines.append(f"|---|---:|")
    md_lines.append(f"| Trung bình rating/user | {user_counts.mean():.1f} |")
    md_lines.append(f"| Trung vị rating/user | {user_counts.median():.1f} |")
    md_lines.append(f"| Min rating/user | {user_counts.min()} |")
    md_lines.append(f"| Max rating/user | {user_counts.max()} |")
    md_lines.append(f"| Users có ≤ 20 ratings (cold-start) | {(user_counts <= 20).sum()} ({(user_counts <= 20).mean()*100:.1f}%) |")
    md_lines.append(f"| Users có > 100 ratings (power users) | {(user_counts > 100).sum()} ({(user_counts > 100).mean()*100:.1f}%) |")
    md_lines.append("")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(user_counts.clip(upper=500), bins=50, color="steelblue", edgecolor="white")
    axes[0].set_title("Phân phối số ratings/user")
    axes[0].set_xlabel("Số ratings")
    axes[0].set_ylabel("Số users")
    axes[0].axvline(x=user_counts.median(), color="red", linestyle="--", label=f"Median={user_counts.median():.0f}")
    axes[0].legend()

    # Log-log plot for power law
    value_counts = user_counts.value_counts().sort_index()
    axes[1].scatter(value_counts.index, value_counts.values, s=10, alpha=0.6, color="coral")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_title("Power-law: User activity (log-log)")
    axes[1].set_xlabel("Số ratings (log)")
    axes[1].set_ylabel("Số users (log)")
    fig.tight_layout()
    path = save_fig("02_user_activity")
    md_lines.append(f"![User activity]({path})\n")
    md_lines.append("**Nhận xét:** Phân phối hoạt động user theo dạng **power-law** — một số ít user rất tích cực (power users), đa số user có ít ratings. Đây là đặc trưng phổ biến của recommendation datasets.\n")

    # ===== 4. ITEM POPULARITY (LONG-TAIL) =====
    md_lines.append("## 4. Phân tích Long-tail phim\n")

    item_counts = interactions.groupby(cols.item).size().sort_values(ascending=False)
    md_lines.append(f"| Thuộc tính | Giá trị |")
    md_lines.append(f"|---|---:|")
    md_lines.append(f"| Trung bình ratings/phim | {item_counts.mean():.1f} |")
    md_lines.append(f"| Trung vị ratings/phim | {item_counts.median():.1f} |")
    md_lines.append(f"| Phim chỉ có 1 rating | {(item_counts == 1).sum()} ({(item_counts == 1).mean()*100:.1f}%) |")
    md_lines.append(f"| Phim có ≤ 5 ratings | {(item_counts <= 5).sum()} ({(item_counts <= 5).mean()*100:.1f}%) |")
    md_lines.append(f"| Top-1% phim chiếm bao nhiêu % interactions | {item_counts.head(max(1, len(item_counts)//100)).sum()/len(interactions)*100:.1f}% |")
    md_lines.append("")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # Long-tail curve
    cumulative = item_counts.cumsum() / item_counts.sum() * 100
    axes[0].plot(range(len(cumulative)), cumulative.values, color="steelblue", linewidth=1.5)
    axes[0].axhline(y=80, color="red", linestyle="--", alpha=0.5, label="80% interactions")
    items_for_80 = (cumulative <= 80).sum()
    axes[0].axvline(x=items_for_80, color="red", linestyle="--", alpha=0.5)
    axes[0].set_title(f"Long-tail: {items_for_80} phim ({items_for_80/len(item_counts)*100:.0f}%) chiếm 80% interactions")
    axes[0].set_xlabel("Phim (sắp xếp theo popularity)")
    axes[0].set_ylabel("% tích lũy interactions")
    axes[0].legend()

    # Item popularity distribution
    axes[1].hist(item_counts.clip(upper=200), bins=50, color="coral", edgecolor="white")
    axes[1].set_title("Phân phối số ratings/phim")
    axes[1].set_xlabel("Số ratings")
    axes[1].set_ylabel("Số phim")
    fig.tight_layout()
    path = save_fig("03_longtail_items")
    md_lines.append(f"![Long-tail analysis]({path})\n")
    md_lines.append(f"**Nhận xét:** Hiện tượng **long-tail** rõ rệt — {items_for_80/len(item_counts)*100:.0f}% phim phổ biến nhất chiếm 80% tổng interactions. Phim ở long-tail (ít interaction) khó recommend bằng CF thuần, cần content-based để bổ trợ.\n")

    # ===== 5. TEMPORAL PATTERNS =====
    md_lines.append("## 5. Phân tích theo thời gian\n")

    ts_col = "timestamp" if "timestamp" in interactions.columns else None
    if ts_col:
        ts = pd.to_datetime(interactions[ts_col], unit="s", errors="coerce")
        interactions["_year"] = ts.dt.year
        interactions["_month"] = ts.dt.month
        interactions["_hour"] = ts.dt.hour
        interactions["_dow"] = ts.dt.dayofweek

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Ratings per year
        year_counts = interactions.groupby("_year").size()
        axes[0, 0].bar(year_counts.index.astype(str), year_counts.values, color="steelblue")
        axes[0, 0].set_title("Số ratings theo năm")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Ratings per month
        month_counts = interactions.groupby("_month").size()
        axes[0, 1].bar(month_counts.index, month_counts.values, color="coral")
        axes[0, 1].set_title("Số ratings theo tháng")
        axes[0, 1].set_xlabel("Tháng")

        # Ratings per hour
        hour_counts = interactions.groupby("_hour").size()
        axes[1, 0].bar(hour_counts.index, hour_counts.values, color="seagreen")
        axes[1, 0].set_title("Số ratings theo giờ trong ngày")
        axes[1, 0].set_xlabel("Giờ")

        # Day of week
        dow_labels = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]
        dow_counts = interactions.groupby("_dow").size()
        axes[1, 1].bar([dow_labels[i] for i in dow_counts.index], dow_counts.values, color="mediumpurple")
        axes[1, 1].set_title("Số ratings theo ngày trong tuần")

        fig.tight_layout()
        path = save_fig("04_temporal_patterns")
        md_lines.append(f"![Temporal patterns]({path})\n")

        interactions.drop(columns=["_year", "_month", "_hour", "_dow"], inplace=True, errors="ignore")
    else:
        md_lines.append("*Dataset này không có timestamp tin cậy, bỏ qua phân tích temporal.*\n")

    # ===== 6. GENRE / CONTENT ANALYSIS =====
    md_lines.append("## 6. Phân tích thể loại và nội dung\n")

    genres = genre_counts(catalog, top_n=20)
    if not genres.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].barh(genres["genre"][::-1], genres["count"][::-1], color=sns.color_palette("viridis", len(genres)))
        axes[0].set_title("Top 20 thể loại phim")
        axes[0].set_xlabel("Số phim")

        # Genre co-occurrence heatmap
        genre_col = "tmdb_genres" if "tmdb_genres" in catalog.columns else "genres"
        if genre_col in catalog.columns:
            genre_matrix = catalog[genre_col].fillna("").astype(str).str.replace(",", "|").str.get_dummies(sep="|")
            genre_matrix = genre_matrix.loc[:, [c for c in genre_matrix.columns if c and c != "(no genres listed)"]]
            top_genres = genre_matrix.sum().nlargest(12).index.tolist()
            if len(top_genres) >= 3:
                co_occur = genre_matrix[top_genres].T.dot(genre_matrix[top_genres])
                co_occur_vals = co_occur.values.copy()
                np.fill_diagonal(co_occur_vals, 0)
                co_occur = pd.DataFrame(co_occur_vals, index=co_occur.index, columns=co_occur.columns)
                sns.heatmap(co_occur, cmap="YlOrRd", annot=True, fmt="d", ax=axes[1], cbar_kws={"shrink": 0.8})
                axes[1].set_title("Ma trận đồng xuất hiện thể loại (Top 12)")
        fig.tight_layout()
        path = save_fig("05_genre_analysis")
        md_lines.append(f"![Genre analysis]({path})\n")

    # Release year distribution
    if "release_year" in catalog.columns:
        year_data = pd.to_numeric(catalog["release_year"], errors="coerce").dropna()
        year_data = year_data[(year_data >= 1900) & (year_data <= 2030)]
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.hist(year_data, bins=60, color="teal", edgecolor="white")
        ax.set_title("Phân phối năm phát hành phim trong catalog")
        ax.set_xlabel("Năm")
        ax.set_ylabel("Số phim")
        path = save_fig("06_release_year")
        md_lines.append(f"![Release year]({path})\n")

    # ===== 7. SPARSITY VISUALIZATION =====
    md_lines.append("## 7. Phân tích Sparsity\n")
    md_lines.append(f"- Ma trận user-item có kích thước: **{stats['users']:,} × {stats['items']:,}** = **{stats['users']*stats['items']:,}** ô")
    md_lines.append(f"- Chỉ có **{stats['interactions']:,}** ô có giá trị → Sparsity = **{stats['sparsity']*100:.2f}%**")
    md_lines.append(f"- Đây là mức sparsity {'rất cao' if stats['sparsity'] > 0.99 else 'cao' if stats['sparsity'] > 0.95 else 'trung bình'}, đặc trưng cho recommendation datasets.\n")

    # Sample interaction matrix
    top_users = interactions.groupby(cols.user).size().nlargest(50).index
    top_items = interactions.groupby(cols.item).size().nlargest(100).index
    subset = interactions[interactions[cols.user].isin(top_users) & interactions[cols.item].isin(top_items)]
    if len(subset) > 10:
        pivot = subset.pivot_table(index=cols.user, columns=cols.item, values=cols.rating, aggfunc="first")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.imshow(pivot.notna().values.astype(float), aspect="auto", cmap="Blues", interpolation="nearest")
        ax.set_title(f"Mẫu ma trận tương tác (Top 50 users × Top 100 items)")
        ax.set_xlabel("Items")
        ax.set_ylabel("Users")
        path = save_fig("07_sparsity_matrix")
        md_lines.append(f"![Sparsity matrix]({path})\n")

    # ===== 8. USER SEGMENTATION =====
    md_lines.append("## 8. Phân cụm người dùng (User Segmentation)\n")
    md_lines.append("Sử dụng K-Means clustering trên feature trung bình của các phim mà user đã xem (thể loại, popularity, vote_average, release_year).\n")

    clusters, genre_cols_list = user_segmentation(interactions, catalog, cols, n_clusters=4, method="kmeans")
    if not clusters.empty and "PCA1" in clusters.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for cluster_name in clusters["cluster_name"].unique():
            mask = clusters["cluster_name"] == cluster_name
            axes[0].scatter(clusters.loc[mask, "PCA1"], clusters.loc[mask, "PCA2"], label=cluster_name, s=20, alpha=0.6)
        axes[0].set_title("User segmentation (PCA + KMeans)")
        axes[0].set_xlabel("PCA1")
        axes[0].set_ylabel("PCA2")
        axes[0].legend(fontsize=8)

        # Cluster profile
        profile = clusters.groupby("cluster_name").agg(
            count=(cols.user, "count"),
            **{c: (c, "mean") for c in ["popularity", "vote_average", "release_year"] if c in clusters.columns}
        ).reset_index()
        cell_text = []
        col_names = list(profile.columns)
        for _, row in profile.iterrows():
            cell_text.append([f"{v:.1f}" if isinstance(v, float) else str(v) for v in row])
        axes[1].axis("off")
        table = axes[1].table(cellText=cell_text, colLabels=col_names, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        axes[1].set_title("Đặc điểm các cụm user")
        fig.tight_layout()
        path = save_fig("08_user_segmentation")
        md_lines.append(f"![User segmentation]({path})\n")

        # Top genre per cluster
        if genre_cols_list:
            top_per_cluster = clusters.groupby("cluster_name")[genre_cols_list].mean()
            for cname in top_per_cluster.index:
                top3 = top_per_cluster.loc[cname].nlargest(3)
                genre_str = ", ".join([f"{g} ({v:.2f})" for g, v in top3.items()])
                md_lines.append(f"- **{cname}**: Top thể loại: {genre_str}")
            md_lines.append("")

    # ===== 9. TOP MOVIES =====
    md_lines.append("## 9. Top phim được tương tác nhiều nhất\n")
    top = top_interacted_movies(interactions, catalog, cols, top_n=15)
    if not top.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(top["title"][::-1], top["interactions"][::-1], color=sns.color_palette("rocket", len(top)))
        ax.set_title(f"Top 15 phim — {info['name']}")
        ax.set_xlabel("Số interactions")
        path = save_fig("09_top_movies")
        md_lines.append(f"![Top movies]({path})\n")

    # ===== 10. SUMMARY =====
    md_lines.append("## 10. Tổng kết EDA\n")
    md_lines.append("### Các phát hiện chính:\n")
    md_lines.append(f"1. **Sparsity cao ({stats['sparsity']*100:.1f}%)** — cần kỹ thuật CF hiệu quả (LightGCN, EASE) và content-based để bổ trợ.")
    md_lines.append(f"2. **Long-tail rõ rệt** — {items_for_80/len(item_counts)*100:.0f}% phim chiếm 80% interactions. Content-based giúp recommend phim ít tương tác.")
    md_lines.append(f"3. **Power-law user activity** — đa số user có ít ratings, cần xử lý cold-start bằng popularity fallback.")
    md_lines.append(f"4. **Rating nghiêng về tích cực** — trung bình {avg_rating:.2f}, phù hợp dùng implicit feedback (threshold ≥ 4.0).")
    if genre_cols_list:
        md_lines.append(f"5. **User segmentation** — phân được {clusters['cluster_name'].nunique()} nhóm user rõ ràng theo sở thích thể loại.")
    md_lines.append("")
    md_lines.append("### Ý nghĩa cho thiết kế hệ thống:\n")
    md_lines.append("- Hybrid approach (CF + Content) phù hợp vì CF tốt cho warm users, Content giúp cold-start và long-tail items.")
    md_lines.append("- LightGCN khai thác graph structure user-item, phù hợp với dữ liệu implicit feedback.")
    md_lines.append("- Min-Max normalization cho hybrid scoring để cân bằng scale giữa các component.")
    md_lines.append("- User segmentation giúp hiểu đối tượng và personalize recommendation strategy.")
    md_lines.append("")

    # Save markdown
    md_text = "\n".join(md_lines)
    report_path = output_dir / "eda_report.md"
    report_path.write_text(md_text, encoding="utf-8")
    print(f"Report: {report_path}")
    return md_text


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent.parent

    datasets_to_run = [args.dataset] if args.dataset != "both" else ["movielens", "letterboxd"]

    for ds in datasets_to_run:
        output_dir = Path(args.output_dir) if args.output_dir else root / "reports" / f"eda_{ds}"
        print(f"\n{'='*60}")
        print(f"Generating EDA for {ds}")
        print(f"{'='*60}")
        generate_eda(ds, output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
