import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("./data") 
ENRICHED_PATH = DATA_DIR / "processed" / "movie_catalog_enriched.parquet"
MOVIELENS_DIR = DATA_DIR / "raw" / "ml-latest-small"
LETTERBOXD_DIR = DATA_DIR / "letterboxd" / "data" / "raw"

st.set_page_config(page_title="EDA Dashboard - Movie Recommendation", layout="wide")
st.title(" EDA Dashboard ")
st.markdown("Thống kê và khám phá dữ liệu tương tác từ MovieLens & Letterboxd kết hợp metadata TMDB.")

@st.cache_data
def load_metadata():
    if ENRICHED_PATH.exists():
        df = pd.read_parquet(ENRICHED_PATH)
    else:
        st.error(f"Không tìm thấy file enriched catalog tại {ENRICHED_PATH}")
        df = pd.DataFrame()
    return df

@st.cache_data
def load_interactions(source_type):
    if source_type == "MovieLens":
        path = MOVIELENS_DIR / "ratings.csv"
        if path.exists():
            df = pd.read_csv(path)
            df.columns = [col.lower() for col in df.columns]
            return df
    else:
        path = LETTERBOXD_DIR / "ratings.csv"
        if path.exists():
            df = pd.read_csv(path)
            df.columns = [col.lower() for col in df.columns]
            return df
    return pd.DataFrame()

@st.cache_data
def perform_user_segmentation(df_inter, df_meta, user_col, item_col, n_clusters=4, method="KMeans"):
    meta_id_col = 'id' if 'id' in df_meta.columns else ('movie_id' if 'movie_id' in df_meta.columns else df_meta.columns[0])
    
    df_inter_str = df_inter.copy()
    df_meta_str = df_meta.copy()
    
    df_inter_str[item_col] = df_inter_str[item_col].apply(lambda x: str(int(float(x))) if pd.notnull(x) else "")
    df_meta_str[meta_id_col] = df_meta_str[meta_id_col].apply(lambda x: str(int(float(x))) if pd.notnull(x) else "")
    
    numeric_cols = ['popularity', 'vote_average', 'release_year']
    available_numeric = [c for c in numeric_cols if c in df_meta_str.columns]
    for col in available_numeric:
        df_meta_str[col] = pd.to_numeric(df_meta_str[col], errors='coerce').fillna(0)
    
    sbert_cols = [c for c in df_meta_str.columns if c.startswith('sbert_') or c.startswith('emb_')]
    
    if 'genres' in df_meta_str.columns:
        df_meta_str['genres'] = df_meta_str['genres'].fillna('')
        genres_dummies = df_meta_str['genres'].str.get_dummies(sep='|')
        genre_list = genres_dummies.columns.tolist()
        df_meta_str = pd.concat([df_meta_str, genres_dummies], axis=1)
    else:
        genre_list = []

    meta_cols = [meta_id_col] + available_numeric + genre_list + sbert_cols
    df_merged = df_inter_str.merge(df_meta_str[meta_cols], left_on=item_col, right_on=meta_id_col, how='inner')
    
    if df_merged.empty:
        return pd.DataFrame(), []
        
    agg_dict = {col: 'mean' for col in available_numeric}
    for col in genre_list:
        agg_dict[col] = 'mean'
    for col in sbert_cols:
        agg_dict[col] = 'mean'
        
    df_user_features = df_merged.groupby(user_col).agg(agg_dict).reset_index()
    
    feature_cols = available_numeric + genre_list + sbert_cols
    X = df_user_features[feature_cols].fillna(0).values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if method == "KMeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df_user_features['cluster'] = model.fit_predict(X_scaled)
    else:
        model = GaussianMixture(n_components=n_clusters, random_state=42)
        df_user_features['cluster'] = model.fit_predict(X_scaled)
        
    cluster_mapping = {}
    if genre_list:
        genre_means = df_user_features.groupby('cluster')[genre_list].mean()
        for cluster_idx in range(n_clusters):
            if cluster_idx in genre_means.index:
                top_genre = genre_means.loc[cluster_idx].idxmax()
                cluster_mapping[cluster_idx] = f"Cụm {cluster_idx}: Fan {top_genre}"
            else:
                cluster_mapping[cluster_idx] = f"Cụm {cluster_idx}"
    else:
        for cluster_idx in range(n_clusters):
            cluster_mapping[cluster_idx] = f"Cụm {cluster_idx}"
            
    df_user_features['cluster_name'] = df_user_features['cluster'].map(cluster_mapping)
    
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    df_user_features['PCA1'] = X_pca[:, 0]
    df_user_features['PCA2'] = X_pca[:, 1]
    
    return df_user_features, genre_list

df_meta = load_metadata()

source = st.sidebar.selectbox("Chọn nguồn dữ liệu tương tác:", ["MovieLens", "Letterboxd"])
df_inter = load_interactions(source)

if df_inter.empty or df_meta.empty:
    st.warning("Vui lòng kiểm tra lại đường dẫn file dữ liệu trong code.")
else:
    user_col = 'userid' if 'userid' in df_inter.columns else ('user_id' if 'user_id' in df_inter.columns else df_inter.columns[0])
    item_col = 'movieid' if 'movieid' in df_inter.columns else ('movie_id' if 'movie_id' in df_inter.columns else df_inter.columns[1])
    rating_col = 'rating' if 'rating' in df_inter.columns else df_inter.columns[2]

    num_users = df_inter[user_col].nunique()
    num_items = df_inter[item_col].nunique()
    num_ratings = len(df_inter)
    
    total_cells = num_users * num_items
    sparsity = (1 - (num_ratings / total_cells)) * 100

    st.header(" Thống kê tổng quan")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Số lượng Users", f"{num_users:,}")
    col2.metric("Số lượng Phim (có tương tác)", f"{num_items:,}")
    col3.metric("Tổng số Ratings", f"{num_ratings:,}")
    col4.metric("Độ thưa ma trận (Sparsity)", f"{sparsity:.2f}%")

    st.markdown("---")

    st.header(" Phân phối dữ liệu")
    
    tab1, tab2, tab3 = st.tabs(["Phân phối Ratings", "Phân phối Thuộc tính phim (TMDB Metadata)", "Định danh & Phân cụm User (User Segmentation)"])
    
    with tab1:
        st.subheader("Biểu đồ phân phối điểm đánh giá (Rating)")
        rating_counts = df_inter[rating_col].value_counts().sort_index().reset_index()
        rating_counts.columns = ['Rating', 'Số lượng']
        fig_rating = px.bar(rating_counts, x='Rating', y='Số lượng', 
                            title=f"Phân phối số lượng theo mức điểm ({source})",
                            text_auto='.2s', color_discrete_sequence=['#ff0066'])
        st.plotly_chart(fig_rating, use_container_width=True)

    with tab2:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if 'release_year' in df_meta.columns:
                fig_year = px.histogram(df_meta, x='release_year', nbins=50,
                                        title="Phân phối năm phát hành của phim",
                                        labels={'release_year': 'Năm phát hành'}, color_discrete_sequence=['#3399ff'])
                st.plotly_chart(fig_year, use_container_width=True)
            
            if 'runtime' in df_meta.columns:
                fig_runtime = px.histogram(df_meta[df_meta['runtime'] > 0], x='runtime', nbins=50,
                                           title="Phân phối thời lượng phim (Runtime)",
                                           labels={'runtime': 'Thời lượng (phút)'}, color_discrete_sequence=['#00cc99'])
                st.plotly_chart(fig_runtime, use_container_width=True)

        with col_m2:
            if 'popularity' in df_meta.columns:
                fig_pop = px.histogram(df_meta, x='popularity', nbins=50, log_y=True,
                                       title="Phân phối độ nổi tiếng (Popularity - Log scale)",
                                       labels={'popularity': 'Độ nổi tiếng'}, color_discrete_sequence=['#ff9900'])
                st.plotly_chart(fig_pop, use_container_width=True)
            
            if 'vote_average' in df_meta.columns:
                fig_vote = px.histogram(df_meta[df_meta['vote_average'] > 0], x='vote_average', nbins=20,
                                        title="Phân phối điểm trung bình (Vote Average)",
                                        labels={'vote_average': 'Điểm TMDB'}, color_discrete_sequence=['#cc66ff'])
                st.plotly_chart(fig_vote, use_container_width=True)

    with tab3:
        st.subheader("Hồ sơ và Phân khúc nhóm hành vi người dùng")
        st.markdown("Tiến hành gom cụm User dựa trên phân phối tỷ lệ thể loại phim đã xem kết hợp hành vi tương tác.")
        
        col_ui1, col_ui2 = st.columns(2)
        with col_ui1:
            algo_method = st.selectbox("Chọn thuật toán phân cụm:", ["KMeans", "GMM (Gaussian Mixture)"])
        with col_ui2:
            num_clusters = st.slider("Chọn số lượng nhóm người dùng (K):", min_value=2, max_value=6, value=4)
        
        with st.spinner(f"Đang xử lý thuật toán {algo_method}..."):
            df_user_clustered, genres = perform_user_segmentation(
                df_inter, df_meta, user_col, item_col, n_clusters=num_clusters, method=algo_method
            )
        
        if df_user_clustered.empty:
            st.error("Không tìm thấy các thuộc tính hợp lệ để tiến hành phân cụm người dùng.")
        else:
            fig_cluster = px.scatter(
                df_user_clustered, 
                x='PCA1', 
                y='PCA2', 
                color='cluster_name',
                hover_data=[user_col, 'vote_average', 'popularity', 'release_year'],
                title=f"Không gian biểu diễn hành vi bằng {algo_method} kết hợp PCA 2D ({source})",
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_cluster.update_traces(marker=dict(size=6, opacity=0.85))
            st.plotly_chart(fig_cluster, use_container_width=True)
        
            st.write("### Bảng chỉ số đặc trưng trung bình của từng Nhóm")
            cluster_profile = df_user_clustered.groupby('cluster_name').mean(numeric_only=True).reset_index()
            show_cols = ['cluster_name', 'vote_average', 'popularity', 'release_year'] + (genres[:6] if genres else [])
        
            st.dataframe(
                cluster_profile[show_cols].style.format({
                    'vote_average': '{:.2f}',
                    'popularity': '{:.2f}',
                    'release_year': '{:.0f}'
                })
            ) 

    st.markdown("---")

    st.header(" Top-K Analytics")
    col_k1, col_k2 = st.columns(2)
    
    with col_k1:
        if 'genres' in df_meta.columns:
            st.subheader(" Top Thể loại phổ biến nhất")
            all_genres = df_meta['genres'].dropna().str.split('|').explode()
            genre_counts = all_genres.value_counts().reset_index()
            genre_counts.columns = ['Genre', 'Số lượng phim']
            
            fig_genre = px.bar(genre_counts.head(15), x='Số lượng phim', y='Genre', orientation='h',
                               color='Số lượng phim', color_continuous_scale='Viridis')
            fig_genre.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_genre, use_container_width=True)

    with col_k2:
        st.subheader(f" Top Phim được xem/rating nhiều nhất ({source})")
        top_interacted = df_inter[item_col].value_counts().reset_index()
        top_interacted.columns = [item_col, 'Lượt tương tác']
        
        meta_id_col = 'id' if 'id' in df_meta.columns else ('movie_id' if 'movie_id' in df_meta.columns else df_meta.columns[0])
        
        if 'title' in df_meta.columns:
            df_inter_clean = df_inter.copy()
            df_meta_clean = df_meta.copy()
            df_inter_clean[item_col] = df_inter_clean[item_col].apply(lambda x: str(int(float(x))) if pd.notnull(x) else "")
            df_meta_clean[meta_id_col] = df_meta_clean[meta_id_col].apply(lambda x: str(int(float(x))) if pd.notnull(x) else "")
            
            top_interacted_clean = df_inter_clean[item_col].value_counts().reset_index()
            top_interacted_clean.columns = [item_col, 'Lượt tương tác']
            
            top_movies = top_interacted_clean.head(10).merge(df_meta_clean[[meta_id_col, 'title']], left_on=item_col, right_on=meta_id_col, how='inner')
            
            if not top_movies.empty:
                fig_movie = px.bar(top_movies, x='Lượt tương tác', y='title', orientation='h',
                                   title=f"Top Phim được tương tác nhiều nhất ({source})",
                                   text='Lượt tương tác', color_discrete_sequence=['#ff4d4d'])
                fig_movie.update_layout(yaxis={'categoryorder':'total ascending'}, yaxis_title="Tên phim")
                st.plotly_chart(fig_movie, use_container_width=True)
            else:
                st.warning(f"Không thể khớp mã phim giữa file Ratings ({source}) và file Metadata. Hiển thị ID gốc:")
                st.dataframe(top_interacted.head(10))
        else:
            st.dataframe(top_interacted.head(10))

    st.subheader(" Top Đạo diễn & Diễn viên nổi bật (Dựa trên số lượng phim)")
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        if 'director' in df_meta.columns:
            top_directors = df_meta['director'].dropna().str.split('|').explode().value_counts().reset_index().head(10)
            top_directors.columns = ['Đạo diễn', 'Số phim']
            fig_dir = px.bar(top_directors, x='Số phim', y='Đạo diễn', orientation='h', color_discrete_sequence=['#45aaf2'])
            fig_dir.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_dir, use_container_width=True)
        else:
            st.info("File enriched của bạn chưa tách riêng trường 'director' hoặc cột tên khác.")
            
    with col_d2:
        if 'cast' in df_meta.columns:
            top_cast = df_meta['cast'].dropna().str.split('|').explode().value_counts().reset_index().head(10)
            top_cast.columns = ['Diễn viên', 'Số phim']
            fig_cast = px.bar(top_cast, x='Số phim', y='Diễn viên', orientation='h', color_discrete_sequence=['#4b7bec'])
            fig_cast.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_cast, use_container_width=True)
        else:
            st.info("File enriched chưa tách riêng trường 'cast' hoặc cột tên khác.")