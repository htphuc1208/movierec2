# App And API

This folder contains the Streamlit frontends for the artifact-based movie recommender.

## Runtime

Start the FastAPI backend first:

```bash
PYTHONPATH=src:. \
ARTIFACTS_DIR=artifacts \
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Then start the main UI:

```bash
API_URL=http://localhost:8000 \
streamlit run app/streamlit_app.py --server.fileWatcherType none
```

Start the EDA dashboard separately when needed:

```bash
PYTHONPATH=src:. \
streamlit run app/eda_app.py --server.fileWatcherType none
```

## Useful API Views

- `POST /recommendations`: artifact-based recommendations.
- `POST /recommend`: compatibility alias for recommendations.
- `GET /movies?query=...`: movie catalog search.
- `GET /movies/trending`: trending movies from catalog metadata.
- `GET /movies/top-rated`: high-rated movies from catalog metadata.
- `GET /movies/latest`: newest movies by release year.
- `GET /movies/genre/{genre}`: genre-filtered catalog.
- `GET /movies/{movie_id}`: enriched movie detail.
- `GET /movies/{movie_id}/similar`: content-similar movies.
- `GET /users/{user_id}/history`: user history from train artifacts and sidecar ratings.
- `POST /rate`: append rating to the sidecar CSV store.
- `POST /chat`: RAG movie chatbot over artifact catalog metadata.

The UI talks to FastAPI only. Training and artifact export are handled by scripts under `scripts/`.
