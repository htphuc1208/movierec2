# App And API

This folder contains the Streamlit UI for the movie recommender.

## Runtime

Start the API first:

```bash
MOVIEREC_DATA_DIR=data/ml-latest-small \
MOVIEREC_ARTIFACT_DIR=artifacts/recommender/latest \
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then start the UI:

```bash
API_URL=http://localhost:8000 \
streamlit run app/streamlit_app.py --server.fileWatcherType none
```

The UI falls back to local inference if the API is not reachable.

## Useful API Views

- `POST /recommend`: hybrid recommendations.
- `GET /movies?search=...`: movie catalog search.
- `GET /movies/trending`: most-rated movies.
- `GET /movies/top-rated`: high average rating with vote threshold.
- `GET /movies/latest`: newest movies by parsed year.
- `GET /movies/genre/{genre}`: genre-filtered catalog.
- `GET /movies/{movie_id}/similar`: content-similar movies.
- `GET /users/{user_id}/history`: user's top rated history.

The API keeps the current artifact-first runtime: it loads `MOVIEREC_ARTIFACT_DIR` when available and otherwise fits the lightweight hybrid model from CSV.
