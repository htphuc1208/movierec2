"""Sidecar storage for user ratings submitted through the demo API."""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


RATING_COLUMNS = ["user_id", "movie_id", "rating", "timestamp"]


@dataclass(frozen=True)
class SidecarRatingStore:
    path: Path

    def append(self, user_id: int, movie_id: int, rating: float, timestamp: int | None = None) -> dict[str, Any]:
        timestamp = int(timestamp or time.time())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=RATING_COLUMNS)
            if needs_header:
                writer.writeheader()
            writer.writerow({"user_id": int(user_id), "movie_id": int(movie_id), "rating": float(rating), "timestamp": timestamp})
        return {"user_id": int(user_id), "movie_id": int(movie_id), "rating": float(rating), "timestamp": timestamp}

    def latest_rating(self, user_id: int, movie_id: int) -> float | None:
        frame = self._read()
        if frame.empty:
            return None
        rows = frame[(frame["user_id"] == int(user_id)) & (frame["movie_id"] == int(movie_id))]
        if rows.empty:
            return None
        return float(rows.sort_values("timestamp").iloc[-1]["rating"])

    def ratings_for_user(self, user_id: int) -> list[dict[str, Any]]:
        frame = self._read()
        if frame.empty:
            return []
        rows = frame[frame["user_id"] == int(user_id)].sort_values("timestamp", ascending=False)
        latest = rows.drop_duplicates(subset=["movie_id"], keep="first")
        return [
            {
                "user_id": int(row.user_id),
                "movie_id": int(row.movie_id),
                "rating": float(row.rating),
                "timestamp": int(row.timestamp),
            }
            for row in latest.itertuples(index=False)
        ]

    def _read(self) -> pd.DataFrame:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return pd.DataFrame(columns=RATING_COLUMNS)
        frame = pd.read_csv(self.path)
        for column in RATING_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.Series(dtype="float64")
        frame = frame[RATING_COLUMNS].copy()
        frame["user_id"] = pd.to_numeric(frame["user_id"], errors="coerce")
        frame["movie_id"] = pd.to_numeric(frame["movie_id"], errors="coerce")
        frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce")
        frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce")
        frame = frame.dropna(subset=RATING_COLUMNS)
        frame["user_id"] = frame["user_id"].astype(int)
        frame["movie_id"] = frame["movie_id"].astype(int)
        frame["timestamp"] = frame["timestamp"].astype(int)
        return frame

