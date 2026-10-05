"""
Genre label embedding cache.

Loaded ONCE at FastAPI startup via warm_up().  Zero per-request latency.

Purpose:
  The composite embedding approach (Approach A) requires that when a user
  skips a movie, we compute:

      composite_neg = normalize(λ · genre_vec + (1-λ) · plot_vec)

  and store that as the negative embedding in the session (instead of the
  raw plot vector).  This makes Rocchio genre-aware: skipping Action no longer
  bleeds into Drama because genre label embeddings are much more separated
  (cos("Action","Drama") ≈ 0.71) than plot embeddings (cos ≈ 0.96).

  Genre label embeddings are ~20 unique strings — negligible to cache.

Cache key: genre string exactly as it appears in movie.genres list.
  e.g. "Action", "Sci-Fi", "Film-Noir"

When a movie has multiple genres (["Action","Drama"]):
  genres_str = " ".join(sorted(genres))  →  "Action Drama"
  The combined string is embedded as a single vector, which naturally
  interpolates between the two genre directions.

Fallback:
  If a genre is not in the cache (e.g. unseen genre label), we return None
  and the caller falls back to the raw plot embedding — graceful degradation.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import httpx
import numpy as np

from app.config import settings
from app.services.vector_math import blend_composite, l2_normalize

logger = logging.getLogger(__name__)

# Module-level cache: {genre_string: normalised float32 array (1024,)}
_cache: Dict[str, np.ndarray] = {}

# All genre labels found in the embedded_movies collection
KNOWN_GENRES: List[str] = [
    "Action", "Adventure", "Animation", "Biography", "Comedy",
    "Crime", "Documentary", "Drama", "Family", "Fantasy",
    "Film-Noir", "History", "Horror", "Music", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Short", "Sport",
    "Thriller", "War", "Western",
]


# ---------------------------------------------------------------------------
# Warm-up (called once at server startup)
# ---------------------------------------------------------------------------

async def warm_up() -> None:
    """
    Embed all known genre label strings via Voyage AI and cache the results.

    Uses input_type="document" because the genre embeddings are used on the
    document side (they are blended into stored movie embeddings and session
    negative embeddings — not used as search queries directly).

    Makes a single batched Voyage API call for all genres.
    Falls back gracefully: if the API is unavailable, the cache stays empty
    and the system degrades to plot-only Rocchio with a warning.
    """
    global _cache
    if _cache:
        logger.info("Genre embedding cache already warm (%d entries).", len(_cache))
        return

    logger.info("Warming genre embedding cache (%d genres)…", len(KNOWN_GENRES))

    url = f"{settings.VOYAGE_BASE_URL.rstrip('/')}/embeddings"
    payload = {
        "input":      KNOWN_GENRES,
        "model":      settings.VOYAGE_MODEL,
        "input_type": "document",
    }
    headers = {
        "Authorization": f"Bearer {settings.VOYAGE_API_KEY}",
        "Content-Type":  "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)

        if response.status_code != 200:
            logger.warning(
                "Genre cache warm-up failed (HTTP %d): %s",
                response.status_code, response.text[:200],
            )
            return

        data = response.json()
        for genre, item in zip(KNOWN_GENRES, data["data"]):
            arr = np.array(item["embedding"], dtype=np.float32)
            _cache[genre] = arr / np.linalg.norm(arr)

        logger.info(
            "Genre embedding cache warmed: %d genres cached, dim=%d",
            len(_cache), next(iter(_cache.values())).shape[0] if _cache else 0,
        )

    except Exception as exc:
        logger.warning(
            "Genre embedding cache warm-up exception: %s — "
            "falling back to plot-only Rocchio.",
            exc,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_genre_vec(genre: str) -> Optional[np.ndarray]:
    """
    Return the cached normalised embedding for a single genre label.
    Returns None if the genre is not cached (graceful fallback).
    """
    return _cache.get(genre)


def get_composite_neg_embedding(
    plot_embedding: np.ndarray,
    genres: List[str],
    lam: float = settings.GENRE_BLEND_LAMBDA,
) -> np.ndarray:
    """
    Compute the composite negative embedding for a skipped movie.

        composite = normalize( λ · embed(genres_str) + (1-λ) · plot_vec )

    When the movie has multiple genres, the genre string is joined:
        ["Action","Drama"] → embed("Action Drama")

    The combined-genre embedding naturally interpolates between genre directions,
    making the penalty proportional to each genre's presence in the movie.

    Falls back to pure plot embedding if genre cache is empty (API unavailable
    at startup) — system still works, just without genre separation.

    Args:
        plot_embedding: Raw plot_embedding_voyage_4 as float32 array.
        genres:         Movie's genres list, e.g. ["Action","Drama"].
        lam:            Genre blend weight λ.

    Returns:
        Composite float32 array, normalised, shape (1024,).
    """
    if not _cache:
        # Cache empty — fall back to plot-only (no genre separation)
        return l2_normalize(plot_embedding.astype(np.float32))

    # Sort genres for reproducibility, join as one string
    genres_str = " ".join(sorted(g for g in genres if g))

    # Try exact combined string first (e.g. "Action Drama")
    genre_vec = _cache.get(genres_str)

    if genre_vec is None:
        # Combined string not cached — average the individual genre vectors
        individual = [_cache[g] for g in genres if g in _cache]
        if not individual:
            # No genres in cache — fall back to plot-only
            return l2_normalize(plot_embedding.astype(np.float32))
        avg = np.mean(individual, axis=0)
        genre_vec = avg / np.linalg.norm(avg)

    return blend_composite(genre_vec, plot_embedding, lam)


def is_warmed() -> bool:
    """Return True if the cache has been populated."""
    return bool(_cache)


def cache_size() -> int:
    """Number of genre labels currently cached."""
    return len(_cache)


async def embed_texts_via_voyage(texts: List[str]) -> List[List[float]]:
    """
    Embed a batch of texts via Voyage AI with input_type='document'.

    Shared helper used by restore_user_vector() in session_service to
    re-embed the seed profile text at demo reset time.

    Returns list of embedding vectors (list[float]) in the same order as texts.
    Raises on HTTP error or API failure.
    """
    url = f"{settings.VOYAGE_BASE_URL.rstrip('/')}/embeddings"
    payload = {
        "input":      texts,
        "model":      settings.VOYAGE_MODEL,
        "input_type": "document",
    }
    headers = {
        "Authorization": f"Bearer {settings.VOYAGE_API_KEY}",
        "Content-Type":  "application/json",
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload, headers=headers)
    response.raise_for_status()
    data = response.json()
    return [item["embedding"] for item in data["data"]]
