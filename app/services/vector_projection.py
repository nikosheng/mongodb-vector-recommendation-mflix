"""
app/services/vector_projection.py
==================================
Projects a 1024-dim composite embedding onto a small set of fixed, human
-readable genre axes so it can be rendered as a radar/spider chart on the
frontend.

Why this works:
  Both `user_profiles.user_vector` and the RFY query vector `V_rfy` live in
  the same composite embedding space as the cached genre-label embeddings
  in `genre_embedding_cache` (see that module's docstring). All vectors are
  L2-normalised, so a dot product between the target vector and a genre
  axis embedding IS the cosine similarity between them — i.e. "how much
  does this vector currently point toward Horror / Sci-Fi / etc."

  Projecting onto a FIXED set of axes (rather than dynamically picking
  genres per request) keeps the radar chart's shape comparable frame to
  frame, so the frontend can animate smooth transitions instead of the
  axes themselves reshuffling.

Used by:
  app/routers/trails.py — computed once per GET /api/trails/{user_id}
  response for both the persistent user_vector and the ephemeral RFY
  query vector, so the frontend can render a "Live Profile Vector" radar
  chart showing both the slow-drifting baseline profile and the
  instantaneous per-request recommendation query vector.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from app.services import genre_embedding_cache
from app.services.vector_math import l2_normalize

logger = logging.getLogger(__name__)

# Fixed radar axes — chosen to give broad, stable coverage of the mflix
# genre space across all demo personas (horror / comedy / sci-fi / etc).
# Kept in sync with RADAR_AXES in app/static/js/app.js.
RADAR_AXES: List[str] = [
    "Action", "Comedy", "Drama", "Horror",
    "Romance", "Sci-Fi", "Thriller", "Documentary",
]


def project_onto_genre_axes(
    vector: np.ndarray,
    axes: List[str] = RADAR_AXES,
) -> Dict[str, float]:
    """
    Project a composite embedding onto each fixed genre axis via cosine
    similarity (dot product of two L2-normalised vectors).

    Args:
        vector: float32/float64 (1024,) composite embedding (any scale —
                will be normalised internally).
        axes:   list of genre labels to project onto. Defaults to the
                fixed RADAR_AXES set.

    Returns:
        {genre: cosine_similarity} — values roughly in [-1, 1]. Missing
        axes (genre not in the warmed cache) map to 0.0 with a warning,
        which should not normally happen since all RADAR_AXES genres are
        part of KNOWN_GENRES warmed at startup.
    """
    if vector is None or len(vector) == 0:
        return {genre: 0.0 for genre in axes}

    unit_vec = l2_normalize(np.asarray(vector, dtype=np.float64))

    scores: Dict[str, float] = {}
    for genre in axes:
        axis_vec = genre_embedding_cache.get_genre_vec(genre)
        if axis_vec is None:
            logger.warning(
                "project_onto_genre_axes: genre '%s' not in cache — "
                "returning 0.0 for this axis.", genre,
            )
            scores[genre] = 0.0
            continue
        scores[genre] = float(np.dot(unit_vec, axis_vec.astype(np.float64)))

    return scores


def vector_shift(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Cosine-distance shift metric between two vectors: 1 - cos(a, b).

    0.0   = identical direction (no shift)
    1.0   = orthogonal
    2.0   = opposite direction

    Used to drive the "shift %" badge in the Live Profile Vector HUD panel.
    """
    a = l2_normalize(np.asarray(vec_a, dtype=np.float64))
    b = l2_normalize(np.asarray(vec_b, dtype=np.float64))
    return 1.0 - float(np.dot(a, b))
