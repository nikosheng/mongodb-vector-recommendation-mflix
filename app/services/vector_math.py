"""
Pure NumPy vector utilities for bi-directional Rocchio feedback projection
and genre-aware composite embedding construction.

BSON Binary sub_type 9 storage format:
  byte 0 : 0x27  (dtype = float32)
  byte 1 : 0x00  (padding)
  bytes 2+: raw IEEE-754 float32 values (little-endian)
  Total: 2 + (1024 * 4) = 4098 bytes

Full bi-directional Rocchio formula:
  V_query = normalize(
      V_user
      + β · mean( V_played_i )     ← attract toward played content
      − α · mean( V_skipped_j )    ← repel away from skipped content
  )

Composite embedding formula (genre-aware):
  composite = normalize( λ · genre_vec + (1-λ) · plot_vec )
  Where λ = GENRE_BLEND_LAMBDA (default 0.35)
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np
from bson import Binary

logger = logging.getLogger(__name__)

_DTYPE_FLOAT32 = 0x27
_PADDING       = 0x00


# ---------------------------------------------------------------------------
# BSON <-> NumPy
# ---------------------------------------------------------------------------

def bson_binary_to_numpy(binary: bytes) -> np.ndarray:
    """
    Convert BSON Binary sub_type 9 (Float32Array) to a float32 NumPy array.
    Strips the 2-byte header (dtype + padding) when present.
    """
    raw = bytes(binary)
    if len(raw) % 4 != 0 and len(raw) >= 2 and (len(raw) - 2) % 4 == 0:
        raw = raw[2:]
    arr = np.frombuffer(raw, dtype=np.float32).copy()
    return arr


def numpy_to_bson_binary(arr: np.ndarray) -> Binary:
    """
    Encode a float32 NumPy array as BSON Vector Binary (subtype 9).
    Header: [0x27, 0x00] + IEEE-754 float32 payload.
    """
    header  = bytes([_DTYPE_FLOAT32, _PADDING])
    payload = arr.astype(np.float32).tobytes()
    return Binary(header + payload, subtype=9)


def list_to_numpy(embedding: List[float]) -> np.ndarray:
    """Convert a plain Python list[float] to float32 NumPy array."""
    return np.asarray(embedding, dtype=np.float32)


def numpy_to_list(arr: np.ndarray) -> List[float]:
    """Convert a NumPy array to a plain Python list[float] for JSON serialisation."""
    return arr.astype(np.float64).tolist()


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalise a vector. Falls back gracefully on near-zero vectors."""
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        logger.warning("l2_normalize: near-zero vector — returning as-is.")
        return vec
    return vec / norm


# ---------------------------------------------------------------------------
# Composite embedding (genre-aware)
# ---------------------------------------------------------------------------

def blend_composite(
    genre_vec: np.ndarray,
    plot_vec: np.ndarray,
    lam: float = 0.35,
) -> np.ndarray:
    """
    Compute a composite embedding that blends genre-label signal with plot signal.

        composite = normalize( λ · genre_vec + (1-λ) · plot_vec )

    Both inputs and the output are float32 arrays of shape (D,).

    Why this helps:
      In pure plot space, all genres cluster together (cos ≈ 0.95+) because
      movies share narrative themes regardless of genre label. Genre label
      embeddings are far more separated (cos ≈ 0.33–0.71).
      The composite shifts genre centroid pairs from 0.96 → 0.82, making
      Rocchio repulsion genre-specific rather than content-blob-specific.

    Args:
        genre_vec: embed("Action Drama") or embed("Horror") — normalised.
        plot_vec:  movie's plot_embedding_voyage_4 — any scale (will normalise).
        lam:       genre weight λ (default from GENRE_BLEND_LAMBDA = 0.35).

    Returns:
        Normalised composite float32 array, same shape as inputs.
    """
    genre_n = l2_normalize(genre_vec.astype(np.float32))
    plot_n  = l2_normalize(plot_vec.astype(np.float32))
    blended = lam * genre_n + (1.0 - lam) * plot_n
    return l2_normalize(blended).astype(np.float32)


# ---------------------------------------------------------------------------
# RFY query vector — orthogonal rejection (for global skips)
# ---------------------------------------------------------------------------

def compute_rfy_query(
    user_vector: np.ndarray,
    negative_embeddings: List[np.ndarray],
    rejection_strength: float = 0.8,
    positive_embeddings: Optional[Sequence[np.ndarray]] = None,
    beta: float = 0.3,
) -> List[float]:
    """
    Compute the RFY query vector using orthogonal rejection for global skips.

    **Why not simple Rocchio subtraction?**
    The user profile vector IS a Horror/Thriller/Action vector.
    Skipping Horror produces a neg_centroid that also points toward Horror.
    cos(user_vector, neg_centroid) ≈ 0.79 → they're nearly parallel.
    Subtracting a parallel vector barely changes direction after normalisation:
        normalize(user_vec − 0.4 · neg_centroid) ≈ user_vec  (cos = 0.96)

    **Orthogonal rejection:**
    Instead of subtracting a fraction of the centroid, we remove the entire
    component of user_vector that points IN THE DIRECTION OF skipped content.
    This works regardless of how aligned the vectors are:

        neg_dir  = normalize( mean(V_skipped) )
        projection = dot(user_vector, neg_dir)   ← how much profile faces skipped
        adjusted = user_vector − rejection_strength · projection · neg_dir
        V_query  = normalize(adjusted)

    With rejection_strength=0.8:
        - Removes 80% of the skipped content's direction from the profile
        - cos(user_vec, V_query) drops from 0.96 → ~0.50  (real visible shift)
        - The RFY will surface noticeably different content

    Positive plays are added AFTER rejection (standard Rocchio attraction):
        V_query += β · mean(V_played)

    Args:
        user_vector:         float32 (D,) — user composite profile vector.
        negative_embeddings: list of float32 (D,) — skipped movie composites.
        rejection_strength:  0.0–1.0. 1.0 = fully remove skipped direction.
        positive_embeddings: optional list of float32 (D,) — played composites.
        beta:                positive attraction weight.

    Returns:
        Normalised query vector as list[float] for Atlas $vectorSearch.
    """
    if not negative_embeddings:
        raise ValueError("compute_rfy_query requires at least one negative embedding.")

    adjusted = user_vector.copy().astype(np.float64)

    # Step 1 — orthogonal rejection of skipped content direction
    neg_matrix  = np.stack(negative_embeddings, axis=0).astype(np.float64)
    neg_centroid = neg_matrix.mean(axis=0)
    neg_dir     = neg_centroid / (np.linalg.norm(neg_centroid) + 1e-9)
    projection  = np.dot(adjusted, neg_dir)   # scalar: how much profile faces skipped content
    adjusted   -= rejection_strength * projection * neg_dir

    # Step 2 — positive attraction (unchanged from bi-directional Rocchio)
    if positive_embeddings:
        pos_matrix   = np.stack(positive_embeddings, axis=0).astype(np.float64)
        pos_centroid = pos_matrix.mean(axis=0)
        adjusted    += beta * pos_centroid

    result = l2_normalize(adjusted.astype(np.float32))

    logger.info(
        "RFY orthogonal rejection: k_neg=%d strength=%.2f  "
        "cos(user,query)=%.4f  shift=%.4f",
        len(negative_embeddings),
        rejection_strength,
        float(np.dot(l2_normalize(user_vector.astype(np.float32)), result)),
        float(np.linalg.norm(result - l2_normalize(user_vector.astype(np.float32)))),
    )

    return numpy_to_list(result)


# ---------------------------------------------------------------------------
# Bi-directional Rocchio (positive-only path — kept for plays)
# ---------------------------------------------------------------------------

def compute_rocchio_query(
    user_vector: np.ndarray,
    negative_embeddings: Sequence[np.ndarray],
    alpha: float = 0.4,
    positive_embeddings: Optional[Sequence[np.ndarray]] = None,
    beta: float = 0.3,
) -> List[float]:
    """
    Bi-directional Rocchio:

        V_query = normalize(
            V_user
            + β · mean( V_positive_i )    [omitted when no plays]
            − α · mean( V_negative_j )    [omitted when no skips]
        )

    In composite-embedding mode, both negative and positive embeddings should
    already be composite vectors (genre-blended) so that the Rocchio centroid
    moves in the correct genre-differentiated direction.

    Raises ValueError if both negative and positive lists are empty.
    """
    has_neg = bool(negative_embeddings)
    has_pos = bool(positive_embeddings)

    if not has_neg and not has_pos:
        raise ValueError(
            "compute_rocchio_query requires at least one embedding "
            "(negative or positive)."
        )

    adjusted = user_vector.copy().astype(np.float64)

    if has_pos:
        pos_matrix   = np.stack(positive_embeddings, axis=0).astype(np.float64)
        pos_centroid = pos_matrix.mean(axis=0)
        adjusted    += beta * pos_centroid

    if has_neg:
        neg_matrix   = np.stack(negative_embeddings, axis=0).astype(np.float64)
        neg_centroid = neg_matrix.mean(axis=0)
        adjusted    -= alpha * neg_centroid

    normalised = l2_normalize(adjusted.astype(np.float32))

    logger.debug(
        "Rocchio: k_neg=%d α=%.2f  k_pos=%d β=%.2f  cos(user,query)=%.4f",
        len(negative_embeddings) if has_neg else 0, alpha,
        len(positive_embeddings) if has_pos else 0, beta,
        float(np.dot(l2_normalize(user_vector.astype(np.float32)), normalised)),
    )

    return numpy_to_list(normalised)


def compute_centroid(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    """Mean (centroid) of a set of embeddings — not normalised."""
    if not embeddings:
        raise ValueError("Cannot compute centroid of empty list.")
    return np.stack(embeddings, axis=0).mean(axis=0)


# ---------------------------------------------------------------------------
# Profile drift
# ---------------------------------------------------------------------------

def apply_drift(
    current_vector: np.ndarray,
    played_vector: np.ndarray,
    drift_rate: float = 0.15,
) -> np.ndarray:
    """
    Exponential moving average drift toward a played movie's composite embedding.

        new_vector = normalize( (1-drift_rate)·V_current + drift_rate·V_played )

    With drift_rate=0.15: after 3 plays the profile has shifted ~38% toward
    the new content direction — visible within a demo session.

    Returns normalised float32 array ready to store as BSON Binary subtype 9.
    """
    blended = (1.0 - drift_rate) * current_vector + drift_rate * played_vector
    return l2_normalize(blended).astype(np.float32)
