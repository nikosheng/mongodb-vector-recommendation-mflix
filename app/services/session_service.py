"""
Real-time session tracking — accumulates ALL events since last reset.

Storage: MongoDB collection `user_realtime_sessions`
Schema:
  {
    "user_id":         str,
    "events":          [ { movie_id, embedding (list[float] COMPOSITE), genre, trail, timestamp } ],
    "positive_events": [ { movie_id, embedding (BinData subtype 9 COMPOSITE), genre, trail, timestamp } ],
    "expires_at":      datetime   # document-level TTL 24h — DB hygiene only, NOT a sliding window
  }

Key design decisions:
  - NO time-based pruning of events.  All skips/plays since the last demo reset
    accumulate and influence the Rocchio formula.  The user resets via ↺ Reset Demo.
  - Negative (skip/abandon) embeddings are stored as composite (genre-aware) list[float].
  - Positive (play/complete) embeddings are stored as composite BSON Binary subtype 9.
  - The composite embedding = normalize(λ·genre_vec + (1-λ)·plot_vec) which gives
    Rocchio genre-specific repulsion/attraction rather than plot-blob-based.
  - genre_embedding_cache must be warmed at startup (called from app/main.py lifespan).
"""
from __future__ import annotations

import asyncio
import logging
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import numpy as np
from bson import Binary, ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.config import settings
from app.models import NegativeEvent, PositiveEvent, SessionState
from app.services import genre_embedding_cache
from app.services.profile_notifier import notify_profile_updated
from app.services.vector_math import (
    apply_drift,
    bson_binary_to_numpy,
    list_to_numpy,
    numpy_to_bson_binary,
    numpy_to_list,
    l2_normalize,
)

logger = logging.getLogger(__name__)

SESSIONS_COL = "user_realtime_sessions"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _genre_counts(events: List[dict]) -> Dict[str, int]:
    counts: Counter = Counter()
    for e in events:
        g = e.get("genre")
        if g:
            counts[g] += 1
    return dict(counts)


def _dominant_genre(counts: Dict[str, int]) -> Optional[str]:
    if not counts:
        return None
    return max(counts, key=lambda k: counts[k])


def _swapped_genres(
    genre_counts: Dict[str, int],
    threshold: int = settings.GENRE_SKIP_THRESHOLD,
) -> List[str]:
    return [g for g, c in genre_counts.items() if c >= threshold]


def _decode_positive_embedding(raw) -> List[float]:
    """Decode a positive event composite embedding (Binary subtype 9 or list[float])."""
    if raw is None:
        return []
    if isinstance(raw, (bytes, bytearray, Binary)):
        arr = bson_binary_to_numpy(raw)
        return arr.astype(np.float64).tolist()
    if isinstance(raw, list):
        return [float(x) for x in raw]
    return []


# ---------------------------------------------------------------------------
# Background profile drift
# ---------------------------------------------------------------------------

async def drift_user_profile(
    db: AsyncIOMotorDatabase,
    user_id: str,
    played_composite: np.ndarray,
) -> None:
    """
    Fire-and-forget: nudge user_profiles.user_vector toward the played movie's
    composite embedding.

        new_vector = normalize( (1-DRIFT_RATE)·V_current + DRIFT_RATE·V_played_composite )

    Written back as BSON Binary subtype 9.  play_count incremented for observability.
    """
    try:
        profiles_col = db.get_collection("user_profiles")
        doc = await profiles_col.find_one({"user_id": user_id}, {"user_vector": 1})
        if not doc:
            return

        raw = doc.get("user_vector")
        if raw is None:
            return

        if isinstance(raw, (bytes, bytearray, Binary)):
            current_arr = bson_binary_to_numpy(raw)
        elif isinstance(raw, list):
            current_arr = list_to_numpy(raw)
        else:
            return

        drifted = apply_drift(current_arr, played_composite, settings.DRIFT_RATE)
        await profiles_col.update_one(
            {"user_id": user_id},
            {
                "$set": {"user_vector": numpy_to_bson_binary(drifted), "updated_at": _utcnow()},
                "$inc": {"play_count": 1},
            },
        )
        logger.info("Profile drift: user=%s rate=%.2f", user_id, settings.DRIFT_RATE)
        # Notify SSE stream so the frontend re-fetches trails immediately
        await notify_profile_updated(user_id, reason="drift")
    except Exception as exc:
        logger.warning("drift_user_profile failed for %s: %s", user_id, exc)


# ---------------------------------------------------------------------------
# Async profile rejection on global skip threshold crossing
# ---------------------------------------------------------------------------

async def reject_user_profile(
    db: AsyncIOMotorDatabase,
    user_id: str,
) -> None:
    """
    Fire-and-forget: persistently update user_profiles.user_vector away from
    the direction of all skipped content when the global skip threshold is
    first crossed.

    Formula (same geometry as compute_rfy_query orthogonal rejection):
        neg_dir    = normalize( mean(all_session_neg_embeddings) )
        projection = dot(user_vector, neg_dir)
        new_vector = normalize( user_vector − SKIP_DRIFT_RATE · projection · neg_dir )

    This is MORE conservative than the ephemeral RFY rejection (REJECTION_STRENGTH=0.8)
    because it permanently alters the baseline profile that all future sessions
    start from.  SKIP_DRIFT_RATE=0.5 removes 50% of the skipped direction.

    Self-limiting: as user_vector moves away from neg_dir, dot(user_vector, neg_dir)
    shrinks on subsequent crossings, so each update has diminishing effect.

    Idempotency is handled by the caller (events.py) which fires this task only
    on the exact skip that first crosses NEGATIVE_THRESHOLD, not on later skips.

    Observability: skip_reject_count field incremented on user_profiles document.
    """
    try:
        # 1. Fetch current user_vector
        profiles_col = db.get_collection("user_profiles")
        doc = await profiles_col.find_one({"user_id": user_id}, {"user_vector": 1})
        if not doc:
            logger.warning("reject_user_profile: no profile for %s", user_id)
            return

        raw = doc.get("user_vector")
        if raw is None:
            return

        if isinstance(raw, (bytes, bytearray, Binary)):
            current_arr = bson_binary_to_numpy(raw).astype(np.float64)
        elif isinstance(raw, list) and raw:
            current_arr = np.array(raw, dtype=np.float64)
        else:
            logger.warning("reject_user_profile: unknown user_vector type for %s", user_id)
            return

        # 2. Fetch all negative embeddings currently in session
        neg_arrays = await get_negative_embeddings(db, user_id)
        if not neg_arrays:
            logger.warning("reject_user_profile: no neg embeddings for %s", user_id)
            return

        # 3. Compute rejection direction (centroid of all skipped composites)
        neg_matrix   = np.stack([a.astype(np.float64) for a in neg_arrays], axis=0)
        neg_centroid = neg_matrix.mean(axis=0)
        neg_norm     = np.linalg.norm(neg_centroid)
        if neg_norm < 1e-9:
            return
        neg_dir    = neg_centroid / neg_norm

        # 4. Measure how much the profile currently faces the skipped direction
        projection = float(np.dot(current_arr, neg_dir))

        # 5. Apply rejection — remove SKIP_DRIFT_RATE fraction of that component
        adjusted   = current_arr - settings.SKIP_DRIFT_RATE * projection * neg_dir
        adj_norm   = np.linalg.norm(adjusted)
        if adj_norm < 1e-9:
            return
        new_vec    = (adjusted / adj_norm).astype(np.float32)

        cos_before_after = float(np.dot(current_arr / np.linalg.norm(current_arr), new_vec))

        # 6. Write back as BSON Binary subtype 9
        await profiles_col.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "user_vector":    numpy_to_bson_binary(new_vec),
                    "updated_at":     _utcnow(),
                },
                "$inc": {"skip_reject_count": 1},
            },
        )
        logger.info(
            "Profile skip-rejection: user=%s neg_count=%d rate=%.2f "
            "projection=%.4f cos(old,new)=%.4f",
            user_id, len(neg_arrays), settings.SKIP_DRIFT_RATE,
            projection, cos_before_after,
        )
        # Notify SSE stream so the frontend re-fetches trails immediately
        await notify_profile_updated(user_id, reason="rejection")

    except Exception as exc:
        logger.warning("reject_user_profile failed for %s: %s", user_id, exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def log_event(
    db: AsyncIOMotorDatabase,
    user_id: str,
    movie_id: str,
    event_type: str,
    genre: Optional[str] = None,
    trail: Optional[str] = None,
    effective_genres: Optional[List[str]] = None,
) -> SessionState:
    """
    Record a user interaction.  All events accumulate indefinitely until reset.

    Negative (skip/abandon):
      - Fetches movie's plot_embedding_voyage_4 from embedded_movies.
      - Computes composite_neg = genre_embedding_cache.get_composite_neg_embedding(
            plot_vec, movie.genres, λ)
        which = normalize(λ·genre_vec + (1-λ)·plot_vec).
      - Stores one events[] entry per attributed genre (multi-genre attribution):
          RFY skips  → effective_genres = all movie genres → all genre skip counts increment
          Genre-row skips → effective_genres = [rowGenre] → only that genre's count increments
      - This ensures genre row removal triggers correctly even when skipping from RFY.

    Positive (play/complete):
      - Fetches movie's composite_embedding (if available) or falls back to plot.
      - Stores as BSON Binary subtype 9 in positive_events[].
      - Fires async drift of user_profiles.user_vector.
    """
    movies_col   = db.get_collection(settings.MOVIES_COLLECTION)
    sessions_col = db[SESSIONS_COL]
    now          = _utcnow()

    if event_type in ("skip", "abandon"):
        # ── Composite negative embedding ───────────────────────────────────
        embedding_list: List[float] = []
        try:
            oid = ObjectId(movie_id)
            movie_doc = await movies_col.find_one(
                {"_id": oid},
                {settings.EMBEDDING_FIELD: 1, settings.COMPOSITE_EMBEDDING_FIELD: 1, "genres": 1},
            )
            if movie_doc:
                genres = movie_doc.get("genres") or []

                # Prefer stored composite_embedding if it already exists
                comp_raw = movie_doc.get(settings.COMPOSITE_EMBEDDING_FIELD)
                if comp_raw is not None:
                    arr = bson_binary_to_numpy(comp_raw)
                    embedding_list = arr.astype(np.float64).tolist()
                    logger.debug("skip: using stored composite for %s", movie_id)
                else:
                    # Fall back to computing composite on-the-fly from plot + genre cache
                    plot_raw = movie_doc.get(settings.EMBEDDING_FIELD)
                    if plot_raw is not None:
                        plot_arr = bson_binary_to_numpy(plot_raw)
                        comp_arr = genre_embedding_cache.get_composite_neg_embedding(
                            plot_arr, genres, settings.GENRE_BLEND_LAMBDA
                        )
                        embedding_list = comp_arr.astype(np.float64).tolist()
                        logger.debug(
                            "skip: computed composite on-the-fly for %s (cache=%s)",
                            movie_id, genre_embedding_cache.is_warmed(),
                        )
        except Exception as exc:
            logger.warning("Could not fetch embedding for movie %s: %s", movie_id, exc)

        # Multi-genre skip attribution.
        # For RFY skips (trail is None or not a genre row), attribute all genres so
        # that every genre's skip count increments — this is what drives genre row
        # removal and the per-genre HUD bars.
        # For genre-row skips (trail = "genre:Horror"), attribute only the row genre
        # so the skip count for the specific row being browsed is incremented.
        is_genre_row_skip = bool(trail and trail.startswith("genre:"))
        if is_genre_row_skip:
            # Single genre: the row the user was browsing
            skip_genres = [genre] if genre else []
        else:
            # RFY (or unknown trail): attribute all genres of the movie
            skip_genres = effective_genres if effective_genres else ([genre] if genre else [])

        skip_events = [
            {
                "movie_id":  movie_id,
                "embedding": embedding_list,   # same composite for all genre entries
                "genre":     g,
                "trail":     trail,
                "timestamp": now,
            }
            for g in skip_genres
        ] if skip_genres else [{
            # Fallback: store one entry with no genre if attribution is unknown
            "movie_id":  movie_id,
            "embedding": embedding_list,
            "genre":     None,
            "trail":     trail,
            "timestamp": now,
        }]

        await sessions_col.update_one(
            {"user_id": user_id},
            {
                "$push":        {"events": {"$each": skip_events}},
                "$setOnInsert": {"expires_at": now + timedelta(hours=24)},
            },
            upsert=True,
        )

    elif event_type in ("play", "complete"):
        # ── Composite positive embedding ───────────────────────────────────
        played_composite: Optional[np.ndarray] = None
        embedding_binary = None

        try:
            oid = ObjectId(movie_id)
            movie_doc = await movies_col.find_one(
                {"_id": oid},
                {settings.EMBEDDING_FIELD: 1, settings.COMPOSITE_EMBEDDING_FIELD: 1, "genres": 1},
            )
            if movie_doc:
                genres = movie_doc.get("genres") or []

                comp_raw = movie_doc.get(settings.COMPOSITE_EMBEDDING_FIELD)
                if comp_raw is not None:
                    played_composite = bson_binary_to_numpy(comp_raw)
                    logger.debug("play: using stored composite for %s", movie_id)
                else:
                    plot_raw = movie_doc.get(settings.EMBEDDING_FIELD)
                    if plot_raw is not None:
                        plot_arr = bson_binary_to_numpy(plot_raw)
                        played_composite = genre_embedding_cache.get_composite_neg_embedding(
                            plot_arr, genres, settings.GENRE_BLEND_LAMBDA
                        )
                        logger.debug("play: computed composite on-the-fly for %s", movie_id)

                if played_composite is not None:
                    embedding_binary = numpy_to_bson_binary(played_composite)
        except Exception as exc:
            logger.warning("Could not fetch embedding for played movie %s: %s", movie_id, exc)

        # Multi-genre attribution: store one positive_events entry per genre.
        # For RFY plays, effective_genres = movie.genres (all genres).
        # For genre-row plays, effective_genres = [rowGenre] (single-element).
        # This ensures positive_genre_counts reflects every genre the user engaged with,
        # so "Because you watched Comedy" rows appear even when playing Action+Comedy movies.
        genres_to_store = effective_genres if effective_genres else ([genre] if genre else ["unknown"])
        push_events = [
            {
                "movie_id":  movie_id,
                "embedding": embedding_binary,   # BSON Binary subtype 9 (same embedding for all genre entries)
                "genre":     g,
                "trail":     trail,
                "timestamp": now,
            }
            for g in genres_to_store
        ]
        await sessions_col.update_one(
            {"user_id": user_id},
            {
                "$push":        {"positive_events": {"$each": push_events}},
                "$setOnInsert": {"expires_at": now + timedelta(hours=24)},
            },
            upsert=True,
        )

        # Fire-and-forget profile drift
        if played_composite is not None:
            asyncio.create_task(
                drift_user_profile(db, user_id, played_composite),
                name=f"drift-{user_id}-{movie_id}",
            )

    return await get_session_state(db, user_id)


async def get_session_state(
    db: AsyncIOMotorDatabase,
    user_id: str,
) -> SessionState:
    """
    Read ALL accumulated events (no time pruning) and return full SessionState.
    Events only clear when the user calls POST /api/demo/reset.
    """
    sessions_col = db[SESSIONS_COL]
    doc = await sessions_col.find_one({"user_id": user_id})

    if not doc:
        return SessionState(
            user_id=user_id,
            negative_count=0,
            threshold=settings.NEGATIVE_THRESHOLD,
            mitigation_active=False,
        )

    # Use ALL stored events — no time-based filtering
    all_neg: List[dict] = doc.get("events", [])
    all_pos: List[dict] = doc.get("positive_events", [])

    # Negative analysis
    # negative_count = distinct movies skipped (not raw entry count).
    # With multi-genre attribution each skip stores N entries (one per genre),
    # so len(all_neg) would overcount. Deduplicating by movie_id keeps the global
    # threshold ("3 movies skipped") on the intended "distinct movies" basis.
    unique_skipped_ids = {e["movie_id"] for e in all_neg}
    count             = len(unique_skipped_ids)
    genre_counts      = _genre_counts(all_neg)
    swapped           = _swapped_genres(genre_counts, settings.GENRE_SKIP_THRESHOLD)
    mitigation_active = count >= settings.NEGATIVE_THRESHOLD
    penalized_genre   = _dominant_genre(genre_counts) if mitigation_active else None

    rocchio_genre: Optional[str] = None
    for e in all_neg:
        g = e.get("genre")
        if g and genre_counts.get(g, 0) >= settings.GENRE_SKIP_THRESHOLD:
            rocchio_genre = g
            break

    negative_events = [
        NegativeEvent(
            movie_id=e["movie_id"],
            embedding=e.get("embedding", []),
            genre=e.get("genre"),
            trail=e.get("trail"),
            timestamp=e["timestamp"],
        )
        for e in all_neg
    ]

    # Positive analysis
    pos_genre_counts = _genre_counts(all_pos)
    positive_events  = [
        PositiveEvent(
            movie_id=e["movie_id"],
            embedding=_decode_positive_embedding(e.get("embedding")),
            genre=e.get("genre"),
            trail=e.get("trail"),
            timestamp=e["timestamp"],
        )
        for e in all_pos
    ]
    # positive_count = distinct movies played (not raw entry count).
    # With multi-genre attribution each play stores N entries (one per genre),
    # so len(all_pos) would overcount by ~genres-per-movie. Deduplicating by
    # movie_id keeps β scaling on the intended "how many movies watched" curve.
    unique_played_ids = {e["movie_id"] for e in all_pos}
    positive_count = len(unique_played_ids)

    return SessionState(
        user_id=user_id,
        negative_count=count,
        threshold=settings.NEGATIVE_THRESHOLD,
        mitigation_active=mitigation_active,
        penalized_genre=penalized_genre,
        negative_events=negative_events,
        genre_skip_counts=genre_counts,
        swapped_genres=swapped,
        rocchio_genre=rocchio_genre,
        positive_count=positive_count,
        positive_events=positive_events,
        positive_genre_counts=pos_genre_counts,
    )


async def get_negative_embeddings(
    db: AsyncIOMotorDatabase,
    user_id: str,
    genre_filter: Optional[str] = None,
) -> List[np.ndarray]:
    """Composite float32 arrays for all negative events (no time filter).

    When genre_filter is None (RFY rejection centroid):
      - Deduplicate by movie_id so a multi-genre skip (which stores N entries)
        contributes its embedding only once to the rejection centroid.
        Without dedup, movies with many genres would dominate the centroid.

    When genre_filter is set:
      - Only events matching that genre are included (no dedup needed —
        genre_filter is not currently used externally but kept for future use).
    """
    state = await get_session_state(db, user_id)
    arrays: List[np.ndarray] = []
    seen_for_rfy: set = set()
    for event in state.negative_events:
        if genre_filter:
            if event.genre != genre_filter:
                continue
        else:
            # RFY arm: deduplicate by movie_id across all genre entries
            if event.movie_id in seen_for_rfy:
                continue
            seen_for_rfy.add(event.movie_id)
        if event.embedding:
            arrays.append(list_to_numpy(event.embedding))
    return arrays


async def get_positive_embeddings(
    db: AsyncIOMotorDatabase,
    user_id: str,
    genre_filter: Optional[str] = None,
) -> List[np.ndarray]:
    """Composite float32 arrays for all positive events (no time filter).

    When genre_filter is set (e.g. for "Because you watched" row centroid):
      - Only events matching that genre are included.
      - The same movie can appear multiple times if it was played from the same
        genre row more than once — that is intentional (repeated plays strengthen
        the centroid toward that movie).

    When genre_filter is None (RFY positive arm centroid):
      - Deduplicate by movie_id so that a single play of [Action, Comedy] (which
        stores two entries) does not contribute its embedding twice to the centroid,
        which would skew the attraction vector toward multi-genre movies.
    """
    state = await get_session_state(db, user_id)
    arrays: List[np.ndarray] = []
    seen_for_rfy: set = set()  # used only when genre_filter is None
    for event in state.positive_events:
        if genre_filter:
            if event.genre != genre_filter:
                continue
        else:
            # RFY arm: deduplicate by movie_id across all genre entries
            if event.movie_id in seen_for_rfy:
                continue
            seen_for_rfy.add(event.movie_id)
        if event.embedding:
            arrays.append(list_to_numpy(event.embedding))
    return arrays


async def get_excluded_ids(
    db: AsyncIOMotorDatabase,
    user_id: str,
) -> List[ObjectId]:
    """ObjectIds of all skipped movies — excluded from $vectorSearch results.

    Deduplicates by movie_id: with multi-genre skip attribution each skip stores
    N entries (one per genre), so iterating negative_events without dedup adds
    the same ObjectId N times, bloating the Atlas $nin filter unnecessarily.
    """
    state = await get_session_state(db, user_id)
    ids: List[ObjectId] = []
    seen: set = set()
    for event in state.negative_events:
        if event.movie_id in seen:
            continue
        seen.add(event.movie_id)
        try:
            ids.append(ObjectId(event.movie_id))
        except Exception:
            pass
    return ids


async def get_played_ids(
    db: AsyncIOMotorDatabase,
    user_id: str,
) -> List[ObjectId]:
    """ObjectIds of all played movies — excluded so they don't flood re-ranked results.

    Deduplicates by movie_id: with multi-genre attribution each play stores N entries
    (one per genre), so iterating positive_events without dedup would add the same
    ObjectId N times, bloating the exclude list unnecessarily.
    """
    state = await get_session_state(db, user_id)
    ids: List[ObjectId] = []
    seen: set = set()
    for event in state.positive_events:
        if event.movie_id in seen:
            continue
        seen.add(event.movie_id)
        try:
            ids.append(ObjectId(event.movie_id))
        except Exception:
            pass
    return ids


async def reset_session(db: AsyncIOMotorDatabase, user_id: str) -> None:
    """Delete the session document — the only way to clear accumulated events."""
    sessions_col = db[SESSIONS_COL]
    await sessions_col.delete_one({"user_id": user_id})
    logger.info("Session reset for user: %s", user_id)


async def restore_user_vector(
    db: AsyncIOMotorDatabase,
    user_id: str,
) -> bool:
    """
    Re-embed the stored profile_text and genres from user_profiles to produce a
    fresh composite user_vector in the correct embedding space, then write it back.

    Called by POST /api/demo/reset so the demo always restarts from a correctly-
    aligned baseline vector.  Without this, repeated drift/rejection operations
    accumulate and the user_vector drifts out of the composite embedding space,
    making rejection mathematically ineffective (low dot product with skipped content).

    Returns True on success, False if the profile or Voyage API is unavailable.
    """
    try:
        profiles_col = db.get_collection("user_profiles")
        doc = await profiles_col.find_one({"user_id": user_id})
        if not doc:
            logger.warning("restore_user_vector: no profile for %s", user_id)
            return False

        profile_text: str = doc.get("profile_text", "")
        genres: List[str] = doc.get("genres", [])
        lam: float        = float(doc.get("lambda", settings.GENRE_BLEND_LAMBDA))

        if not profile_text and not genres:
            logger.warning("restore_user_vector: no profile_text/genres for %s", user_id)
            return False

        genres_str = " ".join(sorted(genres)) if genres else profile_text[:100]

        # Embed both genre label string and profile text via Voyage AI
        embs = await genre_embedding_cache.embed_texts_via_voyage(
            [genres_str, profile_text]
        )
        if len(embs) < 2:
            logger.warning("restore_user_vector: embedding returned < 2 vectors")
            return False

        genre_vec   = np.array(embs[0], dtype=np.float32)
        profile_vec = np.array(embs[1], dtype=np.float32)
        genre_vec   = l2_normalize(genre_vec)
        profile_vec = l2_normalize(profile_vec)

        composite = lam * genre_vec + (1.0 - lam) * profile_vec
        composite = l2_normalize(composite).astype(np.float32)

        await profiles_col.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "user_vector":       numpy_to_bson_binary(composite),
                    "updated_at":        _utcnow(),
                    "skip_reject_count": 0,
                    "play_count":        0,
                },
            },
        )
        logger.info(
            "restore_user_vector: restored composite user_vector for %s "
            "(genres=%s λ=%.2f)",
            user_id, genres, lam,
        )
        return True

    except Exception as exc:
        logger.warning("restore_user_vector failed for %s: %s", user_id, exc)
        return False


async def get_user_profile(
    db: AsyncIOMotorDatabase,
    user_id: str,
) -> Optional[dict]:
    """
    Fetch user_profiles document.
    Handles both list[float] (legacy) and Binary subtype 9 (composite).
    """
    profiles_col = db.get_collection("user_profiles")
    doc = await profiles_col.find_one({"user_id": user_id})
    if not doc:
        return None

    raw = doc.get("user_vector")
    if raw is None:
        return None

    if isinstance(raw, (bytes, bytearray, Binary)):
        arr = bson_binary_to_numpy(raw)
        vec = arr.astype(np.float64).tolist()
    elif isinstance(raw, list) and len(raw) > 0:
        vec = [float(x) for x in raw]
    else:
        return None

    return {
        "user_vector":  vec,
        "genres":       doc.get("genres", []),
        "profile_text": doc.get("profile_text", ""),
        "play_count":   doc.get("play_count", 0),
    }
