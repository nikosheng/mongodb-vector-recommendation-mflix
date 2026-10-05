"""
Atlas Vector Search execution service — composite-embedding aware.

All $vectorSearch queries use the composite_embedding field and
composite_vector_index so that the query vector and stored vectors
are in the same genre-aware composite space.

Fallback: if composite index is unavailable (e.g. not yet created),
queries silently fall back to plot_embedding_voyage_4 via the original index.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.config import settings
from app.models import MovieOut
from app.services.vector_math import l2_normalize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def _build_pipeline(
    query_vector: List[float],
    limit: int,
    num_candidates: int,
    exclude_ids: Optional[List[ObjectId]] = None,
    genre_filter: Optional[str] = None,
    genre_exclusion: Optional[List[str]] = None,
    genre_inclusion: Optional[List[str]] = None,
    use_composite: bool = True,
) -> list:
    """
    Build an Atlas $vectorSearch aggregation pipeline.

    Uses composite_embedding + composite_vector_index when use_composite=True
    (default), falling back to plot_embedding_voyage_4 + vector_index_voyage4.

    genre_exclusion: $nin filter — used by get_discovery_row() dynamic mode to
        exclude the user's current active genre rows.

    genre_inclusion: $in filter — used by get_discovery_row() fixed mode to
        restrict results to a specific set of genres configured in .env.
        Both genre_inclusion and genre_exclusion can coexist in one pipeline
        (e.g. include Action|Horror AND exclude Comedy|Romance).
    """
    emb_field   = settings.COMPOSITE_EMBEDDING_FIELD if use_composite else settings.EMBEDDING_FIELD
    index_name  = settings.COMPOSITE_VECTOR_INDEX    if use_composite else settings.VECTOR_INDEX_NAME

    vs: dict = {
        "index":         index_name,
        "path":          emb_field,
        "queryVector":   query_vector,
        "numCandidates": num_candidates,
        "limit":         limit,
    }

    filters = []
    if exclude_ids:
        filters.append({"_id": {"$nin": exclude_ids}})
    if genre_filter:
        filters.append({"genres": {"$eq": genre_filter}})
    if genre_exclusion:
        filters.append({"genres": {"$nin": genre_exclusion}})
    if genre_inclusion:
        # $in on an array field: matches docs where genres array contains
        # at least one element from genre_inclusion list
        filters.append({"genres": {"$in": genre_inclusion}})

    if len(filters) == 1:
        vs["filter"] = filters[0]
    elif len(filters) > 1:
        vs["filter"] = {"$and": filters}

    return [
        {"$vectorSearch": vs},
        {
            "$project": {
                "_id": 1, "title": 1, "genres": 1,
                "year": 1, "plot": 1, "poster": 1,
                "vectorSearchScore": {"$meta": "vectorSearchScore"},
            }
        },
    ]


def _apply_genre_penalty(
    movies: List[MovieOut],
    genre_skip_counts: Dict[str, int],
    positive_genre_counts: Dict[str, int],
    penalty_strength: float,
    max_skips: int,
    limit: int,
) -> List[MovieOut]:
    """
    Soft, reversible genre penalty applied after $vectorSearch.

    For each movie, computes net_affinity per genre:
        net_affinity(genre) = play_count(genre) - skip_count(genre)

    If any genre on the movie has a negative net affinity (user skipped more
    than played), the movie's vector_score is multiplied by a penalty factor:

        severity     = clamp(|net_affinity| / max_skips, 0.0, 1.0)
        penalty_factor = 1.0 - penalty_strength × severity

    The worst (most negative) genre drives the penalty — an Action+Horror movie
    is penalised by whichever genre is more disliked.

    Penalty is fully reversible:
        - Skip 3 Action, play 0  → net = -3 → factor = 1 - 0.4×0.6 = 0.76
        - Skip 3 Action, play 2  → net = -1 → factor = 1 - 0.4×0.2 = 0.92
        - Skip 3 Action, play 3  → net =  0 → factor = 1.0  (fully restored)
        - Skip 3 Action, play 5  → net = +2 → no penalty  (Rocchio handles boost)

    Movies are re-sorted by adjusted score and the top `limit` are returned.
    The larger fetch pool (RFY_FETCH_MULTIPLIER × limit) ensures non-penalised
    movies can float up to fill the visible slots.
    """
    if not genre_skip_counts:
        # No skips at all — nothing to penalise, just truncate
        return movies[:limit]

    for movie in movies:
        worst_net = 0  # most negative net affinity across this movie's genres
        for genre in movie.genres:
            plays = positive_genre_counts.get(genre, 0)
            skips = genre_skip_counts.get(genre, 0)
            net   = plays - skips
            if net < worst_net:
                worst_net = net

        if worst_net < 0:
            severity           = min(abs(worst_net) / max_skips, 1.0)
            penalty_factor     = 1.0 - penalty_strength * severity
            movie.vector_score = round(movie.vector_score * penalty_factor, 4)

    movies.sort(key=lambda m: m.vector_score, reverse=True)
    return movies[:limit]


def _parse_docs(docs: list) -> List[MovieOut]:
    results: List[MovieOut] = []
    for doc in docs:
        try:
            results.append(MovieOut(
                id=str(doc["_id"]),
                title=doc.get("title", "Unknown"),
                genres=doc.get("genres") or [],
                year=doc.get("year"),
                plot=doc.get("plot"),
                poster=doc.get("poster"),
                vector_score=round(float(doc.get("vectorSearchScore", 0.0)), 4),
            ))
        except Exception as exc:
            logger.warning("Skipping malformed doc %s: %s", doc.get("_id"), exc)
    return results


async def _run_pipeline(
    db: AsyncIOMotorDatabase,
    pipeline: list,
    limit: int,
    exclude_ids: Optional[List[ObjectId]] = None,
    use_composite: bool = True,
) -> List[MovieOut]:
    """Execute pipeline, fallback to plot-only if composite index doesn't exist yet."""
    movies_col = db.get_collection(settings.MOVIES_COLLECTION)
    try:
        cursor = movies_col.aggregate(pipeline)
        docs   = await cursor.to_list(length=limit)
        results = _parse_docs(docs)
        # If composite index returned 0 results, it likely doesn't exist yet.
        # Transparently retry with the plot-only index so the demo still works.
        if not results and use_composite:
            logger.warning(
                "Composite index returned 0 results — index may not exist yet. "
                "Falling back to plot-only index. "
                "Create composite_vector_index in Atlas UI (run 'make composite' first)."
            )
            vs_stage = pipeline[0].get("$vectorSearch", {})
            vs_stage["index"] = settings.VECTOR_INDEX_NAME
            vs_stage["path"]  = settings.EMBEDDING_FIELD
            cursor2 = movies_col.aggregate(pipeline)
            docs2   = await cursor2.to_list(length=limit)
            return _parse_docs(docs2)
        return results
    except Exception as exc:
        err_str = str(exc).lower()
        # If composite index doesn't exist yet, fall back transparently
        if use_composite and any(kw in err_str for kw in
                                  ("index", "unknown", "doesn't exist", "not found",
                                   "no such", "invalid", "namespace")):
            logger.warning(
                "Composite vector index not ready — falling back to plot-only index. "
                "Create composite_vector_index in Atlas UI (see 'make composite' output). "
                "Error: %s", exc
            )
            # Rebuild pipeline with plot-only index
            vs_stage = pipeline[0].get("$vectorSearch", {})
            vs_stage["index"] = settings.VECTOR_INDEX_NAME
            vs_stage["path"]  = settings.EMBEDDING_FIELD
            cursor = movies_col.aggregate(pipeline)
            docs   = await cursor.to_list(length=limit)
            return _parse_docs(docs)
        # Catch-all fallback for any vectorSearch failure with composite index
        if use_composite:
            logger.warning(
                "Unexpected composite index error — falling back to plot-only. Error: %s", exc
            )
            vs_stage = pipeline[0].get("$vectorSearch", {})
            vs_stage["index"] = settings.VECTOR_INDEX_NAME
            vs_stage["path"]  = settings.EMBEDDING_FIELD
            cursor = movies_col.aggregate(pipeline)
            try:
                docs = await cursor.to_list(length=limit)
                return _parse_docs(docs)
            except Exception as exc2:
                logger.error("Plot-only fallback also failed: %s", exc2)
                return []
        # Pre-filter failure — retry without filter
        if exclude_ids and "filter" in pipeline[0].get("$vectorSearch", {}):
            logger.warning("Pre-filter failed, retrying without $nin. Error: %s", exc)
            vs_stage = pipeline[0]["$vectorSearch"]
            del vs_stage["filter"]
            vs_stage["numCandidates"] += len(exclude_ids) * 2
            vs_stage["limit"]         += len(exclude_ids)
            cursor = movies_col.aggregate(pipeline)
            docs   = await cursor.to_list(vs_stage["limit"])
            excl   = {str(oid) for oid in exclude_ids}
            docs   = [d for d in docs if str(d["_id"]) not in excl][:limit]
            return _parse_docs(docs)
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def get_recommendations(
    db: AsyncIOMotorDatabase,
    query_vector: List[float],
    exclude_ids: Optional[List[ObjectId]] = None,
    limit: int = settings.RECOMMENDATION_LIMIT,
    genre_skip_counts: Optional[Dict[str, int]] = None,
    positive_genre_counts: Optional[Dict[str, int]] = None,
) -> List[MovieOut]:
    """
    Recommended For You row — composite $vectorSearch with soft genre penalty.

    query_vector must be in composite space (Rocchio output or user_vector).

    Soft genre penalty (replaces the old hard $nin filter):
      After $vectorSearch, each movie's score is multiplied by a penalty factor
      derived from net_affinity(genre) = play_count - skip_count.  Negative net
      means the user skipped more than played → score reduced.  Positive or zero
      net means no penalty.  The penalty lifts automatically as the user plays
      movies in previously-skipped genres — no hard ban, fully reversible.

      A larger candidate pool (limit × RFY_FETCH_MULTIPLIER) is fetched so
      non-penalised movies can float up to fill the visible slots after reranking.
    """
    fetch_limit = limit * settings.RFY_FETCH_MULTIPLIER
    pipeline = _build_pipeline(
        query_vector=query_vector,
        limit=fetch_limit,
        num_candidates=settings.NUM_CANDIDATES,
        exclude_ids=exclude_ids,
        use_composite=True,
    )
    candidates = await _run_pipeline(db, pipeline, fetch_limit, exclude_ids, use_composite=True)

    # Apply soft genre penalty and rerank
    results = _apply_genre_penalty(
        movies=candidates,
        genre_skip_counts=genre_skip_counts or {},
        positive_genre_counts=positive_genre_counts or {},
        penalty_strength=settings.GENRE_PENALTY_STRENGTH,
        max_skips=settings.GENRE_PENALTY_MAX_SKIPS,
        limit=limit,
    )

    logger.info(
        "RFY: %d candidates → %d results after penalty (excluded=%d skip_genres=%s)",
        len(candidates), len(results), len(exclude_ids or []),
        {g: c for g, c in (genre_skip_counts or {}).items() if c > 0},
    )
    return results


async def get_genre_row(
    db: AsyncIOMotorDatabase,
    query_vector: List[float],
    genre: str,
    exclude_ids: Optional[List[ObjectId]] = None,
    limit: int = settings.GENRE_ROW_LIMIT,
) -> List[MovieOut]:
    """
    Genre row — composite $vectorSearch + genres pre-filter.
    Uses the Rocchio-adjusted query_vector (same as RFY row).
    """
    pipeline = _build_pipeline(
        query_vector=query_vector,
        limit=limit,
        num_candidates=settings.GENRE_NUM_CANDIDATES,
        exclude_ids=exclude_ids,
        genre_filter=genre,
        use_composite=True,
    )
    results = await _run_pipeline(db, pipeline, limit, exclude_ids, use_composite=True)
    logger.info("Genre row [%s]: %d results", genre, len(results))
    return results


async def get_played_similar_row(
    db: AsyncIOMotorDatabase,
    played_embeddings: List[np.ndarray],
    genre: str,
    exclude_ids: Optional[List[ObjectId]] = None,
    limit: int = settings.GENRE_ROW_LIMIT,
) -> List[MovieOut]:
    """
    'Because You Watched' row — uses the CENTROID of played movie composite
    embeddings as the query vector, NOT the Rocchio-adjusted user vector.

    This finds movies semantically similar to what the user specifically watched
    (not what their overall profile says they like).  Genre pre-filter applied.

    Example:
      User plays 2 fast-paced Action movies → centroid points toward that
      specific semantic region → row surfaces similar fast-paced Action films,
      not just anything matching the generic Action profile.

    Args:
        played_embeddings: Composite float32 arrays for played movies of this genre.
        genre:             Genre to filter ($eq pre-filter in Atlas index).
        exclude_ids:       Movies already seen — excluded from results.
        limit:             Max results.
    """
    if not played_embeddings:
        logger.warning("get_played_similar_row called with empty played_embeddings for %s", genre)
        return []

    # Centroid of played movie composite embeddings → normalised query vector
    centroid = np.stack(played_embeddings, axis=0).mean(axis=0)
    centroid = l2_normalize(centroid.astype(np.float32))
    query_vector = centroid.astype(np.float64).tolist()

    pipeline = _build_pipeline(
        query_vector=query_vector,
        limit=limit,
        num_candidates=settings.GENRE_NUM_CANDIDATES,
        exclude_ids=exclude_ids,
        genre_filter=genre,
        use_composite=True,
    )
    results = await _run_pipeline(db, pipeline, limit, exclude_ids, use_composite=True)
    logger.info(
        "Played-similar row [%s]: centroid from %d plays → %d results",
        genre, len(played_embeddings), len(results),
    )
    return results


async def get_discovery_row(
    db: AsyncIOMotorDatabase,
    query_vector: List[float],
    exclude_genres: Optional[List[str]] = None,
    include_genres: Optional[List[str]] = None,
    seed_genres: Optional[List[str]] = None,
    exclude_ids: Optional[List[ObjectId]] = None,
    limit: int = settings.GENRE_ROW_LIMIT,
) -> List[MovieOut]:
    """
    Discovery row — shows content outside the user's current taste profile.

    Two operating modes, selected by the caller:

    Fixed inclusion mode  (include_genres is set — configured via DISCOVERY_GENRE_MAP):
        Shows movies that:
          - contain ≥1 genre from include_genres  ($in filter)
          - do NOT contain any genre from seed_genres ($nin filter)
        This ensures the row shows content genuinely outside the user's bubble —
        e.g. a Comedy Fan's Discovery row shows Action/Horror but never Comedy/Romance.
        Falls back to include_genres-only (without seed exclusion) if no results.

    Dynamic exclusion mode  (include_genres is None — default behaviour):
        Shows movies that do NOT contain any genre from exclude_genres ($nin filter).
        exclude_genres is typically the user's current active genre rows.
        Falls back to unrestricted results if exclusion is too broad.
    """
    if include_genres:
        # ── Fixed inclusion mode ─────────────────────────────────────────
        pipeline = _build_pipeline(
            query_vector=query_vector,
            limit=limit,
            num_candidates=settings.NUM_CANDIDATES,
            exclude_ids=exclude_ids,
            genre_inclusion=include_genres,
            genre_exclusion=seed_genres or None,   # exclude user's seed genres
            use_composite=True,
        )
        results = await _run_pipeline(db, pipeline, limit, exclude_ids, use_composite=True)
        if not results and seed_genres:
            # Fallback 1: drop seed exclusion, keep inclusion filter
            pipeline_fb1 = _build_pipeline(
                query_vector=query_vector,
                limit=limit,
                num_candidates=settings.NUM_CANDIDATES,
                exclude_ids=exclude_ids,
                genre_inclusion=include_genres,
                use_composite=True,
            )
            results = await _run_pipeline(db, pipeline_fb1, limit, exclude_ids, use_composite=True)
        if not results:
            # Fallback 2: no filters at all
            pipeline_fb2 = _build_pipeline(
                query_vector=query_vector,
                limit=limit,
                num_candidates=settings.NUM_CANDIDATES,
                exclude_ids=exclude_ids,
                use_composite=True,
            )
            results = await _run_pipeline(db, pipeline_fb2, limit, exclude_ids, use_composite=True)
        logger.info(
            "Discovery row (fixed): %d results (include=%s exclude_seed=%s)",
            len(results), include_genres, seed_genres or [],
        )
    else:
        # ── Dynamic exclusion mode ───────────────────────────────────────
        pipeline = _build_pipeline(
            query_vector=query_vector,
            limit=limit,
            num_candidates=settings.NUM_CANDIDATES,
            exclude_ids=exclude_ids,
            genre_exclusion=exclude_genres or None,
            use_composite=True,
        )
        results = await _run_pipeline(db, pipeline, limit, exclude_ids, use_composite=True)
        if not results and exclude_genres:
            # Fallback — exclusion too broad, retry without genre filter
            pipeline_fb = _build_pipeline(
                query_vector=query_vector,
                limit=limit,
                num_candidates=settings.NUM_CANDIDATES,
                exclude_ids=exclude_ids,
                use_composite=True,
            )
            results = await _run_pipeline(db, pipeline_fb, limit, exclude_ids, use_composite=True)
        logger.info(
            "Discovery row (dynamic): %d results (excluded_genres=%s)",
            len(results), exclude_genres or [],
        )

    return results


async def get_user_vector(
    db: AsyncIOMotorDatabase,
    user_id: str,
) -> Optional[List[float]]:
    """Fetch user's composite baseline vector from user_profiles."""
    profiles_col = db.get_collection("user_profiles")
    doc = await profiles_col.find_one({"user_id": user_id}, {"user_vector": 1})
    if doc and "user_vector" in doc:
        vec = doc["user_vector"]
        if isinstance(vec, list) and len(vec) > 0:
            return [float(x) for x in vec]
    return None
