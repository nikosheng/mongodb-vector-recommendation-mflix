"""
GET /api/trails/{user_id}

Clean separation of responsibilities:

  RFY row  ← driven by user_vector + global skip signals (orthogonal rejection)
  Genre rows ← use rfy_query when mitigation_active, else plain user_vector
               per-genre skip controls VISIBILITY (row disappears) AND query direction

Design:

  Global skip (≥ NEGATIVE_THRESHOLD = 3 skips of any genre):
    → RFY query = orthogonal_rejection(user_vector, all_skipped_embeddings)
    → Removes the skipped content's direction from the profile
    → RFY surfaces noticeably different content
    → Profile/swapped genre rows ALSO switch to rfy_query so their content shifts
      (genre pre-filter is still applied — an Action row still only shows Action)
    → "Because you watched" played rows keep their played-movie centroid unchanged

  Per-genre skip (≥ GENRE_SKIP_THRESHOLD = 2 skips of same genre):
    → That genre's row DISAPPEARS, replaced by next profile genre
    → Skipped genres are SOFT-PENALISED in RFY via score multiplier (not hard-banned):
        penalty_factor = 1 - GENRE_PENALTY_STRENGTH × clamp(net_skips/MAX, 0, 1)
        net_skips = skip_count - play_count
        → Penalty lifts automatically as user plays movies in those genres
    → HUD shows per-genre bar filling up toward threshold

  Play events (≥ PLAY_ROW_THRESHOLD = 2 plays of same genre):
    → RFY query += β_eff · mean(played_embeddings)   [positive attraction]
    → "Because you watched X" row appears using played-movie centroid
    → Positive arm applies to RFY only, not genre rows

  Combined (global skips + plays):
    → RFY query = rejection(user_vector, all_neg) + β·mean(V_played)
    → query_source = "rfy_full"

  query_source values:
    "user_vector"     — no signals
    "rfy_rejected"    — global skips only (orthogonal rejection applied)
    "rfy_positive"    — plays only (attraction applied)
    "rfy_full"        — both rejection + attraction
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException, Path, Query, status

from app.config import settings
from app.database import get_db
from app.models import GenreRow, SessionState, TrailsResponse
from app.services import recommendation_service, session_service
from app.services.boost_service import apply_boosts_to_row, get_active_boost_map
from app.services.vector_math import (
    compute_rfy_query,
    compute_rocchio_query,
    list_to_numpy,
)
from app.services.vector_projection import project_onto_genre_axes, vector_shift

logger = logging.getLogger(__name__)
router = APIRouter(tags=["trails"])


# ---------------------------------------------------------------------------
# Scaled β for positive arm
# ---------------------------------------------------------------------------

def _effective_beta(positive_count: int) -> float:
    return settings.BETA * min(positive_count, settings.BETA_MAX_MULTIPLIER)


# ---------------------------------------------------------------------------
# RFY query vector — orthogonal rejection + positive attraction
# ---------------------------------------------------------------------------

def _decide_rfy_query(
    user_vector_list: List[float],
    all_neg: List,          # all composite negative embeddings
    all_pos: List,          # all composite positive embeddings
    beta_effective: float,
) -> Tuple[List[float], bool, str]:
    """
    Compute the RFY query vector.

    Global skip signals  → orthogonal rejection removes skipped direction from profile.
    Play signals         → positive attraction pulls toward played content.
    Per-genre skips      → do NOT affect this vector (they only control row visibility).

    Returns (query_vector, rocchio_applied, query_source).
    """
    user_arr  = list_to_numpy(user_vector_list)
    has_neg   = bool(all_neg)
    has_pos   = bool(all_pos)

    if has_neg and has_pos:
        # Both: rejection first, then add positive attraction
        try:
            qv = compute_rfy_query(
                user_arr, all_neg,
                rejection_strength=settings.REJECTION_STRENGTH,
                positive_embeddings=all_pos,
                beta=beta_effective,
            )
            return qv, True, "rfy_full"
        except Exception as exc:
            logger.warning("RFY full (rejection+positive) failed: %s", exc)

    if has_neg:
        # Global skips only — orthogonal rejection
        try:
            qv = compute_rfy_query(
                user_arr, all_neg,
                rejection_strength=settings.REJECTION_STRENGTH,
            )
            return qv, True, "rfy_rejected"
        except Exception as exc:
            logger.warning("RFY rejection failed: %s", exc)

    if has_pos:
        # Plays only — positive-only Rocchio (unchanged)
        try:
            qv = compute_rocchio_query(
                user_arr, [], settings.ALPHA,
                positive_embeddings=all_pos,
                beta=beta_effective,
            )
            return qv, True, "rfy_positive"
        except Exception as exc:
            logger.warning("RFY positive-only Rocchio failed: %s", exc)

    # No signals — baseline profile
    return user_vector_list, False, "user_vector"


# ---------------------------------------------------------------------------
# Genre row slot computation — weighted by seed rank + play counts
# ---------------------------------------------------------------------------

def _compute_weighted_genre_slots(
    profile_genres: List[str],
    swapped_genres: List[str],
    count: int,
    positive_genre_counts: Dict[str, int],
    play_genre_weight: float,
    emerging_rows_count: int = 2,
    play_row_threshold: int = 2,
) -> List[Tuple[str, Optional[str], str]]:
    """
    Build genre row slots as (display_genre, swapped_from, row_source).

    Returns two groups of slots:

    ① Main slots (up to `count`):
        Genres compete by combined weight:
            weight = seed_rank_weight + play_count × play_genre_weight
        Non-seed genres with enough plays displace lower-ranked seed genres,
        demonstrating live preference evolution.

    ② Emerging slots (up to `emerging_rows_count`, appended after main slots):
        Non-seed genres that have ≥1 play but < play_row_threshold plays AND
        did NOT win a main slot.  These appear as "You might like [genre]" rows
        using the user_vector query (not the centroid), signalling growing interest
        before the user has watched enough to warrant a centroid-based row.

    row_source values:
        "played"   — non-seed, ≥ play_row_threshold plays → centroid query
        "profile"  — seed genre that won its main slot
        "emerging" — non-seed, 1 ≤ plays < play_row_threshold → user_vector query
        "swapped"  — a seed genre was skip-swapped (rare with weighted logic)
    """
    swapped_set = set(swapped_genres)
    seed_set    = set(profile_genres)
    n_seed      = len(profile_genres)

    # seed_rank_weight: higher weight for earlier seed genres
    seed_rank_weight: Dict[str, float] = {
        g: float(n_seed - rank) for rank, g in enumerate(profile_genres)
    }

    # Build scored list for the main slot competition
    scored: List[Tuple[float, int, str]] = []
    for order, g in enumerate(
        list(profile_genres) + [g for g in positive_genre_counts if g not in seed_set]
    ):
        if g in swapped_set:
            continue
        w = seed_rank_weight.get(g, 0.0) + positive_genre_counts.get(g, 0) * play_genre_weight
        if w > 0:
            scored.append((w, order, g))

    # Sort descending by weight; ties broken by insertion order (seed genres first)
    scored.sort(key=lambda x: (-x[0], x[1]))

    # ── ① Main slots ────────────────────────────────────────────────────────
    main_slots: List[Tuple[str, Optional[str], str]] = []
    won_genres: set = set()
    for _, _, genre in scored[:count]:
        plays  = positive_genre_counts.get(genre, 0)
        if genre not in seed_set:
            source = "played" if plays >= play_row_threshold else "emerging"
        else:
            source = "profile"
        main_slots.append((genre, None, source))
        won_genres.add(genre)

    # ── ② Bonus slots for non-seed genres that didn't win a main slot ──────────
    #
    # Two sub-groups, both capped together at emerging_rows_count total:
    #
    # a) "played" bonus — non-seed genres with ≥ play_row_threshold plays that
    #    didn't displace a seed slot (weight was too low to win main competition).
    #    These get full "Because you watched" centroid-query rows.
    #
    # b) "emerging" bonus — non-seed genres with 1 ≤ plays < play_row_threshold.
    #    These get "You might like" user_vector-query rows.
    #
    # Sorted within each group by play count descending.
    # "played" bonus rows come before "emerging" bonus rows.
    bonus_candidates = sorted(
        [
            (plays, g) for g, plays in positive_genre_counts.items()
            if g not in seed_set
            and g not in won_genres
            and g not in swapped_set
            and plays > 0
        ],
        reverse=True,   # most-played first
    )

    bonus_slots: List[Tuple[str, Optional[str], str]] = []
    for plays, genre in bonus_candidates:
        if len(bonus_slots) >= emerging_rows_count:
            break
        source = "played" if plays >= play_row_threshold else "emerging"
        bonus_slots.append((genre, None, source))

    return main_slots + bonus_slots


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------

@router.get(
    "/trails/{user_id}",
    response_model=TrailsResponse,
    summary="Multi-row personalised trails — global skip shapes RFY, per-genre skip controls row visibility",
)
async def get_trails(
    user_id: str = Path(..., description="User identifier"),
    trail_limit: int = Query(default=settings.TRAIL_LIMIT, ge=1, le=50),
    genre_rows_count: int = Query(default=settings.GENRE_ROWS_COUNT, ge=0, le=8),
    genre_row_limit: int = Query(default=settings.GENRE_ROW_LIMIT, ge=1, le=24),
) -> TrailsResponse:
    """
    Returns all content trails.

    **RFY row:** query vector is the user profile adjusted by global skip signals
    (orthogonal rejection) and positive play signals (attraction). Visibly shifts
    when ≥ 3 skips of any genre accumulate.

    **Genre rows:** always use plain user_vector + genre pre-filter. Per-genre
    skips (≥ 2 same-genre) control whether a row appears or is swapped out —
    they do not change the query direction.

    **'Because you watched' rows:** appear when ≥ 2 plays of the same genre occur.
    Use the centroid of played movie embeddings as their query vector.
    """
    db = get_db()

    # 1. User profile
    profile = await session_service.get_user_profile(db, user_id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No user profile for '{user_id}'. Run 'make seed' first.",
        )
    user_vector_list: List[float] = profile["user_vector"]
    profile_genres:   List[str]   = profile.get("genres", [])

    # 2. Session
    session     = await session_service.get_session_state(db, user_id)
    exclude_ids = await session_service.get_excluded_ids(db, user_id)
    played_ids  = await session_service.get_played_ids(db, user_id)
    all_exclude = list({*exclude_ids, *played_ids}) if played_ids else exclude_ids

    # 3. Embedding arrays
    # Global: ALL negative embeddings → drive RFY orthogonal rejection
    all_neg_arrays = await session_service.get_negative_embeddings(db, user_id)
    # Positive: ALL play embeddings → drive RFY positive attraction + "Because you watched"
    all_pos_arrays = await session_service.get_positive_embeddings(db, user_id)

    # 4. RFY query vector (global skips + plays)
    beta_eff = _effective_beta(session.positive_count)

    # Only fire rejection when global threshold is crossed (≥ NEGATIVE_THRESHOLD skips)
    neg_for_rfy = all_neg_arrays if session.mitigation_active else []

    rfy_query, rfy_rocchio, rfy_source = _decide_rfy_query(
        user_vector_list,
        neg_for_rfy,
        all_pos_arrays,
        beta_eff,
    )
    session.query_source = rfy_source

    # Log what drove the RFY change
    if rfy_rocchio:
        logger.info(
            "RFY query: source=%s neg_count=%d pos_count=%d "
            "global_threshold=%s beta_eff=%.2f",
            rfy_source, len(neg_for_rfy), len(all_pos_arrays),
            session.mitigation_active, beta_eff,
        )

    # 5. Active genre row slots — weighted by seed rank + play counts.
    # weight(genre) = seed_rank_weight + play_count × PLAY_GENRE_WEIGHT
    # Non-seed genres with enough plays can displace lower-ranked seed genres,
    # demonstrating live preference evolution.
    active_genre_slots = _compute_weighted_genre_slots(
        profile_genres=profile_genres,
        swapped_genres=session.swapped_genres,
        count=genre_rows_count,
        positive_genre_counts=session.positive_genre_counts,
        play_genre_weight=settings.PLAY_GENRE_WEIGHT,
        emerging_rows_count=settings.EMERGING_ROWS_COUNT,
        play_row_threshold=settings.PLAY_ROW_THRESHOLD,
    )

    # 6. Re-order slots so the page always reads:
    #      RFY → BYW rows → "You might like" rows → profile rows → Discovery
    #
    #   "played"   (0) — "Because you watched": centroid query, ≥ PLAY_ROW_THRESHOLD plays
    #   "emerging" (1) — "You might like":       user_vector query, 1 play below threshold
    #   "profile"  (2) — seed genre rows
    #   "swapped"  (3) — skip-rotated seed rows
    #
    # Stable sort: within each priority group, original weight order is preserved.
    _SLOT_PRIORITY = {"played": 0, "emerging": 1, "profile": 2, "swapped": 3}
    active_genre_slots = sorted(
        active_genre_slots,
        key=lambda slot: _SLOT_PRIORITY.get(slot[2], 99),
    )

    # 7. Boost map
    boost_map = await get_active_boost_map(db)

    # 8. Positive embedding lookup for "Because you watched" rows
    pos_events_with_arrays = list(zip(session.positive_events, all_pos_arrays))

    # 9. Parallel $vectorSearch
    # RFY  → rfy_query (shifted by global skips + plays) + soft genre penalty reranking
    # Genre rows → genre_query (rfy_query when mitigation_active, else user_vector_list)
    #
    # Soft genre penalty in get_recommendations():
    #   Skipped genres are deprioritised via score multiplier, not hard-filtered.
    #   penalty_factor = 1 - GENRE_PENALTY_STRENGTH × clamp(net_skips / MAX, 0, 1)
    #   net_skips(genre) = skip_count - play_count
    #   Penalty lifts automatically as user plays movies in those genres.
    #   No hard $nin filter — genres are never permanently banned from RFY.
    rfy_task = recommendation_service.get_recommendations(
        db=db,
        query_vector=rfy_query,
        exclude_ids=all_exclude or None,
        limit=trail_limit,
        genre_skip_counts=session.genre_skip_counts,
        positive_genre_counts=session.positive_genre_counts,
    )

    # When mitigation is active (≥ NEGATIVE_THRESHOLD global skips), profile/swapped
    # genre rows switch from user_vector_list to rfy_query so their content shifts
    # away from the skipped direction.  The genre pre-filter ($eq) is still applied —
    # an Action row still only shows Action movies, but now from the rotated query
    # vector that deprioritises heavily-skipped Action content.
    # "Because you watched" (played) rows always use the played-movie centroid — they
    # are already personalised to watched content and should not be rejection-shifted.
    genre_query = rfy_query if session.mitigation_active else user_vector_list

    genre_tasks = []
    for genre, _, row_source in active_genre_slots:
        if row_source == "played":
            # "Because you watched" row → centroid of played movie embeddings
            # Dedup by movie_id so multi-genre play entries don't skew the centroid
            seen_played_ids: set = set()
            genre_pos_arrays = []
            for ev, arr in pos_events_with_arrays:
                if ev.genre == genre and ev.movie_id not in seen_played_ids:
                    seen_played_ids.add(ev.movie_id)
                    genre_pos_arrays.append(arr)
            if genre_pos_arrays:
                genre_tasks.append(recommendation_service.get_played_similar_row(
                    db=db,
                    played_embeddings=genre_pos_arrays,
                    genre=genre,
                    exclude_ids=all_exclude or None,
                    limit=genre_row_limit,
                ))
            else:
                genre_tasks.append(recommendation_service.get_genre_row(
                    db=db,
                    query_vector=genre_query,
                    genre=genre,
                    exclude_ids=all_exclude or None,
                    limit=genre_row_limit,
                ))
        else:
            # Profile / emerging rows — use genre_query (rfy_query when mitigation_active)
            genre_tasks.append(recommendation_service.get_genre_row(
                db=db,
                query_vector=genre_query,
                genre=genre,
                exclude_ids=all_exclude or None,
                limit=genre_row_limit,
            ))

    # Discovery row — always present, shows content OUTSIDE current active genres.
    # Gives the audience something from unfamiliar genres to click, triggering
    # the preference-evolution story without waiting for profile rows to change.
    #
    # Mode A — fixed inclusion (DISCOVERY_GENRE_MAP has an entry for this user):
    #   Shows only movies from the configured genres, explicitly excluding the
    #   user's seed genres so the content is genuinely outside their bubble.
    #
    # Mode B — dynamic exclusion (no DISCOVERY_GENRE_MAP entry):
    #   Excludes movies from the user's current active genre rows.
    active_genres_set = [g for g, _, _ in active_genre_slots]
    _discovery_genre_map = settings.discovery_genre_map
    _user_discovery_genres = _discovery_genre_map.get(user_id)

    if _user_discovery_genres:
        # Mode A: fixed inclusion — show only the configured genres,
        # and exclude the user's seed genres to keep content genuinely new.
        logger.info(
            "Discovery row: fixed mode user=%s include=%s seed_exclude=%s",
            user_id, _user_discovery_genres, profile_genres,
        )
        discovery_task = recommendation_service.get_discovery_row(
            db=db,
            query_vector=genre_query,
            include_genres=_user_discovery_genres,
            seed_genres=profile_genres or None,
            exclude_ids=all_exclude or None,
            limit=genre_row_limit,
        )
    else:
        # Mode B: dynamic exclusion — exclude the user's current active genre rows
        logger.info(
            "Discovery row: dynamic mode user=%s exclude=%s",
            user_id, active_genres_set,
        )
        discovery_task = recommendation_service.get_discovery_row(
            db=db,
            query_vector=genre_query,
            exclude_genres=active_genres_set or None,
            exclude_ids=all_exclude or None,
            limit=genre_row_limit,
        )

    results = await asyncio.gather(rfy_task, *genre_tasks, discovery_task, return_exceptions=True)

    rfy_movies      = results[0]    if not isinstance(results[0], Exception)    else []
    discovery_movies = results[-1]  if not isinstance(results[-1], Exception)   else []
    genre_results   = results[1:-1]

    if isinstance(results[0], Exception):
        logger.error("RFY query failed: %s", results[0])
    if isinstance(results[-1], Exception):
        logger.warning("Discovery row failed: %s", results[-1])

    # 10. Boost injection (RFY only)
    rfy_movies = await apply_boosts_to_row(
        db=db, movies=rfy_movies, boost_map=boost_map,
        max_boosted_slots=settings.MAX_BOOSTED_SLOTS,
    )

    # 11. Build genre rows
    genre_rows: List[GenreRow] = []
    for idx, (task_result, (display_genre, swapped_from, row_source)) in enumerate(
        zip(genre_results, active_genre_slots)
    ):
        genre_movies = task_result if not isinstance(task_result, Exception) else []
        if isinstance(task_result, Exception):
            logger.error("Genre row [%s] failed: %s", display_genre, task_result)

        genre_rows.append(GenreRow(
            genre=display_genre,
            movies=genre_movies,
            row_index=idx,
            swapped_from=swapped_from,
            rocchio_applied=(row_source not in ("played", "emerging") and session.mitigation_active),
            row_source=row_source,
        ))

    # Append discovery row last
    if discovery_movies:
        genre_rows.append(GenreRow(
            genre="Discover Something New",
            movies=discovery_movies,
            row_index=len(genre_rows),
            swapped_from=None,
            rocchio_applied=False,
            row_source="discovery",
        ))

    # 12. Emerging genre counts — genres with plays > 0 but NOT currently in
    # active rows. Shown in the HUD "Emerging Interests" section.
    active_genre_names = {g for g, _, _ in active_genre_slots}
    emerging_genre_counts = {
        g: c for g, c in session.positive_genre_counts.items()
        if g not in active_genre_names and c > 0
    }

    # 13. Live Profile Vector projections — persistent baseline profile vs.
    # the ephemeral per-request RFY query vector, both projected onto the
    # same fixed 8-genre radar axes so the frontend can render two overlaid
    # radar polygons and a "shift %" badge.
    user_arr = list_to_numpy(user_vector_list)
    rfy_arr  = list_to_numpy(rfy_query)
    user_vector_projection = project_onto_genre_axes(user_arr)
    rfy_query_projection   = project_onto_genre_axes(rfy_arr)
    shift = vector_shift(user_arr, rfy_arr)

    return TrailsResponse(
        recommended_for_you=rfy_movies,
        genre_rows=genre_rows,
        session=session,
        rocchio_applied=rfy_rocchio,
        query_source=rfy_source,
        active_genre_skips=session.genre_skip_counts,
        positive_genre_counts=session.positive_genre_counts,
        profile_genres=profile_genres,
        beta_effective=beta_eff,
        emerging_genre_counts=emerging_genre_counts,
        user_vector_projection=user_vector_projection,
        rfy_query_projection=rfy_query_projection,
        vector_shift=shift,
    )
