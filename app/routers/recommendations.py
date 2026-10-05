"""
GET /api/recommendations/{user_id}

Full recommendation pipeline:
  1. Fetch user_vector from user_profiles collection.
  2. Read the current sliding-window session state.
  3. If negative_count >= NEGATIVE_THRESHOLD:
       apply Rocchio projection → ephemeral V_query
     Else:
       use user_vector directly as V_query
  4. Execute Atlas $vectorSearch with V_query + $nin exclusion.
  5. Return movies + session state + telemetry flags.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Path, Query, status

from app.config import settings
from app.database import get_db
from app.models import RecommendationResponse, SessionState
from app.services import recommendation_service, session_service
from app.services.vector_math import compute_rocchio_query, list_to_numpy

logger = logging.getLogger(__name__)
router = APIRouter(tags=["recommendations"])


@router.get(
    "/recommendations/{user_id}",
    response_model=RecommendationResponse,
    summary="Get personalised movie recommendations with churn mitigation",
)
async def get_recommendations(
    user_id: str = Path(..., description="User identifier (e.g. 'demo_user')"),
    limit: int = Query(
        default=settings.RECOMMENDATION_LIMIT,
        ge=1,
        le=50,
        description="Max number of movies to return",
    ),
) -> RecommendationResponse:
    """
    Returns a personalised feed with dynamic Rocchio negative feedback re-ranking.

    **Rocchio projection** is applied when ≥ 3 skip/abandon events accumulate
    within the last 10 minutes.  The ephemeral query vector is:

        V_query = normalize( V_user − α · mean( V_skipped_i ) )

    The user's persistent `user_vector` in `user_profiles` is **never modified**.
    """
    db = get_db()

    # -----------------------------------------------------------------------
    # 1. Fetch user vector
    # -----------------------------------------------------------------------
    user_vector_list = await recommendation_service.get_user_vector(db, user_id)
    if user_vector_list is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No user profile found for '{user_id}'. "
                "Run scripts/seed_demo_user.py first."
            ),
        )

    # -----------------------------------------------------------------------
    # 2. Read session & collect negative embeddings
    # -----------------------------------------------------------------------
    session = await session_service.get_session_state(db, user_id)
    exclude_ids = await session_service.get_excluded_ids(db, user_id)

    # -----------------------------------------------------------------------
    # 3. Decide query vector
    # -----------------------------------------------------------------------
    rocchio_applied = False
    query_source = "user_vector"

    if session.mitigation_active and len(session.negative_events) >= settings.NEGATIVE_THRESHOLD:
        neg_arrays = await session_service.get_negative_embeddings(db, user_id)
        if neg_arrays:
            try:
                user_arr = list_to_numpy(user_vector_list)
                query_vector = compute_rocchio_query(
                    user_vector=user_arr,
                    negative_embeddings=neg_arrays,
                    alpha=settings.ALPHA,
                )
                rocchio_applied = True
                query_source = "rocchio"
                logger.info(
                    "Rocchio applied for user=%s: %d negatives, α=%.2f",
                    user_id, len(neg_arrays), settings.ALPHA,
                )
            except Exception as exc:
                logger.warning(
                    "Rocchio computation failed for user=%s, falling back to user_vector: %s",
                    user_id, exc,
                )
                query_vector = user_vector_list
        else:
            logger.warning(
                "Mitigation active for user=%s but no embeddings available; "
                "falling back to user_vector.",
                user_id,
            )
            query_vector = user_vector_list
    else:
        query_vector = user_vector_list

    # -----------------------------------------------------------------------
    # 4. Atlas $vectorSearch
    # -----------------------------------------------------------------------
    movies = await recommendation_service.get_recommendations(
        db=db,
        query_vector=query_vector,
        exclude_ids=exclude_ids if exclude_ids else None,
        limit=limit,
    )

    # -----------------------------------------------------------------------
    # 5. Return full response
    # -----------------------------------------------------------------------
    return RecommendationResponse(
        movies=movies,
        session=session,
        rocchio_applied=rocchio_applied,
        query_source=query_source,
    )
