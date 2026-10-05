"""
Marketing admin boost management endpoints.

POST   /api/admin/boost              — create / update a movie boost
DELETE /api/admin/boost/{movie_id}   — deactivate a boost
GET    /api/admin/boosts             — list all active boosts
GET    /api/admin/movies/search      — search movies by title (for the UI picker)

No authentication — this is a PoC demo. In production, protect with JWT / API key.
"""
from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, HTTPException, Path, Query, status

from app.database import get_db
from app.models import BoostIn, BoostOut
from app.services import boost_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["admin"], prefix="/admin")


@router.post(
    "/boost",
    response_model=BoostOut,
    summary="Create or update a movie boost",
    status_code=status.HTTP_200_OK,
)
async def upsert_boost(body: BoostIn) -> BoostOut:
    """
    Set a marketing boost on a specific movie.

    The boosted movie will be guaranteed a top-3 slot in the
    **Recommended For You** row for all users.

    - **boost_score**: multiplier ≥ 1.0 (e.g. 2.0 = strong push to front)
    - **label**: badge shown on the card — `Hot`, `Trending`, `Sponsored`
    - **expires_hours**: auto-deactivate after N hours (omit = never)
    """
    db = get_db()
    try:
        return await boost_service.upsert_boost(
            db=db,
            movie_id=body.movie_id,
            boost_score=body.boost_score,
            label=body.label,
            expires_hours=body.expires_hours,
        )
    except Exception as exc:
        logger.exception("upsert_boost failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


@router.delete(
    "/boost/{movie_id}",
    summary="Deactivate a movie boost",
    status_code=status.HTTP_200_OK,
)
async def deactivate_boost(
    movie_id: str = Path(..., description="MongoDB ObjectId of the movie"),
) -> dict:
    """Remove a movie from the boosted set. The movie remains in the catalogue."""
    db = get_db()
    try:
        removed = await boost_service.deactivate_boost(db, movie_id)
        if not removed:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No active boost found for movie_id={movie_id}",
            )
        return {"success": True, "movie_id": movie_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("deactivate_boost failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get(
    "/boosts",
    response_model=List[BoostOut],
    summary="List all active movie boosts",
)
async def list_boosts() -> List[BoostOut]:
    """Return all currently active, non-expired boosts with joined movie titles."""
    db = get_db()
    return await boost_service.list_active_boosts(db)


@router.get(
    "/movies/search",
    summary="Search movies by title (for the boost picker UI)",
)
async def search_movies(
    q: str = Query(..., min_length=1, description="Title search query"),
    limit: int = Query(default=10, ge=1, le=30),
) -> list:
    """
    Case-insensitive title search against the embedded_movies collection.
    Used by the admin modal to find movies before boosting them.
    """
    db = get_db()
    return await boost_service.search_movies_by_title(db, q, limit)
