"""
POST /api/demo/reset

Resets the demo session — deletes all accumulated skip/play events AND
restores the user_vector to a freshly embedded composite baseline.

Why restore the vector:
  Repeated drift_user_profile() and reject_user_profile() calls accumulate
  over a session.  If the demo is run multiple times without re-seeding,
  the user_vector drifts out of the composite embedding space, making the
  rejection math ineffective (low dot product with skipped content → no visible
  change).  Restoring on reset guarantees every demo run starts from a clean,
  correctly-aligned vector.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, status

from app.config import settings
from app.database import get_db
from app.models import ResetRequest, ResetResponse, SessionState
from app.services import session_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["demo"])


@router.post(
    "/demo/reset",
    response_model=ResetResponse,
    summary="Reset the demo session — clears all accumulated events",
    status_code=status.HTTP_200_OK,
)
async def reset_demo(body: ResetRequest = None) -> ResetResponse:
    """
    Delete all skip/play events for the user.

    - Removes the `user_realtime_sessions` document entirely.
    - All genre skip counts, swap states, and play signals reset to zero.
    - `user_profiles.user_vector` is NOT touched — the baseline profile is preserved.

    Events accumulate indefinitely between resets (no time window).
    Use this endpoint to restart the churn mitigation demo from a clean slate.
    """
    if body is None:
        body = ResetRequest()

    user_id = body.user_id
    try:
        db = get_db()

        # 1. Clear the session (skip/play events)
        await session_service.reset_session(db, user_id)

        # 2. Restore user_vector to a fresh composite baseline.
        #    This re-embeds the stored profile_text + genres via Voyage AI so the
        #    vector is always correctly aligned with the composite movie embeddings.
        #    Also resets skip_reject_count and play_count to 0.
        vector_restored = await session_service.restore_user_vector(db, user_id)
        if not vector_restored:
            logger.warning(
                "Demo reset: session cleared but user_vector restore failed for %s "
                "(Voyage API may be unavailable). Run 'make seed' to fix manually.",
                user_id,
            )

        fresh_session = SessionState(
            user_id=user_id,
            negative_count=0,
            threshold=settings.NEGATIVE_THRESHOLD,
            mitigation_active=False,
            penalized_genre=None,
            genre_skip_counts={},
            swapped_genres=[],
            rocchio_genre=None,
            positive_count=0,
            positive_genre_counts={},
        )

        logger.info(
            "Demo session reset for user: %s (vector_restored=%s)",
            user_id, vector_restored,
        )
        return ResetResponse(
            success=True,
            user_id=user_id,
            message=(
                f"Session cleared for '{user_id}'. "
                f"All skip/play signals reset. "
                f"{'User vector restored to baseline.' if vector_restored else 'Vector restore failed — run make seed.'}"
            ),
            session=fresh_session,
        )
    except Exception as exc:
        logger.exception("Failed to reset demo for user %s: %s", user_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reset failed: {exc}",
        )


def _to_display_name(user_id: str) -> str:
    """Convert user_id to a human-readable display name.
    Prefers the display_name field stored in user_profiles; falls back to
    title-casing the part after the last underscore.
    e.g. 'demo_horror' → 'Horror Fan' (if stored), else 'Horror'
    """
    # strip 'demo_' prefix if present, title-case remainder
    label = user_id.removeprefix("demo_").replace("_", " ").title()
    return label


@router.get(
    "/demo/personas",
    summary="List all demo persona profiles for the persona switcher",
    response_model=List[Dict[str, Any]],
)
async def list_personas() -> List[Dict[str, Any]]:
    """
    Returns profile metadata for every demo user defined in DEMO_USERS.
    Used by the frontend persona switcher to populate pills with genre tags.

    Fields returned per persona:
      user_id        — unique identifier
      display_name   — human-readable label (from profile or derived)
      genres         — seed genre list
      profile_excerpt — first 120 chars of profile_text
    """
    db       = get_db()
    personas = []
    for uid in settings.DEMO_USERS.split(","):
        uid = uid.strip()
        if not uid:
            continue
        try:
            doc = await db["user_profiles"].find_one(
                {"user_id": uid},
                {"user_id": 1, "display_name": 1, "genres": 1, "profile_text": 1},
            )
            if doc:
                personas.append({
                    "user_id":        uid,
                    "display_name":   doc.get("display_name") or _to_display_name(uid),
                    "genres":         doc.get("genres", []),
                    "profile_excerpt": (doc.get("profile_text") or "")[:120],
                })
            else:
                # Profile not yet seeded — return minimal stub so UI still renders
                personas.append({
                    "user_id":        uid,
                    "display_name":   _to_display_name(uid),
                    "genres":         [],
                    "profile_excerpt": "Profile not seeded yet. Run make seed-all.",
                })
        except Exception as exc:
            logger.warning("list_personas: error fetching %s: %s", uid, exc)
    return personas
