"""
POST /api/events

Accepts a single user interaction event (skip, abandon, play, complete),
records it in the session, and returns the updated SessionState.

Skip / abandon:
  - Composite negative embedding stored in session events[].
  - Returns trigger_refresh when a threshold is crossed.
  - When global skip threshold (NEGATIVE_THRESHOLD) is FIRST crossed:
      → fires asyncio.create_task(reject_user_profile()) — updates user_vector
        persistently away from the skipped direction (SKIP_DRIFT_RATE=0.5).
        This is separate from the ephemeral per-request rejection (REJECTION_STRENGTH=0.8).

Play / complete:
  - Composite positive embedding stored in session positive_events[].
  - Fires asyncio.create_task(drift_user_profile()) toward played content.
  - Returns trigger_refresh always.
"""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, status

from app.config import settings
from app.database import get_db
from app.models import EventIn, EventResponse
from app.services import session_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["events"])


@router.post(
    "/events",
    response_model=EventResponse,
    summary="Log a user interaction event",
    status_code=status.HTTP_200_OK,
)
async def log_event(event: EventIn) -> EventResponse:
    """
    Record a user interaction and return the updated session state.

    **Skip / abandon:**
      - Stored in `events[]` with composite embedding.
      - `trigger_refresh=True` when a threshold is crossed.
      - When **global threshold** (`NEGATIVE_THRESHOLD`) is crossed for the first time,
        fires an async background task that persistently updates `user_profiles.user_vector`
        using orthogonal rejection (`SKIP_DRIFT_RATE=0.5`).  Never fires again for the
        same session until the user resets.

    **Play / complete:**
      - Stored in `positive_events[]`.
      - Async drift of `user_profiles.user_vector` toward played content (`DRIFT_RATE`).
      - `trigger_refresh=True` always.
    """
    try:
        db = get_db()

        # Capture state BEFORE this event to detect threshold crossings
        prev_session = await session_service.get_session_state(db, event.user_id)
        was_below_global = not prev_session.mitigation_active

        # Resolve effective genres for multi-genre attribution on play events.
        # For RFY rows the frontend sends all_genres = movie.genres (full list).
        # For genre rows it sends all_genres = [rowGenre] (single-element).
        # Falls back to [genre] when all_genres is absent (e.g. legacy / skip events).
        effective_genres: list[str] = (
            event.all_genres
            if event.all_genres
            else ([event.genre] if event.genre else [])
        )

        # Record the event
        session = await session_service.log_event(
            db=db,
            user_id=event.user_id,
            movie_id=event.movie_id,
            event_type=event.event_type,
            genre=event.genre,
            trail=event.trail,
            effective_genres=effective_genres,
        )

        message        = ""
        trigger_refresh = False
        trigger_reason  = None

        if event.event_type in ("skip", "abandon"):
            genre_count = session.genre_skip_counts.get(event.genre or "", 0)

            # ── Global threshold just crossed for the first time this session ──
            just_crossed_global = (
                was_below_global
                and session.mitigation_active
                and session.negative_count == settings.NEGATIVE_THRESHOLD
            )
            if just_crossed_global:
                asyncio.create_task(
                    session_service.reject_user_profile(db, event.user_id),
                    name=f"skip-reject-{event.user_id}",
                )
                logger.info(
                    "Global threshold crossed for user=%s (neg=%d) — "
                    "async profile rejection queued (SKIP_DRIFT_RATE=%.2f)",
                    event.user_id, session.negative_count, settings.SKIP_DRIFT_RATE,
                )

            # ── Build response message ─────────────────────────────────────────
            if session.mitigation_active:
                message = (
                    f"Global skip threshold reached ({session.negative_count}/"
                    f"{session.threshold}). "
                    f"RFY re-ranked via orthogonal rejection. "
                    f"Profile persistently updated (async, rate={settings.SKIP_DRIFT_RATE})."
                ) if just_crossed_global else (
                    f"Global skip active — {session.negative_count} skips. "
                    f"RFY re-ranked."
                )
                trigger_refresh = True
                trigger_reason  = "rfy_rejected"
            else:
                remaining       = session.threshold - session.negative_count
                genre_remaining = max(0, settings.GENRE_SKIP_THRESHOLD - genre_count)
                message = (
                    f"Skip recorded ({session.negative_count}/{session.threshold}). "
                    f"'{event.genre or 'genre'}': {genre_count}/{settings.GENRE_SKIP_THRESHOLD}. "
                    f"{remaining} more global / {genre_remaining} more same-genre to trigger."
                )
                # Still trigger refresh so HUD and genre rows update immediately
                trigger_refresh = True
                trigger_reason  = "skip_recorded"

        elif event.event_type in ("play", "complete"):
            genres_label = ", ".join(effective_genres) if effective_genres else (event.genre or "unknown")
            message = (
                f"Play recorded — '{genres_label}' "
                f"({session.positive_count} total). "
                f"RFY attracted toward played content. "
                f"Profile drift queued (DRIFT_RATE={settings.DRIFT_RATE})."
            )
            trigger_refresh = True
            trigger_reason  = "rfy_positive"

        logger.info(
            "Event: user=%s type=%s genre=%s trail=%s "
            "neg=%d/%d pos=%d genre_skips=%s mitigation=%s",
            event.user_id, event.event_type, event.genre, event.trail,
            session.negative_count, session.threshold,
            session.positive_count, session.genre_skip_counts,
            session.mitigation_active,
        )

        return EventResponse(
            success=True,
            session=session,
            message=message,
            trigger_refresh=trigger_refresh,
            trigger_reason=trigger_reason,
        )

    except Exception as exc:
        logger.exception("Failed to log event for user %s: %s", event.user_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log event: {exc}",
        )
