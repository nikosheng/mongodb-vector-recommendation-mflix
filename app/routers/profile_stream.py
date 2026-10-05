"""
GET /api/profile/stream/{user_id}

Server-Sent Events (SSE) endpoint that pushes real-time profile update
notifications to the browser.

Flow:
  1. Browser opens EventSource("/api/profile/stream/demo_user") on page load.
  2. This endpoint registers a per-user asyncio.Queue via profile_notifier.
  3. When reject_user_profile() or drift_user_profile() finishes writing
     user_profiles.user_vector, they call notify_profile_updated() which
     puts an event on the queue.
  4. The generator below dequeues it and yields an SSE frame.
  5. The browser EventSource fires the 'profile_updated' event → immediately
     calls fetchTrails() — no timer, no polling, no race condition.

Heartbeat:
  Every 15 seconds a 'heartbeat' comment frame is sent to prevent proxy /
  browser timeout disconnects.  The browser ignores heartbeat events.

Reconnect:
  EventSource auto-reconnects after ~3s on disconnect.  The browser will
  re-register and receive the next notification normally.

Error handling:
  On client disconnect (GeneratorExit / asyncio.CancelledError) the queue
  is unregistered cleanly.
"""
from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Path
from fastapi.responses import StreamingResponse

from app.services import profile_notifier

logger = logging.getLogger(__name__)
router = APIRouter(tags=["profile-stream"])

_HEARTBEAT_INTERVAL = 15   # seconds between keep-alive frames


@router.get(
    "/profile/stream/{user_id}",
    summary="SSE stream — pushes profile_updated events when user_vector changes",
    response_class=StreamingResponse,
)
async def profile_stream(
    user_id: str = Path(..., description="User identifier"),
) -> StreamingResponse:
    """
    Opens a persistent SSE connection for the given user.

    **Events emitted:**

    | Event name        | When                                         | Data fields              |
    |-------------------|----------------------------------------------|--------------------------|
    | `profile_updated` | After `reject_user_profile()` write succeeds | `reason: "rejection"`    |
    | `profile_updated` | After `drift_user_profile()` write succeeds  | `reason: "drift"`        |
    | `heartbeat`       | Every 15 s (keep-alive)                      | `{}`                     |

    The browser should call `fetchTrails()` on every `profile_updated` event.
    """
    async def event_generator():
        # Register this user's notification queue
        q = profile_notifier.register(user_id)
        logger.info("SSE: client connected user=%s", user_id)
        try:
            while True:
                # Wait for a notification or heartbeat timeout
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=_HEARTBEAT_INTERVAL)
                    # Real notification — send as a named SSE event
                    data = json.dumps(payload)
                    yield f"event: profile_updated\ndata: {data}\n\n"
                    logger.info(
                        "SSE: sent profile_updated to user=%s reason=%s",
                        user_id, payload.get("reason"),
                    )
                except asyncio.TimeoutError:
                    # No notification within heartbeat window — send keep-alive
                    yield "event: heartbeat\ndata: {}\n\n"
        except (GeneratorExit, asyncio.CancelledError):
            # Client disconnected cleanly
            logger.info("SSE: client disconnected user=%s", user_id)
        except Exception as exc:
            logger.warning("SSE: unexpected error for user=%s: %s", user_id, exc)
        finally:
            # Always clean up the queue on disconnect
            profile_notifier.unregister(user_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            # Disable buffering so events arrive immediately
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # nginx
            "Connection": "keep-alive",
        },
    )
