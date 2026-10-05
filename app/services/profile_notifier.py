"""
app/services/profile_notifier.py
=================================
In-process pub/sub bus for real-time profile update notifications.

When reject_user_profile() or drift_user_profile() successfully write
a new user_vector to MongoDB, they call notify_profile_updated().  The
SSE endpoint in app/routers/profile_stream.py reads from the per-user
queue and pushes the event to the connected browser, which immediately
re-fetches GET /api/trails — no polling, no timer hacks.

Design:
  - One asyncio.Queue per active user_id (created on first SSE connect).
  - Queues are bounded (maxsize=20) to prevent unbounded growth during
    rapid skip/play sequences.  Excess notifications are dropped silently
    (the frontend will already be re-fetching from the prior notification).
  - Queue entries are plain dicts: {"type": "profile_updated", "reason": str}
  - When no SSE client is connected for a user, notify_profile_updated()
    is a no-op (queue does not exist).
  - Queues are cleaned up automatically when the SSE stream disconnects.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Module-level registry: user_id → asyncio.Queue
_queues: Dict[str, asyncio.Queue] = {}

_QUEUE_MAXSIZE = 20


def register(user_id: str) -> asyncio.Queue:
    """
    Create and register a queue for user_id.
    Called when an SSE client connects.  If a queue already exists
    (e.g. reconnect) it is replaced with a fresh empty one.
    """
    q: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
    _queues[user_id] = q
    logger.debug("profile_notifier: registered queue for user=%s", user_id)
    return q


def unregister(user_id: str) -> None:
    """
    Remove the queue for user_id.
    Called when the SSE client disconnects.
    """
    _queues.pop(user_id, None)
    logger.debug("profile_notifier: unregistered queue for user=%s", user_id)


async def notify_profile_updated(user_id: str, reason: str = "updated") -> None:
    """
    Push a profile_updated notification to the SSE stream for user_id.

    Args:
        user_id: The user whose profile was just updated.
        reason:  Human-readable reason string ("rejection" | "drift" | "updated").
                 Sent to the frontend for logging/debugging.

    No-op if no SSE client is currently connected for this user.
    Drops the notification silently if the queue is full (bounded queue).
    """
    q: Optional[asyncio.Queue] = _queues.get(user_id)
    if q is None:
        # No SSE client connected — nothing to notify
        return
    payload = {"type": "profile_updated", "reason": reason}
    try:
        q.put_nowait(payload)
        logger.debug(
            "profile_notifier: notified user=%s reason=%s queue_size=%d",
            user_id, reason, q.qsize(),
        )
    except asyncio.QueueFull:
        # Queue is full — a previous notification is already pending.
        # The frontend will re-fetch from that notification; this one can be dropped.
        logger.debug(
            "profile_notifier: queue full for user=%s, dropping notification (reason=%s)",
            user_id, reason,
        )
