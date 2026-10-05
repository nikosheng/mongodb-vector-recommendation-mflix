"""
Async MongoDB client management using Motor.
Provides a single shared AsyncIOMotorClient and database handle.
TTL index on user_realtime_sessions is created at startup.
"""
from __future__ import annotations

import logging
from typing import Optional

import certifi
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING

from app.config import settings

logger = logging.getLogger(__name__)

# Module-level singletons
_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


async def startup() -> None:
    """Initialize the Motor client and ensure required indexes exist."""
    global _client, _db

    logger.info("Connecting to MongoDB Atlas...")
    _client = AsyncIOMotorClient(settings.MONGO_URI, tlsCAFile=certifi.where())
    _db = _client[settings.DB_NAME]

    # Verify connectivity
    await _client.admin.command("ping")
    logger.info("MongoDB connection established: db=%s", settings.DB_NAME)

    # Ensure TTL index on user_realtime_sessions (expires after 24h)
    sessions_col = _db["user_realtime_sessions"]
    await sessions_col.create_index(
        [("expires_at", ASCENDING)],
        expireAfterSeconds=0,   # TTL driven by the expires_at field value
        name="ttl_expires_at",
        background=True,
    )
    logger.info("TTL index on user_realtime_sessions verified.")

    # Ensure unique index on user_id for O(1) session lookups
    await sessions_col.create_index(
        [("user_id", ASCENDING)],
        name="unique_user_id",
        unique=True,
        background=True,
    )

    # Ensure index on user_profiles.user_id
    profiles_col = _db["user_profiles"]
    await profiles_col.create_index(
        [("user_id", ASCENDING)],
        name="unique_user_id",
        unique=True,
        background=True,
    )
    logger.info("user_profiles index verified.")


async def shutdown() -> None:
    """Close the Motor client on application shutdown."""
    global _client, _db
    if _client is not None:
        _client.close()
        logger.info("MongoDB connection closed.")
    _client = None
    _db = None


def get_db() -> AsyncIOMotorDatabase:
    """Return the active database handle. Call after startup()."""
    if _db is None:
        raise RuntimeError("Database not initialised — call startup() first.")
    return _db


def get_client() -> AsyncIOMotorClient:
    """Return the active Motor client."""
    if _client is None:
        raise RuntimeError("Database not initialised — call startup() first.")
    return _client
