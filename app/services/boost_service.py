"""
Marketing admin boost service.

Storage: MongoDB collection `movie_boosts`
Schema:
  {
    "_id":         ObjectId,
    "movie_id":    ObjectId,    # ref to embedded_movies
    "boost_score": float,       # multiplier ≥ 1.0
    "label":       str,         # "Hot" | "Trending" | "Sponsored"
    "active":      bool,
    "created_at":  datetime,
    "expires_at":  datetime | None
  }

Boost injection strategy (MAX_BOOSTED_SLOTS = 3):
  1. Fetch all active, non-expired boosts.
  2. For each boosted movie already present in the vector-search result list,
     move it up to the front (preserving its badge metadata).
  3. For boosted movies NOT in the vector-search list, fetch them from DB
     and insert at the front (up to MAX_BOOSTED_SLOTS total).
  4. The remaining non-boosted movies fill positions after the boosted slots.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.config import settings
from app.models import BoostOut, MovieOut

logger = logging.getLogger(__name__)

BOOSTS_COL = "movie_boosts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _active_filter() -> dict:
    now = _utcnow()
    return {
        "active": True,
        "$or": [
            {"expires_at": None},
            {"expires_at": {"$gt": now}},
        ],
    }


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

async def upsert_boost(
    db: AsyncIOMotorDatabase,
    movie_id: str,
    boost_score: float,
    label: str,
    expires_hours: Optional[int],
) -> BoostOut:
    """Create or update a boost for a movie. Returns the resulting BoostOut."""
    col = db[BOOSTS_COL]
    movies_col = db.get_collection(settings.MOVIES_COLLECTION)

    oid = ObjectId(movie_id)
    now = _utcnow()
    expires_at = now + timedelta(hours=expires_hours) if expires_hours else None

    doc = {
        "movie_id": oid,
        "boost_score": float(boost_score),
        "label": label,
        "active": True,
        "created_at": now,
        "expires_at": expires_at,
    }

    result = await col.find_one_and_update(
        {"movie_id": oid},
        {"$set": doc},
        upsert=True,
        return_document=True,
    )

    # Join movie title for the response
    movie_doc = await movies_col.find_one({"_id": oid}, {"title": 1})
    title = movie_doc.get("title") if movie_doc else None

    return BoostOut(
        id=str(result["_id"]),
        movie_id=movie_id,
        title=title,
        boost_score=result["boost_score"],
        label=result["label"],
        active=result["active"],
        created_at=result["created_at"],
        expires_at=result.get("expires_at"),
    )


async def deactivate_boost(
    db: AsyncIOMotorDatabase,
    movie_id: str,
) -> bool:
    """Deactivate (soft-delete) a boost. Returns True if a document was matched."""
    col = db[BOOSTS_COL]
    result = await col.update_one(
        {"movie_id": ObjectId(movie_id)},
        {"$set": {"active": False}},
    )
    return result.matched_count > 0


async def list_active_boosts(db: AsyncIOMotorDatabase) -> List[BoostOut]:
    """Return all currently active, non-expired boosts with joined movie titles."""
    col = db[BOOSTS_COL]
    movies_col = db.get_collection(settings.MOVIES_COLLECTION)

    cursor = col.find(_active_filter()).sort("created_at", -1)
    docs = await cursor.to_list(length=100)

    results: List[BoostOut] = []
    for doc in docs:
        movie_doc = await movies_col.find_one(
            {"_id": doc["movie_id"]}, {"title": 1}
        )
        results.append(
            BoostOut(
                id=str(doc["_id"]),
                movie_id=str(doc["movie_id"]),
                title=movie_doc.get("title") if movie_doc else None,
                boost_score=doc["boost_score"],
                label=doc["label"],
                active=doc["active"],
                created_at=doc["created_at"],
                expires_at=doc.get("expires_at"),
            )
        )
    return results


async def get_active_boost_map(
    db: AsyncIOMotorDatabase,
) -> dict[str, dict]:
    """
    Return {movie_id_str: {boost_score, label}} for all active boosts.
    Used during recommendation post-processing.
    """
    col = db[BOOSTS_COL]
    cursor = col.find(_active_filter(), {"movie_id": 1, "boost_score": 1, "label": 1})
    docs = await cursor.to_list(length=200)
    return {
        str(d["movie_id"]): {"boost_score": d["boost_score"], "label": d["label"]}
        for d in docs
    }


# ---------------------------------------------------------------------------
# Boost injection
# ---------------------------------------------------------------------------

async def apply_boosts_to_row(
    db: AsyncIOMotorDatabase,
    movies: List[MovieOut],
    boost_map: Optional[dict[str, dict]] = None,
    max_boosted_slots: int = settings.MAX_BOOSTED_SLOTS,
) -> List[MovieOut]:
    """
    Inject boosted movies into the top positions of a movie row.

    Algorithm:
      1. Tag any movies already in `movies` that appear in `boost_map`.
      2. Collect boosted-but-absent movie_ids — fetch from DB, build MovieOut.
      3. Sort boosted (present or fetched) by boost_score descending, take top N.
      4. Front-load boosted movies; fill remaining positions with non-boosted.

    Args:
        movies:            Vector-search results (ordered by similarity).
        boost_map:         {movie_id: {boost_score, label}} — pass in to avoid
                           repeated DB calls across parallel rows.
        max_boosted_slots: How many top positions are reserved for boosted movies.
    """
    if not boost_map:
        boost_map = await get_active_boost_map(db)

    if not boost_map:
        return movies  # nothing to boost

    present_ids = {m.id for m in movies}
    boosted_present: List[MovieOut] = []
    non_boosted: List[MovieOut] = []

    for m in movies:
        if m.id in boost_map:
            bdata = boost_map[m.id]
            boosted_present.append(m.model_copy(update={
                "boost_score": bdata["boost_score"],
                "boost_label": bdata["label"],
                "is_boosted": True,
            }))
        else:
            non_boosted.append(m)

    # Fetch boosted movies not returned by vector search
    absent_ids = [mid for mid in boost_map if mid not in present_ids]
    fetched_boosted: List[MovieOut] = []

    if absent_ids:
        movies_col = db.get_collection(settings.MOVIES_COLLECTION)
        cursor = movies_col.find(
            {"_id": {"$in": [ObjectId(mid) for mid in absent_ids]}},
            {"title": 1, "genres": 1, "year": 1, "plot": 1, "poster": 1},
        )
        fetched_docs = await cursor.to_list(length=max_boosted_slots)
        for doc in fetched_docs:
            mid_str = str(doc["_id"])
            bdata = boost_map[mid_str]
            fetched_boosted.append(MovieOut(
                id=mid_str,
                title=doc.get("title", "Unknown"),
                genres=doc.get("genres") or [],
                year=doc.get("year"),
                plot=doc.get("plot"),
                poster=doc.get("poster"),
                vector_score=0.0,   # not from vector search — boosted inject
                boost_score=bdata["boost_score"],
                boost_label=bdata["label"],
                is_boosted=True,
            ))

    # Merge boosted lists and sort by boost_score descending
    all_boosted = boosted_present + fetched_boosted
    all_boosted.sort(key=lambda m: m.boost_score or 0, reverse=True)
    all_boosted = all_boosted[:max_boosted_slots]

    logger.info(
        "Boost injection: %d boosted (present=%d, fetched=%d) into top %d slots",
        len(all_boosted), len(boosted_present), len(fetched_boosted), max_boosted_slots,
    )

    return all_boosted + non_boosted


async def search_movies_by_title(
    db: AsyncIOMotorDatabase,
    query: str,
    limit: int = 10,
) -> List[dict]:
    """Simple title search for the admin boost UI movie picker."""
    movies_col = db.get_collection(settings.MOVIES_COLLECTION)
    import re
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    cursor = movies_col.find(
        {"title": {"$regex": pattern}},
        {"title": 1, "genres": 1, "year": 1, "poster": 1},
    ).limit(limit)
    docs = await cursor.to_list(length=limit)
    return [
        {
            "id": str(d["_id"]),
            "title": d.get("title", ""),
            "genres": d.get("genres", []),
            "year": d.get("year"),
            "poster": d.get("poster"),
        }
        for d in docs
    ]
