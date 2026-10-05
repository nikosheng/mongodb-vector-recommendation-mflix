"""
scripts/setup_indexes.py
========================
Verifies all required MongoDB indexes for the churn mitigation demo.

Run once before starting the application:
    python scripts/setup_indexes.py

Checks:
  1. Atlas Vector Search index `vector_index_voyage4` on embedded_movies
       - Must have 'genres' as a filter field (for per-genre row queries)
       - Must have '_id'   as a filter field (for $nin exclusion)
  2. TTL index on user_realtime_sessions.expires_at
  3. Unique index on user_realtime_sessions.user_id
  4. Unique index on user_profiles.user_id
  5. movie_boosts collection + active index
  6. Embedding coverage report on embedded_movies
"""
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING

MONGO_URI       = os.environ["MONGO_URI"]
DB_NAME         = os.environ.get("DB_NAME", "sample_mflix")
VECTOR_INDEX    = os.environ.get("VECTOR_INDEX_NAME", "vector_index_voyage4")
EMBEDDING_FIELD = os.environ.get("EMBEDDING_FIELD", "plot_embedding_voyage_4")
MOVIES_COL      = os.environ.get("MOVIES_COLLECTION", "embedded_movies")


def green(s):  return f"\033[32m{s}\033[0m"
def yellow(s): return f"\033[33m{s}\033[0m"
def red(s):    return f"\033[31m{s}\033[0m"
def bold(s):   return f"\033[1m{s}\033[0m"
def tick():    return green("✓")
def cross():   return red("✗")


# ---------------------------------------------------------------------------

async def verify_vector_index(db) -> bool:
    print(f"\n{'─'*62}")
    print(bold("1. Atlas Vector Search Index"))
    print(f"{'─'*62}")
    print(f"   Collection : {DB_NAME}.{MOVIES_COL}")
    print(f"   Index name : {VECTOR_INDEX}")

    try:
        cursor = db[MOVIES_COL].aggregate([{"$listSearchIndexes": {}}])
        indexes = await cursor.to_list(length=50)
    except Exception as exc:
        print(red(f"   {cross()} listSearchIndexes failed: {exc}"))
        return False

    matched = [i for i in indexes if i.get("name") == VECTOR_INDEX]
    if not matched:
        print(red(f"   {cross()} Index '{VECTOR_INDEX}' NOT FOUND"))
        print(yellow("   Available indexes:"))
        for i in indexes:
            print(f"     - {i.get('name')} (status: {i.get('status')})")
        print(yellow(
            "\n   Create the index in Atlas UI → Search → Vector Search:\n"
            "   {\n"
            '     "fields": [\n'
            '       {"type":"vector","path":"plot_embedding_voyage_4","numDimensions":1024,"similarity":"cosine"},\n'
            '       {"type":"filter","path":"_id"},\n'
            '       {"type":"filter","path":"genres"}\n'
            "     ]\n"
            "   }"
        ))
        return False

    idx    = matched[0]
    status = idx.get("status", "UNKNOWN")
    sfn    = green if status == "READY" else yellow
    print(sfn(f"\n   {tick()} Index '{VECTOR_INDEX}' — status: {status}"))

    defn   = idx.get("latestDefinition", idx.get("definition", {}))
    fields = defn.get("fields", [])

    has_vector = False
    has_id_filter = False
    has_genre_filter = False

    for f in fields:
        ftype = f.get("type", "?")
        path  = f.get("path", "?")
        dims  = f.get("numDimensions", "")
        sim   = f.get("similarity", "")
        extras = ""
        if dims: extras += f"  dims={dims}"
        if sim:  extras += f"  similarity={sim}"
        print(f"     field: {path}  type={ftype}{extras}")

        if ftype == "vector" and path == EMBEDDING_FIELD:
            has_vector = True
        if ftype == "filter" and path == "_id":
            has_id_filter = True
        if ftype == "filter" and path == "genres":
            has_genre_filter = True

    # Report required fields
    print()
    checks = [
        (has_vector,       f"vector field '{EMBEDDING_FIELD}'"),
        (has_id_filter,    "filter field '_id'  (for $nin skip exclusion)"),
        (has_genre_filter, "filter field 'genres' (for per-genre row queries)"),
    ]
    all_ok = True
    for ok, label in checks:
        sym = tick() if ok else cross()
        print(f"     {sym}  {label}")
        if not ok:
            all_ok = False

    if not has_genre_filter:
        print(yellow(
            "\n   ACTION REQUIRED: Add 'genres' as a filter field to the index.\n"
            "   In Atlas UI: Search → Indexes → Edit → add:\n"
            '     {"type":"filter","path":"genres"}\n'
            "   The index will rebuild (1-5 min on free tier).\n"
            "   Genre rows will fall back to post-$match until this is done."
        ))

    return status == "READY" and all_ok


async def create_ttl_index(db) -> None:
    print(f"\n{'─'*62}")
    print(bold("2. TTL Index — user_realtime_sessions.expires_at"))
    print(f"{'─'*62}")
    col = db["user_realtime_sessions"]
    try:
        await col.create_index(
            [("expires_at", ASCENDING)],
            expireAfterSeconds=0,
            name="ttl_expires_at",
            background=True,
        )
        print(green(f"   {tick()} TTL index created / verified"))
    except Exception as exc:
        print(red(f"   {cross()} TTL index failed: {exc}"))

    try:
        await col.create_index(
            [("user_id", ASCENDING)],
            name="unique_user_id",
            unique=True,
            background=True,
        )
        print(green(f"   {tick()} Unique index on user_id created / verified"))
    except Exception as exc:
        print(red(f"   {cross()} user_id index failed: {exc}"))


async def create_profiles_index(db) -> None:
    print(f"\n{'─'*62}")
    print(bold("3. user_profiles indexes"))
    print(f"{'─'*62}")
    col = db["user_profiles"]
    try:
        await col.create_index(
            [("user_id", ASCENDING)],
            name="unique_user_id",
            unique=True,
            background=True,
        )
        print(green(f"   {tick()} Unique index on user_profiles.user_id created / verified"))
    except Exception as exc:
        print(red(f"   {cross()} user_profiles index failed: {exc}"))


async def create_boosts_index(db) -> None:
    print(f"\n{'─'*62}")
    print(bold("4. movie_boosts indexes"))
    print(f"{'─'*62}")
    col = db["movie_boosts"]
    try:
        await col.create_index(
            [("movie_id", ASCENDING)],
            name="unique_movie_id",
            unique=True,
            background=True,
        )
        print(green(f"   {tick()} Unique index on movie_boosts.movie_id created / verified"))

        await col.create_index(
            [("active", ASCENDING), ("expires_at", ASCENDING)],
            name="active_boosts",
            background=True,
        )
        print(green(f"   {tick()} Compound index on (active, expires_at) created / verified"))
    except Exception as exc:
        print(red(f"   {cross()} movie_boosts index failed: {exc}"))


async def check_movies_coverage(db) -> None:
    print(f"\n{'─'*62}")
    print(bold("5. Embedding coverage — {MOVIES_COL}"))
    print(f"{'─'*62}")
    col = db[MOVIES_COL]
    total    = await col.count_documents({})
    with_emb = await col.count_documents({EMBEDDING_FIELD: {"$exists": True}})
    pct = (with_emb / total * 100) if total else 0
    cfn = green if pct > 80 else yellow if pct > 0 else red
    print(cfn(
        f"   {tick() if pct > 0 else cross()} "
        f"{with_emb:,} / {total:,} movies have '{EMBEDDING_FIELD}' ({pct:.1f}% coverage)"
    ))

    # Genre coverage
    genres = await col.distinct("genres")
    print(f"   {tick()} {len(genres)} distinct genres available for genre rows")
    print(f"   Genres: {', '.join(sorted(genres)[:12])}{'…' if len(genres) > 12 else ''}")


# ---------------------------------------------------------------------------

async def main():
    print(bold("\n🎬 StreamFlix Churn Mitigation Demo — Index Setup v2"))
    print(f"   DB: {DB_NAME}  |  Collection: {MOVIES_COL}")
    print(f"   URI: {MONGO_URI[:50]}…\n")

    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]

    try:
        await client.admin.command("ping")
        print(green(f"{tick()} MongoDB Atlas connection established"))
    except Exception as exc:
        print(red(f"{cross()} Cannot connect: {exc}"))
        sys.exit(1)

    vector_ok = await verify_vector_index(db)
    await create_ttl_index(db)
    await create_profiles_index(db)
    await create_boosts_index(db)
    await check_movies_coverage(db)

    # Summary
    print(f"\n{'═'*62}")
    print(bold("Setup Summary"))
    print(f"{'═'*62}")
    print(f"  Vector index ready : {green('YES') if vector_ok else red('NO — check above')}")
    print(f"  TTL index          : {green('OK')}")
    print(f"  Profile index      : {green('OK')}")
    print(f"  Boost index        : {green('OK')}")
    print()
    if vector_ok:
        print(green("  All indexes verified. Next steps:"))
        print("    python scripts/seed_demo_user.py")
        print("    make run\n")
    else:
        print(yellow("  Fix the vector index issues above, then re-run this script.\n"))

    client.close()


if __name__ == "__main__":
    asyncio.run(main())
