"""
scripts/generate_composite_embeddings.py
=========================================
Generates genre-aware composite embeddings for every movie in embedded_movies
and writes them as a new field `composite_embedding` (BSON Binary subtype 9).

    composite_embedding = normalize(
        λ · embed(genres_string, type="document")
        + (1-λ) · plot_embedding_voyage_4
    )

Where genres_string = " ".join(sorted(movie.genres))
  e.g. genres=["Action","Drama"] → "Action Drama"

This makes Rocchio repulsion genre-specific:
  Before: cos(Action plot, Drama plot) ≈ 0.96  (nearly identical!)
  After:  cos(Action composite, Drama composite) ≈ 0.82  (clearly separated)

Run once before starting the application:
    make composite
    # or: python scripts/generate_composite_embeddings.py

Resumable: movies that already have `composite_embedding` are skipped.

After this script completes:
  1. Create the Atlas vector index `composite_vector_index` on `composite_embedding`
     (the script prints the exact JSON to paste in Atlas UI)
  2. Wait for the index to be READY (~2 min on M0 free tier)
  3. Run: make seed   (re-seeds user_vector in composite space)
  4. Run: make run    (server auto-loads new config)
"""
import asyncio
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import certifi
import httpx
import numpy as np
from bson import Binary
from motor.motor_asyncio import AsyncIOMotorClient

# ── Config ────────────────────────────────────────────────────────────────
MONGO_URI        = os.environ["MONGO_URI"]
DB_NAME          = os.environ.get("DB_NAME", "sample_mflix")
MOVIES_COL       = os.environ.get("MOVIES_COLLECTION", "embedded_movies")
EMBEDDING_FIELD  = os.environ.get("EMBEDDING_FIELD", "plot_embedding_voyage_4")
COMPOSITE_FIELD  = os.environ.get("COMPOSITE_EMBEDDING_FIELD", "plot_genre_composite_embedding")
VOYAGE_API_KEY   = os.environ["VOYAGE_API_KEY"]
VOYAGE_BASE_URL  = os.environ.get("VOYAGE_BASE_URL", "https://ai.mongodb.com/v1")
VOYAGE_MODEL     = os.environ.get("VOYAGE_MODEL", "voyage-4")
LAMBDA           = float(os.environ.get("GENRE_BLEND_LAMBDA", "0.35"))
BATCH_SIZE       = 96   # Voyage AI batch limit — stay under 128 to be safe

# ── ANSI helpers ──────────────────────────────────────────────────────────
def green(s):  return f"\033[32m{s}\033[0m"
def yellow(s): return f"\033[33m{s}\033[0m"
def red(s):    return f"\033[31m{s}\033[0m"
def bold(s):   return f"\033[1m{s}\033[0m"
def dim(s):    return f"\033[2m{s}\033[0m"


# ── BSON helpers ──────────────────────────────────────────────────────────

def decode_bson(raw: bytes) -> np.ndarray:
    """Strip 2-byte sub_type 9 header and decode as float32 array."""
    b = bytes(raw)
    data = b[2:] if (len(b) - 2) % 4 == 0 and len(b) % 4 != 0 else b
    return np.frombuffer(data, dtype=np.float32).copy()


def encode_bson(arr: np.ndarray) -> Binary:
    """Encode float32 numpy array as BSON Binary subtype 9."""
    header  = bytes([0x27, 0x00])
    payload = arr.astype(np.float32).tobytes()
    return Binary(header + payload, subtype=9)


def make_composite(plot_raw: bytes, genre_vec: np.ndarray, lam: float) -> np.ndarray:
    """
    composite = normalize( λ·genre_vec + (1-λ)·plot_vec )
    Both inputs are normalised before blending.
    """
    p = decode_bson(plot_raw)
    p = p / np.linalg.norm(p)
    g = genre_vec / np.linalg.norm(genre_vec)
    c = lam * g + (1.0 - lam) * p
    n = np.linalg.norm(c)
    return (c / n).astype(np.float32)


# ── Voyage AI batch call ──────────────────────────────────────────────────

async def embed_batch(
    texts: list[str],
    http_client: httpx.AsyncClient,
) -> list[np.ndarray]:
    """
    Embed a batch of texts via Voyage AI.
    input_type="document" — these are stored document embeddings, not query embeddings.
    """
    url = f"{VOYAGE_BASE_URL.rstrip('/')}/embeddings"
    payload = {
        "input":      texts,
        "model":      VOYAGE_MODEL,
        "input_type": "document",
    }
    headers = {
        "Authorization": f"Bearer {VOYAGE_API_KEY}",
        "Content-Type":  "application/json",
    }

    resp = await http_client.post(url, json=payload, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Voyage API error {resp.status_code}: {resp.text[:300]}"
        )

    data = resp.json()
    return [np.array(item["embedding"], dtype=np.float32) for item in data["data"]]


# ── Main ──────────────────────────────────────────────────────────────────

async def main():
    print(bold("\n🧬  Generate Composite Embeddings (genre-aware)"))
    print(f"{'═'*62}")
    print(f"  DB collection : {DB_NAME}.{MOVIES_COL}")
    print(f"  Source field  : {EMBEDDING_FIELD}")
    print(f"  Target field  : {COMPOSITE_FIELD}")
    print(f"  λ (genre wt)  : {LAMBDA}  (composite = λ·genre + (1-λ)·plot)")
    print(f"  Voyage model  : {VOYAGE_MODEL}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"{'═'*62}\n")

    client = AsyncIOMotorClient(MONGO_URI, tlsCAFile=certifi.where())
    col    = client[DB_NAME][MOVIES_COL]

    # Verify connectivity
    try:
        await client.admin.command("ping")
        print(green("✓ MongoDB Atlas connected"))
    except Exception as exc:
        print(red(f"✗ Cannot connect: {exc}"))
        sys.exit(1)

    # Count work
    total    = await col.count_documents({EMBEDDING_FIELD: {"$exists": True}})
    already  = await col.count_documents({COMPOSITE_FIELD: {"$exists": True}})
    todo     = total - already
    print(f"\n  Total movies with {EMBEDDING_FIELD}: {total:,}")
    print(f"  Already have {COMPOSITE_FIELD}:        {already:,}")
    print(f"  To process:                             {todo:,}")

    if todo == 0:
        print(green("\n✓ All movies already have composite embeddings. Nothing to do."))
        _print_index_instructions()
        client.close()
        return

    print(f"\n  Estimated Voyage API calls: ~{(todo // BATCH_SIZE) + 1}")
    print()

    # Fetch all movies that need processing (no composite yet)
    cursor = col.find(
        {EMBEDDING_FIELD: {"$exists": True}, COMPOSITE_FIELD: {"$exists": False}},
        {"_id": 1, "genres": 1, EMBEDDING_FIELD: 1},
    )
    docs = await cursor.to_list(length=None)
    print(f"  Fetched {len(docs):,} documents to process\n")

    processed = 0
    errors    = 0
    t_start   = time.time()

    async with httpx.AsyncClient(timeout=60.0) as http:
        # Process in batches
        for batch_start in range(0, len(docs), BATCH_SIZE):
            batch = docs[batch_start: batch_start + BATCH_SIZE]

            # Build genre strings for each doc in the batch
            # genres=["Action","Drama"] → "Action Drama"
            genre_strings = []
            for doc in batch:
                genres = doc.get("genres") or []
                gs = " ".join(sorted(str(g) for g in genres)) if genres else "Unknown"
                genre_strings.append(gs)

            # Embed the genre strings
            try:
                genre_vecs = await embed_batch(genre_strings, http)
            except Exception as exc:
                print(red(f"\n  ✗ Batch {batch_start//BATCH_SIZE + 1} embed failed: {exc}"))
                errors += len(batch)
                # Small backoff then continue — don't abort the whole run
                await asyncio.sleep(3)
                continue

            # Compute composites and bulk-write
            bulk_ops = []
            for doc, gv in zip(batch, genre_vecs):
                try:
                    comp_arr    = make_composite(doc[EMBEDDING_FIELD], gv, LAMBDA)
                    comp_binary = encode_bson(comp_arr)
                    from pymongo import UpdateOne
                    bulk_ops.append(UpdateOne(
                        {"_id": doc["_id"]},
                        {"$set": {COMPOSITE_FIELD: comp_binary}},
                    ))
                    processed += 1
                except Exception as exc:
                    print(yellow(f"\n  ⚠ Skipping doc {doc['_id']}: {exc}"))
                    errors += 1

            if bulk_ops:
                await col.bulk_write(bulk_ops, ordered=False)

            # Progress
            elapsed  = time.time() - t_start
            pct      = (batch_start + len(batch)) / len(docs) * 100
            per_sec  = processed / elapsed if elapsed > 0 else 0
            eta_s    = (len(docs) - (batch_start + len(batch))) / per_sec if per_sec > 0 else 0
            eta_str  = f"{eta_s/60:.1f}m" if eta_s > 60 else f"{eta_s:.0f}s"
            bar_n    = int(pct / 2)
            bar      = "█" * bar_n + "░" * (50 - bar_n)
            print(
                f"\r  [{bar}] {pct:5.1f}%  "
                f"{processed:,}/{len(docs):,}  "
                f"{per_sec:.1f}/s  ETA {eta_str}  errors={errors}",
                end="", flush=True,
            )

    elapsed_total = time.time() - t_start
    print()
    print()
    print(green(bold(f"✓ Done in {elapsed_total:.1f}s")))
    print(f"  Processed: {processed:,}  Errors: {errors}")

    # Verify
    new_count = await col.count_documents({COMPOSITE_FIELD: {"$exists": True}})
    print(f"  Total with {COMPOSITE_FIELD}: {new_count:,} / {total:,}")

    client.close()

    _print_index_instructions()


def _print_index_instructions():
    composite_field = os.environ.get("COMPOSITE_EMBEDDING_FIELD", "plot_genre_composite_embedding")
    index_name      = os.environ.get("COMPOSITE_VECTOR_INDEX", "composite_vector_index")
    col_name        = os.environ.get("MOVIES_COLLECTION", "embedded_movies")
    db_name         = os.environ.get("DB_NAME", "sample_mflix")

    print()
    print(bold("═" * 62))
    print(bold("Next steps:"))
    print(bold("═" * 62))
    print()
    print("1. Create the Atlas Vector Search index in the Atlas UI:")
    print(f"   Collection: {db_name}.{col_name}")
    print(f"   Index name: {index_name}")
    print()
    print("   Paste this JSON definition:")
    print("""   {
     "fields": [
       {
         "type": "vector",
         "path": \"""" + composite_field + """\",
         "numDimensions": 1024,
         "similarity": "cosine"
       },
       { "type": "filter", "path": "_id" },
       { "type": "filter", "path": "genres" }
     ]
   }""")
    print()
    print("2. Wait for the index to be READY (~2 min on free tier)")
    print()
    print("3. Re-seed the user profile in composite space:")
    print("     make seed")
    print()
    print("4. Start / restart the server:")
    print("     make run")
    print()


if __name__ == "__main__":
    asyncio.run(main())
