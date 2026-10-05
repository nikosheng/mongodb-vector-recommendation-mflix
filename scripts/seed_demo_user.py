"""
scripts/seed_demo_user.py
=========================
Seeds a demo user profile with a Horror / Action interest bias.
The user_vector is stored in COMPOSITE space:

    user_vector = normalize(
        λ · embed(seed_genres_string, type="document")
        + (1-λ) · embed(profile_text,  type="document")
    )

This ensures the user_vector is in the same composite space as the stored
composite_embedding field and the Rocchio query vectors — all comparisons
are in the same space.

Run after generate_composite_embeddings.py and index creation:
    make seed
"""
import asyncio
import os
import sys
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

MONGO_URI       = os.environ["MONGO_URI"]
DB_NAME         = os.environ.get("DB_NAME", "sample_mflix")
VOYAGE_API_KEY  = os.environ["VOYAGE_API_KEY"]
VOYAGE_BASE_URL = os.environ.get("VOYAGE_BASE_URL", "https://ai.mongodb.com/v1")
VOYAGE_MODEL    = os.environ.get("VOYAGE_MODEL", "voyage-4")
USER_ID         = os.environ.get("DEMO_USER_ID", "demo_user")
LAMBDA          = float(os.environ.get("GENRE_BLEND_LAMBDA", "0.35"))

# Interest profile text — biased toward Horror, Thriller, Action, Crime, Mystery
PROFILE_TEXT = (
    "I love intense psychological horror films, dark thrillers, and action-packed "
    "suspense movies. My favourite genres are horror, thriller, action, and crime drama. "
    "I enjoy films with twist endings, supernatural elements, serial killers, survival "
    "scenarios, dark atmosphere, and adrenaline-driven plots. Directors like Hitchcock, "
    "Fincher, Kubrick, and Nolan. Films like The Shining, Silence of the Lambs, "
    "Se7en, The Dark Knight, Psycho, Get Out, Hereditary, and No Country for Old Men."
)

SEED_GENRES = ["Horror", "Thriller", "Action", "Crime", "Mystery"]
# The genre label string used for the composite blend
SEED_GENRES_STR = " ".join(sorted(SEED_GENRES))   # "Action Crime Horror Mystery Thriller"

def green(s):  return f"\033[32m{s}\033[0m"
def yellow(s): return f"\033[33m{s}\033[0m"
def red(s):    return f"\033[31m{s}\033[0m"
def bold(s):   return f"\033[1m{s}\033[0m"


def float_list_to_bson_binary(floats: list[float]) -> Binary:
    """Encode list[float] as BSON Vector Binary (subtype 9, float32)."""
    arr     = np.array(floats, dtype=np.float32)
    header  = bytes([0x27, 0x00])
    payload = arr.tobytes()
    return Binary(header + payload, subtype=9)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via Voyage AI (document type)."""
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
    print(f"\n   Calling Voyage AI: {url}")
    print(f"   Model: {VOYAGE_MODEL}  |  input_type: document  |  batch: {len(texts)}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload, headers=headers)

    if response.status_code != 200:
        raise RuntimeError(f"Voyage API {response.status_code}: {response.text}")

    data = response.json()
    results = [item["embedding"] for item in data["data"]]
    print(f"   ✓ Received {len(results)} embeddings, dim={len(results[0])}")
    return results


async def main():
    print(bold("\n🌱 StreamFlix — Seed Demo User (v3 — composite space)"))
    print(f"{'═'*64}")
    print(f"  User ID      : {USER_ID}")
    print(f"  Genres       : {', '.join(SEED_GENRES)}")
    print(f"  Genres str   : '{SEED_GENRES_STR}'")
    print(f"  λ            : {LAMBDA}  (composite = λ·genre + (1-λ)·plot)")
    print(f"  Voyage       : {VOYAGE_BASE_URL}  |  model={VOYAGE_MODEL}")
    print(f"  MongoDB      : {DB_NAME}")
    print(f"{'═'*64}")

    client = AsyncIOMotorClient(MONGO_URI, tlsCAFile=certifi.where())
    db     = client[DB_NAME]

    try:
        await client.admin.command("ping")
        print(green("\n✓ MongoDB Atlas connected"))
    except Exception as exc:
        print(red(f"\n✗ Cannot connect: {exc}"))
        sys.exit(1)

    # Step 1 — Embed BOTH genre label string AND profile text
    print(bold("\n[1/3] Embedding genre labels + profile text via Voyage AI"))
    print(f"   Genre string : '{SEED_GENRES_STR}'")
    print(f"   Profile text : '{PROFILE_TEXT[:80]}…'")

    try:
        embeddings = await embed_texts([SEED_GENRES_STR, PROFILE_TEXT])
    except Exception as exc:
        print(red(f"\n✗ Embedding failed: {exc}"))
        client.close()
        sys.exit(1)

    genre_vec   = np.array(embeddings[0], dtype=np.float32)
    profile_vec = np.array(embeddings[1], dtype=np.float32)
    genre_vec   /= np.linalg.norm(genre_vec)
    profile_vec /= np.linalg.norm(profile_vec)

    # Step 2 — Compute composite user_vector
    composite = LAMBDA * genre_vec + (1.0 - LAMBDA) * profile_vec
    composite /= np.linalg.norm(composite)

    print(f"\n   cos(genre_vec, profile_vec)    = {np.dot(genre_vec, profile_vec):.4f}")
    print(f"   cos(genre_vec, composite)      = {np.dot(genre_vec, composite):.4f}")
    print(f"   cos(profile_vec, composite)    = {np.dot(profile_vec, composite):.4f}")
    print(f"   composite L2 norm              = {np.linalg.norm(composite):.6f}  (must be 1.0)")

    composite_binary = float_list_to_bson_binary(composite.tolist())
    total_bytes = len(bytes(composite_binary))
    print(f"   Encoded: {len(composite)} dims → {total_bytes} bytes (Binary subtype 9)")

    # Step 3 — Upsert user_profiles
    print(bold("\n[2/3] Upserting user_profiles"))
    profiles_col = db["user_profiles"]
    doc = {
        "user_id":        USER_ID,
        "user_vector":    composite_binary,          # BSON Binary subtype 9, composite space
        "profile_text":   PROFILE_TEXT,
        "genres":         SEED_GENRES,
        "play_count":     0,
        "created_at":     datetime.now(timezone.utc),
        "updated_at":     datetime.now(timezone.utc),
        "seeded_by":      "seed_demo_user.py v3",
        "vector_format":  "composite_bson_subtype9_float32",
        "lambda":         LAMBDA,
    }
    result = await profiles_col.replace_one({"user_id": USER_ID}, doc, upsert=True)
    if result.upserted_id:
        print(green(f"   ✓ INSERTED  (id: {result.upserted_id})"))
    else:
        print(green(f"   ✓ UPDATED   (matched: {result.matched_count})"))

    # Step 4 — Clear session
    print(bold("\n[3/3] Clearing demo session"))
    sessions_col = db["user_realtime_sessions"]
    result2 = await sessions_col.delete_one({"user_id": USER_ID})
    if result2.deleted_count > 0:
        print(yellow(f"   ↻ Cleared existing session for '{USER_ID}'"))
    else:
        print(f"   — No existing session to clear")

    # Verify
    print(bold("\n[Verification]"))
    stored = await profiles_col.find_one({"user_id": USER_ID}, {"user_vector": 1, "genres": 1, "lambda": 1})
    if stored:
        raw = stored["user_vector"]
        b = bytes(raw)
        d = b[2:] if (len(b)-2)%4==0 and len(b)%4!=0 else b
        arr = np.frombuffer(d, dtype=np.float32)
        norm = np.linalg.norm(arr)
        print(green(f"   ✓ user_vector: {len(arr)} dims  L2 norm={norm:.6f}  λ={stored.get('lambda')}"))
        print(f"   genres: {stored.get('genres')}")
    else:
        print(red("   ✗ Profile not found after upsert"))

    print(f"\n{'═'*64}")
    print(green(bold("✓ Seed complete — user_vector is in composite space.")))
    print()
    print("  Next:")
    print("    make run   (starts server, warms genre cache, serves UI)")
    print()
    print("  Demo flow:")
    print("    1. Load page → Horror/Thriller/Action rows from composite profile")
    print("    2. Skip 2 Action movies → Action row disappears, Rocchio fires")
    print("       (genre-aware: Drama penalty reduced from 0.96 → 0.82)")
    print("    3. Play 2 Comedy movies → 'Because you watched Comedy' row appears")
    print("    4. Skip + Play together → bi-directional Rocchio: rocchio_full")
    print("    5. ↺ Reset Demo to replay\n")

    client.close()


if __name__ == "__main__":
    asyncio.run(main())
