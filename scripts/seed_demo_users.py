"""
scripts/seed_demo_users.py
==========================
Seeds ALL demo persona profiles in a single run.

Three personas with distinct genre preferences:
  demo_horror  — Horror / Thriller / Mystery / Crime / Drama
  demo_comedy  — Comedy / Romance / Family / Animation / Music
  demo_scifi   — Sci-Fi / Action / Adventure / Fantasy / Thriller

Each user_vector is stored in COMPOSITE space:

    user_vector = normalize(
        λ · embed(seed_genres_string, type="document")
        + (1-λ) · embed(profile_text,  type="document")
    )

All 6 texts (3 genre strings + 3 profile texts) are embedded in a single
batched Voyage AI call to minimise API round-trips.

Run once after generate_composite_embeddings.py and index creation:
    make seed-all
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
LAMBDA          = float(os.environ.get("GENRE_BLEND_LAMBDA", "0.35"))

# ---------------------------------------------------------------------------
# Persona definitions
# ---------------------------------------------------------------------------

PERSONAS = [
    {
        "user_id":      "demo_horror",
        "display_name": "Horror Fan",
        "genres":       ["Horror", "Thriller", "Mystery", "Crime", "Drama"],
        "profile_text": (
            "I love intense psychological horror films, dark thrillers, and crime dramas. "
            "My favourites include serial killer mysteries, supernatural horror, ghost stories, "
            "and atmospheric suspense films. I enjoy movies with twist endings, unreliable "
            "narrators, and dark atmosphere. Directors like Hitchcock, Fincher, Kubrick, and "
            "Ari Aster. Films like The Shining, Silence of the Lambs, Se7en, Get Out, "
            "Hereditary, Psycho, No Country for Old Men, and Midsommar."
        ),
    },
    {
        "user_id":      "demo_comedy",
        "display_name": "Comedy Fan",
        "genres":       ["Comedy", "Romance", "Family", "Animation", "Music"],
        "profile_text": (
            "I enjoy feel-good romantic comedies, animated family films, and light-hearted "
            "musicals. I love movies that make me laugh, cry happy tears, or leave me "
            "smiling. I prefer uplifting stories, witty dialogue, and heart-warming romance. "
            "Directors like Nora Ephron, Judd Apatow, and Pixar's creative team. Films like "
            "When Harry Met Sally, Toy Story, La La Land, The Grand Budapest Hotel, Crazy "
            "Rich Asians, Mamma Mia, Bridesmaids, and About Time."
        ),
    },
    {
        "user_id":      "demo_scifi",
        "display_name": "Sci-Fi Fan",
        "genres":       ["Sci-Fi", "Action", "Adventure", "Fantasy", "Thriller"],
        "profile_text": (
            "I am obsessed with science fiction, space exploration, and futuristic worlds. "
            "I love superhero films, tech-noir thrillers, dystopian stories, and epic fantasy "
            "adventures. I enjoy spectacular visual effects, complex world-building, and "
            "big-idea storytelling. Directors like Nolan, Villeneuve, and the Russo Brothers. "
            "Films like Interstellar, Blade Runner 2049, The Matrix, Avengers: Endgame, Dune, "
            "Mad Max: Fury Road, Arrival, and 2001: A Space Odyssey."
        ),
    },
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def green(s):  return f"\033[32m{s}\033[0m"
def yellow(s): return f"\033[33m{s}\033[0m"
def red(s):    return f"\033[31m{s}\033[0m"
def bold(s):   return f"\033[1m{s}\033[0m"


def float_list_to_bson_binary(floats: list[float]) -> Binary:
    """Encode list[float] as BSON Vector Binary (subtype 9, float32)."""
    arr    = np.array(floats, dtype=np.float32)
    header = bytes([0x27, 0x00])
    return Binary(header + arr.tobytes(), subtype=9)


async def embed_batch(texts: list[str]) -> list[list[float]]:
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
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        raise RuntimeError(f"Voyage API {response.status_code}: {response.text[:300]}")
    data = response.json()
    results = [item["embedding"] for item in data["data"]]
    print(green(f"   Received {len(results)} embeddings, dim={len(results[0])}"))
    return results


def make_composite(genre_emb: np.ndarray, profile_emb: np.ndarray, lam: float) -> np.ndarray:
    """normalize(λ·genre_vec + (1-λ)·profile_vec)"""
    g = genre_emb / np.linalg.norm(genre_emb)
    p = profile_emb / np.linalg.norm(profile_emb)
    composite = lam * g + (1.0 - lam) * p
    return (composite / np.linalg.norm(composite)).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    print(bold("\n🎬 StreamFlix — Seed All Demo Personas"))
    print(f"{'═'*64}")
    for p in PERSONAS:
        genres_str = " ".join(sorted(p["genres"]))
        print(f"  {p['display_name']:15s} ({p['user_id']})  genres: {genres_str}")
    print(f"\n  λ={LAMBDA}  model={VOYAGE_MODEL}  db={DB_NAME}")
    print(f"{'═'*64}")

    client = AsyncIOMotorClient(MONGO_URI, tlsCAFile=certifi.where())
    db     = client[DB_NAME]

    try:
        await client.admin.command("ping")
        print(green("\n✓ MongoDB Atlas connected"))
    except Exception as exc:
        print(red(f"\n✗ Cannot connect: {exc}"))
        client.close()
        sys.exit(1)

    # Step 1 — Build one big text batch: [genre_str_0, genre_str_1, genre_str_2,
    #                                     profile_0,   profile_1,   profile_2]
    print(bold(f"\n[1/3] Embedding {len(PERSONAS)*2} texts in one Voyage AI batch…"))
    genre_texts   = [" ".join(sorted(p["genres"])) for p in PERSONAS]
    profile_texts = [p["profile_text"]               for p in PERSONAS]
    all_texts     = genre_texts + profile_texts

    for i, p in enumerate(PERSONAS):
        print(f"  [{i}] genres : '{genre_texts[i]}'")
        print(f"  [{i}] profile: '{profile_texts[i][:70]}…'")

    try:
        embeddings = await embed_batch(all_texts)
    except Exception as exc:
        print(red(f"\n✗ Embedding failed: {exc}"))
        client.close()
        sys.exit(1)

    n = len(PERSONAS)
    genre_embs   = [np.array(embeddings[i],   dtype=np.float32) for i in range(n)]
    profile_embs = [np.array(embeddings[n+i], dtype=np.float32) for i in range(n)]

    # Step 2 — Compute composite vectors
    print(bold(f"\n[2/3] Computing composite user_vectors (λ={LAMBDA})…"))
    composites = []
    for i, p in enumerate(PERSONAS):
        c = make_composite(genre_embs[i], profile_embs[i], LAMBDA)
        composites.append(c)
        print(f"  {p['display_name']:15s}: norm={np.linalg.norm(c):.6f}  "
              f"cos(genre,profile)={np.dot(genre_embs[i]/np.linalg.norm(genre_embs[i]), profile_embs[i]/np.linalg.norm(profile_embs[i])):.4f}")

    # Step 3 — Upsert user_profiles + clear sessions
    print(bold(f"\n[3/3] Upserting {n} user_profiles + clearing sessions…"))
    profiles_col = db["user_profiles"]
    sessions_col = db["user_realtime_sessions"]
    now          = datetime.now(timezone.utc)

    for i, p in enumerate(PERSONAS):
        doc = {
            "user_id":        p["user_id"],
            "display_name":   p["display_name"],
            "user_vector":    float_list_to_bson_binary(composites[i].tolist()),
            "profile_text":   p["profile_text"],
            "genres":         p["genres"],
            "play_count":     0,
            "skip_reject_count": 0,
            "created_at":     now,
            "updated_at":     now,
            "seeded_by":      "seed_demo_users.py v1",
            "vector_format":  "composite_bson_subtype9_float32",
            "lambda":         LAMBDA,
        }
        result = await profiles_col.replace_one({"user_id": p["user_id"]}, doc, upsert=True)
        tag    = green("INSERTED") if result.upserted_id else green("UPDATED ")
        print(f"  {tag}  {p['user_id']:15s} ({p['display_name']})")

        r2 = await sessions_col.delete_one({"user_id": p["user_id"]})
        if r2.deleted_count:
            print(f"    ↻ Session cleared for {p['user_id']}")

    # Verification
    print(bold("\n[Verification]"))
    for p in PERSONAS:
        stored = await profiles_col.find_one(
            {"user_id": p["user_id"]},
            {"user_vector": 1, "genres": 1, "lambda": 1, "display_name": 1},
        )
        if stored:
            raw = stored["user_vector"]
            b   = bytes(raw)
            d   = b[2:] if (len(b)-2) % 4 == 0 and len(b) % 4 != 0 else b
            arr = np.frombuffer(d, dtype=np.float32)
            print(green(f"  ✓ {p['user_id']:15s}  dims={len(arr)}  "
                        f"norm={np.linalg.norm(arr):.6f}  "
                        f"genres={stored.get('genres')}"))
        else:
            print(red(f"  ✗ {p['user_id']} not found after upsert"))

    print(f"\n{'═'*64}")
    print(green(bold("✓ All personas seeded.  Run 'make run' to start the server.")))
    print()
    print("  Personas available:")
    for p in PERSONAS:
        print(f"    {p['user_id']:15s} → {p['display_name']}  {p['genres'][:3]}")
    print()

    client.close()


if __name__ == "__main__":
    asyncio.run(main())
