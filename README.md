# StreamFlix — Real-Time Preference Evolution Demo

A production-ready PoC demonstrating how MongoDB Atlas Vector Search + Voyage AI can power
a Netflix-style recommendation engine that **evolves in real time** as users watch and skip
content — including multi-persona switching, live genre row displacement, two-step preference
discovery, and persistent profile drift.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  Browser — Netflix-themed SPA (Tailwind CSS + Vanilla JS)                             │
│                                                                                       │
│  ┌──────────────┐  ┌─────────────────────────────────────────────────────────────┐   │
│  │ Hero Banner  │  │ ✨ Recommended For You                    query: rfy_full    │   │
│  │ (organic #1) │  │  [🔥 Boosted] [movie] [movie] ...                           │   │
│  └──────────────┘  └─────────────────────────────────────────────────────────────┘   │
│                    ┌─────────────────────────────────────────────────────────────┐   │
│                    │ ▶ Because you watched Action  (played-movie centroid query)  │   │
│                    └─────────────────────────────────────────────────────────────┘   │
│                    ┌─────────────────────────────────────────────────────────────┐   │
│                    │ ✨ You might like Horror  (user_vector + genre filter)       │   │
│                    └─────────────────────────────────────────────────────────────┘   │
│                    ┌─────────────────────────────────────────────────────────────┐   │
│                    │ 😂 Comedy  (profile row, user_vector + genre filter)         │   │
│                    └─────────────────────────────────────────────────────────────┘   │
│                    ┌─────────────────────────────────────────────────────────────┐   │
│                    │ 🔭 Discover Something New  (genre exclusion query)           │   │
│                    └─────────────────────────────────────────────────────────────┘   │
│                                                                                       │
│  ┌──────────────────────────────────┐  Persona Switcher (nav dropdown):              │
│  │ ⚡ Churn Mitigation HUD          │  ┌────────────────────────────────────────┐    │
│  │ Global Skips → RFY shift ██ 2/3 │  │ Horror Fan | Comedy Fan | Sci-Fi Fan   │    │
│  │ Per-Genre Skips:                 │  │ (each with genre tags, click to switch) │    │
│  │   Action  ████ 2/2 ⚡ row gone  │  └────────────────────────────────────────┘    │
│  │ ✦ Positive Signals:              │                                                │
│  │   Action  ████ 2 plays           │                                                │
│  │ 🌱 Emerging Interests:           │                                                │
│  │   Horror  ██░░ 1/2 plays         │                                                │
│  └──────────────────────────────────┘                                                │
└──────────────────────────────────────────────────────────────────────────────────────┘
         │  GET /api/trails/{user_id}           POST /api/events
         │  POST /api/demo/reset                GET /api/profile/stream/{user_id} (SSE)
         │  POST /api/admin/boost               GET /api/demo/personas
         ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  FastAPI + Motor (async Python)                                                       │
│                                                                                       │
│  routers/trails.py           ← orchestrator: RFY + weighted genre slots + discovery  │
│  routers/events.py           ← skip/play logging, multi-genre attribution            │
│  routers/demo.py             ← session reset + user_vector restore, persona list     │
│  routers/profile_stream.py   ← SSE push for real-time profile update notification    │
│  routers/admin.py            ← boost CRUD                                            │
│                                                                                       │
│  services/session_service.py        event store, drift, rejection, SSE notify        │
│  services/vector_math.py            orthogonal rejection, Rocchio, blend_composite   │
│  services/recommendation_service.py $vectorSearch pipelines, genre penalty, discovery│
│  services/genre_embedding_cache.py  genre label embeddings (warmed at startup)       │
│  services/boost_service.py          boost map + injection                            │
│  services/profile_notifier.py       in-process asyncio.Queue for SSE pub/sub        │
└──────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  MongoDB Atlas — sample_mflix                                                         │
│                                                                                       │
│  embedded_movies    plot_embedding_voyage_4            (BinData subtype 9, 1024-dim) │
│                     plot_genre_composite_embedding     (λ·genre + (1-λ)·plot)        │
│                     composite_vector_index  (cosine, filter: _id + genres)           │
│                     vector_index_voyage4    (fallback, filter: _id + genres)         │
│                                                                                       │
│  user_profiles      user_vector  (BinData subtype 9, composite space, evolves)       │
│                     genres: ["Comedy","Romance","Family","Animation","Music"]         │
│                     display_name, profile_text, lambda                                │
│                     play_count, skip_reject_count  (observability)                   │
│                                                                                       │
│  user_realtime_sessions   events[]          (composite neg embeddings, multi-genre)  │
│                           positive_events[] (composite pos embeddings, multi-genre)  │
│                           expires_at (TTL 24h — DB hygiene only)                     │
│                                                                                       │
│  movie_boosts       boost_score, label, active, expires_at                           │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Design Principles

### 1. Genre-Aware Composite Embeddings

All stored movie embeddings **and** the user profile vector live in **composite space**:

```
composite = normalize( λ · embed(genres_str) + (1-λ) · plot_embedding )
                       ↑ genre label signal    ↑ plot semantic signal
                       λ = 0.35 (GENRE_BLEND_LAMBDA)
```

**Why this matters:** In pure plot space, all genre centroids are nearly identical
(`cos(Action, Drama) = 0.96`). Skipping Action penalises Drama almost as strongly.

| Genre pair | Plot-only | Composite (λ=0.35) |
|---|---|---|
| Action vs Drama | 0.959 | 0.819 |
| Action vs Romance | 0.963 | 0.797 |
| Action vs Comedy | 0.915 | 0.761 |

With composite embeddings, skipping Action correctly targets the Action direction
without bleeding into Drama or Comedy.

`genres_str = " ".join(sorted(movie.genres))` — multi-genre movies (e.g. `["Action","Drama"]`)
get a joint embedding `"Action Drama"` that naturally interpolates between the two directions.

Genre label embeddings are cached in memory at startup (`genre_embedding_cache.py`) via a
single Voyage AI batch call — zero per-request overhead for all subsequent composite
computations.

---

### 2. RFY Query: Orthogonal Rejection + Bidirectional Rocchio

```
neg_dir    = normalize( mean(composite_embeddings_of_all_skipped_movies) )
projection = dot(user_vector, neg_dir)        # how much profile faces skipped direction
V_rfy      = normalize(
               user_vector − REJECTION_STRENGTH · projection · neg_dir   ← reject
               + β_eff · mean(composite_embeddings_of_played_movies)      ← attract
             )
```

**Why orthogonal rejection instead of simple Rocchio subtraction:**

The user profile vector points toward the user's taste (e.g. Comedy). Skipped movies also
point toward that taste. They are nearly parallel (`cos ≈ 0.79`). Simple Rocchio
`normalize(user_vec − 0.4·neg) ≈ user_vec` (cos = 0.96 — barely moves).

Orthogonal rejection removes *exactly* the skipped direction's component regardless of
alignment: `cos(user, V_rfy)` drops from 0.96 → **0.80**, shift = **0.64**. RFY content
visibly changes after 3 skips.

**Scaled β for positive attraction:**
```
β_eff = BETA × min(positive_count, BETA_MAX_MULTIPLIER)
        0.6  ×  min(3, 5)  =  1.8    (3 plays → strong attraction)
```

---

### 3. Dynamic Genre Penalty (Soft, Reversible)

After `$vectorSearch` returns candidates for RFY, a post-processing penalty is applied
to deprioritise heavily-skipped genres **without hard-banning them**:

```
net_affinity(genre) = play_count(genre) − skip_count(genre)

penalty_factor = 1.0 − GENRE_PENALTY_STRENGTH × clamp(|net_affinity| / MAX_SKIPS, 0, 1)
                 only applied when net_affinity < 0 (net skipped)

movie.vector_score *= penalty_factor    (per movie, worst genre across its genres list)
```

**Examples:**
| State | net_Action | penalty_factor | Effect |
|---|---|---|---|
| 3 skips, 0 plays | -3 | 1 - 0.4×0.6 = 0.76 | Action movies demoted |
| 3 skips, 2 plays | -1 | 1 - 0.4×0.2 = 0.92 | Penalty lifting |
| 3 skips, 3 plays | 0 | 1.0 | Fully restored |

The penalty lifts automatically as the user plays movies — no session reset needed. A larger
fetch pool (`RFY_FETCH_MULTIPLIER × trail_limit`) is fetched first so non-penalised movies
can float up after reranking.

---

### 4. Genre Row Weighted Slot Competition

Genre rows are not statically assigned from the seed profile. Genres compete for visible
slots by combined weight:

```
weight(genre) = seed_rank_weight(genre) + play_count(genre) × PLAY_GENRE_WEIGHT

seed_rank_weight:
  Comedy  (rank 0 in 5-genre profile) → weight 5
  Romance (rank 1)                    → weight 4
  Family  (rank 2)                    → weight 3
  Action  (non-seed, 3 plays)         → weight 0 + 3×1.0 = 3  ← ties Family!
```

After enough plays of a non-seed genre, it **displaces lower-ranked seed genres** from the
visible rows. This makes genre rows reflect the user's evolving taste, not just their initial
seed profile.

---

### 5. Two-Step Preference Discovery Row Progression

When a user watches movies outside their usual genres, a two-step progression appears:

**Step 1 — 1 play (below `PLAY_ROW_THRESHOLD`):**
→ **"✨ You might like [Genre]"** row appears
→ Query: `user_vector` + `$eq: genre` filter
→ Shows genre content compatible with the user's overall taste
→ HUD "🌱 Emerging Interests" bar shows progress toward threshold

**Step 2 — 2+ plays (≥ `PLAY_ROW_THRESHOLD`):**
→ **"▶ Because you watched [Genre]"** row appears (upgrades from "You might like")
→ Query: **centroid of the played movie embeddings** for that genre
→ Finds movies similar to the *specific films the user watched*, not generic genre content
→ Positive signals HUD section shows play count

This gives the demo audience a visible moment of "levelling up" as they watch more content.

**Bonus rows:** Genres that don't win a weighted main slot but have plays still appear as
bonus rows (up to `EMERGING_ROWS_COUNT = 2`), below the main rows.

---

### 6. Row Ordering — Fixed Priority

Regardless of weight, the page always renders in this order:

```
1. Recommended For You (RFY)
2. "Because you watched" rows  (played, ≥ threshold — centroid query)
3. "You might like" rows       (emerging, < threshold — user_vector query)
4. Profile / seed genre rows   (profile, user_vector query)
5. Discover Something New      (discovery, always last)
```

---

### 7. Discovery Row

A permanent row showing content **outside** the user's current taste bubble, designed to
give the demo audience something to click to trigger preference evolution.

**Two modes, configured per-user in `.env` via `DISCOVERY_GENRE_MAP`:**

**Fixed inclusion mode** (recommended for demos):
```
DISCOVERY_GENRE_MAP=demo_comedy:Action|Horror|Sci-Fi,...
```
- Atlas `$vectorSearch` filter: `{"$and": [{"genres": {"$in": ["Action","Horror","Sci-Fi"]}}, {"genres": {"$nin": seed_genres}}]}`
- Shows only movies that contain ≥1 discovery genre AND none of the user's seed genres
- A Comedy Fan's Discovery row shows Action/Horror/Sci-Fi — never Comedy or Romance

**Dynamic exclusion mode** (no `.env` entry):
- Atlas filter: `{"genres": {"$nin": active_genre_rows}}`
- Excludes movies from the user's current active genre rows

---

### 8. Multi-Genre Attribution

When a user plays or skips a movie with multiple genres (e.g. `["Action", "Comedy"]`):

**From RFY row:** all genres are attributed
```
all_genres = ["Action", "Comedy"]
→ positive_genre_counts["Action"] += 1
→ positive_genre_counts["Comedy"] += 1
```

**From a genre row** (e.g. Horror row): only the row's genre is attributed
```
trail = "genre:Horror" → genre = "Horror" only
```

This ensures that playing an Action+Comedy movie from the RFY row correctly credits both
genres, enabling both an "Action" and a "Comedy" preference signal to build simultaneously.

For skip events: RFY skips attribute all movie genres to `genre_skip_counts`; genre-row
skips attribute only the row genre.

**Deduplication:** `positive_count` (used for β scaling) counts distinct movies played, not
raw event entries. `get_played_ids()` deduplicates by `movie_id` so the exclude list isn't
bloated by multi-genre entries.

---

### 9. Persistent Profile Evolution — Two Paths

`user_profiles.user_vector` evolves asynchronously in two directions:

| Trigger | Direction | Rate | Function |
|---|---|---|---|
| Every play event | **Toward** played content | `DRIFT_RATE = 0.35` | `drift_user_profile()` |
| Global skip threshold first crossed | **Away from** skipped content | `SKIP_DRIFT_RATE = 0.5` once | `reject_user_profile()` |

Both fire as `asyncio.create_task()` — they never block the HTTP response.

**Drift math:**
```
new_vec = normalize( (1 − DRIFT_RATE) · user_vec + DRIFT_RATE · played_composite )
```
With `DRIFT_RATE = 0.35`: after 3 plays the profile has moved ~58% toward the new genre.

**Rejection math:**
```
neg_dir    = normalize( mean(all_session_neg_composites) )
projection = dot(user_vec, neg_dir)
new_vec    = normalize( user_vec − SKIP_DRIFT_RATE · projection · neg_dir )
```
Self-limiting: as `user_vec` moves away from `neg_dir`, `projection` shrinks → diminishing
effect on subsequent crossings.

**Demo reset restores the vector:** `POST /api/demo/reset` re-embeds the stored `profile_text`
and `genres` via Voyage AI and writes a fresh composite `user_vector`. This ensures every demo
run starts from a correctly aligned baseline regardless of how many drift/rejection cycles ran.

---

### 10. Real-Time Profile Update via SSE

When `drift_user_profile()` or `reject_user_profile()` successfully writes a new
`user_vector` to MongoDB, it pushes a notification to the frontend via **Server-Sent Events**:

```
Backend async task
  update_one(user_profiles, user_vector = new_vec)   ← MongoDB write
  notify_profile_updated(user_id, reason="drift")    ← push to asyncio.Queue

GET /api/profile/stream/{user_id}   ← persistent SSE connection from browser
  EventSource → 'profile_updated' event fires
  → fetchTrails() called immediately          ← no timer, no polling
```

`profile_notifier.py` maintains an in-process `asyncio.Queue` per user_id. The SSE endpoint
reads from the queue with a 15-second heartbeat timeout to keep the connection alive through
proxies and load balancers.

---

### 11. Multi-Persona Demo System

Three independent demo users, each with a distinct taste profile:

| User ID | Display Name | Seed Genres | Discovery Row |
|---|---|---|---|
| `demo_horror` | Horror Fan | Horror, Thriller, Mystery, Crime, Drama | Comedy, Romance, Animation |
| `demo_comedy` | Comedy Fan | Comedy, Romance, Family, Animation, Music | Action, Horror, Sci-Fi |
| `demo_scifi` | Sci-Fi Fan | Sci-Fi, Action, Adventure, Fantasy, Thriller | Comedy, Drama, Documentary |

The nav bar shows a compact dropdown selector. Clicking a persona:
1. Resets that persona's session and restores their `user_vector` via Voyage AI re-embedding
2. Clears all local HUD state
3. Reopens the SSE stream for the new user_id
4. Re-fetches trails — showing completely different RFY and genre rows

**Reset Demo** button silently resets **all** personas in parallel via `Promise.all`.

Each persona's user_vector is seeded as:
```
user_vector = normalize( λ · embed(genres_str) + (1-λ) · embed(profile_text) )
```
All seeded in a **single Voyage AI batch call** (6 texts: 3 genre strings + 3 profile texts)
to minimise API round-trips.

---

## Full Request Flow

### Page Load — `GET /api/trails/{user_id}`

```
Browser
  │  (SSE connection already open: GET /api/profile/stream/{user_id})
  │
  └─► GET /api/trails/demo_comedy
            │
            ├─ 1. Load user_profiles → user_vector, profile_genres, profile_text
            │
            ├─ 2. Load user_realtime_sessions (ALL events — no time filter)
            │       Derive negative_count, genre_skip_counts, swapped_genres,
            │       positive_genre_counts, positive_count
            │
            ├─ 3. Decide RFY query vector (_decide_rfy_query):
            │       if mitigation_active (neg ≥ NEGATIVE_THRESHOLD):
            │         V_rfy = orthogonal_rejection(user_vector, neg_embeddings)
            │                 + β_eff · mean(pos_embeddings)    query_source="rfy_full"
            │       elif pos_count > 0:
            │         V_rfy = user_vector + β_eff·mean(pos_embeddings) "rfy_positive"
            │       else:
            │         V_rfy = user_vector                               "user_vector"
            │
            ├─ 4. Compute weighted genre slots (_compute_weighted_genre_slots):
            │       weight(g) = seed_rank_weight(g) + play_count(g) × PLAY_GENRE_WEIGHT
            │       Top GENRE_ROWS_COUNT slots → main rows (played/profile/emerging)
            │       Bonus slots for non-seed genres with plays → up to EMERGING_ROWS_COUNT
            │
            ├─ 5. Sort slots by priority: played → emerging → profile → swapped
            │
            ├─ 6. asyncio.gather() — ALL $vectorSearch queries in parallel:
            │
            │   RFY:
            │     fetch trail_limit × RFY_FETCH_MULTIPLIER candidates
            │     → _apply_genre_penalty() reranks by net_affinity
            │     → return top trail_limit
            │
            │   "Because you watched" rows (played):
            │     centroid = mean(composite_embeddings_of_played_movies_in_genre)
            │     $vectorSearch(centroid, filter: genres=$eq:genre)
            │
            │   "You might like" rows (emerging):
            │     $vectorSearch(genre_query, filter: genres=$eq:genre)
            │
            │   Profile rows:
            │     $vectorSearch(genre_query, filter: genres=$eq:genre)
            │
            │   Discovery row:
            │     Fixed mode:   $vectorSearch(genre_query,
            │                     filter: {$and: [{genres:{$in:[...]}},
            │                                     {genres:{$nin:seed_genres}}]})
            │     Dynamic mode: $vectorSearch(genre_query,
            │                     filter: {genres:{$nin:active_genre_rows}})
            │
            ├─ 7. Boost injection into RFY top MAX_BOOSTED_SLOTS positions
            │
            ├─ 8. Compute emerging_genre_counts
            │       (genres with plays > 0 not in active rows → HUD sidebar)
            │
            └─► TrailsResponse {
                  recommended_for_you, genre_rows, session,
                  query_source, beta_effective,
                  positive_genre_counts, emerging_genre_counts,
                  profile_genres
                }
```

---

### Skip Event — `POST /api/events`

```
User skips movie with genres=["Action","Comedy"] from the RFY row
  │
  ├─ trail = "rfy"  →  all_genres = ["Action","Comedy"]  (multi-genre attribution)
  │   trail = "genre:Horror"  →  all_genres = ["Horror"]  (single genre)
  │
  └─► POST /api/events { event_type:"skip", genre:"Action", trail:"rfy",
                         all_genres:["Action","Comedy"] }
            │
            ├─ Fetch movie composite embedding from embedded_movies
            │
            ├─ Push one events[] entry per attributed genre:
            │     {movie_id, embedding, genre:"Action", trail, timestamp}
            │     {movie_id, embedding, genre:"Comedy", trail, timestamp}
            │   → genre_skip_counts["Action"] += 1
            │   → genre_skip_counts["Comedy"] += 1
            │
            ├─ negative_count = distinct movies skipped (deduped by movie_id)
            │
            ├─ On FIRST crossing of NEGATIVE_THRESHOLD:
            │   asyncio.create_task(reject_user_profile())    ← non-blocking
            │     neg_dir    = normalize(mean(neg_composites))
            │     projection = dot(user_vec, neg_dir)
            │     new_vec    = normalize(user_vec − 0.5·projection·neg_dir)
            │     update_one(user_vector = new_vec)
            │     notify_profile_updated("rejection")         ← SSE push
            │
            └─► EventResponse { trigger_refresh:true, session:{...} }

Frontend:
  syncSession() → HUD updates immediately
  scheduleRefetch(350ms) → fetchTrails() with session-level rejection
  [async] SSE 'profile_updated' fires when write completes → fetchTrails() again
          with updated persistent user_vector
```

---

### Play Event — `POST /api/events`

```
User plays movie with genres=["Action","Drama"] from RFY
  │
  └─► POST /api/events { event_type:"play", genre:"Action", trail:"rfy",
                         all_genres:["Action","Drama"] }
            │
            ├─ Fetch composite embedding (plot_genre_composite_embedding)
            │
            ├─ Push one positive_events[] entry per genre:
            │     {movie_id, embedding, genre:"Action", ...}
            │     {movie_id, embedding, genre:"Drama",  ...}
            │   → positive_genre_counts["Action"] += 1
            │   → positive_genre_counts["Drama"]  += 1
            │
            ├─ positive_count = distinct movies played (deduped by movie_id)
            │   (used for β_eff scaling — must not overcount multi-genre plays)
            │
            ├─ asyncio.create_task(drift_user_profile())    ← non-blocking
            │     new_vec = normalize((1-0.35)·user_vec + 0.35·played_composite)
            │     update_one(user_vector = new_vec)
            │     notify_profile_updated("drift")            ← SSE push
            │
            └─► trigger_refresh=True, trigger_reason="rfy_positive"

After 1 Action play:
  → positive_genre_counts["Action"] = 1 (< PLAY_ROW_THRESHOLD=2)
  → "✨ You might like Action" row appears  (emerging, user_vector query)
  → HUD: 🌱 Action 1/2 plays progress bar

After 2nd Action play:
  → positive_genre_counts["Action"] = 2 (≥ PLAY_ROW_THRESHOLD)
  → "▶ Because you watched Action" row appears (played, centroid query)
  → "You might like Action" row removed
```

---

### SSE Profile Stream — `GET /api/profile/stream/{user_id}`

```
Browser connects at page load:
  EventSource("/api/profile/stream/demo_comedy")
    │
    │  [persistent HTTP connection, server pushes events]
    │
    ├─ Every 15s: event:heartbeat → ignored (keeps connection alive)
    │
    └─ When profile write completes:
         event:profile_updated
         data: {"type":"profile_updated","reason":"drift"|"rejection"}
           │
           └─► JS: fetchTrails() immediately
                   (no 1500ms hack — push is guaranteed post-write)
```

---

## Algorithms

### Composite Embedding

```python
genres_str = " ".join(sorted(movie.genres))     # "Action Drama" for ["Action","Drama"]
genre_vec  = embed(genres_str, input_type="document")
plot_vec   = plot_embedding_voyage_4             # already stored in Atlas

composite = normalize( λ · genre_vec + (1-λ) · plot_vec )
            λ = GENRE_BLEND_LAMBDA = 0.35
```

User profile vector is seeded with the same formula:
```python
composite = normalize( λ · embed(genres_joined) + (1-λ) · embed(profile_text) )
```

Both must be in the same composite space for cosine similarity to be meaningful.

---

### RFY Orthogonal Rejection

```python
neg_matrix   = stack([composite_emb for each skipped movie])
neg_centroid = mean(neg_matrix, axis=0)
neg_dir      = neg_centroid / norm(neg_centroid)

projection   = dot(user_vector, neg_dir)         # scalar: alignment with skipped direction
adjusted     = user_vector − REJECTION_STRENGTH · projection · neg_dir
               + β_eff · mean(pos_matrix)         # optional positive attraction

V_rfy = normalize(adjusted)

# Measured effect (Horror Fan skipping Horror movies):
# cos(user_vec, neg_dir) = 0.78  ← profile and neg centroid nearly aligned
# cos(user_vec, V_rfy)   = 0.80  ← query vector has moved noticeably
# shift magnitude        = 0.43
```

---

### Genre Penalty Reranking

```python
# After $vectorSearch returns RFY_FETCH_MULTIPLIER × trail_limit candidates:
for movie in candidates:
    worst_net = min(
        plays(g) - skips(g) for g in movie.genres
    )
    if worst_net < 0:
        severity           = min(abs(worst_net) / GENRE_PENALTY_MAX_SKIPS, 1.0)
        penalty_factor     = 1.0 - GENRE_PENALTY_STRENGTH × severity
        movie.vector_score *= penalty_factor

candidates.sort(by=vector_score, descending=True)
return candidates[:trail_limit]
```

---

### Weighted Genre Slot Competition

```python
# Main slots (GENRE_ROWS_COUNT):
for genre in all_candidates:
    weight = seed_rank_weight.get(genre, 0) + positive_genre_counts.get(genre, 0) × PLAY_GENRE_WEIGHT
    # seed_rank_weight: Comedy(rank 0 in 5) → 5, Romance → 4, Family → 3, ...

top_slots = sorted(candidates, by=-weight)[:GENRE_ROWS_COUNT]

# Bonus slots (EMERGING_ROWS_COUNT) for non-seed genres with plays not in top_slots:
bonus = non_seed_genres_with_plays_not_in_top_slots
bonus.sort(by=-play_count)[:EMERGING_ROWS_COUNT]

# Tag source:
"played"   if genre not in seed AND plays ≥ PLAY_ROW_THRESHOLD
"emerging" if genre not in seed AND 0 < plays < PLAY_ROW_THRESHOLD
"profile"  if genre in seed
```

---

### Discovery Row Atlas Filters

**Fixed inclusion mode (`DISCOVERY_GENRE_MAP` configured):**
```python
{
  "$and": [
    {"genres": {"$in":  discovery_genres}},   # must contain ≥1 discovery genre
    {"genres": {"$nin": seed_genres}},         # must NOT contain any seed genre
  ]
}
# e.g. Comedy Fan: $in=["Action","Horror","Sci-Fi"] AND $nin=["Comedy","Romance","Family",...]
```

**Dynamic exclusion mode (no config entry):**
```python
{"genres": {"$nin": active_genre_row_genres}}  # excludes genres already in rows
```

Fallback chain on 0 results: drop seed exclusion → drop all genre filters.

---

## RFY Algorithm — Step by Step with Atlas Vector Search Integration

### Step 1 — Load the user vector (MongoDB read, not Atlas Search)

```
user_vector ∈ ℝ¹⁰²⁴  (composite space, L2-normalised)

Source: user_profiles.find_one({"user_id": user_id}, {"user_vector": 1})
Stored: BinData subtype 9 (2-byte header + IEEE-754 float32 payload)
Decoded: bson_binary_to_numpy() → strip 2-byte header → np.frombuffer(dtype=float32)
```

Re-read from MongoDB on every request — no application-level cache. The vector evolves
persistently between sessions via drift and rejection background writes.

---

### Step 2 — Build the negative centroid (pure Python / NumPy)

```python
# Source: user_realtime_sessions.find_one({"user_id": user_id})
neg_embeddings = [
    np.array(e["embedding"])          # composite float64 list, stored per skip event
    for e in session.events
    # deduplicated by movie_id — multi-genre skips store N entries per movie
]

neg_matrix   = np.stack(neg_embeddings, axis=0)   # shape: (k, 1024)
neg_centroid = neg_matrix.mean(axis=0)             # shape: (1024,)
neg_dir      = neg_centroid / np.linalg.norm(neg_centroid)   # unit vector
```

Each skip stores the movie's `plot_genre_composite_embedding` as `list[float]` in the
session document. `get_negative_embeddings()` deduplicates by `movie_id` before stacking
so a movie with genres `["Action","Drama"]` — which creates two session entries — contributes
its embedding only once to the centroid.

---

### Step 3 — Orthogonal rejection (pure Python / NumPy)

```python
projection = np.dot(user_vector, neg_dir)
             # scalar ∈ [0, 1]: how much profile faces the skipped direction
             # e.g. 0.78 for a Horror Fan skipping Horror movies

adjusted = user_vector - REJECTION_STRENGTH * projection * neg_dir
           # 0.8 × 0.78 × neg_dir removed from profile
```

**Why `projection × neg_dir` rather than a fixed subtraction:**
The removal scales with how aligned the profile actually is with the skipped content.
If the user's profile only weakly overlaps the skipped direction (`projection = 0.10`),
almost nothing is removed. Self-calibrating — no overcorrection on tangential skips.

**No Atlas call here.** This is a NumPy operation on two 1024-dim arrays taking < 1ms.

---

### Step 4 — Positive attraction (pure Python / NumPy)

```python
pos_embeddings = [
    bson_binary_to_numpy(e["embedding"])   # composite, BinData subtype 9
    for e in session.positive_events
    # deduplicated by movie_id (multi-genre plays store N entries per movie)
]

pos_centroid = np.stack(pos_embeddings).mean(axis=0)   # shape: (1024,)

β_eff = BETA * min(positive_count, BETA_MAX_MULTIPLIER)
      # positive_count = distinct movies played (not raw entry count)
      # 0.6 × min(3, 5) = 1.8 for 3 plays

adjusted += β_eff * pos_centroid
```

Added **after** rejection so the attraction operates in the already-rejection-shifted space.
`positive_count` uses distinct `movie_id` count — not raw `positive_events` length — to
prevent multi-genre plays from inflating β.

---

### Step 5 — Normalise to unit vector (pure Python / NumPy)

```python
V_rfy = l2_normalize(adjusted.astype(np.float32))
      = adjusted / np.linalg.norm(adjusted)
```

Result: a 1024-dim unit vector in composite space. Its direction encodes everything —
away from skipped genres, toward played genres, anchored near the original profile.

---

### Step 6 — `query_source` tagging (routing logic)

| Condition | `query_source` | Formula |
|---|---|---|
| No signals | `user_vector` | `V_rfy = user_vector` |
| Only plays | `rfy_positive` | `V_rfy = normalize(user_vector + β_eff·pos_centroid)` |
| Only skips ≥ threshold | `rfy_rejected` | `V_rfy = normalize(user_vector − strength·proj·neg_dir)` |
| Skips + plays | `rfy_full` | Both rejection and attraction applied |

---

### Step 7 — Atlas `$vectorSearch` (the Atlas integration point)

```python
# app/services/recommendation_service.py — _build_pipeline()

pipeline = [
    {
        "$vectorSearch": {
            "index":         "composite_vector_index",   # HNSW index on composite field
            "path":          "plot_genre_composite_embedding",
            "queryVector":   V_rfy.astype(np.float64).tolist(),   # list[float], 1024 dims
            "numCandidates": NUM_CANDIDATES,             # 200 — HNSW graph traversal depth
            "limit":         trail_limit * RFY_FETCH_MULTIPLIER,   # 20 × 3 = 60
            "filter": {
                "_id": {"$nin": exclude_ids}             # skip already-seen movies
            }
        }
    },
    {
        "$project": {
            "_id": 1, "title": 1, "genres": 1,
            "year": 1, "plot": 1, "poster": 1,
            "vectorSearchScore": {"$meta": "vectorSearchScore"}
        }
    }
]
```

**What Atlas does inside `$vectorSearch`:**

Atlas computes cosine similarity between `V_rfy` and every stored
`plot_genre_composite_embedding` using an **HNSW (Hierarchical Navigable Small World)**
approximate nearest-neighbour index. It does not scan all 3403 movies linearly — it
traverses the graph, considers `numCandidates = 200` documents, and returns the `limit = 60`
highest-cosine results.

Since both `V_rfy` and `plot_genre_composite_embedding` are L2-normalised unit vectors,
cosine similarity equals their dot product:

```
vectorSearchScore = dot(V_rfy, movie_composite)    (both are unit vectors)
                  ∈ [−1, 1]
```

A movie whose composite points in the same direction as `V_rfy` → score near 1.0.
A movie in the rejected direction → score near −1.0 or low positive.
The rejection step in Python already moved `V_rfy` away from the skipped genre direction,
so those movies naturally rank lower in Atlas cosine ranking — **without any genre filter**.

**The `$nin` pre-filter on `_id`:**
Atlas evaluates `{"_id": {"$nin": exclude_ids}}` **inside** the HNSW traversal — skipped
movie IDs are excluded during graph traversal, not post-scan. This requires `_id` to be
declared as `"type": "filter"` in the index definition. Without it, Atlas would throw an
error or fall back to a full scan.

**Why `limit = 60` instead of `limit = 20`:**
The next step (genre penalty) adjusts scores downward for net-skipped genres. Movies
initially ranked 21–60 by raw cosine may float into the top 20 after adjustment. Fetching
3× the final limit ensures the penalty reranker has enough candidates.

**Same index, different query vectors:**
Every row type (RFY, genre rows, BYW, discovery) uses the same `composite_vector_index`.
What differs is the `queryVector` and the `filter`:

| Row type | `queryVector` | `filter` |
|---|---|---|
| RFY | `V_rfy` (rejection-adjusted) | `$nin exclude_ids` |
| Profile/emerging row | `genre_query` (`V_rfy` or `user_vector`) | `$nin exclude_ids` AND `genres $eq genre` |
| "Because you watched" | centroid of played embeddings | `$nin exclude_ids` AND `genres $eq genre` |
| Discovery (fixed) | `genre_query` | `$nin exclude_ids` AND `genres $in discovery` AND `genres $nin seed` |
| Discovery (dynamic) | `genre_query` | `$nin exclude_ids` AND `genres $nin active_rows` |

All these queries run in `asyncio.gather()` — in parallel, not sequentially. The total
wall-clock time is the slowest single query, not the sum.

**Why composite space matters for Atlas:**
If `V_rfy` (built from composite embeddings) were sent to the plot-only index
(`vector_index_voyage4` with `plot_embedding_voyage_4`), the cosine scores would be
meaningless — the two spaces are different (`cos(composite, plot_only) ≈ 0.40` for the
same movie). The composite index and composite query vector must match.

The **fallback path** handles the case where `composite_vector_index` is not yet built:
`_run_pipeline()` detects 0 results or an index error and retries with `vector_index_voyage4`
and `plot_embedding_voyage_4` automatically. Genre separation is weaker (Action vs Drama
cos = 0.96 instead of 0.82) but the system functions.

---

### Step 8 — Genre penalty reranking (back in Python, not Atlas)

```python
# app/services/recommendation_service.py — _apply_genre_penalty()

for movie in candidates:     # 60 movies from Atlas
    worst_net = min(
        positive_genre_counts.get(g, 0) - genre_skip_counts.get(g, 0)
        for g in movie.genres
    )
    if worst_net < 0:
        severity           = min(abs(worst_net) / GENRE_PENALTY_MAX_SKIPS, 1.0)
        penalty_factor     = 1.0 - GENRE_PENALTY_STRENGTH * severity
        movie.vector_score *= penalty_factor

candidates.sort(key=lambda m: m.vector_score, reverse=True)
return candidates[:trail_limit]   # top 20
```

**Why this cannot be done inside Atlas:**
`genre_skip_counts` and `positive_genre_counts` are per-session, per-user, per-request data
stored in `user_realtime_sessions`. Atlas Vector Search has no access to this — it only
knows the static document fields in `embedded_movies`. The net affinity per genre changes
every time the user plays or skips something, so it must be evaluated in the application
layer after results arrive.

---

### Complete RFY pipeline (Python + Atlas + Python)

```
Python (NumPy):
  user_vector   ← find_one(user_profiles)                    ~50ms MongoDB read
  neg_centroid  ← find_one(user_realtime_sessions)           ~50ms MongoDB read
  neg_dir       = normalize(mean(neg_embeddings))            <1ms
  projection    = dot(user_vector, neg_dir)                  <1ms
  adjusted      = user_vector − 0.8 × projection × neg_dir  <1ms
                + β_eff × mean(pos_embeddings)               <1ms
  V_rfy         = normalize(adjusted)                        <1ms

Atlas HNSW (parallel gather):
  $vectorSearch(V_rfy,    composite_index, 60 candidates)    ~30ms  ← RFY
  $vectorSearch(genre_q,  composite_index, genre filter)     ~20ms  ← Comedy row
  $vectorSearch(centroid, composite_index, genre filter)     ~20ms  ← BYW row
  $vectorSearch(genre_q,  composite_index, discovery filter) ~20ms  ← Discovery
  ─────────────────────────────────────────────────────────────────
  asyncio.gather() wall time ≈ 30ms  (slowest query, not sum)

Python (post-processing):
  _apply_genre_penalty(60 candidates)                        <1ms
  sort + top 20                                              <1ms
  build TrailsResponse                                       <1ms

Total request latency: ~130–200ms on Atlas free tier
                       ~50–80ms  on Atlas dedicated M10+
```

---

## Prerequisites

- Python 3.11+
- MongoDB Atlas cluster with `sample_mflix` dataset
- Voyage AI API key (direct or Atlas AI gateway `https://ai.mongodb.com/v1`)
- Two Atlas Vector Search indexes on `embedded_movies` (see below)

---

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Fill in MONGO_URI and VOYAGE_API_KEY

# 3. Generate composite embeddings (one-time, ~30s for 3403 movies)
make composite

# 4. Create composite_vector_index in Atlas UI (JSON printed by make composite)
#    Wait for READY (~2 min)

# 5. Verify indexes + create TTL/boost indexes
make setup

# 6. Seed all three demo personas
make seed-all

# 7. Start server
make run     # foreground, hot-reload
make start   # background + open browser
```

### One-shot (first time)

```bash
make demo    # install → composite → seed-all → start → open browser
```

### Day-to-day

```bash
make start     # start in background + open browser
make stop      # kill background server
make reset     # clear all persona sessions (keeps profiles)
make seed-all  # re-seed all 3 personas (reset vectors to baseline)
make seed      # re-seed legacy demo_user only
make trails    # smoke-test /api/trails endpoint
make health    # check config + genre cache status
make logs      # tail server log
```

---

## Atlas Vector Search Indexes

Both indexes need `genres` as a filter field for genre row queries and discovery row
genre inclusion/exclusion filters.

### Primary — `composite_vector_index` on `embedded_movies`

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "plot_genre_composite_embedding",
      "numDimensions": 1024,
      "similarity": "cosine"
    },
    { "type": "filter", "path": "_id" },
    { "type": "filter", "path": "genres" }
  ]
}
```

### Fallback — `vector_index_voyage4` on `embedded_movies`

Used automatically while `composite_vector_index` is building:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "plot_embedding_voyage_4",
      "numDimensions": 1024,
      "similarity": "cosine"
    },
    { "type": "filter", "path": "_id" },
    { "type": "filter", "path": "genres" }
  ]
}
```

---

## Demo Flow

### Preference Evolution Story (Comedy Fan → discovers Action)

| Step | Action | What Happens |
|---|---|---|
| 1 | Load page | Comedy Fan loads. RFY shows Comedy/Romance/Fantasy content. Genre rows: Comedy · Romance · Family · Discover. `query_source: user_vector`. |
| 2 | Play 1 Action movie from Discovery | `positive_genre_counts["Action"]=1`. **"✨ You might like Action"** row appears. HUD: 🌱 Action 1/2. Vector drifts 35% toward Action. |
| 3 | Play 2nd Action movie | `positive_genre_counts["Action"]=2`. **"▶ Because you watched Action"** row replaces "You might like". Centroid query — shows movies similar to the specific films watched. RFY shifts toward Action content. |
| 4 | Play 5+ Action movies | Action wins a weighted slot, displacing Family from the profile rows. Genre rows: Comedy · **Action** (played) · Romance · Discover. RFY dominated by Action. |
| 5 | Skip 3 Action+Comedy movies from RFY | Global threshold: `rfy_rejected`. All movie genres attributed (`Action`, `Comedy`). Action row begins penalty. Async: user_vector pushed away from Action direction. SSE push → immediate re-fetch. |
| 6 | ↺ Reset Demo | All 3 personas reset silently in parallel. Each user_vector re-embedded from their seed profile via Voyage AI. Starts fresh. |

### Skip Mitigation Story (Horror Fan)

| Step | Action | What Happens |
|---|---|---|
| 1 | Skip 2 Horror movies from Horror row | `genre_skip_counts["Horror"]=2`. Horror row removed → Crime replaces it (skip-swapped). |
| 2 | Skip a 3rd movie (any genre) | Global threshold 3/3. `query_source: rfy_rejected`. RFY shifts away from all skipped content. Async: `user_vector` persistently updated (`SKIP_DRIFT_RATE=0.5`). SSE push fires when write completes. |
| 3 | Play 2 Comedy movies | "▶ Because you watched Comedy" row appears (centroid query). RFY attracted toward Comedy (`rfy_full`). |

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/trails/{user_id}` | Multi-row trail: RFY + genre rows + discovery + boost injection |
| `POST` | `/api/events` | Log skip/play. Multi-genre attribution. Async profile update. |
| `POST` | `/api/demo/reset` | Reset session + restore user_vector from seed for one user |
| `GET` | `/api/demo/personas` | List all demo personas with display_name, genres, profile excerpt |
| `GET` | `/api/profile/stream/{user_id}` | SSE: push `profile_updated` when user_vector write completes |
| `GET` | `/api/health` | Health + full config dump |
| `POST` | `/api/admin/boost` | Create / update a movie boost |
| `DELETE` | `/api/admin/boost/{movie_id}` | Deactivate a boost |
| `GET` | `/api/admin/boosts` | List active boosts |
| `GET` | `/api/admin/movies/search?q=` | Title search for boost picker |
| `GET` | `/api/docs` | OpenAPI / Swagger UI |

### POST /api/events — request body

```json
{
  "user_id":    "demo_comedy",
  "movie_id":   "573a1390f29313caabcd4323",
  "event_type": "play",
  "genre":      "Action",
  "trail":      "rfy",
  "all_genres": ["Action", "Drama"]
}
```

- `event_type`: `skip | abandon | play | complete`
- `trail`: `rfy` | `genre:Horror` | `genre:Comedy` | ...
- `genre`: the **row's** genre, not `movie.genres[0]`. Extracted from `trail` string in JS: `"genre:Horror".slice(6) → "Horror"`. For RFY: `movie.genres[0]`.
- `all_genres`: full movie genres list. For RFY plays/skips: all genres attributed. For genre-row plays/skips: `[rowGenre]` only.

### GET /api/trails — response shape (key fields)

```json
{
  "recommended_for_you": [...],
  "genre_rows": [
    { "genre": "Action",  "row_source": "played",   "rocchio_applied": false },
    { "genre": "Horror",  "row_source": "emerging",  "rocchio_applied": false },
    { "genre": "Comedy",  "row_source": "profile",   "rocchio_applied": true  },
    { "genre": "Discover Something New", "row_source": "discovery" }
  ],
  "session": {
    "negative_count": 3, "mitigation_active": true,
    "genre_skip_counts": { "Horror": 2, "Action": 1 },
    "swapped_genres": ["Horror"],
    "positive_count": 2,
    "positive_genre_counts": { "Action": 2, "Drama": 1 }
  },
  "query_source": "rfy_full",
  "beta_effective": 1.2,
  "emerging_genre_counts": { "Horror": 1 },
  "profile_genres": ["Comedy","Romance","Family","Animation","Music"]
}
```

`row_source` values:
| Value | Meaning | Query vector |
|---|---|---|
| `"played"` | ≥ PLAY_ROW_THRESHOLD plays — "Because you watched" | Centroid of played embeddings |
| `"emerging"` | 1 ≤ plays < threshold — "You might like" | `user_vector` + genre filter |
| `"profile"` | Seed genre row | `rfy_query` (or `user_vector` if no mitigation) |
| `"swapped"` | Seed genre was skip-removed, replaced | `rfy_query` |
| `"discovery"` | Always-last discovery row | Fixed/dynamic genre filter |

---

## Configuration Reference

All settings in `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `MONGO_URI` | — | Atlas connection string |
| `DB_NAME` | `sample_mflix` | Database name |
| `MOVIES_COLLECTION` | `embedded_movies` | Collection with embeddings |
| `VOYAGE_API_KEY` | — | Voyage AI API key |
| `VOYAGE_BASE_URL` | `https://ai.mongodb.com/v1` | Voyage endpoint (Atlas gateway or direct) |
| `VOYAGE_MODEL` | `voyage-4` | Embedding model (1024-dim) |
| `EMBEDDING_FIELD` | `plot_embedding_voyage_4` | Original plot embedding field |
| `COMPOSITE_EMBEDDING_FIELD` | `plot_genre_composite_embedding` | Genre-aware composite field |
| `COMPOSITE_VECTOR_INDEX` | `composite_vector_index` | Primary Atlas index name |
| `VECTOR_INDEX_NAME` | `vector_index_voyage4` | Fallback Atlas index name |
| `GENRE_BLEND_LAMBDA` | `0.35` | λ: genre weight in composite (0=plot-only, 1=genre-only) |
| `REJECTION_STRENGTH` | `0.8` | Ephemeral RFY rejection fraction per request |
| `ALPHA` | `0.4` | Rocchio α (used in positive-arm formula) |
| `BETA` | `0.6` | Base positive attraction weight per play |
| `BETA_MAX_MULTIPLIER` | `5` | β_eff cap: `BETA × min(play_count, 5)` |
| `DRIFT_RATE` | `0.35` | Profile drift per play toward played content (0.35 = visible in demo) |
| `SKIP_DRIFT_RATE` | `0.5` | Profile rejection on global threshold crossing (persistent, once) |
| `NEGATIVE_THRESHOLD` | `3` | Global skips to trigger RFY shift + persistent profile update |
| `GENRE_SKIP_THRESHOLD` | `2` | Same-genre skips to remove that genre's row |
| `PLAY_ROW_THRESHOLD` | `2` | Plays to upgrade "You might like" → "Because you watched" |
| `PLAY_GENRE_WEIGHT` | `1.0` | Each play's weight in genre slot competition (1 = 1 rank unit) |
| `EMERGING_ROWS_COUNT` | `2` | Max "You might like" bonus rows shown simultaneously |
| `GENRE_PENALTY_STRENGTH` | `0.4` | Max score multiplier reduction for net-skipped genres in RFY |
| `GENRE_PENALTY_MAX_SKIPS` | `5` | Net skip count at which penalty reaches full strength |
| `RFY_FETCH_MULTIPLIER` | `3` | Fetch `trail_limit × N` candidates before genre penalty reranking |
| `TRAIL_LIMIT` | `20` | Movies in RFY row |
| `GENRE_ROWS_COUNT` | `3` | Number of main genre row slots |
| `GENRE_ROW_LIMIT` | `12` | Movies per genre row |
| `NUM_CANDIDATES` | `200` | Atlas `numCandidates` for RFY and discovery rows |
| `GENRE_NUM_CANDIDATES` | `100` | Atlas `numCandidates` for genre rows |
| `MAX_BOOSTED_SLOTS` | `3` | Max boosted positions at front of RFY |
| `DEMO_USERS` | `demo_horror,demo_comedy,demo_scifi` | Comma-separated persona user IDs |
| `DISCOVERY_GENRE_MAP` | `""` | Per-user fixed Discovery genres: `user_id:Genre1\|Genre2,user_id2:...` |

---

## Project Structure

```
mongodb-mflix-behavior-analysis/
├── .env / .env.example
├── requirements.txt
├── Makefile
├── app/
│   ├── config.py                     pydantic-settings — all tunables + DISCOVERY_GENRE_MAP parser
│   ├── database.py                   Motor client + TTL/unique index creation
│   ├── main.py                       FastAPI app + genre cache warm-up + SSE router
│   ├── models.py                     Pydantic v2 schemas (EventIn.all_genres, TrailsResponse.emerging_genre_counts)
│   ├── services/
│   │   ├── vector_math.py            orthogonal rejection, Rocchio, blend_composite, BSON↔NumPy
│   │   ├── genre_embedding_cache.py  genre label embeddings + embed_texts_via_voyage()
│   │   ├── session_service.py        event store, drift, rejection, restore_user_vector, SSE notify
│   │   ├── recommendation_service.py $vectorSearch pipelines, genre penalty, get_discovery_row()
│   │   ├── boost_service.py          boost CRUD + injection
│   │   └── profile_notifier.py       asyncio.Queue per user for SSE pub/sub
│   ├── routers/
│   │   ├── trails.py                 weighted slot logic, discovery, emerging_genre_counts
│   │   ├── events.py                 multi-genre attribution, threshold detection
│   │   ├── profile_stream.py         GET /api/profile/stream/{user_id} SSE endpoint
│   │   ├── demo.py                   reset (restore vector + session), GET /api/demo/personas
│   │   ├── admin.py                  POST/DELETE/GET /api/admin/boost
│   │   └── recommendations.py        GET /api/recommendations (legacy single-row)
│   └── static/
│       ├── css/style.css             HUD, boost badge, admin modal, persona switcher, emerging bars
│       └── js/app.js                 trail rendering, persona switcher dropdown, SSE client,
│                                     multi-genre attribution, emerging HUD, discovery row labels
├── templates/
│   └── index.html                    Netflix-themed SPA (Tailwind CDN + DEMO_CONFIG injection)
└── scripts/
    ├── generate_composite_embeddings.py  batch embed + write plot_genre_composite_embedding
    ├── seed_demo_users.py                seed all 3 personas (single Voyage batch call)
    ├── seed_demo_user.py                 seed legacy demo_user
    ├── setup_indexes.py                  verify indexes, create TTL/boost indexes
    └── wait_for_server.py                health-check loop for make start
```
