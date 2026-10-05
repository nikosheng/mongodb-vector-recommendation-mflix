"""
Application configuration using pydantic-settings.
All settings are read from environment variables or .env file.
"""
from typing import Dict, List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # MongoDB
    MONGO_URI: str = "mongodb://localhost:27017"
    DB_NAME: str = "sample_mflix"

    # Voyage AI
    VOYAGE_API_KEY: str = ""
    VOYAGE_BASE_URL: str = "https://ai.mongodb.com/v1"
    VOYAGE_MODEL: str = "voyage-4"

    # ── Original plot-only embedding (kept as fallback) ──────────────────
    EMBEDDING_DIM: int = 1024
    VECTOR_INDEX_NAME: str = "vector_index_voyage4"
    EMBEDDING_FIELD: str = "plot_embedding_voyage_4"
    MOVIES_COLLECTION: str = "embedded_movies"

    # ── Composite embedding (genre-aware) ────────────────────────────────
    # Field written by scripts/generate_composite_embeddings.py
    COMPOSITE_EMBEDDING_FIELD: str = "plot_genre_composite_embedding"
    # Atlas vector index on composite_embedding (create manually after script runs)
    COMPOSITE_VECTOR_INDEX: str = "composite_vector_index"
    # λ: weight of the genre label signal vs plot signal
    # composite = normalize(λ·genre_vec + (1-λ)·plot_vec)
    # λ=0.35 → Action vs Drama separation: 0.959 (plot-only) → 0.819 (composite)
    GENRE_BLEND_LAMBDA: float = 0.35

    # RFY query vector — orthogonal rejection for global skips
    # Removes the skipped content's direction component from user_vector.
    # 1.0 = fully remove the skipped genre direction from profile
    # 0.8 = remove 80% — strong, visible shift without going to extremes
    # Much more effective than simple Rocchio subtraction (α=0.4) when
    # user_vector and neg_centroid are nearly aligned (common for on-profile skips).
    REJECTION_STRENGTH: float = 0.8

    # Rocchio — negative repulsion (kept for positive arm β calculation)
    ALPHA: float = 0.4

    # Rocchio — positive attraction (play signals)
    BETA: float = 0.6
    BETA_MAX_MULTIPLIER: int = 5

    # Genre row promotion from positive plays.
    # 2 = two-step progression:
    #   1 play  → "You might like [genre]"  (emerging row, user_vector query)
    #   2 plays → "Because you watched [genre]" (played row, centroid query)
    PLAY_ROW_THRESHOLD: int = 2

    # Background profile drift per play event (toward played content).
    # 0.35 = aggressive shift for demo visibility (vector moves ~50% after 3 plays).
    DRIFT_RATE: float = 0.35

    # How much each play of a genre contributes to displacing seed genre rows.
    # weight(genre) = seed_rank_weight + play_count × PLAY_GENRE_WEIGHT
    # 1.0 = each play is worth 1 rank unit (matches seed rank scale).
    PLAY_GENRE_WEIGHT: float = 1.0

    # Max number of "You might like [genre]" rows to show simultaneously.
    # These appear when a non-seed genre has ≥1 play but < PLAY_ROW_THRESHOLD plays.
    # They are extra rows beyond the GENRE_ROWS_COUNT main slots.
    EMERGING_ROWS_COUNT: int = 2

    # Persistent profile rejection on global skip threshold crossing.
    # Applied once (async) when negative_count first reaches NEGATIVE_THRESHOLD.
    # Removes this fraction of the skipped direction from user_profiles.user_vector.
    # Lower than REJECTION_STRENGTH (0.8) — persistent update should be conservative
    # so the profile doesn't overcorrect from a single session's skips.
    # 0.5 = removes 50% of the skipped direction from the persistent profile.
    SKIP_DRIFT_RATE: float = 0.5

    # Session tracking — NO time window (all events accumulate until explicit reset)
    # Only kept for document-level TTL (24h), not for event pruning
    NEGATIVE_THRESHOLD: int = 3
    GENRE_SKIP_THRESHOLD: int = 2

    # Trail / row layout
    TRAIL_LIMIT: int = 20
    GENRE_ROWS_COUNT: int = 3
    GENRE_ROW_LIMIT: int = 12
    NUM_CANDIDATES: int = 200
    GENRE_NUM_CANDIDATES: int = 100

    # Boost injection
    MAX_BOOSTED_SLOTS: int = 3

    # ── Dynamic genre penalty (soft, reversible) ─────────────────────────
    # Applied in get_recommendations() after $vectorSearch via score reranking.
    # Replaces the hard $nin genre exclusion filter so skipped genres are
    # deprioritised — not banned — and the penalty lifts automatically as
    # the user plays movies in those genres.
    #
    # penalty_factor = 1.0 - GENRE_PENALTY_STRENGTH × clamp(|net_skips| / MAX, 0, 1)
    #   net_skips(genre) = skip_count - play_count  (negative net = liked, no penalty)
    #
    # Example:  Action skip=3, play=0  → factor = 1 - 0.4×0.6 = 0.76
    #           Action skip=3, play=2  → factor = 1 - 0.4×0.2 = 0.92  (lifting)
    #           Action skip=3, play=3  → factor = 1.0              (fully restored)
    GENRE_PENALTY_STRENGTH: float = 0.4
    # net skip count at which penalty reaches full GENRE_PENALTY_STRENGTH
    GENRE_PENALTY_MAX_SKIPS: int = 5
    # get_recommendations fetches trail_limit × this factor before reranking
    # so penalised movies still enter the pool and non-penalised ones can float up
    RFY_FETCH_MULTIPLIER: int = 3

    # ── Demo persona registry ─────────────────────────────────────────────
    # Comma-separated list of all demo user IDs.
    # Must match the user_ids seeded by scripts/seed_demo_users.py.
    # The first entry is the default persona shown on page load.
    DEMO_USERS: str = "demo_horror,demo_comedy,demo_scifi"

    # ── Per-user Discovery row genre sets ────────────────────────────────
    # Controls what genres appear in each persona's "Discover Something New" row.
    #
    # Format:  user_id1:Genre1|Genre2|Genre3,user_id2:Genre1|Genre2
    #   - Pipe (|) separates genres within a user
    #   - Comma separates users
    #
    # Behaviour when a user IS listed (fixed inclusion mode):
    #   - Discovery row shows movies that contain ≥1 of the listed genres
    #   - AND do NOT contain any of the user's seed genres (profile_genres)
    #   - This ensures the row shows content genuinely outside the user's bubble
    #
    # Behaviour when a user is NOT listed (dynamic exclusion mode):
    #   - Discovery row excludes movies from the user's current active genre rows
    #   - Falls back to current default behaviour
    #
    # Example:
    #   demo_comedy:Action|Horror|Sci-Fi,demo_horror:Comedy|Romance|Animation
    DISCOVERY_GENRE_MAP: str = ""

    @property
    def discovery_genre_map(self) -> Dict[str, List[str]]:
        """
        Parse DISCOVERY_GENRE_MAP into {user_id: [genres]} dict.
        Returns empty dict if the setting is unset or blank.
        """
        if not self.DISCOVERY_GENRE_MAP.strip():
            return {}
        result: Dict[str, List[str]] = {}
        for entry in self.DISCOVERY_GENRE_MAP.split(","):
            entry = entry.strip()
            if ":" not in entry:
                continue
            user_id, genres_str = entry.split(":", 1)
            genres = [g.strip() for g in genres_str.split("|") if g.strip()]
            if user_id.strip() and genres:
                result[user_id.strip()] = genres
        return result

    # Legacy alias
    RECOMMENDATION_LIMIT: int = 20


# Singleton instance imported throughout the app
settings = Settings()
