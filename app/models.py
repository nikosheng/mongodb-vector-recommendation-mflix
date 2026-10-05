"""
Pydantic v2 schemas for request/response validation and serialization.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Inbound
# ---------------------------------------------------------------------------

class EventIn(BaseModel):
    """A single user interaction event sent from the frontend."""
    user_id: str = Field(..., description="Unique user identifier")
    movie_id: str = Field(..., description="MongoDB ObjectId as hex string")
    event_type: Literal["skip", "abandon", "play", "complete"] = Field(
        ..., description="Type of interaction"
    )
    genre: Optional[str] = Field(None, description="Row-aware genre (row genre for genre rows, primary genre for RFY)")
    # All genres to attribute for play events. For RFY rows this is movie.genres (all genres).
    # For genre rows this is [rowGenre] (single-element). Falls back to [genre] if absent.
    all_genres: Optional[List[str]] = Field(None, description="All genres to attribute (multi-genre attribution for play events)")
    # Which trail row this event originated from: 'rfy' or 'genre:Horror' etc.
    trail: Optional[str] = Field(None, description="Trail row identifier")


class ResetRequest(BaseModel):
    """Payload for the demo reset endpoint."""
    user_id: str = Field(default="demo_user")


# ---------------------------------------------------------------------------
# Boost admin
# ---------------------------------------------------------------------------

class BoostIn(BaseModel):
    """Payload to create or update a movie boost."""
    movie_id: str = Field(..., description="MongoDB ObjectId of the movie to boost")
    boost_score: float = Field(
        default=1.5, ge=1.0, le=10.0,
        description="Sort key among boosted movies. Higher = closer to slot 0."
    )
    label: str = Field(default="Hot", description="Hot | Trending | Sponsored")
    expires_hours: Optional[int] = Field(None, description="Auto-expire after N hours.")


class BoostOut(BaseModel):
    """A boost record as returned by the admin endpoints."""
    id: str
    movie_id: str
    title: Optional[str] = None
    boost_score: float
    label: str
    active: bool
    created_at: datetime
    expires_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Internal / service layer
# ---------------------------------------------------------------------------

class NegativeEvent(BaseModel):
    """A skip / abandon interaction stored in the sliding window session."""
    movie_id: str
    embedding: List[float] = Field(
        default_factory=list,
        description="plot_embedding_voyage_4 as list[float] for Rocchio math",
    )
    genre: Optional[str] = None
    trail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PositiveEvent(BaseModel):
    """A play / complete interaction stored in the sliding window session."""
    movie_id: str
    embedding: List[float] = Field(
        default_factory=list,
        description="plot_embedding_voyage_4 decoded from BSON Binary subtype 9",
    )
    genre: Optional[str] = None
    trail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SessionState(BaseModel):
    """
    Real-time session state returned to the frontend for the telemetry HUD.
    Covers both negative (skip/abandon) and positive (play/complete) signals
    within the current sliding window.
    """
    user_id: str
    # Negative signals
    negative_count: int = 0
    threshold: int = 3
    mitigation_active: bool = False
    penalized_genre: Optional[str] = None
    negative_events: List[NegativeEvent] = Field(default_factory=list)
    genre_skip_counts: Dict[str, int] = Field(default_factory=dict)
    swapped_genres: List[str] = Field(default_factory=list)
    rocchio_genre: Optional[str] = None
    # Positive signals
    positive_count: int = 0
    positive_events: List[PositiveEvent] = Field(default_factory=list)
    positive_genre_counts: Dict[str, int] = Field(default_factory=dict)
    # Shared
    window_seconds: Optional[int] = None   # None = no time window (accumulate until reset)
    # Combined query source (set by trails router)
    query_source: str = "user_vector"


# ---------------------------------------------------------------------------
# Outbound — movies
# ---------------------------------------------------------------------------

class MovieOut(BaseModel):
    """Movie document as returned to the frontend."""
    id: str = Field(..., description="MongoDB ObjectId as hex string")
    title: str
    genres: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    plot: Optional[str] = None
    poster: Optional[str] = None
    vector_score: float = Field(0.0, description="Atlas vectorSearchScore (cosine similarity)")
    # Boost metadata (None when not boosted)
    boost_score: Optional[float] = None
    boost_label: Optional[str] = None
    is_boosted: bool = False

    @field_validator("genres", mode="before")
    @classmethod
    def coerce_genres(cls, v):
        if v is None:
            return []
        return v

    @field_validator("year", mode="before")
    @classmethod
    def coerce_year(cls, v):
        if isinstance(v, dict):
            return int(next(iter(v.values()), 0))
        if v is None:
            return None
        return int(v)


# ---------------------------------------------------------------------------
# Outbound — trail responses
# ---------------------------------------------------------------------------

class GenreRow(BaseModel):
    """One horizontal genre row returned by the trails endpoint."""
    genre: str
    movies: List[MovieOut]
    row_index: int
    swapped_from: Optional[str] = None
    rocchio_applied: bool = False
    # "profile" | "swapped" | "played" | "emerging" | "discovery"
    row_source: str = "profile"


class TrailsResponse(BaseModel):
    """Full multi-row response from GET /api/trails/{user_id}."""
    recommended_for_you: List[MovieOut]
    genre_rows: List[GenreRow]
    session: SessionState
    rocchio_applied: bool = False
    # Possible values:
    #   "user_vector"           — no signals, baseline only
    #   "rocchio_positive"      — only play signals active
    #   "rocchio_genre:{genre}" — per-genre skip Rocchio (+ any play signals)
    #   "rocchio_global"        — global skip Rocchio   (+ any play signals)
    #   "rocchio_full"          — both positive plays AND negative skips active
    query_source: str = "user_vector"
    active_genre_skips: Dict[str, int] = Field(default_factory=dict)
    positive_genre_counts: Dict[str, int] = Field(default_factory=dict)
    profile_genres: List[str] = Field(default_factory=list)
    # Genres accumulating play counts but not yet visible in active genre rows.
    # Shown in the HUD "Emerging Interests" section so the audience can see
    # preference evolution in real time before the rows change.
    emerging_genre_counts: Dict[str, int] = Field(default_factory=dict)
    beta_effective: float = 0.0   # actual β used this request (BETA × play_count)
    # ── Live Profile Vector HUD ──────────────────────────────────────────
    # Both projected onto the same fixed 8-genre radar axes (RADAR_AXES in
    # vector_projection.py) via cosine similarity, so the frontend can
    # render two overlaid radar polygons:
    #   user_vector_projection — the persistent, slow-drifting baseline
    #     profile (updated async by drift_user_profile/reject_user_profile).
    #   rfy_query_projection   — the ephemeral per-request RFY query vector
    #     (orthogonal rejection + positive attraction applied fresh here).
    user_vector_projection: Dict[str, float] = Field(default_factory=dict)
    rfy_query_projection: Dict[str, float] = Field(default_factory=dict)
    # Cosine-distance shift between the two vectors above: 1 - cos(user, rfy).
    # 0.0 = query identical to baseline profile; larger = more visible shift.
    vector_shift: float = 0.0


class RecommendationResponse(BaseModel):
    """Full response from GET /api/recommendations/{user_id} (legacy single-row)."""
    movies: List[MovieOut]
    session: SessionState
    rocchio_applied: bool = False
    query_source: Literal["user_vector", "rocchio"] = "user_vector"


class EventResponse(BaseModel):
    """Response from POST /api/events."""
    success: bool = True
    session: SessionState
    message: str = ""
    trigger_refresh: bool = False
    trigger_reason: Optional[str] = None


class ResetResponse(BaseModel):
    """Response from POST /api/demo/reset."""
    success: bool = True
    user_id: str
    message: str
    session: SessionState
