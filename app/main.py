"""
FastAPI application entry point.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import database
from app.config import settings
from app.routers import demo, events, recommendations
from app.routers import trails as trails_router
from app.routers import admin as admin_router
from app.routers import profile_stream as profile_stream_router
from app.services import genre_embedding_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await database.startup()

    # Warm the genre label embedding cache (single Voyage API batch call)
    # Used to compute composite negative embeddings on-the-fly for skip events
    await genre_embedding_cache.warm_up()

    logger.info(
        "StreamFlix started — DB=%s composite_index=%s λ=%.2f α=%.2f β=%.2f "
        "neg_threshold=%d genre_threshold=%d genre_cache=%d",
        settings.DB_NAME, settings.COMPOSITE_VECTOR_INDEX,
        settings.GENRE_BLEND_LAMBDA, settings.ALPHA, settings.BETA,
        settings.NEGATIVE_THRESHOLD, settings.GENRE_SKIP_THRESHOLD,
        genre_embedding_cache.cache_size(),
    )
    yield
    await database.shutdown()


app = FastAPI(
    title="StreamFlix — Churn Mitigation Demo",
    description=(
        "Genre-aware composite embeddings + bi-directional Rocchio + "
        "multi-row trails + admin boost injection."
    ),
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="templates")

app.include_router(events.router,               prefix="/api")
app.include_router(recommendations.router,      prefix="/api")
app.include_router(trails_router.router,        prefix="/api")
app.include_router(admin_router.router,         prefix="/api")
app.include_router(demo.router,                 prefix="/api")
app.include_router(profile_stream_router.router, prefix="/api")


@app.get("/api/health", tags=["health"])
async def health() -> dict:
    try:
        await database.get_client().admin.command("ping")
        db_status = "connected"
    except Exception as exc:
        db_status = f"error: {exc}"
    return {
        "status": "ok",
        "db": db_status,
        "config": {
            "db_name":                  settings.DB_NAME,
            "movies_collection":        settings.MOVIES_COLLECTION,
            "composite_field":          settings.COMPOSITE_EMBEDDING_FIELD,
            "composite_index":          settings.COMPOSITE_VECTOR_INDEX,
            "genre_blend_lambda":       settings.GENRE_BLEND_LAMBDA,
            "plot_field":               settings.EMBEDDING_FIELD,
            "rejection_strength":        settings.REJECTION_STRENGTH,
            "alpha":                    settings.ALPHA,
            "beta":                     settings.BETA,
            "beta_max_multiplier":      settings.BETA_MAX_MULTIPLIER,
            "play_row_threshold":       settings.PLAY_ROW_THRESHOLD,
            "drift_rate":               settings.DRIFT_RATE,
            "genre_skip_threshold":     settings.GENRE_SKIP_THRESHOLD,
            "negative_threshold":       settings.NEGATIVE_THRESHOLD,
            "session_mode":             "accumulate_until_reset",
            "genre_cache_warmed":       genre_embedding_cache.is_warmed(),
            "genre_cache_size":         genre_embedding_cache.cache_size(),
            "genre_rows_count":         settings.GENRE_ROWS_COUNT,
            "max_boosted_slots":        settings.MAX_BOOSTED_SLOTS,
        },
    }


@app.get("/", include_in_schema=False)
async def index(request: Request):
    # First entry in DEMO_USERS is the default persona shown on page load
    default_user = settings.DEMO_USERS.split(",")[0].strip()
    return templates.TemplateResponse(
        "index.html",
        {
            "request":             request,
            "api_base":            "/api",
            "default_user":        default_user,
            "demo_users":          settings.DEMO_USERS,
            "negative_threshold":  settings.NEGATIVE_THRESHOLD,
            "genre_skip_threshold":settings.GENRE_SKIP_THRESHOLD,
            "rejection_strength":  settings.REJECTION_STRENGTH,
            "alpha":               settings.ALPHA,
            "beta":                settings.BETA,
            "beta_max_multiplier": settings.BETA_MAX_MULTIPLIER,
            "play_row_threshold":  settings.PLAY_ROW_THRESHOLD,
            "drift_rate":          settings.DRIFT_RATE,
            "genre_blend_lambda":  settings.GENRE_BLEND_LAMBDA,
            "genre_rows_count":    settings.GENRE_ROWS_COUNT,
            "max_boosted_slots":   settings.MAX_BOOSTED_SLOTS,
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )
