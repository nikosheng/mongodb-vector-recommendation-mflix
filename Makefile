# ==============================================================================
# StreamFlix Churn Mitigation Demo — Makefile
# ==============================================================================
# Usage:
#   make install       Install Python dependencies
#   make setup         Verify Atlas indexes + create TTL/unique/boost indexes
#   make seed          Seed demo_user profile via Voyage AI
#   make run           Start the FastAPI server in foreground (hot-reload)
#   make start         Start server in background + open browser when ready
#   make demo          Full pipeline: install → setup → seed → start
#   make stop          Kill the background server
#   make open          Open the demo UI (server must already be running)
#   make reset         Clear demo_user session via the API
#   make health        Hit the health-check endpoint
#   make trails        Quick smoke-test the /api/trails endpoint
#   make boosts        List all active admin boosts
#   make logs          Tail the background server log
#   make clean         Remove __pycache__, .pyc files, and server.pid/log
#   make lint          Syntax-check all Python source files
# ==============================================================================

PYTHON     := python3
PIP        := $(PYTHON) -m pip
UVICORN    := $(PYTHON) -m uvicorn

HOST       := 0.0.0.0
PORT       := 8000
APP        := app.main:app
BASE_URL   := http://localhost:$(PORT)
DEMO_USER  := demo_user

# Background server bookkeeping
PID_FILE   := .server.pid
LOG_FILE   := .server.log

# Colour helpers
BOLD  := \033[1m
RED   := \033[31m
GREEN := \033[32m
YELLOW:= \033[33m
CYAN  := \033[36m
RESET := \033[0m

.DEFAULT_GOAL := help

# ------------------------------------------------------------------------------
# help
# ------------------------------------------------------------------------------
.PHONY: help
help:
	@echo ""
	@printf "$(BOLD)StreamFlix — Churn Mitigation Demo$(RESET)\n"
	@echo "──────────────────────────────────────────────────────"
	@printf "  $(CYAN)make install$(RESET)   Install Python dependencies\n"
	@printf "  $(CYAN)make setup$(RESET)     Verify Atlas vector indexes + create TTL indexes\n"
	@printf "  $(CYAN)make composite$(RESET) Generate composite embeddings for all movies\n"
	@printf "  $(CYAN)make seed$(RESET)      Seed demo_user profile in composite space\n"
	@printf "  $(CYAN)make run$(RESET)       Start server in foreground (Ctrl+C to stop)\n"
	@printf "  $(CYAN)make start$(RESET)     Start server in background + open browser\n"
	@printf "  $(CYAN)make demo$(RESET)      Full pipeline: install → setup → composite → seed → start\n"
	@printf "  $(CYAN)make stop$(RESET)      Kill the background server\n"
	@printf "  $(CYAN)make open$(RESET)      Open $(BASE_URL) in default browser\n"
	@printf "  $(CYAN)make reset$(RESET)     Clear demo_user session (keeps profile)\n"
	@printf "  $(CYAN)make health$(RESET)    Check API health endpoint\n"
	@printf "  $(CYAN)make trails$(RESET)    Smoke-test /api/trails (row counts + Rocchio status)\n"
	@printf "  $(CYAN)make boosts$(RESET)    List all active admin boosts\n"
	@printf "  $(CYAN)make logs$(RESET)      Tail background server log\n"
	@printf "  $(CYAN)make clean$(RESET)     Remove cache, pid, and log files\n"
	@printf "  $(CYAN)make lint$(RESET)      Syntax-check all Python source files\n"
	@echo "──────────────────────────────────────────────────────"
	@echo ""

# ------------------------------------------------------------------------------
# install
# ------------------------------------------------------------------------------
.PHONY: install
install:
	@printf "$(BOLD)Installing dependencies…$(RESET)\n"
	$(PIP) install -r requirements.txt
	@printf "$(GREEN)✓ Dependencies installed$(RESET)\n"

# ------------------------------------------------------------------------------
# setup
# ------------------------------------------------------------------------------
.PHONY: setup
setup:
	@printf "$(BOLD)Running index setup…$(RESET)\n"
	$(PYTHON) scripts/setup_indexes.py

# ------------------------------------------------------------------------------
# composite — generate genre-aware composite embeddings for all movies
# Must run before make seed and make run (first time only, then resumable)
# After this: create composite_vector_index in Atlas UI (script prints JSON)
# ------------------------------------------------------------------------------
.PHONY: composite
composite:
	@printf "$(BOLD)Generating composite embeddings (genre-aware)…$(RESET)\n"
	@printf "  λ = GENRE_BLEND_LAMBDA (default 0.35)\n"
	@printf "  composite = normalize(λ·genre_vec + (1-λ)·plot_vec)\n\n"
	$(PYTHON) scripts/generate_composite_embeddings.py

# ------------------------------------------------------------------------------
# seed  (legacy single-user)
# ------------------------------------------------------------------------------
.PHONY: seed
seed:
	@printf "$(BOLD)Seeding demo user profile…$(RESET)\n"
	$(PYTHON) scripts/seed_demo_user.py

# ------------------------------------------------------------------------------
# seed-all  — seed all three demo personas (horror / comedy / sci-fi)
# ------------------------------------------------------------------------------
.PHONY: seed-all
seed-all:
	@printf "$(BOLD)Seeding all demo personas (horror / comedy / sci-fi)…$(RESET)\n"
	$(PYTHON) scripts/seed_demo_users.py

# ------------------------------------------------------------------------------
# run — foreground server (hot-reload, blocks terminal)
# ------------------------------------------------------------------------------
.PHONY: run
run:
	@printf "$(BOLD)Starting FastAPI server at $(BASE_URL)$(RESET)\n"
	@printf "  Press $(BOLD)Ctrl+C$(RESET) to stop\n\n"
	$(UVICORN) $(APP) \
		--host $(HOST) \
		--port $(PORT) \
		--reload \
		--reload-dir app \
		--reload-dir templates \
		--log-level info

# ------------------------------------------------------------------------------
# start — launch server in background, wait until ready, then open browser
# ------------------------------------------------------------------------------
.PHONY: start
start: _kill_existing
	@printf "$(BOLD)Starting server in background…$(RESET)\n"
	@$(UVICORN) $(APP) \
		--host $(HOST) \
		--port $(PORT) \
		--reload \
		--reload-dir app \
		--reload-dir templates \
		--log-level info \
		> $(LOG_FILE) 2>&1 & echo $$! > $(PID_FILE)
	@printf "  PID: $$(cat $(PID_FILE))  |  Log: $(LOG_FILE)\n"
	@$(PYTHON) scripts/wait_for_server.py "$(BASE_URL)/" 60
	@$(MAKE) --no-print-directory open
	@printf "$(GREEN)✓ Server running — $(BASE_URL)$(RESET)\n"
	@printf "  $(YELLOW)make stop$(RESET) to shut down  |  $(YELLOW)make logs$(RESET) to tail output\n\n"

# ------------------------------------------------------------------------------
# demo — full first-time pipeline
# ------------------------------------------------------------------------------
.PHONY: demo
demo: install setup composite seed-all
	@echo ""
	@printf "$(GREEN)$(BOLD)✓ Setup complete — launching server$(RESET)\n\n"
	@printf "  NOTE: After 'make composite' completes, create the\n"
	@printf "        composite_vector_index in Atlas UI (JSON printed above),\n"
	@printf "        wait for READY, then re-run 'make seed-all' and 'make run'.\n\n"
	@$(MAKE) --no-print-directory start

# ------------------------------------------------------------------------------
# stop — kill the background server
# ------------------------------------------------------------------------------
.PHONY: stop
stop:
	@if [ -f $(PID_FILE) ]; then \
		PID=$$(cat $(PID_FILE)); \
		printf "$(BOLD)Stopping server (PID $$PID)…$(RESET)\n"; \
		kill $$PID 2>/dev/null && printf "$(GREEN)✓ Server stopped$(RESET)\n" \
			|| printf "$(YELLOW)Process $$PID already gone$(RESET)\n"; \
		rm -f $(PID_FILE); \
	else \
		printf "$(YELLOW)No $(PID_FILE) found — killing any uvicorn on port $(PORT)$(RESET)\n"; \
		lsof -ti :$(PORT) | xargs kill 2>/dev/null || true; \
	fi

# Internal: kill anything already on the port before starting
.PHONY: _kill_existing
_kill_existing:
	@if [ -f $(PID_FILE) ]; then \
		PID=$$(cat $(PID_FILE)); \
		kill $$PID 2>/dev/null; rm -f $(PID_FILE); \
		printf "$(YELLOW)Stopped previous server (PID $$PID)$(RESET)\n"; \
	fi
	@lsof -ti :$(PORT) | xargs kill 2>/dev/null || true

# ------------------------------------------------------------------------------
# open — open browser (server must be running)
# ------------------------------------------------------------------------------
.PHONY: open
open:
	@open "$(BASE_URL)" 2>/dev/null \
		|| xdg-open "$(BASE_URL)" 2>/dev/null \
		|| printf "$(CYAN)→ Open manually: $(BASE_URL)$(RESET)\n"

# ------------------------------------------------------------------------------
# reset — clear session via live API
# ------------------------------------------------------------------------------
.PHONY: reset
reset:
	@printf "$(BOLD)Resetting demo session for '$(DEMO_USER)'…$(RESET)\n"
	@curl -s -X POST "$(BASE_URL)/api/demo/reset" \
		-H "Content-Type: application/json" \
		-d '{"user_id":"$(DEMO_USER)"}' \
		| $(PYTHON) -m json.tool
	@echo ""

# ------------------------------------------------------------------------------
# health
# ------------------------------------------------------------------------------
.PHONY: health
health:
	@printf "$(BOLD)API health — $(BASE_URL)/api/health$(RESET)\n"
	@curl -s "$(BASE_URL)/api/health" | $(PYTHON) -m json.tool
	@echo ""

# ------------------------------------------------------------------------------
# trails — smoke-test the /api/trails endpoint
# ------------------------------------------------------------------------------
.PHONY: trails
trails:
	@printf "$(BOLD)Trails smoke-test — $(BASE_URL)/api/trails/$(DEMO_USER)$(RESET)\n"
	@curl -s "$(BASE_URL)/api/trails/$(DEMO_USER)" | $(PYTHON) -c "\
import sys, json; d=json.load(sys.stdin); \
rfy=d.get('recommended_for_you',[]); rows=d.get('genre_rows',[]); \
print(f'  RFY movies      : {len(rfy)}'); \
print(f'  Genre rows      : {len(rows)}'); \
[print(f'    [{r[\"row_index\"]}] {r[\"genre\"]}: {len(r[\"movies\"])} movies  swapped_from={r[\"swapped_from\"]}') for r in rows]; \
print(f'  rocchio_applied : {d[\"rocchio_applied\"]}'); \
print(f'  query_source    : {d[\"query_source\"]}'); \
print(f'  session neg     : {d[\"session\"][\"negative_count\"]}/{d[\"session\"][\"threshold\"]}'); \
print(f'  genre_skips     : {d[\"session\"][\"genre_skip_counts\"]}') \
"
	@echo ""

# ------------------------------------------------------------------------------
# boosts — list active admin boosts
# ------------------------------------------------------------------------------
.PHONY: boosts
boosts:
	@printf "$(BOLD)Active boosts — $(BASE_URL)/api/admin/boosts$(RESET)\n"
	@curl -s "$(BASE_URL)/api/admin/boosts" | $(PYTHON) -m json.tool
	@echo ""

# ------------------------------------------------------------------------------
# logs — tail background server log
# ------------------------------------------------------------------------------
.PHONY: logs
logs:
	@if [ -f $(LOG_FILE) ]; then \
		printf "$(BOLD)Tailing $(LOG_FILE) (Ctrl+C to stop)…$(RESET)\n\n"; \
		tail -f $(LOG_FILE); \
	else \
		printf "$(YELLOW)No log file found ($(LOG_FILE)). Is the server running via 'make start'?$(RESET)\n"; \
	fi

# ------------------------------------------------------------------------------
# clean
# ------------------------------------------------------------------------------
.PHONY: clean
clean:
	@printf "$(BOLD)Cleaning up…$(RESET)\n"
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	@find . -name "*.pyc" -delete 2>/dev/null; true
	@rm -f $(PID_FILE) $(LOG_FILE)
	@printf "$(GREEN)✓ Clean$(RESET)\n"

# ------------------------------------------------------------------------------
# lint
# ------------------------------------------------------------------------------
.PHONY: lint
lint:
	@printf "$(BOLD)Linting Python source files…$(RESET)\n"
	@$(PYTHON) -m py_compile \
		app/config.py \
		app/database.py \
		app/models.py \
		app/main.py \
		app/services/vector_math.py \
		app/services/session_service.py \
		app/services/recommendation_service.py \
		app/services/boost_service.py \
		app/routers/events.py \
		app/routers/recommendations.py \
		app/routers/trails.py \
		app/routers/admin.py \
		app/routers/demo.py \
		scripts/setup_indexes.py \
		scripts/seed_demo_user.py
	@printf "$(GREEN)✓ All files: syntax OK$(RESET)\n"
