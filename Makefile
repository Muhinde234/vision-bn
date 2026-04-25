## VisionDx – common deployment commands
## Usage: make <target>

.PHONY: help dev prod down logs shell migrate seed

help:
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

# ── Development ───────────────────────────────────────────────────────────────
dev: ## Start local dev server (SQLite, no Docker)
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

migrate-dev: ## Run alembic migrations against dev SQLite DB
	alembic upgrade head

# ── Production (Docker Compose) ───────────────────────────────────────────────
prod: ## Build & start all containers (requires .env.production)
	docker compose --env-file .env.production up --build -d

down: ## Stop all containers
	docker compose down

restart: ## Restart API + worker without full rebuild
	docker compose restart api worker

logs: ## Tail logs for all services
	docker compose logs -f

logs-api: ## Tail API logs only
	docker compose logs -f api

logs-worker: ## Tail Celery worker logs
	docker compose logs -f worker

# ── Database ──────────────────────────────────────────────────────────────────
migrate: ## Run alembic migrations inside running api container
	docker compose exec api alembic upgrade head

shell-db: ## Open psql shell in the postgres container
	docker compose exec db psql -U visiondx -d visiondx

# ── Maintenance ───────────────────────────────────────────────────────────────
shell: ## Open a bash shell in the api container
	docker compose exec api bash

ps: ## Show container statuses
	docker compose ps

clean: ## Remove containers, networks, and volumes (DESTRUCTIVE)
	docker compose down -v --remove-orphans
