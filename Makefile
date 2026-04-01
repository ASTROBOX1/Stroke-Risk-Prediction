.PHONY: help install train run-app run-api run-all test clean docker-build docker-up docker-down lint format check

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)Stroke Risk Prediction - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-18s$(NC) %s\n", $$1, $$2}'

install: ## Install Python dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)✓ Installation complete$(NC)"

train: ## Train the ML models
	@echo "$(BLUE)Training models...$(NC)"
	python train_model.py
	@echo "$(GREEN)✓ Training complete$(NC)"

run-app: ## Run Streamlit dashboard (port 8501)
	@echo "$(BLUE)Starting Streamlit dashboard...$(NC)"
	streamlit run app.py --server.port=8501 --server.address=0.0.0.0

run-api: ## Run FastAPI server (port 8000)
	@echo "$(BLUE)Starting FastAPI server...$(NC)"
	uvicorn api:app --host 0.0.0.0 --port 8000 --reload

run-all: ## Run both dashboard and API (requires tmux or separate terminals)
	@echo "$(BLUE)Starting all services...$(NC)"
	@echo "$(YELLOW)Note: Run 'make run-app' and 'make run-api' in separate terminals$(NC)"

test: ## Run unit tests with coverage
	@echo "$(BLUE)Running tests...$(NC)"
	pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html
	@echo "$(GREEN)✓ Tests complete. Open htmlcov/index.html for coverage report$(NC)"

test-api: ## Test API endpoints (requires running API)
	@echo "$(BLUE)Testing API health...$(NC)"
	@curl -s http://localhost:8000/health | python -m json.tool || echo "$(YELLOW)API not running on port 8000$(NC)"

clean: ## Clean up cache files and logs
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -f logs/*.log 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker-compose build
	@echo "$(GREEN)✓ Docker image built$(NC)"

docker-up: ## Start Docker containers
	@echo "$(BLUE)Starting Docker containers...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✓ Containers started$(NC)"
	@echo "$(YELLOW)Dashboard: http://localhost:8501$(NC)"
	@echo "$(YELLOW)API: http://localhost:8000/api/docs$(NC)"

docker-down: ## Stop Docker containers
	@echo "$(BLUE)Stopping Docker containers...$(NC)"
	docker-compose down
	@echo "$(GREEN)✓ Containers stopped$(NC)"

docker-logs: ## Show Docker container logs
	docker-compose logs -f

lint: ## Run code linting (flake8)
	@echo "$(BLUE)Running linter...$(NC)"
	@command -v flake8 >/dev/null 2>&1 || (echo "$(YELLOW)Installing flake8...$(NC)" && pip install flake8)
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	@echo "$(GREEN)✓ Linting complete$(NC)"

format: ## Format code with black
	@echo "$(BLUE)Formatting code...$(NC)"
	@command -v black >/dev/null 2>&1 || (echo "$(YELLOW)Installing black...$(NC)" && pip install black)
	black . --line-length 100
	@echo "$(GREEN)✓ Formatting complete$(NC)"

check: test lint ## Run all checks (tests + linting)
	@echo "$(GREEN)✓ All checks passed$(NC)"

deps-update: ## Show outdated dependencies
	@echo "$(BLUE)Checking for outdated dependencies...$(NC)"
	pip list --outdated

env-setup: ## Create .env file from .env.example
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN)✓ Created .env file from .env.example$(NC)"; \
		echo "$(YELLOW)⚠ Please update .env with your settings$(NC)"; \
	else \
		echo "$(YELLOW).env file already exists$(NC)"; \
	fi

# Development shortcuts
dev: install train run-app ## Quick setup for development (install, train, run)

prod-check: ## Check production readiness
	@echo "$(BLUE)Checking production readiness...$(NC)"
	@echo ""
	@echo "$(YELLOW)Dependencies:$(NC)"
	@pip check && echo "$(GREEN)✓ No dependency conflicts$(NC)" || echo "$(YELLOW)⚠ Dependency issues found$(NC)"
	@echo ""
	@echo "$(YELLOW)Environment:$(NC)"
	@test -f .env && echo "$(GREEN)✓ .env file exists$(NC)" || echo "$(YELLOW)⚠ .env file missing$(NC)"
	@echo ""
	@echo "$(YELLOW)Models:$(NC)"
	@test -f models/best_stroke_model.joblib && echo "$(GREEN)✓ Model file exists$(NC)" || echo "$(YELLOW)⚠ Model not trained$(NC)"
	@echo ""
	@echo "$(YELLOW)Data:$(NC)"
	@test -f data/healthcare-dataset-stroke-data.csv && echo "$(GREEN)✓ Data file exists$(NC)" || echo "$(YELLOW)⚠ Data file missing$(NC)"
	@echo ""

info: ## Show project information
	@echo "$(BLUE)═══════════════════════════════════════════════$(NC)"
	@echo "$(GREEN)  Stroke Risk Prediction Analytics Platform$(NC)"
	@echo "$(BLUE)═══════════════════════════════════════════════$(NC)"
	@echo ""
	@echo "$(YELLOW)Version:$(NC) 2.2.0"
	@echo "$(YELLOW)Python:$(NC) $$(python --version 2>&1)"
	@echo "$(YELLOW)Location:$(NC) $$(pwd)"
	@echo ""
	@echo "$(YELLOW)Key Files:$(NC)"
	@find . -maxdepth 2 -type f -name "*.py" | wc -l | xargs echo "  Python files:"
	@test -f models/best_stroke_model.joblib && echo "  Model: $(GREEN)✓ trained$(NC)" || echo "  Model: $(YELLOW)⚠ not trained$(NC)"
	@echo ""
	@echo "Run '$(GREEN)make help$(NC)' for available commands"
