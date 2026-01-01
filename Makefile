.PHONY: help install install-dev test lint format clean run-api run-dashboard docker-build docker-up

# Default target
help:
	@echo "MAPSO - Multi-Agent Production Scheduling Optimizer"
	@echo ""
	@echo "Available commands:"
	@echo "  make install          Install package and dependencies"
	@echo "  make install-dev      Install with development dependencies"
	@echo "  make test             Run test suite with coverage"
	@echo "  make lint             Run code quality checks (flake8, mypy)"
	@echo "  make format           Format code with black and isort"
	@echo "  make clean            Clean build artifacts and cache"
	@echo "  make run-api          Start FastAPI server"
	@echo "  make run-dashboard    Start Streamlit dashboard"
	@echo "  make docker-build     Build Docker images"
	@echo "  make docker-up        Start Docker containers"
	@echo "  make generate-data    Generate synthetic dataset"
	@echo "  make optimize         Run optimization example"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/ -v --cov=mapso --cov-report=term-missing --cov-report=html

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

test-benchmark:
	pytest tests/benchmarks/ -v --benchmark-only

# Code quality
lint:
	flake8 mapso/ dashboard/ tests/
	mypy mapso/

format:
	black mapso/ dashboard/ tests/ scripts/
	isort mapso/ dashboard/ tests/ scripts/

check-format:
	black --check mapso/ dashboard/ tests/ scripts/
	isort --check mapso/ dashboard/ tests/ scripts/

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# Running services
run-api:
	uvicorn mapso.api.main:app --reload --host 0.0.0.0 --port 8000

run-dashboard:
	streamlit run dashboard/app.py --server.port 8501

# Docker
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Data and optimization
generate-data:
	python scripts/generate_dataset.py --config configs/default.yaml

optimize:
	python scripts/run_optimization.py --input data/synthetic/sample_dataset.json --solver cpsat

# Documentation
docs:
	cd docs && make html

# Git
git-status:
	@git status

commit:
	@echo "Please use git commands directly or specify commit message with: make commit-msg MSG='your message'"

commit-msg:
	git add .
	git commit -m "$(MSG)"

# Pre-commit hooks
pre-commit:
	pre-commit run --all-files
