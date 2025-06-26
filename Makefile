# Hovercraft Analysis Pipeline - Developer Makefile
# This Makefile provides common development commands for the project

.PHONY: help install dev test lint format clean docs check-quality run-dashboard

# Default target - show help
help:
	@echo "Hovercraft Analysis Pipeline - Development Commands"
	@echo "=================================================="
	@echo ""
	@echo "Installation:"
	@echo "  make install      Install package in editable mode"
	@echo "  make dev          Install with development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test         Run all tests"
	@echo "  make test-fast    Run tests without coverage"
	@echo "  make test-cov     Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint         Run linting checks (flake8, mypy)"
	@echo "  make format       Auto-format code (black, isort)"
	@echo "  make check-quality Run all quality checks without modifying"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs         Build documentation"
	@echo "  make docs-serve   Build and serve documentation locally"
	@echo ""
	@echo "Development:"
	@echo "  make run-dashboard Launch the analysis dashboard"
	@echo "  make clean        Remove build artifacts and caches"
	@echo ""

# Installation targets
install:
	pip install -e .
	@echo "✓ Package installed successfully"

dev:
	pip install -e ".[dev,notebook]"
	@echo "✓ Development environment installed successfully"

# Testing targets
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

# Code quality targets
lint:
	@echo "Running flake8..."
	flake8 src/ tests/ --config=.flake8 || true
	@echo ""
	@echo "Running mypy..."
	mypy src/ --config-file pyproject.toml || true

format:
	@echo "Running isort..."
	isort src/ tests/
	@echo "Running black..."
	black src/ tests/
	@echo "✓ Code formatted successfully"

check-quality:
	@echo "Checking isort..."
	isort --check-only src/hovercraft_analysis tests/
	@echo "Checking black..."
	black --check src/hovercraft_analysis tests/
	@echo "Checking flake8..."
	flake8 src/ tests/ --config=.flake8
	@echo "Checking mypy..."
	mypy src/ --config-file pyproject.toml

# Documentation targets
docs:
	@echo "Building documentation..."
	cd docs && sphinx-build -b html . _build/html
	@echo "Documentation built in docs/_build/html/index.html"

docs-serve: docs
	@echo "Serving documentation at http://localhost:8000"
	cd docs/_build/html && python -m http.server

# Development utilities
run-dashboard:
	@echo "Starting Hovercraft Analysis Dashboard..."
	hovercraft-dashboard

clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf src/*.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleaned successfully"

# Quick commands for common workflows
.PHONY: quick-test quick-check

quick-test:
	pytest tests/ -v -x

quick-check: format lint test-fast
	@echo "✓ Quick check completed"