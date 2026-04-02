#!/usr/bin/env bash
# NOTE: use with source or .
#   source scripts/enable-venv.sh

VENV_DIR=".venv"

# Guard statement to exit early if already in venv
if [ -n "$VIRTUAL_ENV" ]; then
  return
fi

# Create venv if directory doesn't exist
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating venv at '$VENV_DIR'..."
  python -m venv "$VENV_DIR" \
    || { echo "venv creation failed"; return 1; }
fi

# Activate venv
echo "Activating venv..."
source "$VENV_DIR/bin/activate" \
|| { echo "venv activation failed"; return 1; }

# Install dependencies — prefer uv, fall back to pip
echo "Installing dependencies..."
if command -v uv &> /dev/null; then
  uv sync --group dev \
    || { echo "uv sync failed"; return 1; }
else
  pip install --upgrade pip && pip install -e ".[dev]" \
    || { echo "pip install failed"; return 1; }
fi

# Install pre-commit hooks if not already installed
if [ ! -f ".git/hooks/pre-commit" ]; then
  echo "Installing pre-commit hooks..."
  pre-commit install \
    || { echo "pre-commit install failed"; return 1; }
fi
