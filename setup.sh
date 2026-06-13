#!/usr/bin/env bash

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Error: Python not found"
  exit 1
fi

"$PYTHON_BIN" -m venv .venv

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
  source .venv/Scripts/activate
else
  echo "Error: venv activation script not found"
  exit 1
fi

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo "Reflex ready"