#!/usr/bin/env bash
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "${RED}[ERROR]: Python not found${NC}"
  exit 1
fi

echo "${YELLOW}[INFO] Creating virtual environment"
"$PYTHON_BIN" -m venv .venv

echo "${YELLOW}[INFO] Activating virtual environment"
source .venv/bin/activate

echo "${YELLOW}[INFO] Upgrading pip"
python -m pip install --upgrade pip

echo "${YELLOW}[INFO] Installing dependencies"
pip install -r requirements.txt
pip install -e .

echo "${GREEN}[SUCCESS] REFLEX READY"

echo "${YELLOW}[INFO] To start reflex run:
- source .venv/bin/activate
- reflex --help
"