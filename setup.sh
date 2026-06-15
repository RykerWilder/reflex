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
  echo -e "${RED}[ERROR]: Python not found${NC}"
  exit 1
fi

echo -e "${YELLOW}[INFO]${NC} Creating virtual environment"
"$PYTHON_BIN" -m venv .venv

echo -e "${YELLOW}[INFO]${NC} Activating virtual environment"
source .venv/bin/activate

echo -e "${YELLOW}[INFO]${NC} Upgrading pip"
python -m pip install --upgrade pip

echo -e "${YELLOW}[INFO]${NC} Installing dependencies"
pip install -r requirements.txt
pip install -e .

echo -e "${GREEN}[SUCCESS]${NC} REFLEX READY"

echo -e "${YELLOW}[INFO]${NC} To start reflex run:
- source .venv/bin/activate
- reflex --help"