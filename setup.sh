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

echo -e "${YELLOW}[INFO] Creating virtual environment${NC}"
"$PYTHON_BIN" -m venv .venv

echo -e "${YELLOW}[INFO] Activating virtual environment${NC}"
source .venv/bin/activate

echo -e "${YELLOW}[INFO] Upgrading pip${NC}"
python -m pip install --upgrade pip

echo -e "${YELLOW}[INFO] Installing dependencies${NC}"
pip install -r requirements.txt
pip install -e .

echo -e "${GREEN}[SUCCESS] REFLEX READY${NC}"

echo -e "${YELLOW}[INFO] To start reflex run:
- source .venv/bin/activate
- reflex --help
${NC}"