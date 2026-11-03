#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
if [ ! -x "$PY" ]; then
  echo "[ERROR] venv not found: $PY"
  echo "  python -m venv ../.venv"
  echo "  ../.venv/bin/pip install -r requirements.txt"
  exit 1
fi
"$PY" -c "import lightgbm, rdkit" || {
  echo "[ERROR] missing lightgbm/rdkit"; exit 1; }
exec "$PY" cpamp_gui.py
