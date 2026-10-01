#!/usr/bin/env bash
# End-to-end pipeline: tests -> Q calibration -> TVCSI table -> sweeps -> figures -> summary -> dataset.
# Usage: bash scripts/run_all.sh   (set PYTHON=/path/to/python to override the interpreter)
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python; else PYTHON=python3; fi
fi
"$PYTHON" -c "import tvcsi" 2>/dev/null || "$PYTHON" -m pip install -q -e .

step() { echo; echo "=== $* ==="; }
step "tests";                 "$PYTHON" -m pytest -q
step "Q calibration";         "$PYTHON" scripts/calibrate_q.py
step "TVCSI table (default)"; "$PYTHON" scripts/fit_tvcsi_table.py
step "sweeps + runtime";      "$PYTHON" -m tvcsi.sweep
step "figures";               "$PYTHON" -m tvcsi.plots
step "summary";               "$PYTHON" scripts/make_summary.py
step "dataset export";        "$PYTHON" scripts/export_dataset.py
echo; echo "done: results/ and figures/"
