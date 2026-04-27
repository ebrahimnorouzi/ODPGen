#!/usr/bin/env bash
# Regenerate the user-evaluation .tex outputs from the CSVs in this directory.
set -euo pipefail
cd "$(dirname "$0")"
python3 analyze_user_eval.py
python3 theme_analysis.py
