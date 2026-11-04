#!/usr/bin/env bash
# Regenerate every user-evaluation statistic reported in the paper
# (expert Likert ratings, adoption, Krippendorff's alpha, per-pattern means,
# pattern-author and student results, and the free-text themes).
#
# Inputs : the pseudonymised CSV exports in this folder.
# Outputs: LaTeX tables/sections in ./latex/ (override with ODPGEN_LATEX_DIR),
#          student.tex in this folder, and a console summary.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

"$PY" analyze_user_eval.py
"$PY" analyze_author_eval.py
"$PY" analyze_students_eval.py
"$PY" theme_analysis.py
