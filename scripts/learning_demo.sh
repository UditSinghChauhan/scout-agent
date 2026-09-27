#!/usr/bin/env bash
# Scout learning demo: cold start, then three runs that show memory reuse and lessons.
# Real LLM and search calls. Run from the repo root with the virtualenv active.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

latest_run() { ls -td runs/*/ 2>/dev/null | head -1 | xargs basename; }

"$PY" -m scout reset --yes --cache

GOALS=(
  "We sell a campus hiring-challenge platform. Research Zoho as a prospect and tell us how to pitch."
  "Prep me for an SDE intern interview at Zoho."
  "We sell a campus hiring-challenge platform. Research Freshworks as a prospect and tell us how to pitch."
)
IDS=()
for goal in "${GOALS[@]}"; do
  echo "=== $goal"
  "$PY" -m scout run "$goal"
  IDS+=("$(latest_run)")
done

"$PY" -m scout insights
STAMP=$(date +%Y%m%d-%H%M%S)
"$PY" scripts/demo_summary.py "runs/learning_demo_${STAMP}.md" "${IDS[@]}"
echo "Summary saved to runs/learning_demo_${STAMP}.md"
