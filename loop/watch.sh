#!/bin/bash
# Keep evals, overlays and the failure map current while generation runs.
cd "$(dirname "$0")/.."
PY=eval/.venv/bin/python
while true; do
  $PY eval/run_all.py --reason >> logs/watch-eval.log 2>&1
  $PY eval/integrity.py >> logs/watch-integrity.log 2>&1
  $PY eval/render_all.py >> logs/watch-render.log 2>&1
  $PY loop/run_loop.py --aggregate-only > /dev/null 2>&1
  done_n=$(grep -cE "^(OK|FAIL)" logs/gen-grid.log)
  echo "$(date +%H:%M:%S) gen=$done_n evals=$(ls results/evals | wc -l) overlays=$(ls results/overlays/*.mp4 2>/dev/null | wc -l)" >> logs/watch.log
  [ "$done_n" -ge 44 ] && [ -f logs/.watch_final ] && break
  [ "$done_n" -ge 44 ] && touch logs/.watch_final
  sleep 20
done
