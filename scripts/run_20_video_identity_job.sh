#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_dir="artifacts/agent-track-labeling/batch-20"
combined_results="$run_dir/combined-results"
combined_dino="$run_dir/combined-dino"
combined_projection="$run_dir/combined-projections"
evidence="$run_dir/evidence"
predictions="$run_dir/predictions-gemini-3.6-flash.jsonl"
models="$run_dir/fixture-models"
recovery="$run_dir/recovery"
labels="$run_dir/labelled-videos"
status="$run_dir/status.txt"

mkdir -p \
  "$combined_results" "$combined_dino" "$combined_projection" \
  "$models" "$recovery" "$labels"
printf '%s\n' "$$" >"$run_dir/pid"
printf 'running\n' >"$status"

finish() {
  exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    printf 'complete\n' >"$status"
  else
    printf 'failed exit_code=%s\n' "$exit_code" >"$status"
  fi
}
trap finish EXIT

for source in \
  artifacts/published-tracking-review/results/yolo26m-botsort-high-recall/*.json \
  artifacts/tactical-coverage-review/results/yolo26m-botsort-high-recall/*.json
do
  ln -sfn "$(realpath "$source")" "$combined_results/$(basename "$source")"
done

for source in \
  artifacts/published-tracking-review/dino-features-high-recall/*.json \
  artifacts/tactical-coverage-review/dino-features/*.json
do
  ln -sfn "$(realpath "$source")" "$combined_dino/$(basename "$source")"
done

for source in \
  artifacts/published-tracking-review/pitch-projections/*.json \
  artifacts/tactical-coverage-review/pitch-projections/*.json
do
  ln -sfn "$(realpath "$source")" "$combined_projection/$(basename "$source")"
done

if [[ -f "$evidence/manifest.json" ]] && .venv/bin/python - "$evidence/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
valid = (
    manifest.get("minimum_confidence") == 0.45
    and manifest.get("minimum_detections") == 10
)
raise SystemExit(0 if valid else 1)
PY
then
  echo "Reusing filtered evidence manifest at $evidence"
else
  .venv/bin/python scripts/run_agent_track_labeling.py prepare \
    --results "$combined_results" \
    --labels artifacts/published-tracking-review/track-labels.json \
    --output "$evidence" \
    --minimum-confidence 0.45 \
    --minimum-detections 10
fi

mapfile -t target_keys < <(
  .venv/bin/python - "$evidence/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
target_clips = {
    path.stem
    for path in Path(
        "artifacts/tactical-coverage-review/results/yolo26m-botsort-high-recall"
    ).glob("*.json")
}
for row in manifest["tracks"]:
    if row["clip_id"] in target_clips and row["split"] != "seed":
        print(row["key"])
PY
)

key_args=()
for key in "${target_keys[@]}"; do
  key_args+=(--key "$key")
done

echo "Selected 20 videos and ${#target_keys[@]} non-seed tracks"
.venv/bin/python scripts/run_concurrent_agent_track_labeling.py \
  --evidence "$evidence" \
  --output "$predictions" \
  --env .env \
  --model gemini-3.6-flash \
  --workers 2 \
  --retries 3 \
  --retry-delay 30 \
  --request-delay 2 \
  "${key_args[@]}"

mapfile -t fixtures < <(
  .venv/bin/python - "$evidence/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
print(*sorted({row["fixture_id"] for row in manifest["tracks"]}), sep="\n")
PY
)

train_args=()
for fixture in "${fixtures[@]}"; do
  train_args+=(--fixture "$fixture")
done

.venv/bin/python scripts/fixture_identity_recovery.py train \
  --evidence "$evidence/manifest.json" \
  --predictions "$predictions" \
  --dino-dir "$combined_dino" \
  --projection-dir "$combined_projection" \
  --output-dir "$models" \
  "${train_args[@]}"

for fixture in "${fixtures[@]}"; do
  .venv/bin/python scripts/fixture_identity_recovery.py recover \
    --model "$models/$fixture.json" \
    --evidence "$evidence/manifest.json" \
    --predictions "$predictions" \
    --dino-dir "$combined_dino" \
    --projection-dir "$combined_projection" \
    --output "$recovery/$fixture.json"
done

.venv/bin/python scripts/run_agent_track_labeling.py reconcile \
  --evidence "$evidence" \
  --predictions "$predictions" \
  --output "$labels"

echo "20-video identity job complete"
