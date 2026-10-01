#!/usr/bin/env bash
# Resolve the project from this file, independently of the caller's working directory.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ $# -eq 0 ]]; then
  echo 'Usage : bash scripts/search_hadith.sh "Je cherche le hadith sur la colère"' >&2
  exit 2
fi
exec docker compose --project-directory "$project_dir" -f "$project_dir/docker-compose.yml" run --rm --no-deps -T \
  --user "$(id -u):$(id -g)" \
  -e HF_HOME=/app/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=1 \
  api python scripts/run_hadith_cli.py "$@"
