#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export BUILDX_BUILDER="${BUILDX_BUILDER:-default}"
docker compose up -d --build --wait
printf 'ReliefBench http://127.0.0.1:%s/\n' "${RELIEFBENCH_PORT:-19958}"
