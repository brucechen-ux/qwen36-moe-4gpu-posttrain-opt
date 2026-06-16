#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/collect_gpu_stats.sh [options]

Options:
  --interval SECONDS   Sampling interval. Default: 2
  --duration SECONDS   Total duration. 0 means run until interrupted. Default: 0
  --output PATH        Output CSV path. Default: logs/gpu_stats.csv
  --help               Show this help message.
EOF
}

log() {
  printf '[collect_gpu_stats] %s\n' "$*" >&2
}

die() {
  log "ERROR: $*"
  exit 1
}

INTERVAL=2
DURATION=0
OUTPUT="logs/gpu_stats.csv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval)
      [[ $# -ge 2 ]] || die "--interval requires a value"
      INTERVAL="$2"
      shift 2
      ;;
    --duration)
      [[ $# -ge 2 ]] || die "--duration requires a value"
      DURATION="$2"
      shift 2
      ;;
    --output)
      [[ $# -ge 2 ]] || die "--output requires a value"
      OUTPUT="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found; run this on a GPU node"

case "${INTERVAL}" in
  ''|*[!0-9.]*)
    die "--interval must be numeric"
    ;;
esac

case "${DURATION}" in
  ''|*[!0-9.]*)
    die "--duration must be numeric"
    ;;
esac

mkdir -p "$(dirname "${OUTPUT}")"

if [[ ! -s "${OUTPUT}" ]]; then
  printf 'sample_time,gpu_timestamp,index,name,memory_used_mb,memory_total_mb,utilization_gpu_pct,power_draw_w,power_limit_w,temperature_gpu_c\n' > "${OUTPUT}"
fi

QUERY='timestamp,index,name,memory.used,memory.total,utilization.gpu,power.draw,power.limit,temperature.gpu'
START_EPOCH="$(date +%s)"

log "writing ${OUTPUT}, interval=${INTERVAL}s, duration=${DURATION}s"
trap 'log "stopped"; exit 0' INT TERM

while true; do
  NOW_ISO="$(date -Iseconds)"
  if ! nvidia-smi --query-gpu="${QUERY}" --format=csv,noheader,nounits |
    awk -v sample_time="${NOW_ISO}" 'BEGIN { FS=", "; OFS="," } { print sample_time,$0 }' >> "${OUTPUT}"; then
    die "nvidia-smi query failed"
  fi

  if [[ "${DURATION}" != "0" ]]; then
    NOW_EPOCH="$(date +%s)"
    ELAPSED=$((NOW_EPOCH - START_EPOCH))
    if (( ELAPSED >= DURATION )); then
      log "completed duration=${DURATION}s"
      exit 0
    fi
  fi

  sleep "${INTERVAL}"
done
