#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://localhost:3000}"
SYMBOL="${2:-AAPL}"
MODE="${3:-intraday}"
MAX_POLLS="${MAX_POLLS:-20}"
SLEEP_SEC="${SLEEP_SEC:-2}"

if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl not found" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: jq not found" >&2
  exit 1
fi

INTERNAL_KEY_HEADER=()
if [[ -n "${INTERNAL_API_KEY:-}" ]]; then
  INTERNAL_KEY_HEADER=(-H "X-Internal-Key: ${INTERNAL_API_KEY}")
fi

echo "[flow] BASE_URL=$BASE_URL"
echo "[flow] SYMBOL=$SYMBOL MODE=$MODE"

echo "[1/4] Health"
curl -sS "${BASE_URL}/api/health" | jq .

echo "[2/4] Submit analyze job"
SUBMIT_JSON="$({
  curl -sS -X POST "${BASE_URL}/api/analyze" \
    -H "Content-Type: application/json" \
    "${INTERNAL_KEY_HEADER[@]}" \
    -d "{\"symbol\":\"${SYMBOL}\",\"mode\":\"${MODE}\"}"
} )"
echo "$SUBMIT_JSON" | jq .

JOB_ID="$(echo "$SUBMIT_JSON" | jq -r '.jobId // empty')"
if [[ -z "$JOB_ID" ]]; then
  echo "ERROR: jobId missing from analyze submission response" >&2
  exit 1
fi

echo "[3/4] Poll analyze status for jobId=$JOB_ID"
STATUS=""
for ((i = 1; i <= MAX_POLLS; i++)); do
  STATUS_JSON="$(curl -sS "${BASE_URL}/api/analyze/status?jobId=${JOB_ID}")"
  STATUS="$(echo "$STATUS_JSON" | jq -r '.status // empty')"
  echo "poll=$i status=${STATUS:-unknown}"

  if [[ "$STATUS" == "succeeded" || "$STATUS" == "failed" ]]; then
    echo "$STATUS_JSON" | jq .
    break
  fi

  sleep "$SLEEP_SEC"
done

if [[ "$STATUS" != "succeeded" && "$STATUS" != "failed" ]]; then
  echo "ERROR: analysis job did not reach terminal state after ${MAX_POLLS} polls" >&2
  exit 1
fi

echo "[4/4] Done"
if [[ "$STATUS" == "succeeded" ]]; then
  echo "Async flow passed."
else
  echo "Async flow completed with failed status; inspect payload above."
fi
