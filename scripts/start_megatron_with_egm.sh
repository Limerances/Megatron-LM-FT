#!/bin/bash
set -euo pipefail

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <launcher command...>" >&2
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EGM_POOL_SIZE_GB="${EGM_POOL_SIZE_GB:-16}"
EGM_NUM_SLOTS="${EGM_NUM_SLOTS:-2}"
EGM_NUMA_NODE_ID="${EGM_NUMA_NODE_ID:-0}"
EGM_DEVICE_ID="${EGM_DEVICE_ID:-0}"
EGM_SOCKET_PATH="${EGM_SOCKET_PATH:-/tmp/megatron_egm_manager.sock}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
EGM_DAEMON_READY_TIMEOUT_S="${EGM_DAEMON_READY_TIMEOUT_S:-60}"

TOTAL_DAEMON_SLOTS="${EGM_DAEMON_NUM_SLOTS:-$((NPROC_PER_NODE * EGM_NUM_SLOTS))}"

DAEMON_CMD=(
    python3 -m megatron.core.egm
    --pool-size-gb "${EGM_POOL_SIZE_GB}"
    --numa-node-id "${EGM_NUMA_NODE_ID}"
    --num-slots "${TOTAL_DAEMON_SLOTS}"
    --device-id "${EGM_DEVICE_ID}"
    --socket-path "${EGM_SOCKET_PATH}"
)

cleanup() {
    if [[ -S "${EGM_SOCKET_PATH}" ]]; then
        (
            cd "${ROOT_DIR}"
            python3 - <<'PY' "${EGM_SOCKET_PATH}" >/dev/null 2>&1 || true
from megatron.core.egm.egm_manager import EGMClient
import sys

client = EGMClient(socket_path=sys.argv[1])
try:
    client.connect()
    client.request_shutdown()
finally:
    client.close()
PY
        ) || true
    fi

    if [[ -n "${EGM_DAEMON_PID:-}" ]] && kill -0 "${EGM_DAEMON_PID}" 2>/dev/null; then
        kill "${EGM_DAEMON_PID}" 2>/dev/null || true
        wait "${EGM_DAEMON_PID}" 2>/dev/null || true
    fi
}

trap cleanup EXIT INT TERM

(
    cd "${ROOT_DIR}"
    "${DAEMON_CMD[@]}"
) &
EGM_DAEMON_PID=$!

deadline=$((SECONDS + EGM_DAEMON_READY_TIMEOUT_S))
while [[ ${SECONDS} -lt ${deadline} ]]; do
    if ! kill -0 "${EGM_DAEMON_PID}" 2>/dev/null; then
        echo "EGM daemon exited before becoming ready" >&2
        wait "${EGM_DAEMON_PID}" || true
        exit 1
    fi
    if [[ -S "${EGM_SOCKET_PATH}" ]] && (
        cd "${ROOT_DIR}" &&
        python3 - <<'PY' "${EGM_SOCKET_PATH}" >/dev/null 2>&1
from megatron.core.egm.egm_manager import EGMClient
import sys

client = EGMClient(socket_path=sys.argv[1])
try:
    client.connect()
    raise SystemExit(0 if client.ping() else 1)
finally:
    client.close()
PY
    ); then
        break
    fi
    sleep 0.5
done

if [[ ! -S "${EGM_SOCKET_PATH}" ]]; then
    echo "EGM daemon did not create socket within ${EGM_DAEMON_READY_TIMEOUT_S}s: ${EGM_SOCKET_PATH}" >&2
    exit 1
fi

if ! (
    cd "${ROOT_DIR}" &&
    python3 - <<'PY' "${EGM_SOCKET_PATH}" >/dev/null 2>&1
from megatron.core.egm.egm_manager import EGMClient
import sys

client = EGMClient(socket_path=sys.argv[1])
try:
    client.connect()
    raise SystemExit(0 if client.ping() else 1)
finally:
    client.close()
PY
); then
    echo "EGM daemon socket exists but ping failed: ${EGM_SOCKET_PATH}" >&2
    exit 1
fi

"$@"
cmd_exit_code=$?
cleanup
exit "${cmd_exit_code}"
