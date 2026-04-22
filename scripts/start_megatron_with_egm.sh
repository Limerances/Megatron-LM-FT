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
EGM_AUTO_TOPOLOGY="${EGM_AUTO_TOPOLOGY:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
EGM_DAEMON_READY_TIMEOUT_S="${EGM_DAEMON_READY_TIMEOUT_S:-60}"

TOTAL_DAEMON_SLOTS="${EGM_DAEMON_NUM_SLOTS:-$((NPROC_PER_NODE * EGM_NUM_SLOTS))}"
EGM_SOCKET_PATH_TEMPLATE="${EGM_SOCKET_PATH_TEMPLATE:-}"
if [[ -z "${EGM_SOCKET_PATH_TEMPLATE}" ]]; then
    if [[ "${EGM_SOCKET_PATH}" == *.sock ]]; then
        EGM_SOCKET_PATH_TEMPLATE="${EGM_SOCKET_PATH%.sock}.numa{numa}.sock"
    else
        EGM_SOCKET_PATH_TEMPLATE="${EGM_SOCKET_PATH}.numa{numa}"
    fi
fi
export EGM_SOCKET_PATH_TEMPLATE
export EGM_AUTO_TOPOLOGY

declare -a EGM_DAEMON_PIDS=()
declare -a EGM_DAEMON_SOCKETS=()

wait_for_socket() {
    local socket_path="$1"
    local daemon_pid="$2"
    local deadline=$((SECONDS + EGM_DAEMON_READY_TIMEOUT_S))
    while [[ ${SECONDS} -lt ${deadline} ]]; do
        if ! kill -0 "${daemon_pid}" 2>/dev/null; then
            echo "EGM daemon exited before becoming ready: ${socket_path}" >&2
            wait "${daemon_pid}" || true
            return 1
        fi
        if [[ -S "${socket_path}" ]] && (
            cd "${ROOT_DIR}" &&
            python3 - <<'PY' "${socket_path}" >/dev/null 2>&1
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
            return 0
        fi
        sleep 0.5
    done
    echo "EGM daemon did not become ready within ${EGM_DAEMON_READY_TIMEOUT_S}s: ${socket_path}" >&2
    return 1
}

cleanup() {
    local socket_path
    for socket_path in "${EGM_DAEMON_SOCKETS[@]}"; do
        if [[ -S "${socket_path}" ]]; then
            (
                cd "${ROOT_DIR}"
                python3 - <<'PY' "${socket_path}" >/dev/null 2>&1 || true
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
    done

    local pid
    for pid in "${EGM_DAEMON_PIDS[@]}"; do
        if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
            wait "${pid}" 2>/dev/null || true
        fi
    done
}

trap cleanup EXIT INT TERM

if [[ "${EGM_AUTO_TOPOLOGY}" == "1" ]]; then
    export FT_EGM_AUTO_TOPOLOGY=1
    export FT_EGM_DAEMON_SOCKET_TEMPLATE="${EGM_SOCKET_PATH_TEMPLATE}"

    TOPOLOGY_JSON="$(
        python3 - <<'PY'
import os
import json
from collections import OrderedDict

try:
    import cuda.bindings.driver as cuda_driver
except ImportError:
    import cuda.cuda as cuda_driver

def norm(err):
    if isinstance(err, tuple):
        return err[0] if err else None
    return err

err = cuda_driver.cuInit(0)
if norm(err) != cuda_driver.CUresult.CUDA_SUCCESS:
    raise SystemExit(f"cuInit failed: {err}")

template = os.environ["EGM_SOCKET_PATH_TEMPLATE"]
num_slots = int(os.environ["EGM_NUM_SLOTS"])

err, count = cuda_driver.cuDeviceGetCount()
if norm(err) != cuda_driver.CUresult.CUDA_SUCCESS:
    raise SystemExit(f"cuDeviceGetCount failed: {err}")

groups = OrderedDict()
device_entries = []
for dev_idx in range(int(count)):
    err, dev = cuda_driver.cuDeviceGet(dev_idx)
    if norm(err) != cuda_driver.CUresult.CUDA_SUCCESS:
        raise SystemExit(f"cuDeviceGet failed for device {dev_idx}: {err}")
    err, numa = cuda_driver.cuDeviceGetAttribute(
        cuda_driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
        dev,
    )
    if norm(err) != cuda_driver.CUresult.CUDA_SUCCESS:
        raise SystemExit(f"cuDeviceGetAttribute(HOST_NUMA_ID) failed for device {dev_idx}: {err}")
    numa = int(numa)
    if numa not in groups:
        groups[numa] = {"device": dev_idx, "count": 0}
    groups[numa]["count"] += 1
    socket_path = template.format(numa=numa, device=groups[numa]["device"])
    device_entries.append(
        {
            "local_rank": dev_idx,
            "device": dev_idx,
            "numa": numa,
            "socket": socket_path,
        }
    )

daemon_entries = []
for numa, info in groups.items():
    socket_path = template.format(numa=numa, device=info["device"])
    total_slots = info["count"] * num_slots
    daemon_entries.append(
        {
            "numa": numa,
            "device": info["device"],
            "count": info["count"],
            "slots": total_slots,
            "socket": socket_path,
        }
    )

print(json.dumps({"daemons": daemon_entries, "devices": device_entries}))
PY
    )"

    if [[ -z "${TOPOLOGY_JSON}" ]]; then
        echo "Failed to detect EGM topology; no daemon groups were created." >&2
        exit 1
    fi

    export FT_EGM_LOCAL_RANK_SOCKET_MAP="$(
        python3 - <<'PY' "${TOPOLOGY_JSON}"
import json
import sys
topology = json.loads(sys.argv[1])
print(json.dumps({str(d["local_rank"]): d["socket"] for d in topology["devices"]}))
PY
    )"
    export FT_EGM_LOCAL_RANK_NUMA_MAP="$(
        python3 - <<'PY' "${TOPOLOGY_JSON}"
import json
import sys
topology = json.loads(sys.argv[1])
print(json.dumps({str(d["local_rank"]): int(d["numa"]) for d in topology["devices"]}))
PY
    )"

    while IFS=$'\t' read -r daemon_numa daemon_device daemon_count daemon_slots daemon_socket; do
        [[ -z "${daemon_socket}" ]] && continue
        echo "Starting EGM daemon for NUMA ${daemon_numa} using device ${daemon_device} (${daemon_count} local GPU(s), slots=${daemon_slots}) at ${daemon_socket}"
        (
            cd "${ROOT_DIR}"
            python3 -m megatron.core.egm \
                --pool-size-gb "${EGM_POOL_SIZE_GB}" \
                --numa-node-id "${daemon_numa}" \
                --num-slots "${daemon_slots}" \
                --device-id "${daemon_device}" \
                --socket-path "${daemon_socket}"
        ) &
        daemon_pid=$!
        EGM_DAEMON_PIDS+=("${daemon_pid}")
        EGM_DAEMON_SOCKETS+=("${daemon_socket}")
        wait_for_socket "${daemon_socket}" "${daemon_pid}"
    done < <(
        python3 - <<'PY' "${TOPOLOGY_JSON}"
import json
import sys
topology = json.loads(sys.argv[1])
for d in topology["daemons"]:
    print(f'{d["numa"]}\t{d["device"]}\t{d["count"]}\t{d["slots"]}\t{d["socket"]}')
PY
    )
else
    export FT_EGM_AUTO_TOPOLOGY=0
    DAEMON_CMD=(
        python3 -m megatron.core.egm
        --pool-size-gb "${EGM_POOL_SIZE_GB}"
        --numa-node-id "${EGM_NUMA_NODE_ID}"
        --num-slots "${TOTAL_DAEMON_SLOTS}"
        --device-id "${EGM_DEVICE_ID}"
        --socket-path "${EGM_SOCKET_PATH}"
    )
    (
        cd "${ROOT_DIR}"
        "${DAEMON_CMD[@]}"
    ) &
    daemon_pid=$!
    EGM_DAEMON_PIDS+=("${daemon_pid}")
    EGM_DAEMON_SOCKETS+=("${EGM_SOCKET_PATH}")
    wait_for_socket "${EGM_SOCKET_PATH}" "${daemon_pid}"
fi

"$@"
cmd_exit_code=$?
cleanup
exit "${cmd_exit_code}"
