#!/usr/bin/env bash

# Run the Westmead 12x1 BRAVEHEART extraction sweeps in separate, sequential
# Python processes. A process must exit before the next model is loaded, so RAM
# from one sweep can be reclaimed before the following sweep starts.
#
# Usage:
#   scripts/run_braveheart_sweeps.sh
#
# Optional environment variables:
#   PYTHON_BIN=/path/to/python  Python executable from the desired environment.
#   SWEEP_LOG_DIR=path         Directory for per-sweep logs and the summary.
#   STOP_ON_ERROR=1            Stop after the first failed sweep (default: 0).

set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${repo_root}"

python_bin="${PYTHON_BIN:-python3}"
stop_on_error="${STOP_ON_ERROR:-0}"
run_id="$(date +%Y%m%d-%H%M%S)"
log_dir="${SWEEP_LOG_DIR:-sandbox/braveheart-sweep-logs/${run_id}}"

# Keep the higher-memory 4500 px resolution sweep last. Each entry has a unique
# DATA.output_path, so the results remain separated as well as the logs.
configs=(
  "src/config/inference_wrapper_westmead_12x1_sweep_balanced.yml"
  "src/config/inference_wrapper_westmead_12x1_sweep_row_identity_045.yml"
  "src/config/inference_wrapper_westmead_12x1_sweep_line_acceptance_085.yml"
  "src/config/inference_wrapper_westmead_12x1_sweep_stripes_008.yml"
  "src/config/inference_wrapper_westmead_12x1_sweep_faint_components_007.yml"
  "src/config/inference_wrapper_westmead_12x1_sweep_resample_4500.yml"
)

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "Python executable not found: ${python_bin}" >&2
  exit 2
fi

for config in "${configs[@]}"; do
  if [[ ! -f "${config}" ]]; then
    echo "Missing sweep config: ${config}" >&2
    exit 2
  fi
done

mkdir -p "${log_dir}"
summary_file="${log_dir}/summary.tsv"
printf 'sweep\tstatus\texit_code\tlog\n' >"${summary_file}"

failures=0
completed=0
total="${#configs[@]}"

for config in "${configs[@]}"; do
  completed=$((completed + 1))
  sweep_name="$(basename "${config}" .yml)"
  log_file="${log_dir}/${sweep_name}.log"

  echo
  echo "[${completed}/${total}] Starting ${sweep_name}"
  echo "Config: ${config}"
  echo "Log:    ${log_file}"

  if PYTHONUNBUFFERED=1 "${python_bin}" -m src.digitize --config "${config}" 2>&1 | tee "${log_file}"; then
    exit_code=0
    status="passed"
  else
    exit_code=$?
    status="failed"
    failures=$((failures + 1))
  fi

  printf '%s\t%s\t%s\t%s\n' "${sweep_name}" "${status}" "${exit_code}" "${log_file}" >>"${summary_file}"
  echo "[${completed}/${total}] ${sweep_name}: ${status} (exit ${exit_code})"

  if [[ "${status}" == "failed" && "${stop_on_error}" == "1" ]]; then
    echo "STOP_ON_ERROR=1; remaining sweeps were not started."
    break
  fi
done

echo
echo "Sweep summary: ${summary_file}"
if ((failures > 0)); then
  echo "Completed with ${failures} failed sweep(s)." >&2
  exit 1
fi

echo "All sweeps completed successfully."
