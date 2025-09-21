#!/usr/bin/env bash
set -euo pipefail

# Run the external prompt generation test and list outputs.
# Environment setup mirrors your sbatch reference (no srun involved).

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"  # LLM_Collaboration_with_MARL

# Optional: conda activation (safe if conda exists)
if command -v conda >/dev/null 2>&1; then
  # Load conda shell hook
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  # Load user bashrc for aliases/env if present
  [ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
  # Activate target env if available
  conda env list | grep -qE '^comlrl\s' && conda activate comlrl || true
fi

export PYTHONPATH="${PYTHONPATH:-}:$REPO_DIR"

PYTHON_BIN=${PYTHON:-python3}
echo "Using Python: $PYTHON_BIN"
echo "Repo dir:   $REPO_DIR"
echo "Script dir: $SCRIPT_DIR"

cd "$REPO_DIR"

# If REAL_EXPERT=1, request real expert edits with selected model (default deepseek-coder)
EXPERT_ARGS=()
if [[ "${REAL_EXPERT:-}" == "1" ]]; then
  : "${EXPERT_MODEL:=deepseek-coder}"
  if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
    echo "[warn] REAL_EXPERT=1 but DEEPSEEK_API_KEY is not set; request may fail." >&2
  fi
  EXPERT_ARGS+=("--real-expert" "--expert-model" "$EXPERT_MODEL")
fi

"$PYTHON_BIN" "$SCRIPT_DIR/dump_external_prompts.py" "${EXPERT_ARGS[@]}"

echo
echo "Generated files:"
ls -1 "$SCRIPT_DIR"/prompts_sa.txt "$SCRIPT_DIR"/prompts_ma.txt 2>/dev/null || echo "No files generated."
