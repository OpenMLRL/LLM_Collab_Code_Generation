#!/usr/bin/env bash
set -euo pipefail

# Batch settings (adjust as needed)
ACCOUNT="bevi-dtai-gh"
PARTITION="ghx4"
NODES=1
GPUS_PER_NODE=1
NTASKS=1
NTASKS_PER_NODE=1
CPUS_PER_TASK=64
MEM="100g"
TIME="12:00:00"

# Runtime env
# Use $HOME to avoid tilde not expanding inside variables
BASE_DIR="${HOME}"
CONDA_ENV="comlrl"
PROJECT_DIR_REL="LLM_Collab_Code_Generation"

# Evaluation controls (unified here)
SAMPLES=${SAMPLES:-20}
GENERATIONS=${GENERATIONS:-3}

# Where to write aggregated results (JSON Lines). Can be overridden via env.
RESULTS_JSON=${RESULTS_JSON:-"${BASE_DIR}/baseline_results.jsonl"}

# Dataset selection interface
# Accepts tokens: HE, CHE (case-insensitive). Space-separated list.
# Example: ./run_all_baseline.sh --datasets "HE CHE"  or  DATASETS="HE" ./run_all_baseline.sh
DATASETS=${DATASETS:-"HE CHE"}

# Models
# Single-agent base (fixed) model; change if desired
SA_BASE_MODEL=${SA_BASE_MODEL:-"Qwen/Qwen2.5-3B"}

# Whether to include MAGRPO single-turn jobs in addition to the 12-per-dataset set
# Set to "true" to include; default is "false" to match the requested 12 combos per dataset
INCLUDE_MAGRPO_SINGLE_TURN=${INCLUDE_MAGRPO_SINGLE_TURN:-false}

submit() {
  local job_name="$1"; shift
  local py_cmd="$*"
  sbatch \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --nodes="${NODES}" \
    --gpus-per-node="${GPUS_PER_NODE}" \
    --ntasks="${NTASKS}" \
    --ntasks-per-node="${NTASKS_PER_NODE}" \
    --cpus-per-task="${CPUS_PER_TASK}" \
    --mem="${MEM}" \
    --time="${TIME}" \
    --job-name "${job_name}" \
    --wrap "cd ${BASE_DIR} && source \$(conda info --base)/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && export PYTHONPATH=\"\$PYTHONPATH:\$PWD/${PROJECT_DIR_REL}\" && ${py_cmd} --samples ${SAMPLES} --generations ${GENERATIONS} --result-json ${RESULTS_JSON}"
}

run_single_agent() {
  local dataset="$1"      # humaneval | coophumaneval
  local model_label="$2"  # base | ft
  local model_name="$3"
  local hf_dataset="$4"
  for mode in plain level_feedback expert_edits; do
    submit "sa2-${dataset}-${model_label}-${mode}" \
      python ${PROJECT_DIR_REL}/baselines/single_agent.py \
        --dataset "${hf_dataset}" \
        --model "${model_name}" \
        --num-turns 2 \
        --external-mode ${mode}
  done
}

run_tti_multi_agent() {
  local dataset="$1"
  local aux_model="$2"
  local main_model="$3"
  local hf_dataset="$4"
  for tti_mode in naive_concat sequential_pipeline one_round_discussion; do
    submit "tti-${dataset}-${tti_mode}" \
      python ${PROJECT_DIR_REL}/baselines/multi_agent_tti.py \
        --dataset "${hf_dataset}" \
        --mode ${tti_mode} \
        --aux-model "${aux_model}" \
        --main-model "${main_model}"
  done
}

run_magrpo_multi_agent() {
  local dataset="$1"
  local aux_model="$2"
  local main_model="$3"
  local hf_dataset="$4"
  # Optional single-turn (disabled by default to keep 12 combos per dataset)
  if [[ "${INCLUDE_MAGRPO_SINGLE_TURN}" == "true" ]]; then
    submit "mag1-${dataset}" \
      python ${PROJECT_DIR_REL}/baselines/multi_agent_ours.py \
        --dataset "${hf_dataset}" \
        --aux-model "${aux_model}" \
        --main-model "${main_model}" \
        --num-turns 1
  fi
  # Two-turn with external transitions
  for mode in plain level_feedback expert_edits; do
    submit "mag2-${dataset}-${mode}" \
      python ${PROJECT_DIR_REL}/baselines/multi_agent_ours.py \
        --dataset "${hf_dataset}" \
        --aux-model "${aux_model}" \
        --main-model "${main_model}" \
        --num-turns 2 \
        --external-mode ${mode}
  done
}

main() {
  # Parse optional CLI args
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --datasets)
        DATASETS="$2"; shift 2 ;;
      --he-hf-dataset)
        HE_HF_DATASET="$2"; shift 2 ;;
      --che-hf-dataset)
        CHE_HF_DATASET="$2"; shift 2 ;;
      *)
        echo "Unknown argument: $1" 1>&2; exit 1 ;;
    esac
  done

  for token in ${DATASETS}; do
    # Normalize dataset token to baseline flag values
    case "${token}" in
      HE|he|Humaneval|humaneval)
        dataset="humaneval" ;;
      CHE|che|CoopHumanEval|coophumaneval)
        dataset="coophumaneval" ;;
      *)
        echo "Unsupported dataset token: ${token} (use HE or CHE)" 1>&2; exit 1 ;;
    esac
    # Resolve dataset-specific models
    if [[ "${dataset}" == "humaneval" ]]; then
      SA_FT_MODEL="OpenMLRL/Single_agent_HE"
      AUX_MODEL="OpenMLRL/DS-HE-r3.04-Agent-0"
      MAIN_MODEL="OpenMLRL/DS-HE-r3.04-Agent-1"
      HF_DATASET="${HE_HF_DATASET:-openai/openai_humaneval}"
    else
      SA_FT_MODEL="OpenMLRL/Single_agent_CHE"
      AUX_MODEL="OpenMLRL/DS-CHE-r3.21-Agent-0"
      MAIN_MODEL="OpenMLRL/DS-CHE-r3.21-Agent-1"
      HF_DATASET="${CHE_HF_DATASET:-OpenMLRL/CoopHumanEval}"
    fi

    # Single-agent: base model and GRPO-FT model, each with 2-turn + 3 external modes
    run_single_agent "${dataset}" base "${SA_BASE_MODEL}" "${HF_DATASET}"
    run_single_agent "${dataset}" ft   "${SA_FT_MODEL}"  "${HF_DATASET}"

    # Multi-agent TTI baselines
    run_tti_multi_agent "${dataset}" "${AUX_MODEL}" "${MAIN_MODEL}" "${HF_DATASET}"

    # MAGRPO (ours): two-turn with external modes; optional single-turn
    run_magrpo_multi_agent "${dataset}" "${AUX_MODEL}" "${MAIN_MODEL}" "${HF_DATASET}"
  done
}

main "$@"
