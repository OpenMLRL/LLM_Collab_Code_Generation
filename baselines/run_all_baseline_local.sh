#!/usr/bin/env bash
set -euo pipefail

# Single-GPU, serial baseline runner (no Slurm).
# Runs the same 12 combos per dataset (HE/CHE) as the sbatch script.

# Runtime env
BASE_DIR="${HOME}"
CONDA_ENV="comlrl"
PROJECT_DIR_REL="LLM_Collab_Code_Generation"

# Use a single GPU by default
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Evaluation controls
SAMPLES=${SAMPLES:-20}
GENERATIONS=${GENERATIONS:-3}

# Aggregated results JSONL
RESULTS_JSON=${RESULTS_JSON:-"${BASE_DIR}/baseline_results_local.jsonl"}

# Dataset selection: tokens HE/CHE (space-separated)
DATASETS=${DATASETS:-"HE CHE"}

# Models
SA_BASE_MODEL=${SA_BASE_MODEL:-"Qwen/Qwen2.5-3B"}

# Include MAGRPO single-turn runs (default false)
INCLUDE_MAGRPO_SINGLE_TURN=${INCLUDE_MAGRPO_SINGLE_TURN:-false}

# Optional CLI: --datasets, --he-hf-dataset, --che-hf-dataset
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

cd "${BASE_DIR}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
export PYTHONPATH="${PYTHONPATH}:${PWD}/${PROJECT_DIR_REL}"

run_single_agent() {
  local token_dataset="$1"   # humaneval | coophumaneval
  local model_label="$2"     # base | ft
  local model_name="$3"
  local hf_dataset="$4"
  for mode in plain level_feedback expert_edits; do
    echo "[single-agent][${token_dataset}][${model_label}][${mode}]"
    python ${PROJECT_DIR_REL}/baselines/single_agent.py \
      --dataset "${hf_dataset}" \
      --model "${model_name}" \
      --num-turns 2 \
      --external-mode ${mode} \
      --samples ${SAMPLES} \
      --generations ${GENERATIONS} \
      --result-json "${RESULTS_JSON}"
  done
}

run_tti_multi_agent() {
  local token_dataset="$1"
  local aux_model="$2"
  local main_model="$3"
  local hf_dataset="$4"
  for tti_mode in naive_concat sequential_pipeline one_round_discussion; do
    echo "[tti][${token_dataset}][${tti_mode}]"
    python ${PROJECT_DIR_REL}/baselines/multi_agent_tti.py \
      --dataset "${hf_dataset}" \
      --mode ${tti_mode} \
      --aux-model "${aux_model}" \
      --main-model "${main_model}" \
      --samples ${SAMPLES} \
      --generations ${GENERATIONS} \
      --result-json "${RESULTS_JSON}"
  done
}

run_magrpo_multi_agent() {
  local token_dataset="$1"
  local aux_model="$2"
  local main_model="$3"
  local hf_dataset="$4"
  if [[ "${INCLUDE_MAGRPO_SINGLE_TURN}" == "true" ]]; then
    echo "[ours][${token_dataset}][single-turn]"
    python ${PROJECT_DIR_REL}/baselines/multi_agent_ours.py \
      --dataset "${hf_dataset}" \
      --aux-model "${aux_model}" \
      --main-model "${main_model}" \
      --num-turns 1 \
      --samples ${SAMPLES} \
      --generations ${GENERATIONS} \
      --result-json "${RESULTS_JSON}"
  fi
  for mode in plain level_feedback expert_edits; do
    echo "[ours][${token_dataset}][2-turn][${mode}]"
    python ${PROJECT_DIR_REL}/baselines/multi_agent_ours.py \
      --dataset "${hf_dataset}" \
      --aux-model "${aux_model}" \
      --main-model "${main_model}" \
      --num-turns 2 \
      --external-mode ${mode} \
      --samples ${SAMPLES} \
      --generations ${GENERATIONS} \
      --result-json "${RESULTS_JSON}"
  done
}

for token in ${DATASETS}; do
  case "${token}" in
    HE|he|Humaneval|humaneval)
      ds_token="humaneval"
      SA_FT_MODEL="OpenMLRL/Single_agent_HE"
      AUX_MODEL="OpenMLRL/DS-HE-r3.04-Agent-0"
      MAIN_MODEL="OpenMLRL/DS-HE-r3.04-Agent-1"
      HF_DATASET="${HE_HF_DATASET:-openai/openai_humaneval}" ;;
    CHE|che|CoopHumanEval|coophumaneval)
      ds_token="coophumaneval"
      SA_FT_MODEL="OpenMLRL/Single_agent_CHE"
      AUX_MODEL="OpenMLRL/DS-CHE-r3.21-Agent-0"
      MAIN_MODEL="OpenMLRL/DS-CHE-r3.21-Agent-1"
      HF_DATASET="${CHE_HF_DATASET:-OpenMLRL/CoopHumanEval}" ;;
    *)
      echo "Unsupported dataset token: ${token} (use HE or CHE)" 1>&2; exit 1 ;;
  esac

  # Single-agent: base model and GRPO-FT model (2-turn x 3 modes)
  run_single_agent "${ds_token}" base "${SA_BASE_MODEL}" "${HF_DATASET}"
  run_single_agent "${ds_token}" ft   "${SA_FT_MODEL}"  "${HF_DATASET}"

  # TTI baselines (3 modes)
  run_tti_multi_agent "${ds_token}" "${AUX_MODEL}" "${MAIN_MODEL}" "${HF_DATASET}"

  # MAGRPO (ours): 2-turn x 3 modes (+ optional single-turn)
  run_magrpo_multi_agent "${ds_token}" "${AUX_MODEL}" "${MAIN_MODEL}" "${HF_DATASET}"
done

echo "All runs completed. Aggregated JSONL: ${RESULTS_JSON}"
