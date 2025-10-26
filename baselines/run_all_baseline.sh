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
GENERATIONS=${GENERATIONS:-15}

# Where to write aggregated results (JSON Lines). Can be overridden via env.
RESULTS_JSON=${RESULTS_JSON:-"${BASE_DIR}/baseline_results.jsonl"}

# Dataset selection interface
# Accepts tokens: HE, CHE (case-insensitive). Space-separated list.
# Example: ./run_all_baseline.sh --datasets "HE CHE"  or  DATASETS="HE" ./run_all_baseline.sh
DATASETS=${DATASETS:-"HE CHE"}

# Models
# Single-agent base (fixed) model; change if desired
SA_BASE_MODEL=${SA_BASE_MODEL:-"Qwen/Qwen2.5-3B"}
# Multi-agent base (fixed) models (aux/main both fixed to same base)
MA_BASE_AUX_MODEL=${MA_BASE_AUX_MODEL:-"Qwen/Qwen2.5-3B"}
MA_BASE_MAIN_MODEL=${MA_BASE_MAIN_MODEL:-"Qwen/Qwen2.5-3B"}

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

# ---------- Single-agent helpers ----------
run_sa_turn1() {
  local dataset="$1"      # humaneval | coophumaneval
  local model_label="$2"  # base | ft
  local model_name="$3"
  local hf_dataset="$4"
  submit "sa1-${dataset}-${model_label}" \
    python ${PROJECT_DIR_REL}/baselines/single_agent.py \
      --dataset "${hf_dataset}" \
      --model "${model_name}" \
      --num-turns 1
}

run_sa_turn2() {
  local dataset="$1"
  local model_label="$2"
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

# ---------- Multi-agent TTI baselines ----------
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

# ---------- Multi-agent (ours) helpers ----------
run_ma_turn1() {
  local dataset="$1"
  local model_label="$2"   # base | ft
  local aux_model="$3"
  local main_model="$4"
  local hf_dataset="$5"
  submit "ma1-${dataset}-${model_label}" \
    python ${PROJECT_DIR_REL}/baselines/multi_agent_ours.py \
      --dataset "${hf_dataset}" \
      --aux-model "${aux_model}" \
      --main-model "${main_model}" \
      --num-turns 1
}

run_ma_turn2() {
  local dataset="$1"
  local model_label="$2"   # base | ft
  local aux_model="$3"
  local main_model="$4"
  local hf_dataset="$5"
  for mode in plain level_feedback expert_edits; do
    submit "ma2-${dataset}-${model_label}-${mode}" \
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
    # Resolve dataset-specific models (fine-tuned names from your screenshot)
    if [[ "${dataset}" == "humaneval" ]]; then
      # Single-agent fine-tuned
      SA_FT_1T_MODEL="OpenMLRL/sa_1t_he"
      SA_FT_2T_MODEL_plain="OpenMLRL/sa_he_2t_plain_slice"
      SA_FT_2T_MODEL_level_feedback="OpenMLRL/sa_he_2t_level_feedback_slice"
      SA_FT_2T_MODEL_expert_edits="OpenMLRL/sa_he_2t_expert_edits_slice"
      # Multi-agent fine-tuned
      MA_FT_1T_AUX="OpenMLRL/he_1t_aux";   MA_FT_1T_MAIN="OpenMLRL/he_1t_main"
      MA_FT_2T_AUX_plain="OpenMLRL/he_2t_plain_aux";   MA_FT_2T_MAIN_plain="OpenMLRL/he_2t_plain_main"
      MA_FT_2T_AUX_level_feedback="OpenMLRL/he_2t_level_feedback_aux"; MA_FT_2T_MAIN_level_feedback="OpenMLRL/he_2t_level_feedback_main"
      # Expert-edits models are now available
      MA_FT_2T_AUX_expert_edits="OpenMLRL/he_2t_expert_edits_aux"; MA_FT_2T_MAIN_expert_edits="OpenMLRL/he_2t_expert_edits_main"
      HF_DATASET="${HE_HF_DATASET:-openai/openai_humaneval}"
    else
      SA_FT_1T_MODEL="OpenMLRL/sa_1t_che"
      SA_FT_2T_MODEL_plain="OpenMLRL/sa_che_2t_plain_slice"
      SA_FT_2T_MODEL_level_feedback="OpenMLRL/sa_che_2t_level_feedback_slice"
      SA_FT_2T_MODEL_expert_edits="OpenMLRL/sa_che_2t_expert_edits_slice"
      MA_FT_1T_AUX="OpenMLRL/che_1t_aux";  MA_FT_1T_MAIN="OpenMLRL/che_1t_main"
      MA_FT_2T_AUX_plain="OpenMLRL/che_2t_plain_aux";  MA_FT_2T_MAIN_plain="OpenMLRL/che_2t_plain_main"
      MA_FT_2T_AUX_level_feedback="OpenMLRL/che_2t_level_feedback_aux"; MA_FT_2T_MAIN_level_feedback="OpenMLRL/che_2t_level_feedback_main"
      MA_FT_2T_AUX_expert_edits="OpenMLRL/che_2t_expert_edits_aux"; MA_FT_2T_MAIN_expert_edits="OpenMLRL/che_2t_expert_edits_main"
      HF_DATASET="${CHE_HF_DATASET:-OpenMLRL/CoopHumanEval}"
    fi

    # ---------------- Single-agent (8 per dataset) ----------------
    # num_turn=1: fixed, fine-tuned
    run_sa_turn1 "${dataset}" base "${SA_BASE_MODEL}" "${HF_DATASET}"
    run_sa_turn1 "${dataset}" ft   "${SA_FT_1T_MODEL}" "${HF_DATASET}"
    # num_turn=2: fixed x3 modes; fine-tuned x3 modes
    run_sa_turn2 "${dataset}" base "${SA_BASE_MODEL}" "${HF_DATASET}"
    # fine-tuned 2-turn per mode
    submit "sa2-${dataset}-ft-plain" \
      python ${PROJECT_DIR_REL}/baselines/single_agent.py --dataset "${HF_DATASET}" --model "${SA_FT_2T_MODEL_plain}" --num-turns 2 --external-mode plain
    submit "sa2-${dataset}-ft-level_feedback" \
      python ${PROJECT_DIR_REL}/baselines/single_agent.py --dataset "${HF_DATASET}" --model "${SA_FT_2T_MODEL_level_feedback}" --num-turns 2 --external-mode level_feedback
    submit "sa2-${dataset}-ft-expert_edits" \
      python ${PROJECT_DIR_REL}/baselines/single_agent.py --dataset "${HF_DATASET}" --model "${SA_FT_2T_MODEL_expert_edits}" --num-turns 2 --external-mode expert_edits

    # ---------------- Multi-agent TTI (3 per dataset) ----------------
    run_tti_multi_agent "${dataset}" "${MA_BASE_AUX_MODEL}" "${MA_BASE_MAIN_MODEL}" "${HF_DATASET}"

    # ---------------- Multi-agent ours (8 per dataset) ----------------
    # num_turn=1: fixed, GRPO-FT
    run_ma_turn1 "${dataset}" base "${MA_BASE_AUX_MODEL}" "${MA_BASE_MAIN_MODEL}" "${HF_DATASET}"
    run_ma_turn1 "${dataset}" ft   "${MA_FT_1T_AUX}" "${MA_FT_1T_MAIN}" "${HF_DATASET}"
    # num_turn=2: fixed x3 modes
    run_ma_turn2 "${dataset}" base "${MA_BASE_AUX_MODEL}" "${MA_BASE_MAIN_MODEL}" "${HF_DATASET}"
    # num_turn=2: GRPO-FT per mode (expert_edits falls back to plain per your note)
    for m in plain level_feedback expert_edits; do
      aux_var="MA_FT_2T_AUX_${m}"
      main_var="MA_FT_2T_MAIN_${m}"
      aux_val="${!aux_var}"
      main_val="${!main_var}"
      submit "ma2-${dataset}-ft-${m}" \
        python ${PROJECT_DIR_REL}/baselines/multi_agent_ours.py --dataset "${HF_DATASET}" --aux-model "${aux_val}" --main-model "${main_val}" --num-turns 2 --external-mode ${m}
    done
  done
}

main "$@"
