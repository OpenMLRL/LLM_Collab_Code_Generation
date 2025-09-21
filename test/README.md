## Run External Output Test (dump_external_prompts.py)

- Recommended Method (Automatically sets environment and aggregates output into two files)
  - `bash LLM_Collaboration_with_MARL/test/run_external.sh`

- Directly Run Script (if located in repository root directory)
  - Optional: `conda activate comlrl`
  - `export PYTHONPATH="${PYTHONPATH}:$(pwd)/LLM_Collaboration_with_MARL"`
  - `python3 LLM_Collaboration_with_MARL/test/dump_external_prompts.py`

- Notes
  - Script runs offline: stubs `anthropic`/`openai` before importing and overrides `expert_edits.add_expert_edits`
  - Output files located in `LLM_Collaboration_with_MARL/test/`:
    - `prompts_sa.txt` (single agent)
    - `prompts_ma.txt` (multi agent)
