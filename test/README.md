## Run External Output Test (dump_external_prompts.py)

- Recommended Method (Automatically sets environment and aggregates output into two files)
  - `bash LLM_Collaboration_with_MARL/test/run_external.sh`

- Directly Run Script (if located in repository root directory)
  - Optional: `conda activate comlrl`
  - `export PYTHONPATH="${PYTHONPATH}:$(pwd)/LLM_Collaboration_with_MARL"`
  - `python3 LLM_Collaboration_with_MARL/test/dump_external_prompts.py`

- Notes
  - For `expert_edits` mode, the script now defaults to calling the real expert model (`deepseek-coder`).
    - Set `DEEPSEEK_API_KEY` in your environment; otherwise requests will fail and may fall back to stub where applicable.
    - To force offline stub, pass `--offline` to the script or set `OFFLINE=1`.
  - Output files located in `LLM_Collaboration_with_MARL/test/`:
    - `prompts_sa.txt` (single agent)
    - `prompts_ma.txt` (multi agent)
