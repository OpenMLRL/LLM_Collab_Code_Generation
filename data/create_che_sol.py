import json
 
import re
from datetime import datetime
from typing import Any, Dict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import utilities (assuming these are in your project structure)
from rewards.code_utils import cleanup_code, extract_specific_function


def extract_function_params_from_prompt(prompt_text):
    """Extract function parameters from the prompt text."""
    # Match pattern like: def function_name(param1: type1, param2: type2) -> return_type:
    match = re.search(r"def\s+\w+\s*\(([^)]+)\)", prompt_text)
    if match:
        params_str = match.group(1)
        # Clean up parameters
        params = [p.strip() for p in params_str.split(",") if p.strip()]
        return params
    return []


def aux_function_formatter(example: Dict[str, Any]) -> str:
    """Formatter for the auxiliary function generator (Agent 0)."""
    prompt = example.get("prompt", "")
    params = extract_function_params_from_prompt(prompt)

    if not params:
        return "Error: Could not extract function information from prompt."

    params_str = ", ".join(params)

    return f"""Create a helper function for this coding problem.

Problem:
{prompt}

IMPORTANT INSTRUCTIONS:
- Output ONLY the function code, no explanations or examples
- Do NOT include markdown code blocks (```python)
- Create a helper function named 'aux' that can assist the main function

def aux(...):
    # your function code here
    return result
"""


def main_function_formatter(example: Dict[str, Any]) -> str:
    """Formatter for the main function generator (Agent 1)."""
    prompt = example.get("prompt", "")
    entry_point = example.get("entry_point", "")
    params = extract_function_params_from_prompt(prompt)

    if not params or not entry_point:
        return "Error: Could not extract function information from prompt."

    params_str = ", ".join(params)

    return f"""Solve this coding problem by implementing the required function.

Problem:
{prompt}

You have access to a helper function: aux(...)

IMPORTANT INSTRUCTIONS:
- Output ONLY the function code, no explanations or examples
- Do NOT include markdown code blocks (```python)
- Implement ONLY the '{entry_point}' function as specified
- You can call aux() within your function if helpful

def {entry_point}({params_str}):
    # your function code here
    return result
"""


def load_agent_from_hf(model_name: str):
    """Load an agent model and tokenizer from Hugging Face Hub."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with GPU support if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        if device == "cpu":
            model = model.to(device)

        return model, tokenizer

    except Exception as e:
        print(f"Failed to load {model_name}: {str(e)}")
        return None, None


def generate_completion(
    model, tokenizer, prompt: str, max_new_tokens: int = 256
) -> str:
    """Generate a completion from the model given a prompt."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        completion_tokens = outputs[0][input_length:]
        completion = tokenizer.decode(completion_tokens, skip_special_tokens=True)

        return completion.strip()

    except Exception as e:
        return f"Error during generation: {str(e)}"


def main():
    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    print("Loading CoopHumanEval dataset...")
    dataset_name = "LovelyBuggies/CoopHumanEval"
    dataset = load_dataset(dataset_name, split="test")

    print(f"Total examples: {len(dataset)}")

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    aux_model_name = "LovelyBuggies/2xQwen2.5-Coder-3B-Griffin-Aux"
    main_model_name = "LovelyBuggies/2xQwen2.5-Coder-3B-Griffin-Main"
    print("Loading models...")
    agent_0, tokenizer_0 = load_agent_from_hf(aux_model_name)  # Aux model
    agent_1, tokenizer_1 = load_agent_from_hf(main_model_name)  # Main model

    if not agent_0 or not agent_1:
        print("Failed to load models. Exiting.")
        return

    # ------------------------------------------------------------------
    # Generate solutions for all examples
    # ------------------------------------------------------------------
    results = []

    print("Generating solutions...")

    for i, example in enumerate(dataset):
        # Show progress without details
        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{len(dataset)}")

        try:
            # Generate auxiliary function (Agent 0)
            aux_prompt = aux_function_formatter(example)
            aux_completion = generate_completion(agent_0, tokenizer_0, aux_prompt)

            # Generate main function (Agent 1)
            main_prompt = main_function_formatter(example)
            main_completion = generate_completion(agent_1, tokenizer_1, main_prompt)

            # Extract specific functions
            aux_func = extract_specific_function(cleanup_code(aux_completion), "aux")
            main_func = extract_specific_function(
                cleanup_code(main_completion), example["entry_point"]
            )

            # Combine the code
            combined_code = ""
            if aux_func:
                combined_code += aux_func
            if main_func:
                if combined_code:
                    combined_code += "\n\n"
                combined_code += main_func

            # Create result in the specified format
            result = {
                "task_id": example.get("task_id", ""),
                "prompt": example.get("prompt", ""),
                "test": example.get("test", ""),
                "entry_point": example.get("entry_point", ""),
                "mlrl_solution": combined_code,
            }
            results.append(result)

        except Exception as e:
            # If error, still add the entry with empty solution
            result = {
                "task_id": example.get("task_id", ""),
                "prompt": example.get("prompt", ""),
                "test": example.get("test", ""),
                "entry_point": example.get("entry_point", ""),
                "mlrl_solution": "",
            }
            results.append(result)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"coophumaneval_mlrl_solutions_{timestamp}.json"

    with open(output_filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nCompleted! Results saved to: {output_filename}")


if __name__ == "__main__":
    main()
