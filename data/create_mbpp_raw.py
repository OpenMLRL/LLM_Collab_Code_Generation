import json
import re
from datasets import load_dataset


def extract_function_name_from_code(code):
    """Extract function name from the code string."""
    # Look for function definition pattern
    match = re.search(r'def\s+(\w+)\s*\(', code)
    if match:
        return match.group(1)
    return None


def convert_mbpp_to_coophumaneval_format():
    """
    Convert MBPP sanitized dataset to coophumaneval format.
    """
    print("Loading MBPP sanitized dataset...")
    dataset = load_dataset("mbpp", "sanitized")
    
    # Get the test split
    test_data = dataset["test"]
    print(f"Loaded {len(test_data)} examples from MBPP sanitized dataset")
    
    converted_data = []
    used_function_names = set()
    
    for i, example in enumerate(test_data):
        # Extract function name from code
        function_name = extract_function_name_from_code(example["code"])
        if not function_name:
            print(f"Warning: Could not extract function name from example {i}")
            continue
        
        # Ensure function name is valid (no spaces, special chars)
        function_name = re.sub(r'[^a-zA-Z0-9_]', '_', function_name)
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', function_name):
            function_name = f"function_{i}"
        
        # Ensure unique function names
        original_function_name = function_name
        counter = 1
        while function_name in used_function_names:
            function_name = f"{original_function_name}_{counter}"
            counter += 1
        used_function_names.add(function_name)
        
        # Extract function parameters from the original code
        code = example["code"]
        params_match = re.search(r'def\s+\w+\s*\(([^)]*)\)', code)
        if params_match:
            params_str = params_match.group(1).strip()
            # Use the original parameters from the code
            prompt = f"def {function_name}({params_str}):\n    '''{example['prompt']}\n    '''\n"
        else:
            # Fallback if no parameters found
            prompt = f"def {function_name}():\n    '''{example['prompt']}\n    '''\n"
        
        # Create test cases in che format
        test_cases = []
        for test in example["test_list"]:
            # Convert assert statements to use candidate() instead of function name
            # Replace function name with candidate in assert statements
            test_with_candidate = test.replace(f"{function_name}(", "candidate(")
            
            # Also handle cases where the test might call a different function name
            # Extract any function calls in the assert statement
            # Find function calls in the assert statement
            func_calls = re.findall(r'(\w+)\s*\(', test_with_candidate)
            for func_call in func_calls:
                if func_call != "candidate" and func_call != "assert":
                    # Replace any function call with candidate
                    test_with_candidate = test_with_candidate.replace(f"{func_call}(", "candidate(")
            
            test_cases.append(test_with_candidate)
        
        # Create test string in che format: def check(candidate): + assert statements
        test_string = "def check(candidate):\n    " + "\n    ".join(test_cases)
        
        # Create the entry point (function name)
        entry_point = function_name
        
        # Create task_id in coophumaneval format
        task_id = f"MBPP/{i}"
        
        converted_example = {
            "task_id": task_id,
            "prompt": prompt,
            "test": test_string,
            "entry_point": entry_point
        }
        
        converted_data.append(converted_example)
    
    print(f"Converted {len(converted_data)} examples")
    
    # Save to JSON file
    output_file = "mbpp_raw.json"
    with open(output_file, "w") as f:
        json.dump(converted_data, f, indent=2)
    
    print(f"Saved converted data to {output_file}")
    
    # Print first example for verification
    if converted_data:
        print("\nFirst converted example:")
        print(json.dumps(converted_data[0], indent=2))
    
    return converted_data


def main():
    """Main function to run the conversion."""
    converted_data = convert_mbpp_to_coophumaneval_format()
    print(f"\nConversion complete! {len(converted_data)} examples converted.")


if __name__ == "__main__":
    main()

