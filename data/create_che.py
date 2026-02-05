import json

 
from datasets import Dataset, DatasetDict, Features, Value
from huggingface_hub import HfApi, login


def upload_json_to_hf_dataset(json_file_path, dataset_name, hf_token=None):
    """
    Upload a JSON file to Hugging Face as a dataset with proper test split and dataset card.

    Args:
        json_file_path (str): Path to the JSON file
        dataset_name (str): Name of the dataset on Hugging Face (e.g., "username/CoopHumanEval")
        hf_token (str, optional): Hugging Face API token. If None, will prompt for login.
    """

    # Load raw JSON and normalize schema

    # Step 1: Load the JSON file
    print(f"Loading JSON file from {json_file_path}...")
    with open(json_file_path, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} examples from JSON file")

    # Step 2: Convert to Hugging Face Dataset
    print("Converting to Hugging Face Dataset format...")

    # Create a dictionary with lists for each field
    dataset_dict = {
        "task_id": [],
        "prompt": [],
        "test": [],
        "entry_point": [],
    }

    for item in data:
        dataset_dict["task_id"].append(item["task_id"])
        dataset_dict["prompt"].append(item["prompt"])
        dataset_dict["test"].append(item["test"])
        dataset_dict["entry_point"].append(item["entry_point"])

    # Define features explicitly
    features = Features(
        {
            "task_id": Value("string"),
            "prompt": Value("string"),
            "test": Value("string"),
            "entry_point": Value("string"),
        }
    )

    # Create the dataset with features
    dataset = Dataset.from_dict(dataset_dict, features=features)

    # Authenticate to Hugging Face
    if hf_token:
        login(token=hf_token)
    else:
        print("Please login to Hugging Face:")
        login()

    # Step 4: Create a DatasetDict with a TEST split (not train)
    dataset_dict = DatasetDict({"test": dataset})  # Changed from 'train' to 'test'

    # Step 5: Calculate dataset size
    # Convert dataset to pandas to estimate size
    df = dataset.to_pandas()
    dataset_size = sum(df.memory_usage(deep=True))

    # Build dataset card (README) with metadata
    readme_content = f"""---
dataset_info:
  features:
  - name: task_id
    dtype: string
  - name: prompt
    dtype: string
  - name: test
    dtype: string
  - name: entry_point
    dtype: string
  splits:
  - name: test
    num_bytes: {dataset_size}
    num_examples: {len(data)}
  download_size: {dataset_size}
  dataset_size: {dataset_size}
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
license: apache-2.0
task_categories:
- text-generation
language:
- en
tags:
- code
- python
- programming
- code-generation
pretty_name: CoopHumanEval
size_categories:
- n<1K
---

# CoopHumanEval Dataset

This dataset contains programming challenges designed for cooperative code generation evaluation.

## Dataset Structure

Each example contains:
- `task_id`: Unique identifier for the task
- `prompt`: The function signature and docstring with examples
- `test`: Test cases to verify the solution
- `entry_point`: The name of the function to implement

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{dataset_name}")

# Access the test split
test_data = dataset['test']

# View first example
print(test_data[0])
```

## Example

```json
{{
  "task_id": "CoopHumanEval/0",
  "prompt": "def find_nth_prime_cube(n):\\n    \\"\\"\\"Please find the cube of the nth prime number.\\n    \\n    Examples:\\n    >>> find_nth_prime_cube(1)\\n    8\\n    >>> find_nth_prime_cube(2)\\n    27\\n    >>> find_nth_prime_cube(3)\\n    125\\n    \\"\\"\\"\\n",
  "test": "def check(candidate):\\n    assert candidate(1) == 8\\n    ...",
  "entry_point": "find_nth_prime_cube"
}}
```

## Dataset Statistics

- **Number of examples**: {len(data)}
- **Split**: test
- **Task type**: Code generation
- **Programming language**: Python

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{{coophumaneval2024,
  title={{CoopHumanEval: A Dataset for Cooperative Code Generation}},
  author={{LovelyBuggies}},
  year={{2024}},
  publisher={{Hugging Face}}
}}
```
"""

    # Step 7: Push to Hugging Face Hub with commit message
    print(f"Uploading dataset to {dataset_name}...")
    dataset_dict.push_to_hub(
        dataset_name,
        private=False,
        commit_message="Upload CoopHumanEval dataset with test split",
    )

    # Step 8: Upload README separately to ensure it's properly saved
    api = HfApi()
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=dataset_name,
        repo_type="dataset",
        commit_message="Add dataset card with proper metadata",
    )

    print(
        f"Successfully uploaded dataset to https://huggingface.co/datasets/{dataset_name}"
    )
    print("The dataset card should now be visible on the Hugging Face website.")


def main():
    # Configuration
    json_file_path = "che_raw.json"

    # Your Hugging Face username
    dataset_name = "LovelyBuggies/CoopHumanEval"

    # Optional: Set your Hugging Face token here, or leave as None to login interactively
    hf_token = None  # or "hf_..."

    # Upload the dataset
    upload_json_to_hf_dataset(json_file_path, dataset_name, hf_token)


if __name__ == "__main__":
    main()


# Alternative: Verify dataset after upload
def verify_dataset(dataset_name):
    """
    Verify the uploaded dataset by loading it back.
    """
    from datasets import load_dataset

    print(f"\nVerifying dataset {dataset_name}...")
    dataset = load_dataset(dataset_name)

    print(f"Available splits: {list(dataset.keys())}")
    print(f"Number of examples in test split: {len(dataset['test'])}")
    print(f"\nFirst example:")
    print(dataset["test"][0])

    return dataset
