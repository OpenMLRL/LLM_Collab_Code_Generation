import json

import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value
from huggingface_hub import HfApi, login


def upload_json_to_hf_dataset(json_file_path, dataset_name, hf_token=None):
    """
    Upload a JSON file to Hugging Face as a dataset with proper test split and dataset card.

    Args:
        json_file_path (str): Path to the JSON file
        dataset_name (str): Name of the dataset on Hugging Face (e.g., "OpenMLRL/MBPP")
        hf_token (str, optional): Hugging Face API token. If None, will prompt for login.
    """

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

    # Step 3: Login to Hugging Face
    if hf_token:
        login(token=hf_token)
    else:
        print("Please login to Hugging Face:")
        login()

    # Step 4: Create a DatasetDict with a TEST split (not train)
    dataset_dict = DatasetDict({"test": dataset})

    # Step 5: Calculate dataset size
    # Convert dataset to pandas to estimate size
    df = dataset.to_pandas()
    dataset_size = sum(df.memory_usage(deep=True))

    # Step 6: Create README content BEFORE pushing
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
pretty_name: MBPP
size_categories:
- 1K<n<10K
---

# MBPP Dataset

This dataset contains programming challenges from the Mostly Basic Python Problems (MBPP) benchmark, converted to the coophumaneval format for cooperative code generation evaluation.

## Dataset Structure

Each example contains:
- `task_id`: Unique identifier for the task (MBPP/{{index}})
- `prompt`: The function signature and docstring
- `test`: Test cases to verify the solution
- `entry_point`: The name of the function to implement

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{{dataset_name}}")

# Access the test split
test_data = dataset['test']

# View first example
print(test_data[0])
```

## Example

```json
{{
  "task_id": "MBPP/0",
  "prompt": "def function_name():\n    '''Write a function to find the shared elements from the given two lists.\n    '''\n",
  "test": "def check(candidate):\n    assert set(candidate((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))\n    assert set(candidate((1, 2, 3, 4),(5, 4, 3, 7))) == set((3, 4))\n    assert set(candidate((11, 12, 14, 13),(17, 15, 14, 13))) == set((13, 14))",
  "entry_point": "function_name"
}}
```

## Dataset Statistics

- **Number of examples**: {len(data)}
- **Split**: test
- **Task type**: Code generation
- **Programming language**: Python
- **Source**: MBPP (Mostly Basic Python Problems) sanitized version

## Original Dataset

This dataset is based on the MBPP (Mostly Basic Python Problems) benchmark, which consists of around 1,000 crowd-sourced Python programming problems designed to be solvable by entry level programmers. We use the sanitized version which contains 427 hand-verified examples.

## Citation

If you use this dataset, please cite both the original MBPP paper and this converted version:

```bibtex
@dataset{{mbpp2025,
  title={{MBPP: A Dataset for Cooperative Code Generation}},
  author={{ryankamiri}},
  year={{2025}},
  publisher={{Hugging Face}},
  based_on={{Austin et al., 2021}}
}}

@article{{austin2021program,
  title={{Program Synthesis with Large Language Models}},
  author={{Austin, Jacob and Odena, Augustus and Nye, Maxwell and Bosma, Maarten and Michalewski, Henryk and Dohan, David and Jiang, Ellen and Cai, Carrie and Terry, Michael and Le, Quoc and others}},
  journal={{arXiv preprint arXiv:2107.03374}},
  year={{2021}}
}}
```
"""

    # Step 7: Push to Hugging Face Hub with commit message
    print(f"Uploading dataset to {dataset_name}...")
    dataset_dict.push_to_hub(
        dataset_name,
        private=False,
        commit_message="Upload MBPP dataset with test split",
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
