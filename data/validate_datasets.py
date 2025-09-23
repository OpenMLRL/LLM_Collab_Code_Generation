#!/usr/bin/env python3
"""
Script to validate that MBPP dataset can be used as a drop-in replacement for CHE dataset.
Compares structure, format, and content to ensure compatibility.
"""

import json
import re
from typing import Dict, List, Any, Tuple

def load_json_dataset(filepath: str) -> List[Dict[str, Any]]:
    """Load a JSON dataset file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def validate_dataset_structure(dataset: List[Dict[str, Any]], dataset_name: str) -> Dict[str, Any]:
    """Validate the basic structure of a dataset."""
    print(f"\n{'='*60}")
    print(f"Validating {dataset_name} structure...")
    print(f"{'='*60}")
    
    if not dataset:
        return {"valid": False, "error": "Dataset is empty"}
    
    # Check if it's a list
    if not isinstance(dataset, list):
        return {"valid": False, "error": "Dataset is not a list"}
    
    # Check first example structure
    first_example = dataset[0]
    required_fields = ["task_id", "prompt", "test", "entry_point"]
    
    missing_fields = [field for field in required_fields if field not in first_example]
    if missing_fields:
        return {"valid": False, "error": f"Missing required fields: {missing_fields}"}
    
    # Check data types
    field_types = {
        "task_id": str,
        "prompt": str,
        "test": str,
        "entry_point": str
    }
    
    type_errors = []
    for field, expected_type in field_types.items():
        if not isinstance(first_example[field], expected_type):
            type_errors.append(f"{field} should be {expected_type.__name__}, got {type(first_example[field]).__name__}")
    
    if type_errors:
        return {"valid": False, "error": f"Type errors: {type_errors}"}
    
    # Validate all examples have same structure
    all_examples_valid = True
    for i, example in enumerate(dataset):
        if not all(field in example for field in required_fields):
            print(f"⚠️  Example {i} missing required fields")
            all_examples_valid = False
    
    result = {
        "valid": True,
        "total_examples": len(dataset),
        "required_fields": required_fields,
        "all_examples_valid": all_examples_valid
    }
    
    print(f"✅ Structure validation passed")
    print(f"   Total examples: {len(dataset)}")
    print(f"   Required fields: {required_fields}")
    print(f"   All examples valid: {all_examples_valid}")
    
    return result

def validate_task_ids(dataset: List[Dict[str, Any]], dataset_name: str) -> Dict[str, Any]:
    """Validate task_id format and uniqueness."""
    print(f"\nValidating {dataset_name} task_ids...")
    
    task_ids = [example["task_id"] for example in dataset]
    unique_task_ids = set(task_ids)
    
    # Check uniqueness
    duplicates = len(task_ids) - len(unique_task_ids)
    
    # Check format
    format_errors = []
    for task_id in task_ids:
        if not isinstance(task_id, str) or not task_id:
            format_errors.append(f"Empty or non-string task_id: {task_id}")
    
    result = {
        "total_task_ids": len(task_ids),
        "unique_task_ids": len(unique_task_ids),
        "duplicates": duplicates,
        "format_errors": format_errors,
        "valid": duplicates == 0 and len(format_errors) == 0
    }
    
    print(f"   Total task_ids: {len(task_ids)}")
    print(f"   Unique task_ids: {len(unique_task_ids)}")
    print(f"   Duplicates: {duplicates}")
    print(f"   Format errors: {len(format_errors)}")
    
    if result["valid"]:
        print("✅ Task ID validation passed")
    else:
        print("❌ Task ID validation failed")
    
    return result

def validate_prompts(dataset: List[Dict[str, Any]], dataset_name: str) -> Dict[str, Any]:
    """Validate prompt format and content."""
    print(f"\nValidating {dataset_name} prompts...")
    
    issues = []
    function_definitions = []
    
    for i, example in enumerate(dataset):
        prompt = example["prompt"]
        
        # Check if prompt starts with function definition
        if not prompt.strip().startswith("def "):
            issues.append(f"Example {i}: Prompt doesn't start with 'def '")
        
        # Extract function name
        match = re.search(r'def\s+(\w+)\s*\(', prompt)
        if match:
            function_definitions.append(match.group(1))
        else:
            issues.append(f"Example {i}: Could not extract function name from prompt")
        
        # Check if prompt has docstring
        if '"""' not in prompt and "'''" not in prompt:
            issues.append(f"Example {i}: Prompt missing docstring")
    
    result = {
        "total_prompts": len(dataset),
        "function_definitions_found": len(function_definitions),
        "issues": issues,
        "valid": len(issues) == 0
    }
    
    print(f"   Total prompts: {len(dataset)}")
    print(f"   Function definitions found: {len(function_definitions)}")
    print(f"   Issues: {len(issues)}")
    
    if result["valid"]:
        print("✅ Prompt validation passed")
    else:
        print("❌ Prompt validation failed")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"   - {issue}")
    
    return result

def validate_tests(dataset: List[Dict[str, Any]], dataset_name: str) -> Dict[str, Any]:
    """Validate test format and content."""
    print(f"\nValidating {dataset_name} tests...")
    
    issues = []
    test_patterns = []
    
    for i, example in enumerate(dataset):
        test = example["test"]
        
        # Check if test starts with "def check(candidate):"
        if not test.strip().startswith("def check(candidate):"):
            issues.append(f"Example {i}: Test doesn't start with 'def check(candidate):'")
        
        # Check for assert statements
        assert_count = test.count("assert ")
        if assert_count == 0:
            issues.append(f"Example {i}: No assert statements found")
        
        # Check for candidate() calls
        if "candidate(" not in test:
            issues.append(f"Example {i}: No candidate() calls found")
        
        # Check for function name calls (should not be present)
        entry_point = example["entry_point"]
        if f"{entry_point}(" in test:
            issues.append(f"Example {i}: Found {entry_point}() call instead of candidate()")
        
        test_patterns.append({
            "assert_count": assert_count,
            "has_candidate": "candidate(" in test,
            "has_function_name": f"{entry_point}(" in test
        })
    
    result = {
        "total_tests": len(dataset),
        "issues": issues,
        "test_patterns": test_patterns,
        "valid": len(issues) == 0
    }
    
    print(f"   Total tests: {len(dataset)}")
    print(f"   Issues: {len(issues)}")
    
    if result["valid"]:
        print("✅ Test validation passed")
    else:
        print("❌ Test validation failed")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"   - {issue}")
    
    return result

def validate_entry_points(dataset: List[Dict[str, Any]], dataset_name: str) -> Dict[str, Any]:
    """Validate entry_point format and consistency with prompts."""
    print(f"\nValidating {dataset_name} entry_points...")
    
    issues = []
    entry_points = []
    
    for i, example in enumerate(dataset):
        entry_point = example["entry_point"]
        prompt = example["prompt"]
        
        # Check if entry_point is a valid function name
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', entry_point):
            issues.append(f"Example {i}: Invalid entry_point format: {entry_point}")
        
        # Check if entry_point matches function name in prompt
        match = re.search(r'def\s+(\w+)\s*\(', prompt)
        if match:
            function_name = match.group(1)
            if function_name != entry_point:
                issues.append(f"Example {i}: entry_point '{entry_point}' doesn't match function name '{function_name}' in prompt")
        
        entry_points.append(entry_point)
    
    unique_entry_points = set(entry_points)
    
    result = {
        "total_entry_points": len(entry_points),
        "unique_entry_points": len(unique_entry_points),
        "issues": issues,
        "valid": len(issues) == 0
    }
    
    print(f"   Total entry_points: {len(entry_points)}")
    print(f"   Unique entry_points: {len(unique_entry_points)}")
    print(f"   Issues: {len(issues)}")
    
    if result["valid"]:
        print("✅ Entry point validation passed")
    else:
        print("❌ Entry point validation failed")
        for issue in issues[:5]:  # Show first 5 issues
            print(f"   - {issue}")
    
    return result

def compare_datasets(che_data: List[Dict[str, Any]], mbpp_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare CHE and MBPP datasets for compatibility."""
    print(f"\n{'='*60}")
    print("COMPARING CHE vs MBPP DATASETS")
    print(f"{'='*60}")
    
    # Basic stats
    che_size = len(che_data)
    mbpp_size = len(mbpp_data)
    
    print(f"CHE dataset size: {che_size}")
    print(f"MBPP dataset size: {mbpp_size}")
    print(f"Size difference: {mbpp_size - che_size}")
    
    # Check if MBPP can replace CHE
    can_replace = True
    issues = []
    
    if mbpp_size < che_size:
        can_replace = False
        issues.append(f"MBPP has fewer examples ({mbpp_size}) than CHE ({che_size})")
    
    # Check structure compatibility
    che_example = che_data[0]
    mbpp_example = mbpp_data[0]
    
    che_keys = set(che_example.keys())
    mbpp_keys = set(mbpp_example.keys())
    
    if che_keys != mbpp_keys:
        can_replace = False
        issues.append(f"Different keys: CHE has {che_keys}, MBPP has {mbpp_keys}")
    
    # Check data types
    for key in che_keys:
        if key in mbpp_keys:
            if type(che_example[key]) != type(mbpp_example[key]):
                can_replace = False
                issues.append(f"Different types for {key}: CHE has {type(che_example[key])}, MBPP has {type(mbpp_example[key])}")
    
    result = {
        "can_replace": can_replace,
        "issues": issues,
        "che_size": che_size,
        "mbpp_size": mbpp_size,
        "structure_compatible": che_keys == mbpp_keys
    }
    
    print(f"\nCompatibility check:")
    print(f"   Can replace CHE: {'✅ YES' if can_replace else '❌ NO'}")
    print(f"   Structure compatible: {'✅ YES' if result['structure_compatible'] else '❌ NO'}")
    
    if issues:
        print(f"   Issues found:")
        for issue in issues:
            print(f"     - {issue}")
    
    return result

def main():
    """Main validation function."""
    print("DATASET VALIDATION SCRIPT")
    print("=" * 60)
    print("Validating CHE and MBPP datasets for compatibility...")
    
    # Load datasets
    try:
        che_data = load_json_dataset("che_raw.json")
        mbpp_data = load_json_dataset("mbpp_raw.json")
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return
    
    # Validate CHE dataset
    che_structure = validate_dataset_structure(che_data, "CHE")
    che_task_ids = validate_task_ids(che_data, "CHE")
    che_prompts = validate_prompts(che_data, "CHE")
    che_tests = validate_tests(che_data, "CHE")
    che_entry_points = validate_entry_points(che_data, "CHE")
    
    # Validate MBPP dataset
    mbpp_structure = validate_dataset_structure(mbpp_data, "MBPP")
    mbpp_task_ids = validate_task_ids(mbpp_data, "MBPP")
    mbpp_prompts = validate_prompts(mbpp_data, "MBPP")
    mbpp_tests = validate_tests(mbpp_data, "MBPP")
    mbpp_entry_points = validate_entry_points(mbpp_data, "MBPP")
    
    # Compare datasets
    comparison = compare_datasets(che_data, mbpp_data)
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    che_valid = all([
        che_structure["valid"],
        che_task_ids["valid"],
        che_prompts["valid"],
        che_tests["valid"],
        che_entry_points["valid"]
    ])
    
    mbpp_valid = all([
        mbpp_structure["valid"],
        mbpp_task_ids["valid"],
        mbpp_prompts["valid"],
        mbpp_tests["valid"],
        mbpp_entry_points["valid"]
    ])
    
    print(f"CHE dataset valid: {'✅ YES' if che_valid else '❌ NO'}")
    print(f"MBPP dataset valid: {'✅ YES' if mbpp_valid else '❌ NO'}")
    print(f"MBPP can replace CHE: {'✅ YES' if comparison['can_replace'] else '❌ NO'}")
    
    if che_valid and mbpp_valid and comparison['can_replace']:
        print(f"\n🎉 SUCCESS: MBPP dataset is a valid drop-in replacement for CHE!")
    else:
        print(f"\n⚠️  WARNING: Issues found that may prevent MBPP from replacing CHE.")
        print("Please review the validation results above.")

if __name__ == "__main__":
    main()
