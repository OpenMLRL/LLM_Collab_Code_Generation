import signal

import numpy as np

from rewards.code_utils import (
    TimeoutException,
    check_aux_call_without_assignment,
    check_aux_function_usage,
    check_function_definition,
    check_syntax,
    cleanup_code,
    concatenate_functions,
    extract_imports_from_prompt,
    extract_specific_function,
    extract_test_cases,
    is_wrapper_function,
    timeout_handler,
)


def code_reward_logger(
    completions1, completions2, test_cases, entry_points, prompts=None
):
    """
    Logger for code generation tasks (HumanEval/CoopHumanEval) with aux + main function collaboration.

    Args:
        completions1: List of aux function completions
        completions2: List of main function completions
        test_cases: List of test cases
        entry_points: List of entry point function names
        prompts: Optional list of prompts for import extraction

    Returns:
        List of metric dictionaries, one per sample
    """
    all_metrics = []
    TEST_TIMEOUT = 10
    MAX_TIMEOUTS = 3

    # Handle case where prompts is not provided
    if prompts is None:
        prompts = [""] * len(completions1)

    for i, (c1, c2, test_code, entry_point, prompt) in enumerate(
        zip(completions1, completions2, test_cases, entry_points, prompts)
    ):
        metrics = {
            "sample_id": i,
            "entry_point": entry_point,
            # Level 1 metrics
            "level_1_reward": 0.0,
            # Level 2 metrics
            "level_2_reward": 0.0,
            # Level 3 metrics
            "level_3_reward": 0.0,
            "test_reward": 0.0,
            "passed_tests": 0,
            "total_tests": 0,
            "passed_rate": 0.0,
            "timeout_num": 0,
            "bonus_reward": 0.0,
            "aux_usage_bonus": 0.0,
            "anti_wrapper_bonus": 0.0,
            "called_wo_used_deduction": 0.0,  # NEW METRIC
            # Overall metrics
            "total_reward": 0.0,
            "gated_total_reward": 0.0,
        }

        # Extract imports from prompt
        imports = extract_imports_from_prompt(prompt)

        # Clean completions
        c1_clean = cleanup_code(c1)
        c2_clean = cleanup_code(c2)

        # Extract specific functions
        aux_func = extract_specific_function(c1_clean, "aux")
        main_func = extract_specific_function(c2_clean, entry_point)

        # ================================================================
        # LEVEL 1: FUNCTION DEFINITION REQUIREMENTS
        # ================================================================

        # Check aux function (+0.4)
        aux_check_passed, _ = check_function_definition(c1_clean, "aux", "Aux function")
        if aux_check_passed:
            metrics["level_1_reward"] += 0.4

        # Check main function (+0.6)
        main_check_passed, _ = check_function_definition(
            c2_clean, entry_point, f"Main function ({entry_point})"
        )
        if main_check_passed:
            metrics["level_1_reward"] += 0.6

        # If main function not defined, stop here
        if not main_check_passed:
            metrics["total_reward"] = metrics["level_1_reward"]
            metrics["gated_total_reward"] = metrics["level_1_reward"]
            all_metrics.append(metrics)
            continue

        # ================================================================
        # LEVEL 2: SYNTAX REQUIREMENTS
        # ================================================================

        # Concatenate functions with imports
        combined_code = concatenate_functions(aux_func, main_func, imports)

        # Check syntax (+0.5)
        syntax_passed, _ = check_syntax(combined_code, "Combined code")
        if syntax_passed:
            metrics["level_2_reward"] = 0.5

        # If syntax failed, stop here
        if not syntax_passed:
            metrics["total_reward"] = (
                metrics["level_1_reward"] + metrics["level_2_reward"]
            )
            metrics["gated_total_reward"] = metrics["total_reward"]
            all_metrics.append(metrics)
            continue

        # ================================================================
        # LEVEL 3: TEST EXECUTION REQUIREMENTS
        # ================================================================

        # Extract test cases
        test_cases_list = extract_test_cases(test_code, entry_point)
        if not test_cases_list:
            metrics["total_reward"] = (
                metrics["level_1_reward"] + metrics["level_2_reward"]
            )
            metrics["gated_total_reward"] = metrics["total_reward"]
            all_metrics.append(metrics)
            continue

        metrics["total_tests"] = len(test_cases_list)

        # Execute tests
        timeout_count = 0
        passed_tests = 0

        try:
            # Load code definitions
            exec_globals = {}
            exec(combined_code, exec_globals)

            # Run individual test cases
            for test_i, test_case in enumerate(test_cases_list):
                # Check timeout limit
                if timeout_count >= MAX_TIMEOUTS:
                    break

                try:
                    # Set timeout for each test
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(TEST_TIMEOUT)

                    # Execute test
                    exec(test_case, exec_globals)
                    passed_tests += 1

                    # Clear timeout
                    signal.alarm(0)

                except TimeoutException:
                    signal.alarm(0)
                    timeout_count += 1

                except (AssertionError, Exception):
                    signal.alarm(0)
                    # Test failed, continue to next test
                    continue

            # Calculate test metrics
            metrics["passed_tests"] = passed_tests
            metrics["timeout_num"] = timeout_count
            if metrics["total_tests"] > 0:
                metrics["passed_rate"] = passed_tests / metrics["total_tests"]
                metrics["test_reward"] = metrics["passed_rate"] * 1.0

        except Exception:
            # Code loading failed
            signal.alarm(0)

        # Level 3 reward is just test reward at this point
        metrics["level_3_reward"] = metrics["test_reward"]

        # ================================================================
        # LEVEL 3 BONUS: COLLABORATION AND COMPLEXITY CHECKS
        # ================================================================

        # Only award bonuses if we have aux function and passed tests
        if passed_tests > 0 and aux_func:
            # Check if main uses aux
            main_uses_aux = check_aux_function_usage(main_func, "aux")

            if main_uses_aux:
                metrics["aux_usage_bonus"] = 0.5
                metrics["bonus_reward"] += 0.5

                # Check if it's NOT a wrapper (anti-wrapper bonus)
                is_wrapper = is_wrapper_function(main_func, "aux")

                if not is_wrapper:
                    metrics["anti_wrapper_bonus"] = 1.0
                    metrics["bonus_reward"] += 1.0

                # Check for aux calls without assignment (deduction)
                has_ignored_calls, ignored_calls = check_aux_call_without_assignment(
                    main_func, "aux"
                )

                if has_ignored_calls:
                    metrics["called_wo_used_deduction"] = 0.5
                    metrics["bonus_reward"] -= 0.5  # Apply deduction to bonus_reward

        # Update level 3 reward to include bonuses (after deduction)
        metrics["level_3_reward"] += metrics["bonus_reward"]

        # ================================================================
        # FINAL REWARD CALCULATION
        # ================================================================

        metrics["total_reward"] = (
            metrics["level_1_reward"]
            + metrics["level_2_reward"]
            + metrics["level_3_reward"]
        )

        # Gated reward considers early stopping
        metrics["gated_total_reward"] = metrics["total_reward"]

        all_metrics.append(metrics)

    return all_metrics


# Legacy single-turn aggregator removed in favor of multi-turn aggregator
