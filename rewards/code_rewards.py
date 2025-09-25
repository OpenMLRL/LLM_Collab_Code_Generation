import re
import signal
from typing import List
import builtins

# Verbose toggle (can be set by training scripts)
VERBOSE = True

from rewards.code_utils import (
    TimeoutException,
    check_aux_call_without_assignment,
    check_aux_used_on_passing_tests,
    check_function_definition,
    check_syntax,
    cleanup_code,
    compute_aux_usefulness_delta,
    concatenate_functions,
    extract_imports_from_prompt,
    extract_specific_function,
    extract_test_cases,
    is_wrapper_function,
    timeout_handler,
    valid_format_gate,
)


def execution_reward_aux(
    completion1: List[str],
    completion2: List[str],
    test_cases: List[str],
    entry_points: List[str],
    prompts: List[str] = None,  # Add prompts parameter
) -> List[float]:
    """
    Reward function for aux + main function collaboration on code tasks:

    FORMAT GATE:
    - Reward = 0 if either completion doesn't contain single valid def ...(): return ... format

    LEVEL 1 (Definitions):
    - +0.4 reward if aux function is properly defined with return statement
    - +0.6 reward if main function is properly defined with return statement

    LEVEL 2 (Syntax):
    - +0.5 reward if concatenated code has no syntax errors

    LEVEL 3 (Tests):
    - +0 to +1.0 reward proportional to test pass rate

    DEPENDENCY GATE:
    - Cap total reward ≤ 2.0 if any tests pass but aux is not called/used

    DENSE COOPERATION BONUSES (only if aux is used on passing tests):
    - Use-aux bonus (0 → 0.5): Proportional to fraction of passing tests where aux contributes
    - Non-wrapper bonus (0 or 0.6): Binary bonus if main is NOT a simple wrapper around aux
    - Aux usefulness Δ (0 → 0.6): Run tests with aux stubbed; reward proportional to pass_rate_true - pass_rate_stub

    PENALTIES:
    - -0.5 if aux is called but return ignored
    - -0.2 if runtime timeout or unsafe operation

    Final reward is clipped between 0.0 and 4.0.
    """
    # Local print override based on VERBOSE
    if not VERBOSE:
        def print(*args, **kwargs):  # type: ignore
            return None
    else:
        print = builtins.print  # type: ignore

    rewards = []
    TEST_TIMEOUT = 10  # Timeout per individual test

    # Handle case where prompts is not provided
    if prompts is None:
        prompts = [""] * len(completion1)

    for c1, c2, test_code, entry_point, prompt in zip(
        completion1, completion2, test_cases, entry_points, prompts
    ):
        reward = 0.0

        print("\n" + "=" * 60)
        print("TESTING HUMANEVAL AUX + MAIN FUNCTION COLLABORATION")
        print("=" * 60)
        print(f"Entry point: {entry_point}")
        print(f"Maximum possible reward: 4.0 (Level 3 max: 2.5)")

        # Extract imports from prompt
        imports = extract_imports_from_prompt(prompt)
        if imports:
            print(f"\n--- EXTRACTED IMPORTS ---")
            print(imports)

        # Print raw completions for debugging
        print(f"\n--- RAW COMPLETION 1 (AUX) ---")
        print(repr(c1))
        print(f"\n--- RAW COMPLETION 2 (MAIN) ---")
        print(repr(c2))

        # ================================================================
        # STRICT FORMAT GATE - Check both completions
        # ================================================================
        print("\n🚪 STRICT FORMAT GATE")
        print("-" * 30)
        
        aux_format_valid = valid_format_gate(c1)
        main_format_valid = valid_format_gate(c2)
        
        print(f"Aux format valid: {aux_format_valid}")
        print(f"Main format valid: {main_format_valid}")
        
        if not aux_format_valid or not main_format_valid:
            print("❌ FORMAT GATE FAILED: One or both completions don't meet strict format requirements")
            print("   Requirements: Single valid def ...(): return ... format")
            print(f"Final reward: 0.0")
            rewards.append(0.0)
            continue
        
        print("✅ FORMAT GATE PASSED: Both completions meet format requirements")

        # Clean completions
        c1_clean = cleanup_code(c1)
        c2_clean = cleanup_code(c2)

        print(f"\n--- CLEANED COMPLETION 1 (AUX) ---")
        print(repr(c1_clean))
        print(f"\n--- CLEANED COMPLETION 2 (MAIN) ---")
        print(repr(c2_clean))

        # Extract specific functions for validation
        aux_func = extract_specific_function(c1, "aux")
        main_func = extract_specific_function(c2, entry_point)

        print(f"\n--- EXTRACTED AUX FUNCTION ---")
        print(repr(aux_func))
        print(f"\n--- EXTRACTED MAIN FUNCTION ---")
        print(repr(main_func))

        # ================================================================
        # LEVEL 1: FUNCTION DEFINITION REQUIREMENTS
        # ================================================================
        print("\n📋 LEVEL 1: FUNCTION DEFINITION REQUIREMENTS")
        print("-" * 50)

        level1_passed = True

        # 1.1 Check aux function in completion1 (+0.4)
        # Only give reward if aux is actually defined, but don't fail if it's empty
        aux_check_passed, aux_message = check_function_definition(
            c1, "aux", "Aux function"
        )

        if aux_check_passed:
            reward += 0.4
            print(f"✅ {aux_message}: +0.4 (total: {reward})")
        else:
            print(f"⚠️  {aux_message} (continuing without aux reward)")
            # Don't set level1_passed = False for aux - it's optional

        # 1.2 Check main function in completion2 (+0.6)
        main_check_passed, main_message = check_function_definition(
            c2, entry_point, f"Main function ({entry_point})"
        )

        if main_check_passed:
            reward += 0.6
            print(f"✅ {main_message}: +0.6 (total: {reward})")
        else:
            print(f"❌ {main_message}")
            level1_passed = False

        print(f"📊 Level 1: {'PASSED' if level1_passed else 'FAILED'}")

        if not level1_passed:
            print("⏹️  STOPPING: Function definition requirements not met")
            print(f"Final reward: {reward}")
            rewards.append(reward)
            continue

        # ================================================================
        # LEVEL 2: SYNTAX REQUIREMENTS
        # ================================================================
        print("\n⚙️  LEVEL 2: SYNTAX REQUIREMENTS")
        print("-" * 40)

        # 2.1 Concatenate functions with imports
        combined_code = concatenate_functions(aux_func, main_func, imports)

        print("\n--- Combined Code ---")
        print(combined_code)
        print("--- End Code ---")

        # 2.2 Check combined syntax (+0.5)
        syntax_passed, syntax_message = check_syntax(combined_code, "Combined code")

        if syntax_passed:
            reward += 0.5
            print(f"✅ {syntax_message}: +0.5 (total: {reward})")
        else:
            print(f"❌ {syntax_message}")
            print("⏹️  STOPPING: Syntax requirements not met")
            print(f"Final reward: {reward}")
            rewards.append(reward)
            continue

        # ================================================================
        # LEVEL 3: TEST EXECUTION REQUIREMENTS
        # ================================================================
        print("\n🧪 LEVEL 3: TEST EXECUTION REQUIREMENTS")
        print("-" * 40)

        # Extract test cases
        test_cases_list = extract_test_cases(test_code, entry_point)
        if not test_cases_list:
            print("❌ No test cases found")
            rewards.append(reward)
            continue

        print(f"📝 Found {len(test_cases_list)} test case(s)")

        # Initialize test tracking variables
        passed_tests = 0
        total_tests = len(test_cases_list)

        # 3.1 Execute tests (+0 to +1.0)
        timeout_count = 0  # Track number of timeouts
        MAX_TIMEOUTS = 3  # Stop testing after 3 timeouts

        try:
            # Create execution environment (no timeout needed for function definitions)
            exec_globals = {}
            exec(combined_code, exec_globals)
            print("✅ Code definitions loaded successfully")

            # Run individual test cases WITH INDIVIDUAL TIMEOUTS
            for i, test_case in enumerate(test_cases_list):
                # Check if we should stop testing due to too many timeouts
                if timeout_count >= MAX_TIMEOUTS:
                    remaining_tests = total_tests - i
                    print(
                        f"🛑 STOPPING TEST EXECUTION: {timeout_count} timeouts reached (limit: {MAX_TIMEOUTS})"
                    )
                    print(f"⏭️  Skipping remaining {remaining_tests} tests")
                    print("⚠️  No bonuses will be awarded due to excessive timeouts")
                    break

                try:
                    # SET TIMEOUT FOR EACH TEST INDIVIDUALLY
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(TEST_TIMEOUT)

                    print(f"🧪 Running Test {i + 1}: {test_case}")

                    # Parse the test case to extract the function call and expected result
                    test_match = re.search(
                        r"assert\s+(\w+)\(([^)]*)\)\s*==\s*(.+)", test_case
                    )
                    if test_match:
                        func_name = test_match.group(1)
                        func_args = test_match.group(2)
                        expected_result = test_match.group(3)

                        # Execute the function call to get actual result
                        func_call = f"{func_name}({func_args})"
                        actual_result = eval(func_call, exec_globals)
                        expected_result_eval = eval(expected_result, exec_globals)

                        print(f"   📞 Function call: {func_call}")
                        print(f"   🎯 Expected: {expected_result_eval}")
                        print(f"   📤 Actual: {actual_result}")

                        # Execute the actual test
                        exec(test_case, exec_globals)
                        passed_tests += 1
                        print(f"✅ Test {i + 1}: PASSED")
                    else:
                        # Fallback if parsing fails
                        exec(test_case, exec_globals)
                        passed_tests += 1
                        print(f"✅ Test {i + 1}: PASSED")

                    # CLEAR TIMEOUT AFTER SUCCESSFUL TEST
                    signal.alarm(0)

                except TimeoutException:
                    signal.alarm(0)  # Clear timeout
                    timeout_count += 1
                    print(
                        f"⏰ Test {i + 1}: TIMEOUT after {TEST_TIMEOUT} seconds (timeout #{timeout_count})"
                    )
                    print("⚠️  Likely infinite recursion or infinite loop in function")

                except AssertionError as e:
                    signal.alarm(0)  # Clear timeout
                    # Try to show more details about the assertion failure
                    if (
                        "test_match" in locals()
                        and test_match
                        and "actual_result" in locals()
                        and "expected_result_eval" in locals()
                    ):
                        print(f"❌ Test {i + 1}: FAILED")
                        print(f"   🎯 Expected: {expected_result_eval}")
                        print(f"   📤 Actual: {actual_result}")
                        print(
                            f"   💥 Assertion failed: {actual_result} != {expected_result_eval}"
                        )
                    else:
                        print(f"❌ Test {i + 1}: FAILED (AssertionError: {str(e)})")

                except Exception as e:
                    signal.alarm(0)  # Clear timeout
                    print(f"❌ Test {i + 1}: FAILED (Error: {str(e)})")
                    print(f"   🧪 Test case: {test_case}")

            # Calculate proportional reward for test cases (0 to +1.0)
            if total_tests > 0:
                test_reward = (passed_tests / total_tests) * 1.0
                reward += test_reward
                print(f"📊 Tests passed: {passed_tests}/{total_tests}")
                print(f"✅ Test reward: +{test_reward:.2f} (total: {reward})")

        except Exception as e:
            print(f"❌ Code loading failed: {str(e)}")
            signal.alarm(0)

        # ================================================================
        # DEPENDENCY GATE: Cap reward at 2.0 if aux unused on passing tests
        # ================================================================
        print("\n🔒 DEPENDENCY GATE")
        print("-" * 20)
        
        aux_used_on_passing = check_aux_used_on_passing_tests(main_func, test_cases_list, "aux")
        print(f"Aux used on passing tests: {aux_used_on_passing}")
        
        if passed_tests > 0 and not aux_used_on_passing:
            reward = min(reward, 2.0)
            print(f"⚠️  DEPENDENCY GATE: Aux not used on passing tests - capping reward at 2.0")
            print(f"   Current reward: {reward}")
        else:
            print("✅ DEPENDENCY GATE: Aux properly used or no passing tests")

        # ================================================================
        # DENSE COOPERATION BONUSES
        # ================================================================
        print("\n🎁 DENSE COOPERATION BONUSES")
        print("-" * 30)
        
        cooperation_bonus = 0.0
        
        # Check if main is a wrapper (do this once and reuse)
        is_wrapper = is_wrapper_function(main_func, "aux") if main_func else False
        
        if passed_tests > 0 and aux_func and aux_used_on_passing:
            # Use-aux bonus (0 → 0.5): Proportional to fraction of passing tests where aux return contributes
            use_aux_bonus = 0.5 * (passed_tests / total_tests)
            cooperation_bonus += use_aux_bonus
            print(f"✅ Use-aux bonus: +{use_aux_bonus:.3f} (proportional to test pass rate)")
            
            # Non-wrapper bonus (0 or 0.6): Binary check if main is not a simple wrapper
            if not is_wrapper:
                non_wrapper_bonus = 0.6
                cooperation_bonus += non_wrapper_bonus
                print(f"✅ Non-wrapper bonus: +{non_wrapper_bonus:.3f} (main is not a simple wrapper)")
            else:
                non_wrapper_bonus = 0.0
                print(f"⚠️  No non-wrapper bonus: main is a simple wrapper")
            
            # Aux usefulness delta (0 → 0.6): Run tests with aux stubbed
            try:
                # Set timeout for aux usefulness calculation (30 seconds total)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)
                
                aux_usefulness = compute_aux_usefulness_delta(combined_code, test_cases_list, "aux")
                aux_usefulness_bonus = 0.6 * aux_usefulness
                cooperation_bonus += aux_usefulness_bonus
                print(f"✅ Aux usefulness bonus: +{aux_usefulness_bonus:.3f} (usefulness: {aux_usefulness:.3f})")
                
                # Clear timeout after successful calculation
                signal.alarm(0)
            except TimeoutException:
                signal.alarm(0)  # Clear timeout
                print("⚠️  Aux usefulness calculation timed out - skipping bonus")
                aux_usefulness_bonus = 0.0
            except Exception as e:
                signal.alarm(0)  # Clear timeout
                print(f"⚠️  Aux usefulness calculation failed: {e} - skipping bonus")
                aux_usefulness_bonus = 0.0
            
            reward += cooperation_bonus
            print(f"🎉 Total cooperation bonus: +{cooperation_bonus:.3f} (total: {reward:.3f})")
        else:
            if passed_tests == 0:
                print("⚠️  No tests passed - no cooperation bonuses")
            if not aux_func:
                print("⚠️  No aux function - no cooperation bonuses")
            if not aux_used_on_passing:
                print("⚠️  Aux not used on passing tests - no cooperation bonuses")

        # ================================================================
        # PENALTIES
        # ================================================================
        print("\n⚠️  PENALTIES")
        print("-" * 15)
        
        penalty = 0.0
        
        # Check for aux calls without assignment (deduction)
        if aux_func and main_func:
            has_ignored_calls, ignored_calls = check_aux_call_without_assignment(main_func, "aux")
            if has_ignored_calls:
                penalty += 0.5
                print(f"⚠️  Aux function called but return value ignored: -0.5")
                print("💡 Problematic aux calls found:")
                for call in ignored_calls:
                    print(f"   📍 {call}")
            else:
                print("✅ All aux function calls properly use return values")
            
        
        # Check for runtime timeout or unsafe operations
        if timeout_count > 0:
            penalty += 0.2
            print(f"⚠️  Runtime timeout detected: -0.2")
        
        reward -= penalty
        if penalty > 0:
            print(f"💰 Total penalties: -{penalty:.1f} (total: {reward:.3f})")
        else:
            print("✅ No penalties applied")

        # Clip final reward between 0 and 4.0
        reward = max(0.0, min(4.0, reward))
        print(f"\n🏆 FINAL REWARD: {reward:.3f} / 4.0")
        rewards.append(reward)

    return rewards
