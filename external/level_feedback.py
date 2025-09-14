import ast
import signal
from typing import Dict, List, Tuple

from rewards.code_utils import (
    extract_imports_from_prompt,
    concatenate_functions,
    extract_specific_function,
    check_function_definition,
    check_syntax,
    extract_test_cases,
    check_aux_function_usage,
    check_aux_call_without_usage,
    is_wrapper_function,
    TimeoutException,
    timeout_handler,
)


def _run_tests(combined_code: str, test_code: str, entry_point: str) -> Tuple[int, int, List[Dict[str, str]]]:
    """
    Execute extracted test cases and return (passed, total, details list).
    Each details item: {status: 'pass'|'fail', case: <assert code>, error_type, error_message}
    """
    test_cases = extract_test_cases(test_code, entry_point)
    if not test_cases:
        return 0, 0, []

    details: List[Dict[str, str]] = []
    passed = 0
    total = len(test_cases)
    timeout_count = 0
    TEST_TIMEOUT = 10
    MAX_TIMEOUTS = 3

    # Prepare execution environment
    exec_globals: Dict[str, object] = {}
    try:
        exec(combined_code, exec_globals)
    except Exception as e:
        # Can't import definitions; mark all as failed with same error
        for tc in test_cases:
            details.append(
                {
                    "status": "fail",
                    "case": tc,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
            )
        return 0, total, details

    for i, tc in enumerate(test_cases):
        try:
            # Install timeout per test
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(TEST_TIMEOUT)

            exec(tc, exec_globals)
            details.append({"status": "pass", "case": tc})
            passed += 1
            signal.alarm(0)
        except TimeoutException as te:
            timeout_count += 1
            details.append(
                {
                    "status": "fail",
                    "case": tc,
                    "error_type": "TimeoutException",
                    "error_message": str(te) or "test execution timed out",
                }
            )
            signal.alarm(0)
            if timeout_count >= MAX_TIMEOUTS:
                # Stop early similar to reward model
                break
        except Exception as e:
            details.append(
                {
                    "status": "fail",
                    "case": tc,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
            )
            signal.alarm(0)

    return passed, total, details


def analyze_code(
    original_prompt: str,
    aux_completion: str,
    main_completion: str,
    test_code: str,
    entry_point: str,
) -> Dict[str, object]:
    """Static/dynamic analysis used by feedback modes."""
    aux_func = extract_specific_function(aux_completion, "aux")
    main_func = extract_specific_function(main_completion, entry_point)

    aux_defined, aux_msg = check_function_definition(aux_completion, "aux", "Aux function")
    main_defined, main_msg = check_function_definition(
        main_completion, entry_point, f"Main function ({entry_point})"
    )

    imports = extract_imports_from_prompt(original_prompt)
    combined_code = concatenate_functions(aux_func, main_func, imports)

    syntax_ok, syntax_msg = check_syntax(combined_code, "Combined code")
    syntax_error = None
    if not syntax_ok:
        # Try to extract more structured info
        try:
            ast.parse(combined_code)
        except SyntaxError as e:
            syntax_error = {"line": e.lineno, "offset": e.offset, "msg": e.msg}

    # Aux usage signals
    aux_called = check_aux_function_usage(main_func, "aux") if main_func else False
    called_but_not_used, problematic = (
        check_aux_call_without_usage(main_func, "aux") if main_func else (False, [])
    )
    wrapper = is_wrapper_function(main_func, "aux") if main_func else False

    # Run tests only if syntax OK
    tests_passed = 0
    tests_total = 0
    test_details: List[Dict[str, str]] = []
    if syntax_ok and test_code and entry_point:
        tests_passed, tests_total, test_details = _run_tests(
            combined_code, test_code, entry_point
        )

    return {
        "aux_func": aux_func,
        "main_func": main_func,
        "aux_defined": aux_defined,
        "aux_message": aux_msg,
        "main_defined": main_defined,
        "main_message": main_msg,
        "syntax_ok": syntax_ok,
        "syntax_message": syntax_msg,
        "syntax_error": syntax_error,
        "aux_called": aux_called,
        "aux_called_but_not_used": called_but_not_used,
        "aux_problematic_calls": problematic,
        "is_wrapper": wrapper,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "test_details": test_details,
    }


def format_followup_prompts(
    original_prompt: str,
    aux_completion: str,
    main_completion: str,
    test_code: str,
    entry_point: str,
) -> Tuple[str, str]:
    """
    Produce detailed level_feedback prompts for each agent using previous code + diagnostics.
    """
    report = analyze_code(
        original_prompt, aux_completion, main_completion, test_code, entry_point
    )

    aux_lines = [
        "Your previous aux(...) implementation:",
        report.get("aux_func") or "<no implementation found>",
        "",
        "Static and execution diagnostics:",
        f"- Aux definition: {'FOUND' if report['aux_defined'] else 'MISSING'} ({report['aux_message']})",
        f"- Main definition: {'FOUND' if report['main_defined'] else 'MISSING'} ({report['main_message']})",
        f"- Syntax: {'OK' if report['syntax_ok'] else 'ERROR'} ({report['syntax_message']})",
    ]
    if report["syntax_error"]:
        se = report["syntax_error"]
        aux_lines.append(
            f"  SyntaxError at line {se.get('line')}, col {se.get('offset')}: {se.get('msg')}"
        )
    if report["tests_total"] > 0:
        aux_lines.append(
            f"- Tests: {report['tests_passed']}/{report['tests_total']} passed"
        )
        # Include all failing details
        for td in report["test_details"]:
            if td.get("status") == "fail":
                aux_lines.append(
                    f"  Failed case: {td.get('case')} -> {td.get('error_type')}: {td.get('error_message')}"
                )
    else:
        aux_lines.append("- Tests: no test cases found")

    main_lines = [
        "Your previous main implementation:",
        report.get("main_func") or "<no implementation found>",
        "",
        "Static and execution diagnostics:",
        f"- Main definition: {'FOUND' if report['main_defined'] else 'MISSING'} ({report['main_message']})",
        f"- Aux definition: {'FOUND' if report['aux_defined'] else 'MISSING'} ({report['aux_message']})",
        f"- Syntax: {'OK' if report['syntax_ok'] else 'ERROR'} ({report['syntax_message']})",
    ]
    if report["syntax_error"]:
        se = report["syntax_error"]
        main_lines.append(
            f"  SyntaxError at line {se.get('line')}, col {se.get('offset')}: {se.get('msg')}"
        )
    if report["tests_total"] > 0:
        main_lines.append(
            f"- Tests: {report['tests_passed']}/{report['tests_total']} passed"
        )
        for td in report["test_details"]:
            if td.get("status") == "fail":
                main_lines.append(
                    f"  Failed case: {td.get('case')} -> {td.get('error_type')}: {td.get('error_message')}"
                )
    else:
        main_lines.append("- Tests: no test cases found")

    # Aux usage specific notes for main agent
    if report["main_defined"]:
        if not report["aux_called"]:
            main_lines.append("- Aux usage: main does NOT call aux")
        elif report["aux_called_but_not_used"]:
            # aux_problematic_calls is a list of formatted strings like 'Line N: reason'
            reasons = ", ".join(report["aux_problematic_calls"][:2])
            main_lines.append(f"- Aux usage: called but misused ({reasons})")
        else:
            main_lines.append("- Aux usage: main calls aux and uses its result")
        if report["is_wrapper"]:
            main_lines.append("- Warning: main appears to be a thin wrapper over aux")

    # Closing instruction
    aux_lines.append(
        "\nRevise your aux(...) accordingly. Output ONLY the function code with no extra text."
    )
    main_lines.append(
        f"\nRevise your {entry_point}(...) accordingly. Output ONLY the function code with no extra text."
    )

    return ("\n".join(aux_lines), "\n".join(main_lines))
