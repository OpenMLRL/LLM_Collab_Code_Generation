import ast
import re


class TimeoutException(Exception):
    """Exception raised when code execution times out."""

    pass


def timeout_handler(signum, frame):
    """Signal handler for timeouts."""
    raise TimeoutException("Code execution timed out")


def extract_imports_from_prompt(prompt):
    """Extract import statements from the prompt text."""
    if not prompt:
        return ""

    # Find all import statements in the prompt
    import_patterns = [
        r"^from\s+[\w\.]+\s+import\s+.*$",  # from module import ...
        r"^import\s+[\w\.,\s]+$",  # import module, module2
    ]

    imports = []
    lines = prompt.split("\n")

    for line in lines:
        line = line.strip()
        for pattern in import_patterns:
            if re.match(pattern, line):
                imports.append(line)
                break

    # Remove duplicates while preserving order
    seen = set()
    unique_imports = []
    for imp in imports:
        if imp not in seen:
            seen.add(imp)
            unique_imports.append(imp)

    return "\n".join(unique_imports)


def cleanup_code(code):
    """
    Extract function definitions from code that may contain explanatory text,
    markdown, and other non-executable content.
    """
    if not code or not isinstance(code, str):
        return ""

    # Step 1: Remove markdown code blocks but keep the content
    code = re.sub(r"```python\s*\n?", "", code)
    code = re.sub(r"```\s*\n?", "", code)

    # Step 2: Split into lines for processing
    lines = code.split("\n")

    # Step 3: Find and extract function definitions
    function_blocks = []
    current_function = []
    in_function = False
    base_indent = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Check if this line starts a function definition
        if re.match(r"^(\s*)def\s+\w+\s*\(", line):
            # If we were already in a function, save it
            if in_function and current_function:
                function_blocks.append("\n".join(current_function))

            # Start new function
            current_function = [line]
            in_function = True
            base_indent = len(line) - len(line.lstrip())

        elif in_function:
            # We're inside a function definition
            if stripped == "":
                # Empty line - include it
                current_function.append(line)
            elif line.startswith(" " * (base_indent + 1)) or line.startswith("\t"):
                # This line is indented more than the function def - it's part of the function
                current_function.append(line)
            elif stripped.startswith('"""') or stripped.startswith("'''"):
                # Handle docstrings that might be at the same level as def
                current_function.append(line)
                # Look for closing docstring
                quote_type = '"""' if stripped.startswith('"""') else "'''"
                if not (stripped.endswith(quote_type) and len(stripped) > 3):
                    # Multi-line docstring, find the end
                    i += 1
                    while i < len(lines):
                        current_function.append(lines[i])
                        if quote_type in lines[i]:
                            break
                        i += 1
            else:
                # This line is not part of the function (back to base level or less)
                # End current function
                if current_function:
                    function_blocks.append("\n".join(current_function))
                current_function = []
                in_function = False

                # Check if this line starts a new function
                if re.match(r"^(\s*)def\s+\w+\s*\(", line):
                    current_function = [line]
                    in_function = True
                    base_indent = len(line) - len(line.lstrip())

        i += 1

    # Don't forget the last function if we ended while still in one
    if in_function and current_function:
        function_blocks.append("\n".join(current_function))

    # Step 4: Join all function blocks
    result = "\n\n".join(function_blocks)

    # Step 5: Final cleanup - remove any remaining explanatory text that might have slipped through
    # Remove lines that look like natural language explanations
    lines = result.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        # Skip lines that are clearly natural language (but keep comments and docstrings)
        if (
            stripped
            and not stripped.startswith("#")
            and not stripped.startswith('"""')
            and not stripped.startswith("'''")
            and not any(
                keyword in line
                for keyword in [
                    "def ",
                    "return ",
                    "if ",
                    "for ",
                    "while ",
                    "=",
                    "(",
                    ")",
                    "[",
                    "]",
                ]
            )
            and re.match(r"^[A-Z][a-z].*[.!?:]$", stripped)
        ):
            continue
        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines).strip()

    # Step 6: Validate that we have actual function definitions
    if not re.search(r"def\s+\w+\s*\(", result):
        return ""

    return result


def extract_specific_function(code, function_name):
    """
    Extract a specific function by name from the code.
    This is useful when you know exactly which function you want.
    """
    if not code or not function_name:
        return ""

    # First clean the code
    cleaned = cleanup_code(code)

    # Then extract the specific function
    lines = cleaned.split("\n")
    function_lines = []
    in_target_function = False
    base_indent = 0

    for line in lines:
        if re.match(rf"^(\s*)def\s+{re.escape(function_name)}\s*\(", line):
            function_lines = [line]
            in_target_function = True
            base_indent = len(line) - len(line.lstrip())
        elif in_target_function:
            if (
                line.strip() == ""
                or line.startswith(" " * (base_indent + 1))
                or line.startswith("\t")
            ):
                function_lines.append(line)
            else:
                # End of function
                break

    return "\n".join(function_lines).strip()


def check_function_definition(code, function_name, description):
    """Check if a function is properly defined with return statement."""
    # Extract just the specific function
    func_code = extract_specific_function(code, function_name)

    if not func_code:
        return False, f"{description} not defined"

    # Check if function has return statement
    has_return = re.search(r"return\s+", func_code) is not None

    if not has_return:
        return False, f"{description} missing return statement"

    return True, f"{description} properly defined with return statement"


def check_syntax(code, description):
    """Check syntax of code."""
    try:
        ast.parse(code)
        return True, f"{description} syntax OK"
    except SyntaxError as e:
        return False, f"{description} has syntax error: {str(e)}"


def concatenate_functions(aux_completion, main_completion, imports=""):
    """Concatenate imports, aux and main functions."""
    aux_clean = cleanup_code(aux_completion)
    main_clean = cleanup_code(main_completion)

    # Build the combined code with imports first
    parts = []

    if imports:
        parts.append(imports)

    if aux_clean:
        parts.append(aux_clean)

    if main_clean:
        parts.append(main_clean)

    combined_code = "\n\n".join(parts)
    return combined_code


def extract_test_cases(test_code, entry_point):
    """Extract individual test cases from the test code and replace candidate with entry_point."""
    if not test_code or not entry_point:
        return []

    # Find the check function
    check_match = re.search(
        r"def check\(candidate\):(.*?)(?=\n\ndef|\Z)", test_code, re.DOTALL
    )
    if not check_match:
        return []

    check_body = check_match.group(1)

    # Find all assert statements - using a more general approach
    # Pattern 1: Direct candidate calls with ==
    direct_equality_pattern = r"assert\s+candidate\([^)]*\)\s*==\s*[^\n]+"

    # Pattern 2: Simple abs difference pattern
    simple_abs_pattern = r"assert\s+abs\(candidate\([^)]*\)\s*-\s*[^)]+\)\s*<\s*[^\n]+"

    # Pattern 3: Math.fabs or other function calls containing candidate
    math_fabs_pattern = r"assert\s+math\.fabs\([^)]*candidate[^)]*\)\s*<\s*[^\n]+"

    # Pattern 4: General assert statements that contain "candidate" - catch-all
    general_candidate_pattern = r"assert\s+[^\n]*candidate[^\n]*"

    # Try patterns in order of specificity (most specific first)
    patterns = [
        direct_equality_pattern,
        simple_abs_pattern,
        math_fabs_pattern,
        general_candidate_pattern,
    ]

    all_matches = []
    check_lines = check_body.strip().split("\n")

    # Process each line to find assert statements
    for line in check_lines:
        line = line.strip()
        if line.startswith("assert") and "candidate" in line:
            # Check if this line matches any of our specific patterns
            matched = False
            for pattern in patterns[:-1]:  # Don't use general pattern in this loop
                if re.match(pattern, line):
                    all_matches.append(line)
                    matched = True
                    break

            # If no specific pattern matched, use the general pattern as fallback
            if not matched and re.match(general_candidate_pattern, line):
                all_matches.append(line)

    # Remove duplicates while preserving order
    seen = set()
    unique_matches = []
    for match in all_matches:
        if match not in seen:
            seen.add(match)
            unique_matches.append(match)

    # Replace 'candidate' with actual entry_point name
    test_cases = []
    for match in unique_matches:
        test_case = match.replace("candidate", entry_point)
        test_cases.append(test_case)

    return test_cases


def check_aux_function_usage(main_code, aux_function_name="aux"):
    """Check if the main function calls the auxiliary function, including indirect usage."""
    if not main_code or not aux_function_name:
        return False

    # Pattern 1: Direct function calls - aux(...)
    direct_call_pattern = rf"\b{re.escape(aux_function_name)}\s*\("

    # Pattern 2: Aux function used in assignments - var = aux(...)
    assignment_pattern = rf"=\s*{re.escape(aux_function_name)}\s*\("

    # Pattern 3: Aux function used in expressions - return aux(...) + something
    expression_pattern = rf"\b{re.escape(aux_function_name)}\s*\([^)]*\)"

    # Check for any of these patterns
    patterns = [direct_call_pattern, assignment_pattern, expression_pattern]

    for pattern in patterns:
        if re.search(pattern, main_code):
            return True

    return False


def is_wrapper_function(main_code, aux_function_name="aux"):
    """
    Check if the main function is just a simple wrapper around the aux function.
    A wrapper is defined as a function that:
    1. Calls aux function once
    2. Directly returns the result without significant processing
    """
    if not main_code or not aux_function_name:
        return False

    # Remove comments and docstrings for cleaner analysis
    lines = main_code.split("\n")
    code_lines = []
    in_docstring = False

    for line in lines:
        stripped = line.strip()

        # Skip empty lines
        if not stripped:
            continue

        # Handle docstrings
        if stripped.startswith('"""') or stripped.startswith("'''"):
            quote_type = '"""' if stripped.startswith('"""') else "'''"
            if stripped.endswith(quote_type) and len(stripped) > 3:
                # Single line docstring, skip it
                continue
            else:
                # Multi-line docstring start
                in_docstring = True
                continue
        elif in_docstring:
            if '"""' in stripped or "'''" in stripped:
                in_docstring = False
            continue

        # Skip comments
        if stripped.startswith("#"):
            continue

        # Skip function definition line
        if stripped.startswith("def "):
            continue

        code_lines.append(stripped)

    # Now analyze the actual code content
    if len(code_lines) == 0:
        return True  # Empty function body is considered a wrapper

    # Count aux function calls
    aux_call_pattern = rf"\b{re.escape(aux_function_name)}\s*\("
    aux_calls = sum(1 for line in code_lines if re.search(aux_call_pattern, line))

    # Pattern 1: Single line that calls aux and returns directly
    # e.g., "return aux(...)"
    if len(code_lines) == 1:
        line = code_lines[0]
        if line.startswith("return ") and aux_function_name in line:
            return True

    # Pattern 2: Two lines - assign aux result and return it
    # e.g., "result = aux(...)" followed by "return result"
    elif len(code_lines) == 2:
        first_line = code_lines[0]
        second_line = code_lines[1]

        # Check if first line assigns aux result to a variable
        aux_assignment_pattern = rf"(\w+)\s*=\s*{re.escape(aux_function_name)}\s*\("
        assignment_match = re.search(aux_assignment_pattern, first_line)

        if assignment_match:
            var_name = assignment_match.group(1)
            # Check if second line just returns that variable
            if second_line == f"return {var_name}":
                return True

    return False


def check_aux_call_without_assignment(main_code, aux_function_name="aux"):
    """
    Check if aux function is called but its return value is ignored (not assigned).
    Uses bad patterns approach - looks for standalone aux calls that waste the return value.

    Returns:
        bool: True if aux is called without assignment (should deduct points)
        list: List of problematic aux calls found
    """

    if not main_code or aux_function_name not in main_code:
        return False, []

    # Find all aux function calls
    aux_call_pattern = rf"{re.escape(aux_function_name)}\s\([^)]\)"

    problematic_calls = []
    lines = main_code.split("\n")

    for line_num, line in enumerate(lines, 1):
        stripped_line = line.strip()

        # Skip empty lines and comments
        if not stripped_line or stripped_line.startswith("#"):
            continue

        # If line contains aux call, check if it's a bad pattern
        if re.search(aux_call_pattern, stripped_line):

            # Bad patterns (aux result is wasted):
            bad_patterns = [
                # Standalone aux call (just aux(...) on its own line)
                rf"^{re.escape(aux_function_name)}\s\([^)]\)$",
                # Aux call followed by semicolon (if someone uses semicolons)
                rf"^{re.escape(aux_function_name)}\s\([^)]\);?\s$",
                # Aux call in expression statement that doesn't use the result
                # (This is the most common bad pattern)
                rf"^\s{re.escape(aux_function_name)}\s\([^)]\)\s*$",
            ]

            # Check if this line matches any bad pattern
            is_bad_usage = any(
                re.search(pattern, stripped_line) for pattern in bad_patterns
            )

            if is_bad_usage:
                problematic_calls.append(f"Line {line_num}: {stripped_line}")

    return len(problematic_calls) > 0, problematic_calls


def check_aux_call_without_usage(main_code, aux_function_name="aux"):
    """
    Check if aux function is called but its return value is ignored (not assigned).
    Uses bad patterns approach - looks for standalone aux calls that waste the return value.

    Returns:
        bool: True if aux is called without assignment (should deduct points)
        list: List of problematic aux calls found
    """
    if not main_code or aux_function_name not in main_code:
        return False, []

    # AST parse may fail on incomplete or special-token-laden generations
    try:
        tree = ast.parse(main_code)
    except SyntaxError:
        return False, []

    analyzer = AuxUsageAnalyzer(aux_function_name, main_code)
    analyzer.visit(tree)
    problematic_calls = analyzer.get_problematic_calls()

    return len(problematic_calls) > 0, problematic_calls


class AuxUsageAnalyzer(ast.NodeVisitor):
    def __init__(self, function_name, code):
        self.function_name = function_name
        self.code = code
        self.parent_map = {}
        self.problematic_calls = []
        self.variable_usage = {}  # Track variable usage

    def visit(self, node):
        # Build parent-child relationships
        for child in ast.iter_child_nodes(node):
            self.parent_map[child] = node
        return super().visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == self.function_name:

            # Check the context of this aux call
            context = self._get_call_context(node)

            if context["is_problematic"]:
                self.problematic_calls.append(
                    {
                        "line": node.lineno,
                        "reason": context["reason"],
                        "variable": context.get("variable"),
                    }
                )

        self.generic_visit(node)

    def visit_Name(self, node):
        # Track variable usage (when variable is being read)
        if isinstance(node.ctx, ast.Load):
            if node.id not in self.variable_usage:
                self.variable_usage[node.id] = []
            self.variable_usage[node.id].append(node.lineno)

        self.generic_visit(node)

    def _get_call_context(self, call_node):
        """Determine if aux call usage is problematic."""
        parent = self.parent_map.get(call_node)

        if parent is None:
            return {"is_problematic": True, "reason": "standalone call (not assigned)"}

        # Good contexts - aux return value is used
        if isinstance(parent, ast.Return):
            return {"is_problematic": False, "reason": "used as return value"}

        if isinstance(parent, ast.Call):
            return {"is_problematic": False, "reason": "used as function argument"}

        if isinstance(parent, (ast.BinOp, ast.UnaryOp, ast.Compare)):
            return {"is_problematic": False, "reason": "used in expression"}

        if isinstance(parent, (ast.If, ast.While)) and parent.test == call_node:
            return {"is_problematic": False, "reason": "used in condition"}

        if isinstance(parent, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            return {"is_problematic": False, "reason": "used in collection"}

        # Assignment context - need to check if assigned variable is used
        if isinstance(parent, ast.Assign):
            for target in parent.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id

                    # Check if variable is never used
                    if self._is_variable_unused(var_name, parent.lineno):
                        return {
                            "is_problematic": True,
                            "reason": f"assigned to {var_name} but never used",
                            "variable": var_name,
                        }

                    # Check if variable is reassigned before use
                    reassign_line = self._get_reassignment_before_use(
                        var_name, parent.lineno
                    )
                    if reassign_line:
                        return {
                            "is_problematic": True,
                            "reason": f"assigned to {var_name} but reassigned on line {reassign_line} before use",
                            "variable": var_name,
                        }

            return {"is_problematic": False, "reason": "assigned and used"}

        # Standalone expression (aux() by itself)
        if isinstance(parent, ast.Expr):
            return {"is_problematic": True, "reason": "standalone call (not assigned)"}

        # Default to safe (not problematic)
        return {"is_problematic": False, "reason": "used in other context"}

    def _is_variable_unused(self, var_name, assignment_line):
        """Check if variable is never used after assignment."""
        if var_name not in self.variable_usage:
            return True

        # Check if there are any usages after the assignment line
        usage_lines = [
            line for line in self.variable_usage[var_name] if line > assignment_line
        ]
        return len(usage_lines) == 0

    def _get_reassignment_before_use(self, var_name, assignment_line):
        """Check if variable is reassigned before being used."""
        if var_name not in self.variable_usage:
            return None

        # Find reassignments and usages after the aux assignment
        reassignments = []
        usages = []

        try:
            tree = ast.parse(self.code)
        except SyntaxError:
            # If code cannot be parsed, we cannot analyze reassignments safely
            return None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and hasattr(node, "lineno")
                and node.lineno > assignment_line
            ):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == var_name:
                        reassignments.append(node.lineno)

            if (
                isinstance(node, ast.Name)
                and node.id == var_name
                and isinstance(node.ctx, ast.Load)
                and hasattr(node, "lineno")
                and node.lineno > assignment_line
            ):
                usages.append(node.lineno)

        if not reassignments:
            return None

        first_reassignment = min(reassignments)
        first_usage = min(usages) if usages else float("inf")

        return first_reassignment if first_reassignment < first_usage else None

    def get_problematic_calls(self):
        """Return list of problematic calls with formatted messages."""
        formatted_calls = []
        for call in self.problematic_calls:
            line = call["line"]
            reason = call["reason"]
            formatted_calls.append(f"Line {line}: {reason}")

        return formatted_calls
