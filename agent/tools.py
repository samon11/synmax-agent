"""
LangChain tools for data analysis using subprocess execution with AST safety checks.
"""

import ast
import sys
import subprocess
import pandas as pd
from pathlib import Path
from langchain_core.tools import tool
from typing import Set


def get_dataset_schema_and_sample(file_path: str) -> str:
    """Load dataset and return schema information with top 5 sample rows.

    Args:
        file_path: Path to the dataset file (CSV or Parquet format)

    Returns:
        Formatted string containing:
        - Dataset shape (rows x columns)
        - Column names with data types
        - Top 5 sample rows
        - Error message if loading fails
    """
    try:
        # Load the dataset
        path = Path(file_path)
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
        elif path.suffix.lower() == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            return f"Error: Unsupported file format: {path.suffix}. Only .csv and .parquet are supported."

        # Build schema information
        schema_parts = []
        schema_parts.append(f"Dataset: {Path(file_path).name}")
        schema_parts.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        schema_parts.append("\nSchema:")

        # Add column information
        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            schema_parts.append(f"  - {col}: {dtype} ({null_count} nulls)")

        # Add sample rows
        schema_parts.append("\nTop 5 Sample Rows:")
        schema_parts.append(df.head(5).to_string())

        return "\n".join(schema_parts)

    except FileNotFoundError:
        return f"Error: Dataset file not found at path: {file_path}"
    except pd.errors.EmptyDataError:
        return f"Error: Dataset file is empty: {file_path}"
    except Exception as e:
        return f"Error: Failed to load dataset file: {str(e)}"


class ASTSecurityChecker(ast.NodeVisitor):
    """
    AST visitor that checks for potentially dangerous operations.

    This implements a "trust but verify" approach by flagging:
    - Dangerous built-in functions (exec, eval, compile, __import__)
    - File write operations (allows read operations)
    - Network operations
    - OS/subprocess operations
    - Code execution via strings

    Allowed operations:
    - File reads (open() with 'r' mode, pathlib read_text/read_bytes)
    - Data processing (pandas, numpy, etc.)
    - Standard library imports (except dangerous ones)
    """

    def __init__(self):
        self.issues: list[str] = []
        self.dangerous_functions: Set[str] = {
            "exec",
            "eval",
            "compile",
            "__import__",
            "input",
        }
        self.dangerous_modules: Set[str] = {
            "os",
            "subprocess",
            "sys",
            "socket",
            "urllib",
            "requests",
            "http",
            "shutil",
        }
        self.write_modes: Set[str] = {
            "w",
            "a",
            "w+",
            "a+",
            "r+",
            "x",
            "x+",
            "wb",
            "ab",
            "wb+",
            "ab+",
            "rb+",
            "xb",
            "xb+",
        }
        self.dangerous_pathlib_methods: Set[str] = {
            "write_text",
            "write_bytes",
            "mkdir",
            "touch",
            "unlink",
            "rmdir",
            "chmod",
        }

    def visit_Call(self, node: ast.Call):
        """Check for dangerous function calls."""
        func_name = None

        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        # Check for dangerous built-in functions
        if func_name in self.dangerous_functions:
            self.issues.append(f"Potentially dangerous function call: {func_name}()")

        # Special handling for open() - only block write modes
        elif func_name == "open":
            self._check_open_call(node)

        # Check for dangerous pathlib methods
        elif func_name in self.dangerous_pathlib_methods:
            self.issues.append(
                f"Potentially dangerous pathlib operation: .{func_name}()"
            )

        self.generic_visit(node)

    def _check_open_call(self, node: ast.Call):
        """Check if open() is being called with write mode."""
        # Check if mode is specified in arguments
        mode_arg = None

        # Check positional args (mode is second argument)
        if len(node.args) >= 2:
            mode_arg = node.args[1]

        # Check keyword args
        for keyword in node.keywords:
            if keyword.arg == "mode":
                mode_arg = keyword.value
                break

        # If mode is a string literal, check if it's a write mode
        if isinstance(mode_arg, ast.Constant) and isinstance(mode_arg.value, str):
            mode = mode_arg.value
            if any(write_mode in mode for write_mode in self.write_modes):
                self.issues.append(
                    f"File write operation detected: open(..., mode='{mode}')"
                )
        elif mode_arg is not None:
            # Mode is specified but not a constant (could be a variable)
            self.issues.append(
                "File open with non-constant mode (cannot verify safety)"
            )

    def visit_Import(self, node: ast.Import):
        """Check for dangerous module imports."""
        for alias in node.names:
            if alias.name in self.dangerous_modules:
                self.issues.append(f"Potentially dangerous import: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Check for dangerous 'from X import Y' statements."""
        if node.module and node.module in self.dangerous_modules:
            self.issues.append(f"Potentially dangerous import from: {node.module}")
        self.generic_visit(node)


def check_code_safety(code: str) -> tuple[bool, list[str]]:
    """
    Parse and check Python code for safety issues using AST.

    Args:
        code: Python code string to analyze

    Returns:
        Tuple of (is_safe, list_of_issues)
    """
    try:
        tree = ast.parse(code)
        checker = ASTSecurityChecker()
        checker.visit(tree)

        is_safe = len(checker.issues) == 0
        return is_safe, checker.issues

    except SyntaxError as e:
        return False, [f"Syntax error: {str(e)}"]
    except Exception as e:
        return False, [f"Failed to parse code: {str(e)}"]


@tool
def execute_python_subprocess(code: str, timeout: int = 30) -> str:
    """Execute Python code in a subprocess with safety checks.

    IMPORTANT: This tool is STATELESS. Each execution runs in a fresh, isolated
    subprocess. All variables, imports, and state are lost after execution.
    Each call starts with an empty session - nothing persists between executions.

    This tool runs Python code in an isolated subprocess after performing
    AST-based safety checks. It uses a 'trust but verify' approach:
    - Blocks dangerous operations (exec, eval, file writes, network, etc.)
    - Allows safe operations (file reads, data processing, computation)
    - Runs code using the same Python interpreter as the main process
    - Runs in subprocess isolation for additional safety
    - Enforces timeout limits

    Allowed operations:
    - File reads: open('file.txt', 'r'), pathlib.Path.read_text()
    - Data processing: pandas, numpy, matplotlib, etc.
    - Computation and analysis

    Blocked operations:
    - File writes: open('file.txt', 'w'), pathlib write methods
    - Code execution: exec(), eval(), compile()
    - System operations: os, subprocess, sys modules
    - Network: socket, urllib, requests, http modules

    Args:
        code: Python code to execute
        timeout: Maximum execution time in seconds (default: 30)

    Returns:
        Execution output (stdout/stderr) or error message with safety warnings
    """
    # Step 1: Check code safety using AST
    is_safe, issues = check_code_safety(code)

    if not is_safe and issues:
        warning_msg = (
            "SECURITY WARNING - Code contains potentially dangerous operations:\n"
        )
        for issue in issues:
            warning_msg += f"  - {issue}\n"
        warning_msg += (
            "\nExecution blocked. Please review the code for safety concerns."
        )
        return warning_msg

    # Step 2: Execute in subprocess using the same Python interpreter
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Combine stdout and stderr
        output_parts = []
        if result.stdout:
            output_parts.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            output_parts.append(f"STDERR:\n{result.stderr}")
        if result.returncode != 0:
            output_parts.append(f"\nExit code: {result.returncode}")

        output = (
            "\n".join(output_parts)
            if output_parts
            else "Code executed successfully (no output)"
        )
        return output

    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out after {timeout} seconds"
    except FileNotFoundError:
        return "Error: Python interpreter not found in PATH"
    except Exception as e:
        return f"Error executing code: {str(e)}"
