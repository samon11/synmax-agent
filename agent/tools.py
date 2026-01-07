"""
LangChain tools for data analysis using Pyodide sandbox with session_bytes.
"""

import pandas as pd
from pathlib import Path
from langchain_core.tools import tool, BaseTool
from langchain_sandbox import PyodideSandbox
from typing import Optional, Dict, Any
from pydantic import Field


class PyodideDatasetTool(BaseTool):
    """
    Pyodide sandbox tool that maintains stateful sessions using session_bytes.

    This tool loads a dataset into a Pyodide session as a Python object (pandas DataFrame)
    and maintains the session state across executions without exposing file contents to the LLM.
    """

    name: str = "pyodide_executor"
    description: str = """Execute Python code in a stateful Pyodide sandbox with dataset access.

    IMPORTANT: The dataset is ALREADY LOADED as a pandas DataFrame named 'data' in the session.
    DO NOT use pd.read_csv() or load the dataset - simply use the 'data' variable directly.

    Use this to run pandas, numpy, or other data analysis code on the dataset.
    REQUIREMENT CRITICAL: YOU MUST ALWAYS INCLUDE A print(...) STATEMENT TO ACCESS ANY RESULTS FROM THIS TOOL."""

    # Dataset configuration
    dataset_path: str = Field(description="Path to the dataset file")

    # Session state (opaque bytes maintained by Pyodide)
    session_bytes: Optional[bytes] = Field(default=None, exclude=True)
    session_metadata: Optional[Dict[str, Any]] = Field(default=None, exclude=True)
    dataset_bootstrapped: bool = Field(default=False, exclude=True)

    # Sandbox configuration
    stateful: bool = Field(default=True, exclude=True)
    allow_net: bool = Field(default=False, exclude=True)

    # Internal sandbox instance
    sandbox: Optional[PyodideSandbox] = Field(default=None, exclude=True, repr=False)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, dataset_path: str, **kwargs):
        super().__init__(dataset_path=dataset_path, **kwargs)
        # Initialize the sandbox
        self.sandbox = PyodideSandbox(
            stateful=self.stateful,
            allow_net=self.allow_net
        )

    async def bootstrap(self) -> str:
        """
        Load the dataset into the Pyodide session as a Python object.

        This runs once during initialization to create the initial session state.
        The dataset is loaded as a pandas DataFrame and stored in the session,
        without ever exposing the raw file contents to the LLM.

        Returns:
            Success message with dataset info
        """
        if self.dataset_bootstrapped:
            return "Dataset already bootstrapped"

        try:
            # 1. Read the file locally (no LLM involvement)
            df = pd.read_csv(Path(self.dataset_path))

            # 2. Bootstrap code that creates the 'data' variable in the session
            bootstrap_code = """
import pandas as pd
# The 'df' variable is passed in from outside
data = df
print(f"Dataset loaded: {data.shape[0]} rows × {data.shape[1]} columns")
"""

            # 3. Execute with the DataFrame passed as a kwarg
            result = await self.sandbox.execute(
                bootstrap_code,
                kwargs={"df": df}
            )

            # 4. Store the session state
            self.session_bytes = result.session_bytes
            self.session_metadata = result.session_metadata
            self.dataset_bootstrapped = True

            return f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns"

        except Exception as e:
            return f"Failed to bootstrap dataset: {str(e)}"

    def _run(self, code: str) -> str:
        """
        Execute code in the stateful Pyodide session.

        This is not async, so we raise an error directing users to use the async version.
        """
        raise NotImplementedError(
            "PyodideDatasetTool only supports async execution. Use _arun instead."
        )

    async def _arun(self, code: str) -> str:
        """
        Execute code in the stateful Pyodide session.

        Args:
            code: Python code to execute

        Returns:
            Execution result or error message
        """
        if not self.dataset_bootstrapped:
            await self.bootstrap()

        try:
            # Execute code with the current session state
            result = await self.sandbox.execute(
                code,
                session_bytes=self.session_bytes,
                session_metadata=self.session_metadata
            )

            # Update session state for next execution
            self.session_bytes = result.session_bytes
            self.session_metadata = result.session_metadata

            # Return stdout/result
            return result.stdout or str(result.result)

        except Exception as e:
            return f"Error executing code: {str(e)}"


@tool
def load_dataset_file(file_path: str) -> str:
    """Load the dataset file content for inspection.

    Args:
        file_path: Path to the dataset file

    Returns:
        First 500 characters of file content or error message
    """
    try:
        content = Path(file_path).read_text()
        preview = content[:500]
        if len(content) > 500:
            preview += f"\n... ({len(content) - 500} more characters)"
        return preview
    except FileNotFoundError:
        return f"Error: Dataset file not found at path: {file_path}"
    except PermissionError:
        return f"Error: Permission denied reading file: {file_path}"
    except Exception as e:
        return f"Error reading dataset file: {str(e)}"


def get_dataset_schema_and_sample(file_path: str) -> str:
    """Load dataset and return schema information with top 5 sample rows.

    Args:
        file_path: Path to the dataset file (CSV format)

    Returns:
        Formatted string containing:
        - Dataset shape (rows x columns)
        - Column names with data types
        - Top 5 sample rows
        - Error message if loading fails
    """
    try:
        # Load the dataset
        df = pd.read_csv(file_path)

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
    except pd.errors.ParserError as e:
        return f"Error: Failed to parse CSV file: {str(e)}"
    except Exception as e:
        return f"Error loading dataset schema: {str(e)}"
