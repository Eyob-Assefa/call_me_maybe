"""Input and output utility functions with graceful error handling."""

import json
import os
import sys
from typing import List
from pydantic import ValidationError

from src.schemas import FunctionCallOutput, FunctionDefinition, PromptItem


def load_function_definitions(filepath: str) -> List[FunctionDefinition]:
    """Load and validate function definitions from a JSON file."""
    if not os.path.exists(filepath):
        print(
            f"Error: Function definitions file not found: {filepath}",
            file=sys.stderr,
        )
        return []

    try:
        with open(filepath, "r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)

        if not isinstance(data, list):
            print(
                f"Error: Expected a JSON array in {filepath}",
                file=sys.stderr,
            )
            return []

        return [FunctionDefinition.model_validate(item) for item in data]

    except json.JSONDecodeError as json_err:
        print(
            f"Error: Malformed JSON in {filepath}: {json_err}",
            file=sys.stderr,
        )
        return []
    except ValidationError as val_err:
        print(
            f"Error: Schema validation failed for {filepath}: {val_err}",
            file=sys.stderr,
        )
        return []
    except OSError as os_err:
        print(f"Error: Unable to read {filepath}: {os_err}", file=sys.stderr)
        return []
    except Exception as err:
        print(
            f"Unexpected error loading {filepath}: {err}",
            file=sys.stderr,
        )
        return []


def load_prompts(filepath: str) -> List[PromptItem]:
    """Load and validate test prompts from a JSON file."""
    if not os.path.exists(filepath):
        print(f"Error: Prompts file not found: {filepath}", file=sys.stderr)
        return []

    try:
        with open(filepath, "r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)

        if not isinstance(data, list):
            print(
                f"Error: Expected a JSON array in {filepath}",
                file=sys.stderr,
            )
            return []

        return [PromptItem.model_validate(item) for item in data]

    except json.JSONDecodeError as json_err:
        print(
            f"Error: Malformed JSON in {filepath}: {json_err}",
            file=sys.stderr,
        )
        return []
    except ValidationError as val_err:
        print(
            f"Error: Schema validation failed for {filepath}: {val_err}",
            file=sys.stderr,
        )
        return []
    except OSError as os_err:
        print(f"Error: Unable to read {filepath}: {os_err}", file=sys.stderr)
        return []
    except Exception as err:
        print(
            f"Unexpected error loading {filepath}: {err}",
            file=sys.stderr,
        )
        return []


def save_function_calls(
    outputs: List[FunctionCallOutput],
    filepath: str,
) -> bool:
    """Serialize and write function call outputs to a JSON file."""
    try:
        output_dir = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        serialized_data = [item.model_dump() for item in outputs]

        with open(filepath, "w", encoding="utf-8") as file_handle:
            json.dump(serialized_data, file_handle, indent=2)

        return True

    except (OSError, TypeError) as err:
        print(
            f"Error: Failed to write output to {filepath}: {err}",
            file=sys.stderr,
        )
        return False
    except Exception as err:
        print(
            f"Unexpected error saving to {filepath}: {err}",
            file=sys.stderr,
        )
        return False
