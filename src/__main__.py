import argparse
import sys
from typing import List

from src.constrained_decoder import ConstrainedDecoder
from src.io_utils import (
    load_function_definitions,
    load_prompts,
    save_function_calls,
)
from src.schemas import FunctionCallOutput
from src.tokenizer_index import TokenizerIndex


"""Main command-line entry point for constrained function calling."""


def parse_arguments() -> argparse.Namespace:
    """Parse and return command-line arguments with project defaults.

    Returns:
        Parsed CLI arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Constrained decoding function calling engine.",
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the JSON file with function definitions.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the input JSON file containing test prompts.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calls.json",
        help="Destination path for the generated function calls JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the end-to-end function calling generation pipeline."""
    args = parse_arguments()

    print("[1/5] Loading input files...")
    function_defs = load_function_definitions(args.functions_definition)
    if not function_defs:
        print("Aborting: No valid function definitions loaded.", file=sys.stderr)
        sys.exit(1)

    test_prompts = load_prompts(args.input)
    if not test_prompts:
        print("Aborting: No test prompts loaded.", file=sys.stderr)
        sys.exit(1)

    print(
        f"Loaded {len(function_defs)} function definitions and "
        f"{len(test_prompts)} prompts."
    )

    print("[2/5] Initializing LLM SDK and vocabulary...")
    try:
        # Import local llm_sdk module directly
        from llm_sdk import Small_LLM_Model  # type: ignore[import-untyped]

        model = Small_LLM_Model()
        vocab_path = model.get_path_to_vocab_file()
    except (ImportError, AttributeError, OSError) as err:
        print(f"Error: Failed to initialize LLM SDK: {err}", file=sys.stderr)
        sys.exit(1)

    tokenizer_index = TokenizerIndex(vocab_path)
    decoder = ConstrainedDecoder(model, tokenizer_index)

    print("[3/5] Processing prompts with constrained decoding...")
    results: List[FunctionCallOutput] = []

    for index, prompt_item in enumerate(test_prompts, start=1):
        print(f"  Processing prompt {index}/{len(test_prompts)}: '{prompt_item.prompt}'")
        try:
            output = decoder.decode_function_call(prompt_item.prompt, function_defs)
            results.append(output)
        except Exception as err:
            print(
                f"Warning: Failed to process prompt '{prompt_item.prompt}': {err}",
                file=sys.stderr,
            )
            # Safe empty fallback to preserve schema integrity
            results.append(
                FunctionCallOutput(
                    prompt=prompt_item.prompt,
                    name=function_defs[0].name,
                    parameters={},
                )
            )

    print(f"[4/5] Saving {len(results)} outputs to {args.output}...")
    success = save_function_calls(results, args.output)
    if not success:
        print(f"Error: Failed to write output to {args.output}", file=sys.stderr)
        sys.exit(1)

    print("[5/5] Done! Execution completed successfully.")


if __name__ == "__main__":
    main()