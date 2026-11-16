*This project has been created as part of the 42 curriculum by eshikur.*

# Call Me Maybe

## Description

Call Me Maybe is a constrained decoding engine that converts natural language prompts into structured JSON function calls.

Small language models, such as `Qwen/Qwen3-0.6B`, can struggle to generate valid JSON. This project solves that problem by controlling which tokens the model can generate at each step. Invalid tokens are blocked, ensuring that the final output follows the provided function definitions and JSON structure.

## Instructions

### Prerequisites

- Python 3.10 or higher
- `uv` package manager

### Installation

Clone the repository, open the project folder, and install the dependencies:

```bash
uv sync
```

### Execution

Run the project using:

```bash
make run
```

You can also run it directly and provide custom file paths:

```bash
uv run python -m src --functions_definition data/input/functions_definition.json --input data/input/function_calling_tests.json --output data/output/function_calls.json
```

### Linting and Type Checking

Run the following command to check code style and types:

```bash
make lint
```

## Algorithm Explanation

The decoding process follows four main steps:

1. **Vocabulary Indexing:** `TokenizerIndex` loads the model vocabulary and maps token IDs to readable tokens. It also handles special characters used for spaces and line breaks.

2. **Logit Masking:** At each generation step, the model assigns scores to possible tokens. The decoder blocks tokens that are not allowed and selects the highest-scoring valid token.

3. **Function Selection:** The available function names and descriptions are added to the prompt. The decoder only allows tokens that match a valid function name and stops when a complete match is found.

4. **Parameter Extraction:** After selecting a function, the engine uses its parameter definitions to extract values such as numbers, strings, and booleans. The JSON structure is built around these values.

## Design Decisions

- **Modular Structure:** The project is divided into separate modules for data validation, file handling, vocabulary lookup, decoding, and main execution.
- **Fixed JSON Structure:** Instead of asking the model to generate the entire JSON output, the program builds the structure itself. The model only generates the function name and parameter values.
- **Precomputed Masks:** Common token groups, such as digits and valid string characters, are prepared at startup to reduce repeated work during decoding.
- **Error Handling:** The program handles invalid files, incorrect JSON, and decoding errors with clear error messages.

## Performance Analysis

- **Accuracy:** Restricting function names to the available definitions prevents the model from generating unknown function names. Providing function descriptions also helps with parameter selection.
- **Speed:** The model takes around 40–90 ms per forward pass on a standard CPU laptop. Since the program generates fewer tokens by building the JSON structure itself, a test set of 10–50 prompts runs in under one minute.
- **Reliability:** Every generated result is checked using Pydantic before being saved, ensuring that the output follows the expected format.

## Challenges Faced

- **Special Characters:** The tokenizer represents spaces and line breaks using special characters such as `Ġ` and `Ċ`. This caused issues when extracting strings and detecting line breaks. The problem was fixed by converting these markers into their correct characters.
- **Token Encoding:** The model's `encode()` method returned a 2D tensor instead of a flat list. A helper function was added to convert it into a standard list of token IDs.
- **String Handling:** Early versions sometimes generated extra text or stopped with empty strings. Using a JSON prefix for string parameters helped the model generate and close strings correctly while still allowing empty values.

## Testing Strategy

The project was tested in three main ways:

1. **Code Quality:** Used `flake8` and `mypy` to check code style and type errors.
2. **Functional Testing:** Tested numbers, multiple parameters, negative and decimal values, strings, and empty strings using `data/input/function_calling_tests.json`.
3. **Error Testing:** Tested missing files, invalid JSON, and empty input arrays to check that the program handles errors correctly.

## Example Usage

### Input

File: `data/input/function_calling_tests.json`

```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Greet shrek" },
  { "prompt": "Reverse the string 'hello'" }
]
```

### Run

```bash
uv run python -m src
```

### Expected Output

File: `data/output/function_calls.json`

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {
      "name": "shrek"
    }
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {
      "s": "hello"
    }
  }
]
```

## Resources

- **Hugging Face Documentation:** Used to understand tokenizers, model vocabularies, and model outputs.
- **Outlines Documentation:** Used as a reference for constrained decoding and token filtering.
- **Pydantic v2 Documentation:** Used for data validation and JSON output handling.

**AI Usage Disclosure:** Gemini was used as a support tool to discuss the decoding approach, troubleshoot tokenizer issues and Python type hints, and review code style. All code was written, tested, and checked manually.