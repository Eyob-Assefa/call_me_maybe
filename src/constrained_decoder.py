from typing import Any, Dict, List, Optional, Set
import numpy as np

from src.schemas import FunctionCallOutput, FunctionDefinition
from src.tokenizer_index import TokenizerIndex

"""Constrained decoding engine for structured function calling."""

class ConstrainedDecoder:
    """Orchestrates schema-guided generation using logit masking."""

    def __init__(self, model: Any, tokenizer_index: TokenizerIndex) -> None:
        """Initialize the decoder with model and token index.

        Args:
            model: An instance of Small_LLM_Model from llm_sdk.
            tokenizer_index: Initialized TokenizerIndex for token filtering.
        """
        self.model = model
        self.index = tokenizer_index

    def mask_and_select(
        self,
        logits: List[float],
        allowed_token_ids: Set[int],
    ) -> int:
        """Apply a negative infinity mask to invalid tokens and select argmax.

        Args:
            logits: Raw unnormalized logit scores from the model.
            allowed_token_ids: Set of token IDs permitted by the current schema state.

        Returns:
            The token ID with the highest score among allowed tokens.
        """
        logits_array = np.array(logits, dtype=np.float32)

        if not allowed_token_ids:
            # Fallback to absolute argmax if no constraints are given
            return int(np.argmax(logits_array))

        # Mask all illegal tokens to negative infinity
        masked_logits = np.full_like(logits_array, -1e9)
        allowed_indices = list(allowed_token_ids)
        masked_logits[allowed_indices] = logits_array[allowed_indices]

        return int(np.argmax(masked_logits))

    def select_function_name(
        self,
        prompt: str,
        function_defs: List[FunctionDefinition],
    ) -> str:
        """Select a valid function name using constrained prefix decoding.

        Args:
            prompt: User natural language request.
            function_defs: List of available function specifications.

        Returns:
            The selected valid function name string.
        """
        function_names = [f.name for f in function_defs]
        query = f"Task: Select function\nUser: {prompt}\nFunction: "
        input_ids: List[int] = list(self.model.encode(query))

        generated_name = ""
        max_tokens = 30

        for _ in range(max_tokens):
            if generated_name in function_names:
                break

            valid_tokens = self.index.get_valid_continuation_tokens(
                generated_name,
                function_names,
            )
            if not valid_tokens:
                break

            logits = self.model.get_logits_from_input_ids(input_ids)
            next_token_id = self.mask_and_select(logits, valid_tokens)
            token_str = self.index.id_to_token.get(next_token_id, "")

            generated_name += token_str
            input_ids.append(next_token_id)

        # Fallback if generation broke prematurely
        for name in function_names:
            if name.startswith(generated_name) or generated_name.startswith(name):
                return name

        return function_names[0] if function_names else ""

    def extract_number_parameter(
        self,
        prompt: str,
        func_name: str,
        param_name: str,
    ) -> float:
        """Extract a numeric parameter value using constrained character tokens.

        Args:
            prompt: Original natural language instruction.
            func_name: Chosen function name.
            param_name: Target parameter key.

        Returns:
            The parsed floating-point number.
        """
        query = (
            f"Context: {prompt}\n"
            f"Function: {func_name}\n"
            f"Extract numeric value for '{param_name}': "
        )
        input_ids: List[int] = list(self.model.encode(query))

        # Allowed characters for valid numbers
        valid_chars = set("0123456789.- ")
        number_tokens: Set[int] = {
            tid for tid, s in self.index.id_to_token.items()
            if s and all(c in valid_chars for c in s)
        }

        generated_str = ""
        for _ in range(15):
            logits = self.model.get_logits_from_input_ids(input_ids)
            next_token_id = self.mask_and_select(logits, number_tokens)
            token_str = self.index.id_to_token.get(next_token_id, "")

            # If model generates whitespace after collecting digits, stop
            if generated_str.strip() and ("\n" in token_str or " " in token_str):
                break

            generated_str += token_str
            input_ids.append(next_token_id)

            if len(generated_str.strip()) >= 10:
                break

        try:
            return float(generated_str.strip())
        except ValueError:
            return 0.0

    def extract_string_parameter(
        self,
        prompt: str,
        func_name: str,
        param_name: str,
    ) -> str:
        """Extract a string parameter value from the prompt context.

        Args:
            prompt: Original natural language instruction.
            func_name: Chosen function name.
            param_name: Target parameter key.

        Returns:
            Extracted string value.
        """
        query = (
            f"Context: {prompt}\n"
            f"Function: {func_name}\n"
            f"Extract string value for '{param_name}': "
        )
        input_ids: List[int] = list(self.model.encode(query))

        generated_str = ""
        for _ in range(30):
            logits = self.model.get_logits_from_input_ids(input_ids)
            # Unconstrained text generation up to newline / quote
            next_token_id = int(np.argmax(logits))
            token_str = self.index.id_to_token.get(next_token_id, "")

            if "\n" in token_str or '"' in token_str:
                break

            generated_str += token_str
            input_ids.append(next_token_id)

        clean_result = generated_str.strip().strip("'\"")
        return clean_result

    def decode_function_call(
        self,
        prompt: str,
        function_defs: List[FunctionDefinition],
    ) -> FunctionCallOutput:
        """Execute the full constrained decoding pipeline for a single prompt.

        Args:
            prompt: Natural language user instruction.
            function_defs: List of all valid function specifications.

        Returns:
            A structured, validated FunctionCallOutput object.
        """
        chosen_name = self.select_function_name(prompt, function_defs)

        # Retrieve parameter schema for the selected function
        target_def: Optional[FunctionDefinition] = next(
            (f for f in function_defs if f.name == chosen_name),
            None,
        )

        extracted_params: Dict[str, Any] = {}

        if target_def and target_def.parameters:
            for param_name, param_spec in target_def.parameters.items():
                if param_spec.type == "number":
                    extracted_params[param_name] = self.extract_number_parameter(
                        prompt,
                        chosen_name,
                        param_name,
                    )
                elif param_spec.type == "string":
                    extracted_params[param_name] = self.extract_string_parameter(
                        prompt,
                        chosen_name,
                        param_name,
                    )
                elif param_spec.type == "boolean":
                    # Simple boolean constrained branch
                    str_val = self.extract_string_parameter(prompt, chosen_name, param_name)
                    extracted_params[param_name] = str_val.lower() == "true"
                else:
                    extracted_params[param_name] = self.extract_string_parameter(
                        prompt,
                        chosen_name,
                        param_name,
                    )

        return FunctionCallOutput(
            prompt=prompt,
            name=chosen_name,
            parameters=extracted_params,
        )