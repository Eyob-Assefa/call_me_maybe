"""Constrained decoding engine for structured function calling."""

import json
from typing import Any, Dict, List, Optional, Set
import numpy as np

from src.schemas import FunctionCallOutput, FunctionDefinition
from src.tokenizer_index import TokenizerIndex


class ConstrainedDecoder:
    """Orchestrates schema-guided generation using logit masking."""

    def __init__(self, model: Any, tokenizer_index: TokenizerIndex) -> None:
        """Initialize decoder with model and vocabulary index."""
        self.model = model
        self.index = tokenizer_index

        self.forbidden_str_tokens: Set[int] = {
            tid for tid, s in self.index.id_to_token.items()
            if "\n" in s or "\r" in s or not s
        }
        digit_chars = set("0123456789.,}")
        self.digit_tokens: Set[int] = {
            tid for tid, s in self.index.id_to_token.items()
            if s and all(c in digit_chars for c in s)
        }
        num_start_chars = set("-0123456789. ")
        self.num_start_tokens: Set[int] = {
            tid for tid, s in self.index.id_to_token.items()
            if s and all(c in num_start_chars for c in s)
            and s.count("-") <= 1
            and ("-" not in s or s.lstrip().startswith("-"))
        }

    def _encode_to_ids(self, text: str) -> List[int]:
        """Encode text via SDK into a flat 1D list of token IDs."""
        encoded = self.model.encode(text)
        if hasattr(encoded, "tolist"):
            raw_list = encoded.tolist()
            if raw_list and isinstance(raw_list[0], list):
                return [int(tid) for tid in raw_list[0]]
            return [int(tid) for tid in raw_list]
        return [int(tid) for tid in encoded]

    def mask_and_select(
        self,
        logits: List[float],
        allowed_token_ids: Set[int],
    ) -> int:
        """Mask disallowed tokens to -inf and return highest-scoring ID."""
        logits_array = np.array(logits, dtype=np.float32)
        if not allowed_token_ids:
            return int(np.argmax(logits_array))

        masked_logits = np.full_like(logits_array, -1e9)
        allowed_indices = list(allowed_token_ids)
        masked_logits[allowed_indices] = logits_array[allowed_indices]
        return int(np.argmax(masked_logits))

    def select_function_name(
        self,
        prompt: str,
        function_defs: List[FunctionDefinition],
    ) -> str:
        """Select a valid function name using constrained prefix decoding."""
        function_names = [f.name for f in function_defs]
        tools_desc = "\n".join(
            f"- {f.name}({', '.join(f.parameters.keys())}): {f.description}"
            for f in function_defs
        )
        query = (
            f"Select the single best function for the user request.\n"
            f"Functions:\n{tools_desc}\n\n"
            f"Request: {prompt}\n"
            f"Selected function: "
        )
        input_ids: List[int] = self._encode_to_ids(query)
        generated_name = ""

        for _ in range(30):
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
            generated_name += self.index.id_to_token.get(next_token_id, "")
            input_ids.append(next_token_id)

        for name in function_names:
            if (
                name.startswith(generated_name)
                or generated_name.startswith(name)
            ):
                return name
        return function_names[0] if function_names else ""

    def _build_param_prefix(
        self,
        prompt: str,
        func_name: str,
        prior_params: Dict[str, Any],
        param_name: str,
    ) -> str:
        """Format JSON context including already-extracted parameters."""
        partial = json.dumps(prior_params)[:-1]
        sep = ", " if prior_params else ""
        return (
            f"User request: {prompt}\n"
            f'JSON call: {{"name": "{func_name}", '
            f'"parameters": {partial}{sep}"{param_name}":'
        )

    def extract_number_parameter(
        self,
        prompt: str,
        func_name: str,
        param_name: str,
        prior_params: Dict[str, Any],
    ) -> float:
        """Extract a numeric parameter using constrained numeric tokens."""
        query = self._build_param_prefix(
            prompt, func_name, prior_params, param_name
        )
        input_ids: List[int] = self._encode_to_ids(query)
        generated_str = ""

        for _ in range(15):
            allowed = (
                self.digit_tokens
                if generated_str.strip()
                else self.num_start_tokens
            )
            logits = self.model.get_logits_from_input_ids(input_ids)
            next_token_id = self.mask_and_select(logits, allowed)
            token_str = self.index.id_to_token.get(next_token_id, "")

            if generated_str.strip() and (
                "," in token_str or "}" in token_str
            ):
                for sep in (",", "}"):
                    if sep in token_str:
                        token_str = token_str.split(sep)[0]
                generated_str += token_str
                break

            generated_str += token_str
            input_ids.append(next_token_id)
            if len(generated_str.strip()) >= 12:
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
        prior_params: Dict[str, Any],
    ) -> str:
        """Extract a string parameter using logit masking."""
        query = (
            self._build_param_prefix(
                prompt, func_name, prior_params, param_name
            )
            + '"'
        )
        input_ids: List[int] = self._encode_to_ids(query)
        allowed = (
            set(self.index.id_to_token.keys()) - self.forbidden_str_tokens
        )
        generated_str = ""

        for _ in range(40):
            logits = self.model.get_logits_from_input_ids(input_ids)
            next_token_id = self.mask_and_select(logits, allowed)
            token_str = self.index.id_to_token.get(next_token_id, "")

            if '"' in token_str:
                generated_str += token_str.split('"')[0]
                break

            generated_str += token_str
            input_ids.append(next_token_id)

        return generated_str

    def decode_function_call(
        self,
        prompt: str,
        function_defs: List[FunctionDefinition],
    ) -> FunctionCallOutput:
        """Run constrained decoding for a single prompt."""
        chosen_name = self.select_function_name(prompt, function_defs)
        target_def: Optional[FunctionDefinition] = next(
            (f for f in function_defs if f.name == chosen_name),
            None,
        )
        extracted: Dict[str, Any] = {}

        if target_def and target_def.parameters:
            for param_name, param_spec in target_def.parameters.items():
                if param_spec.type == "number":
                    extracted[param_name] = self.extract_number_parameter(
                        prompt, chosen_name, param_name, extracted
                    )
                elif param_spec.type == "integer":
                    val_num = self.extract_number_parameter(
                        prompt, chosen_name, param_name, extracted
                    )
                    extracted[param_name] = int(round(val_num))
                elif param_spec.type == "boolean":
                    val_str = self.extract_string_parameter(
                        prompt, chosen_name, param_name, extracted
                    )
                    extracted[param_name] = val_str.lower() == "true"
                else:
                    extracted[param_name] = self.extract_string_parameter(
                        prompt, chosen_name, param_name, extracted
                    )

        return FunctionCallOutput(
            prompt=prompt,
            name=chosen_name,
            parameters=extracted,
        )
