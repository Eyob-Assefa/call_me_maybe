"""Vocabulary and token indexing for fast constrained decoding."""

import json
from typing import Dict, List, Set
from pydantic import BaseModel, Field


class TokenizerConfig(BaseModel):
    """Configuration wrapper for the indexed vocabulary."""

    vocab_size: int = Field(..., description="Total count of unique tokens")


class TokenizerIndex:
    """Index mapping vocabulary strings to token IDs for masking."""

    def __init__(self, vocab_path: str) -> None:
        """Load and index the model vocabulary from disk."""
        with open(vocab_path, "r", encoding="utf-8") as file_handle:
            raw_vocab = json.load(file_handle)

        self.id_to_token: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {}

        if isinstance(raw_vocab, dict):
            for token_str, token_id in raw_vocab.items():
                tid = int(token_id)
                clean_str = self._clean_token(token_str)
                self.id_to_token[tid] = clean_str
                self.token_to_id[clean_str] = tid
        elif isinstance(raw_vocab, list):
            for tid, token_str in enumerate(raw_vocab):
                clean_str = self._clean_token(str(token_str))
                self.id_to_token[tid] = clean_str
                self.token_to_id[clean_str] = tid

        self.config = TokenizerConfig(vocab_size=len(self.id_to_token))

    @staticmethod
    def _clean_token(token: str) -> str:
        """Normalize subword whitespace and newline indicators."""
        return (
            token.replace("Ġ", " ")
            .replace(" ", " ")
            .replace("Ċ", "\n")
            .replace("ĉ", "\t")
        )

    def get_valid_continuation_tokens(
        self,
        current_text: str,
        target_words: List[str],
    ) -> Set[int]:
        """Find token IDs that legally continue current_text."""
        valid_token_ids: Set[int] = set()

        for target in target_words:
            if not target.startswith(current_text):
                continue

            remaining = target[len(current_text):]
            for token_id, token_str in self.id_to_token.items():
                if token_str and remaining.startswith(token_str):
                    valid_token_ids.add(token_id)

        return valid_token_ids
