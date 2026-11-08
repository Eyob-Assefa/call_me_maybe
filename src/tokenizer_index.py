import json
from typing import Dict, List, Optional, Set
from pydantic import BaseModel, Field

"""Vocabulary and token indexing for fast constrained decoding."""


class TrieNode:
    """Node in a character-level prefix tree (Trie)."""

    def __init__(self) -> None:
        """Initialize child links and completion marker."""
        self.children: Dict[str, "TrieNode"] = {}
        self.is_terminal: bool = False


class PrefixTrie:
    """Trie structure to query valid string continuations efficiently."""

    def __init__(self) -> None:
        """Initialize root node."""
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """Insert a valid target word into the trie.

        Args:
            word: Target string to register.
        """
        curr = self.root
        for char in word:
            if char not in curr.children:
                curr.children[char] = TrieNode()
            curr = curr.children[char]
        curr.is_terminal = True

    def get_node(self, prefix: str) -> Optional[TrieNode]:
        """Traverse to the node representing the given prefix.

        Args:
            prefix: The current string prefix.

        Returns:
            The TrieNode at the end of prefix, or None if prefix is invalid.
        """
        curr = self.root
        for char in prefix:
            if char not in curr.children:
                return None
            curr = curr.children[char]
        return curr


class TokenizerConfig(BaseModel):
    """Configuration wrapper for the indexed vocabulary."""

    vocab_size: int = Field(..., description="Total count of unique tokens")


class TokenizerIndex:
    """Index mapping vocabulary strings to token IDs for constrained masking."""

    def __init__(self, vocab_path: str) -> None:
        """Load and index the model vocabulary from disk.

        Args:
            vocab_path: Filepath to the tokenizer vocabulary JSON.
        """
        with open(vocab_path, "r", encoding="utf-8") as file_handle:
            raw_vocab = json.load(file_handle)

        self.id_to_token: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {}

        # Handle both {"token": id} and [token, ...] structures
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
        """Normalize subword whitespace representations to standard spaces.

        Args:
            token: Raw token string from the tokenizer vocabulary.

        Returns:
            Normalized token string.
        """
        # Common BPE / SentencePiece space indicators
        return token.replace("Ġ", " ").replace(" ", " ")

    def get_valid_continuation_tokens(
        self,
        current_text: str,
        target_words: List[str],
    ) -> Set[int]:
        """Find all token IDs that legally continue current_text towards any target word.

        Args:
            current_text: The string generated so far.
            target_words: Complete valid target strings (e.g. valid function names).

        Returns:
            A set of token IDs that are legal next tokens.
        """
        valid_token_ids: Set[int] = set()

        for target in target_words:
            # Current text must still match the start of this target
            if not target.startswith(current_text):
                continue

            remaining = target[len(current_text):]

            # Look through vocab to see which tokens match the beginning of remaining
            for token_id, token_str in self.id_to_token.items():
                if not token_str:
                    continue
                if remaining.startswith(token_str):
                    valid_token_ids.add(token_id)

        return valid_token_ids