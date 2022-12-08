import collections
import json
import os
import re
import string
from typing import Dict, Iterable, List, Tuple, Union, Callable

from sentence_transformers.models import tokenizer
import pickle


def split_whitespace(s):
    """
    this function is explicitly defined because just using lambda expression would screw up pickling
    """
    return s.split()


class CustomTokenizer(tokenizer.WordTokenizer):
    """
    Custom sentence transformers tokenizer

    tokenize_fn function can be specified by the user
    """

    def __init__(
        self,
        vocab: Iterable[str] = [],
        stop_words: Iterable[str] = [],
        do_lower_case: bool = False,
        tokenize_fn: Callable[[str], List[str]] = split_whitespace,
    ):
        self.stop_words = set(stop_words)
        self.do_lower_case = do_lower_case
        self.set_vocab(vocab)
        self.tokenize_fn = tokenize_fn

    def get_vocab(self):
        return self.vocab

    def set_vocab(self, vocab: Iterable[str]):
        self.vocab = vocab
        self.word2idx = collections.OrderedDict(
            [(word, idx) for idx, word in enumerate(vocab)]
        )

    def is_token_a_word(self, token):
        number_match = re.match(r"\d+\.?\d+", token)
        return number_match is None or number_match.string != token

    def tokenize(self, text: str) -> List[int]:
        if self.do_lower_case:
            text = text.lower()

        tokens = self.tokenize_fn(text)

        tokens_filtered = []
        for token in tokens:
            if token in self.stop_words:
                continue
            elif token in self.word2idx:
                tokens_filtered.append(self.word2idx[token])
                continue

            token = token.strip(string.punctuation)
            if token in self.stop_words:
                continue
            elif len(token) > 0 and token in self.word2idx:
                tokens_filtered.append(self.word2idx[token])
                continue

            token = token.lower()
            if token in self.stop_words:
                continue
            elif token in self.word2idx:
                tokens_filtered.append(self.word2idx[token])
                continue

        return tokens_filtered

    def save(self, output_path: str):
        with open(
            os.path.join(output_path, "whitespacetokenizer_config.json"), "w"
        ) as fOut:
            json.dump(
                {
                    "vocab": list(self.word2idx.keys()),
                    "stop_words": list(self.stop_words),
                    "do_lower_case": self.do_lower_case,
                },
                fOut,
            )
        with open(os.path.join(output_path, "tokenize_fn.pkl"), "wb") as f:
            pickle.dump(self.tokenize_fn, f)

    @staticmethod
    def load(input_path: str):
        with open(
            os.path.join(input_path, "whitespacetokenizer_config.json"), "r"
        ) as fIn:
            config = json.load(fIn)

        with open(os.path.join(input_path, "tokenize_fn.pkl"), "rb") as f:
            config["tokenize_fn"] = pickle.load(f)

        return CustomTokenizer(**config)
