import gzip
from typing import Dict, Callable
import numpy as np


class CompressedKeyedVectors(object):
    def __init__(self, vocab_path: str, embedding_path: str, to_lowercase: bool = True):
        """
        Class from sdadas polish-nlp-resources
        https://github.com/sdadas/polish-nlp-resources
        I need to get it somewhere from where I can import it easily for using with custom BentoML model
        """
        self.vocab_path: str = vocab_path
        self.embedding_path: str = embedding_path
        self.to_lower: bool = to_lowercase
        self.vocab: Dict[str, int] = self.__load_vocab(vocab_path)
        embedding = np.load(embedding_path)
        self.codes: np.ndarray = embedding[embedding.files[0]]
        self.codebook: np.ndarray = embedding[embedding.files[1]]
        self.m = self.codes.shape[1]
        self.k = int(self.codebook.shape[0] / self.m)
        self.dim: int = self.codebook.shape[1]

    def __load_vocab(self, vocab_path: str) -> Dict[str, int]:
        open_func: Callable = gzip.open if vocab_path.endswith(".gz") else open
        with open_func(vocab_path, "rt", encoding="utf-8") as input_file:
            return {line.strip(): idx for idx, line in enumerate(input_file)}

    def vocab_vector(self, word: str):
        if word == "<pad>":
            return np.zeros(self.dim)
        val: str = word.lower() if self.to_lower else word
        index: int = self.vocab.get(val, self.vocab["<unk>"])
        codes = self.codes[index]
        code_indices = np.array(
            [idx * self.k + offset for idx, offset in enumerate(np.nditer(codes))]
        )
        return np.sum(self.codebook[code_indices], axis=0)

    def __getitem__(self, key):
        return self.vocab_vector(key)
