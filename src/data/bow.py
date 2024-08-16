from collections import Counter
from collections.abc import Iterable

import numpy as np

from src.data.normalizer import TextNormalizer  # type: ignore


class BagOfWords:
    def __init__(self,
                 stopwords_path: str,
                 min_df: int = 1):
        self.normalizer = TextNormalizer(stopwords_path)
        self._min_df = min_df
        self._inverse_transform: dict[int, str] = {}
        self._vocabulary: dict[str, int] = {}

    def fit(self,
            texts: Iterable[str]):
        tokens = []
        for text in texts:
            tokens += self.normalizer(text)
        counter_tokens = Counter(tokens)
        filtered_tokens = sorted(
            list(
                map(lambda pair: pair[0],
                    filter(lambda pair: pair[1] >= self._min_df,
                           counter_tokens.items()))
            )
        )
        self._inverse_transform = {i: token
                                   for i, token in enumerate(filtered_tokens)}
        self._vocabulary = {token: i
                            for i, token in enumerate(filtered_tokens)}

    def _transform_text(self,
                        text: str) -> np.ndarray:
        vector = np.zeros(shape=len(self._vocabulary))
        tokens = self.normalizer(text)
        for token in tokens:
            vector[self._vocabulary[token]] += 1
        return vector

    def transform(self,
                  texts: Iterable[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            vectors.append(self._transform_text(text))
        return np.array(vectors)
