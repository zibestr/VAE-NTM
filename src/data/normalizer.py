import re
from typing import Iterator

from pymorphy3 import MorphAnalyzer  # type: ignore


class TextNormalizer:
    def __init__(self, stopwords_path: str):
        self.analyzer = MorphAnalyzer()
        with open(stopwords_path, 'r') as f:
            self.stopwords = f.read().split('\n')

    @staticmethod
    def _delete_punctuation(text: str) -> str:
        text = re.sub(r'[!"#\$%&\'\(\)\*\+,-\.\/:;<=>\?@\[\\\]\^_`\{\|\}~—–]+',
                      ' ', text)
        return text

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = text.split()
        return tokens

    def _delete_stopwords(self, tokens: list[str]) -> Iterator[str]:
        return filter(lambda token: token not in self.stopwords,
                      tokens)

    def _lemmatize(self, tokens: Iterator[str]) -> list[str]:
        return [self.analyzer.normal_forms(token)[0]
                for token in tokens]

    def __call__(self, text: str) -> list[str]:
        text = text.lower()
        text = self._delete_punctuation(text)
        tokens = self._tokenize(text)
        filtered_tokens = self._delete_stopwords(tokens)
        lemms = self._lemmatize(filtered_tokens)
        return lemms
