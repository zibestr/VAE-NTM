from torch.utils.data import Dataset
import torch
from typing import Iterable
from src.data.normalizer import TextNormalizer
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore


class TextDataset(Dataset):
    def __init__(self,
                 texts: Iterable[str],
                 stopwords_path: str,
                 min_df: int = 1,
                 bigrams: bool = False,
                 device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else
                'mps' if torch.backends.mps.is_available() else
                'cpu'
            )
        else:
            self.device = torch.device(device)

        self._normalizer = TextNormalizer(stopwords_path)
        texts = [' '.join(self._normalizer(text)) for text in texts]
        self.transformer = TfidfVectorizer(min_df=min_df,
                                           ngram_range=(1 + bigrams,
                                                        1 + bigrams))
        self.transformer.fit(texts)
        self._content = torch.tensor(
            self.transformer.transform(texts).toarray(),
            dtype=torch.float32,
            device=self.device
        )

    @property
    def vocab_size(self) -> int:
        return len(self.transformer.vocabulary_)

    def __len__(self) -> int:
        return len(self._content)

    def __getitem__(self, key: int | list[int]) -> torch.Tensor:
        if isinstance(key, int):
            return self._content[key]
        if len(key) == 1:
            return self._content[key].reshape(1, -1)
        return torch.vstack([self._content[k] for k in key])
