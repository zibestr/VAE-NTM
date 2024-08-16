from torch.utils.data import Dataset
import torch
from typing import Iterable
from src.data.bow import BagOfWords  # type: ignore


class TextDataset(Dataset):
    def __init__(self,
                 texts: Iterable[str],
                 stopwords_path: str,
                 min_df: int = 1,
                 device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else
                'mps' if torch.backends.mps.is_available() else
                'cpu'
            )
        else:
            self.device = torch.device(device)

        self._bow_transformer = BagOfWords(stopwords_path,
                                           min_df=min_df)
        self._bow_transformer.fit(texts)
        self._content = torch.tensor(
            self._bow_transformer.transform(texts),
            dtype=torch.float32,
            device=self.device
        )

    @property
    def vocab_size(self) -> int:
        return len(self._bow_transformer._vocabulary)

    def __len__(self) -> int:
        return len(self._content)

    def __getitem__(self, key: int) -> torch.Tensor:
        return self._content[key]
