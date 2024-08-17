import os

from src.data.dataset import TextDataset


class DataOrganizer:
    def __init__(self,
                 data_path: str,
                 stopwords_path: str,
                 min_df: int = 1,
                 bigrams: bool = False,
                 encoding: str = 'UTF-8'):
        self.docs = {}
        texts = []
        for i, doc in enumerate(os.listdir(data_path)):
            full_path = os.path.join(data_path, doc)
            with open(full_path, 'r', encoding=encoding) as f:
                self.docs[full_path] = i
                texts.append(f.read())
        self.data = TextDataset(texts,
                                stopwords_path,
                                bigrams=bigrams,
                                min_df=min_df)
