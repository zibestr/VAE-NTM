from typing import Literal

import numpy as np
import pandas as pd  # type: ignore
from random import seed as set_seed
from os import environ
import torch
import plotly.graph_objects as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore

from src.app.messager import MessageHandler
from src.data.organizer import DataOrganizer
from src.model.trainer import UnsupervisedTrainer
from src.model.vae import VariationalAutoencoder


class Application:
    def __init__(self,
                 topics_num: int,
                 data_path: str,
                 stopwords_path: str,
                 random_state: int,
                 min_df: int = 1,
                 bigrams: bool = False,
                 display: Literal['console',
                                  'logs',
                                  'webpanel'] = 'console',
                 learning_rate: float = 1e-3,
                 epochs: int = 20):
        self.__set_seed(random_state)

        self.organizer = DataOrganizer(data_path, stopwords_path,
                                       min_df=min_df, bigrams=bigrams)
        self.messager = MessageHandler(display)
        self.topic_model = VariationalAutoencoder(
            topics_num,
            self.organizer.data.vocab_size
        )
        self.trainer = UnsupervisedTrainer(self.topic_model,
                                           learning_rate,
                                           self.organizer.data,
                                           2,
                                           self.messager)
        self.fit(epochs)

    def fit(self, epochs: int):
        losses = self.trainer.train(epochs)
        self.messager.receive(f'Train losses {losses}')

    def _get_normalized_texts(self, doc_paths: list[str]) -> torch.Tensor:
        num_texts: list[int] = []
        for doc_path in doc_paths:
            if doc_path not in self.organizer.docs:
                raise ValueError(f'Document not found: {doc_path}')
            num_texts.append(self.organizer.docs[doc_path])
        return self.organizer.data[num_texts]

    def get_topics(
        self,
        doc_paths: list[str],
        return_chart: bool = False
    ) -> np.ndarray | tuple[np.ndarray, go.Figure]:
        normalized_texts = self._get_normalized_texts(doc_paths)

        distributions = self.topic_model.topic_distribution(
            normalized_texts
        ).cpu().numpy()

        if return_chart:
            return distributions, self._distribution_charts(distributions)
        return distributions

    @staticmethod
    def _distribution_charts(distributions: np.ndarray) -> go.Figure:
        fig = make_subplots(cols=1, rows=distributions.shape[0])
        title = 'Topics probabilities '
        for i, dist in enumerate(distributions):
            df = pd.DataFrame(data={
                'topic': [f'topic #{i}' for i in range(len(dist))],
                'probability': dist
            })
            fig.add_trace(
                go.Bar(x=df['topic'], y=df['probability'],
                       name=title + str(i + 1)),
                row=i + 1, col=1
            )

        fig.update_layout(height=200 * distributions.shape[0],
                          width=200 * distributions.shape[1],
                          title_text="Topics probabilities of docs")
        return fig

    def get_keywords(self,
                     doc_paths: list[str],
                     num_words: int = 5) -> list[str]:
        normalized_texts = self._get_normalized_texts(doc_paths)

        inds = self.topic_model.sample_words(
            normalized_texts,
            num_words
        ).cpu().numpy()
        words = self.organizer.data.transformer.get_feature_names_out()

        return [words[ind].tolist() for ind in inds]

    def get_topics_with_keywords(
        self,
        doc_paths: list[str],
        num_words: int = 5,
        return_chart: bool = False
    ) -> tuple[go.Figure, list[str]] | tuple[np.ndarray, list[str]]:
        normalized_texts = self._get_normalized_texts(doc_paths)
        distributions, inds = self.topic_model.topic_distribution_with_words(
            normalized_texts,
            num_words
        )
        words = self.organizer.data.transformer.get_feature_names_out()
        if return_chart:
            return (self._distribution_charts(distributions.cpu().numpy()),
                    [words[ind].tolist() for ind in inds.cpu().numpy()])
        return (distributions,
                [words[ind].tolist() for ind in inds.cpu().numpy()])

    @staticmethod
    def __set_seed(seed: int):
        np.random.seed(seed)
        set_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.mps.manual_seed(seed)
        environ["PYTHONHASHSEED"] = str(seed)
