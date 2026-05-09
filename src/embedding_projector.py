import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class EmbeddingProjector:
    def __init__(self, model_name="Qwen/Qwen3-Embedding-0.6B"):
        self.model_name = model_name
        self._model = None

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def compute_embeddings(self, summaries: list, anchors: dict, debug_mode=False):
        anchor_texts = [anchors['pro'], anchors['con']]
        all_texts = summaries + anchor_texts

        if debug_mode:
            print(all_texts)

        embeddings = self._load_model().encode(all_texts, show_progress_bar=True)
        speaker_embeddings = embeddings[:len(summaries)]
        anchor_embeddings = embeddings[len(summaries):]
        return speaker_embeddings, anchor_embeddings

    def axis_of_controversy(self, sum_df: pd.DataFrame, speaker_embeddings, anchor_embeddings, issue: str) -> pd.DataFrame:
        df = sum_df.copy().reset_index(drop=True)
        df['embedding'] = list(speaker_embeddings)

        party_centroids = (
            df
            .groupby("party")['embedding']
            .apply(lambda x: np.mean(list(x), axis=0))
            .reset_index()
            .rename(columns={'embedding': 'centroid'})
        )

        pro_emb, con_emb = anchor_embeddings[0], anchor_embeddings[1]
        midpoint = (pro_emb + con_emb) / 2
        axis = pro_emb - con_emb
        axis = axis / np.linalg.norm(axis)

        centroids_matrix = np.stack(party_centroids['centroid'].values)
        party_centroids['controversy_score'] = (centroids_matrix - midpoint) @ axis

        party_df = party_centroids[['party']].copy()
        party_df['controversy_score'] = party_centroids['controversy_score']
        party_df['issue'] = issue
        return party_df

    def cosine_similarity(self, vecA, vecB) -> float:
        dot_product = np.dot(vecA, vecB)
        normA = np.linalg.norm(vecA)
        normB = np.linalg.norm(vecB)
        if normA == 0 or normB == 0:
            return 0.0
        return dot_product / (normA * normB)
