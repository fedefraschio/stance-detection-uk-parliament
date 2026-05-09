import pandas as pd
from setfit import SetFitModel


class OpinionClassifier:
    def __init__(self, model_hf="andreacristiano/stancedetection"):
        self.model = SetFitModel.from_pretrained(model_hf)

    def classify(self, df_filtered: pd.DataFrame) -> pd.DataFrame:
        texts = df_filtered['text'].tolist()
        predictions = self.model.predict(texts)
        df = df_filtered.copy()
        df['classification'] = predictions
        df_classified = df[df['classification'] == 'opinion']
        speaker_counts = df_classified['speaker'].value_counts()
        valid_speakers = speaker_counts[speaker_counts > 1].index
        return df_classified[df_classified['speaker'].isin(valid_speakers)]
