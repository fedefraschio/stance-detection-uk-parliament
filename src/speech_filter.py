import pandas as pd


class SpeechFilter:
    def filter(self, df: pd.DataFrame, keywords: list, policyarea: list, years=None) -> pd.DataFrame:
        df = df.copy()
        if years is not None:
            df = df[df['year'].isin(years)]
        if policyarea:
            df = df[df['policyarea'].isin(policyarea)]
        df = df[df['text'].str.contains('|'.join(keywords), case=False, na=False)]
        return df
