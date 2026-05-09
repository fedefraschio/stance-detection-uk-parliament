import requests
import pandas as pd


class StanceSummarizer:
    _TOPIC_VERBOSE = {
        'nuclear': 'the use of nuclear energy as an energy source for the future'
    }

    def __init__(self, model_name='gemma3', random_seed=42):
        self.model_name = model_name
        self.random_seed = random_seed

    def summarize_speaker(self, speaker_name: str, topic: str, speaker_sentences: pd.DataFrame) -> pd.DataFrame:
        topic_verbose = self._TOPIC_VERBOSE.get(topic, topic)
        full_text = "\n".join(speaker_sentences['text'].tolist())

        context_prompt = (
            f"Based on the politician’s statements in the following parliamentary speeches, "
            f"infer and summarize this politician’s stance on {topic_verbose}. "
            f"This summary is intended for those who have little knowledge of British politics. "
            f"Please summarize in a way that is easy to understand even for those who are not interested in politics. "
            f"\n{full_text}."
        )
        sys_prompt = (
            "You are an expert in UK parliamentary politics. "
            "Provide only a single, clear, concise, and purely factual summary sentence of the politician's stance. "
            "Do not use introductory phrases like 'Okay,', 'Based on,', or 'From this,'. "
            "Do not use colloquial language, filler words, or explanations. "
            "Output the summary sentence directly."
        )

        payload = {
            "model": self.model_name,
            "prompt": context_prompt,
            "system": sys_prompt,
            "stream": False,
            "options": {"seed": self.random_seed}
        }
        try:
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
            response.raise_for_status()
            summary = response.json()['response'].strip()
        except requests.exceptions.RequestException as e:
            print(f"\nError generating summary for speaker {speaker_name}: {e}")
            summary = f"ERROR: {str(e)}"

        return pd.DataFrame([{
            'summary': summary,
            'party': speaker_sentences['party'].iloc[0],
            'speaker': speaker_name
        }])

    def summarize_all(self, classified_df: pd.DataFrame, topic: str) -> pd.DataFrame:
        speakers = classified_df['speaker'].unique()
        summaries = [
            self.summarize_speaker(speaker, topic, classified_df[classified_df['speaker'] == speaker])
            for speaker in speakers
        ]
        result_df = pd.concat(summaries, ignore_index=True)
        result_df = result_df[~result_df['summary'].str.startswith('Please provide')]
        print("Summarization completed for topic:", topic)
        return result_df
