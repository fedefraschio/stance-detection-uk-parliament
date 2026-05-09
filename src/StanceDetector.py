import pandas as pd

from speech_filter import SpeechFilter
from opinion_classifier import OpinionClassifier
from stance_summarizer import StanceSummarizer
from anchor_generator import AnchorGenerator
from embedding_projector import EmbeddingProjector
from visualizer import Visualizer
from evaluator import Evaluator


class StanceDetector:
    """
    Orchestrate the full stance-detection pipeline for parliamentary speeches.

    Typical workflow:
    1. add_record      — register a topic with keywords and policy areas.
    2. filter_speeches — filter speeches by topic (and optionally year).
    3. classify_filtered_sentences — keep only opinion-bearing sentences.
    4. summarize_all_sentences     — generate per-speaker stance summaries.
    5. generate_anchors            — produce PRO/CON controversy anchors.
    6. compute_embeddings          — embed summaries and anchors.
    7. axis_of_controversy         — project parties onto the controversy axis.
    8. generate_gold_standard      — get LLM reference ordering for validation.
    9. evaluate_ordering           — compute Spearman/Kendall/LCS metrics.

    Visualization helpers: plot_axis_of_controversy, compute_umap_embeddings,
    plot_umap_party_averages.

    All intermediate results are stored in the internal record dict keyed by
    topic name, so each step can read the previous step's output automatically.
    """

    def __init__(self, speech, records, cl_model_hf="andreacristiano/stancedetection", random_seed=42):
        self.__speeches_df = speech
        self.__record = records
        self.random_seed = random_seed

        self._filter = SpeechFilter()
        self._classifier = OpinionClassifier(cl_model_hf)
        self._summarizer = StanceSummarizer(random_seed=random_seed)
        self._anchor_gen = AnchorGenerator(random_seed=random_seed)
        self._projector = EmbeddingProjector()
        self._visualizer = Visualizer()
        self._evaluator = Evaluator()

    # ==================== GETTERS ====================

    def get_records(self):
        return self.__record

    def get_speeches(self):
        return self.__speeches_df

    def get_filtered_speeches(self, topic):
        return self.__record[topic]['df_filtered']

    def get_classified_speeches(self, topic):
        return self.__record[topic]['df_classified']

    # ==================== SETTERS ====================

    def set_summarization_for_topic(self, topic, df_summarized_speaker):
        self.__record[topic]['df_summarized_speaker'] = df_summarized_speaker

    # ==================== RECORD ====================

    def add_record(self, topic, keywords, policyarea):
        self.__record[topic] = {'keywords': keywords, 'policyarea': policyarea}

    def __get_keywords_policyarea(self, topic):
        return self.__record[topic]['keywords'], self.__record[topic]['policyarea']

    # ==================== PIPELINE STEPS ====================

    def filter_speeches(self, topic, years=None):
        keywords, policyarea = self.__get_keywords_policyarea(topic)
        print("Filtering speeches for topic:", topic)
        df = self._filter.filter(self.__speeches_df, keywords, policyarea, years)
        self.__record[topic]['df_filtered'] = df
        suffix = f" in years {years}" if years is not None else ""
        print(f"Number of speeches after filtering for topic '{topic}'{suffix}: {len(df)}")
        return df

    def classify_filtered_sentences(self, topic):
        print("Classifying filtered speeches for topic:", topic)
        df_classified = self._classifier.classify(self.__record[topic]['df_filtered'])
        self.__record[topic]['df_classified'] = df_classified
        print(f"Number of opinionated speeches for {topic}: {len(df_classified)}")
        return df_classified

    def sum_member_speeches(self, speaker_name, topic, model_name='gemma3'):
        self._summarizer.model_name = model_name
        speaker_sentences = self.__record[topic]['df_classified']
        speaker_sentences = speaker_sentences[speaker_sentences['speaker'] == speaker_name]
        return self._summarizer.summarize_speaker(speaker_name, topic, speaker_sentences)

    def summarize_all_sentences(self, topic, model_name='gemma3'):
        self._summarizer.model_name = model_name
        result_df = self._summarizer.summarize_all(self.__record[topic]['df_classified'], topic)
        self.__record[topic]['df_summarized_speaker'] = result_df
        return result_df

    def generate_anchors(self, topic, general=False, temperature=0, model_name='gemma3'):
        print("Generating stance anchors for topic:", topic)
        self._anchor_gen.model_name = model_name
        summaries = self.__record[topic]['df_summarized_speaker']
        text = "\n".join(summaries['summary'].dropna().astype(str))
        return self._anchor_gen.generate(text, topic, general=general, temperature=temperature)

    def compute_embeddings(self, topic, anchors, model_name="Qwen/Qwen3-Embedding-0.6B", debug_mode=False):
        print("Computing embeddings for topic:", topic)
        if self._projector.model_name != model_name:
            self._projector.model_name = model_name
            self._projector._model = None
        summaries = self.__record[topic]['df_summarized_speaker']['summary'].tolist()
        return self._projector.compute_embeddings(summaries, anchors, debug_mode)

    def axis_of_controversy(self, topic, issue, speaker_embeddings, anchor_embeddings):
        return self._projector.axis_of_controversy(
            self.__record[topic]['df_summarized_speaker'], speaker_embeddings, anchor_embeddings, issue
        )

    def cosine_similarity(self, vecA, vecB):
        return self._projector.cosine_similarity(vecA, vecB)

    def compute_umap_embeddings(self, topic, anchors, model_name="Qwen/Qwen3-Embedding-0.6B",
                                n_components=2, n_neighbors=10, min_dist=0.1, metric="cosine"):
        print("Computing UMAP embeddings for topic:", topic)
        if self._projector.model_name != model_name:
            self._projector.model_name = model_name
            self._projector._model = None
        sum_df = self.__record[topic]['df_summarized_speaker'].copy()
        speaker_embeddings, anchor_embeddings = self._projector.compute_embeddings(sum_df['summary'].tolist(), anchors)
        return self._visualizer.compute_umap_embeddings(
            sum_df, speaker_embeddings, anchor_embeddings,
            n_components=n_components, n_neighbors=n_neighbors,
            min_dist=min_dist, metric=metric,
            random_seed=self.random_seed, anchors=anchors
        )

    # ==================== VISUALIZATION ====================

    def plot_axis_of_controversy(self, party_df, issue, anchors=None):
        self._visualizer.plot_axis_of_controversy(party_df, issue, anchors)

    def plot_umap_party_averages(self, umap_data, show_speeches=True, show_party_averages=True,
                                 show_speaker_labels=True, label_fontsize=8):
        self._visualizer.plot_umap_party_averages(
            umap_data, show_speeches, show_party_averages, show_speaker_labels, label_fontsize
        )

    # ==================== EVALUATION ====================

    def generate_gold_standard(self, parties, anchors, years, model="qwen3:8b", debug_mode=False):
        return self._evaluator.generate_gold_standard(parties, anchors, years, model, self.random_seed, debug_mode)

    def evaluate_ordering(self, pred_ordering: list, gold_ordering: list) -> dict:
        return self._evaluator.evaluate_ordering(pred_ordering, gold_ordering)
