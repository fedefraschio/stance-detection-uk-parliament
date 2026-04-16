from setfit import SetFitModel
import requests
import numpy as np
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import random
import ollama
import re
from scipy.stats import spearmanr, kendalltau




class StanceDetector:
    
    
    """
    Detect and analyze political stances in parliamentary speeches.

    Typical pipeline:
    1. Register a topic with keywords and policy areas.
    2. Filter speeches for the topic (optionally by year).
    3. Classify filtered speeches as opinion/non-opinion.
    4. Summarize each speaker stance with an LLM.
    5. Generate contested issue anchors (pro/con positions).
    6. Build embeddings and project parties on controversy axes.
    7. Visualize positions with UMAP and/or axis plots.

    Internal state is stored in a topic-keyed record dictionary, where each
    topic can contain keys such as: keywords, policyarea, df_filtered,
    df_classified, df_summarized_speaker.

    Expected speech DataFrame columns include at least:
    date, agenda, speaker, party, text, policyarea.
    If year-based filtering is used, a year column is also required.
    """
    __record=None
    __speeches_df=None

    def __init__( self, speech,records, cl_model_hf="andreacristiano/stancedetection", random_seed=42):

        """
        Initialize the detector with input data, topic records, and classifier.

        Args:
            speech: DataFrame containing parliamentary speeches.
            records: Dictionary storing topic configuration and intermediate data.
            cl_model_hf: SetFit model id used for opinion classification.
            random_seed: Seed used for reproducible stochastic operations.
        """

        self.__speeches_df = speech
        self.__record= records
        self.model= SetFitModel.from_pretrained(cl_model_hf)
        self.random_seed = random_seed

    # ==================== GETTER METHODS ====================
    def get_records(self):
        """Return the full internal topic record dictionary."""
        return self.__record
    
    def get_speeches(self):
        """Return the source speeches DataFrame."""
        return self.__speeches_df
    
    def get_filtered_speeches(self, topic):
        """Return the filtered speeches DataFrame for a topic."""
        return self.__record[topic]['df_filtered']
    
    def get_classified_speeches(self, topic):
        """Return the classified (opinion-only) DataFrame for a topic."""
        return self.__record[topic]['df_classified']
    

    # ==================== SETTER METHODS ====================

    def set_summarization_for_topic(self, topic, df_summarized_speaker):
        """
        Set or overwrite speaker-level summarization results for a topic.

        Useful for testing and for manually editing summaries before anchor
        generation or downstream visualizations.
        """
        self.__record[topic]['df_summarized_speaker'] = df_summarized_speaker

    # ====================  METHODS ====================

        
    def add_record(self, topic, keywords, policyarea):
        """
        Register a new topic with its keyword and policy-area filters.

        Args:
            topic: Topic name (for example, 'nuclear' or 'immigration').
            keywords: List of keywords to match in speech text.
            policyarea: List of policy-area codes used for filtering.
        """
        self.__record[topic] = {
            'keywords':keywords,
            'policyarea':policyarea,
        }

    def __get_keywords_policyarea(self,topic):
        """
        Retrieve keyword and policy-area filters configured for a topic.

        Args:
            topic: Topic key present in the internal record.

        Returns:
            Tuple[keywords_list, policyarea_list].
        """
        return self.__record[topic]['keywords'], self.__record[topic]['policyarea']
    

    # ==================== MAIN WORKFLOW METHODS ====================

    def filter_speeches(self, topic, years=None):
        """
        Filter speeches by topic, and optionally by year.
        
        This is STEP 1 of the analysis workflow.
        Applies three filters:
        1. Year filter (optional)
        2. Policy area filter
        3. Keyword filter (speeches must contain at least one keyword)
        
        Args:
            topic: The topic to filter for
            years: Optional list of years to include (e.g., [2019, 2020])
            
        Returns:
            DataFrame of filtered speeches
        """
        print("Filtering speeches for topic:", topic)

        keywords, policyarea = self.__get_keywords_policyarea(topic)

        # Start from full dataset
        df = self.__speeches_df.copy()

         # --- Year filtering (optional) ---
        if years is not None:
            df = df[df['year'].isin(years)]

        # --- Policy area filtering ---
        if policyarea:
            df = df[df['policyarea'].isin(policyarea)]

        # --- Keyword filtering ---
        df = df[df['text'].str.contains('|'.join(keywords), case=False, na=False)]

        # Store result
        self.__record[topic]['df_filtered'] = df

        print(
            f"Number of speeches after filtering for topic '{topic}'"
            + (f" in years {years}" if years is not None else "")
            + f": {len(df)}"
        )

        return df
     
    
    def classify_filtered_sentences(self, topic):
        """
        Classify filtered speeches and retain opinionated entries.

        Behavior:
        - Reads df_filtered for the topic.
        - Adds a classification column using the SetFit model.
        - Keeps only rows labeled opinion.
        - Drops speakers with a single opinionated speech.

        Args:
            topic: Topic key with an existing df_filtered DataFrame.

        Returns:
            DataFrame containing opinionated speeches after speaker filtering.
        """
        print("Classifying filtered speeches for topic:", topic)

        #Extract the filtered dataframe
        df_filtered = self.__record[topic]['df_filtered']

        # Get the texts to classify
        texts = df_filtered['text'].tolist()

        # Perform classification
        predictions = self.model.predict(texts)

        # Add predictions as a new column
        df_filtered['classification'] = predictions

        # Keep only opinionated sentences
        df_classified = df_filtered[df_filtered['classification'] == 'opinion']

        # --- Remove speakers with only one datapoint ---
        speaker_counts = df_classified['speaker'].value_counts()
        valid_speakers = speaker_counts[speaker_counts > 1].index
        df_classified = df_classified[df_classified['speaker'].isin(valid_speakers)]

        # Store result
        self.__record[topic]['df_classified'] = df_classified

        print(f'"Number of opinionated speeches for {topic}: {len(df_classified)}')
        return df_classified
    

    def __get_speaker_sentences(self, speaker_name, topic):
        """
        Retrieve all classified rows for a speaker on a topic.

        Args:
            speaker_name: Speaker name to filter on.
            topic: Topic key with an existing df_classified DataFrame.

        Returns:
            DataFrame subset of the classified speeches for that speaker.
        """
        classified_df = self.__record[topic]['df_classified']
        return classified_df[classified_df['speaker'] == speaker_name]
        
    
    def sum_member_speeches(self, speaker_name, topic, model_name='gemma3'):
        
        """
        Summarize one speaker stance on a topic using Ollama.

        The method collects all opinionated speeches for the speaker and asks
        the LLM to produce one concise factual stance sentence.

        Args:
            speaker_name: Speaker to summarize.
            topic: Topic key.
            model_name: Ollama model name.

        Returns:
            Single-row DataFrame with columns: summary, party, speaker.
        """

        # Use verbose topic description for better LLM understanding
        topic_verbose=topic
        if topic=='nuclear':
            topic_verbose = 'the use of nuclear energy as an energy source for the future'
        
        # Get all sentences from this speaker about this topic
        speaker_sentences = self.__get_speaker_sentences(speaker_name, topic)

        # Combine all sentences into one text block using \n as separator
        full_text = "\n".join(speaker_sentences['text'].tolist())

        # Prompt for LLM: summarize the speaker's stance
        context_prompt=f'Based on the politician’s statements in the following parliamentary speeches, infer and summarize this politician’s stance on {topic_verbose}. This summary is intended for those who have little knowledge of British politics. Please summarize in a way that is easy to understand even for those who are not interested in politics. \n{full_text}.'

        # System prompt: ensure factual, concise output
        sys_prompt="You are an expert in UK parliamentary politics. Provide only a single, clear, concise, and purely factual summary sentence of the politician's stance. Do not use introductory phrases like 'Okay,', 'Based on,', or 'From this,'. Do not use colloquial language, filler words, or explanations. Output the summary sentence directly."
        
        # Prepare the request to Ollama API
        payload = {
            "model": model_name,
            "prompt": context_prompt,
            "system": sys_prompt,
            "stream": False,
            "options": {
                "seed": self.random_seed
            }
        }
        try:
            # Call Ollama API (default runs on localhost:11434)
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            # Extract the summary from response
            summary = response.json()['response'].strip()
            
        except requests.exceptions.RequestException as e:
            print(f"\nError generating summary for speaker {speaker_name}: {e}")
            summary = "ERROR: {str(e)}"

        # Return as a single-row DataFrame
        return pd.DataFrame([{
            'summary': summary,
            'party': speaker_sentences['party'].iloc[0],
            'speaker': speaker_name
        }])

    def summarize_all_sentences(self, topic, model_name='gemma3'):
        """
        Summarize stance for every speaker in a topic classified dataset.

        Calls sum_member_speeches for each speaker, concatenates the output,
        stores it as df_summarized_speaker, and removes obvious fallback rows
        that start with Please provide.

        Args:
            topic: Topic key.
            model_name: Ollama model name.

        Returns:
            DataFrame with columns: summary, party, speaker.
        """

        # Get all unique speakers who have opinionated sentences
        classified_df=self.__record[topic]['df_classified']
        speakers = classified_df['speaker'].unique()

        # Summarize each speaker's stance
        summaries = []
        for speaker in speakers:
            summary_df = self.sum_member_speeches(speaker, topic, model_name=model_name)
            summaries.append(summary_df)

        # Combine all summaries
        result_df = pd.concat(summaries, ignore_index=True)
        self.__record[topic]['df_summarized_speaker'] = result_df

        # Remove any rows where LLM failed to generate a proper summary (they usually start with 'Please provide')
        self.__record[topic]['df_summarized_speaker'] = self.__record[topic]['df_summarized_speaker'][~self.__record[topic]['df_summarized_speaker']['summary'].str.startswith('Please provide')]
        print("Summarization completed for topic:", topic)
        return self.__record[topic]['df_summarized_speaker']
        
    
    def generate_anchors(self, topic, general= False, temperature=0, model_name='gemma3'):

        """
        Generate contested issue anchors (pro/con pairs) from summaries.

        Uses an LLM to infer debated issues from topic summaries and returns
        structured anchors with topic, pro, and con fields.

        Args:
            topic: Topic key.
            general: If True, generate one main issue; else generate multiple.
            temperature: Decoding temperature for Ollama.
            model_name: Ollama model name.

        Returns:
            If general is False: list[dict] with keys topic, pro, con.
            If general is True: a single dict with keys topic, pro, con.
        """
        print("Generating stance anchors for topic:", topic)
        summarizations = self.__record[topic]['df_summarized_speaker']

        # Concatenate all summaries into one text
        text = "\n".join(summarizations['summary'].dropna().astype(str))

        system_message = """
        You are an expert in political discourse analysis and debate extraction.

        Your task is to identify contested policy issues from collections of political statements and reconstruct structured debate pairs.

        Always follow the requested output format exactly.
        Only use information grounded in the provided text.
        Do not invent arguments or add commentary.
        Be concise, neutral, and analytical.
        """

        prompt = f"""
        You are given a dataset of summaries of political statements about the topic: "{topic}".

        Each summary represents the opinion of one politician.

        Your task is to reconstruct the main contested policy issues discussed in the dataset and express them as structured debate pairs.

        Method:
        1. Read all summaries.
        2. Identify recurring policy questions or normative disputes.
        3. Group similar viewpoints.
        4. Detect two opposing positions on each issue.
        5. Formulate a neutral issue statement and one "For" and one "Against" argument.

        Constraints:
        - Use ONLY positions supported by the summaries.
        - Do NOT invent arguments.
        - Issues must be clear political questions.
        - "For" and "Against" must represent genuinely opposing positions.

        Writing rules:
        - Issue: 10–18 words
        - For / Against: 20–35 words
        - Neutral analytical tone.
        - Do not mention politicians or parties.
        - Avoid duplicate issues.

        Output format (strict):

        Issue: <Neutral statement of the contested issue>
        For: <Argument supporting the issue>
        Against: <Argument opposing the issue>

        Issue: <Neutral statement of the contested issue>
        For: <Argument supporting the issue>
        Against: <Argument opposing the issue>

        Generate between 5 and 8 issues.

        Dataset:
        {text}
        """

        if general:

            prompt = f"""
                You are given a dataset of summaries of political statements about the topic: "{topic}".
                
                Each summary represents the opinion of one politician.

                Your task is to identify the single most important contested policy issue from the dataset and express it as a structured debate pair.

                Method:
                1. Read all summaries.
                2. Identify the most central policy question or normative dispute.
                3. Group similar viewpoints.
                4. Detect two opposing positions on this issue.
                5. Formulate a neutral issue statement and one "For" and one "Against" argument.

                Constraints:
                - Use ONLY positions supported by the summaries.
                - Do NOT invent arguments.
                - The issue must be a clear political question.
                - "For" and "Against" must represent genuinely opposing positions.
                
                Writing rules:
                - Issue: 10–18 words
                - For / Against: 20–35 words
                - Neutral analytical tone.
                - Do not mention politicians or parties.
                
                Output format (strict):

                Issue: <Neutral statement of the main contested issue>
                For: <Position emphasising one set of causes, actors, and solutions>
                Against: <Position emphasising a completely different set of causes, actors, and solutions>

                Generate EXACTLY ONE issue.

                Dataset:
                {text}
                """
            

        # Request structured JSON output from Ollama
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt},
                      {"role": "system", "content": system_message}],
            # removed 'format' as it can slow down response and is not always necessary if the prompt is clear enough
            # temperature set to 0 might slow down decoding
            options={"temperature": temperature, 'seed': self.random_seed}
        )

        content = response["message"]["content"]

        # create a regex pattern to extract the issues and their pro/con positions
        pattern = r"Issue:\s*(.*?)\nFor:\s*(.*?)\nAgainst:\s*(.*?)(?=\nIssue:|\Z)"
        matches = re.findall(pattern, content, re.DOTALL)

        if not matches:
            print("Warning: no anchors extracted. Raw response:\n", content)
            return []


        anchors = [
            {"topic": m[0].strip(), "pro": m[1].strip(), "con": m[2].strip()}
            for m in matches
            ]
        
        if general:
            return anchors[0]  # Return the single general anchor as a dict

        return anchors

        # return json.loads(content)
        #return content # DEBUG

    # cosine similarity to compare anchors (useful for debugging) 
    def cosine_similarity(self, vecA, vecB):
        """
        Compute cosine similarity between two vectors.

        Args:
            vecA: First vector.
            vecB: Second vector.

        Returns:
            Float cosine similarity score, or 0.0 when a norm is zero.
        """
        dot_product = np.dot(vecA, vecB)
        normA = np.linalg.norm(vecA)
        normB = np.linalg.norm(vecB)
        if normA == 0 or normB == 0:
            return 0.0
        else:
            return dot_product / (normA * normB)
    

    def compute_embeddings(self, topic, anchors, model_name="Qwen/Qwen3-Embedding-0.6B", debug_mode=False):
        """
        Compute embeddings for speaker summaries and a single anchor pair.

        Args:
            topic: Topic key.
            anchors: Dict containing pro and con strings.
            model_name: SentenceTransformer model name.
            debug_mode: If True, print additional debugging information.
        Returns:
            Tuple (speaker_embeddings, anchor_embeddings).
        """
        print("Computing embeddings for topic:", topic)
        
        # Get speaker summaries
        sum_df = self.__record[topic]['df_summarized_speaker'].copy()
        summaries = sum_df["summary"].tolist()

        # Extract anchor texts (pro and con positions)
        anchor_texts = [anchors['pro'], anchors['con']]
        
        all_texts = summaries + anchor_texts

        # DEBUG
        if debug_mode:
            print(all_texts)

        # Load embedding model
        model = SentenceTransformer(model_name)
        
        # Embed both speaker summaries AND anchors together
        # (This ensures they're in the same embedding space)
        embeddings = model.encode(all_texts, show_progress_bar=True)

        # Split back into speeches and anchors
        speaker_embeddings = embeddings[:len(summaries)]
        anchor_embeddings = embeddings[len(summaries):]

        return speaker_embeddings, anchor_embeddings
    
    
    # TODO: for each subtopic, create axis of controversy and projecting party averages onto it 

    def axis_of_controversy(self, topic, issue, speaker_embeddings, anchor_embeddings):

        """
        Project party positions onto a controversy axis defined by anchors.

        The axis is built from con to pro anchor embeddings, centered at their
        midpoint. Party centroids are then projected onto this axis.

        Args:
            topic: Topic key.
            issue: Issue label associated with this anchor pair.
            speaker_embeddings: Array of speaker summary embeddings.
            anchor_embeddings: Array with pro and con anchor embeddings.

        Returns:
            DataFrame with columns: issue, party, controversy_score.
        """

        # Compute party centroids in the original embedding space
        sum_df = self.__record[topic]['df_summarized_speaker'].copy().reset_index(drop=True)
        sum_df['embedding'] = list(speaker_embeddings)  # Add embeddings to DataFrame

        party_centroids = (
            sum_df
            .groupby("party")['embedding']
            .apply(lambda x: np.mean(list(x), axis=0))
            .reset_index()
            .rename(columns={'embedding': 'centroid'})
        )

        # Axis: CON → PRO, centred at midpoint
        pro_emb, con_emb = anchor_embeddings[0], anchor_embeddings[1]
        midpoint = (pro_emb + con_emb) / 2
        axis = (pro_emb - con_emb)
        axis = axis / np.linalg.norm(axis)

        centroids_matrix = np.stack(party_centroids['centroid'].values)
        party_centroids['controversy_score'] = (centroids_matrix - midpoint) @ axis

        # Create a DataFrame with parties and their controversy scores
        party_df = party_centroids[['party']].copy()
        party_df['controversy_score'] = party_centroids['controversy_score']
        party_df['issue'] = issue

        return party_df


    def plot_axis_of_controversy(self, party_df, issue, anchors=None):
        """
        Visualize the axis of controversy with party positions.

        Plots parties on a horizontal line by controversy_score, adds PRO/CON
        direction markers, and optionally shows anchor text.

        Args:
            party_df: DataFrame with columns issue, party, controversy_score.
            issue: Issue label for the plot title.
            anchors: Optional dict with pro and con strings.
        """
        show_anchors = anchors is not None
        fig, ax = plt.subplots(figsize=(14, 5 if show_anchors else 3.5))

        df = party_df.sort_values('controversy_score').reset_index(drop=True)

        # Assign distinct colors per party
        unique_parties = sorted(df['party'].unique())
        cmap = plt.get_cmap('tab10')
        party_to_color = {p: cmap(i % 10) for i, p in enumerate(unique_parties)}
        colors = [party_to_color[p] for p in df['party']]

        x_min, x_max = df['controversy_score'].min(), df['controversy_score'].max()
        pad = max((x_max - x_min) * 0.25, 0.05)

        # Main axis line
        ax.axhline(0, color='#444444', linewidth=2, zorder=2)

        # Party points
        ax.scatter(
            df['controversy_score'], np.zeros(len(df)),
            c=colors, s=150, zorder=5, linewidths=0.8, edgecolors='white'
        )

        # Stagger labels alternately above/below to reduce overlap
        stagger_heights = [0.10, -0.14, 0.19, -0.24]
        for i, (_, row) in enumerate(df.iterrows()):
            yo = stagger_heights[i % len(stagger_heights)]
            c = party_to_color[row['party']]
            ax.annotate(
                row['party'],
                xy=(row['controversy_score'], 0.01 if yo > 0 else -0.01),
                xytext=(row['controversy_score'], yo),
                ha='center', va='center',
                fontsize=8.5, color=c, fontweight='bold',
                arrowprops=dict(arrowstyle='-', color=c, lw=0.8, alpha=0.5),
                path_effects=[pe.withStroke(linewidth=2.5, foreground='white')]
            )

        # Neutral reference line
        ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
        ax.text(0, -0.34, 'neutral', ha='center', fontsize=7, color='gray', style='italic')

        # PRO / CON end labels
        ax.text(
            x_max + pad * 0.75, 0, 'PRO ▶',
            ha='left', fontsize=10, color='steelblue', fontweight='bold', va='center'
        )
        ax.text(
            x_min - pad * 0.75, 0, '◀ CON',
            ha='right', fontsize=10, color='firebrick', fontweight='bold', va='center'
        )

        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(-0.40, 0.32)
        ax.set_title(f"Axis of Controversy: {issue}", fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel("Controversy Score", fontsize=9, color='#666666')
        ax.set_yticks([])
        for spine in ['left', 'right', 'top']:
            ax.spines[spine].set_visible(False)
        ax.grid(axis='x', alpha=0.2)

        # Anchor descriptions at the bottom of the figure
        if show_anchors:
            plt.subplots_adjust(bottom=0.38)
            fig.text(
                0.05, 0.24,
                f"PRO:  {anchors['pro']}",
                ha='left', va='bottom', fontsize=8,
                color='steelblue', style='italic'
            )
            fig.text(
                0.05, 0.08,
                f"CON:  {anchors['con']}",
                ha='left', va='bottom', fontsize=8,
                color='firebrick', style='italic'
            )

        plt.show()

    
    def compute_umap_embeddings(self,
                           topic,
                           anchors,
                           model_name="Qwen/Qwen3-Embedding-0.6B",
                           n_components=2,
                           n_neighbors=10,
                           min_dist=0.1,
                           metric="cosine",):
        """
        Compute UMAP coordinates for speaker summaries and anchor points.

        Speaker and anchor texts are embedded in the same space and reduced
        jointly with UMAP.

        Args:
            topic: Topic key.
            anchors: Dict containing topic, pro, con.
            model_name: SentenceTransformer model name.
            n_components: Number of output dimensions.
            n_neighbors: UMAP neighborhood size.
            min_dist: UMAP minimum distance.
            metric: UMAP distance metric.

        Returns:
            Dict with keys:
            - df: speaker DataFrame with umap_x and umap_y.
            - reduced_embeddings: full reduced matrix (speakers + anchors).
            - reduced_anchors: reduced coordinates for pro/con anchors.
            - anchors: input anchor dictionary.
        """

        print("Computing UMAP embeddings for topic:", topic)
        
        # Get speaker summaries
        sum_df = self.__record[topic]['df_summarized_speaker'].copy()
        summaries = sum_df["summary"].tolist()

        # Generate embeddings using the compute_embeddings method
        speaker_embeddings, anchor_embeddings = self.compute_embeddings(topic, anchors, model_name)
        
        # Combine embeddings for UMAP reduction
        embeddings = np.vstack([speaker_embeddings, anchor_embeddings])

        # Reduce embeddings to 2D using UMAP
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=self.random_seed
        )

        reduced_embeddings = reducer.fit_transform(embeddings)
        
        # Split back into speeches and anchors
        reduced_speeches = reduced_embeddings[:len(summaries)]
        reduced_anchors = reduced_embeddings[len(summaries):]

        # Add UMAP coordinates to the speaker DataFrame
        sum_df["umap_x"] = reduced_speeches[:, 0]
        sum_df["umap_y"] = reduced_speeches[:, 1]

        return {
            'df': sum_df,
            'reduced_embeddings': reduced_embeddings,
            'reduced_anchors': reduced_anchors,
            'anchors': anchors
        }


    def plot_umap_party_averages(self,
                             umap_data,                            
                             show_speeches=True,
                             show_party_averages=True,
                             show_speaker_labels=True,
                             label_fontsize=8):
        """
        Plot UMAP speaker points, party centroids, and stance anchors.

        Args:
            umap_data: Output dictionary from compute_umap_embeddings.
            show_speeches: Whether to draw individual speaker points.
            show_party_averages: Whether to draw party centroid markers.
            show_speaker_labels: Whether to annotate each speaker point.
            label_fontsize: Font size for speaker labels.
        """
        # Extract data from the UMAP results
        sum_df = umap_data['df']
        reduced_anchors = umap_data['reduced_anchors']
        anchors = umap_data['anchors']
        anchor_labels = ['pro', 'con']

        # Calculate party centroids (average UMAP position per party)
        party_centroids = (
            sum_df
            .groupby("party")
            .agg({
                "umap_x": "mean",
                "umap_y": "mean",
                "party": "size"  # Count speeches per party
            })
            .rename(columns={"party": "speech_count"})
            .reset_index()
        )

        # Assign colors to parties
        unique_parties = sorted(sum_df["party"].unique())
        party_to_color = {party: i for i, party in enumerate(unique_parties)}
        cmap = plt.get_cmap("tab10")

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 12))

        # Plot each party's data
        for party in unique_parties:
            color = cmap(party_to_color[party])

            # Plot individual speakers if requested
            if show_speeches:
                mask = sum_df["party"] == party
                party_data = sum_df.loc[mask]
                
                ax.scatter(
                    party_data["umap_x"],
                    party_data["umap_y"],
                    alpha=0.85,
                    color=color,
                    label=party
                )
                
                # Add speaker labels to each datapoint
                if show_speaker_labels:
                    for _, row in party_data.iterrows():
                        ax.annotate(
                            row['speaker'],
                            xy=(row['umap_x'], row['umap_y']),
                            xytext=(3, 3),  # Small offset from the point
                            textcoords='offset points',
                            fontsize=label_fontsize,
                            alpha=0.7,
                            color=color
                        )

            # Plot party centroid if requested
            if show_party_averages:
                centroid = party_centroids[party_centroids["party"] == party]
                # Marker size scales with number of speeches
                marker_size = 100 + centroid["speech_count"].values[0] * 80
                ax.scatter(
                    centroid["umap_x"],
                    centroid["umap_y"],
                    s=marker_size,
                    color=color,
                    edgecolor="black",
                    linewidth=1.2,
                    zorder=5  # Draw on top of individual points
                )

        # Plot stance anchors (pro/con positions)
        if reduced_anchors is not None:
            ax.scatter(
                reduced_anchors[:, 0],
                reduced_anchors[:, 1],
                s=200,
                color='black',
                marker='D',  # Diamond shape
                edgecolor='white',
                linewidth=1.5,
                label='Stance Anchors',
                zorder=10  # Draw on top of everything
            )

            # Add labels to anchors with arrows
            for x, y, label in zip(
                reduced_anchors[:, 0],
                reduced_anchors[:, 1],
                anchor_labels
            ):
                ax.annotate(
                    label.upper(),
                    xy=(x, y),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(
                        boxstyle='round,pad=0.3',
                        facecolor='yellow',
                        alpha=0.7
                    ),
                    arrowprops=dict(
                        arrowstyle='->',
                        connectionstyle='arc3,rad=0',
                        lw=1
                    )
                )

        # Add titles and labels
        ax.set_title(f"UMAP Party Averages – {anchors['topic']}")
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")

        # Add legend
        ax.legend(
            title="Parties & Anchors",
            bbox_to_anchor=(1.05, 1),
            loc="upper left"
        )

        # Add anchor descriptions at the bottom of the figure
        anchor_text = (
            f"*PRO*: {anchors['pro']}\n\n"
            f"*CON*: {anchors['con']}"
        )

        fig.text(
            0.02,  # x position (left side)
            0.02,  # y position (bottom)
            anchor_text,
            ha="left",
            va="bottom",
            fontsize=14,
            wrap=True
        )

        # Make space at the bottom for anchor text
        plt.subplots_adjust(bottom=0.30)
        plt.show()

    def generate_gold_standard(self, parties, anchors, years, model="qwen3:8b", debug_mode=False): #llama3.2:latest # qwen3:0.6b
        """
        Generate an LLM-based reference ordering of parties on the issue axis.

        Args:
            parties: Party names to rank from CON to PRO.
            anchors: Dict with topic, pro, and con descriptions.
            years: Year interval used as historical context in the prompt.
            model: Ollama model name.
            debug_mode: If True, print debug information.

        Returns:
            List of party names ordered from most CON-aligned to most PRO-aligned.
        """

        # shuffle a copy of the parties list to avoid any bias from the original order
        parties = parties.copy()
        random.shuffle(parties)

        parties_json = json.dumps(parties)

        example = json.dumps(parties[::-1], ensure_ascii=False)  # usa i nomi reali

        prompt = f"""You are a political scientist specialising in UK parliamentary politics ({years[0]}–{years[-1]}).

            Rank these parties from most aligned with POSITION_A to most aligned with POSITION_B:

            ISSUE: {anchors['topic']}
            POSITION_A: {anchors['con']}
            POSITION_B: {anchors['pro']}

            Parties to rank (use these exact strings):
            {parties_json}

            Return ONLY a JSON array ordered from most POSITION_A-aligned to most POSITION_B-aligned.
            Example format (NOT the correct answer):
            {example}

            OUTPUT:"""
        
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            think=True, # if model supports thinking
            options={
                "temperature": 0.25,
                "seed": self.random_seed,
            }
        )

        content = response["message"]["content"].strip()

        # Extract JSON array safely
        match = re.search(r"\[[^\]]+\]", content)
        if not match:
            raise ValueError(f"No JSON array found in response:\n{content}")

        stripped_array = match.group().strip("\n")

        if stripped_array.count("[") != 1 or stripped_array.count("]") != 1:
            raise ValueError(f"Invalid JSON array format:\n{stripped_array}")
        
        # DEBUG
        if debug_mode:  
            print("Parties to rank:", parties)
            print("Raw LLM response:", content)
        
        ranked_parties = json.loads(stripped_array)

        # Validation
        if set(ranked_parties) != set(parties):
            raise ValueError(
                f"Party mismatch.\nExpected: {parties}\nGot: {ranked_parties}\nRaw: {content}"
            )

        if len(ranked_parties) != len(parties):
            raise ValueError(
                f"Duplicate or missing parties.\nExpected {len(parties)} got {len(ranked_parties)}"
            )

        return ranked_parties
    
    def evaluate_ordering(self, pred_ordering: list, gold_ordering: list) -> dict:
        """
        Evaluate predicted ordering against a reference ordering.

        Computes Spearman rho, Kendall tau, and LCS ratio on parties that are
        present in both input lists.

        Args:
            pred_ordering: Party names ordered CON to PRO from model projection.
            gold_ordering: Party names ordered CON to PRO from reference ranking.

        Returns:
            Dict with keys:
            spearman_rho, spearman_p, kendall_tau, kendall_p, lcs_ratio, n_parties.
        """

        # Align on common parties only
        common = [p for p in pred_ordering if p in gold_ordering]
        n = len(common)

        if n < 2:
            print(f"Warning: only {n} party/parties in common — metrics not computable.")
            return {
                "spearman_rho": np.nan, "spearman_p": np.nan,
                "kendall_tau":  np.nan, "kendall_p":  np.nan,
                "lcs_ratio":    np.nan, "n_parties":  n
            }

        # Rank positions (0-indexed, lower = more CON)
        pred_ranks = [pred_ordering.index(p) for p in common]
        gold_ranks = [gold_ordering.index(p) for p in common]

        rho, p_rho = spearmanr(pred_ranks, gold_ranks)
        tau, p_tau = kendalltau(pred_ranks, gold_ranks)

        # LCS
        gold_common = [p for p in gold_ordering if p in common]
        m, n_gold = len(common), len(gold_common)
        dp = [[0] * (n_gold + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n_gold + 1):
                if common[i-1] == gold_common[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        lcs = dp[m][n_gold] / max(m, n_gold)

        return {
            "spearman_rho": round(rho,  4),
            "spearman_p":   round(p_rho, 4),
            "kendall_tau":  round(tau,  4),
            "kendall_p":    round(p_tau, 4),
            "lcs_ratio":    round(lcs,  4),
            "n_parties":    n
        }