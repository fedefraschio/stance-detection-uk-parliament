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
    A class for detecting and analyzing political stances in parliamentary speeches.
    
    WORKFLOW:
    1. Load parliamentary speeches dataset
    2. Filter speeches by topic (using keywords and policy areas)
    3. Classify sentences as opinionated or non-opinionated
    4. Keep only opinionated sentences for analysis
    5. Summarize each speaker's stance on the topic
    6. Generate stance anchors (opposing viewpoints)
    7. Create UMAP visualizations to map political positions
    
    DATA STRUCTURE:
    The class stores topics in a record dictionary:
    {
        "nuclear": {
            'keywords': [...],           # Words to search for
            'policyarea': [7, 8],        # Policy area codes
            'df_filtered': ...,          # Speeches matching topic
            'df_classified': ...,        # Opinionated sentences only
            'df_summarized_speaker': ... # Summarized stances per speaker
        }
    }
    
    INPUT DATASET COLUMNS:
    [ 'date', 'agenda', 'speaker', 'party', 'text', 'policyarea']
    """
    __record=None
    __speeches_df=None

    def __init__( self, speech,records, cl_model_hf="andreacristiano/stancedetection", random_seed=42):

        """
        Initialize the StanceDetector.
        
        Args:
            speech: DataFrame containing all parliamentary speeches
            records: Dictionary to store topic-specific data
            cl_model_hf: HuggingFace model for opinion classification
            random_seed: Seed for reproducible random operations
        """

        self.__speeches_df = speech
        self.__record= records
        self.model= SetFitModel.from_pretrained(cl_model_hf)
        self.random_seed = random_seed

    # ==================== GETTER METHODS ====================
    def get_records(self):
        return self.__record
    
    def get_speeches(self):
        return self.__speeches_df
    
    def get_filtered_speeches(self, topic):
        return self.__record[topic]['df_filtered']
    
    def get_classified_speeches(self, topic):
        return self.__record[topic]['df_classified']
    

    # ==================== SETTER METHODS ====================

    # It will be used to set the summarization results, mainly for testing purposes, but it can also be useful if we want to modify the summarization results before using them for anchor generation or visualization.
    def set_summarization_for_topic(self, topic, df_summarized_speaker):
        self.__record[topic]['df_summarized_speaker'] = df_summarized_speaker

    # ====================  METHODS ====================

        
    def add_record(self, topic, keywords, policyarea):
        """
        Add a new topic to track.
        
        Args:
            topic: Name of the topic (e.g., 'nuclear', 'immigration')
            keywords: List of keywords to search for in speeches
            policyarea: List of policy area codes to filter by
        """
        self.__record[topic] = {
            'keywords':keywords,
            'policyarea':policyarea,
        }

    def __get_keywords_policyarea(self,topic):
        """
        Internal method to retrieve keywords and policy areas for a topic.
        
        Returns:
            Tuple of (keywords_list, policyarea_list)
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
        IN: filtered dataframe about a topic
        OUT: classified dataframe containing opinionated sentences only
            + a new column 'classification' in filtered dataframe
            + speakers with only one datapoint removed
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
        Retrieve all sentences spoken by a specific speaker on a given topic.
        
        """
        classified_df = self.__record[topic]['df_classified']
        return classified_df[classified_df['speaker'] == speaker_name]
        
    
    def sum_member_speeches(self, speaker_name, topic, model_name='gemma3'):
        
        """
        Summarize one speaker's stance on a topic using an LLM.
        
        This is STEP 3a (per speaker) of the analysis workflow.
        - Collects all opinionated sentences from the speaker
        - Sends them to an LLM for summarization
        - Returns a one-sentence summary of their stance
        
        Args:
            speaker_name: Name of the speaker to summarize
            topic: The topic to summarize about
            model_name: Ollama model to use for summarization
            
        Returns:
            DataFrame with columns: summary, party, speaker
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
        Summarize stances for all speakers on a topic.
        
        This is STEP 3 of the analysis workflow.
        - Calls sum_member_speeches() for each speaker
        - Combines all summaries into one DataFrame
        - Removes any failed summaries
        
        Args:
            topic: The topic to summarize
            model_name: Ollama model to use
            
        Returns:
            DataFrame with columns: summary, party, speaker
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
    

    
    
    def generate_anchors(self, topic, model_name='gemma3'):

        """
        Generate stance anchors (opposing viewpoints) for a topic.
        
        This is STEP 4 of the analysis workflow.
        - Takes all speaker summaries
        - Uses LLM to identify key issues and opposing views
        - Returns structured anchors (pro/con positions)
        
        Args:
            topic: The topic to generate anchors for
            model_name: Ollama model to use
            
        Returns:
            List of dictionaries with keys: topic, pro, con
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

        # Request structured JSON output from Ollama
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt},
                      {"role": "system", "content": system_message}],
            # removed 'format' as it can slow down response and is not always necessary if the prompt is clear enough
            # temperature set to 0 might slow down decoding
            options={"temperature": 0, 'seed': self.random_seed}
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
        
        return anchors

        # return json.loads(content)
        #return content # DEBUG

    def compute_embeddings(self, topic, anchors, model_name="Qwen/Qwen3-Embedding-0.6B"):
        """
        Compute embeddings for speaker summaries and stance anchors.
        
        This is STEP 5a of the analysis workflow.
        - Converts speaker summaries to numerical embeddings
        - Includes stance anchors (pro/con) in the embedding space
        
        Args:
            topic: The topic to compute embeddings for
            anchors: Dictionary with 'pro' and 'con' stance descriptions
            model_name: SentenceTransformer model for embeddings
        Returns:
            Tuple of (speaker_embeddings, anchor_embeddings, top_politicians)
        """
        print("Computing embeddings for topic:", topic)
        
        # Get speaker summaries
        sum_df = self.__record[topic]['df_summarized_speaker'].copy()
        summaries = sum_df["summary"].tolist()

        # pick politicians that have the most speeches (max 10)
        politician_counts = sum_df['speaker'].value_counts()
        top_politicians = politician_counts.head(10).index.tolist()
        sum_df = sum_df[sum_df['speaker'].isin(top_politicians)]

        # Persist the exact speakers used for embeddings to keep downstream alignment
        self.__record[topic]['df_embeddings_speaker'] = sum_df[['speaker', 'party']].copy()

        # Considering summaries for top politicians
        summaries = sum_df["summary"].tolist()

        # Extract anchor texts (pro and con positions)
        anchor_texts = [anchors['pro'], anchors['con']]
        
        all_texts = summaries + anchor_texts

        # DEBUG
        print(all_texts)

        # Load embedding model
        model = SentenceTransformer(model_name)
        
        # Embed both speaker summaries AND anchors together
        # (This ensures they're in the same embedding space)
        embeddings = model.encode(all_texts, show_progress_bar=True)

        # Split back into speeches and anchors
        speaker_embeddings = embeddings[:len(summaries)]
        anchor_embeddings = embeddings[len(summaries):]

        return speaker_embeddings, anchor_embeddings, top_politicians
    
    

    def axis_of_controversy(self, topic, issue, speaker_embeddings, anchor_embeddings):

        """
        Create an axis of controversy based on the stance anchors and project speaker positions onto it.
        
        This is STEP 5b of the analysis workflow.
        - Uses the pro and con anchor embeddings to define a controversy axis
        - Copmute party aberages in the original embedding space
        - Project party averages onto the controversy axis to get a controversy score
        - Higher scores indicate alignment with the "pro" position, lower scores with "con"
        
        Args:
            topic: The topic to analyze
            issue: The specific issue (from the generated anchors) to analyze
            speaker_embeddings: Embeddings for each speaker's summary
            anchor_embeddings: Embeddings for the pro and con anchors
            
        Returns:
            DataFrame with columns: issue, party, controversy_score
        """

        # Axis: CON → PRO, centred at midpoint
        pro_emb, con_emb = anchor_embeddings[0], anchor_embeddings[1]
        midpoint = (pro_emb + con_emb) / 2
        axis = (pro_emb - con_emb)
        axis = axis / np.linalg.norm(axis)


        # project each speaker embedding onto the axis
        controversy_scores = []
        for emb in speaker_embeddings:
            score = np.dot(emb - midpoint, axis)
            controversy_scores.append(score)
        # Create DataFrame with controversy scores
        if 'df_embeddings_speaker' in self.__record[topic]:
            party_df = self.__record[topic]['df_embeddings_speaker'].copy()
        else:
            party_df = self.__record[topic]['df_summarized_speaker'][['speaker', 'party']].copy()
        party_df['issue'] = issue
        party_df['controversy_score'] = controversy_scores

        return party_df


    def plot_axis_of_controversy(self, speaker_df, anchors):
        """
        Visualize the controversy axis at the individual speaker level.

        Each party gets its own row; individual speakers are shown as dots,
        with the party centroid overlaid as a larger marker.
        Styled consistently with plot_axis_of_controversy (party-level plot).

        Args:
            speaker_df: DataFrame with columns: speaker, party, controversy_score
            anchors: Dict with keys 'topic', 'pro', 'con'
        """
        import matplotlib.patheffects as pe

        # Sort parties by their mean score (left = CON, right = PRO)
        party_order = (
            speaker_df.groupby('party')['controversy_score']
            .mean()
            .sort_values()
            .index.tolist()
        )
        n_parties = len(party_order)

        show_anchors = anchors is not None
        fig_height = max(4, n_parties * 0.7) + (2.5 if show_anchors else 0)
        fig, ax = plt.subplots(figsize=(14, fig_height))

        # Color palette consistent with party-level plot
        cmap = plt.get_cmap('tab10')
        party_to_color = {p: cmap(i % 10) for i, p in enumerate(party_order)}

        x_min = speaker_df['controversy_score'].min()
        x_max = speaker_df['controversy_score'].max()
        pad = max((x_max - x_min) * 0.20, 0.05)

        for row_idx, party in enumerate(party_order):
            subset = speaker_df[speaker_df['party'] == party]
            color = party_to_color[party]

            # Horizontal guide line for this party
            ax.axhline(row_idx, color='#dddddd', linewidth=0.8, zorder=1)

            # Jitter individual speakers vertically within their row
            jitter = np.random.uniform(-0.18, 0.18, size=len(subset))
            ax.scatter(
                subset['controversy_score'],
                row_idx + jitter,
                color=color,
                alpha=0.55,
                s=28,
                edgecolor='white',
                linewidth=0.4,
                zorder=3,
            )

            # Party centroid — larger, solid, with white halo
            centroid = subset['controversy_score'].mean()
            ax.scatter(
                centroid, row_idx,
                color=color,
                s=160,
                zorder=5,
                edgecolor='white',
                linewidth=1.2,
                marker='D',  # diamond to distinguish from speakers
            )

            # Party label on the left
            ax.text(
                x_min - pad * 0.15, row_idx,
                party,
                ha='right', va='center',
                fontsize=8.5, color=color, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=2, foreground='white')]
            )

        # Neutral reference line
        ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.4, zorder=1)
        ax.text(0, -0.65, 'neutral', ha='center', fontsize=7, color='gray', style='italic')

        # PRO / CON end labels
        ax.text(
            x_max + pad * 0.6, (n_parties - 1) / 2, 'PRO ▶',
            ha='left', fontsize=10, color='steelblue', fontweight='bold', va='center'
        )
        ax.text(
            x_min - pad * 0.6, (n_parties - 1) / 2, '◀ CON',
            ha='right', fontsize=10, color='firebrick', fontweight='bold', va='center'
        )

        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(-0.6, n_parties - 0.4)
        ax.set_title(f"Axis of Controversy – Speakers: {anchors['topic']}", fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel("Controversy Score (CON → PRO)", fontsize=9, color='#666666')
        ax.set_yticks([])
        for spine in ['left', 'right', 'top']:
            ax.spines[spine].set_visible(False)
        ax.grid(axis='x', alpha=0.2)

        # Anchor descriptions below the plot
        if show_anchors:
            plt.subplots_adjust(bottom=0.28)
            fig.text(0.05, 0.18, f"PRO:  {anchors['pro']}",
                    ha='left', va='bottom', fontsize=8, color='steelblue', style='italic')
            fig.text(0.05, 0.06, f"CON:  {anchors['con']}",
                    ha='left', va='bottom', fontsize=8, color='firebrick', style='italic')

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
        Create UMAP embeddings to visualize speaker positions on a topic.
        
        This is STEP 5 of the analysis workflow.
        - Converts speaker summaries to numerical embeddings
        - Includes stance anchors (pro/con) in the embedding space
        - Reduces to 2D using UMAP for visualization
        
        Args:
            topic: The topic to visualize
            anchors: Dictionary with 'pro' and 'con' stance descriptions
            model_name: SentenceTransformer model for embeddings
            n_components: Number of UMAP dimensions (usually 2 for plotting)
            n_neighbors: UMAP parameter (affects local vs global structure)
            min_dist: UMAP parameter (affects point spacing)
            metric: Distance metric for UMAP
            
        Returns:
            Dictionary containing:
            - 'df': DataFrame with UMAP coordinates (umap_x, umap_y)
            - 'reduced_embeddings': Full UMAP array (speakers + anchors)
            - 'reduced_anchors': UMAP coordinates for pro/con anchors
            - 'anchors': The input anchor dictionary
        """

        print("Computing UMAP embeddings for topic:", topic)
        
        # Get speaker summaries
        sum_df = self.__record[topic]['df_summarized_speaker'].copy()
        summaries = sum_df["summary"].tolist()

        # Extract anchor texts (pro and con positions)
        anchor_texts = [anchors['pro'], anchors['con']]
        
        # Load embedding model
        model = SentenceTransformer(model_name)
        
        # Embed both speaker summaries AND anchors together
        # (This ensures they're in the same embedding space)
        all_texts = summaries + anchor_texts
        embeddings = model.encode(all_texts, show_progress_bar=True)
        

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
        Visualize speaker positions and party centroids on a 2D UMAP plot.
        
        This is STEP 6 (visualization) of the analysis workflow.
        - Plots individual speakers colored by party
        - Shows party average positions (centroids)
        - Displays stance anchors (pro/con) as black diamonds
        - Includes anchor descriptions below the plot
        
        Args:
            umap_data: Output from compute_umap_embeddings()
            show_speeches: Whether to show individual speaker points
            show_party_averages: Whether to show party centroid markers
            show_speaker_labels: Whether to show speaker names on datapoints
            label_fontsize: Font size for speaker labels
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

    # generate a gold standard ordering of parties on the controversy axis based on the generated anchors
    def generate_gold_standard(self, parties, anchors, years, model="qwen3:0.6b"):
        """
        Generate a gold standard ordering of parties on the controversy axis.

        Args:
            parties: list[str]
            anchors: dict with 'topic', 'pro', 'con'
            years: list[int]
            model: ollama model

        Returns:
            list[str] ordered from most CON-aligned to most PRO-aligned
        """

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
            think=True,
            options={
                "temperature": 0,
                "seed": self.random_seed
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
        Evaluate a predicted party ordering against a gold standard.

        Computes Spearman's rho, Kendall's tau, and LCS ratio.
        Only parties present in BOTH orderings are evaluated.

        Args:
            pred_ordering: list of party names ordered CON → PRO (from axis_of_controversy)
            gold_ordering: list of party names ordered CON → PRO (from generate_gold_standard)

        Returns:
            dict with keys: spearman_rho, spearman_p, kendall_tau, kendall_p, lcs_ratio, n_parties
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