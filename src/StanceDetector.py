from setfit import SetFitModel 
import requests
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import umap
import matplotlib.pyplot as plt
import random
import ollama



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
        Classify sentences as opinionated or non-opinionated.
        
        This is STEP 2 of the analysis workflow.
        - Uses few shot model to classify each sentence
        - Keeps only opinionated sentences
        - Removes speakers with only one sentence (insufficient data) to reduce noise
        
        Args:
            topic: The topic to classify
            
        Returns:
            DataFrame containing only opinionated sentences
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

        # Remove speakers with only one datapoint
        speaker_counts = df_classified['speaker'].value_counts()
        valid_speakers = speaker_counts[speaker_counts > 1].index
        df_classified = df_classified[df_classified['speaker'].isin(valid_speakers)]

        # Store result
        self.__record[topic]['df_classified'] = df_classified

        print(f'"Number of opinionated speeches for {topic}: {len(df_classified)}')
        return df_classified
    

    def __get_speaker_sentences(self, speaker_name, topic):
        """
        Internal method to get all opinionated sentences from one speaker on a topic.
        
        Args:
            speaker_name: Name of the speaker
            topic: The topic to retrieve sentences for
            
        Returns:
            DataFrame of sentences from this speaker about this topic
        """
        classified_df = self.__record[topic]['df_classified']
        return classified_df[classified_df['speaker'] == speaker_name]
        
    
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

        # Prompt LLM to identify issues and opposing views
        prompt = f"""
            Below is a text summarizing the stance on {topic} of several legislators extracted from the parliamentary proceedings.
            Please use this text as a basis for identifying issues related to this topic, and describe the polar opposing views on these issues.

            Response Format (Please be sure to respond in this format)
            Issue: Outline of the issue
            For: Opinion in favor of the issue
            Against: Opinion against the issue

            Text:
            {text}
            """

        # Request structured JSON output from Ollama
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            format={
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string"},
                        "pro": {"type": "string"},
                        "con": {"type": "string"}
                    },
                    "required": ["topic", "pro", "con"]
                }
            },
            options={"temperature": 0, 'seed': self.random_seed}
        )

        content = response["message"]["content"]

        return json.loads(content)
    
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

