import torch
import pandas as pd

## Utility functions to process data ##

class StanceDetector:
    
    
    ''''
    GENERAL FUNCTIONING:
    1. Load speeches dataset in it, in speeches_df
    2. Filter the dataset according to a topic and its keyword list, put it in filtered_df 
        (this way it means that each instance of a class contains just a single topic since there is just one filtered_df
        which contains filtered speeches about one single topic, it can be solved in the future)
    3. Classify all the sentences in filtered_df in (non)opinionated. Add a classification column to filtered_df
    4. Take only the classified as opinionated and put them in classified_df
        At this point classified_df contains all the sentences with an opinion about a single topic.
        This is enough to track the amount of data available for each year.
        Classified_df will still contain metadata like date, author and party
    5. Summarize all the 'text' from classified_df into a new dataframe
    '''

    """
    The dataframe we'll be working on has the following columns:
    ['instance_id', 'date', 'agenda', 'speechnumber', 'sentencenumber',
       'speaker', 'party', 'text', 'parliament', 'iso3country', 'chair', 'eu',
       'policyarea', 'cmp_party']

    There is a big jsonfile we'll be working on, with the following structure:
    {
        "nuclear":{
            'keywords':[...],
            'policyarea':[7, 8],
            'df_filtered':...,
            'df_classified':...

        },
        
    }
    Thus there is going to be one class instance for each database. In each instance we'll have multiple topics 
    every topic will have its own keyword list, policyarea list, filtered dataframe and classified dataframe.
    """
    record=None
    filtered_df=None
    classified_df=None
    topic = None
    policyarea=[]

    def __init__( self, speech,records, classification_model_path='./../models/setfit-fewshot-opinions-1', cl_model_hf="andreacristiano/stancedetection"):
        self.speeches_df = speech
        self.record= records    
        self.classification_model_path= classification_model_path
        self.model= SetFitModel.from_pretrained(cl_model_hf)
    #AUXILIARY FUNCTIONS
    def get_records(self):
        return self.records
    
    def get_speeches(self):
        return self.speeches_df
    
    def get_filtered_speeches(self, topic):
        return self.records[topic]['df_filtered']
    
    def get_classified_speeches(self, topic):
        return self.records[topic]['df_classified']

    def format_discussion(self,agenda, date):
        
        """
        IN: a speech, identified uniquely by its agenda and date
        OUT: speech in dialogue format

        Format:
        speaker1 (party1): text
        speaker2 (party2): text
        ...
        """
        
        # Filter rows by both agenda and date
        discussion_df = self.speeches_df[(self.speeches_df['agenda'] == agenda) & (self.speeches_df['date'] == date)].copy()

        if discussion_df.empty:
            return f"No discussion found for agenda '{agenda}' on date '{date}'."

        # Sort by speechnumber and sentencenumber to maintain chronological order
        discussion_df.sort_values(by=['speechnumber', 'sentencenumber'], inplace=True)

        # Group sentences belonging to the same speech (same speechnumber, speaker, and party)
        grouped = (
            discussion_df
            .groupby(['speechnumber', 'speaker', 'party'])['text']
            .apply(' '.join)
            .reset_index()
        )

        # Format each speaker's speech
        formatted_lines = [f"{row['speaker']} ({row['party']}): {row['text']}" for _, row in grouped.iterrows()]

        # Join everything into one formatted discussion
        return "\n".join(formatted_lines)

        
    def add_record(self, topic, keywords, policyarea):
        """
        IN: topic name, list of keywords, list of policyarea codes
        """
        self.records[topic] = {
            'keywords':keywords,
            'policyarea':policyarea,
        }

    def __get_keywords_policyarea(self,topic):
        return self.record[topic]['keywords'], self.record[topic]['policyarea']

    #WORKFLOW FUNCTIONS
    def filter_speeches(self,topic):
        #https://www.comparativeagendas.net/pages/master-codebook 
        #here there's a list of policiarea codes for each topic, these will be integrated in the filtering to make it more effective.
        keywords, policyarea = self.__get_keywords_policyarea(topic)
        #filter the dataframe according to the policyarea values. policyarea is an array containing integers
        
        self.record[topic]['df_filtered']= self.speeches_df[self.speeches_df['policyarea'].isin(policyarea)]
        self.record[topic]['df_filtered'] = self.speeches_df[self.speeches_df['text'].str.contains('|'.join(keywords), case=False, na=False)]
        print("BRO")
        
        return self.record[topic]['df_filtered']

    def classify_filtered_sentences(self, topic):
        #TODO: an easier method to access single topic data inside the class can be made
        """"
        IN: filtered dataframe about a topic
        OUT: classified dataframe containing opinonated sentences only
             + a new column 'classification' in filtered dataframe
        """
        #model = SetFitModel.from_pretrained("andreacristiano/stancedetection")
        #model = SetFitModel.from_pretrained(self.classification_model_path)
        # Get the texts to classify from self.filtered_df
        texts = self.record[topic]['df_filtered']['text'].tolist()
        # Perform classification
        predictions = self.model.predict(texts)
        # Add predictions as a new column in self.filtered_df
        self.record[topic]['df_filtered']['classification'] = predictions
        self.record[topic]['df_classified'] = self.record[topic]['df_filtered'][self.record[topic]['df_filtered']['classification'] == 'opinion']
        return self.record[topic]['df_classified']
    
    def get_full_discussions(self, topic):
        """
        IN: topic name
        OUT: dataframe with full discussions for each unique (agenda, date) pair
            in the classified sentences
        
        Creates a new dataframe with columns:
        - text: full formatted discussion using format_discussion()
        - speakers: list of speakers involved
        - parties: list of parties involved
        - date: discussion date
        - agenda: discussion agenda
        
        Saves result in self.records[topic]['full_classified_df']
        """
        classified_df = self.record[topic]['df_classified']
        
        if classified_df.empty:
            print(f"No classified sentences found for topic '{topic}'")
            self.record[topic]['full_classified_df'] = pd.DataFrame(
                columns=['text', 'speakers', 'parties', 'date', 'agenda']
            )
            return self.record[topic]['full_classified_df']
        
        # Get unique (agenda, date) pairs from classified sentences
        unique_discussions = classified_df[['agenda', 'date']].drop_duplicates()
        
        full_discussions = []
        
        for _, row in unique_discussions.iterrows():
            agenda = row['agenda']
            date = row['date']
            
            # Get the formatted full discussion text
            discussion_text = self.format_discussion(agenda, date)
            
            # Get all speakers and parties from this discussion in the original dataset
            discussion_df = self.speeches_df[
                (self.speeches_df['agenda'] == agenda) & 
                (self.speeches_df['date'] == date)
            ]
            
            # Extract unique speakers and parties (maintaining order of appearance)
            speakers = discussion_df.sort_values('speechnumber')['speaker'].unique().tolist()
            parties = discussion_df.sort_values('speechnumber')['party'].unique().tolist()
            
            full_discussions.append({
                'text': discussion_text,
                'speakers': speakers,
                'parties': parties,
                'date': date,
                'agenda': agenda
            })
        
        # Create the dataframe
        self.record[topic]['full_classified_df'] = pd.DataFrame(full_discussions)
        
        print(f"Retrieved {len(full_discussions)} full discussions for topic '{topic}'")
        
        return self.record[topic]['full_classified_df']
        


    def generate_anchors(self, topic, model_name='gemma3'):
        """
        Generate anchor sentences representing different stances on a topic.
        
        Parameters:
        -----------
        topic : str
            The topic to generate anchors for
        model_name : str
            Name of the Ollama model to use (default: 'gemma3')
        
        Returns:
        --------
        dict
            Dictionary with stance levels as keys and generated sentences as values
            Keys: 'strongly_favor', 'moderately_favor', 'neutral', 'moderately_against', 'strongly_against'
        """
        
        stance_levels = {
            'strongly_favor': f"Write a single parliamentary-style sentence expressing strong support for {topic}. Be clear and assertive.",
            'moderately_favor': f"Write a single parliamentary-style sentence expressing moderate support for {topic}. Show cautious approval.",
            'neutral': f"Write a single parliamentary-style sentence expressing a neutral or balanced view on {topic}. Show no clear preference.",
            'moderately_against': f"Write a single parliamentary-style sentence expressing moderate opposition to {topic}. Show concerns but not extreme rejection.",
            'strongly_against': f"Write a single parliamentary-style sentence expressing strong opposition to {topic}. Be clear and firm in disagreement."
        }
        
        system_prompt = """You are a UK parliamentary speech writer. Generate realistic parliamentary statements.
        Keep sentences concise and natural. Use appropriate parliamentary language.
        Return ONLY the sentence itself, without any preamble, explanation, or quotation marks."""
        
        anchors = {}
        
        print(f"Generating stance anchors for topic: {topic}")
        print("=" * 80)
        
        for stance, prompt in stance_levels.items():
            payload = {
                "model": model_name,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "temperature": 0.7
            }
            
            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json=payload,
                    timeout=60
                )
                response.raise_for_status()
                
                # Extract and clean the generated sentence
                generated_text = response.json()['response'].strip()
                # Remove quotes if present
                generated_text = generated_text.strip('"').strip("'")
                
                anchors[stance] = generated_text
                
                print(f"\n{stance.upper().replace('_', ' ')}:")
                print(f"  {generated_text}")
                
            except requests.exceptions.RequestException as e:
                print(f"\nError generating anchor for {stance}: {e}")
                anchors[stance] = f"ERROR: {str(e)}"
        
        print("\n" + "=" * 80)
        print(f"Generated {len(anchors)} anchor sentences")
        
        # Store anchors in the record for this topic
        self.record[topic]['anchors'] = anchors
        
        return anchors

    def generate_latent_space(self, topic):
        #TODO: generate latent space representation of the classified and summarized speeches
        pass

    #for testing purposes only  
    def __extract_keywords_from_sentence(self, sentence, topic):
        """
        Extract keywords that appear in a given sentence.
        
        Parameters:
        -----------
        sentence : str
            The sentence to search for keywords
        keywords : list
            List of keywords to search for
        
        Returns:
        --------
        list
            List of keywords found in the sentence
        """
        # Convert sentence to lowercase for case-insensitive matching
        sentence_lower = sentence.lower()
        
        # Find keywords that appear in the sentence
        found_keywords = []
        kws=self.record[topic]['keywords']
        for keyword in kws:
            if keyword.lower() in sentence_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def summarize(self, topic, model_name='gemma3'):

        system_prompt = f"""You are summarizing speeches from the UK Parliament about {topic}. 
        Focus on capturing the speaker's stance or position regarding {topic}. 
        Provide brief, concise summaries that clearly indicate whether the speaker supports, opposes, or has a nuanced view on {topic}.
        Keep summaries to 1-2 sentences maximum. if the speech does not express any stance on {topic}, reply NO"""

        classified_df=self.record[topic]['df_classified']
        #summarize every sentence in classified_df and add it to a new column 'summary'
        summaries = []
        for idx, row in classified_df.iterrows():
            speech_text = row['text']
            # Prepare the request to Ollama API
            payload = {
                "model": model_name,
                "prompt": f"Summarize this parliamentary speech about {topic}, focusing on the speaker's stance:\n\n{speech_text}",
                "system": system_prompt,
                "stream": False
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
                summaries.append(summary)
            except requests.exceptions.RequestException as e:
                print(f"\nError processing sample {idx}: {e}")
                summaries.append(f"ERROR: {str(e)}")
        #remove all the summaries whose text is 'NO' or start with 'Please provide'
        classified_df['summary'] = summaries
        classified_df = classified_df[
            (classified_df['summary'] != 'NO') &
            (~classified_df['summary'].str.lower().str.startswith('please provide'))
        ]
        self.record[topic]['df_classified'] = classified_df
        return classified_df
    
    def summarize_parliamentary_speeches(self, model_name, num_samples, topic):
        """
        Summarize random samples from UK parliamentary speeches using Ollama.
        Computes ROUGE scores, extracts keywords, and displays results.
        
        Parameters:
        -----------
        we start on a classified dataframe
        model_name : str
            Name of the Ollama model to use (e.g., 'llama2', 'mistral')
        num_samples : int
            Number of random samples to extract and summarize
        topic : str
            The topic that all speeches are about (e.g., 'climate change', 'Brexit')
        keywords : list
            List of keywords to search for in each sentence
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: 'original_text', 'summary', 'keywords', 'rouge1', 'rouge2', 'rougeL'
        """
        # Extract random samples keeping only the 'text' column
        classified_df=self.record[topic]['df_classified']

        samples = classified_df[['text']].sample(n=num_samples, random_state=None)
        
        #TODO: generalize system prompt for every parliament
        # System prompt for UK parliamentary context with focus on stance
        system_prompt = f"""You are summarizing speeches from the UK Parliament about {topic}. 
        Focus on capturing the speaker's stance or position regarding {topic}. 
        Provide brief, concise summaries that clearly indicate whether the speaker supports, opposes, or has a nuanced view on {topic}.
        Keep summaries to 1-2 sentences maximum. if the speech does not express any stance on {topic}, reply NO"""
        
        # Initialize ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        results = []
        
        print("=" * 100)
        print(f"PARLIAMENTARY SPEECH SUMMARIZATION WITH ROUGE SCORES")
        print(f"TOPIC: {topic}")
        print("=" * 100)
        
        for idx, row in enumerate(samples.iterrows(), 1):
            speech_text = row[1]['text']
            
            # Extract keywords from the original text
            found_keywords = self.__extract_keywords_from_sentence(speech_text, topic)
            
            # Prepare the request to Ollama API
            payload = {
                "model": model_name,
                "prompt": f"Summarize this parliamentary speech about {topic}, focusing on the speaker's stance:\n\n{speech_text}",
                "system": system_prompt,
                "stream": False
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
                
                # Compute ROUGE scores
                scores = scorer.score(speech_text, summary)
                
                # Extract F1 scores
                rouge1_f1 = scores['rouge1'].fmeasure
                rouge2_f1 = scores['rouge2'].fmeasure
                rougeL_f1 = scores['rougeL'].fmeasure
                
                results.append({
                    'original_text': speech_text,
                    'summary': summary,
                    'keywords': found_keywords,
                    'rouge1': rouge1_f1,
                    'rouge2': rouge2_f1,
                    'rougeL': rougeL_f1
                })
                
                # Print results for this sample
                print(f"\n{'─' * 100}")
                print(f"SAMPLE {idx}/{num_samples}")
                print(f"{'─' * 100}")
                print(f"\nORIGINAL TEXT:")
                print(f"{speech_text}\n")
                print(f"SUMMARY:")
                print(f"{summary}\n")
                print(f"KEYWORDS FOUND: {', '.join(found_keywords) if found_keywords else 'None'}\n")
                print(f"ROUGE SCORES:")
                print(f"  ROUGE-1: {rouge1_f1:.4f}")
                print(f"  ROUGE-2: {rouge2_f1:.4f}")
                print(f"  ROUGE-L: {rougeL_f1:.4f}")
                
            except requests.exceptions.RequestException as e:
                print(f"\nError processing sample {idx}: {e}")
                results.append({
                    'original_text': speech_text,
                    'summary': f"ERROR: {str(e)}",
                    'keywords': found_keywords,
                    'rouge1': 0.0,
                    'rouge2': 0.0,
                    'rougeL': 0.0
                })
        
        print(f"\n{'=' * 100}")
        print("SUMMARY STATISTICS")
        print(f"{'=' * 100}")
        
        df_results = pd.DataFrame(results)
        
        if len(df_results) > 0:
            print(f"\nAverage ROUGE Scores:")
            print(f"  ROUGE-1: {df_results['rouge1'].mean():.4f}")
            print(f"  ROUGE-2: {df_results['rouge2'].mean():.4f}")
            print(f"  ROUGE-L: {df_results['rougeL'].mean():.4f}")
        
        return df_results

    def plot_umap_party_averages(self,
                             topic,
                             model_name="all-MiniLM-L6-v2",
                             n_components=2,
                             n_neighbors=10,
                             min_dist=0.1,
                             metric="cosine",
                             show_speeches=True):

        # --- Load data ---
        classified_df = self.record[topic]['df_classified'].copy()
        summaries = classified_df["summary"].tolist()
        parties = classified_df["party"].tolist()

        # --- Sentence embeddings (per speech) ---
        model = SentenceTransformer(model_name)
        embeddings = model.encode(summaries, show_progress_bar=True)

        # --- Global UMAP ---
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42
        )
        reduced_embeddings = reducer.fit_transform(embeddings)

        # --- Attach UMAP coords ---
        classified_df["umap_x"] = reduced_embeddings[:, 0]
        classified_df["umap_y"] = reduced_embeddings[:, 1]

        # --- Party centroids (AFTER UMAP) ---
        party_centroids = (
            classified_df
            .groupby("party")[["umap_x", "umap_y"]]
            .mean()
            .reset_index()
        )

        # --- Color mapping (same as before) ---
        unique_parties = sorted(classified_df["party"].unique())
        party_to_color = {party: i for i, party in enumerate(unique_parties)}
        cmap = plt.get_cmap("tab10")

        # --- Plot ---
        plt.figure(figsize=(10, 8))

        for party in unique_parties:
            color = cmap(party_to_color[party])

            # speeches
            if show_speeches:
                mask = classified_df["party"] == party
                plt.scatter(
                    classified_df.loc[mask, "umap_x"],
                    classified_df.loc[mask, "umap_y"],
                    alpha=0.25,
                    color=color,
                    label=party
                )

            # centroid (bigger circle, same color)
            centroid = party_centroids[party_centroids["party"] == party]
            plt.scatter(
                centroid["umap_x"],
                centroid["umap_y"],
                s=300,
                color=color,
                edgecolor="black",
                linewidth=1.2
            )

        plt.title(f"UMAP Party Averages – {topic.capitalize()}")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")

        # --- Legend: same behavior as before ---
        plt.legend(title="Parties", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.show()

        return reduced_embeddings




