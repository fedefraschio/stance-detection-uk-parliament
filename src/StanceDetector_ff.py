from setfit import SetFitModel


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

    def __init__( self, speech,records,cl_model_hf="andreacristiano/stancedetection"):
        self.speeches_df = speech
        self.record= records    
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
        


    def generate_anchors(self, topic):
        #TODO: use language model to generate few sentences, strongly in favour, moderately in favour, neutral, moderately against, strongly against the topic
        pass

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
