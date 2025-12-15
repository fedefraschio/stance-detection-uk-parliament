from setfit import SetFitModel 
import json
import os
import re

class DatasetFilterer:

    record = None
    topic = None
    filtered_dataset_df = None
    classified_df = None


    # TODO: migliorare gestione errori caricamento dati e modelli
    def __init__( self, speech, records, cl_model_hf="andreacristiano/stancedetection"):
        self.speeches_df = speech

        if records is None:
            raise ValueError("Records data could not be loaded.")
        
        self.record = records

        self.model= SetFitModel.from_pretrained(cl_model_hf)

        if self.model is None:
            raise ValueError("Classification model could not be loaded.")


    #AUXILIARY FUNCTIONS
    def get_records(self):
        return self.record
    
    def get_speeches(self):
        return self.speeches_df
    
    def get_filtered_speeches(self, topic):
        return self.record[topic]['df_filtered']
    
    def get_classified_speeches(self, topic):
        return self.record[topic]['df_classified']
    
    def __get_keywords_policyarea(self,topic):
        return self.record[topic]['keywords'], self.record[topic]['policyarea']
    
    #WORKFLOW FUNCTIONS

    def filter_speeches(self, topic):
        keywords, policyarea = self.__get_keywords_policyarea(topic)

        # Filtering per policy area (faster than keyword search) - shrinks dataframe
        df = self.speeches_df[self.speeches_df['policyarea'].isin(policyarea)]

        # Creating regex pattern for keyword search
        pattern = re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)

        # Filtering the text on the already shrunk dataframe
        df = df[df["text"].str.contains(pattern, na=False)]

        # save
        self.record[topic]['df_filtered'] = df

        return df

    def classify_sentences(self, topic):
        # TODO: optimize method
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