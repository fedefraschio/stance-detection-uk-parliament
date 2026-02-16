import torch
import pandas as pd
import requests
from rouge_score import rouge_scorer

## Utility functions to process data ##


def format_discussion(df: pd.DataFrame, agenda: str, date: str) -> str:
    """
    Given a DataFrame of UK parliamentary speeches, an agenda, and a date,
    returns the formatted text of the discussion.

    Format:
    speaker1 (party1): text
    speaker2 (party2): text
    ...
    """
    # Filter rows by both agenda and date
    discussion_df = df[(df['agenda'] == agenda) & (df['date'] == date)].copy()

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


#for testing purposes only  

def extract_keywords_from_sentence(keywords, sentence, topic):
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
    kws=keywords
    for keyword in kws:
        if keyword.lower() in sentence_lower:
            found_keywords.append(keyword)
    
    return found_keywords



def summarize_parliamentary_speeches(classified_df,keywords,  model_name, num_samples, topic):
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
        found_keywords = extract_keywords_from_sentence(keywords,speech_text, topic)
        
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


