from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd

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



## Utility function to work with models ##

def load_model(model_checkpoint):
    """
    Load the model and tokenizer.
    """

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        device_map="auto", # <--- Set "cpu", "cuda" or "mps" to explicitly set the device of interest
        dtype=torch.bfloat16 # <--- What's this format? Weren't there just float16 and float32?
        # T4 GPU does not support bfloat :(, it might either give an error or automatically cast to float
    )

    # pad token setting for decoder-only models
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer