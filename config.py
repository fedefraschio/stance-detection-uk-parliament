# Configuration for Stance Detection Pipeline

# Topics to analyze and their date ranges
TOPICS_CONFIG = {
    'Gaza': {
        'years': [2011, 2012, 2013, 2014],
        'keywords': ['Gaza', 'Palestine', 'Israel', 'Palestinian'],
        'description': 'Israeli-Palestinian conflict analysis'
    },
    'climate change': {
        'years': [2014],
        'keywords': ['climate', 'green', 'renewable', 'carbon', 'fossil'],
        'description': 'UK energy transition analysis'
    }
}

# Model configuration
MODEL_CONFIG = {
    'embedding_model': 'sentence-transformers/all-MiniLM-L6-v2',
    'llm_model': 'qwen3:8b',
    'llm_with_thinking': True,
    'random_seed': 41
}

# Data paths
DATA_CONFIG = {
    'raw_data': './../data/raw/ParlEE_UK_plenary_speeches.csv',
    'external_records': './../data/external/records.json',
    'processed_dir': './../data/processed/',
    'outputs_dir': './../outputs/'
}

# Pipeline parameters
PIPELINE_CONFIG = {
    'min_opinion_sentences': 3,  # Minimum opinion-bearing sentences per speaker
    'min_speakers_per_party': 1,  # Minimum speakers to include party (recommend: 3-5)
    'embedding_batch_size': 32,
    'umap_n_neighbors': 15,
    'umap_min_dist': 0.1,
    'umap_metric': 'cosine'
}

# Validation parameters
VALIDATION_CONFIG = {
    'metric': 'spearman',  # 'spearman', 'kendall', or 'lcs'
    'significance_level': 0.05,
    'min_anchor_similarity': 0.3  # Minimum cosine sim between anchors (quality gate)
}

# Feature flags
FEATURES = {
    'use_cache': True,
    'save_embeddings': True,
    'generate_visualizations': True,
    'debug_mode': False
}
