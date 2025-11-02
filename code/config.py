"""
Configuration file for memory-attention trade-off experiment
"""

# Experiment parameters
SEQUENCE_LENGTHS = [5, 10, 15, 20, 25, 30, 40, 50]
NUM_TRIALS_PER_LENGTH = 10
RESULTS_FOLDER = "final_results"

# Model configurations
MODELS_CONFIG = [
    ("GPT-2", "gpt2", "causal"),
    ("DistilBERT", "distilbert-base-uncased", "masked"),
    ("Microsoft Phi-2", "microsoft/phi-2", "causal"),
    ("GPT-Neo 125M", "EleutherAI/gpt-neo-125M", "causal"),
]

# Analysis parameters
FOCUS_WINDOW = 3  # For attention focus calculation
EPS = 1e-12  # For numerical stability