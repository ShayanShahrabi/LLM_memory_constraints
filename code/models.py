"""
Model loading and management
"""

import torch
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    AutoModelForCausalLM, AutoTokenizer,
    AutoModelForMaskedLM,
    GPTNeoForCausalLM
)
from config import MODELS_CONFIG

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(model_name, model_id, model_type):
    """Load model and tokenizer based on model type"""
    try:
        if model_type == "causal":
            if "phi" in model_id.lower():
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    output_attentions=True,
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                ).to(device)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    output_attentions=True,
                    torch_dtype=torch.float16
                ).to(device)

            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

        elif model_type == "masked":
            model = AutoModelForMaskedLM.from_pretrained(
                model_id,
                output_attentions=True,
                torch_dtype=torch.float16
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_id)

        print(f"✓ Successfully loaded {model_name}")
        return model, tokenizer, model_type

    except Exception as e:
        print(f"✗ Failed to load {model_name}: {e}")
        return None, None, None

def load_all_models():
    """Load all models specified in configuration"""
    print("Loading models...")
    models = []
    for config in MODELS_CONFIG:
        model, tokenizer, model_type = load_model_and_tokenizer(*config)
        if model is not None:
            models.append((config[0], model, tokenizer, model_type))
    
    print(f"Loaded {len(models)} models successfully")
    return models