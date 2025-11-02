"""
Experiment execution and model evaluation
"""

import numpy as np
import torch
from tqdm import tqdm
from utils import generate_sequence, robust_attention_entropy, calculate_attention_focus
from config import SEQUENCE_LENGTHS, NUM_TRIALS_PER_LENGTH

def evaluate_model_with_robust_entropy(sequence, model, tokenizer, model_name, model_type,
                                     save_map=False, seq_len=None, trial=None):
    """Unified evaluation with robust entropy calculation"""
    try:
        # Tokenize based on model type
        if model_type == "causal":
            inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        else:  # masked
            inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=512, padding=True).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            attentions = outputs.attentions
            logits = outputs.logits

        # Calculate accuracy based on model type
        accuracy = calculate_accuracy(inputs, logits, model_type)

        # ROBUST attention metrics calculation
        entropy_vals, focus_vals, layer_entropies = calculate_attention_metrics(attentions)

        avg_entropy = np.mean(entropy_vals) if entropy_vals else 0
        avg_focus = np.mean(focus_vals) if focus_vals else 0
        num_heads = attentions[0].shape[1] if attentions else 0
        num_layers = len(attentions) if attentions else 0

        return {
            'accuracy': accuracy,
            'avg_attention_entropy': avg_entropy,
            'avg_attention_focus': avg_focus,
            'layer_entropies': layer_entropies,
            'num_attention_heads': num_heads,
            'num_layers': num_layers
        }

    except Exception as e:
        print(f"Error evaluating {model_name} at seq_len {seq_len}: {e}")
        return {
            'accuracy': 0,
            'avg_attention_entropy': 0,
            'avg_attention_focus': 0,
            'layer_entropies': [],
            'num_attention_heads': 0,
            'num_layers': 0
        }

def calculate_accuracy(inputs, logits, model_type):
    """Calculate prediction accuracy based on model type"""
    if model_type == "causal":
        input_ids = inputs["input_ids"][0]
        correct, total = 0, len(input_ids) - 1
        for i in range(total):
            predicted_id = torch.argmax(logits[0, i]).item()
            if predicted_id == input_ids[i + 1].item():
                correct += 1
        return correct / total if total > 0 else 0
    else:  # masked
        input_ids = inputs["input_ids"][0]
        mask_positions = random.sample(range(1, len(input_ids)-1), min(3, len(input_ids)-2))
        correct, total = 0, len(mask_positions)
        for pos in mask_positions:
            original_id = input_ids[pos].item()
            predicted_id = torch.argmax(logits[0, pos]).item()
            if predicted_id == original_id:
                correct += 1
        return correct / total if total > 0 else 0

def calculate_attention_metrics(attentions):
    """Calculate attention entropy and focus metrics"""
    entropy_vals = []
    focus_vals = []
    layer_entropies = []

    if attentions is not None:
        for layer_idx, layer_attn in enumerate(attentions):
            layer_entropy = []
            batch, heads, seq_len_attn, _ = layer_attn.shape

            # Sample multiple heads and tokens for better statistics
            sampled_heads = min(4, heads)  # Sample up to 4 heads
            sampled_tokens = min(6, seq_len_attn)  # Sample up to 6 tokens

            for head_idx in range(0, heads, max(1, heads // sampled_heads)):
                head_entropies = []
                for token_idx in range(0, seq_len_attn, max(1, seq_len_attn // sampled_tokens)):
                    try:
                        attn_weights = layer_attn[0, head_idx, token_idx]
                        entropy_val = robust_attention_entropy(attn_weights)
                        head_entropies.append(entropy_val)
                        entropy_vals.append(entropy_val)
                    except:
                        continue

                # Also calculate focus for this head
                try:
                    focus_val = calculate_attention_focus(layer_attn[0, head_idx])
                    focus_vals.append(focus_val)
                except:
                    continue

            if head_entropies:
                layer_entropies.append(np.mean(head_entropies))

    return entropy_vals, focus_vals, layer_entropies

def run_final_experiment(models):
    """Run the final comprehensive experiment"""
    all_results = []
    model_performance_data = {model[0]: {'accuracies': [], 'entropies': []} for model in models}

    print("Starting FINAL memory-attention trade-off analysis...")
    print(f"Testing {len(models)} models across {len(SEQUENCE_LENGTHS)} sequence lengths")

    for model_name, model, tokenizer, model_type in tqdm(models, desc="Models"):
        print(f"\n🔍 Testing {model_name} ({model_type})...")

        model_accuracies = []
        model_entropies = []

        for seq_len in tqdm(SEQUENCE_LENGTHS, desc=f"Sequence lengths", leave=False):
            seq_accuracies = []
            seq_entropies = []

            for trial in range(NUM_TRIALS_PER_LENGTH):
                sequence, task_info = generate_sequence(seq_len)

                results = evaluate_model_with_robust_entropy(
                    sequence, model, tokenizer, model_name, model_type,
                    save_map=False, seq_len=seq_len, trial=trial
                )

                # Store results
                result_entry = {
                    'model': model_name,
                    'model_type': model_type,
                    'sequence_length': seq_len,
                    'trial': trial,
                    'accuracy': results['accuracy'],
                    'avg_attention_entropy': results['avg_attention_entropy'],
                    'avg_attention_focus': results['avg_attention_focus'],
                    'num_attention_heads': results['num_attention_heads'],
                    'num_layers': results['num_layers'],
                    'task_type': task_info['task_type']
                }

                all_results.append(result_entry)
                seq_accuracies.append(results['accuracy'])
                seq_entropies.append(results['avg_attention_entropy'])

            # Store averages for this sequence length
            model_accuracies.append(np.mean(seq_accuracies))
            model_entropies.append(np.mean(seq_entropies))

        # Store model-level data for trade-off analysis
        model_performance_data[model_name]['accuracies'] = model_accuracies
        model_performance_data[model_name]['entropies'] = model_entropies

    return all_results, model_performance_data