"""
Data analysis and statistical functions
"""

import numpy as np
import pandas as pd
from scipy import stats
from config import SEQUENCE_LENGTHS, RESULTS_FOLDER
import os

def analyze_memory_attention_tradeoff(accuracy_curve, entropy_curve, sequence_lengths):
    """Enhanced trade-off analysis"""

    if len(accuracy_curve) != len(entropy_curve) or len(accuracy_curve) < 3:
        return {
            'memory_capacity': sequence_lengths[-1] if accuracy_curve else 0,
            'memory_limit_index': len(sequence_lengths) - 1,
            'pre_limit_entropy': np.mean(entropy_curve) if entropy_curve else 0,
            'post_limit_entropy': np.mean(entropy_curve) if entropy_curve else 0,
            'entropy_change': 0,
            'entropy_change_abs': 0,
            'accuracy_entropy_correlation': 0,
            'correlation_p_value': 1,
            'max_accuracy': max(accuracy_curve) if accuracy_curve else 0,
            'min_accuracy': min(accuracy_curve) if accuracy_curve else 0
        }

    # Find point where accuracy drops below 0.5 (or significant drop)
    accuracy_array = np.array(accuracy_curve)

    # Find significant drop (more than 30% decrease from max)
    max_acc = np.max(accuracy_array)
    threshold = max_acc * 0.7  # 30% drop threshold

    drop_points = np.where(accuracy_array < threshold)[0]
    if len(drop_points) > 0:
        memory_limit_idx = drop_points[0]
    else:
        # If no clear drop, use the point where accuracy is lowest
        memory_limit_idx = np.argmin(accuracy_array)

    memory_capacity = sequence_lengths[memory_limit_idx]

    # Calculate entropy changes
    if memory_limit_idx > 0 and memory_limit_idx < len(entropy_curve):
        pre_limit_entropy = np.mean(entropy_curve[:memory_limit_idx])
        post_limit_entropy = np.mean(entropy_curve[memory_limit_idx:])
        entropy_change = post_limit_entropy - pre_limit_entropy
    else:
        pre_limit_entropy = np.mean(entropy_curve)
        post_limit_entropy = np.mean(entropy_curve)
        entropy_change = 0

    # Calculate correlation
    try:
        correlation, p_value = stats.pearsonr(accuracy_curve, entropy_curve)
    except:
        correlation, p_value = 0, 1

    return {
        'memory_capacity': memory_capacity,
        'memory_limit_index': memory_limit_idx,
        'pre_limit_entropy': pre_limit_entropy,
        'post_limit_entropy': post_limit_entropy,
        'entropy_change': entropy_change,
        'entropy_change_abs': abs(entropy_change),
        'accuracy_entropy_correlation': correlation,
        'correlation_p_value': p_value,
        'max_accuracy': max_acc,
        'min_accuracy': min(accuracy_array)
    }

def calculate_tradeoff_results(model_performance_data):
    """Calculate trade-off results for all models"""
    tradeoff_results = {}
    for model_name in model_performance_data.keys():
        accuracies = model_performance_data[model_name]['accuracies']
        entropies = model_performance_data[model_name]['entropies']
        tradeoff_results[model_name] = analyze_memory_attention_tradeoff(accuracies, entropies, SEQUENCE_LENGTHS)
    
    return tradeoff_results

def save_analysis_results(df_all, tradeoff_results):
    """Save analysis results to files"""
    # Save raw data
    df_all.to_csv(os.path.join(RESULTS_FOLDER, "final_detailed_analysis.csv"), index=False)

    # Save trade-off results
    tradeoff_df = pd.DataFrame.from_dict(tradeoff_results, orient='index')
    tradeoff_df.reset_index(inplace=True)
    tradeoff_df.rename(columns={'index': 'model'}, inplace=True)
    tradeoff_df.to_csv(os.path.join(RESULTS_FOLDER, "final_tradeoff_results.csv"), index=False)

    # Save summary statistics
    summary_stats = df_all.groupby('model').agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'avg_attention_entropy': ['mean', 'std', 'min', 'max'],
        'avg_attention_focus': ['mean', 'std', 'min', 'max'],
        'sequence_length': 'count'
    }).round(4)

    summary_stats.to_csv(os.path.join(RESULTS_FOLDER, "final_summary_statistics.csv"))
    
    return tradeoff_df, summary_stats