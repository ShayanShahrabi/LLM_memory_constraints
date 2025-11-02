"""
Visualization functions for results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from config import SEQUENCE_LENGTHS, RESULTS_FOLDER

def create_comprehensive_analysis(df_all, model_performance_data, tradeoff_results):
    """Create complete analysis and visualizations"""
    
    # Create separate visualizations

    # 1. Accuracy vs Sequence Length
    plt.figure(figsize=(10, 6))
    for model_name in df_all['model'].unique():
        model_data = df_all[df_all['model'] == model_name]
        avg_data = model_data.groupby('sequence_length')['accuracy'].mean()
        plt.plot(avg_data.index, avg_data.values, marker='o', linewidth=2, label=model_name)

    plt.xlabel('Sequence Length')
    plt.ylabel('Prediction Accuracy')
    plt.title('Model Performance vs Sequence Length')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, '1_accuracy_vs_sequence_length.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Entropy vs Sequence Length
    plt.figure(figsize=(10, 6))
    for model_name in df_all['model'].unique():
        model_data = df_all[df_all['model'] == model_name]
        avg_data = model_data.groupby('sequence_length')['avg_attention_entropy'].mean()
        plt.plot(avg_data.index, avg_data.values, marker='s', linewidth=2, label=model_name)

    plt.xlabel('Sequence Length')
    plt.ylabel('Average Attention Entropy')
    plt.title('Attention Entropy vs Sequence Length')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, '2_entropy_vs_sequence_length.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Memory Capacity Comparison
    plt.figure(figsize=(10, 6))
    capacities = [tradeoff_results[model]['memory_capacity'] for model in tradeoff_results]
    models_list = list(tradeoff_results.keys())
    bars = plt.bar(models_list, capacities, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    plt.xlabel('Model')
    plt.ylabel('Memory Capacity (tokens)')
    plt.title('Comparative Memory Capacity')
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, capacity in zip(bars, capacities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{capacity}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, '3_memory_capacity_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Entropy Change at Capacity Limit
    plt.figure(figsize=(10, 6))
    entropy_changes = [tradeoff_results[model]['entropy_change'] for model in tradeoff_results]
    colors = ['green' if x > 0 else 'red' for x in entropy_changes]
    bars = plt.bar(models_list, entropy_changes, alpha=0.7, color=colors)
    plt.xlabel('Model')
    plt.ylabel('Entropy Change at Capacity Limit')
    plt.title('Attention Strategy at Memory Limit')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xticks(rotation=45)

    # Add value labels
    for bar, change in zip(bars, entropy_changes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.01 if change > 0 else -0.03),
                f'{change:.3f}', ha='center', va='bottom' if change > 0 else 'top')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, '4_entropy_change_at_capacity_limit.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Create trade-off visualization for each model
    for idx, model_name in enumerate(model_performance_data.keys()):
        plt.figure(figsize=(10, 6))

        accuracies = model_performance_data[model_name]['accuracies']
        entropies = model_performance_data[model_name]['entropies']

        # Primary y-axis (accuracy)
        color_acc = 'tab:blue'
        plt.plot(SEQUENCE_LENGTHS, accuracies, marker='o', color=color_acc, linewidth=2, label='Accuracy')
        plt.xlabel('Sequence Length')
        plt.ylabel('Accuracy', color=color_acc)
        plt.tick_params(axis='y', labelcolor=color_acc)
        plt.grid(True, alpha=0.3)

        # Secondary y-axis (entropy)
        color_ent = 'tab:red'
        ax2 = plt.twinx()
        ax2.plot(SEQUENCE_LENGTHS, entropies, marker='s', color=color_ent, linewidth=2, label='Entropy')
        ax2.set_ylabel('Attention Entropy', color=color_ent)
        ax2.tick_params(axis='y', labelcolor=color_ent)

        # Mark memory capacity
        mem_cap = tradeoff_results[model_name]['memory_capacity']
        plt.axvline(x=mem_cap, color='black', linestyle='--', alpha=0.7,
                   label=f'Capacity: {mem_cap}')

        plt.title(f'{model_name}\nCapacity: {mem_cap} tokens')

        # Add correlation
        corr = tradeoff_results[model_name]['accuracy_entropy_correlation']
        plt.text(0.05, 0.95, f'r = {corr:.3f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_FOLDER, f'5_tradeoff_analysis_{model_name.replace(" ", "_").lower()}.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    return tradeoff_results