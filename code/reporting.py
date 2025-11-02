"""
Report generation functions
"""

import pandas as pd
import os
from config import RESULTS_FOLDER

def generate_final_report(df_all, tradeoff_results):
    """Generate comprehensive final report"""

    # Generate readable report
    with open(os.path.join(RESULTS_FOLDER, "FINAL_EXPERIMENT_REPORT.txt"), 'w') as f:
        f.write("FINAL MEMORY-ATTENTION TRADE-OFF EXPERIMENT REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("EXPERIMENT OVERVIEW:\n")
        f.write(f"- Total data points: {len(df_all)}\n")
        f.write(f"- Sequence lengths tested: {list(df_all['sequence_length'].unique())}\n")
        f.write(f"- Trials per length: {df_all.groupby(['model', 'sequence_length']).size().iloc[0]}\n")
        f.write(f"- Models tested: {', '.join(df_all['model'].unique())}\n\n")

        f.write("KEY FINDINGS:\n")
        f.write("=" * 30 + "\n\n")

        for model_name, metrics in tradeoff_results.items():
            model_data = df_all[df_all['model'] == model_name]
            avg_accuracy = model_data['accuracy'].mean()
            avg_entropy = model_data['avg_attention_entropy'].mean()

            f.write(f"{model_name}:\n")
            f.write(f"  - Average Accuracy: {avg_accuracy:.4f}\n")
            f.write(f"  - Average Entropy: {avg_entropy:.4f}\n")
            f.write(f"  - Memory Capacity: {metrics['memory_capacity']} tokens\n")
            f.write(f"  - Entropy Change at Limit: {metrics['entropy_change']:.4f}\n")
            f.write(f"  - Accuracy-Entropy Correlation: {metrics['accuracy_entropy_correlation']:.4f}\n")

            if metrics['entropy_change'] > 0:
                f.write(f"  - Strategy: INCREASES entropy when memory overloaded\n")
            else:
                f.write(f"  - Strategy: DECREASES entropy when memory overloaded\n")
            f.write("\n")

        f.write("CONCLUSIONS:\n")
        f.write("=" * 20 + "\n")
        f.write("1. Models show different memory capacity limits\n")
        f.write("2. Attention entropy patterns reveal memory management strategies\n")
        f.write("3. Correlation between accuracy and entropy indicates trade-off behavior\n")
        f.write("4. Masked vs causal models show fundamentally different patterns\n")

    print(f"\n📊 FINAL RESULTS SUMMARY:")
    print("=" * 50)
    for model_name, metrics in tradeoff_results.items():
        trend = "↑ increases" if metrics['entropy_change'] > 0 else "↓ decreases"
        print(f"   {model_name:15} | Capacity: {metrics['memory_capacity']:2d} tokens | "
              f"Entropy {trend} | r = {metrics['accuracy_entropy_correlation']:.3f}")