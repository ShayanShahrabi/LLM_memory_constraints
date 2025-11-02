"""
Main execution script for memory-attention trade-off analysis
Paper: [Your Paper Title]
Author: [Your Name]
"""

import os
import pandas as pd
from config import RESULTS_FOLDER
from models import load_all_models
from experiments import run_final_experiment
from analysis import calculate_tradeoff_results, save_analysis_results
from visualization import create_comprehensive_analysis
from reporting import generate_final_report

def main():
    print("🚀 FINAL Memory-Attention Trade-off Analysis")
    print("=" * 60)

    # Create results directory
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    # Load models
    models = load_all_models()
    if not models:
        print("❌ No models loaded successfully. Exiting.")
        return

    # Run the final experiment
    all_results, model_performance_data = run_final_experiment(models)

    # Convert to DataFrame
    df_all = pd.DataFrame(all_results)

    # Calculate trade-off results
    tradeoff_results = calculate_tradeoff_results(model_performance_data)

    # Save analysis results
    save_analysis_results(df_all, tradeoff_results)

    # Create comprehensive analysis and visualizations
    create_comprehensive_analysis(df_all, model_performance_data, tradeoff_results)

    # Generate final report
    generate_final_report(df_all, tradeoff_results)

    print(f"\n✅ FINAL Experiment Completed Successfully!")
    print(f"📁 All results saved to '{RESULTS_FOLDER}' folder")
    print(f"📈 Ready for research paper writing!")

if __name__ == "__main__":
    main()