#!/usr/bin/env python3
"""
Comprehensive Oversampling Methods Experiment Runner

This script runs all available oversampling methods with all models and saves
comprehensive results including metrics, plots, and comparisons.

Usage:
    python run_comprehensive_experiments.py [--dynamic-threshold]
"""

import argparse
import sys
import time
import os
from datetime import datetime
from multi_model_runner import run_all_models

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive oversampling experiments')
    parser.add_argument('--dynamic-threshold', action='store_true', 
                       help='Use dynamic threshold in active learning')
    
    args = parser.parse_args()
    
    # Set base directory based on threshold type
    base_dir = "results_dynamic_threshold" if args.dynamic_threshold else "results_static_threshold"
    
    print("="*80)
    print("COMPREHENSIVE OVERSAMPLING METHODS EXPERIMENT")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Dynamic threshold: {'Enabled' if args.dynamic_threshold else 'Disabled'}")
    print(f"Results will be saved in: {base_dir}/")
    print("="*80)
    
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
    
    start_time = time.time()
    
    try:
        # Run all experiments with base directory
        all_results = run_all_models(use_dynamic_threshold=args.dynamic_threshold, base_dir=base_dir)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Total duration: {duration/60:.2f} minutes")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\nExperiment results structure:")
        print(f"├── {base_dir}/")
        print(f"│   ├── overall_comparison/")
        print(f"│   │   ├── comprehensive_results.csv")
        print(f"│   │   ├── accuracy_heatmap.png")
        print(f"│   │   └── metrics_comparison.png")
        
        for method in all_results.keys():
            print(f"│   ├── {method}/")
            print(f"│   │   ├── classification_reports (*.txt)")
            print(f"│   │   ├── metrics (*.csv)")
            print(f"│   │   ├── confusion_matrix_*.png")
            print(f"│   │   ├── embedding_*_pca.png")
            print(f"│   │   ├── learning_curves_*.png")
            print(f"│   │   ├── roc_curve_*.png")
            print(f"│   │   ├── accuracy_comparison.png")
            print(f"│   │   └── learning_curves_comparison.png")
        
        return all_results
        
    except Exception as e:
        print(f"\nERROR: Experiment failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    results = main()
