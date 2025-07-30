import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
from pathlib import Path
import numpy as np

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def parse_classification_report(file_path):
    """Parse a classification report text file and extract metrics"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Extract basic info
        model_match = re.search(r'Model: (.+)', content)
        oversampling_match = re.search(r'Oversampling Method: (.+)', content)
        timestamp_match = re.search(r'Timestamp: (.+)', content)
        
        # Extract summary metrics
        accuracy_match = re.search(r'Accuracy: ([\d.]+)', content)
        precision_match = re.search(r'Precision \(weighted\): ([\d.]+)', content)
        recall_match = re.search(r'Recall \(weighted\): ([\d.]+)', content)
        f1_match = re.search(r'F1-Score \(weighted\): ([\d.]+)', content)
        
        return {
            'model': model_match.group(1) if model_match else 'unknown',
            'oversampling_method': oversampling_match.group(1) if oversampling_match else 'unknown',
            'timestamp': timestamp_match.group(1) if timestamp_match else 'unknown',
            'accuracy': float(accuracy_match.group(1)) if accuracy_match else 0.0,
            'precision': float(precision_match.group(1)) if precision_match else 0.0,
            'recall': float(recall_match.group(1)) if recall_match else 0.0,
            'f1_score': float(f1_match.group(1)) if f1_match else 0.0,
            'file_path': file_path
        }
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def read_all_reports_from_directory(base_dir="results_static_threshold"):
    """Read all classification reports from the results directory"""
    all_results = []
    
    # Find all classification report files
    pattern = os.path.join(base_dir, "*", "classification_report_*.txt")
    report_files = glob.glob(pattern)
    
    print(f"Found {len(report_files)} classification report files in {base_dir}")
    
    for file_path in report_files:
        result = parse_classification_report(file_path)
        if result:
            # Extract oversampling method from folder path
            path_parts = Path(file_path).parts
            folder_name = path_parts[-2]  # Get folder name (oversampling method)
            result['oversampling_method'] = folder_name
            all_results.append(result)
            print(f"✓ Parsed: {folder_name}/{result['model']}")
    
    return pd.DataFrame(all_results)

# Read all results
print("=== READING CLASSIFICATION REPORTS ===")
df_results = read_all_reports_from_directory("results_static_threshold")

print(f"\n=== SUMMARY ===")
print(f"Total results loaded: {len(df_results)}")
print(f"Oversampling methods: {sorted(df_results['oversampling_method'].unique())}")
print(f"Models: {sorted(df_results['model'].unique())}")

# Display the data
print("\nFirst few rows:")
display(df_results.head())