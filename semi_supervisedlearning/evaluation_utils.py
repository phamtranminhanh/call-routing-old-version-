# --- evaluation_utils.py ---
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc

def plot_confidence_distribution(confidences, title="Confidence Histogram"):
    plt.figure(figsize=(8, 4))
    plt.hist(confidences, bins=20, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel("Confidence")
    plt.ylabel("Sample Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_embeddings(X, y, method='pca', title="2D Embedding", oversampling_method=None, model_name=None):
    if method == 'pca':
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)

    X_reduced = reducer.fit_transform(X)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=y, palette='tab10', s=30, alpha=0.8)
    plt.title(f"{title} using {method.upper()}")
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    
    if oversampling_method and model_name:
        import os
        os.makedirs(oversampling_method, exist_ok=True)
        plt.savefig(f"{oversampling_method}/embedding_{model_name}_{method}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def evaluate_and_report(model, X_test, y_test, model_name=None, oversampling_method=None):
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nComprehensive Metrics for {model_name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}")
    
    print("\nDetailed Classification Report:")
    class_report = classification_report(y_test, y_pred)
    print(class_report)
    
    # Use model class name if no model_name provided
    if model_name is None:
        model_name = model.__class__.__name__
    
    # Create directory structure
    import os
    if oversampling_method:
        os.makedirs(oversampling_method, exist_ok=True)
        log_dir = oversampling_method
    else:
        os.makedirs("logs_smote", exist_ok=True)
        log_dir = "logs_smote"

    # Save detailed report
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f"{log_dir}/classification_report_{model_name}_{timestamp}.txt"
    metrics_filename = f"{log_dir}/metrics_{model_name}_{timestamp}.csv"
    
    with open(report_filename, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Oversampling Method: {oversampling_method}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Summary Metrics:\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision (weighted): {precision:.4f}\n")
        f.write(f"Recall (weighted): {recall:.4f}\n")
        f.write(f"F1-Score (weighted): {f1:.4f}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(class_report)
    
    # Save metrics as CSV
    import pandas as pd
    metrics_df = pd.DataFrame({
        'Model': [model_name],
        'Oversampling_Method': [oversampling_method],
        'Accuracy': [accuracy],
        'Precision_Weighted': [precision],
        'Recall_Weighted': [recall],
        'F1_Score_Weighted': [f1],
        'Timestamp': [timestamp]
    })
    metrics_df.to_csv(metrics_filename, index=False)
        
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    
    if oversampling_method:
        plt.savefig(f"{log_dir}/confusion_matrix_{model_name}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'model_name': model_name,
        'oversampling_method': oversampling_method
    }

def plot_learning_curves(train_losses, val_accuracies, oversampling_method=None, model_name=None):
    epochs = range(1, len(train_losses) + 1)
    
    fig, ax1 = plt.subplots(figsize=(8,5))

    # Plot training loss on left y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(epochs, train_losses, marker='o', color=color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Create second y-axis for validation accuracy
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Validation Accuracy', color=color)
    ax2.plot(epochs, val_accuracies, marker='s', linestyle='--', color=color, label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    # Title and legend
    title = f'Learning Curves - {model_name}' if model_name else 'Learning Curves'
    plt.title(title)
    fig.tight_layout()
    
    # Combine legends of both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='best')

    if oversampling_method and model_name:
        import os
        os.makedirs(oversampling_method, exist_ok=True)
        plt.savefig(f"{oversampling_method}/learning_curves_{model_name}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_roc_curve(model, X_test, y_test, oversampling_method=None, model_name=None):
    from sklearn.preprocessing import label_binarize

    y_score = model.predict_proba(X_test)
    classes = np.unique(y_test)
    y_test_binarized = label_binarize(y_test, classes=classes)
    plt.figure(figsize=(8,6))

    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {c} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    title = f'Multi-class ROC curve - {model_name}' if model_name else 'Multi-class ROC curve'
    plt.title(title)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    
    if oversampling_method and model_name:
        import os
        os.makedirs(oversampling_method, exist_ok=True)
        plt.savefig(f"{oversampling_method}/roc_curve_{model_name}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()