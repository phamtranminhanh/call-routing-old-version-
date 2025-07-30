# --- multi_model_runner.py ---
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from pprint import pprint
import numpy as np
from evaluation_utils import evaluate_and_report, visualize_embeddings, plot_learning_curves, plot_roc_curve
from data_loader import *
from config import MODEL_REGISTRY, LABELED_RATIO
from active_learning import active_learning_cycle
from checkpoint_manager import save_checkpoint
import os
def sample_and_label(X, y, labeled_ratio=LABELED_RATIO):
    n_labeled = int(len(X) * labeled_ratio)
    indices = np.random.permutation(len(X))
    labeled_indices = indices[:n_labeled]
    unlabeled_indices = indices[n_labeled:]
    return X[labeled_indices], y[labeled_indices], X[unlabeled_indices], y[unlabeled_indices]

def train_model(X_train, y_train, model_class, model_params):
    model = model_class(**model_params)
    model.fit(X_train, y_train)
    return model

def run_single_oversampling_method(oversampling_method, data_loader_func, use_dynamic_threshold=False, base_dir=""):
    """Run a single oversampling method with all models"""
    import os
    
    # Create method-specific directory under base_dir
    method_dir = os.path.join(base_dir, oversampling_method) if base_dir else oversampling_method
    os.makedirs(method_dir, exist_ok=True)
    
    print(f"Loading data using {oversampling_method}...")
    X, y, feature_names, original_df, scaler = data_loader_func()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    learning_curves = {}
    total_epochs = 100

    for name, (model_class, model_params) in MODEL_REGISTRY.items():
        print(f"\nTraining {name} with {oversampling_method}...")
        
        X_labeled, y_labeled, X_unlabeled, y_unlabeled = sample_and_label(X_train, y_train)
        X_val, y_val = X_test, y_test
        
        model = train_model(X_labeled, y_labeled, model_class, model_params)
        
        losses = []
        val_accuracies = []
        
        for epoch in range(1, total_epochs + 1):
            model, X_labeled, y_labeled, loss, X_unlabeled, y_unlabeled = active_learning_cycle(
                model, X_labeled, y_labeled, X_unlabeled, y_unlabeled, X_val, y_val,
                epoch, total_epochs, use_dynamic_threshold
            )
            losses.append(loss)
            val_acc = model.score(X_val, y_val)
            val_accuracies.append(val_acc)
            
            if len(X_unlabeled) == 0:
                print(f"No more unlabeled data at epoch {epoch}")
                break
        
        final_acc = model.score(X_test, y_test)
        
        # Evaluate and get metrics - pass method_dir as oversampling_method
        metrics = evaluate_and_report(model, X_test, y_test, model_name=name, oversampling_method=method_dir)
        
        # Generate additional plots in the method directory
        y_pred = model.predict(X_test)
        visualize_embeddings(X_test, y_pred, method='pca', 
                           title=f"{name} Prediction Embedding", 
                           oversampling_method=method_dir, model_name=name)
        plot_learning_curves(losses, val_accuracies, 
                           oversampling_method=method_dir, 
                           model_name=name)
        plot_roc_curve(model, X_test, y_test, 
                      oversampling_method=method_dir, 
                      model_name=name)

        results[name] = {
            'final_acc': final_acc, 
            'losses': losses, 
            'val_accuracies': val_accuracies,
            'metrics': metrics
        }
        learning_curves[name] = losses

    # Save comparison plots for this oversampling method in method_dir
    save_comparison_plots(results, learning_curves, method_dir)
    
    return results

def save_comparison_plots(results, learning_curves, method_dir):
    """Save comparison plots for an oversampling method in the specified directory"""
    
    # Extract method name from path for titles
    method_name = os.path.basename(method_dir)
    
    # Plot accuracy bar chart
    plt.figure(figsize=(12, 6))
    models = list(results.keys())
    accuracies = [results[m]['final_acc'] for m in models]
    plt.bar(models, accuracies)
    plt.ylabel('Accuracy')
    plt.title(f'Model Performance Comparison - {method_name}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(method_dir, "accuracy_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot learning curves comparison
    plt.figure(figsize=(12, 6))
    for name, losses in learning_curves.items():
        plt.plot(range(1, len(losses)+1), losses, marker='o', label=name)
    plt.xlabel('Active Learning Iteration')
    plt.ylabel('Training Loss')
    plt.title(f'Training Loss Across Active Learning Iterations - {method_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(method_dir, "learning_curves_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

def run_all_models(use_dynamic_threshold=False, base_dir=""):
    """Run all models with all oversampling methods"""
    
    # Define oversampling methods
    oversampling_methods = {
        'No_Oversampling': load_call_quality_data,
        'SMOTE': load_call_quality_data_using_Smote,
        'ADASYN': load_call_quality_data_using_ADASYN,
        'KMeansSMOTE': load_call_quality_data_using_KMeansSMOTE,
        'SVMSMOTE': load_call_quality_data_using_SVMSMOTE
        'BorderlineSMOTE': load_call_quality_data_using_BORDERSMOTE,
    }
    
    all_results = {}
    
    for method_name, data_loader_func in oversampling_methods.items():
        print(f"\n{'='*60}")
        print(f"Starting {method_name}...")
        print(f"{'='*60}")
        
        try:
            results = run_single_oversampling_method(method_name, data_loader_func, use_dynamic_threshold, base_dir)
            if results:  # Only add if results are not empty
                all_results[method_name] = results
                print(f"✓ {method_name} completed successfully")
            else:
                print(f"✗ {method_name} returned empty results")
        except Exception as e:
            print(f"✗ Error with {method_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Only create comparison if we have results
    if all_results:
        print(f"\nCreating overall comparison with {len(all_results)} successful methods...")
        try:
            create_overall_comparison(all_results, base_dir)
        except Exception as e:
            print(f"Error creating overall comparison: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("No successful results to compare!")
    
    return all_results

def create_overall_comparison(all_results, base_dir=""):
    """Create comparison plots across all oversampling methods"""
    import os
    import pandas as pd
    import seaborn as sns
    
    # Check if we have any results
    if not all_results:
        print("No results to compare!")
        return
    
    # Create overall comparison directory under base_dir
    comparison_dir = os.path.join(base_dir, "overall_comparison") if base_dir else "overall_comparison"
    os.makedirs(comparison_dir, exist_ok=True)
    
    try:
        # Get the first non-empty result to extract model names
        first_result = None
        for method_results in all_results.values():
            if method_results:
                first_result = method_results
                break
        
        if not first_result:
            print("All results are empty!")
            return
            
        methods = list(all_results.keys())
        models = list(first_result.keys())
        
        print(f"Creating comparison for {len(methods)} methods and {len(models)} models")
        
        # Create comprehensive metrics comparison
        metrics_data = []
        for method in methods:
            for model in models:
                if model in all_results[method]:
                    result = all_results[method][model]
                    metrics = result.get('metrics', {})
                    
                    metrics_data.append({
                        'Oversampling_Method': method,
                        'Model': model,
                        'Accuracy': metrics.get('accuracy', result.get('final_acc', 0)),
                        'Precision': metrics.get('precision', 0),
                        'Recall': metrics.get('recall', 0),
                        'F1_Score': metrics.get('f1_score', 0)
                    })
        
        if not metrics_data:
            print("No metrics data found!")
            return
            
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(os.path.join(comparison_dir, "comprehensive_results.csv"), index=False)
        print("✓ Saved comprehensive results CSV")
        
        # Create accuracy comparison heatmap
        accuracy_pivot = metrics_df.pivot(index='Model', columns='Oversampling_Method', values='Accuracy')
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(accuracy_pivot, annot=True, cmap='Blues', fmt='.3f')
        plt.title('Accuracy Comparison Across Models and Oversampling Methods')
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, "accuracy_heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved accuracy heatmap")
        
        # Create metrics comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i//2, i%2]
            pivot_data = metrics_df.pivot(index='Model', columns='Oversampling_Method', values=metric)
            pivot_data.plot(kind='bar', ax=ax)
            ax.set_title(f'{metric} Comparison')
            ax.set_ylabel(metric)
            ax.legend(title='Oversampling Method', bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, "metrics_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved metrics comparison plot")
        
        print(f"Overall comparison completed successfully in {comparison_dir}!")
        
    except Exception as e:
        print(f"Error in create_overall_comparison: {e}")
        import traceback
        traceback.print_exc()

def run_all_models_old(use_dynamic_threshold=False):
    X, y, feature_names, original_df, scaler = load_call_quality_data_using_Smote()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    learning_curves = {}
    total_epochs = 100

    for name, (model_class, model_params) in MODEL_REGISTRY.items():
        print(f"\nTraining model: {name}")
        X_labeled, y_labeled, X_unlabeled, y_unlabeled = sample_and_label(X_train, y_train)
        model = train_model(X_labeled, y_labeled, model_class, model_params)

        losses = []
        for iteration in range(total_epochs):
            model, X_labeled, y_labeled, train_loss, X_unlabeled, y_unlabeled = active_learning_cycle(
                model, X_labeled, y_labeled, X_unlabeled, y_unlabeled,
                X_val=X_test, y_val=y_test,
                epoch=iteration + 1, total_epochs=total_epochs,
                use_dynamic_threshold=use_dynamic_threshold
            )
            losses.append(train_loss)  # collect training loss per iteration
            save_checkpoint(model, name, iteration)

        y_pred = model.predict(X_test)
        final_acc = accuracy_score(y_test, y_pred)

        evaluate_and_report(model, X_test, y_test, model_name=name)
        visualize_embeddings(X_test, y_pred, method='pca', title=f"{name} Prediction Embedding")
        plot_learning_curves(losses, [accuracy_score(y_test, y_pred)] * len(losses))
        plot_roc_curve(model, X_test, y_test)

        results[name] = {'final_acc': final_acc}
        learning_curves[name] = losses

    # Plot accuracy bar chart
    plt.figure(figsize=(12, 6))
    models = list(results.keys())
    accuracies = [results[m]['final_acc'] for m in models]
    plt.bar(models, accuracies)
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # New plot: Learning curves (training loss) per model
    plt.figure(figsize=(12, 6))
    for name, losses in learning_curves.items():
        plt.plot(range(1, len(losses)+1), losses, marker='o', label=name)
    plt.xlabel('Active Learning Iteration')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Across Active Learning Iterations')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(results)