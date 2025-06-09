"""
Analysis template for assembly vs environment experiments.

This script provides example code for analyzing experiment results and
creating visualizations for model comparisons.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))
from experiment_tracker import ExperimentTracker


def analyze_experiment(experiment_name: str):
    """Comprehensive analysis of an experiment."""
    
    print(f"Analyzing experiment: {experiment_name}")
    
    # Load experiment
    tracker = ExperimentTracker(experiment_name)
    
    # Get experiment summary
    summary = tracker.get_experiment_summary()
    print(f"Models trained: {list(summary['metadata']['models'].keys())}")
    
    # Load all predictions
    try:
        combined = tracker.combine_all_predictions()
        print(f"Total predictions: {len(combined)}")
        print(f"Prediction columns: {[col for col in combined.columns if col.startswith('pred_')]}")
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return
    
    # Evaluate models
    eval_results = tracker.evaluate_models(['mse', 'mae', 'r2', 'pearson'])
    
    # Print performance summary
    print("\n=== Model Performance (Mean ± Std) ===")
    summary_stats = eval_results.groupby('model').agg({
        'mse': ['mean', 'std'],
        'mae': ['mean', 'std'],
        'r2': ['mean', 'std'],
        'pearson': ['mean', 'std']
    }).round(4)
    
    for model in summary_stats.index:
        print(f"\n{model.upper()}:")
        print(f"  MSE: {summary_stats.loc[model, ('mse', 'mean')]:.4f} ± {summary_stats.loc[model, ('mse', 'std')]:.4f}")
        print(f"  MAE: {summary_stats.loc[model, ('mae', 'mean')]:.4f} ± {summary_stats.loc[model, ('mae', 'std')]:.4f}")
        print(f"  R²:  {summary_stats.loc[model, ('r2', 'mean')]:.4f} ± {summary_stats.loc[model, ('r2', 'std')]:.4f}")
        print(f"  ρ:   {summary_stats.loc[model, ('pearson', 'mean')]:.4f} ± {summary_stats.loc[model, ('pearson', 'std')]:.4f}")
    
    return tracker, combined, eval_results


def create_comparison_plots(combined: pd.DataFrame, save_dir: Path = None):
    """Create comparison plots for model predictions."""
    
    if save_dir is None:
        save_dir = Path("plots")
    save_dir.mkdir(exist_ok=True)
    
    # Find prediction columns
    pred_cols = [col for col in combined.columns if col.startswith('pred_')]
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Prediction vs Truth scatter plots
    fig, axes = plt.subplots(1, len(pred_cols), figsize=(5*len(pred_cols), 4))
    if len(pred_cols) == 1:
        axes = [axes]
    
    for i, pred_col in enumerate(pred_cols):
        model_name = pred_col.replace('pred_', '').upper()
        
        # Remove NaN values
        mask = ~(np.isnan(combined[pred_col]) | np.isnan(combined['target']))
        x = combined.loc[mask, 'target']
        y = combined.loc[mask, pred_col]
        
        # Scatter plot
        axes[i].scatter(x, y, alpha=0.5, s=1)
        
        # Perfect prediction line
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        # Calculate R²
        r2 = stats.pearsonr(x, y)[0]**2
        axes[i].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[i].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        axes[i].set_xlabel('True log(M*/M☉)')
        axes[i].set_ylabel(f'{model_name} Prediction')
        axes[i].set_title(f'{model_name} vs Truth')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / "prediction_vs_truth.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Residuals comparison
    fig, axes = plt.subplots(1, len(pred_cols), figsize=(5*len(pred_cols), 4))
    if len(pred_cols) == 1:
        axes = [axes]
    
    for i, pred_col in enumerate(pred_cols):
        model_name = pred_col.replace('pred_', '').upper()
        
        # Calculate residuals
        mask = ~(np.isnan(combined[pred_col]) | np.isnan(combined['target']))
        residuals = combined.loc[mask, pred_col] - combined.loc[mask, 'target']
        
        # Histogram of residuals
        axes[i].hist(residuals, bins=50, alpha=0.7, density=True)
        
        # Statistics
        mean_res = residuals.mean()
        std_res = residuals.std()
        axes[i].axvline(mean_res, color='red', linestyle='--', 
                       label=f'Mean: {mean_res:.3f}')
        axes[i].text(0.05, 0.95, f'σ = {std_res:.3f}', transform=axes[i].transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        axes[i].set_xlabel('Residuals (Pred - True)')
        axes[i].set_ylabel('Density')
        axes[i].set_title(f'{model_name} Residuals')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / "residuals_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Model comparison (if multiple models)
    if len(pred_cols) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Box plot of R² scores by fold
        eval_data = []
        for pred_col in pred_cols:
            model_name = pred_col.replace('pred_', '')
            for fold in combined['fold'].unique():
                fold_data = combined[combined['fold'] == fold]
                mask = ~(np.isnan(fold_data[pred_col]) | np.isnan(fold_data['target']))
                if mask.sum() > 0:
                    r2 = stats.pearsonr(fold_data.loc[mask, pred_col], 
                                      fold_data.loc[mask, 'target'])[0]**2
                    eval_data.append({'model': model_name, 'fold': fold, 'r2': r2})
        
        eval_df = pd.DataFrame(eval_data)
        sns.boxplot(data=eval_df, x='model', y='r2', ax=axes[0])
        axes[0].set_title('R² by Model')
        axes[0].set_ylabel('R²')
        
        # Direct model comparison scatter
        if len(pred_cols) == 2:
            model1, model2 = pred_cols[0], pred_cols[1]
            mask = ~(np.isnan(combined[model1]) | np.isnan(combined[model2]))
            
            axes[1].scatter(combined.loc[mask, model1], combined.loc[mask, model2], alpha=0.5, s=1)
            
            # Perfect agreement line
            min_val = min(combined.loc[mask, model1].min(), combined.loc[mask, model2].min())
            max_val = max(combined.loc[mask, model1].max(), combined.loc[mask, model2].max())
            axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
            
            axes[1].set_xlabel(f'{model1.replace("pred_", "").upper()} Prediction')
            axes[1].set_ylabel(f'{model2.replace("pred_", "").upper()} Prediction')
            axes[1].set_title('Model Predictions Comparison')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()


def analyze_feature_importance(combined: pd.DataFrame, feature_cols: List[str] = None):
    """Analyze which features correlate most with prediction errors."""
    
    if feature_cols is None:
        # Default feature columns
        feature_cols = [col for col in combined.columns 
                       if any(x in col.lower() for x in ['halo', 'vmax', 'mass', 'central', 'overdensity'])]
    
    pred_cols = [col for col in combined.columns if col.startswith('pred_')]
    
    print("=== Feature Correlation with Prediction Errors ===")
    
    for pred_col in pred_cols:
        model_name = pred_col.replace('pred_', '').upper()
        print(f"\n{model_name}:")
        
        # Calculate residuals
        mask = ~(np.isnan(combined[pred_col]) | np.isnan(combined['target']))
        residuals = combined.loc[mask, pred_col] - combined.loc[mask, 'target']
        
        # Calculate correlations with features
        correlations = {}
        for feature in feature_cols:
            if feature in combined.columns:
                feature_mask = mask & ~np.isnan(combined[feature])
                if feature_mask.sum() > 10:  # Need enough data points
                    corr = stats.pearsonr(
                        combined.loc[feature_mask, feature], 
                        residuals[feature_mask]
                    )[0]
                    correlations[feature] = corr
        
        # Sort by absolute correlation
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for feature, corr in sorted_corr[:5]:  # Top 5
            print(f"  {feature}: {corr:.3f}")


def example_usage():
    """Example of how to use the analysis functions."""
    
    # Replace with your actual experiment name
    experiment_name = "comparison_2024"
    
    # Run analysis
    tracker, combined, eval_results = analyze_experiment(experiment_name)
    
    # Create plots
    create_comparison_plots(combined, tracker.exp_dir / "plots")
    
    # Analyze feature importance
    analyze_feature_importance(combined)
    
    # Example of accessing specific data
    print("\n=== Example Data Access ===")
    print(f"Available columns: {combined.columns.tolist()}")
    print(f"Shape: {combined.shape}")
    print(f"Folds: {sorted(combined['fold'].unique())}")
    
    # Example: Get specific model predictions for fold 0
    fold_0 = combined[combined['fold'] == 0]
    if 'pred_env_gnn' in fold_0.columns:
        env_gnn_predictions = fold_0['pred_env_gnn'].values
        targets = fold_0['target'].values
        print(f"Fold 0 Environment GNN R²: {stats.pearsonr(env_gnn_predictions, targets)[0]**2:.4f}")


if __name__ == "__main__":
    # Run example analysis
    # You can modify this to analyze your specific experiments
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    args = parser.parse_args()
    
    tracker, combined, eval_results = analyze_experiment(args.experiment)
    create_comparison_plots(combined, tracker.exp_dir / "plots")
    analyze_feature_importance(combined)