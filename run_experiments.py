#!/usr/bin/env python3
"""
Main script for running assembly vs environment experiments.

This script demonstrates how to use the experiment tracking system for
reproducible model comparisons and analysis.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from train_with_tracking import run_comparison_experiment, run_residual_experiment
from experiment_tracker import ExperimentTracker


def main():
    parser = argparse.ArgumentParser(description="Run assembly vs environment experiments")
    parser.add_argument("--experiment", type=str, required=True,
                       help="Name of the experiment")
    parser.add_argument("--models", nargs="+", default=["mlp", "env_gnn", "merger_gnn"],
                       choices=["mlp", "env_gnn", "merger_gnn"],
                       help="Models to train")
    parser.add_argument("--residual", action="store_true",
                       help="Run residual experiment after main experiment")
    parser.add_argument("--residual-only", action="store_true",
                       help="Only run residual experiment using existing base model")
    parser.add_argument("--base-model", type=str, default="env_gnn",
                       help="Base model for residual learning")
    parser.add_argument("--evaluate-only", action="store_true",
                       help="Only evaluate existing experiment results")
    
    args = parser.parse_args()
    
    if args.evaluate_only:
        # Just evaluate existing results
        tracker = ExperimentTracker(args.experiment)
        eval_results = tracker.evaluate_models()
        print("Evaluation complete!")
        
        # Print summary table separated by target type
        print("\n=== Model Performance Summary ===")
        
        # Group by both model and target_type for meaningful averages
        if 'target_type' in eval_results.columns:
            for target_type in eval_results['target_type'].unique():
                print(f"\n--- {target_type} predictions ---")
                target_results = eval_results[eval_results['target_type'] == target_type]
                summary_stats = target_results.groupby('model').agg({
                    'mse': ['mean', 'std'],
                    'mae': ['mean', 'std'], 
                    'r2': ['mean', 'std'],
                    'pearson': ['mean', 'std']
                }).round(4)
                print(summary_stats)
        else:
            # Fallback for single-output case
            summary_stats = eval_results.groupby('model').agg({
                'mse': ['mean', 'std'],
                'mae': ['mean', 'std'], 
                'r2': ['mean', 'std'],
                'pearson': ['mean', 'std']
            }).round(4)
            print(summary_stats)
        
    elif args.residual_only:
        # Run only residual experiment using existing base model
        print(f"Running residual-only experiment using base model: {args.base_model}")
        residual_tracker = run_residual_experiment(
            args.experiment, 
            args.base_model, 
            ["merger_gnn"]
        )
        print(f"\nResidual experiment complete! Results saved in: {residual_tracker.exp_dir}")
        
    else:
        # Run full comparison experiment
        print(f"Running comparison experiment: {args.experiment}")
        print(f"Models: {args.models}")
        
        tracker = run_comparison_experiment(args.experiment, args.models)
        
        # Optionally run residual experiment
        if args.residual and "env_gnn" in args.models:
            print(f"\nRunning residual experiment with base model: {args.base_model}")
            residual_tracker = run_residual_experiment(
                args.experiment, 
                args.base_model, 
                ["merger_gnn"]
            )
        
        print(f"\nExperiment complete! Results saved in: {tracker.exp_dir}")
        print("\nTo analyze results, you can:")
        print(f"1. Load predictions: tracker = ExperimentTracker('{args.experiment}')")
        print("2. Get combined predictions: combined = tracker.combine_all_predictions()")
        print("3. Evaluate models: results = tracker.evaluate_models()")


if __name__ == "__main__":
    main()