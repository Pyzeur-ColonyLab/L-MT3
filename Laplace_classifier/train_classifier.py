"""
Train Laplace Instrument Classifier
====================================

Complete training pipeline for the Laplace-based instrument classifier.

Steps:
1. Load extracted features from NSynth
2. Train classifier (RandomForest, XGBoost, or Neural Network)
3. Evaluate and save model

Usage:
    # Using pre-extracted features
    python train_classifier.py \
        --features nsynth_features.npz \
        --output laplace_classifier.pkl \
        --model_type random_forest

    # Full pipeline (extract + train)
    python train_classifier.py \
        --nsynth_dir ./nsynth-train \
        --output laplace_classifier.pkl \
        --model_type xgboost

Author: Aurel (ColonyLab)
Date: November 2025
"""

import numpy as np
import argparse
from pathlib import Path
import json
from typing import Dict, Optional, Tuple
import warnings

# Import our modules
from laplace_classifier import (
    LaplaceInstrumentClassifier,
    ClassifierConfig,
    create_classifier,
    FEATURE_NAMES
)

try:
    from extract_nsynth_features import extract_nsynth_features, LaplaceFeatureExtractor
    EXTRACTOR_AVAILABLE = True
except ImportError:
    EXTRACTOR_AVAILABLE = False

# Visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    warnings.warn("matplotlib/seaborn not available for plotting")


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def load_features(features_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load pre-extracted features from .npz file.
    
    Args:
        features_path: Path to .npz file
        
    Returns:
        Tuple of (features, labels)
    """
    data = np.load(features_path)
    features = data['features']
    labels = data['labels']
    
    print(f"Loaded features: {features.shape}")
    print(f"Loaded labels: {labels.shape}")
    
    return features, labels


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = 'random_forest',
    output_path: Optional[str] = None,
    plot_results: bool = True,
    verbose: bool = True
) -> Tuple[LaplaceInstrumentClassifier, Dict]:
    """
    Train classifier and evaluate.
    
    Args:
        X: Feature matrix (N, 26)
        y: Labels (N,)
        model_type: 'random_forest', 'xgboost', or 'gradient_boosting'
        output_path: Path to save model
        plot_results: Generate visualizations
        verbose: Print progress
        
    Returns:
        Tuple of (trained classifier, results dictionary)
    """
    print("="*60)
    print("TRAINING LAPLACE INSTRUMENT CLASSIFIER")
    print("="*60)
    
    # Create classifier
    if verbose:
        print(f"\n1. Creating {model_type} classifier...")
    
    classifier = create_classifier(model_type)
    
    # Train
    if verbose:
        print("\n2. Training...")
    
    train_stats = classifier.train(X, y, validate=True, verbose=verbose)
    
    # Cross-validation
    if verbose:
        print("\n3. Cross-validation (5-fold)...")
    
    cv_results = classifier.cross_validate(X, y, cv=5)
    
    print(f"   CV Accuracy: {cv_results['cv_mean']:.2%} ± {cv_results['cv_std']:.2%}")
    
    # Full evaluation
    if verbose:
        print("\n4. Full evaluation...")
    
    eval_results = classifier.evaluate(X, y, verbose=verbose)
    
    # Combine results
    results = {
        'training': train_stats,
        'cross_validation': cv_results,
        'evaluation': eval_results
    }
    
    # Plot results
    if plot_results and PLOTTING_AVAILABLE:
        if verbose:
            print("\n5. Generating visualizations...")
        
        plot_training_results(classifier, eval_results, output_path)
    
    # Save model
    if output_path:
        if verbose:
            print(f"\n6. Saving model to {output_path}...")
        
        classifier.save(output_path)
        
        # Also save results as JSON
        results_path = Path(output_path).with_suffix('.json')
        
        # Convert numpy arrays to lists for JSON serialization
        results_json = {
            'training': {
                k: v for k, v in train_stats.items() 
                if not isinstance(v, (np.ndarray, dict))
            },
            'cross_validation': cv_results,
            'evaluation': {
                'accuracy': eval_results['accuracy'],
                'f1_weighted': eval_results['f1_weighted'],
                'f1_macro': eval_results['f1_macro']
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"   Results saved to {results_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    return classifier, results


def plot_training_results(
    classifier: LaplaceInstrumentClassifier,
    eval_results: Dict,
    output_path: Optional[str] = None
):
    """
    Generate visualizations of training results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    class_names = list(classifier.config.class_names)
    
    # 1. Confusion Matrix
    ax1 = axes[0, 0]
    cm = np.array(eval_results['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title('Confusion Matrix')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    plt.setp(ax1.get_yticklabels(), rotation=0)
    
    # 2. Per-class F1 scores
    ax2 = axes[0, 1]
    report = eval_results['classification_report']
    f1_scores = [report[name]['f1-score'] for name in class_names]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(class_names)))
    bars = ax2.barh(class_names, f1_scores, color=colors)
    ax2.set_xlabel('F1 Score')
    ax2.set_title('Per-Class F1 Scores')
    ax2.set_xlim(0, 1)
    for bar, score in zip(bars, f1_scores):
        ax2.text(score + 0.02, bar.get_y() + bar.get_height()/2,
                 f'{score:.2f}', va='center')
    
    # 3. Feature Importance
    ax3 = axes[1, 0]
    if hasattr(classifier.model, 'feature_importances_'):
        importances = classifier.model.feature_importances_
        sorted_idx = np.argsort(importances)
        
        # Show top 15 features
        top_n = 15
        top_idx = sorted_idx[-top_n:]
        
        ax3.barh(range(top_n), importances[top_idx], color='steelblue')
        ax3.set_yticks(range(top_n))
        ax3.set_yticklabels([FEATURE_NAMES[i] for i in top_idx])
        ax3.set_xlabel('Importance')
        ax3.set_title(f'Top {top_n} Feature Importances')
    else:
        ax3.text(0.5, 0.5, 'Feature importance\nnot available',
                 ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Feature Importance')
    
    # 4. Summary metrics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    TRAINING SUMMARY
    ================
    
    Model: {classifier.config.model_type}
    
    Overall Metrics:
    • Accuracy: {eval_results['accuracy']:.2%}
    • F1 (weighted): {eval_results['f1_weighted']:.2%}
    • F1 (macro): {eval_results['f1_macro']:.2%}
    
    Best Classes:
    • {class_names[np.argmax(f1_scores)]}: {max(f1_scores):.2%}
    
    Worst Classes:
    • {class_names[np.argmin(f1_scores)]}: {min(f1_scores):.2%}
    
    Features: 26 (7 Prony + 8 VQT + 11 Gammatone)
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_path:
        plot_path = Path(output_path).with_suffix('.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"   Plot saved to {plot_path}")
    
    plt.show()


# =============================================================================
# COMPARISON OF MODELS
# =============================================================================

def compare_models(X: np.ndarray, y: np.ndarray) -> Dict:
    """
    Compare different model types.
    
    Args:
        X: Features
        y: Labels
        
    Returns:
        Comparison results
    """
    print("="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    model_types = ['random_forest', 'gradient_boosting']
    
    try:
        from xgboost import XGBClassifier
        model_types.append('xgboost')
    except ImportError:
        print("XGBoost not available, skipping...")
    
    results = {}
    
    for model_type in model_types:
        print(f"\n--- {model_type.upper()} ---")
        
        classifier = create_classifier(model_type)
        cv_results = classifier.cross_validate(X, y, cv=5)
        
        print(f"CV Accuracy: {cv_results['cv_mean']:.2%} ± {cv_results['cv_std']:.2%}")
        
        results[model_type] = {
            'cv_mean': cv_results['cv_mean'],
            'cv_std': cv_results['cv_std'],
            'cv_scores': cv_results['cv_scores']
        }
    
    # Find best model
    best_model = max(results.keys(), key=lambda k: results[k]['cv_mean'])
    print(f"\n🏆 Best model: {best_model} ({results[best_model]['cv_mean']:.2%})")
    
    return results


# =============================================================================
# QUICK DEMO WITH SYNTHETIC DATA
# =============================================================================

def demo_with_synthetic_data():
    """
    Demo the training pipeline with synthetic data.
    """
    print("="*60)
    print("DEMO WITH SYNTHETIC DATA")
    print("="*60)
    
    # Generate synthetic features
    np.random.seed(42)
    n_samples = 5000
    n_features = 26
    n_classes = 11
    
    # Create class-specific feature distributions
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)
    
    samples_per_class = n_samples // n_classes
    
    for class_idx in range(n_classes):
        start_idx = class_idx * samples_per_class
        end_idx = start_idx + samples_per_class
        
        # Each class has different mean features
        class_mean = np.random.randn(n_features) * (class_idx + 1)
        class_std = np.abs(np.random.randn(n_features)) + 0.5
        
        X[start_idx:end_idx] = np.random.randn(samples_per_class, n_features) * class_std + class_mean
        y[start_idx:end_idx] = class_idx
    
    print(f"Generated {n_samples} synthetic samples")
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    # Train and evaluate
    classifier, results = train_and_evaluate(
        X, y,
        model_type='random_forest',
        output_path=None,
        plot_results=PLOTTING_AVAILABLE,
        verbose=True
    )
    
    # Test prediction
    print("\n--- Testing Prediction ---")
    test_features = np.random.randn(26)
    result = classifier.predict(test_features)
    print(f"Predicted: {result['label']} (confidence: {result['confidence']:.2%})")
    
    return classifier


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train Laplace Instrument Classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train from pre-extracted features
    python train_classifier.py --features nsynth_features.npz --output model.pkl

    # Train with XGBoost
    python train_classifier.py --features nsynth_features.npz --output model.pkl --model_type xgboost

    # Extract features and train (full pipeline)
    python train_classifier.py --nsynth_dir ./nsynth-train --output model.pkl

    # Compare different models
    python train_classifier.py --features nsynth_features.npz --compare

    # Demo with synthetic data
    python train_classifier.py --demo
        """
    )
    
    parser.add_argument('--features', type=str,
                        help='Path to pre-extracted features (.npz)')
    parser.add_argument('--nsynth_dir', type=str,
                        help='Path to NSynth directory (will extract features)')
    parser.add_argument('--output', type=str, default='laplace_classifier.pkl',
                        help='Output path for trained model')
    parser.add_argument('--model_type', type=str, default='random_forest',
                        choices=['random_forest', 'xgboost', 'gradient_boosting'],
                        help='Model type to train')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to use for training')
    parser.add_argument('--compare', action='store_true',
                        help='Compare different model types')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo with synthetic data')
    parser.add_argument('--no_plot', action='store_true',
                        help='Disable plotting')
    
    args = parser.parse_args()
    
    # Demo mode
    if args.demo:
        demo_with_synthetic_data()
        return
    
    # Load or extract features
    if args.features:
        X, y = load_features(args.features)
    elif args.nsynth_dir:
        if not EXTRACTOR_AVAILABLE:
            raise ImportError("Feature extractor not available")
        
        # First extract features
        features_path = Path(args.nsynth_dir).parent / 'nsynth_features.npz'
        X, y = extract_nsynth_features(
            args.nsynth_dir,
            str(features_path),
            max_samples=args.max_samples
        )
    else:
        print("Error: Must provide either --features or --nsynth_dir")
        print("       Or use --demo for synthetic data demo")
        return
    
    # Limit samples if specified
    if args.max_samples and len(X) > args.max_samples:
        indices = np.random.choice(len(X), args.max_samples, replace=False)
        X = X[indices]
        y = y[indices]
        print(f"Using {len(X)} samples")
    
    # Compare models
    if args.compare:
        compare_models(X, y)
        return
    
    # Train
    train_and_evaluate(
        X, y,
        model_type=args.model_type,
        output_path=args.output,
        plot_results=not args.no_plot,
        verbose=True
    )


if __name__ == '__main__':
    main()
