"""
Laplace Instrument Classifier
==============================

A machine learning classifier that uses Laplace-domain features
(Prony decay, VQT spectral, Gammatone perceptual) to classify
musical instruments.

Trained on NSynth dataset (300k samples, 11 instrument families).

Author: Aurel (ColonyLab)
Date: November 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json
import warnings

# ML Libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Install with: pip install scikit-learn")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ClassifierConfig:
    """Configuration for the Laplace Classifier"""
    
    # Feature dimensions
    n_prony_features: int = 7
    n_vqt_features: int = 8
    n_gammatone_features: int = 11
    n_total_features: int = 26  # 7 + 8 + 11
    
    # NSynth classes
    n_classes: int = 11
    class_names: Tuple[str, ...] = (
        'bass', 'brass', 'flute', 'guitar', 'keyboard',
        'mallet', 'organ', 'reed', 'string', 'synth_lead', 'vocal'
    )
    
    # MIDI program mapping (NSynth family → General MIDI program)
    class_to_midi_program: Dict[str, int] = None
    
    # Model hyperparameters
    model_type: str = 'random_forest'  # 'random_forest', 'xgboost', 'gradient_boosting'
    
    # Random Forest params
    rf_n_estimators: int = 200
    rf_max_depth: int = 20
    rf_min_samples_split: int = 5
    rf_min_samples_leaf: int = 2
    
    # XGBoost params
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    
    # Training params
    test_size: float = 0.2
    random_state: int = 42
    use_scaler: bool = True
    
    def __post_init__(self):
        if self.class_to_midi_program is None:
            self.class_to_midi_program = {
                'bass': 32,       # Acoustic Bass
                'brass': 56,      # Trumpet
                'flute': 73,      # Flute
                'guitar': 24,     # Acoustic Guitar (nylon)
                'keyboard': 0,    # Acoustic Grand Piano
                'mallet': 11,     # Vibraphone
                'organ': 16,      # Drawbar Organ
                'reed': 65,       # Alto Sax
                'string': 40,     # Violin
                'synth_lead': 80, # Square Lead
                'vocal': 52       # Choir Aahs
            }


# Feature names for interpretability
FEATURE_NAMES = [
    # Prony features (7)
    'prony_mean_damping',
    'prony_std_damping', 
    'prony_median_damping',
    'prony_damping_range',
    'prony_mean_freq',
    'prony_freq_spread',
    'prony_spectral_centroid',
    # VQT features (8)
    'vqt_spectral_centroid',
    'vqt_spectral_spread',
    'vqt_spectral_skewness',
    'vqt_temporal_var_mean',
    'vqt_temporal_var_max',
    'vqt_harmonic_ratio',
    'vqt_phase_coherence',
    'vqt_damping_estimate',
    # Gammatone features (11)
    'gt_mean_attack_time',
    'gt_std_attack_time',
    'gt_mean_decay_time',
    'gt_std_decay_time',
    'gt_energy_centroid',
    'gt_energy_spread',
    'gt_low_energy_ratio',
    'gt_mid_energy_ratio',
    'gt_high_energy_ratio',
    'gt_onset_strength',
    'gt_envelope_flatness'
]


# =============================================================================
# MAIN CLASSIFIER CLASS
# =============================================================================

class LaplaceInstrumentClassifier:
    """
    Classifier for musical instruments based on Laplace-domain features.
    
    Features:
    - 7 Prony features (decay characteristics)
    - 8 VQT features (spectral characteristics)
    - 11 Gammatone features (perceptual/envelope)
    
    Usage:
        # Training
        classifier = LaplaceInstrumentClassifier()
        classifier.train(X_train, y_train)
        classifier.save('model.pkl')
        
        # Inference
        classifier = LaplaceInstrumentClassifier()
        classifier.load('model.pkl')
        result = classifier.predict(features)
    """
    
    def __init__(self, config: Optional[ClassifierConfig] = None):
        """
        Initialize the classifier.
        
        Args:
            config: Configuration object. If None, uses defaults.
        """
        self.config = config or ClassifierConfig()
        self.model = None
        self.scaler = StandardScaler() if self.config.use_scaler else None
        self.is_trained = False
        self.training_stats = {}
        
        # Initialize model based on config
        self._init_model()
    
    def _init_model(self):
        """Initialize the ML model based on config."""
        if self.config.model_type == 'random_forest':
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn required for RandomForest")
            self.model = RandomForestClassifier(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                min_samples_split=self.config.rf_min_samples_split,
                min_samples_leaf=self.config.rf_min_samples_leaf,
                random_state=self.config.random_state,
                n_jobs=-1,  # Use all cores
                class_weight='balanced'  # Handle class imbalance
            )
        
        elif self.config.model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost required. Install with: pip install xgboost")
            self.model = XGBClassifier(
                n_estimators=self.config.xgb_n_estimators,
                max_depth=self.config.xgb_max_depth,
                learning_rate=self.config.xgb_learning_rate,
                random_state=self.config.random_state,
                n_jobs=-1,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
        
        elif self.config.model_type == 'gradient_boosting':
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn required for GradientBoosting")
            self.model = GradientBoostingClassifier(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                learning_rate=0.1,
                random_state=self.config.random_state
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    # -------------------------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------------------------
    
    def train(self, 
              X: np.ndarray, 
              y: np.ndarray,
              validate: bool = True,
              verbose: bool = True) -> Dict:
        """
        Train the classifier.
        
        Args:
            X: Feature matrix, shape (N, 26)
            y: Labels, shape (N,) - integers 0-10 or string class names
            validate: Whether to perform train/val split
            verbose: Print training progress
            
        Returns:
            Dictionary with training statistics
        """
        # Validate input
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        
        if X.shape[1] != self.config.n_total_features:
            raise ValueError(f"Expected {self.config.n_total_features} features, got {X.shape[1]}")
        
        # Convert string labels to integers if needed
        if y.dtype.kind in ('U', 'S', 'O'):  # String types
            y = np.array([self.config.class_names.index(label) for label in y])
        
        if verbose:
            print(f"Training {self.config.model_type} classifier...")
            print(f"  Samples: {X.shape[0]}")
            print(f"  Features: {X.shape[1]}")
            print(f"  Classes: {self.config.n_classes}")
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Split data
        if validate:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        # Scale features
        if self.scaler is not None:
            X_train = self.scaler.fit_transform(X_train)
            if X_val is not None:
                X_val = self.scaler.transform(X_val)
        
        # Train model
        if verbose:
            print("  Training...")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Compute statistics
        train_acc = self.model.score(X_train, y_train)
        self.training_stats = {
            'train_accuracy': train_acc,
            'n_samples': X.shape[0],
            'n_train': X_train.shape[0],
            'model_type': self.config.model_type
        }
        
        if verbose:
            print(f"  Train accuracy: {train_acc:.2%}")
        
        # Validation
        if validate and X_val is not None:
            val_acc = self.model.score(X_val, y_val)
            y_pred = self.model.predict(X_val)
            val_f1 = f1_score(y_val, y_pred, average='weighted')
            
            self.training_stats['val_accuracy'] = val_acc
            self.training_stats['val_f1'] = val_f1
            self.training_stats['n_val'] = X_val.shape[0]
            
            if verbose:
                print(f"  Val accuracy: {val_acc:.2%}")
                print(f"  Val F1 (weighted): {val_f1:.2%}")
        
        # Feature importance (for tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            importance_dict = dict(zip(FEATURE_NAMES, importances))
            self.training_stats['feature_importance'] = importance_dict
            
            if verbose:
                print("\n  Top 5 most important features:")
                sorted_imp = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                for name, imp in sorted_imp[:5]:
                    print(f"    {name}: {imp:.3f}")
        
        return self.training_stats
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Labels
            cv: Number of folds
            
        Returns:
            Cross-validation results
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        
        # Convert string labels if needed
        if y.dtype.kind in ('U', 'S', 'O'):
            y = np.array([self.config.class_names.index(label) for label in y])
        
        # Scale features
        if self.scaler is not None:
            X = self.scaler.fit_transform(X)
        
        # Cross-validation
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        return {
            'cv_scores': scores.tolist(),
            'cv_mean': scores.mean(),
            'cv_std': scores.std()
        }
    
    # -------------------------------------------------------------------------
    # PREDICTION
    # -------------------------------------------------------------------------
    
    def predict(self, features: np.ndarray) -> Dict:
        """
        Predict instrument family for given features.
        
        Args:
            features: Feature vector(s), shape (26,) or (N, 26)
            
        Returns:
            Dictionary with:
            - label: Predicted class name
            - label_index: Predicted class index
            - confidence: Probability of predicted class
            - probabilities: Dict of all class probabilities
            - midi_program: Suggested MIDI program number
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        
        features = np.asarray(features, dtype=np.float32)
        
        # Handle single sample
        single_sample = features.ndim == 1
        if single_sample:
            features = features.reshape(1, -1)
        
        # Validate shape
        if features.shape[1] != self.config.n_total_features:
            raise ValueError(f"Expected {self.config.n_total_features} features, got {features.shape[1]}")
        
        # Handle missing values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        # Predict
        pred_indices = self.model.predict(features)
        pred_proba = self.model.predict_proba(features)
        
        # Format results
        results = []
        for i in range(len(pred_indices)):
            pred_idx = int(pred_indices[i])
            pred_label = self.config.class_names[pred_idx]
            confidence = float(pred_proba[i, pred_idx])
            
            result = {
                'label': pred_label,
                'label_index': pred_idx,
                'confidence': confidence,
                'probabilities': dict(zip(self.config.class_names, pred_proba[i].tolist())),
                'midi_program': self.config.class_to_midi_program[pred_label]
            }
            results.append(result)
        
        return results[0] if single_sample else results
    
    def predict_batch(self, features_list: List[np.ndarray]) -> List[Dict]:
        """
        Predict for multiple samples.
        
        Args:
            features_list: List of feature vectors
            
        Returns:
            List of prediction dictionaries
        """
        features = np.array(features_list)
        return self.predict(features)
    
    # -------------------------------------------------------------------------
    # EVALUATION
    # -------------------------------------------------------------------------
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> Dict:
        """
        Evaluate the classifier on test data.
        
        Args:
            X: Feature matrix
            y: True labels
            verbose: Print detailed report
            
        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        
        # Convert string labels if needed
        if y.dtype.kind in ('U', 'S', 'O'):
            y = np.array([self.config.class_names.index(label) for label in y])
        
        # Handle missing values and scale
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Predict
        y_pred = self.model.predict(X)
        
        # Metrics
        accuracy = (y_pred == y).mean()
        f1_weighted = f1_score(y, y_pred, average='weighted')
        f1_macro = f1_score(y, y_pred, average='macro')
        
        # Per-class metrics
        report = classification_report(
            y, y_pred,
            target_names=list(self.config.class_names),
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        results = {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        if verbose:
            print("\n" + "="*60)
            print("EVALUATION RESULTS")
            print("="*60)
            print(f"\nAccuracy: {accuracy:.2%}")
            print(f"F1 (weighted): {f1_weighted:.2%}")
            print(f"F1 (macro): {f1_macro:.2%}")
            print("\nPer-class metrics:")
            print(classification_report(y, y_pred, target_names=list(self.config.class_names)))
        
        return results
    
    # -------------------------------------------------------------------------
    # PERSISTENCE
    # -------------------------------------------------------------------------
    
    def save(self, path: Union[str, Path]):
        """
        Save the trained model to disk.
        
        Args:
            path: File path (will create .pkl file)
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model.")
        
        path = Path(path)
        
        save_dict = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'training_stats': self.training_stats,
            'feature_names': FEATURE_NAMES,
            'version': '1.0'
        }
        
        joblib.dump(save_dict, path)
        print(f"Model saved to: {path}")
    
    def load(self, path: Union[str, Path]):
        """
        Load a trained model from disk.
        
        Args:
            path: Path to saved model file
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        save_dict = joblib.load(path)
        
        self.model = save_dict['model']
        self.scaler = save_dict['scaler']
        self.config = save_dict['config']
        self.training_stats = save_dict.get('training_stats', {})
        self.is_trained = True
        
        print(f"Model loaded from: {path}")
        print(f"  Model type: {self.config.model_type}")
        if 'val_accuracy' in self.training_stats:
            print(f"  Val accuracy: {self.training_stats['val_accuracy']:.2%}")
    
    # -------------------------------------------------------------------------
    # UTILITIES
    # -------------------------------------------------------------------------
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance (for tree-based models)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained.")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError(f"Model type {self.config.model_type} doesn't support feature importance")
        
        return dict(zip(FEATURE_NAMES, self.model.feature_importances_))
    
    def label_to_midi_program(self, label: str) -> int:
        """Convert class label to MIDI program number."""
        return self.config.class_to_midi_program.get(label, 0)
    
    @property
    def classes(self) -> Tuple[str, ...]:
        """Get class names."""
        return self.config.class_names


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_classifier(model_type: str = 'random_forest', **kwargs) -> LaplaceInstrumentClassifier:
    """
    Factory function to create a classifier.
    
    Args:
        model_type: 'random_forest', 'xgboost', or 'gradient_boosting'
        **kwargs: Additional config parameters
        
    Returns:
        Configured classifier instance
    """
    config = ClassifierConfig(model_type=model_type, **kwargs)
    return LaplaceInstrumentClassifier(config)


def quick_train(X: np.ndarray, y: np.ndarray, model_type: str = 'random_forest') -> LaplaceInstrumentClassifier:
    """
    Quick training with default parameters.
    
    Args:
        X: Features
        y: Labels
        model_type: Model type
        
    Returns:
        Trained classifier
    """
    classifier = create_classifier(model_type)
    classifier.train(X, y)
    return classifier


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == '__main__':
    # Demo with synthetic data
    print("="*60)
    print("LAPLACE INSTRUMENT CLASSIFIER - DEMO")
    print("="*60)
    
    # Generate synthetic data (in real use, extract from NSynth)
    np.random.seed(42)
    n_samples = 1000
    n_features = 26
    n_classes = 11
    
    # Synthetic features
    X_demo = np.random.randn(n_samples, n_features)
    y_demo = np.random.randint(0, n_classes, n_samples)
    
    print("\n1. Creating classifier...")
    classifier = create_classifier('random_forest')
    
    print("\n2. Training...")
    stats = classifier.train(X_demo, y_demo, verbose=True)
    
    print("\n3. Predicting single sample...")
    test_features = np.random.randn(26)
    result = classifier.predict(test_features)
    print(f"  Predicted: {result['label']} (confidence: {result['confidence']:.2%})")
    print(f"  MIDI program: {result['midi_program']}")
    
    print("\n4. Cross-validation...")
    cv_results = classifier.cross_validate(X_demo, y_demo, cv=5)
    print(f"  CV accuracy: {cv_results['cv_mean']:.2%} ± {cv_results['cv_std']:.2%}")
    
    print("\n5. Saving model...")
    classifier.save('/tmp/laplace_classifier_demo.pkl')
    
    print("\n6. Loading model...")
    classifier2 = LaplaceInstrumentClassifier()
    classifier2.load('/tmp/laplace_classifier_demo.pkl')
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
