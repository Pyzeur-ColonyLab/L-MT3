# 🎯 Guide de Développement du Laplace Classifier

## Vue d'Ensemble

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DÉVELOPPEMENT DU CLASSIFIER                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ÉTAPE 1: Télécharger NSynth (300k samples)                         │
│           ↓                                                          │
│  ÉTAPE 2: Extraire 26 features Laplace par sample                   │
│           ↓                                                          │
│  ÉTAPE 3: Entraîner le modèle (RandomForest/XGBoost)                │
│           ↓                                                          │
│  ÉTAPE 4: Évaluer et sauvegarder                                    │
│           ↓                                                          │
│  ÉTAPE 5: Intégrer dans le pipeline MR-MT3                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📦 Fichiers Fournis

| Fichier | Description | Taille |
|---------|-------------|--------|
| `laplace_classifier.py` | Classe principale du classifier | ~15 KB |
| `extract_nsynth_features.py` | Extraction des 26 features | ~20 KB |
| `train_classifier.py` | Script d'entraînement | ~10 KB |

---

## 🚀 Quick Start (5 minutes)

### Option A: Demo avec données synthétiques

```bash
# Tester le pipeline sans NSynth
python train_classifier.py --demo
```

**Sortie attendue:**
```
DEMO WITH SYNTHETIC DATA
========================
Generated 5000 synthetic samples
Training random_forest classifier...
  Train accuracy: 98.75%
  Val accuracy: 85.20%
CV Accuracy: 84.50% ± 2.30%
```

### Option B: Entraînement complet avec NSynth

```bash
# 1. Télécharger NSynth (~25GB)
wget https://storage.googleapis.com/magentadata/datasets/nsynth/nsynth-train.jsonwav.tar.gz
tar -xzf nsynth-train.jsonwav.tar.gz

# 2. Extraire les features (2-3 heures sur CPU)
python extract_nsynth_features.py \
    --nsynth_dir ./nsynth-train \
    --output ./nsynth_features.npz \
    --n_workers 8

# 3. Entraîner le classifier (5-10 minutes)
python train_classifier.py \
    --features ./nsynth_features.npz \
    --output ./laplace_classifier.pkl \
    --model_type random_forest
```

---

## 📊 Architecture du Classifier

### Input: 26 Features

```python
features = np.array([
    # ═══════════ PRONY (7) - Decay ═══════════
    -2500.0,    # [0]  prony_mean_damping
    800.0,      # [1]  prony_std_damping
    -2300.0,    # [2]  prony_median_damping
    3500.0,     # [3]  prony_damping_range
    440.0,      # [4]  prony_mean_freq
    150.0,      # [5]  prony_freq_spread
    520.0,      # [6]  prony_spectral_centroid
    
    # ═══════════ VQT (8) - Spectral ═══════════
    1200.0,     # [7]  vqt_spectral_centroid
    450.0,      # [8]  vqt_spectral_spread
    0.35,       # [9]  vqt_spectral_skewness
    0.12,       # [10] vqt_temporal_var_mean
    0.45,       # [11] vqt_temporal_var_max
    2.5,        # [12] vqt_harmonic_ratio
    0.78,       # [13] vqt_phase_coherence
    -1500.0,    # [14] vqt_damping_estimate
    
    # ═══════════ GAMMATONE (11) - Perceptual ═══════════
    0.015,      # [15] gt_mean_attack_time
    0.008,      # [16] gt_std_attack_time
    0.25,       # [17] gt_mean_decay_time
    0.12,       # [18] gt_std_decay_time
    0.45,       # [19] gt_energy_centroid
    0.22,       # [20] gt_energy_spread
    0.30,       # [21] gt_low_energy_ratio
    0.45,       # [22] gt_mid_energy_ratio
    0.25,       # [23] gt_high_energy_ratio
    0.85,       # [24] gt_onset_strength
    0.62,       # [25] gt_envelope_flatness
])
```

### Output: Prediction

```python
result = {
    'label': 'keyboard',           # Nom de la famille
    'label_index': 4,              # Index (0-10)
    'confidence': 0.87,            # Probabilité
    'midi_program': 0,             # Programme MIDI suggéré
    'probabilities': {             # Distribution complète
        'bass': 0.02,
        'brass': 0.01,
        'flute': 0.03,
        'guitar': 0.04,
        'keyboard': 0.87,  # ← Plus haute
        'mallet': 0.01,
        'organ': 0.01,
        'reed': 0.005,
        'string': 0.005,
        'synth_lead': 0.005,
        'vocal': 0.005
    }
}
```

---

## 🔧 Utilisation du Classifier

### Entraînement

```python
from laplace_classifier import LaplaceInstrumentClassifier
import numpy as np

# Charger les features extraites
data = np.load('nsynth_features.npz')
X = data['features']  # (N, 26)
y = data['labels']    # (N,)

# Créer et entraîner
classifier = LaplaceInstrumentClassifier()
stats = classifier.train(X, y, validate=True, verbose=True)

# Sauvegarder
classifier.save('laplace_classifier.pkl')
```

### Inférence

```python
from laplace_classifier import LaplaceInstrumentClassifier
from extract_nsynth_features import LaplaceFeatureExtractor
import librosa

# Charger le modèle entraîné
classifier = LaplaceInstrumentClassifier()
classifier.load('laplace_classifier.pkl')

# Charger un audio
audio, sr = librosa.load('note.wav', sr=16000)

# Extraire les features
extractor = LaplaceFeatureExtractor()
features = extractor.extract(audio, sr)  # (26,)

# Prédire
result = classifier.predict(features)
print(f"Instrument: {result['label']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"MIDI Program: {result['midi_program']}")
```

### Intégration avec MR-MT3

```python
# Dans phase1_mrmt3_enhancement.py

class LaplaceEnhancedMRMT3:
    def __init__(self, classifier_path: str):
        # Charger le classifier entraîné
        self.classifier = LaplaceInstrumentClassifier()
        self.classifier.load(classifier_path)
        self.extractor = LaplaceFeatureExtractor()
    
    def enhance_midi(self, raw_midi, audio, sr):
        """
        Améliorer le MIDI avec le classifier.
        """
        for instrument in raw_midi.instruments:
            # Extraire l'audio de cet instrument
            inst_audio = self.extract_instrument_audio(instrument, audio, sr)
            
            # Extraire les features
            features = self.extractor.extract(inst_audio, sr)
            
            # Prédire l'instrument
            result = self.classifier.predict(features)
            
            # Mettre à jour si confiance suffisante
            if result['confidence'] > 0.7:
                instrument.program = result['midi_program']
        
        return raw_midi
```

---

## 📈 Résultats Attendus

### Avec NSynth (300k samples)

| Modèle | Accuracy | F1 (weighted) | Temps entraînement |
|--------|----------|---------------|-------------------|
| RandomForest | 85-90% | 84-89% | 5-10 min |
| XGBoost | 88-92% | 87-91% | 10-15 min |
| GradientBoosting | 82-87% | 81-86% | 20-30 min |

### Features les plus importantes (typiquement)

1. `vqt_spectral_centroid` - Brillance
2. `prony_mean_damping` - Taux de décroissance
3. `gt_mean_attack_time` - Temps d'attaque
4. `vqt_harmonic_ratio` - Harmoniques vs bruit
5. `gt_energy_centroid` - Distribution d'énergie

---

## 🛠️ Commandes Utiles

### Extraction des features

```bash
# Full dataset (~3 heures)
python extract_nsynth_features.py \
    --nsynth_dir ./nsynth-train \
    --output ./features_train.npz \
    --n_workers 8

# Test rapide (1000 samples, ~5 min)
python extract_nsynth_features.py \
    --nsynth_dir ./nsynth-train \
    --output ./features_test.npz \
    --max_samples 1000 \
    --n_workers 4
```

### Entraînement

```bash
# RandomForest (recommandé pour commencer)
python train_classifier.py \
    --features ./features_train.npz \
    --output ./model_rf.pkl \
    --model_type random_forest

# XGBoost (meilleure accuracy)
python train_classifier.py \
    --features ./features_train.npz \
    --output ./model_xgb.pkl \
    --model_type xgboost

# Comparer tous les modèles
python train_classifier.py \
    --features ./features_train.npz \
    --compare
```

### Évaluation

```bash
# Évaluer sur le test set
python -c "
from laplace_classifier import LaplaceInstrumentClassifier
import numpy as np

# Charger
classifier = LaplaceInstrumentClassifier()
classifier.load('laplace_classifier.pkl')

# Évaluer
data = np.load('features_test.npz')
results = classifier.evaluate(data['features'], data['labels'])
"
```

---

## ⚠️ Troubleshooting

### Erreur: "ModuleNotFoundError: No module named 'librosa'"

```bash
pip install librosa
```

### Erreur: "Memory error" pendant l'extraction

```bash
# Réduire le nombre de workers
python extract_nsynth_features.py --n_workers 2

# Ou traiter par batches
python extract_nsynth_features.py --max_samples 50000
```

### Erreur: "XGBoost not available"

```bash
pip install xgboost
```

### Accuracy faible (<70%)

1. Vérifier que les features sont correctement extraites
2. Augmenter le nombre de samples d'entraînement
3. Ajuster les hyperparamètres du modèle:

```python
config = ClassifierConfig(
    rf_n_estimators=500,      # Plus d'arbres
    rf_max_depth=30,          # Plus profond
    rf_min_samples_leaf=1     # Moins de régularisation
)
classifier = LaplaceInstrumentClassifier(config)
```

---

## 📋 Checklist de Développement

- [ ] Installer les dépendances (`pip install librosa scikit-learn xgboost`)
- [ ] Télécharger NSynth (~25GB)
- [ ] Tester avec `--demo` d'abord
- [ ] Extraire les features (2-3h)
- [ ] Entraîner le modèle (5-10 min)
- [ ] Évaluer (accuracy > 85%)
- [ ] Intégrer dans le pipeline MR-MT3
- [ ] Tester end-to-end sur un fichier audio

---

## 📚 Fichiers de Sortie

Après l'entraînement, vous aurez:

```
./
├── nsynth_features.npz      # Features extraites (peut être réutilisé)
├── laplace_classifier.pkl   # Modèle entraîné
├── laplace_classifier.json  # Métriques d'entraînement
└── laplace_classifier.png   # Visualisations (confusion matrix, etc.)
```

---

## 🎯 Next Steps

1. **Phase 2A**: Entraîner sur NSynth complet → 85-90% accuracy
2. **Phase 2B**: Fine-tune sur Slakh2100 stems → Adaptation au domaine
3. **Phase 3**: Intégrer dans architecture end-to-end MR-MT3
