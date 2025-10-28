
# Modeling Human Activity States Using Hidden Markov Models

## Project Overview

This project implements a **Hidden Markov Model (HMM)** for **human activity recognition** using smartphone sensor data. The system processes accelerometer and gyroscope signals to classify activities (standing, walking, jumping, still) by modeling them as hidden states that generate observable sensor feature sequences.

## Key Features

- **Multi-sensor fusion**: Combines accelerometer and gyroscope data
- **Comprehensive feature extraction**: 163 time and frequency-domain features
- **Sequential modeling**: HMM captures temporal dependencies between activities
- **Real-time capability**: Sliding window approach for continuous classification
- **High accuracy**: 95.45% overall accuracy on test data

## Objectives

* Collect and preprocess real-world sequential motion data from smartphone sensors
* Extract meaningful time and frequency-domain features using sliding windows
* Train a Gaussian HMM with supervised initialization followed by Baum-Welch refinement
* Evaluate model performance using Viterbi decoding on unseen data
* Analyze transition probabilities between activity states
* Deploy the model for real-time activity prediction

---

## 1. Data Collection & Preprocessing

### Sensor Data Sources
Data is recorded using motion logging apps (**Sensor Logger** or **Physics Toolbox Accelerometer**) with the following specifications:

**Sensors Used:**
- **Accelerometer** (linear acceleration: x, y, z, total)
- **Gyroscope** (angular velocity: x, y, z)

**Activities Recorded:**
| Activity | Duration | Samples | Description |
|----------|----------|---------|-------------|
| Standing | 17.82s | 8,910 | Phone steady at waist level |
| Walking | 24.26s | 12,129 | Normal walking pace |
| Jumping | 18.60s | 9,298 | Continuous jumping |
| Still | 12.48s | 6,241 | Phone on flat surface |

### Data Preprocessing Pipeline
```python
1. File Combination: Merge multiple recording sessions per activity
2. Time Alignment: Synchronize accelerometer and gyroscope timestamps
3. Resampling: Standardize to 500Hz sampling rate
4. Interpolation: Create common time axis for sensor fusion
5. Harmonization: Ensure consistent column naming and structure
```

### File Structure
```
/content/
├── standing/
│   ├── linear standing 1.csv
│   └── standing 1.csv
├── walking/
│   ├── linear walking 1.csv
│   └── walking 1.csv
└── [activity]_combined_harmonized.csv (output files)
```

---

## 2. Feature Extraction Engine

### Sliding Window Configuration
- **Window size**: 2.0 seconds (1000 samples at 500Hz)
- **Overlap**: 50% (500 sample step size)
- **Total windows**: 67 across all activities

### Feature Categories (163 total features)

#### Time-Domain Features (per axis)
- Statistical: Mean, STD, Variance, Min, Max, Range, Median
- Distribution: Q25, Q75, IQR, Skewness, Kurtosis
- Signal Properties: RMS, Energy, Zero-Crossing Rate

#### Frequency-Domain Features (per axis)
- Spectral Analysis: Dominant Frequency, Dominant Magnitude
- Band Energy: Low (<2Hz), Mid (2-5Hz), High (>5Hz)
- Spectral Properties: Spectral Energy, Spectral Entropy, Spectral Centroid

#### Multi-axis Features
- Signal Magnitude Area (SMA)
- Inter-axis correlations (XY, XZ, YZ)
- Magnitude statistics (Mean, STD, Max, Min, Range)
- Motion intensity metrics

---

## 3. HMM Architecture

### Model Components
| Element | Specification | Implementation |
|---------|---------------|----------------|
| **Hidden States** | 4 activities: standing, walking, jumping, still | `n_components=4` |
| **Observations** | 163-dimensional feature vectors | Gaussian emissions |
| **Transition Matrix** | Learned via Baum-Welch | 4×4 probability matrix |
| **Emission Probabilities** | Multivariate Gaussian | Diagonal covariance |
| **Initial Probabilities** | Data-driven initialization | From training distribution |

### Training Strategy
```python
1. Supervised Initialization:
   - Means: Calculated from labeled training data
   - Covariances: Variance per feature + regularization
   - Transition Matrix: 85% self-transition prior
   - Initial Probabilities: Empirical distribution

2. Baum-Welch Refinement:
   - Iterations: 200 maximum
   - Convergence: Monitor log-likelihood
   - Parameters: Update means, covariances, transitions
```

### Key Algorithms
- **Viterbi Algorithm**: Most likely state sequence decoding
- **Forward-Backward Algorithm**: Probability calculations
- **Baum-Welch Algorithm**: Parameter estimation

---

## 4. Model Implementation

### Dependencies
```python
hmmlearn==0.3.3
numpy>=1.10
scikit-learn>=0.16
scipy>=0.19
pandas
matplotlib
seaborn
```

### Core Classes
1. **`ActivityFeatureExtractor`**: Handles windowing and feature extraction
2. **`ActivityHMM`**: Manages HMM training, evaluation, and visualization
3. **Data preprocessing functions**: Combine and harmonize sensor data

### Training Process
```python
# Data splitting: 70% train, 30% test per activity
standing: Train=11, Test=5 windows
walking:  Train=16, Test=7 windows  
jumping:  Train=11, Test=6 windows
still:    Train=7,  Test=4 windows
```

---

## 5. Evaluation Results

### Performance Metrics
| Activity | Samples | Sensitivity | Specificity | Overall Accuracy |
|----------|---------|-------------|-------------|------------------|
| Jumping  | 6       | 1.0000      | 1.0000      | 0.9545 |
| Standing | 5       | 1.0000      | 0.9412      | 0.9545 |
| Still    | 4       | 1.0000      | 1.0000      | 0.9545 |
| Walking  | 7       | 0.8571      | 1.0000      | 0.9545 |

### Transition Matrix Analysis
```
From         → To           | Probability
---------------------------------------------
jumping      → jumping      | 1.0000
standing     → standing     | 0.9091
still        → still        | 1.0000
walking      → walking      | 1.0000
```

### Real-world Testing
- **Unseen data**: 19.28 seconds walking activity
- **Prediction**: WALKING with 94.4% confidence
- **Windows analyzed**: 18 windows
- **Breakdown**: 17 walking, 1 standing

---

## 6. Visualization Suite

The project includes comprehensive visualizations:

1. **Sensor Data Plots**: Raw and processed accelerometer/gyroscope signals
2. **Transition Matrix Heatmap**: State transition probabilities
3. **Initial Probabilities**: Bar chart of starting state likelihoods
4. **Confusion Matrix**: Classification performance
5. **Activity Sequences**: Predicted vs. true state progression over time
6. **Real-time Prediction**: Window-by-window activity classification

---

## 7. Usage Guide

### Training New Model
```python
# Load and preprocess data
sequences = load_activity_sequences(base_path="/content", 
                                  activities=["standing", "walking", "jumping", "still"])

# Extract features
extractor = ActivityFeatureExtractor(window_size=2.0, overlap=0.5, sampling_rate=500)
activity_features = extractor.process_sequences(sequences)

# Train HMM
hmm_model = ActivityHMM(n_iter=200)
hmm_model.train(train_features)
```

### Predicting New Data
```python
# Prepare test data
test_df = combine_and_prepare_test_data(accel_csv, gyro_csv, resample_rate=500)

# Predict activity
predicted_activity, pred_sequence, confidence = predict_activity(
    hmm_model, test_df, extractor
)
```

---

## 8. Analysis and Insights

### Key Findings
- **Best classified**: Jumping and Still (100% sensitivity/specificity)
- **Most challenging**: Walking (85.7% sensitivity due to occasional misclassification as standing)
- **Temporal consistency**: High self-transition probabilities reflect real activity persistence
- **Feature importance**: Multi-axis correlations and frequency features significantly improve discrimination

### Limitations & Improvements
1. **Data scarcity**: Limited to 67 total windows - collect more diverse data
2. **Activity complexity**: Add transitions between activities for more realistic modeling
3. **Feature selection**: Implement automated feature selection to reduce dimensionality
4. **Personalization**: User-specific model adaptation
5. **Real-time optimization**: Reduce computational complexity for mobile deployment

### Practical Applications
- Fitness tracking and workout monitoring
- Elderly fall detection and activity monitoring
- Healthcare rehabilitation progress tracking
- Smartphone-based context awareness

---

## Repository Structure
```
HMM-Activity-Model/
├── data/                   # Raw and processed sensor data
├── notebooks/              # Jupyter notebooks with full implementation
├── src/
│   ├── preprocessing.py    # Data combination and harmonization
│   ├── feature_extraction.py # Sliding window and feature computation
│   ├── hmm_model.py        # HMM training and evaluation
│   └── visualization.py    # Plotting and results display
├── models/                 # Saved trained models
├── results/                # Evaluation metrics and confusion matrices
├── tests/                  # Test data for validation
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── report.pdf              # Final comprehensive report
```

---

## Submission Checklist

- [ ] Well-labeled dataset files (`.csv`) with consistent formatting
- [ ] Complete Python implementation (`.ipynb` or `.py` files)
- [ ] Trained HMM model with saved parameters
- [ ] Comprehensive evaluation results and visualizations
- [ ] Test predictions on unseen data with confidence scores
- [ ] 4-5 page report including:
  - Background & motivation
  - Data collection and preprocessing methodology
  - HMM architecture and training procedure
  - Results with quantitative metrics and visualizations
  - Discussion of limitations and future work
  - Conclusion and practical implications

---

## References

1. Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition.
2. Mannini, A., & Sabatini, A. M. (2010). Machine learning methods for classifying human physical activity from on-body accelerometers.
3. hmmlearn documentation: https://hmmlearn.readthedocs.io
