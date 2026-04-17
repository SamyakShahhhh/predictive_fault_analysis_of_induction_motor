# Motor Fault Classification using Machine Learning

A machine learning project that classifies faults in three-phase induction motors using electrical and mechanical features extracted from motor signals.

---

## Table of Contents

- [Theory](#theory)
- [Fault Types](#fault-types)
- [Project Structure](#project-structure)
- [Features Used](#features-used)
- [Pipeline Stages](#pipeline-stages)
- [Models Trained](#models-trained)
- [Requirements](#requirements)
- [Setup and Usage](#setup-and-usage)
- [Output](#output)

---

## Theory

### Why Motor Fault Detection?

Three-phase induction motors are widely used in industrial applications. Unexpected motor failures can cause production downtime and safety hazards. Early fault detection using machine learning helps prevent such failures by analyzing patterns in motor signals before a complete breakdown occurs.

### How it Works

Motor faults change the electrical and mechanical behavior of a motor. For example:
- A broken rotor bar causes asymmetry in current waveforms
- Inter-turn short circuits (ITSC) increase winding temperature and distort current
- Voltage unbalance causes unequal current draw across phases

By extracting statistical and frequency-domain features from motor current, torque, and speed signals, a classifier can be trained to distinguish between healthy and faulty operating conditions.

### Feature Extraction

Raw time-domain signals are not directly fed into the classifier. Instead, meaningful features are computed:
- **Statistical features** — RMS, standard deviation, kurtosis, skewness, peak, crest factor, shape factor
- **Frequency features** — dominant frequency of phase current
- **Power features** — power factor, slip, voltage unbalance, current unbalance
- **Mechanical features** — torque ripple, speed ripple, torque P2P

---

## Fault Types

| Label | Fault | Description |
|-------|-------|-------------|
| 0 | Healthy | Normal motor operation |
| 1 | RotorEcc | Rotor eccentricity — uneven air gap between rotor and stator |
| 2 | BrokenBar | One or more broken rotor bars causing torque pulsations |
| 3 | ITSC | Inter-Turn Short Circuit — partial short in stator windings |
| 4 | P2P | Phase-to-Phase fault — electrical short between two phases |
| 5 | VoltageUnbal | Unbalanced supply voltage causing unequal phase currents |
| 6 | SuddenLoad | Abrupt change in mechanical load on the motor shaft |

---

## Project Structure

```
motor_fault_project/
│
├── main.py              # Entry point — runs all 6 stages in sequence
├── config.py            # All constants: paths, feature names, fault labels
├── data_loader.py       # Stage 1 & 2: load CSVs and assign fault_type labels
├── visualize.py         # Stage 3.1: exploratory data analysis plots
├── preprocess.py        # Stage 3.2: scaling and feature selection
├── train.py             # Stage 4: train all 6 models
├── evaluate.py          # Stage 5: accuracy, confusion matrix, F1, CV scores
├── predict.py           # Stage 6: predict fault on new input samples
│
└── plots/               # All saved charts (auto-created on first run)
```

---

## Features Used

17 features are extracted per sample:

| Feature | Description |
|---------|-------------|
| ia_rms | RMS value of phase-A current |
| ia_std | Standard deviation of phase-A current |
| ia_kurtosis | Kurtosis — measures impulsive spikes in current |
| ia_peak | Peak amplitude of phase-A current |
| ia_crest | Crest factor = peak / RMS |
| ia_domFreq | Dominant frequency in phase-A current spectrum |
| current_unbalance | Difference in RMS across three phases |
| te_mean | Mean electromagnetic torque |
| te_std | Standard deviation of torque |
| te_p2p | Peak-to-peak torque variation |
| torque_ripple | Periodic fluctuation in torque |
| speed_ripple | Fluctuation in rotor speed |
| ia_skewness | Asymmetry in the current distribution |
| power_factor | Ratio of real power to apparent power |
| slip | Difference between synchronous and actual rotor speed |
| ia_shape_factor | RMS / mean absolute value of current |
| voltage_unbalance | Degree of unbalance in supply voltage |

---

## Pipeline Stages

### Stage 1 — Load Data
Reads all 7 CSV files from the dataset directory and validates column order.

### Stage 2 — Combine and Label
Merges all files into one DataFrame. Assigns integer `fault_type` labels (0–6). Removes duplicate rows and shuffles the data.

### Stage 3.1 — Visualisation
Generates exploratory plots: class distribution, feature distributions, correlation heatmap, and box plots per fault type.

### Stage 3.2 — Preprocessing
- Splits data into 80% train / 20% test
- Applies `StandardScaler` to normalize features
- Uses `SelectKBest` (ANOVA F-test) to select the top 10 most discriminative features

### Stage 4 — Model Training
Trains 6 classifiers on the preprocessed training data.

### Stage 5 — Evaluation
For each model:
- Test accuracy
- Classification report (precision, recall, F1 per class)
- Confusion matrix heatmap (saved as PNG)
- 5-Fold Stratified Cross-Validation score
- Per-class F1 bar chart (saved as PNG)
- Final model comparison chart

### Stage 6 — Prediction Demo
Picks the best-performing model and runs it on 3 random samples from the test set to demonstrate single-sample prediction.

---

## Models Trained

| Model | Notes |
|-------|-------|
| Logistic Regression | Linear baseline classifier |
| K-Nearest Neighbours (KNN) | Distance-based, k=5 |
| Decision Tree | Depth limited to 10 to reduce overfitting |
| Support Vector Machine (SVM) | RBF kernel, one-vs-rest |
| Random Forest | 100 trees, class-weight balanced |
| XGBoost | Gradient boosted trees, multi-class softmax |

---

## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
```

---

## Setup and Usage

### 1. Clone the repository

```bash
git clone https://github.com/your-username/motor_fault_project.git
cd motor_fault_project
```

### 2. Place the dataset

Download or copy your dataset CSVs into a folder. Then open `config.py` and update the `DATA_DIR` path to point to that folder:

```python
DATA_DIR = r"C:\your\path\to\dataset"
```

The following CSV files are expected:

```
healthy_features_cleaned.csv
RotorEcc_clause.csv
RotorBroke_features_balanced.csv
itsc_features_Claude.csv
P_P_FAULT_claude.csv
Voltage_unbal_claude.csv
SuddenLoad_claude.csv
```

### 3. Run the pipeline

```bash
python main.py
```

The pipeline will execute all 6 stages automatically and print progress to the terminal. All plots are saved to the `plots/` folder.

---

## Output

After a successful run you will find:

- **Terminal output** — stage-by-stage logs, accuracy scores, classification reports, and CV scores
- **plots/** — the following PNG files:
  - Class distribution bar chart
  - Feature correlation heatmap
  - Box plots per fault type
  - Confusion matrix for each of the 6 models
  - Per-class F1 score chart for each model
  - Model comparison accuracy chart
