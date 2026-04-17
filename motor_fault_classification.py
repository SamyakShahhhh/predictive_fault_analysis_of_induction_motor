# =============================================================================
# Motor Fault Classification — Complete ML Pipeline
# BTech Project | Stages 1 through 6
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd

# matplotlib.use('Agg') MUST be called before importing pyplot
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — saves plots to files without needing a display

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# =============================================================================
# PATHS — edit DATA_DIR to point at your CSV folder if you move things
# =============================================================================
DATA_DIR  = r"C:\Users\Swayam\Downloads\claude_dataset"
PLOTS_DIR = r"C:\Users\Swayam\Downloads\motor_fault_project\plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Fixed reference column order — every loaded file will be reordered to this
FEATURE_COLS = [
    'ia_rms', 'ia_std', 'ia_kurtosis', 'ia_peak', 'ia_crest',
    'ia_domFreq', 'current_unbalance', 'te_mean', 'te_std', 'te_p2p',
    'torque_ripple', 'speed_ripple', 'ia_skewness', 'power_factor',
    'slip', 'ia_shape_factor', 'voltage_unbalance'
]
REFERENCE_COLS = FEATURE_COLS + ['label']   # expected final column order

# Human-readable names for each fault_type integer
FAULT_NAMES = {
    0: 'Healthy',
    1: 'RotorEcc',
    2: 'BrokenBar',
    3: 'ITSC',
    4: 'P2P',
    5: 'VoltageUnbal',
    6: 'SuddenLoad'
}
TARGET_NAMES = list(FAULT_NAMES.values())   # used in classification report


# =============================================================================
# STAGE 1 — LOAD DATA
# =============================================================================

def load_files(file_map):
    """
    Load every CSV listed in file_map and reorder columns to REFERENCE_COLS.

    Parameters
    ----------
    file_map : dict  { filename (str) : fault_label_name (str) }
               Adding a new file in future only requires one new entry here.

    Returns
    -------
    dict  { fault_label_name : DataFrame }
    """
    loaded = {}

    for filename, label_name in file_map.items():
        filepath = os.path.join(DATA_DIR, filename)

        try:
            df = pd.read_csv(filepath)

            # Reorder columns to the fixed reference order so every file is consistent
            # This is critical for P_P_FAULT_claude.csv which has a different column order
            df = df[REFERENCE_COLS]

            print(f"\n--- {label_name} ({filename}) ---")
            print(f"    Shape : {df.shape}")
            print(df.head())

            loaded[label_name] = df

        except Exception as e:
            print(f"[ERROR] Could not load {filename}: {e}")

    return loaded


# Map from CSV filename → internal label name
FILE_MAP = {
    'healthy_features_cleaned.csv'    : 'healthy',
    'RotorEcc_clause.csv'             : 'rotor_ecc',
    'RotorBroke_features_balanced.csv': 'broken_bar',
    'itsc_features_Claude.csv'        : 'itsc',
    'P_P_FAULT_claude.csv'            : 'p2p',
    'Voltage_unbal_claude.csv'        : 'voltage_unbal',
    'SuddenLoad_claude.csv'           : 'sudden_load',
}

print("=" * 60)
print("STAGE 1 — LOADING DATA")
print("=" * 60)

try:
    dataframes = load_files(FILE_MAP)
    print("\n[OK] All files loaded successfully.")
except Exception as e:
    print(f"[ERROR] Stage 1 failed: {e}")
    dataframes = {}


# =============================================================================
# STAGE 2 — COMBINE & LABEL
# =============================================================================

print("\n" + "=" * 60)
print("STAGE 2 — COMBINING AND LABELLING")
print("=" * 60)

try:
    # fault_type integer assigned to each file
    FAULT_LABEL_MAP = {
        'rotor_ecc'   : 1,
        'broken_bar'  : 2,
        'itsc'        : 3,
        'p2p'         : 4,
        'voltage_unbal': 5,
        'sudden_load' : 6,
    }

    all_pieces = []   # we will collect individual DataFrames and concatenate once

    # --- Healthy rows from each FAULT file (label == 1 means healthy in binary files) ---
    healthy_from_faults = []

    for label_name, fault_type_id in FAULT_LABEL_MAP.items():
        df = dataframes[label_name].copy()

        # Rows where binary label == 0 are the actual fault samples
        fault_rows = df[df['label'] == 0].copy()
        fault_rows['fault_type'] = fault_type_id

        # Rows where binary label == 1 are healthy samples embedded in the fault file
        healthy_rows = df[df['label'] == 1].copy()
        healthy_rows['fault_type'] = 0

        all_pieces.append(fault_rows)
        healthy_from_faults.append(healthy_rows)

    # --- All rows from the dedicated healthy file ---
    healthy_df = dataframes['healthy'].copy()
    healthy_df['fault_type'] = 0
    all_pieces.append(healthy_df)

    # --- Add healthy rows collected from fault files ---
    for h in healthy_from_faults:
        all_pieces.append(h)

    # Drop the original binary label column — we now use fault_type
    for i, piece in enumerate(all_pieces):
        all_pieces[i] = piece.drop(columns=['label'])

    # Combine everything into one DataFrame
    df_final = pd.concat(all_pieces, ignore_index=True)

    # Shuffle to remove any ordering bias — important before train/test split
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

    # --- Class distribution ---
    print("\nClass distribution (fault_type counts):")
    dist = df_final['fault_type'].value_counts().sort_index()
    for ft, count in dist.items():
        print(f"    {ft} — {FAULT_NAMES[ft]:<15} : {count}")

    # --- Duplicate check ---
    dup_count = df_final.duplicated().sum()
    print(f"\nDuplicate rows found : {dup_count}")
    if dup_count > 0:
        df_final = df_final.drop_duplicates().reset_index(drop=True)
        print(f"Duplicates removed. Rows remaining : {len(df_final)}")

    print(f"\nFinal df_final shape : {df_final.shape}")
    print("[OK] Stage 2 complete.")

except Exception as e:
    print(f"[ERROR] Stage 2 failed: {e}")


# =============================================================================
# STAGE 3.1 — DATA VISUALIZATION
# =============================================================================

print("\n" + "=" * 60)
print("STAGE 3.1 — DATA VISUALIZATION")
print("=" * 60)

# Helper: save figure and close cleanly
def save_plot(filename):
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    Saved → {path}")


# ---- Plot 1: Class Distribution Bar Chart ----
try:
    print("\n[Plot 1] Class distribution bar chart...")
    counts = df_final['fault_type'].value_counts().sort_index()
    labels = [FAULT_NAMES[i] for i in counts.index]

    plt.figure(figsize=(10, 5))
    plt.bar(labels, counts.values, color='steelblue', edgecolor='black')
    plt.title('Class Distribution — Fault Types')
    plt.xlabel('Fault Type')
    plt.ylabel('Sample Count')
    plt.xticks(rotation=15)
    plt.tight_layout()
    save_plot('01_class_distribution.png')

except Exception as e:
    print(f"    [ERROR] Plot 1 failed: {e}")


# ---- Plot 2: Box Plots per Feature grouped by fault_type ----
try:
    print("[Plot 2] Box plots for each feature grouped by fault_type...")
    n_cols = 3
    n_rows = 6   # ceil(17 / 3)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 24))
    axes = axes.flatten()

    for idx, feature in enumerate(FEATURE_COLS):
        ax = axes[idx]
        # Group data by fault_type and collect values for each group
        groups = [df_final[df_final['fault_type'] == ft][feature].values
                  for ft in sorted(FAULT_NAMES.keys())]
        ax.boxplot(groups, labels=list(FAULT_NAMES.values()), patch_artist=True)
        ax.set_title(feature, fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=7)

    # Hide any empty subplot axes (we have 18 slots, only need 17)
    for idx in range(len(FEATURE_COLS), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Box Plots — Features Grouped by Fault Type', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_plot('02_boxplots_per_feature.png')

except Exception as e:
    print(f"    [ERROR] Plot 2 failed: {e}")


# ---- Plot 3: Correlation Heatmap ----
try:
    print("[Plot 3] Correlation heatmap...")
    corr_matrix = df_final[FEATURE_COLS].corr()

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        linewidths=0.5,
        annot_kws={'size': 7}
    )
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    save_plot('03_correlation_heatmap.png')

except Exception as e:
    print(f"    [ERROR] Plot 3 failed: {e}")


# ---- Plot 4: Feature Distribution Histograms by fault_type ----
try:
    print("[Plot 4] Feature distribution histograms...")
    n_cols = 3
    n_rows = 6
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 24))
    axes = axes.flatten()

    # colormaps API changed in matplotlib 3.7+ — use matplotlib.colormaps instead of plt.cm.get_cmap
    cmap = matplotlib.colormaps.get_cmap('tab10')

    for idx, feature in enumerate(FEATURE_COLS):
        ax = axes[idx]
        for ft in sorted(FAULT_NAMES.keys()):
            subset = df_final[df_final['fault_type'] == ft][feature]
            ax.hist(subset, bins=30, alpha=0.5,
                    label=FAULT_NAMES[ft], color=cmap(ft / 9))
        ax.set_title(feature, fontsize=9)
        ax.legend(fontsize=6)

    for idx in range(len(FEATURE_COLS), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Feature Distributions by Fault Type', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    save_plot('04_feature_histograms.png')

except Exception as e:
    print(f"    [ERROR] Plot 4 failed: {e}")


# ---- Plot 5: Outlier Count Bar Chart (Z-Score method) ----
try:
    print("[Plot 5] Outlier count bar chart (Z-score ±3)...")
    outlier_counts = []

    for feature in FEATURE_COLS:
        z_scores = np.abs(stats.zscore(df_final[feature].dropna()))
        count = (z_scores > 3).sum()
        outlier_counts.append(count)

    plt.figure(figsize=(14, 5))
    plt.bar(FEATURE_COLS, outlier_counts, color='tomato', edgecolor='black')
    plt.title('Outlier Count per Feature (Z-Score |z| > 3)')
    plt.xlabel('Feature')
    plt.ylabel('Number of Outliers')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_plot('05_outlier_counts_zscore.png')

except Exception as e:
    print(f"    [ERROR] Plot 5 failed: {e}")


# ---- Plot 6: Pairplot of Top 5 Most Correlated Features ----
try:
    print("[Plot 6] Pairplot of top 5 most correlated features...")
    # Find 5 features with highest average absolute correlation to all others
    corr_matrix = df_final[FEATURE_COLS].corr().abs()
    avg_corr = corr_matrix.mean().sort_values(ascending=False)
    top5_features = avg_corr.head(5).index.tolist()
    print(f"    Top 5 features selected: {top5_features}")

    pairplot_df = df_final[top5_features + ['fault_type']].copy()
    pairplot_df['fault_type'] = pairplot_df['fault_type'].astype(str)

    pair_fig = sns.pairplot(pairplot_df, hue='fault_type', diag_kind='kde',
                            palette='tab10', plot_kws={'alpha': 0.4})
    pair_fig.fig.suptitle('Pairplot — Top 5 Features', y=1.02, fontsize=13)
    pair_fig.savefig(os.path.join(PLOTS_DIR, '06_pairplot_top5_features.png'),
                     bbox_inches='tight', dpi=150)
    plt.close('all')
    print(f"    Saved → {os.path.join(PLOTS_DIR, '06_pairplot_top5_features.png')}")

except Exception as e:
    print(f"    [ERROR] Plot 6 failed: {e}")


# ---- Plot 7: Class-wise Feature Mean Grouped Bar Chart ----
try:
    print("[Plot 7] Class-wise feature mean grouped bar chart...")
    # Compute mean of each feature for each fault_type
    class_means = df_final.groupby('fault_type')[FEATURE_COLS].mean()

    x = np.arange(len(FEATURE_COLS))
    bar_width = 0.11
    # colormaps API changed in matplotlib 3.7+ — use matplotlib.colormaps
    cmap7 = matplotlib.colormaps.get_cmap('tab10')

    plt.figure(figsize=(20, 7))
    for i, ft in enumerate(sorted(FAULT_NAMES.keys())):
        offset = (i - 3) * bar_width   # centre the group of bars
        plt.bar(x + offset, class_means.loc[ft], width=bar_width,
                label=FAULT_NAMES[ft], color=cmap7(i / 9), alpha=0.85)

    plt.xticks(x, FEATURE_COLS, rotation=45, ha='right', fontsize=8)
    plt.title('Class-wise Feature Means')
    plt.xlabel('Feature')
    plt.ylabel('Mean Value')
    plt.legend(title='Fault Type', fontsize=8)
    plt.tight_layout()
    save_plot('07_classwise_feature_means.png')

except Exception as e:
    print(f"    [ERROR] Plot 7 failed: {e}")

print("[OK] Stage 3.1 complete.")


# =============================================================================
# STAGE 3.2 — PREPROCESSING
# =============================================================================

print("\n" + "=" * 60)
print("STAGE 3.2 — PREPROCESSING")
print("=" * 60)

try:
    # ---- Step 1: Missing Value Check ----
    print("\n[Step 1] Checking for missing values...")
    null_counts = df_final.isnull().sum()
    print(null_counts)

    if null_counts.any():
        # Fill each numeric feature column with its own mean
        for col in FEATURE_COLS:
            col_mean = df_final[col].mean()
            df_final[col] = df_final[col].fillna(col_mean)
        print("Missing values filled with column means.")
    else:
        print("No missing values found — no action needed.")

except Exception as e:
    print(f"[ERROR] Step 1 failed: {e}")


try:
    # ---- Step 2: Label Encoding Verification ----
    print("\n[Step 2] Verifying fault_type label encoding...")
    unique_labels = sorted(df_final['fault_type'].unique())
    print(f"Unique fault_type values : {unique_labels}")

    # Confirm all values are in expected range 0–6
    expected = list(range(7))
    if unique_labels == expected:
        print("Label encoding is correct (0 to 6).")
    else:
        print(f"[WARNING] Unexpected labels found. Expected {expected}")

    print("\nCounts per label:")
    print(df_final['fault_type'].value_counts().sort_index())

except Exception as e:
    print(f"[ERROR] Step 2 failed: {e}")


try:
    # ---- Step 3: Outlier Treatment — IQR Winsorization ----
    print("\n[Step 3] Winsorizing outliers using IQR method (clip only, no rows dropped)...")

    for col in FEATURE_COLS:
        Q1  = df_final[col].quantile(0.25)
        Q3  = df_final[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Count how many values fall outside the bounds before clipping
        capped = ((df_final[col] < lower_bound) | (df_final[col] > upper_bound)).sum()

        # Clip values to the bounds — rows are NOT removed
        df_final[col] = df_final[col].clip(lower=lower_bound, upper=upper_bound)

        print(f"    {col:<25} : {capped} values capped")

    print("Winsorization complete.")

except Exception as e:
    print(f"[ERROR] Step 3 failed: {e}")


try:
    # ---- Step 4: Train-Test Split ----
    print("\n[Step 4] Splitting into 80% train / 20% test...")

    X = df_final[FEATURE_COLS]
    y = df_final['fault_type']

    # Store the original (unscaled) test set for Stage 6 prediction demo
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    # Reset indices so iloc-based access in Stage 6 is always safe
    X_train_raw = X_train_raw.reset_index(drop=True)
    X_test_raw  = X_test_raw.reset_index(drop=True)
    y_train     = y_train.reset_index(drop=True)
    y_test      = y_test.reset_index(drop=True)

    print(f"    X_train : {X_train_raw.shape}  |  y_train : {y_train.shape}")
    print(f"    X_test  : {X_test_raw.shape}   |  y_test  : {y_test.shape}")

except Exception as e:
    print(f"[ERROR] Step 4 failed: {e}")


try:
    # ---- Step 5: Feature Scaling ----
    print("\n[Step 5] Applying StandardScaler (fit on train only)...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled  = scaler.transform(X_test_raw)

    # Wrap back into DataFrames so column names are preserved for selector
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=FEATURE_COLS)
    X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=FEATURE_COLS)

    # Print before vs after for 3 sample features
    sample_features = ['ia_rms', 'te_mean', 'slip']
    print("\n    Before scaling (raw train values):")
    print(X_train_raw[sample_features].describe().loc[['mean', 'std']].round(4))
    print("\n    After scaling (scaled train values):")
    print(X_train_scaled[sample_features].describe().loc[['mean', 'std']].round(4))

except Exception as e:
    print(f"[ERROR] Step 5 failed: {e}")


try:
    # ---- Step 6: Feature Selection — SelectKBest (top 10) ----
    print("\n[Step 6] Selecting top 10 features with SelectKBest + f_classif...")

    selector = SelectKBest(score_func=f_classif, k=10)
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel  = selector.transform(X_test_scaled)

    # Find out which 10 features were selected
    selected_mask     = selector.get_support()
    selected_features = [col for col, keep in zip(FEATURE_COLS, selected_mask) if keep]
    print(f"    Selected features : {selected_features}")

except Exception as e:
    print(f"[ERROR] Step 6 failed: {e}")


try:
    # ---- Step 7: Class Imbalance — SMOTE ----
    print("\n[Step 7] Applying SMOTE to balance training classes...")

    print("    Class counts BEFORE SMOTE:")
    before = y_train.value_counts().sort_index()
    for ft, cnt in before.items():
        print(f"        {ft} — {FAULT_NAMES[ft]:<15} : {cnt}")

    smote = SMOTE(random_state=42)
    X_train_final, y_train_final = smote.fit_resample(X_train_sel, y_train)

    print("\n    Class counts AFTER SMOTE:")
    after = pd.Series(y_train_final).value_counts().sort_index()
    for ft, cnt in after.items():
        print(f"        {ft} — {FAULT_NAMES[ft]:<15} : {cnt}")

except Exception as e:
    print(f"[ERROR] Step 7 failed: {e}")


try:
    # ---- Step 8: Final Shape Check ----
    print("\n[Step 8] Final shapes after all preprocessing steps:")
    print(f"    X_train : {X_train_final.shape}")
    print(f"    X_test  : {X_test_sel.shape}")
    print(f"    y_train : {y_train_final.shape}")
    print(f"    y_test  : {y_test.shape}")
    print("[OK] Stage 3.2 complete.")

except Exception as e:
    print(f"[ERROR] Step 8 failed: {e}")


# =============================================================================
# STAGE 4 — MODEL TRAINING
# =============================================================================

print("\n" + "=" * 60)
print("STAGE 4 — MODEL TRAINING")
print("=" * 60)

# Dictionary to hold trained model objects
models = {}

# Dictionary to hold test predictions from each model
predictions = {}


def train_model(name, model):
    """Train a single model and store it and its predictions."""
    try:
        model.fit(X_train_final, y_train_final)
        preds = model.predict(X_test_sel)
        models[name]      = model
        predictions[name] = preds
        print(f"    [OK] {name} trained.")
    except Exception as e:
        print(f"    [ERROR] {name} failed: {e}")


# ---- 1. Logistic Regression ----
print("\nTraining Logistic Regression...")
# multi_class='multinomial' was deprecated in sklearn 1.5 and removed in 1.7+
# sklearn auto-selects the right solver for multi-class — no need to specify it
lr = LogisticRegression(max_iter=1000, random_state=42)
train_model('Logistic Regression', lr)

# ---- 2. K-Nearest Neighbors ----
print("Training KNN...")
knn = KNeighborsClassifier(n_neighbors=5)
train_model('KNN', knn)

# ---- 3. Decision Tree ----
print("Training Decision Tree...")
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
train_model('Decision Tree', dt)

# ---- 4. Support Vector Machine ----
print("Training SVM...")
svm = SVC(kernel='rbf', decision_function_shape='ovr', random_state=42)
train_model('SVM', svm)

# ---- 5. Random Forest ----
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
train_model('Random Forest', rf)

# ---- 6. XGBoost ----
print("Training XGBoost...")
xgb = XGBClassifier(
    objective='multi:softmax',
    num_class=7,
    eval_metric='mlogloss',
    random_state=42,
    verbosity=0
)
train_model('XGBoost', xgb)

print("\n[OK] Stage 4 complete.")


# =============================================================================
# STAGE 5 — EVALUATION
# =============================================================================

print("\n" + "=" * 60)
print("STAGE 5 — EVALUATION")
print("=" * 60)

# Store accuracy scores for the comparison chart at the end
accuracy_scores = {}

# Stratified 5-fold cross-validation object — reused for all models
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models.items():
    print(f"\n{'─' * 50}")
    print(f"Model : {model_name}")
    print('─' * 50)

    preds = predictions[model_name]

    # ---- 5a: Accuracy Score ----
    try:
        acc = accuracy_score(y_test, preds)
        accuracy_scores[model_name] = acc
        print(f"Test Accuracy : {acc:.4f}  ({acc * 100:.2f}%)")
    except Exception as e:
        print(f"[ERROR] Accuracy failed: {e}")

    # ---- 5b: Classification Report ----
    try:
        report = classification_report(y_test, preds, target_names=TARGET_NAMES)
        print("\nClassification Report:")
        print(report)
    except Exception as e:
        print(f"[ERROR] Classification report failed: {e}")

    # ---- 5c: Confusion Matrix Heatmap ----
    try:
        # labels=list(range(7)) ensures the matrix is always 7x7
        # even if some classes are absent in predictions
        cm = confusion_matrix(y_test, preds, labels=list(range(7)))
        plt.figure(figsize=(9, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
        plt.title(f'Confusion Matrix — {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()

        safe_name = model_name.replace(' ', '_').lower()
        save_plot(f'confusion_matrix_{safe_name}.png')

    except Exception as e:
        print(f"[ERROR] Confusion matrix plot failed: {e}")

    # ---- 5d: Stratified 5-Fold Cross-Validation ----
    try:
        cv_scores = cross_val_score(model, X_train_final, y_train_final,
                                    cv=cv_strategy, scoring='accuracy')
        print(f"5-Fold CV Accuracy : {cv_scores.mean():.4f}  ± {cv_scores.std():.4f}")
    except Exception as e:
        print(f"[ERROR] Cross-validation failed: {e}")

    # ---- 5f: Per-Class F1 Score Bar Chart ----
    try:
        f1_scores = f1_score(y_test, preds, average=None, labels=list(range(7)))

        plt.figure(figsize=(10, 5))
        plt.bar(TARGET_NAMES, f1_scores, color='mediumseagreen', edgecolor='black')
        plt.title(f'Per-Class F1 Score — {model_name}')
        plt.xlabel('Fault Type')
        plt.ylabel('F1 Score')
        plt.ylim(0, 1.05)
        plt.xticks(rotation=15)
        plt.tight_layout()

        safe_name = model_name.replace(' ', '_').lower()
        save_plot(f'f1_per_class_{safe_name}.png')

    except Exception as e:
        print(f"[ERROR] F1 bar chart failed: {e}")


# ---- 5e: Model Comparison Bar Chart (all models side by side) ----
try:
    print("\n[Plot] Model comparison bar chart...")
    plt.figure(figsize=(10, 5))
    plt.bar(list(accuracy_scores.keys()), list(accuracy_scores.values()),
            color='steelblue', edgecolor='black')
    plt.title('Model Comparison — Test Accuracy')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=15, ha='right')

    # Annotate each bar with the accuracy value
    for i, (model_name, acc) in enumerate(accuracy_scores.items()):
        plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_plot('model_comparison_accuracy.png')

except Exception as e:
    print(f"[ERROR] Model comparison chart failed: {e}")

print("\n[OK] Stage 5 complete.")


# =============================================================================
# STAGE 6 — PREDICTION FUNCTION
# =============================================================================

print("\n" + "=" * 60)
print("STAGE 6 — PREDICTION FUNCTION")
print("=" * 60)


def predict_fault(model, scaler, selector, sample):
    """
    Predict fault type for a single raw input sample.

    Parameters
    ----------
    model    : trained classifier
    scaler   : fitted StandardScaler
    selector : fitted SelectKBest selector
    sample   : list or array of 17 raw feature values (same order as FEATURE_COLS)

    Returns
    -------
    int — predicted fault_type (also prints result)
    """
    # Step 1: Convert the raw sample list into a one-row DataFrame with correct column names
    sample_df = pd.DataFrame([sample], columns=FEATURE_COLS)

    # Step 2: Scale the sample using the same scaler fitted on training data
    sample_scaled = scaler.transform(sample_df)

    # Step 3: Select the same top features the model was trained on
    sample_selected = selector.transform(sample_scaled)

    # Step 4: Predict the fault_type integer
    predicted_label = model.predict(sample_selected)[0]

    # Step 5: Print human-readable result
    fault_name = FAULT_NAMES[predicted_label]
    if predicted_label == 0:
        print(f"    Prediction : HEALTHY")
    else:
        print(f"    Prediction : FAULT DETECTED — {fault_name}")

    return predicted_label


# ---- Test predict_fault on 3 random samples from the unscaled test set ----
try:
    print("\nTesting predict_fault() on 3 random samples from X_test_raw...\n")

    # Pick 3 random indices from the test set
    np.random.seed(42)
    test_indices = np.random.choice(len(X_test_raw), size=3, replace=False)

    # Use the best model (highest accuracy) for the demo
    best_model_name = max(accuracy_scores, key=accuracy_scores.get)
    best_model      = models[best_model_name]
    print(f"Using best model : {best_model_name} (accuracy = {accuracy_scores[best_model_name]:.4f})\n")

    for i, idx in enumerate(test_indices):
        raw_sample   = X_test_raw.iloc[idx].values.tolist()
        true_label   = y_test.iloc[idx]
        true_name    = FAULT_NAMES[true_label]

        print(f"  Sample {i + 1}:")
        print(f"    True label : {true_label} — {true_name}")
        predicted = predict_fault(best_model, scaler, selector, raw_sample)
        match = "CORRECT" if predicted == true_label else "WRONG"
        print(f"    Result     : {match}")
        print()

except Exception as e:
    print(f"[ERROR] Stage 6 failed: {e}")

print("[OK] Stage 6 complete.")
print("\n" + "=" * 60)
print("ALL STAGES COMPLETE")
print(f"Plots saved in : {PLOTS_DIR}")
print("=" * 60)
