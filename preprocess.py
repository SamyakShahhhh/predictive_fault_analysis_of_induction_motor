# =============================================================================
# preprocess.py — Stage 3.2
# run_preprocessing() runs all 8 steps in order and returns everything
# needed by train.py, evaluate.py, and predict.py
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

from config import FEATURE_COLS, FAULT_NAMES


def check_missing(df):
    """Step 1 — Print null counts. Fill with column mean if any found."""
    print("\n[Step 1] Checking for missing values...")
    null_counts = df.isnull().sum()
    print(null_counts)

    if null_counts.any():
        for col in FEATURE_COLS:
            df[col] = df[col].fillna(df[col].mean())
        print("Missing values filled with column means.")
    else:
        print("No missing values found — no action needed.")

    return df


def verify_labels(df):
    """Step 2 — Confirm fault_type values are integers 0 through 6."""
    print("\n[Step 2] Verifying fault_type label encoding...")
    unique = sorted(df['fault_type'].unique())
    print(f"Unique fault_type values : {unique}")

    if unique == list(range(7)):
        print("Label encoding is correct (0 to 6).")
    else:
        print(f"[WARNING] Unexpected labels. Expected {list(range(7))}")

    print("\nCounts per label:")
    print(df['fault_type'].value_counts().sort_index())


def winsorize(df):
    """Step 3 — Clip outliers using IQR method. No rows are removed."""
    print("\n[Step 3] Winsorizing outliers (IQR method, clip only)...")

    for col in FEATURE_COLS:
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        capped = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"    {col:<25} : {capped} values capped")

    print("Winsorization complete.")
    return df


def split_data(df):
    """Step 4 — 80/20 stratified train-test split."""
    print("\n[Step 4] Splitting into 80% train / 20% test...")

    X = df[FEATURE_COLS]
    y = df['fault_type']

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # Reset indices so iloc-based access is always safe
    X_train_raw = X_train_raw.reset_index(drop=True)
    X_test_raw  = X_test_raw.reset_index(drop=True)
    y_train     = y_train.reset_index(drop=True)
    y_test      = y_test.reset_index(drop=True)

    print(f"    X_train : {X_train_raw.shape}  |  y_train : {y_train.shape}")
    print(f"    X_test  : {X_test_raw.shape}   |  y_test  : {y_test.shape}")

    return X_train_raw, X_test_raw, y_train, y_test


def scale_features(X_train_raw, X_test_raw):
    """Step 5 — StandardScaler fitted on train only, applied to both."""
    print("\n[Step 5] Applying StandardScaler (fit on train only)...")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled  = scaler.transform(X_test_raw)

    # Wrap back into DataFrames so column names survive into feature selection
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=FEATURE_COLS)
    X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=FEATURE_COLS)

    # Print mean and std before/after for 3 sample features
    sample_cols = ['ia_rms', 'te_mean', 'slip']
    print("\n    Before scaling (raw train):")
    print(X_train_raw[sample_cols].describe().loc[['mean', 'std']].round(4))
    print("\n    After scaling (scaled train):")
    print(X_train_scaled[sample_cols].describe().loc[['mean', 'std']].round(4))

    return X_train_scaled, X_test_scaled, scaler


def select_features(X_train_scaled, X_test_scaled, y_train):
    """Step 6 — SelectKBest top 10 features, fitted on train only."""
    print("\n[Step 6] Selecting top 10 features (SelectKBest + f_classif)...")

    selector = SelectKBest(score_func=f_classif, k=10)
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel  = selector.transform(X_test_scaled)

    selected = [col for col, keep in zip(FEATURE_COLS, selector.get_support()) if keep]
    print(f"    Selected features : {selected}")

    return X_train_sel, X_test_sel, selector


def apply_smote(X_train_sel, y_train):
    """Step 7 — SMOTE to balance class distribution in training data."""
    print("\n[Step 7] Applying SMOTE to balance training classes...")

    print("    Class counts BEFORE SMOTE:")
    for ft, cnt in y_train.value_counts().sort_index().items():
        print(f"        {ft} — {FAULT_NAMES[ft]:<15} : {cnt}")

    smote = SMOTE(random_state=42)
    X_train_final, y_train_final = smote.fit_resample(X_train_sel, y_train)

    print("\n    Class counts AFTER SMOTE:")
    for ft, cnt in pd.Series(y_train_final).value_counts().sort_index().items():
        print(f"        {ft} — {FAULT_NAMES[ft]:<15} : {cnt}")

    return X_train_final, y_train_final


def run_preprocessing(df):
    """
    Run all 8 preprocessing steps in order.

    Returns
    -------
    X_train_final : numpy array — SMOTE-balanced training features
    X_test_sel    : numpy array — test features after feature selection
    y_train_final : numpy array — SMOTE-balanced training labels
    y_test        : pandas Series — true test labels
    scaler        : fitted StandardScaler (needed for predict.py)
    selector      : fitted SelectKBest  (needed for predict.py)
    X_test_raw    : pandas DataFrame — original unscaled test features (needed for Stage 6 demo)
    """
    try:
        df = check_missing(df)
    except Exception as e:
        print(f"[ERROR] Step 1 failed: {e}")

    try:
        verify_labels(df)
    except Exception as e:
        print(f"[ERROR] Step 2 failed: {e}")

    try:
        df = winsorize(df)
    except Exception as e:
        print(f"[ERROR] Step 3 failed: {e}")

    try:
        X_train_raw, X_test_raw, y_train, y_test = split_data(df)
    except Exception as e:
        print(f"[ERROR] Step 4 failed: {e}")
        return None

    try:
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train_raw, X_test_raw)
    except Exception as e:
        print(f"[ERROR] Step 5 failed: {e}")
        return None

    try:
        X_train_sel, X_test_sel, selector = select_features(X_train_scaled, X_test_scaled, y_train)
    except Exception as e:
        print(f"[ERROR] Step 6 failed: {e}")
        return None

    try:
        X_train_final, y_train_final = apply_smote(X_train_sel, y_train)
    except Exception as e:
        print(f"[ERROR] Step 7 failed: {e}")
        return None

    # Step 8 — Final shape check
    print("\n[Step 8] Final shapes after all preprocessing:")
    print(f"    X_train : {X_train_final.shape}")
    print(f"    X_test  : {X_test_sel.shape}")
    print(f"    y_train : {y_train_final.shape}")
    print(f"    y_test  : {y_test.shape}")
    print("[OK] Preprocessing complete.")

    return X_train_final, X_test_sel, y_train_final, y_test, scaler, selector, X_test_raw
