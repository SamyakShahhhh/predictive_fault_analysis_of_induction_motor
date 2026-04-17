# =============================================================================
# config.py — All shared constants, paths, and labels
# Import this file in every other module so settings stay in one place
# =============================================================================

import os
import matplotlib

# Set non-interactive backend BEFORE any pyplot import anywhere in the project
matplotlib.use('Agg')

# =============================================================================
# PATHS
# =============================================================================

DATA_DIR  = r"C:\Users\Swayam\Downloads\claude_dataset"
PLOTS_DIR = r"C:\Users\Swayam\Downloads\motor_fault_project\plots"

# Create plots folder if it doesn't exist yet
os.makedirs(PLOTS_DIR, exist_ok=True)

# =============================================================================
# FEATURE COLUMNS
# =============================================================================

# Fixed reference order — every CSV will be reordered to match this
FEATURE_COLS = [
    'ia_rms', 'ia_std', 'ia_kurtosis', 'ia_peak', 'ia_crest',
    'ia_domFreq', 'current_unbalance', 'te_mean', 'te_std', 'te_p2p',
    'torque_ripple', 'speed_ripple', 'ia_skewness', 'power_factor',
    'slip', 'ia_shape_factor', 'voltage_unbalance'
]

# Expected columns in every loaded CSV (features + binary label)
REFERENCE_COLS = FEATURE_COLS + ['label']

# =============================================================================
# FAULT LABELS
# =============================================================================

# Integer label → human-readable name
FAULT_NAMES = {
    0: 'Healthy',
    1: 'RotorEcc',
    2: 'BrokenBar',
    3: 'ITSC',
    4: 'P2P',
    5: 'VoltageUnbal',
    6: 'SuddenLoad'
}

# Ordered list used in classification reports and plot axes
TARGET_NAMES = [FAULT_NAMES[i] for i in range(7)]

# =============================================================================
# FILE MAP  —  add new CSV files here without touching any other file
# =============================================================================

# Maps CSV filename → internal key used in combine_and_label()
FILE_MAP = {
    'healthy_features_cleaned.csv'    : 'healthy',
    'RotorEcc_clause.csv'             : 'rotor_ecc',
    'RotorBroke_features_balanced.csv': 'broken_bar',
    'itsc_features_Claude.csv'        : 'itsc',
    'P_P_FAULT_claude.csv'            : 'p2p',
    'Voltage_unbal_claude.csv'        : 'voltage_unbal',
    'SuddenLoad_claude.csv'           : 'sudden_load',
}

# Maps internal key → fault_type integer (healthy file handled separately)
FAULT_LABEL_MAP = {
    'rotor_ecc'    : 1,
    'broken_bar'   : 2,
    'itsc'         : 3,
    'p2p'          : 4,
    'voltage_unbal': 5,
    'sudden_load'  : 6,
}
