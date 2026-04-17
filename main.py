# =============================================================================
# main.py — Entry point
# Run this file to execute the full pipeline from Stage 1 to Stage 6
#
# Usage:
#   python main.py
# =============================================================================

from config import FILE_MAP

from data_loader import load_files, combine_and_label
from visualize   import run_all_plots
from preprocess  import run_preprocessing
from train       import train_all_models
from evaluate    import run_evaluation
from predict     import run_prediction_demo


def main():

    # =========================================================================
    # STAGE 1 — Load all CSV files
    # =========================================================================
    print("=" * 60)
    print("STAGE 1 — LOADING DATA")
    print("=" * 60)

    try:
        dataframes = load_files(FILE_MAP)
        print("\n[OK] Stage 1 complete.")
    except Exception as e:
        print(f"[ERROR] Stage 1 failed: {e}")
        return   # cannot continue without data

    # =========================================================================
    # STAGE 2 — Combine and assign fault_type labels
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 2 — COMBINING AND LABELLING")
    print("=" * 60)

    try:
        df_final = combine_and_label(dataframes)
        print("[OK] Stage 2 complete.")
    except Exception as e:
        print(f"[ERROR] Stage 2 failed: {e}")
        return

    # =========================================================================
    # STAGE 3.1 — Data visualisation (7 plots saved to plots/)
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 3.1 — DATA VISUALISATION")
    print("=" * 60)

    try:
        run_all_plots(df_final)
    except Exception as e:
        print(f"[ERROR] Stage 3.1 failed: {e}")
        # not fatal — continue to preprocessing

    # =========================================================================
    # STAGE 3.2 — Preprocessing pipeline
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 3.2 — PREPROCESSING")
    print("=" * 60)

    result = run_preprocessing(df_final)
    if result is None:
        print("[ERROR] Preprocessing failed — cannot continue.")
        return

    X_train_final, X_test_sel, y_train_final, y_test, scaler, selector, X_test_raw = result

    # =========================================================================
    # STAGE 4 — Train all 6 models
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 4 — MODEL TRAINING")
    print("=" * 60)

    try:
        models, predictions = train_all_models(X_train_final, y_train_final, X_test_sel)
    except Exception as e:
        print(f"[ERROR] Stage 4 failed: {e}")
        return

    # =========================================================================
    # STAGE 5 — Evaluate all models
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 5 — EVALUATION")
    print("=" * 60)

    try:
        accuracy_scores = run_evaluation(
            models, predictions, y_test, X_train_final, y_train_final
        )
    except Exception as e:
        print(f"[ERROR] Stage 5 failed: {e}")
        accuracy_scores = {}

    # =========================================================================
    # STAGE 6 — Prediction demo
    # =========================================================================
    print("\n" + "=" * 60)
    print("STAGE 6 — PREDICTION DEMO")
    print("=" * 60)

    if accuracy_scores and models:
        run_prediction_demo(
            models, accuracy_scores, scaler, selector, X_test_raw, y_test
        )
    else:
        print("[SKIP] No models available for prediction demo.")

    # =========================================================================
    print("\n" + "=" * 60)
    print("ALL STAGES COMPLETE")
    print(f"Plots saved in : C:\\Users\\Swayam\\Downloads\\motor_fault_project\\plots")
    print("=" * 60)


if __name__ == '__main__':
    main()
