# =============================================================================
# predict.py — Stage 6
# predict_fault() : takes 17 raw feature values, returns fault type
# run_prediction_demo() : tests on 3 random samples from the test set
# =============================================================================

import numpy as np
import pandas as pd

from config import FEATURE_COLS, FAULT_NAMES


def predict_fault(model, scaler, selector, sample):
    """
    Predict the fault type for a single raw input sample.

    Parameters
    ----------
    model    : any trained sklearn/xgboost classifier
    scaler   : the StandardScaler fitted during preprocessing
    selector : the SelectKBest selector fitted during preprocessing
    sample   : list or array of 17 raw feature values
               (must follow the order in FEATURE_COLS)

    Returns
    -------
    int — predicted fault_type label
    """
    # Step 1: wrap raw values in a DataFrame so the scaler gets named columns
    sample_df = pd.DataFrame([sample], columns=FEATURE_COLS)

    # Step 2: apply the same scaling that was fitted on training data
    sample_scaled = scaler.transform(sample_df)

    # Step 3: select the same top-10 features the model was trained on
    sample_selected = selector.transform(sample_scaled)

    # Step 4: predict the fault_type integer
    predicted = int(model.predict(sample_selected)[0])

    # Step 5: print a human-readable result
    fault_name = FAULT_NAMES[predicted]
    if predicted == 0:
        print(f"    Prediction : HEALTHY")
    else:
        print(f"    Prediction : FAULT DETECTED — {fault_name}")

    return predicted


def run_prediction_demo(models, accuracy_scores, scaler, selector, X_test_raw, y_test):
    """
    Test predict_fault() on 3 random samples from the unscaled test set.
    Uses the best-performing model automatically.
    """
    try:
        # Pick the model with the highest test accuracy
        best_name  = max(accuracy_scores, key=accuracy_scores.get)
        best_model = models[best_name]
        print(f"Using best model : {best_name}  "
              f"(accuracy = {accuracy_scores[best_name]:.4f})\n")

        # Pick 3 random positional indices from the test set
        np.random.seed(42)
        indices = np.random.choice(len(X_test_raw), size=3, replace=False)

        for i, idx in enumerate(indices):
            raw_sample = X_test_raw.iloc[idx].values.tolist()
            true_label = y_test.iloc[idx]

            predicted = predict_fault(best_model, scaler, selector, raw_sample)

    except Exception as e:
        print(f"[ERROR] Prediction demo failed: {e}")
