# =============================================================================
# evaluate.py — Stage 5
# For each model: accuracy, classification report, confusion matrix,
# 5-fold CV, per-class F1 chart, and a final model comparison chart
# =============================================================================

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from config import TARGET_NAMES, PLOTS_DIR


def save_plot(filename):
    """Save current figure to PLOTS_DIR and close cleanly."""
    path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    Saved → {path}")


def evaluate_one_model(model_name, model, preds, y_test, X_train, y_train):
    """
    Run all evaluation steps for a single model.

    Returns
    -------
    float — test accuracy (stored by caller for comparison chart)
    """
    print(f"\n{'─' * 50}")
    print(f"Model : {model_name}")
    print('─' * 50)

    acc = None

    # ---- Accuracy score ----
    try:
        acc = accuracy_score(y_test, preds)
        print(f"Test Accuracy : {acc:.4f}  ({acc * 100:.2f}%)")
    except Exception as e:
        print(f"[ERROR] Accuracy failed: {e}")

    # ---- Classification report ----
    try:
        report = classification_report(y_test, preds, target_names=TARGET_NAMES)
        print("\nClassification Report:")
        print(report)
    except Exception as e:
        print(f"[ERROR] Classification report failed: {e}")

    # ---- Confusion matrix heatmap ----
    try:
        # labels=list(range(7)) forces a 7×7 matrix even if a class has 0 predictions
        cm = confusion_matrix(y_test, preds, labels=list(range(7)))
        plt.figure(figsize=(9, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=TARGET_NAMES, yticklabels=TARGET_NAMES)
        plt.title(f'Confusion Matrix — {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()

        safe = model_name.replace(' ', '_').lower()
        save_plot(f'confusion_matrix_{safe}.png')

    except Exception as e:
        print(f"[ERROR] Confusion matrix failed: {e}")

    # ---- Stratified 5-Fold Cross-Validation ----
    try:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=cv, scoring='accuracy')
        print(f"5-Fold CV Accuracy : {cv_scores.mean():.4f}  ± {cv_scores.std():.4f}")
    except Exception as e:
        print(f"[ERROR] Cross-validation failed: {e}")

    # ---- Per-class F1 score bar chart ----
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

        safe = model_name.replace(' ', '_').lower()
        save_plot(f'f1_per_class_{safe}.png')

    except Exception as e:
        print(f"[ERROR] F1 bar chart failed: {e}")

    return acc


def plot_model_comparison(accuracy_scores):
    """Plot all model test accuracies side by side in one bar chart."""
    try:
        plt.figure(figsize=(10, 5))
        plt.bar(list(accuracy_scores.keys()), list(accuracy_scores.values()),
                color='steelblue', edgecolor='black')
        plt.title('Model Comparison — Test Accuracy')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.05)
        plt.xticks(rotation=15, ha='right')

        # Annotate each bar with its accuracy value
        for i, (name, acc) in enumerate(accuracy_scores.items()):
            plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        save_plot('model_comparison_accuracy.png')

    except Exception as e:
        print(f"[ERROR] Model comparison chart failed: {e}")


def run_evaluation(models, predictions, y_test, X_train, y_train):
    """
    Evaluate every model and produce the comparison chart at the end.

    Returns
    -------
    dict { model_name : test_accuracy }
    """
    accuracy_scores = {}

    for model_name, model in models.items():
        acc = evaluate_one_model(
            model_name, model, predictions[model_name],
            y_test, X_train, y_train
        )
        if acc is not None:
            accuracy_scores[model_name] = acc

    print("\n[Plot] Model comparison bar chart...")
    plot_model_comparison(accuracy_scores)

    print("\n[OK] Evaluation complete.")
    return accuracy_scores
