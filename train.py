# =============================================================================
# train.py — Stage 4
# Trains all 6 models and returns them in a dictionary
# =============================================================================

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def _fit_and_predict(name, model, X_train, y_train, X_test):
    """
    Train one model, predict on test set, return (model, predictions).
    Returns (None, None) if training fails.
    """
    try:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        print(f"    [OK] {name} trained.")
        return model, preds
    except Exception as e:
        print(f"    [ERROR] {name} failed: {e}")
        return None, None


def train_all_models(X_train, y_train, X_test):
    """
    Train all 6 classifiers on (X_train, y_train) and predict on X_test.

    Returns
    -------
    models      : dict { model_name : fitted model }
    predictions : dict { model_name : predicted labels on X_test }
    """
    models      = {}
    predictions = {}

    # Define each model with its hyperparameters
    model_defs = [
        (
            'Logistic Regression',
            # multi_class removed in sklearn 1.7+ — handled automatically
            LogisticRegression(max_iter=1000, random_state=42)
        ),
        (
            'KNN',
            KNeighborsClassifier(n_neighbors=5)
        ),
        (
            'Decision Tree',
            DecisionTreeClassifier(max_depth=10, random_state=42)
        ),
        (
            'SVM',
            SVC(kernel='rbf', decision_function_shape='ovr', random_state=42)
        ),
        (
            'Random Forest',
            RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        ),
        (
            'XGBoost',
            XGBClassifier(
                objective='multi:softmax',
                num_class=7,
                eval_metric='mlogloss',
                random_state=42,
                verbosity=0
            )
        ),
    ]

    for name, model in model_defs:
        print(f"\nTraining {name}...")
        fitted_model, preds = _fit_and_predict(name, model, X_train, y_train, X_test)
        if fitted_model is not None:
            models[name]      = fitted_model
            predictions[name] = preds

    print("\n[OK] All models trained.")
    return models, predictions
