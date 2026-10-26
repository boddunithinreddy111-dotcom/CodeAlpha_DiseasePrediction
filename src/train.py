"""
Training and Benchmarking Pipeline for Disease Prediction - CodeAlpha Task 4.
Evaluates SVM, Logistic Regression, Random Forest, and Gradient Boosting / XGBoost.
Selects best clinical model, serializes weights, and generates publication plots.
"""

import sys
from pathlib import Path
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from src.preprocessing import prepare_medical_dataset, MedicalDataPreprocessor
from src.evaluate import (
    calculate_medical_metrics,
    plot_clinical_confusion_matrix,
    plot_clinical_roc_curve,
    plot_clinical_feature_importance,
    plot_medical_model_benchmark
)

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def get_clinical_models():
    """
    Returns candidate models required by the CodeAlpha task.
    """
    models = {
        "SVM (RBF Kernel)": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=6,
            n_jobs=-1,
            random_state=42
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.08,
            max_depth=6,
            random_state=42
        ),
    }

    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=5,
            eval_metric="logloss",
            random_state=42
        )
    except ImportError:
        pass

    return models


def run_medical_training_pipeline(csv_path: Path = DATA_PATH):
    print("=" * 70)
    print("CodeAlpha ML Internship - Task 4: Disease Prediction from Medical Data")
    print("=" * 70)

    if not csv_path.exists():
        print(f"[INFO] Medical dataset not found at: {csv_path}")
        from src.download_data import download_medical_data
        download_medical_data()

    print(f"\n[1/5] Loading and Preprocessing Clinical Records: {csv_path.name}")
    X_train, X_test, y_train, y_test, preprocessor = prepare_medical_dataset(csv_path)
    print(f"      Training patients : {len(X_train)} records")
    print(f"      Testing patients  : {len(X_test)} records")
    print(f"      Engineered Features : {len(preprocessor.feature_names)} features")

    preprocessor_path = MODELS_DIR / "preprocessor.joblib"
    preprocessor.save(preprocessor_path)

    models = get_clinical_models()
    results = []
    trained_models = {}

    print("\n[2/5] Training Candidate Clinical Models (SVM, Logistic Regression, Random Forest, Boosting)...")
    for name, model in models.items():
        print(f"      --> Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

        metrics = calculate_medical_metrics(model, X_test, y_test)
        results.append({
            "Model": name,
            "Accuracy": round(metrics["Accuracy"], 4),
            "Precision": round(metrics["Precision"], 4),
            "Sensitivity (Recall)": round(metrics["Sensitivity (Recall)"], 4),
            "Specificity": round(metrics["Specificity"], 4),
            "F1-Score": round(metrics["F1-Score"], 4),
            "ROC-AUC": round(metrics["ROC-AUC"], 4) if not pd.isna(metrics["ROC-AUC"]) else 0.0,
        })

    results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_csv = REPORTS_DIR / "model_benchmark_results.csv"
    results_df.to_csv(results_csv, index=False)

    print("\n[3/5] Clinical Model Benchmark Summary:")
    print("-" * 80)
    print(results_df.to_string(index=False))
    print("-" * 80)
    print(f"Benchmark summary saved to: {results_csv}")

    best_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]
    best_f1 = results_df.iloc[0]["F1-Score"]
    print(f"\n[4/5] Selected Best Clinical Model: '{best_name}' (F1 = {best_f1:.4f})")

    best_model_path = MODELS_DIR / "best_disease_model.joblib"
    joblib.dump(best_model, best_model_path)
    print(f"      Saved best model artifact to: {best_model_path}")

    print("\n[5/5] Generating Visual Evaluation Figures...")
    best_metrics = calculate_medical_metrics(best_model, X_test, y_test)
    print("\nDetailed Clinical Classification Report:")
    print(best_metrics["Classification Report"])

    plot_medical_model_benchmark(results_df)
    plot_clinical_confusion_matrix(best_metrics["Confusion Matrix"], model_name=best_name)
    plot_clinical_roc_curve(best_model, X_test, y_test, model_name=best_name)

    if hasattr(best_model, "feature_importances_"):
        plot_clinical_feature_importance(best_model, preprocessor.feature_names, model_name=best_name)
    elif "Random Forest" in trained_models and hasattr(trained_models["Random Forest"], "feature_importances_"):
        plot_clinical_feature_importance(
            trained_models["Random Forest"], preprocessor.feature_names, model_name="Random Forest"
        )

    print("\n" + "=" * 70)
    print("Clinical Pipeline Completed Successfully!")
    print(f"Artifacts ready in: {MODELS_DIR}")
    print(f"Reports ready in: {PROJECT_ROOT / 'reports' / 'figures'}")
    print("=" * 70)

    return best_model, preprocessor, results_df


if __name__ == "__main__":
    run_medical_training_pipeline()
