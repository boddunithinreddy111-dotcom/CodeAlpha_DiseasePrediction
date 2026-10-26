"""
Clinical Diagnostic Inference Module.
Evaluates patient biomarkers against the trained disease prediction model
and generates diagnostic recommendations.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import MedicalDataPreprocessor

MODEL_PATH = PROJECT_ROOT / "models" / "best_disease_model.joblib"
PREPROCESSOR_PATH = PROJECT_ROOT / "models" / "preprocessor.joblib"


def load_medical_artifacts():
    """
    Loads saved disease model and preprocessor.
    """
    if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Clinical model or preprocessor missing.\n"
            f"Run 'python src/train.py' first to train the models."
        )
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


def get_clinical_advice(is_positive: bool, disease_probability: float, patient_data: Dict[str, Any]) -> str:
    """
    Produces actionable clinical recommendations based on risk level and biomarkers.
    """
    if is_positive:
        warnings = []
        if patient_data.get("ExerciseAngina") == "Y":
            warnings.append("exercise-induced angina observed")
        if patient_data.get("ST_Slope") in ["Flat", "Down"]:
            warnings.append("abnormal ST segment depression during stress")
        if patient_data.get("Cholesterol", 0) > 240:
            warnings.append("hypercholesterolemia (>240 mg/dl)")
        if patient_data.get("RestingBP", 0) >= 140:
            warnings.append("Stage 2 hypertension")

        warning_str = f" Critical findings: {', '.join(warnings)}." if warnings else ""

        return (
            f"ELEVATED CARDIAC RISK (Confidence: {disease_probability * 100:.1f}%).{warning_str} "
            f"Immediate referral to a cardiologist is strongly advised. "
            f"Recommended diagnostics: 12-lead resting ECG, echocardiogram, and treadmill stress test."
        )
    else:
        return (
            f"LOW CARDIAC RISK (Confidence: {(1 - disease_probability) * 100:.1f}%). "
            f"Patient biomarkers are within manageable limits. "
            f"Recommend routine annual cardiovascular wellness screening and heart-healthy lifestyle habits."
        )


def predict_patient_risk(patient_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluates a single patient's clinical records.
    """
    model, preprocessor = load_medical_artifacts()

    df = pd.DataFrame([patient_dict])
    X_processed = preprocessor.transform(df)

    pred = int(model.predict(X_processed)[0])

    disease_prob = 0.5
    if hasattr(model, "predict_proba"):
        disease_prob = float(model.predict_proba(X_processed)[0, 1])

    diagnosis = "Heart Disease Detected (High Risk)" if pred == 1 else "Normal / Low Risk"
    advice = get_clinical_advice(pred == 1, disease_prob, patient_dict)

    return {
        "Diagnosis": diagnosis,
        "Risk_Score": pred,
        "Disease_Probability": round(disease_prob, 4),
        "Healthy_Probability": round(1.0 - disease_prob, 4),
        "Clinical_Advice": advice,
    }


if __name__ == "__main__":
    sample_patient = {
        "Age": 58,
        "Sex": "M",
        "ChestPainType": "ASY",
        "RestingBP": 140,
        "Cholesterol": 289,
        "FastingBS": 1,
        "RestingECG": "LVH",
        "MaxHR": 130,
        "ExerciseAngina": "Y",
        "Oldpeak": 2.2,
        "ST_Slope": "Flat"
    }

    try:
        print("[TEST] Evaluating sample patient biomarkers...")
        res = predict_patient_risk(sample_patient)
        print(f"--> Diagnostic Finding : {res['Diagnosis']}")
        print(f"--> Disease Risk       : {res['Disease_Probability'] * 100:.1f}%")
        print(f"--> Clinical Advice    : {res['Clinical_Advice']}")
    except FileNotFoundError as e:
        print(e)
