"""
Streamlit Clinical Decision Support System - CodeAlpha Task 4.
Disease Prediction from Medical Data.
Run locally with: streamlit run visualizer/app.py
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import predict_patient_risk, load_medical_artifacts

st.set_page_config(
    page_title="Cardiovascular Disease Diagnostic System | CodeAlpha Task 4",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #991B1B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .alert-high-risk {
        background-color: #FEF2F2;
        border-left: 6px solid #DC2626;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .alert-low-risk {
        background-color: #ECFDF5;
        border-left: 6px solid #059669;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🫀 AI Clinical Decision Support: Disease Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">CodeAlpha Machine Learning Internship — Task 4: Structured Medical Data Classification</div>', unsafe_allow_html=True)

# Sidebar Patient Data
st.sidebar.header("📋 Patient Clinical Biomarkers")

st.sidebar.subheader("👤 Demographics")
age = st.sidebar.slider("Age (Years)", min_value=25, max_value=85, value=54)
sex = st.sidebar.selectbox("Biological Sex", ["M", "F"], index=0)

st.sidebar.subheader("🩺 Vitals & Hemodynamics")
resting_bp = st.sidebar.slider("Resting Blood Pressure (mm Hg)", min_value=80, max_value=210, value=130)
max_hr = st.sidebar.slider("Maximum Heart Rate Achieved (bpm)", min_value=60, max_value=210, value=145)
resting_ecg = st.sidebar.selectbox(
    "Resting Electrocardiogram (ECG)",
    ["Normal", "LVH (Left Ventricular Hypertrophy)", "ST (ST-T Wave Abnormality)"]
)
resting_ecg_code = "Normal" if "Normal" in resting_ecg else ("LVH" if "LVH" in resting_ecg else "ST")

st.sidebar.subheader("🧪 Metabolic Profile")
cholesterol = st.sidebar.slider("Serum Cholesterol (mg/dl)", min_value=100, max_value=500, value=220)
fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"], index=0)
fasting_bs_code = 1 if fasting_bs == "Yes" else 0

st.sidebar.subheader("⚡ Cardiac Stress & Angina Profile")
chest_pain = st.sidebar.selectbox(
    "Chest Pain Classification",
    [
        "ASY (Asymptomatic - Silent Ischemia)",
        "NAP (Non-Anginal Pain)",
        "ATA (Atypical Angina)",
        "TA (Typical Angina)"
    ]
)
cp_code = chest_pain.split(" ")[0]

exercise_angina = st.sidebar.selectbox("Exercise-Induced Angina?", ["No", "Yes"], index=0)
angina_code = "Y" if exercise_angina == "Yes" else "N"

oldpeak = st.sidebar.slider("ST Depression Induced by Exercise (Oldpeak mm)", min_value=-2.0, max_value=6.0, value=1.0, step=0.1)
st_slope = st.sidebar.selectbox("Peak Exercise ST Segment Slope", ["Up (Upsloping)", "Flat (Flat)", "Down (Downsloping)"])
slope_code = st_slope.split(" ")[0]

# Main Dashboard
col1, col2 = st.columns([1.1, 0.9])

with col1:
    st.subheader("Patient Clinical Snapshot")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Patient Age", f"{age} yrs")
    p2.metric("Blood Pressure", f"{resting_bp} mm Hg")
    p3.metric("Cholesterol", f"{cholesterol} mg/dl")
    p4.metric("Max Heart Rate", f"{max_hr} bpm")

    patient_payload = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": cp_code,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs_code,
        "RestingECG": resting_ecg_code,
        "MaxHR": max_hr,
        "ExerciseAngina": angina_code,
        "Oldpeak": oldpeak,
        "ST_Slope": slope_code,
    }

    st.write("---")
    eval_btn = st.button("🚀 Run Comprehensive Medical Diagnostic Assessment", type="primary", use_container_width=True)

with col2:
    st.subheader("Diagnostic Assessment Output")

    if eval_btn:
        try:
            with st.spinner("Analyzing patient biomarkers against clinical models..."):
                result = predict_patient_risk(patient_payload)

            is_high_risk = result["Risk_Score"] == 1
            disease_prob = result["Disease_Probability"]
            healthy_prob = result["Healthy_Probability"]
            advice = result["Clinical_Advice"]

            if is_high_risk:
                st.markdown(f"""
                <div class="alert-high-risk">
                    <h3>⚠️ HIGH CARDIAC RISK DETECTED</h3>
                    <p><strong>Predicted Condition:</strong> Heart Failure / Coronary Disease Present</p>
                    <p><strong>Disease Probability:</strong> {disease_prob * 100:.1f}%</p>
                    <hr>
                    <p><strong>Clinical Recommendation:</strong> {advice}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="alert-low-risk">
                    <h3>✅ LOW CARDIAC RISK (NORMAL)</h3>
                    <p><strong>Predicted Condition:</strong> No Significant Coronary Pathology Detected</p>
                    <p><strong>Healthy Probability:</strong> {healthy_prob * 100:.1f}%</p>
                    <hr>
                    <p><strong>Clinical Recommendation:</strong> {advice}</p>
                </div>
                """, unsafe_allow_html=True)

            st.write("---")
            st.subheader("Cardiovascular Risk Distribution")
            chart_df = pd.DataFrame({
                "Diagnostic Class": ["Healthy / Low Risk", "Heart Disease (High Risk)"],
                "Confidence Score": [healthy_prob, disease_prob]
            }).set_index("Diagnostic Class")
            st.bar_chart(chart_df)

        except FileNotFoundError:
            st.warning("⚠️ Model weights not found. Run `python src/train.py` first to train and generate the model artifacts.")
        except Exception as e:
            st.error(f"Error evaluating patient: {e}")
    else:
        st.info("👈 Enter patient clinical vitals on the left sidebar and click **'Run Comprehensive Medical Diagnostic Assessment'** to evaluate risk.")

st.write("---")
st.markdown("""
**CodeAlpha Task 4 Medical Criteria:**
- **Clinical Algorithms Benchmarked:** Support Vector Machines (SVM), Logistic Regression, Random Forest, Gradient Boosting / XGBoost.
- **Biomarkers Evaluated:** Blood Pressure, Serum Cholesterol, Fasting Glucose, Resting ECG, ST-Segment Slope, Angina symptoms.
- **Primary Clinical Metric:** Sensitivity (Recall), Specificity, and ROC-AUC.
""")
