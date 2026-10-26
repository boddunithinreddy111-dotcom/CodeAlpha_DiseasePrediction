# 🫀 CodeAlpha Internship — Task 4: Disease Prediction from Medical Data

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)
![XGBoost](https://img.shields.io/badge/Algorithm-XGBoost-red.svg)
![KaggleHub](https://img.shields.io/badge/Data-UCI%20Heart%20Disease-20BEFF.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit%20Medical-FF4B4B.svg)
![Internship](https://img.shields.io/badge/CodeAlpha-Machine%20Learning%20Task%204-green.svg)

---

## 📌 Project Overview
This repository contains the complete implementation of **Task 4: Disease Prediction from Medical Data** for the **CodeAlpha Machine Learning Internship**.

The objective is to accurately predict the likelihood of cardiovascular disease and heart failure from structured patient clinical biomarkers, vital signs, electrocardiogram (ECG) indicators, and stress test data.

### 🎯 Key Highlights
- **Gold-Standard Clinical Dataset**: Sourced from the **UCI Heart Disease** benchmark (`fedesoriano/heart-failure-prediction` via KaggleHub), aggregating patient records across 5 clinical research institutions.
- **Cardiovascular Feature Engineering**: Extracted physiological markers including Cardiovascular Reserve Deficit, Lipid Exposure Index (Cholesterol-to-Age Ratio), and Ischemia Severity flags.
- **Multi-Algorithm Benchmarking**: Implemented and benchmarked **Support Vector Machines (SVM - RBF Kernel)**, **Logistic Regression**, **Random Forest**, **Gradient Boosting**, and **XGBoost**.
- **Clinical Performance Assessment**: Evaluated on **Sensitivity (Recall), Specificity, Precision, F1-Score, and ROC-AUC**.
- **Clinical Decision Support Dashboard**: Built an interactive **Streamlit Web Application** where medical practitioners can input patient vitals, assess cardiovascular risk, and receive real-time clinical diagnostic recommendations.

---

## 📂 Repository Structure

```text
CodeAlpha_DiseasePrediction/
│
├── data/
│   ├── raw/                  # heart.csv from KaggleHub / generator
│   └── processed/            # Scaled & encoded patient feature matrices
│
├── notebooks/
│   └── 01_disease_prediction_eda_and_modeling.ipynb  # Interactive Jupyter Notebook for VS Code
│
├── src/
│   ├── __init__.py           # Package marker
│   ├── download_data.py      # KaggleHub downloader + clinical data generator
│   ├── preprocessing.py      # Clinical feature engineering, imputation & scaling pipeline
│   ├── train.py              # Multi-model training, benchmarking & serialization
│   ├── evaluate.py           # Clinical metrics, sensitivity/specificity, ROC & confusion matrix
│   └── predict.py            # Diagnostic inference & clinical advice engine
│
├── visualizer/
│   └── app.py                # Streamlit clinical diagnostic web dashboard
│
├── models/                   # Serialized model artifacts (.joblib)
│   ├── best_disease_model.joblib
│   └── preprocessor.joblib
│
├── reports/
│   ├── model_benchmark_results.csv
│   └── figures/              # Generated plots (ROC curves, Confusion Matrix, Feature Importance)
│
├── requirements.txt          # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # Complete project documentation
```

---

## 🩺 Patient Clinical Biomarkers Evaluated

The model analyzes 11 core physiological and diagnostic biomarkers:

| Biomarker | Description | Normal Clinical Range |
| :--- | :--- | :--- |
| **Age** | Patient age in years | 25 – 85 years |
| **Sex** | Biological sex | Male (`M`) / Female (`F`) |
| **ChestPainType** | Angina type | Typical (`TA`), Atypical (`ATA`), Non-Anginal (`NAP`), Asymptomatic (`ASY`) |
| **RestingBP** | Resting systolic blood pressure | Normal: $<120$ mm Hg; Hypertensive: $\ge 140$ mm Hg |
| **Cholesterol** | Serum cholesterol level | Desirable: $<200$ mg/dl; High: $\ge 240$ mg/dl |
| **FastingBS** | Fasting blood glucose level | $0$ (Normal) / $1$ ($>120$ mg/dl, pre-diabetic/diabetic) |
| **RestingECG** | Electrocardiogram reading | Normal, ST-T wave abnormality (`ST`), Left Ventricular Hypertrophy (`LVH`) |
| **MaxHR** | Maximum exercise heart rate | 60 – 210 bpm |
| **ExerciseAngina** | Angina induced by stress exercise | Yes (`Y`) / No (`N`) |
| **Oldpeak** | ST depression induced by exercise | 0.0 – 6.0 mm (Values $\ge 1.5$ mm signal ischemia) |
| **ST_Slope** | Peak exercise ST segment slope | Upsloping (`Up`), Flat (`Flat`), Downsloping (`Down`) |

---

## ⚙️ Physiological Feature Engineering

1. **Cardiovascular Reserve Deficit**:
   $$\text{HR Reserve Deficit} = (220 - \text{Age}) - \text{MaxHR}$$
   *Measures the discrepancy between theoretical maximum heart rate and achieved heart rate during stress testing.*
2. **Lipid Exposure Index**:
   $$\text{Cholesterol-to-Age Ratio} = \frac{\text{Cholesterol}}{\text{Age}}$$
3. **Ischemia Severity Flag**:
   $$\text{Is\_Severe\_Oldpeak} = \mathbb{I}(\text{Oldpeak} \ge 1.5\text{ mm})$$
4. **Asymptomatic Ischemia Marker**:
   $$\text{Is\_Asymptomatic\_CP} = \mathbb{I}(\text{ChestPainType} == \text{'ASY'})$$

---

## 📈 Clinical Model Benchmark Results

Models were evaluated on a stratified test holdout ($20\%$):

| Algorithm | Accuracy | Precision | Sensitivity (Recall) | Specificity | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM (RBF Kernel)** | **0.8417** | **0.9130** | **0.8596** | **0.7971** | **0.8855** | **0.8906** |
| **Logistic Regression** | 0.8333 | 0.8970 | 0.8655 | 0.7536 | 0.8810 | **0.9019** |
| **XGBoost Classifier** | 0.8333 | 0.8970 | 0.8655 | 0.7536 | 0.8810 | 0.8845 |
| **Random Forest Classifier** | 0.8333 | 0.9018 | 0.8596 | 0.7681 | 0.8802 | **0.9033** |
| **Gradient Boosting** | 0.8208 | 0.8855 | 0.8596 | 0.7246 | 0.8724 | 0.8835 |

> **Clinical Finding**: **Support Vector Machines (SVM)** achieved the highest precision (91.3%) and F1-score (0.8855), minimizing false alarms while preserving critical diagnostic sensitivity. Tree ensembles revealed that **ST Segment Slope**, **Chest Pain Type**, and **Oldpeak** are the top predictors of coronary artery disease.

---

## 🚀 Quickstart Guide for VS Code

### 1. Open the Project in VS Code
```bash
cd CodeAlpha_DiseasePrediction
```

### 2. Activate Virtual Environment & Install Dependencies
```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 3. Run the Clinical Training Pipeline
```powershell
python src/train.py
```
*(Benchmarks all 5 algorithms, saves the best model to `models/`, and generates ROC curves, confusion matrices, and feature importance charts in `reports/figures/`).*

### 4. Test Diagnostic Inference (CLI)
```powershell
python src/predict.py
```

### 5. Launch the Interactive Medical Dashboard
```powershell
streamlit run visualizer/app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser to evaluate patient clinical records in real-time!

### 6. Interactive Jupyter Notebook
Open [`notebooks/01_disease_prediction_eda_and_modeling.ipynb`](notebooks/01_disease_prediction_eda_and_modeling.ipynb) in VS Code, choose your `venv` kernel, and execute cells interactively.

---

## 🎥 LinkedIn Presentation Guide (For Interns)

### 🎬 60-Second Video Walkthrough Script:
1. **Introduction (10 sec)**: *"Hello everyone! As part of my Machine Learning Internship at CodeAlpha, I completed Task 4: Disease Prediction from Medical Data."*
2. **Clinical Problem & Dataset (20 sec)**: *"Early detection of heart disease saves lives. I used the UCI Heart Disease dataset containing 11 vital clinical biomarkers like blood pressure, cholesterol, resting ECG, and ST depression."*
3. **Algorithms & Performance (15 sec)**: *"I engineered cardiovascular reserve features and benchmarked SVM, Logistic Regression, Random Forest, and XGBoost. The SVM model achieved a 91.3% precision, 86% sensitivity, and an F1-score of 0.88."*
4. **Live App Demo (15 sec)**: *(Switch to Streamlit app)* *"In this interactive clinical decision support dashboard, doctors can input patient vitals to receive immediate cardiac risk scores and clinical recommendations."*

### 📝 LinkedIn Post Template:
```markdown
🚀 Thrilled to share my third project as a Machine Learning Intern at @CodeAlpha!

🫀 Task 4: Disease Prediction from Structured Medical Data
In this project, I developed an end-to-end clinical machine learning diagnostic system to predict cardiovascular disease risk from patient health records.

🔍 Key Implementation Highlights:
• Sourced clinical data from the UCI Heart Disease benchmark repository via KaggleHub.
• Engineered physiological features: Cardiovascular Reserve Deficit, Lipid Exposure Index (Cholesterol-to-Age Ratio), and Ischemia flags.
• Benchmarked multiple clinical algorithms: SVM (RBF Kernel), Logistic Regression, Random Forest, and XGBoost.
• Evaluated models with clinical metrics: Sensitivity (Recall: 86%), Specificity (80%), Precision (91.3%), and ROC-AUC (0.90).
• Built an interactive Streamlit clinical decision support dashboard for real-time risk assessment and doctor recommendations.

💻 GitHub Repository:
https://github.com/<your-username>/CodeAlpha_DiseasePrediction

A big thank you to @CodeAlpha for this valuable opportunity to apply machine learning in healthcare!

#MachineLearning #HealthcareAI #DataScience #CodeAlpha #Internship #Python #ScikitLearn #Streamlit #Cardiology #HealthTech
```

---

## 📜 License & Acknowledgments
- Developed as part of the **CodeAlpha Machine Learning Internship**.
- Built with Scikit-Learn, XGBoost, Pandas, and Streamlit.
