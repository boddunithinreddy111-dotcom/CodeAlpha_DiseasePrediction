"""
Clinical Data Preprocessing and Medical Feature Engineering Pipeline.
Handles clinical zero-imputation, physiological ratio engineering,
encoding, and feature scaling.
"""

from pathlib import Path
from typing import Tuple, List
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

TARGET_COL = "HeartDisease"


def engineer_medical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies clinical domain logic and features to raw medical records.
    """
    df = df.copy()

    # In clinical datasets, 0 BP or 0 Cholesterol are measurement errors / missing values
    if "RestingBP" in df.columns:
        df["RestingBP"] = df["RestingBP"].replace(0, np.nan)
    if "Cholesterol" in df.columns:
        df["Cholesterol"] = df["Cholesterol"].replace(0, np.nan)

    # 1. Cardiovascular Reserve Deficit: (220 - Age) - MaxHR
    if "Age" in df.columns and "MaxHR" in df.columns:
        theoretical_max = 220.0 - df["Age"]
        df["HR_Reserve_Deficit"] = theoretical_max - df["MaxHR"]

    # 2. Cholesterol to Age Ratio (Long-term lipid exposure indicator)
    if "Cholesterol" in df.columns and "Age" in df.columns:
        df["Cholesterol_to_Age"] = df["Cholesterol"] / df["Age"].clip(lower=1.0)

    # 3. High Risk Chest Pain Flag (Asymptomatic 'ASY' has highest ischemia probability)
    if "ChestPainType" in df.columns:
        df["Is_Asymptomatic_CP"] = (df["ChestPainType"] == "ASY").astype(int)

    # 4. Oldpeak Stress Indicator (ST depression > 1.5 indicates myocardial ischemia)
    if "Oldpeak" in df.columns:
        df["Is_Severe_Oldpeak"] = (df["Oldpeak"] >= 1.5).astype(int)

    return df


class MedicalDataPreprocessor:
    """
    Scikit-learn compatible clinical feature pipeline.
    """

    def __init__(self):
        self.pipeline = None
        self.num_cols = []
        self.cat_cols = []
        self.feature_names = []

    def fit(self, X: pd.DataFrame):
        X_eng = engineer_medical_features(X)

        self.cat_cols = [
            c for c in ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
            if c in X_eng.columns
        ]
        self.num_cols = [
            c for c in X_eng.columns if c not in self.cat_cols and c != TARGET_COL
        ]

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        self.pipeline = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, self.num_cols),
                ("cat", cat_pipeline, self.cat_cols)
            ]
        )

        self.pipeline.fit(X_eng)

        cat_encoder = self.pipeline.named_transformers_["cat"].named_steps["onehot"]
        cat_feature_names = list(cat_encoder.get_feature_names_out(self.cat_cols))
        self.feature_names = self.num_cols + cat_feature_names

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_eng = engineer_medical_features(X)
        return self.pipeline.transform(X_eng)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, filepath: str or Path):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[SAVED] Clinical preprocessor saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str or Path):
        return joblib.load(filepath)


def prepare_medical_dataset(
    csv_path: str or Path, test_size: float = 0.2, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MedicalDataPreprocessor]:
    """
    Loads raw clinical CSV, engineers features, performs stratified split,
    and returns matrices, labels, and preprocessor.
    """
    df = pd.read_csv(csv_path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' in dataset.")

    y = df[TARGET_COL].astype(int).values
    X = df.drop(columns=[TARGET_COL])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = MedicalDataPreprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    return X_train, X_test, y_train, y_test, preprocessor
