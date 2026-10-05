from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import shap


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "src" / "data" / "raw_placement_data.csv"
OUT = ROOT / "reports" / "figures"

OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Features and target
# --------------------------------------------------

FEATURES = [
    "branch",
    "college_tier",
    "cgpa",
    "backlogs",
    "coding_skill_score",
    "communication_skill_score",
    "internships_count",
    "projects_count"
]

TARGET = "placement_status"


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

def load():

    df = pd.read_csv(DATA)

    df.columns = df.columns.str.strip()

    # Keep only the required Lab 8 columns
    df = df[FEATURES + [TARGET]].dropna()

    # Convert college tier:
    # "Tier 1" -> 1
    # "Tier 2" -> 2
    # "Tier 3" -> 3
    if not pd.api.types.is_numeric_dtype(df["college_tier"]):

        df["college_tier"] = (
            df["college_tier"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
            .astype(float)
        )

    # One-hot encode branch
    X = pd.get_dummies(
        df[FEATURES],
        columns=["branch"],
        dtype=float
    )

    # Process target
    y = df[TARGET]

    if not pd.api.types.is_numeric_dtype(y):

        labels = sorted(y.astype(str).unique())

        if len(labels) != 2:
            raise ValueError(
                "placement_status must be binary"
            )

        y = (
            y.astype(str)
            .map({
                labels[0]: 0,
                labels[1]: 1
            })
        )

    return X, y


# --------------------------------------------------
# SHAP analysis
# --------------------------------------------------

def shap_report(model, X, name):

    print(f"\nGenerating SHAP analysis for {name}...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):

        if len(shap_values) > 1:
            values = shap_values[1]
        else:
            values = shap_values[0]

    elif len(getattr(shap_values, "shape", ())) == 3:

        values = shap_values[:, :, 1]

    else:

        values = shap_values

    # --------------------------------------------------
    # Feature importance
    # --------------------------------------------------

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Mean_Absolute_SHAP": np.abs(values).mean(axis=0)
    })

    importance = importance.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    )

    importance.to_csv(
        OUT / f"{name}_shap_feature_importance.csv",
        index=False
    )

    # --------------------------------------------------
    # SHAP summary plot
    # --------------------------------------------------

    plt.figure()

    shap.summary_plot(
        values,
        X,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        OUT / f"{name}_shap_summary.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------
    # SHAP dependence plot
    # --------------------------------------------------

    top_feature = importance.iloc[0]["Feature"]

    plt.figure()

    shap.dependence_plot(
        top_feature,
        values,
        X,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        OUT / f"{name}_shap_dependence.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nTop features ({name}):")
    print(
        importance.head(10).to_string(index=False)
    )


# --------------------------------------------------
# Train XGBoost and LightGBM
# --------------------------------------------------

def run():

    print("Loading placement dataset...")

    X, y = load()

    print(f"Dataset shape: {X.shape}")

    # --------------------------------------------------
    # Train-test split
    # --------------------------------------------------

    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # --------------------------------------------------
    # Models
    # --------------------------------------------------

    models = {

        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42
        ),

        "lightgbm": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )
    }

    # --------------------------------------------------
    # Train models
    # --------------------------------------------------

    rows = []

    for name, model in models.items():

        print("\n" + "=" * 40)
        print(f"Training {name.upper()}")
        print("=" * 40)

        start_time = time.perf_counter()

        model.fit(
            Xtr,
            ytr
        )

        training_time = round(
            time.perf_counter() - start_time,
            4
        )

        # Predictions
        predictions = model.predict(Xte)

        # Accuracy
        accuracy = round(
            accuracy_score(
                yte,
                predictions
            ),
            4
        )

        print(
            f"{name.upper()} | "
            f"Accuracy: {accuracy} | "
            f"Training Time: {training_time}s"
        )

        print("\nClassification Report:")
        print(
            classification_report(
                yte,
                predictions
            )
        )

        # SHAP
        shap_report(
            model,
            Xte,
            name
        )

        rows.append([
            name,
            accuracy,
            training_time
        ])

    # --------------------------------------------------
    # Save model comparison
    # --------------------------------------------------

    comparison = pd.DataFrame(
        rows,
        columns=[
            "Model",
            "Accuracy",
            "Training_Time_Seconds"
        ]
    )

    comparison.to_csv(
        OUT / "xgb_lightgbm_comparison.csv",
        index=False
    )

    print("\n" + "=" * 40)
    print("MODEL COMPARISON")
    print("=" * 40)

    print(comparison)

    print("\nAll Lab 8 outputs saved in:")
    print(OUT)


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":
    run()