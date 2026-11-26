import json
import os
from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from predictive_maintenance_mlops.features import DERIVED_NAMES, DerivedFeatures

TARGET = "Machine failure"
# Not features: identifiers + the target decomposed (leakage)
DROP = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
CAT = ["Type"]
MODEL_NAME = "ai4i-failure-model"

# Our own classes, safe to trust when MLflow saves the model with skops
TRUSTED_TYPES = [
    "predictive_maintenance_mlops.features.DerivedFeatures",
    "xgboost.core.Booster",
    "xgboost.sklearn.XGBClassifier",
]


def make_model(name, seed, xgb_params, scale_pos_weight):
    if name == "logreg":
        return LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=seed
        )
    if name == "xgboost":
        return XGBClassifier(
            **xgb_params,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=seed,
        )
    raise ValueError(f"unknown model: {name}")


def build_pipeline(exp, base_num, seed, xgb_params, scale_pos_weight):
    num_cols = base_num + (DERIVED_NAMES if exp["derived"] else [])
    steps = []
    if exp["derived"]:
        steps.append(("features", DerivedFeatures()))
    steps.append(
        (
            "prep",
            ColumnTransformer(
                [
                    ("cat", OneHotEncoder(), CAT),
                    ("num", StandardScaler(), num_cols),
                ]
            ),
        )
    )
    steps.append(
        ("model", make_model(exp["model"], seed, xgb_params, scale_pos_weight))
    )
    return Pipeline(steps)


def evaluate(pipeline, X, y, cv):
    scoring = {
        "pr_auc": "average_precision",
        "recall": "recall",
        "precision": "precision",
        "f2": make_scorer(fbeta_score, beta=2),
    }
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)
    return {
        name: {
            "mean": float(scores[f"test_{name}"].mean()),
            "std": float(scores[f"test_{name}"].std()),
        }
        for name in scoring
    }


def log_to_mlflow(experiments, results, best_name, best_pipeline, params, columns):
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    mlflow.set_experiment("ai4i-failure")
    for exp in experiments:
        with mlflow.start_run(run_name=exp["name"]):
            mlflow.log_params(
                {
                    "model": exp["model"],
                    "derived_features": exp["derived"],
                    "n_splits": params["n_splits"],
                    "n_repeats": params["n_repeats"],
                    "seed": params["seed"],
                    "input_columns": ",".join(columns),
                }
            )
            for metric, v in results[exp["name"]].items():
                mlflow.log_metric(f"{metric}_mean", v["mean"])
                mlflow.log_metric(f"{metric}_std", v["std"])

            if exp["name"] == best_name:
                mlflow.sklearn.log_model(
                    best_pipeline,
                    name="model",
                    registered_model_name=MODEL_NAME,
                    skops_trusted_types=TRUSTED_TYPES,
                )

    # A new model is always a challenger. The promotion gate (step 2.10)
    # decides later whether it replaces the champion.
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    newest = max(versions, key=lambda v: int(v.version))
    client.set_registered_model_alias(MODEL_NAME, "challenger", newest.version)


def main() -> None:
    all_params = yaml.safe_load(open("params.yaml"))
    params = all_params["train"]
    xgb_params = all_params["xgboost"]
    split_params = all_params["split"]

    train = pd.read_csv(split_params["train_path"])
    X = train.drop(columns=DROP + [TARGET])
    y = train[TARGET]
    base_num = [c for c in X.columns if c not in CAT]

    # Class ratio on the train set only (a single global constant)
    scale_pos_weight = float((y == 0).sum() / (y == 1).sum())

    cv = RepeatedStratifiedKFold(
        n_splits=params["n_splits"],
        n_repeats=params["n_repeats"],
        random_state=params["seed"],
    )

    experiments = params["experiments"]
    results = {}
    for exp in experiments:
        pipe = build_pipeline(
            exp, base_num, params["seed"], xgb_params, scale_pos_weight
        )
        results[exp["name"]] = evaluate(pipe, X, y, cv)
        pr = results[exp["name"]]["pr_auc"]
        print(f"{exp['name']:<18} PR-AUC {pr['mean']:.3f} +/- {pr['std']:.3f}")

    # Selection uses cross-validation on the train set only, never the test set
    best_name = max(results, key=lambda n: results[n]["pr_auc"]["mean"])
    best_exp = next(e for e in experiments if e["name"] == best_name)
    print(f"best: {best_name}")

    best_pipeline = build_pipeline(
        best_exp, base_num, params["seed"], xgb_params, scale_pos_weight
    )
    best_pipeline.fit(X, y)

    model_path = Path(params["model_path"])
    metrics_path = Path(params["metrics_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_pipeline, model_path)
    with open(metrics_path, "w") as f:
        json.dump({"best": best_name, "experiments": results}, f, indent=2)

    # Log to MLflow only if credentials are configured, so a teammate or
    # the CI can still run the pipeline without them
    if os.getenv("MLFLOW_TRACKING_URI"):
        log_to_mlflow(
            experiments, results, best_name, best_pipeline, params, list(X.columns)
        )


if __name__ == "__main__":
    main()