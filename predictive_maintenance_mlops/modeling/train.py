import json
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

TARGET = "Machine failure"
# Not features: identifiers + the target decomposed (leakage)
DROP = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
CAT = ["Type"]


def build_pipeline(num_cols: list[str], seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("cat", OneHotEncoder(), CAT),
                        ("num", StandardScaler(), num_cols),
                    ]
                ),
            ),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=seed
                ),
            ),
        ]
    )


def main() -> None:
    all_params = yaml.safe_load(open("params.yaml"))
    params = all_params["train"]
    split_params = all_params["split"]

    train = pd.read_csv(split_params["train_path"])
    X = train.drop(columns=DROP + [TARGET])
    y = train[TARGET]
    num_cols = [c for c in X.columns if c not in CAT]

    pipeline = build_pipeline(num_cols, params["seed"])

    cv = RepeatedStratifiedKFold(
        n_splits=params["n_splits"],
        n_repeats=params["n_repeats"],
        random_state=params["seed"],
    )
    scoring = {
        "pr_auc": "average_precision",
        "recall": "recall",
        "precision": "precision",
        "f2": make_scorer(fbeta_score, beta=2),
    }
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)

    metrics = {
        name: {
            "mean": float(scores[f"test_{name}"].mean()),
            "std": float(scores[f"test_{name}"].std()),
        }
        for name in scoring
    }
    print(json.dumps(metrics, indent=2))

    # Final model on the whole train set
    pipeline.fit(X, y)

    model_path = Path(params["model_path"])
    metrics_path = Path(params["metrics_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()