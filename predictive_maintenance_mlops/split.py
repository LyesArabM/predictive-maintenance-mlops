from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))["split"]

    df = pd.read_csv(params["raw_path"])

    # Stratify on failure AND product type 
    strat = df["Machine failure"].astype(str) + "_" + df["Type"]

    train, test = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["seed"],
        stratify=strat,
    )

    train_path = Path(params["train_path"])
    test_path = Path(params["test_path"])
    train_path.parent.mkdir(parents=True, exist_ok=True)

    # All columns are kept; the features are chosen in train.py
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    print(f"train: {len(train)} rows, failure rate {train['Machine failure'].mean():.4f}")
    print(f"test:  {len(test)} rows, failure rate {test['Machine failure'].mean():.4f}")


if __name__ == "__main__":
    main()