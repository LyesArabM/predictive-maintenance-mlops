from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def main() -> None:
    p = yaml.safe_load(Path("params.yaml").read_text())["split"]

    df = pd.read_csv(p["raw_path"])

    # Stratification sur la panne and le type de produit 
    strata = df["Machine failure"].astype(str) + "_" + df["Type"].astype(str)

    train, test = train_test_split(
        df,
        test_size=p["test_size"],
        random_state=p["seed"],
        stratify=strata,
    )

    Path(p["train_path"]).parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(p["train_path"], index=False)
    test.to_csv(p["test_path"], index=False)

    print(f"train: {len(train)} lignes, taux de panne {train['Machine failure'].mean():.4f}")
    print(f"test : {len(test)} lignes, taux de panne {test['Machine failure'].mean():.4f}")


if __name__ == "__main__":
    main()