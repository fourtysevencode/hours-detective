"Runs the entire training and predciton pipeline on train.csv and test.csv placed next to this file"

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test.csv"
OUTPUT_PATH = ROOT / "predictions.csv"

def add_features(frame: pd.DataFrame) -> pd.DataFrame: # yay type checking
    frame = frame.copy # make a copy

    if {"num_commits", "active_commit_days"}.issubset(frame.columns):
        days = pd.to_numeric(frame["active_commit_days"], errors="coerce")
        commits = pd.to_numeric(frame["num_commits"], errors="coerce")
        frame["commits_per_active_day"] = commits.div(days.replace(0, np.nan))
        frame["commits_per_active_day"] = frame["commits_per_active_day"].replace(
            [np.inf, -np.inf], np.nan # handle division by 0 if no commits in an active day, edge case
        )
    return frame # with engineered features

def main():
    if not TRAIN_PATH.is_file() or not TEST_PATH.is_file():
        raise FileNotFoundError("Place train.csv and test.csv next to predict.py")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test.csv"
OUTPUT_PATH = ROOT / "predictions.csv"


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if {"num_commits", "active_commit_days"}.issubset(frame.columns):
        days = pd.to_numeric(frame["active_commit_days"], errors="coerce")
        commits = pd.to_numeric(frame["num_commits"], errors="coerce")
        frame["commits_per_active_day"] = commits.div(days.replace(0, np.nan))
        frame["commits_per_active_day"] = frame["commits_per_active_day"].replace(
            [np.inf, -np.inf], np.nan
        )
    return frame


def main() -> None:
    if not TRAIN_PATH.is_file() or not TEST_PATH.is_file():
        raise FileNotFoundError("Place train.csv and test.csv next to predict.py")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    if "project_id" not in test.columns:
        raise ValueError("test.csv must contain a project_id column")
    if "hours_spent" not in train.columns:
        raise ValueError("train.csv must contain the hours_spent target")

    project_ids = test["project_id"].copy()
    y = pd.to_numeric(train["hours_spent"], errors="raise").clip(lower=0)

    train_features = add_features(train)
    test_features = add_features(test)

    drop_columns = ["project_id", "hours_spent", "log_hours", "hours_bucket"]
    train_features = train_features.drop(columns=[c for c in drop_columns if c in train_features.columns])
    test_features = test_features.drop(columns=[c for c in drop_columns if c in test_features.columns])

    feature_columns = train_features.columns.tolist()
    for column in feature_columns:
        if column not in test_features.columns:
            test_features[column] = np.nan
    test_features = test_features[feature_columns]

    categorical_columns = train_features.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    for column in categorical_columns:
        values = pd.concat(
            [train_features[column], test_features[column]], ignore_index=True
        ).dropna().unique().tolist()
        categories = sorted(values, key=lambda value: str(value))
        train_features[column] = pd.Categorical(train_features[column], categories=categories)
        test_features[column] = pd.Categorical(test_features[column], categories=categories)

    model = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=42)

    model.fit(train_features, np.log1p(y))

    predicted_log_hours = model.predict(test_features)
    predictions = np.maximum(np.expm1(predicted_log_hours), 0.0)

    output = pd.DataFrame(
        {
            "project_id": project_ids,
            "hours_spent": predictions,
        }
    ) # prescribed by 'what we run'
    output.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(output)} predictions to {OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()