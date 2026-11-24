import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error

TEST_FILE = "test.csv"
PREDICTIONS_FILE = "predictions.csv"
ID_COLUMN = "project_id"
TARGET_COLUMN = "hours_spent"


def main():
    actual = pd.read_csv(TEST_FILE)
    predicted = pd.read_csv(PREDICTIONS_FILE)

    for name, df in [("test.csv", actual), ("predictions.csv", predicted)]:
        if ID_COLUMN not in df.columns:
            raise ValueError(f"{name} is missing required column: {ID_COLUMN}")
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"{name} is missing required column: {TARGET_COLUMN}")


    merged = actual[[ID_COLUMN, TARGET_COLUMN]].merge(
        predicted[[ID_COLUMN, TARGET_COLUMN]],
        on=ID_COLUMN,
        how="inner",
        suffixes=("_actual", "_predicted"),
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError("No matching project_id values found between the two files.")

    y_true = pd.to_numeric(merged[f"{TARGET_COLUMN}_actual"], errors="raise")
    y_pred = pd.to_numeric(merged[f"{TARGET_COLUMN}_predicted"], errors="raise")

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Actual or predicted values contain missing/non-finite values.")
    if (y_true < 0).any() or (y_pred < 0).any():
        raise ValueError("RMSLE requires non-negative actual and predicted values.")

    score = np.sqrt(mean_squared_log_error(y_true, y_pred))
    print(f"Projects scored: {len(merged)}")
    print(f"RMSLE: {score:.6f}")


if __name__ == "__main__":
    main()
