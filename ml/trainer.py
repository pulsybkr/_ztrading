import lightgbm as lgb
import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path
from typing import Optional

from ml.features import build_features, FEATURE_COLUMNS, signals_to_dataframe
from ml.labeling import label_signals
from core.types import Signal, BacktestConfig


DEFAULT_MODEL_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "n_estimators": 400,
    "max_depth": 5,
    "num_leaves": 20,
    "learning_rate": 0.02,
    "min_child_samples": 300,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.5,
    "reg_lambda": 0.5,
    "verbose": -1,
    "n_jobs": -1,
}


def prepare_training_data(signals: list[Signal], candles_m5: pd.DataFrame,
                         candles_h1: pd.DataFrame = None,
                         config: BacktestConfig = None) -> pd.DataFrame:
    df_features = signals_to_dataframe(signals, candles_m5, candles_h1)
    df_labels = label_signals(signals, candles_m5, config)
    
    df = df_features.merge(df_labels[["time", "signal_idx", "label", "barrier", "return"]], 
                          on=["time", "signal_idx"], how="left")
    df = df.fillna(0)
    
    return df


def walk_forward_train(
    signals: list[Signal],
    candles_m5: pd.DataFrame,
    candles_h1: pd.DataFrame = None,
    train_weeks: int = 8,
    test_weeks: int = 2,
    step_weeks: int = 2,
    model_params: dict = None,
    config: BacktestConfig = None,
) -> list[dict]:
    if model_params is None:
        model_params = DEFAULT_MODEL_PARAMS.copy()

    df = prepare_training_data(signals, candles_m5, candles_h1, config)
    df = df.sort_values("time").reset_index(drop=True)

    start = df["time"].min()
    end = df["time"].max()

    train_delta = timedelta(weeks=train_weeks)
    test_delta = timedelta(weeks=test_weeks)
    step_delta = timedelta(weeks=step_weeks)

    fold_results = []
    fold = 0
    current = start

    while current + train_delta + test_delta <= end:
        train_end = current + train_delta
        test_end = train_end + test_delta

        train_mask = (df["time"] >= current) & (df["time"] < train_end)
        test_mask = (df["time"] >= train_end) & (df["time"] < test_end)

        X_train = df.loc[train_mask, FEATURE_COLUMNS]
        y_train = df.loc[train_mask, "label"]
        X_test = df.loc[test_mask, FEATURE_COLUMNS]
        y_test = df.loc[test_mask, "label"]

        if len(X_train) < 50 or len(X_test) < 10:
            current += step_delta
            continue

        model = lgb.LGBMClassifier(**model_params)
        
        effective_min_child = max(30, min(model_params.get("min_child_samples", 300), len(X_train) // 10))
        model.set_params(min_child_samples=effective_min_child)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.log_evaluation(0)],
        )

        probas = model.predict_proba(X_test)[:, 1]
        preds = (probas >= 0.62).astype(int)

        accuracy = (preds == y_test.values).mean()
        
        tp = ((preds == 1) & (y_test.values == 1)).sum()
        fp = ((preds == 1) & (y_test.values == 0)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        recall = (preds[y_test.values == 1] == 1).mean() if (y_test.values == 1).sum() > 0 else 0

        fold_results.append({
            "fold": fold,
            "train_start": current.date(),
            "train_end": train_end.date(),
            "test_start": train_end.date(),
            "test_end": test_end.date(),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "model": model,
        })

        print(f"  Fold {fold}: train {len(X_train)} | test {len(X_test)} | "
              f"acc={accuracy:.3f} | prec={precision:.3f} | rec={recall:.3f}")

        current += step_delta
        fold += 1

    return fold_results


def train_final_model(df: pd.DataFrame, model_params: dict = None) -> lgb.LGBMClassifier:
    if model_params is None:
        model_params = DEFAULT_MODEL_PARAMS.copy()

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    model = lgb.LGBMClassifier(**model_params)
    model.fit(X, y)

    return model


def save_model(model: lgb.LGBMClassifier, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(path)
    print(f"Model saved to {path}")


def load_model(path: str) -> lgb.LGBMClassifier:
    booster = lgb.Booster(model_file=path)
    model = lgb.LGBMClassifier()
    model._Booster__Booster = booster
    return model
