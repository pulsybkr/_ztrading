import optuna
from optuna.samplers import TPESampler
import numpy as np
from typing import Optional

from ml.trainer import walk_forward_train, DEFAULT_MODEL_PARAMS
from core.types import BacktestConfig


def optimize_hyperparameters(
    signals,
    candles_m5,
    candles_h1,
    n_trials: int = 50,
    train_weeks: int = 8,
    test_weeks: int = 2,
    config: BacktestConfig = None,
) -> dict:
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "num_leaves": trial.suggest_int("num_leaves", 10, 40),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 100, 500),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.8),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 1.0, log=True),
            "objective": "binary",
            "metric": "binary_logloss",
            "verbose": -1,
            "n_jobs": -1,
        }

        try:
            fold_results = walk_forward_train(
                signals=signals,
                candles_m5=candles_m5,
                candles_h1=candles_h1,
                train_weeks=train_weeks,
                test_weeks=test_weeks,
                step_weeks=test_weeks,
                model_params=params,
                config=config,
            )

            if not fold_results:
                return 0.0

            avg_precision = np.mean([f["precision"] for f in fold_results])
            avg_accuracy = np.mean([f["accuracy"] for f in fold_results])

            return avg_precision * 0.7 + avg_accuracy * 0.3

        except Exception as e:
            return 0.0

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
    }
