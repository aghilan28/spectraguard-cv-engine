"""Hyperparameter optimization and cross-validation for ML models."""

import json
import os
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from ..models.config import TrainingConfig
from ..models.trainer import ModelTrainer


class HyperparameterTuner:
    """
    Executes randomized cross-validation search strategies to find
    optimal parameters and returns configured ModelTrainers.
    """

    def __init__(
        self,
        model_type: str,
        param_distributions: Dict[str, Any],
        n_iter: int = 10,
        cv: int = 3,
        random_state: int = 42,
    ):
        if model_type not in ["random_forest", "xgboost"]:
            raise ValueError(
                "Unsupported model_type. Must be 'random_forest' or 'xgboost'."
            )

        self.model_type = model_type
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.cv = cv
        self.random_state = random_state

        self.best_params_: Dict[str, Any] = {}
        self.best_score_: float = 0.0

    def optimize(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[Dict[str, Any], float]:
        """
        Runs the randomized search cross-validation on the provided training set.

        Returns:
            Tuple containing the best parameters dictionary and the best CV accuracy score.
        """
        if self.model_type == "random_forest":
            base_estimator = RandomForestClassifier(random_state=self.random_state)
        elif self.model_type == "xgboost":
            base_estimator = XGBClassifier(
                random_state=self.random_state, eval_metric="logloss", n_jobs=-1
            )

        search = RandomizedSearchCV(
            estimator=base_estimator,
            param_distributions=self.param_distributions,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring="accuracy",
            random_state=self.random_state,
            n_jobs=-1,
        )

        search.fit(X_train, y_train)

        self.best_params_ = search.best_params_
        self.best_score_ = float(search.best_score_)

        return self.best_params_, self.best_score_

    def get_best_trainer(self, checkpoint_dir: str) -> ModelTrainer:
        """
        Constructs a ModelTrainer initialized with the optimized parameters.
        """
        if not self.best_params_:
            raise RuntimeError(
                "Must call optimize() successfully before retrieving best trainer."
            )

        config = TrainingConfig(
            model_type=self.model_type,
            random_seed=self.random_state,
            checkpoint_dir=checkpoint_dir,
            hyperparameters=self.best_params_,
        )
        return ModelTrainer(config)

    def save_report(self, filepath: str) -> None:
        """
        Persists the optimization results and metadata to a JSON report.
        """
        if not self.best_params_:
            raise RuntimeError(
                "Must call optimize() successfully before saving report."
            )

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

        report = {
            "model_type": self.model_type,
            "best_cv_accuracy": self.best_score_,
            "best_parameters": self.best_params_,
            "optimization_config": {
                "cv_folds": self.cv,
                "n_iterations": self.n_iter,
                "random_seed": self.random_state,
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)
