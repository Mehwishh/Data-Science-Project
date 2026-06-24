import os
import sys
import catboost
import numpy as np

from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.metrics import r2_score

from xgboost import XGBRegressor

from src.pipeline.exception import CustomException
from src.pipeline.logger import logging
from src.pipeline.utilis import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):

        try:

            logging.info(
                "Splitting training and testing data"
            )

            train_array = np.asarray(train_array)
            test_array = np.asarray(test_array)

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {

                "XGBRegressor":
                    XGBRegressor(),

                "RandomForestRegressor":
                    RandomForestRegressor(),

                "KNeighborsRegressor":
                    KNeighborsRegressor(),

                "GradientBoostingRegressor":
                    GradientBoostingRegressor(),

                "DecisionTreeRegressor":
                    DecisionTreeRegressor(),

                "LinearRegression":
                    LinearRegression(),

                "CatBoostRegressor":
                    catboost.CatBoostRegressor(
                        verbose=False
                    ),

                "AdaBoostRegressor":
                    AdaBoostRegressor()
            }

            param = {

                "DecisionTreeRegressor": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson"
                    ]
                },

                "GradientBoostingRegressor": {
                    "learning_rate": [
                        0.1,
                        0.01,
                        0.05,
                        0.001
                    ],
                    "n_estimators": [
                        8,
                        16,
                        32,
                        64,
                        128,
                        256
                    ]
                },

                "RandomForestRegressor": {
                    "n_estimators": [
                        8,
                        16,
                        32,
                        64,
                        128,
                        256
                    ]
                },

                "XGBRegressor": {
                    "learning_rate": [
                        0.1,
                        0.01,
                        0.05,
                        0.001
                    ],
                    "n_estimators": [
                        8,
                        16,
                        32,
                        64,
                        128,
                        256
                    ]
                },

                "KNeighborsRegressor": {
                    "n_neighbors": [
                        5,
                        7,
                        9,
                        11
                    ]
                },

                "CatBoostRegressor": {
                    "depth": [
                        6,
                        8,
                        10
                    ],
                    "learning_rate": [
                        0.1,
                        0.01,
                        0.05,
                        0.001
                    ],
                    "iterations": [
                        30,
                        50,
                        100
                    ]
                },

                "AdaBoostRegressor": {
                    "learning_rate": [
                        0.1,
                        0.01,
                        0.05,
                        0.001
                    ],
                    "n_estimators": [
                        8,
                        16,
                        32,
                        64,
                        128,
                        256
                    ]
                },

                "LinearRegression": {}
            }

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=param
            )

            print("\nModel Performance Report")
            print("-" * 50)

            for model_name, score in model_report.items():
                print(
                    f"{model_name}: "
                    f"{score:.4f} "
                    f"({score * 100:.2f}%)"
                )

            logging.info(
                f"Model Report: {model_report}"
            )

            best_model_score = max(
                model_report.values()
            )

            best_model_name = list(
                model_report.keys()
            )[
                list(model_report.values()).index(
                    best_model_score
                )
            ]

            best_model = models[
                best_model_name
            ]

            print("\nBest Model")
            print("-" * 50)
            print(f"Model Name : {best_model_name}")
            print(f"R2 Score   : {best_model_score:.4f}")
            print(f"Percentage : {best_model_score * 100:.2f}%")

            logging.info(
                f"Best Model Found: {best_model_name}"
            )

            best_model.fit(
                X_train,
                y_train
            )

            predicted = best_model.predict(
                X_test
            )

            r2_square = r2_score(
                y_test,
                predicted
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info(
                "Best model saved successfully"
            )

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)