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

            # Convert DataFrame to NumPy array if needed
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

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )
            print("\nModel Performance Report")
            print("-" * 50)

            for model_name, score in model_report.items():
               print(f"{model_name}: {score:.4f} ({score*100:.2f}%)")
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

            logging.info(
                f"Best Model Found: {best_model_name}"
            )

            logging.info(
                f"Best Model Score: {best_model_score}"
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