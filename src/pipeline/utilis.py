import pickle
import sys
import os
import numpy as np
from dataclasses import dataclass
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.pipeline.exception import CustomException  

def save_object(file_path:str,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
def evaluate_models(X_train,y_train,X_test,y_test,models:dict,param:dict):
    try:

        report = {}

        for model_name, model in models.items():

            gs = GridSearchCV(
                estimator=model,
                param_grid=param[model_name],
                cv=3
            )

            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)

            model.fit(X_train,y_train)

            y_test_pred = model.predict(X_test)

            test_model_score = r2_score(
                y_test,
                y_test_pred
            )

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e,sys)