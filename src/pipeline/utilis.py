import pickle
import sys
import os
import importlib
import numpy as np
from dataclasses import dataclass
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.pipeline.exception import CustomException  


class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _set_sklearn_compatibility_attrs(obj):
    try:
        if hasattr(obj, "feature_names_in_") and not hasattr(obj, "_feature_names_in"):
            setattr(obj, "_feature_names_in", getattr(obj, "feature_names_in_"))
    except Exception:
        pass

    try:
        if hasattr(obj, "sparse_output") and not hasattr(obj, "sparse"):
            setattr(obj, "sparse", getattr(obj, "sparse_output"))
    except Exception:
        pass

    if hasattr(obj, "steps"):
        for _, step in getattr(obj, "steps"):
            _set_sklearn_compatibility_attrs(step)

    if hasattr(obj, "transformers"):
        for transformer in getattr(obj, "transformers"):
            if len(transformer) >= 2:
                sub_obj = transformer[1]
                if sub_obj not in ("drop", "passthrough"):
                    _set_sklearn_compatibility_attrs(sub_obj)

    if hasattr(obj, "named_transformers_"):
        for sub_obj in getattr(obj, "named_transformers_").values():
            if sub_obj not in ("drop", "passthrough"):
                _set_sklearn_compatibility_attrs(sub_obj)


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
    
def load_object(file_path:str):
    try:
        with open(file_path, "rb") as file_obj:
            obj = CompatUnpickler(file_obj).load()
            _set_sklearn_compatibility_attrs(obj)
            return obj

    except Exception as e:
        raise CustomException(e,sys)