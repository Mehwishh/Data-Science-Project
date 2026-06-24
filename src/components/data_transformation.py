import sys
import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.pipeline.exception import CustomException
from src.pipeline.logger import logging
from src.pipeline.utilis import save_object

class DataTransformationConfig:

    preprocessor_obj_file_path = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):

        try:

            numerical_columns = [
                "writing_score",
                "reading_score"
            ]

            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]


            numerical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )


            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "onehotencoder",
                        OneHotEncoder()
                    ),
                    (
                        "scaler",
                        StandardScaler(with_mean=False)
                    )
                ]
            )


            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "numerical_pipeline",
                        numerical_pipeline,
                        numerical_columns
                    ),
                    (
                        "categorical_pipeline",
                        categorical_pipeline,
                        categorical_columns
                    )
                ]
            )


            logging.info("Preprocessor object created")

            return preprocessor


        except Exception as e:
            raise CustomException(e, sys)



    def initiate_data_transformation(
            self,
            train_path,
            test_path
    ):

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)


            logging.info(
                "Train and test data loaded"
            )


            preprocessing_obj = self.get_data_transformer_object()


            target_column_name = "math_score"


            # Independent features
            input_feature_train_df = train_df.drop(
                columns=[target_column_name]
            )


            input_feature_test_df = test_df.drop(
                columns=[target_column_name]
            )


            # Dependent feature
            target_feature_train_df = train_df[
                target_column_name
            ]


            target_feature_test_df = test_df[
                target_column_name
            ]


            logging.info(
                "Applying preprocessing on train data"
            )


            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )


            logging.info(
                "Applying preprocessing on test data"
            )


            input_feature_test_arr = preprocessing_obj.transform(
                input_feature_test_df
            )


            train_arr = pd.DataFrame(
                input_feature_train_arr
            )


            test_arr = pd.DataFrame(
                input_feature_test_arr
            )


            train_arr[target_column_name] = (
                target_feature_train_df.values
            )


            test_arr[target_column_name] = (
                target_feature_test_df.values
            )


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )


            logging.info(
                "Preprocessor saved successfully"
            )


            return (
                train_arr,
                test_arr
            )


        except Exception as e:
            raise CustomException(e, sys)



if __name__ == "__main__":

    from src.components.data_ingestion import DataIngestion

    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()

    train_arr, test_arr = (
        data_transformation.initiate_data_transformation(
            train_data,
            test_data
        )
    )

    print(train_arr.head())
    print(test_arr.head())