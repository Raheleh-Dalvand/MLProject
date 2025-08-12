import os
import sys
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self) -> None:
        self.config = DataTransformationConfig()

    def _build_preprocessor(self, dataframe: pd.DataFrame) -> ColumnTransformer:
        try:
            # Identify columns by dtype, assuming numeric scores and categorical demographics
            numeric_columns = [
                column_name
                for column_name in dataframe.columns
                if dataframe[column_name].dtype in ["int64", "float64", "int32", "float32"]
            ]

            # Fallback: if numeric columns were read as objects (due to quotes), coerce them later
            candidate_numeric_columns = [
                "math score",
                "reading score",
                "writing score",
            ]
            if not numeric_columns:
                numeric_columns = [
                    column_name for column_name in candidate_numeric_columns if column_name in dataframe.columns
                ]

            categorical_columns = [
                column_name for column_name in dataframe.columns if column_name not in numeric_columns
            ]

            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_pipeline, numeric_columns),
                    ("cat", categorical_pipeline, categorical_columns),
                ]
            )
            return preprocessor
        except Exception as error:
            raise CustomException(error, sys)

    def initiate_data_transformation(
        self, train_csv_path: str, test_csv_path: str
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        try:
            train_dataframe = pd.read_csv(train_csv_path)
            test_dataframe = pd.read_csv(test_csv_path)

            # Attempt to coerce score columns to numeric in case they were parsed as strings
            for column_name in ["math score", "reading score", "writing score"]:
                if column_name in train_dataframe.columns:
                    train_dataframe[column_name] = pd.to_numeric(
                        train_dataframe[column_name], errors="coerce"
                    )
                if column_name in test_dataframe.columns:
                    test_dataframe[column_name] = pd.to_numeric(
                        test_dataframe[column_name], errors="coerce"
                    )

            # Define a simple modeling target to enable a full pipeline
            target_column_name = "math score" if "math score" in train_dataframe.columns else None

            if target_column_name is None:
                raise CustomException("Target column 'math score' not found in training data", sys)

            input_feature_train_df = train_dataframe.drop(columns=[target_column_name])
            target_feature_train_df = train_dataframe[target_column_name]

            input_feature_test_df = test_dataframe.drop(columns=[target_column_name])
            target_feature_test_df = test_dataframe[target_column_name]

            preprocessor = self._build_preprocessor(input_feature_train_df)

            input_feature_train_array = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessor.transform(input_feature_test_df)

            # Save the fitted preprocessor
            save_object(self.config.preprocessor_obj_file_path, preprocessor)

            # Combine features and target for downstream training
            train_array = np.c_[input_feature_train_array.toarray() if hasattr(input_feature_train_array, "toarray") else input_feature_train_array, target_feature_train_df.to_numpy()]
            test_array = np.c_[input_feature_test_array.toarray() if hasattr(input_feature_test_array, "toarray") else input_feature_test_array, target_feature_test_df.to_numpy()]

            return train_array, test_array, self.config.preprocessor_obj_file_path
        except Exception as error:
            raise CustomException(error, sys)