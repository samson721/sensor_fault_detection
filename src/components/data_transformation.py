import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import  RobustScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import customException
from src.logger import logging
from src.utils import save_object
import os

@dataclass
class DatatransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DatatransformationConfig()

    def get_data_transformer_object(self):
        try:

            # define Custom function to replace "NA" with np.nan
            replace_na_with_nan = lambda X: np.where(X == 'na', np.nan, X) 
            
            #define the steps for preprocessor pipleline
            nan_replacement_step = ('nan_replacement',FunctionTransformer(replace_na_with_nan))

            imputer_step = ('imputer', SimpleImputer(strategy='constant', fill_value=0))
            scaler_step = ('scaler', RobustScaler())

            preprocessor = Pipeline(
                steps=[
                    nan_replacement_step,
                    imputer_step,
                    scaler_step
                ]
            )

            return preprocessor


        except Exception as e:
            raise customException(e,sys)