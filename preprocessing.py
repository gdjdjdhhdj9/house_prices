import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
# from sklearn.feature_selection import VarianceThreshold


def create_preprocessor(X, categorical_columns, numeric_columns):
    low_card_cols = [
        col for col in categorical_columns
        if X[col].nunique() < 5
    ]

    high_card_cols = [
        col for col in categorical_columns
        if X[col].nunique() >= 5
    ]

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        # ('variance_selector', VarianceThreshold(threshold=0.1))

    ])

    categorical_imputer = SimpleImputer(strategy='constant', fill_value='missing')  

    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', Pipeline([
                ('imputer', categorical_imputer),
                ('encoder', OneHotEncoder(
                drop='first',
                handle_unknown='ignore',
                sparse_output=False
            ))
            ]), low_card_cols),

            ('target', Pipeline([
                ('imputer', categorical_imputer),
                ('target',  TargetEncoder(
                target_type='continuous',
                random_state=42
            ))]), high_card_cols),

            ('numeric', numeric_transformer, numeric_columns)
        ],
        verbose_feature_names_out=False
    )

    return preprocessor