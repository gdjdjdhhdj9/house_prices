import numpy as np
import pandas as pd

def prepare_data(data:pd.DataFrame) -> pd.DataFrame:
    X = data.copy()
    X = data.drop('Id', axis=1, errors='ignore')

    # добовляем новые фичи
    X['HouseAge'] = X['YrSold'] - X['YearBuilt']
    X['RemodAge'] = X['YrSold'] - X['YearRemodAdd']
    X['TotalBath'] = X['FullBath'] + 0.5 * X['HalfBath']
    X['HasGarage'] = (X['GarageArea'].fillna(0) > 0).astype(int)
    X['HasBasement'] = (X['TotalBsmtSF'].fillna(0) > 0).astype(int)
    X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']

    #удаляем ненужные фичи
    X = X.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'],axis=1)

    #заполняем пропуски в категориальных фичах
    categorical_colums = X.select_dtypes(include=['object', 'string']).columns
    for col in categorical_colums:
        X[col] = X[col].fillna(f'{col}_nan')

    return X