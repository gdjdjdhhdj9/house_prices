import numpy as np
import pandas as pd

def prepare_data(data:pd.DataFrame) -> pd.DataFrame:
    X = data.copy()
    X = data.drop('Id', axis=1, errors='ignore')

    X['HouseAge'] = X['YrSold'] - X['YearBuilt']
    X['RemodAge'] = X['YrSold'] - X['YearRemodAdd']
    X['TotalBath'] = X['FullBath'] + 0.5 * X['HalfBath']
    X['HasGarage'] = (X['GarageArea'].fillna(0) > 0).astype(int)
    X['HasBasement'] = (X['TotalBsmtSF'].fillna(0) > 0).astype(int)
    X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']
    X = X.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'],axis=1)

    return X