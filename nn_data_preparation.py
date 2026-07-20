import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, TensorDataset

from feature_engineering import engineer_features
from preprocessing import create_preprocessor
from config import config


class DataPreparation():
    def __init__(self, train):
        self.train = train

    def prepare_data(self, data):
        if self.train:
        
            y = np.log1p(data['SalePrice'])
            X = engineer_features(data.drop('SalePrice',axis=1))

            y = y[X.index]

            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=config.general.seed)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(0.2/0.9), random_state=config.general.seed)

            numeric_columns = X_train.select_dtypes(include=['number']).columns
            categorical_columns = X_train.select_dtypes(include=['object', 'string']).columns

            preprocessor = create_preprocessor(X_train, categorical_columns, numeric_columns)
            

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            generator = torch.Generator().manual_seed(config.general.seed)

            # Применяем препроцессор
            X_train_processed = preprocessor.fit_transform(X_train, y_train)
            X_val_processed = preprocessor.transform(X_val)
            X_test_processed = preprocessor.transform(X_test)

            joblib.dump(preprocessor, Path(config.paths.path_to_preprocessor))
            # Преобразуем данные в тензоры PyTorch
            X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)

            X_val_tensor = torch.tensor(X_val_processed, dtype=torch.float32).to(device)
            y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

            X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32).to(device)
            y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(device)

            # Создаем TensorDataset
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

            train_loader = DataLoader(dataset=train_dataset, batch_size=config.dataloader_params.batch_size, shuffle=True, generator=generator)
            val_loader = DataLoader(dataset=val_dataset, batch_size=config.dataloader_params.batch_size, shuffle=config.dataloader_params.shuffle)
            test_loader = DataLoader(dataset=test_dataset, batch_size=config.dataloader_params.batch_size, shuffle=config.dataloader_params.shuffle)
            
            return train_loader, val_loader, test_loader
        
        else:

            test_ids = data['Id']

            preprocessor = joblib.load(Path(config.paths.path_to_preprocessor))

            # 3. Предобработка тестовых данных
            X_test_final = engineer_features(data)
            X_test_processed_final = preprocessor.transform(X_test_final)

            # Конвертируем в тензор
            X_test_tensor_final = torch.tensor(X_test_processed_final, dtype=torch.float32)

            return X_test_tensor_final, test_ids
