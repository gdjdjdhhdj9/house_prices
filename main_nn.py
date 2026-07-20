import numpy as np
import pandas as pd
from pathlib import Path
import joblib

import torch
import torch.nn as nn

from train_nn import train
from nn_model import MyNN
from nn_data_preparation import DataPreparation
from config import config


if not Path(config.paths.path_to_best_nn_model).exists() or not any(Path(config.paths.path_to_best_nn_model).glob("*.pt")):
    Path(config.paths.path_to_best_nn_model).mkdir(parents=True, exist_ok=True)
    train()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_test_tensor_final,test_ids = DataPreparation(train=False).prepare_data(pd.read_csv(Path(config.paths.path_to_test_data)))
print("Данные готовы к предсказанию.")

model = MyNN(X_test_tensor_final.shape[1], 1).to(device)

model.load_state_dict(torch.load(f'{Path(config.paths.path_to_best_nn_model)}/best_nn_model.pt', map_location=device))
model.eval()
with torch.no_grad():
    # Получаем предсказания
    log_preds = model(X_test_tensor_final).numpy().flatten()

final_preds = np.expm1(log_preds)

# Создаем DataFrame для отправки
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': final_preds
})

# Сохраняем в CSV
submission.to_csv(Path(config.paths.path_to_nn_submission), index=False)

print("Файл submission.csv успешно создан!")
