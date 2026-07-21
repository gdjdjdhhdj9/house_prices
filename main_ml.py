import numpy as np
import pandas as pd

from pathlib import Path
import joblib

from train_ml import train_model
from feature_engineering import engineer_features
from config import config


if not Path(config.paths.path_to_best_ml_model).exists() or not any(Path(config.paths.path_to_best_ml_model).iterdir()):
    Path(config.paths.path_to_best_ml_model).mkdir(parents=True, exist_ok=True)
    train_model()

model = joblib.load(f'{Path(config.paths.path_to_best_ml_model)}/best_model.pkl')

test_data = pd.read_csv(Path(config.paths.path_to_test_data))
y_pred = np.expm1(model.predict(engineer_features(test_data)))

submission = pd.DataFrame({
    "Id": test_data["Id"],
    "SalePrice": y_pred
})

submission.to_csv(Path(config.paths.path_to_ml_submission), index=False)

print("Файл submission.csv успешно создан!")