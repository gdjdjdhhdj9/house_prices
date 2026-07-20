import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import root_mean_squared_error

import joblib
from pathlib import Path

from feature_engineering import engineer_features
from preprocessing import create_preprocessor
from models_list import models
from metrics import rmse_scorer
from config import config


def train_model():

    data = pd.read_csv(Path(config.paths.path_to_train_data))

    y = np.log1p(data['SalePrice'])
    X = engineer_features(data.drop('SalePrice',axis=1))

    y = y[X.index]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        shuffle=True,
        random_state=42,
    )

    numeric_columns = X_train.select_dtypes(include=['number']).columns
    categorical_columns = X_train.select_dtypes(include=['object', 'string']).columns

    preprocessor = create_preprocessor(X_train, categorical_columns, numeric_columns)    

    # объявляем массив, куда мы будем записывать лучшие параметры моделей
    results = []
    best_score = None

    output_dir = Path(config.paths.path_to_best_ml_model)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Запуск цикла обучения 
    for i, (name, (model, params)) in enumerate(models.items(), 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(models)}] Обучение модели: {name}")
        print(f"Количество комбинаций параметров: {len(params) if isinstance(params, list) else 'N/A'}")
        print('='*60)
        
        if name == 'CatBoostRegressor':
            pipe = Pipeline([
                ('model', model),
            ])
            grid = GridSearchCV(pipe, params, cv=5, scoring=rmse_scorer, n_jobs=1, verbose=3)
            print("Запуск GridSearchCV с cat_features...")
            grid.fit(X_train, y_train, model__cat_features = categorical_columns.tolist())
        else:
            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model),
            ])
            grid = GridSearchCV(pipe, params, cv=5, scoring=rmse_scorer, n_jobs=1, verbose=3)
            print(f'Запуск GridSearchCV с {model}...')
            grid.fit(X_train, y_train)

        test_rmse = root_mean_squared_error(
            np.expm1(y_test),
            np.expm1(grid.predict(X_test)),
        )
        
        print(f"\n✓ Модель {name} обучена")
        print(f"  Лучшие параметры: {grid.best_params_}")
        print(f"  CV RMSE (логарифмическая шкала): {-grid.best_score_:.6f}")
        print(f"  Test RMSE (исходная шкала): {test_rmse:.2f}")
        
        results.append({
            'model': name,
            'best_cv_score': -grid.best_score_,
            'test_score': test_rmse,
            'best_params': grid.best_params_,
            'best_estimator': grid.best_estimator_
        })

        if best_score is None or test_rmse < best_score:
            best_score = test_rmse
            joblib.dump(grid.best_estimator_, output_dir / 'best_model.pkl')
            print(f"⭐ Сохранена как лучшая модель!")

    print(f"\n{'='*60}")
    print("✓ Обучение завершено. Рассчитываем финальные метрики...")
    print('='*60)