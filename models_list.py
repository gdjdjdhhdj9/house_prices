import numpy as np

from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


models = {
    'Lasso': (
        Lasso(),
        {
            'model__alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            'model__max_iter': [1000, 5000],
            'model__tol': [1e-4, 1e-5],
            'model__positive': [False],
            'model__selection': ['cyclic', 'random'],
        }

        # best params
        # {
            # 'model__alpha': [0.0001],
            # 'model__max_iter': [5000],
            # 'model__positive': [False],
            # 'model__selection': ['random'],
            # 'model__tol': [1e-05],
        # }
    ),
    
    'Ridge': (
        Ridge(),
        {
            'model__alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1.0],
            'model__fit_intercept': [True],
            'model__max_iter': [1000, 5000],
            'model__tol': [1e-4, 1e-5],
            'model__solver': ['auto', 'svd', 'cholesky'],
        }

        # best params
        # {
            # 'model__alpha': [1.0],
            # 'model__fit_intercept': [True],
            # 'model__max_iter': [1000],
            # 'model__solver': ['svd'],
            # 'model__tol': [0.0001],
        # }
    ),

    'tree': (
        DecisionTreeRegressor(random_state=42),
        {
            'model__criterion': ['squared_error', 'friedman_mse'],
            'model__max_depth': [3, 5, 7, 9, None],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 4, 8],
            'model__max_leaf_nodes': [None, 10, 20, 50, 100],
        }

        # best params
        # {
            # 'model__criterion': ['friedman_mse'],
            # 'model__max_depth': [7],
            # 'model__max_leaf_nodes': [100],
            # 'model__min_samples_leaf': [8],
            # 'model__min_samples_split': [2]
        # }
    ),

    'forest': (
        RandomForestRegressor(random_state=42),
        {
            'model__n_estimators': [50, 100, 200, 300],
            'model__max_depth': [None, 5, 10, 15, 20],
            'model__min_samples_leaf': [1, 2, 4],
            'model__min_samples_split': [2, 5, 10],
            'model__max_features': ['sqrt', 'log2', 0.8, 1.0],
        }

        # best params
        # {
        #     'model__max_depth': [20], 
        #     'model__max_features': [0.8], 
        #     'model__min_samples_leaf': [2], 
        #     'model__min_samples_split': [2], 
        #     'model__n_estimators': [200],
        # }
    ),

    'extra_trees': (
        ExtraTreesRegressor(random_state=42),
        {
            'model__n_estimators': [50, 100, 200, 300],
            'model__max_depth': [None, 5, 10, 15, 20],
            'model__min_samples_leaf': [1, 2, 4],
            'model__min_samples_split': [2, 5, 10],
            'model__max_features': ['sqrt', 'log2', 0.8, 1.0],
        }

        # best params
        # {
        #     'model__max_depth': [20],
        #     'model__max_features': [1.0],
        #     'model__min_samples_leaf': [2],
        #     'model__min_samples_split': [5],
        #     'model__n_estimators': [200],
        # }
    ),

    'gradient_boosting': (
        GradientBoostingRegressor(random_state=42),
        {
            'model__n_estimators': [50, 100, 200, 500],
            'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'model__max_depth': [3, 5, 7],
            'model__subsample': [0.6, 0.8, 1.0],
            'model__max_features': ['sqrt', 'log2', None],
        }

        # best params
        # {
        #     'model__learning_rate': [0.05],
        #     'model__max_depth': [3],
        #     'model__max_features': ['sqrt'],
        #     'model__n_estimators': [500],
        #     'model__subsample': [0.8],
        # }
    ),

    'XGBRegressor' :( 
        XGBRegressor( 
            objective="reg:squarederror", 
            eval_metric="rmse",
            random_state=42,
            n_jobs=1,
            verbosity=False,
        ),
        {
            'model__n_estimators': [50, 100, 200, 300],
            'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'model__max_depth': [3, 5, 7, 9],
            'model__subsample': [0.6, 0.8, 1.0],
            'model__colsample_bytree': [0.6, 0.8, 1.0],
            'model__gamma': [0, 0.1, 0.2],
        },

        # best params
        # {
        #     'model__colsample_bytree': [0.6],
        #     'model__gamma': [0],
        #     'model__learning_rate': [0.05],
        #     'model__max_depth': [3],
        #     'model__n_estimators': [300],
        #     'model__subsample': [0.6],
        # }
    ),

    'LGBMRegressor':( 
        LGBMRegressor( 
            objective="regression", 
            random_state=42,
            n_jobs=1,
            verbose=False,
        ),
        {
            'model__n_estimators': [50, 100, 200, 300],
            'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'model__max_depth': [-1, 3, 5, 7, 9],
            'model__num_leaves': [20, 31, 40],
            'model__min_child_samples': [20, 30, 40],
            'model__subsample': [0.6, 0.8, 1.0],
            'model__colsample_bytree': [0.6, 0.8, 1.0],
        },

        # best params
        # {
        #     'model__colsample_bytree': [0.6],
        #     'model__learning_rate': [0.05],
        #     'model__max_depth': [5],
        #     'model__min_child_samples': [20],
        #     'model__n_estimators': [200],
        #     'model__num_leaves': [20],
        #     'model__subsample': [0.6],
        # }
    ),

    'CatBoostRegressor':(
        CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=42,
            thread_count=1,
            verbose=False,
            allow_writing_files=False,
            nan_mode='Min'
        ),
        {
            'model__iterations': [100, 200, 300, 500, 1000],
            'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'model__depth': [3, 5, 7, 9],
            'model__l2_leaf_reg': [1, 3, 5, 10],
            'model__random_strength': [0.1, 0.5, 1, 2],
            'model__border_count': [32, 64, 128, 254],
        }

        # 'model__iterations': [100, 200, 300, 500],
        # 'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        # 'model__depth': [3, 5, 7, 9],
        # 'model__l2_leaf_reg': [1, 3, 5, 7],
        # 'model__random_strength': [0.1, 0.5, 1, 2],
        # 'model__border_count': [32, 64, 128],

        # best params
        # {
                # [CV 4/5] END model__border_count=32, model__depth=5, model__iterations=500, model__l2_leaf_reg=3, model__learning_rate=0.1, model__random_strength=1;, score=-19237.159 total time=  10.1s
        # }
    ),
}
