from omegaconf import OmegaConf
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

config = {
    'general': {
        'seed': 42,
    },
    'paths': {
        'path_to_train_data': BASE_DIR / 'data' / 'train.csv',
        'path_to_test_data': BASE_DIR / 'data' / 'test.csv',
        'path_to_best_nn_model': BASE_DIR / 'models' / 'nn',
        'path_to_best_ml_model': BASE_DIR / 'models' / 'ml',
        'path_to_preprocessor': BASE_DIR / 'models' / 'nn' / 'preprocessor.pkl',
        'path_to_ml_submission': BASE_DIR / 'data' / 'ml_submission' / 'submission.csv',
        'path_to_nn_submission': BASE_DIR / 'data' / 'nn_submission' / 'submission.csv',

    },
    'training': {
        'num_epochs': 5000,
        'early_stopping_epochs': 1000,
    },
    'dataloader_params': {
        'batch_size': 32,
        'shuffle': False,
    },
    'optimizer': {
        'name': 'AdamW',
        'params': {
            'lr': 0.1,
            'weight_decay' : 0.01,
        },
    },
    'scheduler': {
        'name': 'ReduceLROnPlateau', 
        'params': {
            'mode' : 'min',
            'factor' : 0.1,  
            'patience' : 10,
            'eps': 1e-8,
        }
    },
    'loss': {
        'name': 'MSELoss',
        'params': {
            # 's': 45,
            # 'm': 0.4,
            # 'crit': 'bce',
            # 'class_weights_norm': "batch",
        }
    },

}

config = OmegaConf.create(config)