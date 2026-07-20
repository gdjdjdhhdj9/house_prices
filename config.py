from omegaconf import OmegaConf

config = {
    'general': {
        'seed': 42,
    },
    'paths': {
        'path_to_train_data': 'D:/проекты/house_price/data/train.csv',
        'path_to_test_data': 'D:/проекты/house_price/data/test.csv',
        'path_to_best_nn_model': 'D:/проекты/house_price/models/nn',
        'path_to_best_ml_model': 'D:/проекты/house_price/models/ml',
        'path_to_preprocessor':'D:/проекты/house_price/models/nn/preprocessor.pkl',
        'path_to_ml_submission':'D:/проекты/house_price/data/ml_submission/submission.csv',
        'path_to_nn_submission':'D:/проекты/house_price/data/nn_submission/submission.csv',

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