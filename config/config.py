class Config:
    SEQUENCE_LENGTH = 30
    N_FEATURES = 2

    # ========================
    # Configurações de Produção
    # ========================
    # N_SAMPLES = 800
    
    # OPTUNA = {
    #     'n_trials': 20
    # }
    
    # TRAINING = {
    #     'epochs': 50,
    #     'final_epochs': 100
    # }
    
    # VALIDATION = {
    #     'method': 'temporal_split',  # 'temporal_split', 'time_series_cv', 'walk_forward'
    #     'test_size': 0.2,
    #     'val_size': 0.2,
    #     'cv_splits': 3,
    #     'walk_forward_window': 100
    # }
    
    # LSTM_PARAMS_BOUNDS = {
    #     'units_min': 32,
    #     'units_max': 128,
    #     'dropout_min': 0.1,
    #     'dropout_max': 0.5,
    #     'lr_min': 0.001,
    #     'lr_max': 0.01,
    #     'batch_sizes': [16, 32, 64]
    # }

    # CALLBACKS = {
    #     'early_stopping_patience': 15,
    #     'lr_reduction_patience': 10,
    #     'lr_reduction_factor': 0.5
    # }

    # VALIDATION_SPECIFIC = {
    #     'cv_epochs': 30,
    #     'walk_forward_epochs': 10,
    #     'walk_forward_batch_size': 16
    # }

    # PREDICTION = {
    #     'horizon': 10,           # Quantos dias prever à frente (1, 3, 7, 14, 30, etc.)
    #     'strategy': 'direct'   # Estratégia de predição: 'direct' (simultânea)
    # }

    # ===========================
    # Configurações de Desenvolvimento/Testes
    # ===========================
    N_SAMPLES = 200

    OPTUNA = {
        'n_trials': 5
    }

    TRAINING = {
        'epochs': 10,
        'final_epochs': 20
    }

    VALIDATION = {
        'method': 'temporal_split',  # 'temporal_split', 'time_series_cv', 'walk_forward'
        'test_size': 0.2,
        'val_size': 0.2,
        'cv_splits': 2,
        'walk_forward_window': 100
    }

    LSTM_PARAMS_BOUNDS = {
        'units_min': 16,
        'units_max': 32,
        'dropout_min': 0.2,
        'dropout_max': 0.3,
        'lr_min': 0.001,
        'lr_max': 0.005,
        'batch_sizes': [16]
    }

    CALLBACKS = {
        'early_stopping_patience': 5,
        'lr_reduction_patience': 3,
        'lr_reduction_factor': 0.5
    }

    VALIDATION_SPECIFIC = {
        'cv_epochs': 10,
        'walk_forward_epochs': 5,
        'walk_forward_batch_size': 16
    }

    PREDICTION = {
        'horizon': 20,
        'strategy': 'direct'
    }

