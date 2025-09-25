class Config:
    SEQUENCE_LENGTH = 30
    N_FEATURES = 2

    # ========================
    # Configurações de Produção
    # ========================
    N_SAMPLES = 800
    
    GENETIC_ALGORITHM = {
        'population_size': 10,
        'generations': 5
    }
    
    TRAINING = {
        'epochs': 50,
        'final_epochs': 100
    }
    
    VALIDATION = {
        'method': 'temporal_split',  # 'temporal_split', 'time_series_cv', 'walk_forward'
        'test_size': 0.2,
        'val_size': 0.2,
        'cv_splits': 3,
        'walk_forward_window': 100
    }
    
    LSTM_PARAMS_BOUNDS = {
        'units_min': 32,
        'units_max': 128,
        'dropout_min': 0.1,
        'dropout_max': 0.5,
        'lr_min': 0.001,
        'lr_max': 0.01,
        'batch_sizes': [16, 32, 64]
    }

    # ===========================
    # Configurações de Desenvolvimento/Testes
    # ===========================
    # N_SAMPLES = 200

    # GENETIC_ALGORITHM = {
    #     'population_size': 3,
    #     'generations': 2
    # }

    # TRAINING = {
    #     'epochs': 10,
    #     'final_epochs': 20
    # }

    # VALIDATION = {
    #     'method': 'temporal_split',  # 'temporal_split', 'time_series_cv', 'walk_forward'
    #     'test_size': 0.2,
    #     'val_size': 0.2,
    #     'cv_splits': 2,
    #     'walk_forward_window': 100
    # }

    # LSTM_PARAMS_BOUNDS = {
    #     'units_min': 16,
    #     'units_max': 32,
    #     'dropout_min': 0.2,
    #     'dropout_max': 0.3,
    #     'lr_min': 0.001,
    #     'lr_max': 0.005,
    #     'batch_sizes': [16]
    # }

