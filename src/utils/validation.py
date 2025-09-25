import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


class TimeSeriesValidator:
    def __init__(self, n_splits=3, test_size=0.2):
        self.n_splits = n_splits
        self.test_size = test_size

    def temporal_split(self, X, y):
        """
        Divide dados temporais em Train/Val/Test preservando ordem temporal

        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        n_samples = len(X)

        # Test: últimos 20% dos dados
        test_start = int(n_samples * (1 - self.test_size))

        # Validation: 20% antes do test
        val_start = int(n_samples * (1 - 2 * self.test_size))

        X_train = X[:val_start]
        y_train = y[:val_start]

        X_val = X[val_start:test_start]
        y_val = y[val_start:test_start]

        X_test = X[test_start:]
        y_test = y[test_start:]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def time_series_cross_validate(self, X, y, model_creator, params):
        """
        Validação cruzada temporal usando TimeSeriesSplit

        Parameters:
        -----------
        X, y : arrays
            Dados de entrada e target
        model_creator : function
            Função que cria modelo dados os parâmetros
        params : list
            Parâmetros do modelo

        Returns:
        --------
        float: RMSE médio das validações
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        rmse_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"    Fold {fold+1}/{self.n_splits}")

            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]

            try:
                model = model_creator(params)

                # Treinar modelo
                model.fit(
                    X_fold_train, y_fold_train,
                    batch_size=[16, 32, 64][int(params[3])],
                    epochs=30,  # Menos épocas para validação cruzada
                    validation_data=(X_fold_val, y_fold_val),
                    verbose=0
                )

                # Avaliar
                y_pred = model.predict(X_fold_val, verbose=0)
                rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
                rmse_scores.append(rmse)

            except Exception as e:
                print(f"      Erro no fold {fold+1}: {e}")
                rmse_scores.append(float('inf'))

        return np.mean(rmse_scores)

    def walk_forward_validation(self, X, y, model_creator, params, window_size=100):
        """
        Validação walk-forward para séries temporais

        Parameters:
        -----------
        window_size : int
            Tamanho da janela móvel para treino
        """
        rmse_scores = []
        n_samples = len(X)

        # Começar após ter dados suficientes
        start_idx = window_size

        for i in range(start_idx, n_samples - 1):
            # Janela móvel de treino
            train_start = max(0, i - window_size)

            X_train_window = X[train_start:i]
            y_train_window = y[train_start:i]

            # Próximo ponto para predição
            X_test_point = X[i:i+1]
            y_test_point = y[i:i+1]

            try:
                model = model_creator(params)

                model.fit(
                    X_train_window, y_train_window,
                    batch_size=16,
                    epochs=10,
                    verbose=0
                )

                y_pred = model.predict(X_test_point, verbose=0)
                rmse = np.sqrt(mean_squared_error(y_test_point, y_pred))
                rmse_scores.append(rmse)

            except Exception as e:
                rmse_scores.append(float('inf'))

        return np.mean(rmse_scores)