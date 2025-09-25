import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ..utils.validation import TimeSeriesValidator
from config.config import Config


class LSTMModel:
    def __init__(self, sequence_length=30, n_features=2):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model = None
        self.validator = TimeSeriesValidator(test_size=Config.VALIDATION['test_size'])

    def prepare_sequences(self, data, target_col='vazao'):
        features = data.drop(columns=[target_col])
        target = data[target_col].values.reshape(-1, 1)

        features_scaled = self.scaler_X.fit_transform(features)
        target_scaled = self.scaler_y.fit_transform(target)

        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(features_scaled[i-self.sequence_length:i])
            y.append(target_scaled[i, 0])

        return np.array(X), np.array(y)

    def create_model(self, params):
        lstm_units, dropout_rate, learning_rate, batch_size_idx = params
        lstm_units = int(lstm_units)

        model = Sequential([
            Input(shape=(self.sequence_length, self.n_features)),
            LSTM(lstm_units, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(lstm_units // 2, return_sequences=False),
            Dropout(dropout_rate),
            Dense(25, activation='relu'),
            Dense(1)
        ])

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        return model

    def train(self, X, y, params, epochs=None):
        epochs = epochs or Config.TRAINING['final_epochs']

        # Usar divisão temporal estruturada
        X_train, X_val, X_test, y_train, y_val, y_test = self.validator.temporal_split(X, y)

        print(f"Divisão temporal final:")
        print(f"   Train: {len(X_train)} amostras ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Val: {len(X_val)} amostras ({len(X_val)/len(X)*100:.1f}%)")
        print(f"   Test: {len(X_test)} amostras ({len(X_test)/len(X)*100:.1f}%)")

        self.model = self.create_model(params)
        batch_size = [16, 32, 64][int(params[3])]

        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5)
            ],
            verbose=1
        )

        # Avaliar no conjunto de teste (nunca visto)
        y_pred = self.model.predict(X_test)
        y_test_orig = self.scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_orig = self.scaler_y.inverse_transform(y_pred).flatten()

        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test_orig, y_pred_orig)),
            'mae': mean_absolute_error(y_test_orig, y_pred_orig),
            'r2': r2_score(y_test_orig, y_pred_orig)
        }

        # Também calcular métricas na validação para comparação
        y_val_pred = self.model.predict(X_val)
        y_val_orig = self.scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
        y_val_pred_orig = self.scaler_y.inverse_transform(y_val_pred).flatten()

        val_metrics = {
            'val_rmse': np.sqrt(mean_squared_error(y_val_orig, y_val_pred_orig)),
            'val_mae': mean_absolute_error(y_val_orig, y_val_pred_orig),
            'val_r2': r2_score(y_val_orig, y_val_pred_orig)
        }

        # Combinar métricas
        all_metrics = {**metrics, **val_metrics}

        return history, all_metrics

    @tf.function(reduce_retracing=True)
    def _predict_function(self, X):
        return self.model(X, training=False)

    def predict(self, X):
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda!")

        predictions = self._predict_function(X)
        return self.scaler_y.inverse_transform(predictions).flatten()