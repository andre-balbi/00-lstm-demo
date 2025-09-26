import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ..utils.validation import TimeSeriesValidator
from config.config import Config


class OperationalEarlyStopping(tf.keras.callbacks.Callback):
    """
    Early stopping baseado em performance operacional para contexto hidroelétrico

    Para training stops se horizontes críticos (dias 1-3) começarem a degradar
    mesmo que a loss geral ainda esteja melhorando
    """

    def __init__(self, patience=5, min_delta=0.001, verbose=1):
        super().__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.wait = 0
        self.best_critical_performance = float('inf')
        self.horizon = Config.PREDICTION.get('horizon', 1)

    def on_epoch_end(self, epoch, logs=None):
        if self.horizon == 1:
            # Single-step: usar loss de validação padrão
            current_performance = logs.get('val_loss', float('inf'))
        else:
            # Multi-step: focar em performance dos primeiros horizontes
            # Como não temos métricas separadas por horizonte durante training,
            # usar val_loss como proxy e aplicar early stopping mais conservador
            current_performance = logs.get('val_loss', float('inf'))

        # Verificar se houve melhoria significativa
        if current_performance < self.best_critical_performance - self.min_delta:
            self.best_critical_performance = current_performance
            self.wait = 0
            if self.verbose:
                print(f"\\n  🟢 Época {epoch+1}: Performance operacional melhorou para {current_performance:.6f}")
        else:
            self.wait += 1
            if self.verbose:
                print(f"\\n  🟡 Época {epoch+1}: Sem melhoria crítica há {self.wait} épocas (performance: {current_performance:.6f})")

            if self.wait >= self.patience:
                if self.verbose:
                    print(f"\\n  🔴 EARLY STOPPING OPERACIONAL: Performance crítica não melhorou há {self.patience} épocas")
                    print(f"     Melhor performance: {self.best_critical_performance:.6f}")
                    print(f"     Performance atual: {current_performance:.6f}")
                    print(f"     Parando treinamento para preservar qualidade operacional")
                self.model.stop_training = True


class HydroelectricCallback(tf.keras.callbacks.Callback):
    """
    Callback especializado para monitoramento de treinamento hidroelétrico

    Fornece insights sobre degradação de performance por horizonte durante treinamento
    """

    def __init__(self, validation_data=None, verbose=1):
        super().__init__()
        self.validation_data = validation_data
        self.verbose = verbose
        self.horizon = Config.PREDICTION.get('horizon', 1)
        self.epoch_history = []

    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None or self.horizon == 1:
            return

        X_val, y_val = self.validation_data

        # Fazer predições para análise
        try:
            y_pred = self.model.predict(X_val, verbose=0)

            # Calcular RMSE por horizonte
            horizon_rmses = []
            for h in range(min(3, self.horizon)):  # Focar nos primeiros 3 dias
                rmse_h = np.sqrt(mean_squared_error(y_val[:, h], y_pred[:, h]))
                horizon_rmses.append(rmse_h)

            self.epoch_history.append({
                'epoch': epoch + 1,
                'horizon_rmses': horizon_rmses,
                'avg_critical_rmse': np.mean(horizon_rmses)
            })

            # Log periódico (a cada 10 épocas ou últimas 5)
            if self.verbose and (epoch % 10 == 0 or epoch >= logs.get('epochs', 50) - 5):
                print(f"\\n  📊 Época {epoch+1} - RMSE Crítico (D1-3): {np.mean(horizon_rmses):.4f}")
                if len(horizon_rmses) > 1:
                    degradation = horizon_rmses[-1] / horizon_rmses[0]
                    print(f"     Degradação D1→D3: {degradation:.2f}x")

        except Exception as e:
            if self.verbose:
                print(f"\\n  ⚠️ Erro no monitoramento hidroelétrico: {e}")


class LSTMModel:
    def __init__(self, sequence_length=30, n_features=2):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.model = None
        self.validator = TimeSeriesValidator(test_size=Config.VALIDATION['test_size'])

    def prepare_sequences(self, data, target_col='vazao'):
        """
        Prepara sequências para predição multi-step sem data leakage

        Para horizon=1: [30 steps históricos] → [step t+1]
        Para horizon=7: [30 steps históricos] → [steps t+1, t+2, ..., t+7]

        Args:
            data: DataFrame com dados temporais
            target_col: Nome da coluna target

        Returns:
            X: Sequências de entrada (n_samples, sequence_length, n_features)
            y: Targets (n_samples,) para horizon=1 ou (n_samples, horizon) para horizon>1
        """
        features = data.drop(columns=[target_col])
        target = data[target_col].values  # Não fazer reshape ainda

        features_scaled = self.scaler_X.fit_transform(features)
        target_scaled = self.scaler_y.fit_transform(target.reshape(-1, 1)).flatten()

        horizon = Config.PREDICTION['horizon']
        X, y = [], []

        print(f"DEBUG: Preparando sequências com horizon={horizon}")
        print(f"DEBUG: Dados totais: {len(data)} amostras")

        # Loop corrigido: garantir que não há vazamento temporal
        for i in range(self.sequence_length, len(data) - horizon + 1):
            # Input: sequência histórica (30 dias ANTES de i)
            X.append(features_scaled[i-self.sequence_length:i])

            # Output: próximos 'horizon' dias A PARTIR de i
            if horizon == 1:
                y.append(target_scaled[i])  # Apenas dia i
            else:
                y.append(target_scaled[i:i+horizon])  # Dias i até i+horizon-1

        print(f"DEBUG: Sequências criadas - X: {len(X)}, y: {len(y)}")
        if len(y) > 0:
            if horizon == 1:
                print(f"DEBUG: Shape y[0]: {np.array(y[0]).shape}")
            else:
                print(f"DEBUG: Shape y[0]: {np.array(y[0]).shape} (deve ser {horizon})")

        return np.array(X), np.array(y)

    def create_model(self, params):
        lstm_units, dropout_rate, learning_rate, batch_size_idx = params
        lstm_units = int(lstm_units)
        horizon = Config.PREDICTION['horizon']

        model = Sequential([
            Input(shape=(self.sequence_length, self.n_features)),
            LSTM(lstm_units, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(lstm_units // 2, return_sequences=False),
            Dropout(dropout_rate),
            Dense(25, activation='relu'),
            Dense(horizon)  # Outputs configuráveis baseados no horizon
        ])

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        return model

    def _proper_denormalize_multistep(self, y_scaled, horizon):
        """
        Desnormalização correta para predições multi-step

        Args:
            y_scaled: Dados escalados (n_samples, horizon) ou (n_samples,)
            horizon: Número de horizontes de predição

        Returns:
            array: Dados desnormalizados no formato original
        """
        if horizon == 1:
            # Caso simples: 1D ou 2D com 1 coluna
            if len(y_scaled.shape) == 1:
                return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
            else:
                return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
        else:
            # Multi-step: desnormalizar cada horizonte separadamente
            n_samples = y_scaled.shape[0]
            y_denorm = np.zeros((n_samples, horizon))

            for h in range(horizon):
                y_denorm[:, h] = self.scaler_y.inverse_transform(
                    y_scaled[:, h].reshape(-1, 1)
                ).flatten()

            return y_denorm

    def _calculate_metrics_simple(self, y_true, y_pred, prefix=''):
        """
        Cálculo de métricas simplificado e intuitivo com desnormalização correta

        Args:
            y_true: Valores reais escalados
            y_pred: Valores preditos escalados
            prefix: Prefixo para as métricas ('test', 'val', etc.)

        Returns:
            dict: Métricas calculadas
        """
        horizon = Config.PREDICTION['horizon']

        if horizon == 1:
            # Caso simples: predição de 1 passo
            y_true_orig = self._proper_denormalize_multistep(y_true, horizon)
            y_pred_orig = self._proper_denormalize_multistep(y_pred, horizon)

            metrics = {
                f'{prefix}_rmse' if prefix else 'rmse': np.sqrt(mean_squared_error(y_true_orig, y_pred_orig)),
                f'{prefix}_mae' if prefix else 'mae': mean_absolute_error(y_true_orig, y_pred_orig),
                f'{prefix}_r2' if prefix else 'r2': r2_score(y_true_orig, y_pred_orig)
            }

            print(f"RESULTADO {prefix.upper()}: Predição de 1 step - RMSE: {metrics[f'{prefix}_rmse' if prefix else 'rmse']:.3f}")

        else:
            # Caso multi-step: métricas por horizonte + geral
            # Desnormalização correta
            y_true_orig = self._proper_denormalize_multistep(y_true, horizon)
            y_pred_orig = self._proper_denormalize_multistep(y_pred, horizon)

            # Métrica geral (todos os horizontes) - PONDERADA para contexto hidroelétrico
            weights = self._get_hydroelectric_weights(horizon)
            rmse_weighted = self._calculate_weighted_rmse(y_true_orig, y_pred_orig, weights)
            rmse_overall = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))  # Para comparação
            mae_overall = mean_absolute_error(y_true_orig, y_pred_orig)
            r2_overall = r2_score(y_true_orig.flatten(), y_pred_orig.flatten())

            metrics = {
                f'{prefix}_rmse_weighted' if prefix else 'rmse_weighted': rmse_weighted,
                f'{prefix}_rmse' if prefix else 'rmse': rmse_overall,
                f'{prefix}_mae' if prefix else 'mae': mae_overall,
                f'{prefix}_r2' if prefix else 'r2': r2_overall
            }

            print(f"\\nRESULTADO {prefix.upper()}: Predição de {horizon} dias consecutivos")
            print(f"RMSE Ponderado (Hidroelétrico): {rmse_weighted:.3f}")
            print(f"RMSE Geral (Não-ponderado): {rmse_overall:.3f}")

            # Métricas por horizonte (dias individuais)
            print("RMSE por dia:")
            for h in range(horizon):
                rmse_h = np.sqrt(mean_squared_error(y_true_orig[:, h], y_pred_orig[:, h]))
                mae_h = mean_absolute_error(y_true_orig[:, h], y_pred_orig[:, h])
                r2_h = r2_score(y_true_orig[:, h], y_pred_orig[:, h])

                metrics[f'{prefix}_rmse_day{h+1}' if prefix else f'rmse_day{h+1}'] = rmse_h
                metrics[f'{prefix}_mae_day{h+1}' if prefix else f'mae_day{h+1}'] = mae_h
                metrics[f'{prefix}_r2_day{h+1}' if prefix else f'r2_day{h+1}'] = r2_h

                weight_info = f" (peso: {weights[h]:.2f})" if horizon > 1 else ""
                print(f"  Dia {h+1} (t+{h+1}): RMSE={rmse_h:.3f}, MAE={mae_h:.3f}, R²={r2_h:.3f}{weight_info}")

        return metrics

    def _get_hydroelectric_weights(self, horizon):
        """
        Pesos operacionais para contexto hidroelétrico

        Prioriza predições de curto prazo (dias 1-3) que são críticas
        para decisões operacionais imediatas
        """
        if horizon == 1:
            return np.array([1.0])

        # Peso exponencial decrescente: dias iniciais são mais críticos
        weights = np.exp(-0.15 * np.arange(horizon))

        # Manter pesos normalizados (soma = 1) para média ponderada correta
        weights = weights / weights.sum()

        return weights

    def _calculate_weighted_rmse(self, y_true, y_pred, weights):
        """
        Calcula RMSE ponderado por horizonte para otimização hidroelétrica

        Retorna média ponderada dos RMSEs, não soma, para manter escala correta
        """
        horizon_rmses = []
        for h in range(y_true.shape[1]):
            rmse_h = np.sqrt(mean_squared_error(y_true[:, h], y_pred[:, h]))
            horizon_rmses.append(rmse_h)

        # Média ponderada: Σ(rmse_h * weight_h) onde Σ(weight_h) = 1
        weighted_rmse = np.average(horizon_rmses, weights=weights)

        return weighted_rmse

    def debug_shapes_and_predictions(self, X, y):
        """
        Função de debugging para verificar shapes e exemplo de predições
        """
        horizon = Config.PREDICTION['horizon']

        print(f"\n{'='*60}")
        print(f"DEBUG: SHAPES E PREDIÇÕES (horizon={horizon})")
        print(f"{'='*60}")

        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

        if len(y.shape) == 1:
            print(f"y é 1D - cada amostra prediz 1 valor")
        else:
            print(f"y é 2D - cada amostra prediz {y.shape[1]} valores")

        # Exemplo de uma sequência
        print(f"\nEXEMPLO - Primeira sequência:")
        print(f"Input (últimos 5 dias de features): {X[0][-5:]}")
        if horizon == 1:
            print(f"Target: {y[0]} (próximo dia)")
        else:
            print(f"Target: {y[0]} (próximos {horizon} dias)")

        # Verificar se há algum modelo treinado
        if self.model is not None:
            print(f"\nTESTE DE PREDIÇÃO:")
            test_pred = self.model.predict(X[:3])  # Primeiras 3 amostras
            print(f"Predição shape: {test_pred.shape}")
            print(f"Predições: {test_pred}")

        print(f"{'='*60}\n")

    def train(self, X, y, params, epochs=None):
        epochs = epochs or Config.TRAINING['final_epochs']

        # DEBUG: Verificar shapes antes do treinamento
        self.debug_shapes_and_predictions(X, y)

        # Usar divisão temporal estruturada
        X_train, X_val, X_test, y_train, y_val, y_test = self.validator.temporal_split(X, y)

        print(f"Divisão temporal final:")
        print(f"   Train: {len(X_train)} amostras ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Val: {len(X_val)} amostras ({len(X_val)/len(X)*100:.1f}%)")
        print(f"   Test: {len(X_test)} amostras ({len(X_test)/len(X)*100:.1f}%)")

        self.model = self.create_model(params)
        batch_size = Config.LSTM_PARAMS_BOUNDS['batch_sizes'][int(params[3])]

        # Callbacks operacionais para contexto hidroelétrico
        callbacks_list = [
            # Early stopping padrão
            tf.keras.callbacks.EarlyStopping(
                patience=Config.CALLBACKS['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            # Redução de learning rate
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=Config.CALLBACKS['lr_reduction_patience'],
                factor=Config.CALLBACKS['lr_reduction_factor'],
                verbose=1
            ),
            # Early stopping operacional (prioriza horizontes críticos)
            OperationalEarlyStopping(
                patience=Config.CALLBACKS['early_stopping_patience'] - 2,  # Mais conservador
                min_delta=0.0001,
                verbose=1
            ),
            # Monitoramento hidroelétrico
            HydroelectricCallback(
                validation_data=(X_val, y_val),
                verbose=1
            )
        ]

        print(f"\\n🏭 TREINAMENTO HIDROELÉTRICO:")
        print(f"   Early Stopping Padrão: {Config.CALLBACKS['early_stopping_patience']} épocas")
        print(f"   Early Stopping Operacional: {Config.CALLBACKS['early_stopping_patience'] - 2} épocas")
        print(f"   Monitoramento por horizonte: {'Ativo' if Config.PREDICTION['horizon'] > 1 else 'Inativo'}")
        print(f"   {'='*50}")

        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=1
        )

        # Avaliar no conjunto de teste (nunca visto)
        y_pred = self.model.predict(X_test)
        horizon = Config.PREDICTION['horizon']

        print(f"\nDEBUG: Avaliação com horizon={horizon}")
        print(f"DEBUG: y_test shape: {y_test.shape}")
        print(f"DEBUG: y_pred shape: {y_pred.shape}")

        # Calcular métricas de forma simplificada
        metrics = self._calculate_metrics_simple(y_test, y_pred, 'test')

        # Também calcular métricas na validação para comparação
        y_val_pred = self.model.predict(X_val)
        val_metrics = self._calculate_metrics_simple(y_val, y_val_pred, 'val')

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
        horizon = Config.PREDICTION['horizon']

        # Usar desnormalização correta
        predictions_numpy = predictions.numpy() if hasattr(predictions, 'numpy') else predictions
        return self._proper_denormalize_multistep(predictions_numpy, horizon)