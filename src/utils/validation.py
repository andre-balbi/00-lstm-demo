import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from config.config import Config


class TimeSeriesValidator:
    def __init__(self, n_splits=3, test_size=0.2):
        self.n_splits = n_splits
        self.test_size = test_size

    def split_for_optimization(self, X, y):
        """
        Separa dados em (train+val) para otimização e test para avaliação final

        Returns:
            tuple: (X_optim, X_test, y_optim, y_test)
        """
        n_samples = len(X)
        test_start = int(n_samples * (1 - self.test_size))

        # Dados para otimização (80%)
        X_optim = X[:test_start]
        y_optim = y[:test_start]

        # Dados para teste final (20%)
        X_test = X[test_start:]
        y_test = y[test_start:]

        return X_optim, X_test, y_optim, y_test

    def temporal_split(self, X, y):
        """
        Divide dados temporais em Train/Val preservando ordem temporal
        NOTA: Usar apenas dados já separados para otimização (sem test)

        Returns:
            tuple: (X_train, X_val, y_train, y_val)
        """
        n_samples = len(X)

        # Validation: últimos 25% dos dados de otimização (equivale a 20% do total)
        val_start = int(n_samples * 0.75)

        X_train = X[:val_start]
        y_train = y[:val_start]

        X_val = X[val_start:]
        y_val = y[val_start:]

        return X_train, X_val, y_train, y_val

    def time_series_cross_validate(self, X, y, model_creator, params):
        """
        Validação cruzada temporal usando TimeSeriesSplit
        NOTA: Operar apenas em dados de otimização (sem dados de teste)

        Parameters:
        -----------
        X, y : arrays
            Dados de entrada e target (já sem dados de teste)
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
                    batch_size=Config.LSTM_PARAMS_BOUNDS['batch_sizes'][int(params[3])],
                    epochs=Config.VALIDATION_SPECIFIC['cv_epochs'],
                    validation_data=(X_fold_val, y_fold_val),
                    verbose=0
                )

                # Avaliar
                y_pred = model.predict(X_fold_val, verbose=0)

                # Compatibilidade com múltiplos outputs
                if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                    # Múltiplos outputs - calcular RMSE da média
                    rmse = np.sqrt(mean_squared_error(y_fold_val.flatten(), y_pred.flatten()))
                else:
                    # Output único
                    rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))

                rmse_scores.append(rmse)

            except Exception as e:
                print(f"      Erro no fold {fold+1}: {e}")
                rmse_scores.append(float('inf'))

        return np.mean(rmse_scores)

    def horizon_aware_cross_validate(self, X, y, model_creator, params, lstm_model=None):
        """
        Validação cruzada temporal com análise por horizonte para otimização hidroelétrica

        Parameters:
        -----------
        X, y : arrays
            Dados de entrada e target (já sem dados de teste)
        model_creator : function
            Função que cria modelo dados os parâmetros
        params : list
            Parâmetros do modelo
        lstm_model : LSTMModel
            Instância do modelo LSTM para acessar o scaler (opcional)

        Returns:
        --------
        tuple: (weighted_rmse, horizon_analysis)
            - weighted_rmse: RMSE ponderado médio para otimização
            - horizon_analysis: Análise detalhada por horizonte
        """
        from config.config import Config

        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        fold_results = []
        horizon = getattr(Config, 'PREDICTION', {}).get('horizon', 1)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"    Fold {fold+1}/{self.n_splits} (horizon-aware)")

            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]

            try:
                model = model_creator(params)

                # Treinar modelo
                model.fit(
                    X_fold_train, y_fold_train,
                    batch_size=Config.LSTM_PARAMS_BOUNDS['batch_sizes'][int(params[3])],
                    epochs=Config.VALIDATION_SPECIFIC['cv_epochs'],
                    validation_data=(X_fold_val, y_fold_val),
                    verbose=0
                )

                # Predições
                y_pred = model.predict(X_fold_val, verbose=0)

                # Análise por horizonte
                if horizon == 1:
                    # Single-step
                    rmse = np.sqrt(mean_squared_error(y_fold_val, y_pred))
                    fold_result = {
                        'fold': fold + 1,
                        'rmse_overall': rmse,
                        'rmse_weighted': rmse,
                        'horizon_metrics': [{'day': 1, 'rmse': rmse}]
                    }
                else:
                    # Multi-step: calcular métricas por horizonte
                    horizon_metrics = []
                    horizon_rmses = []

                    for h in range(horizon):
                        rmse_h = np.sqrt(mean_squared_error(y_fold_val[:, h], y_pred[:, h]))
                        horizon_metrics.append({'day': h+1, 'rmse': rmse_h})
                        horizon_rmses.append(rmse_h)

                    # RMSE ponderado usando pesos hidroelétricos
                    weights = self._get_hydroelectric_weights(horizon)
                    weighted_rmse = np.average(horizon_rmses, weights=weights)

                    # RMSE geral para comparação
                    rmse_overall = np.sqrt(mean_squared_error(y_fold_val, y_pred))

                    fold_result = {
                        'fold': fold + 1,
                        'rmse_overall': rmse_overall,
                        'rmse_weighted': weighted_rmse,
                        'horizon_metrics': horizon_metrics
                    }

                fold_results.append(fold_result)

            except Exception as e:
                print(f"      Erro no fold {fold+1}: {e}")
                # Fallback para RMSE alto em caso de erro
                fold_results.append({
                    'fold': fold + 1,
                    'rmse_overall': float('inf'),
                    'rmse_weighted': float('inf'),
                    'horizon_metrics': []
                })

        # Análise consolidada
        valid_folds = [f for f in fold_results if f['rmse_weighted'] != float('inf')]
        if not valid_folds:
            return float('inf'), {}

        # RMSE ponderado médio para otimização
        avg_weighted_rmse = np.mean([f['rmse_weighted'] for f in valid_folds])

        # Análise detalhada por horizonte
        horizon_analysis = {
            'avg_weighted_rmse': avg_weighted_rmse,
            'avg_overall_rmse': np.mean([f['rmse_overall'] for f in valid_folds]),
            'fold_results': fold_results,
            'horizon': horizon,
            'n_valid_folds': len(valid_folds)
        }

        if horizon > 1:
            # Estatísticas por horizonte
            horizon_stats = {}
            for h in range(horizon):
                day_rmses = [f['horizon_metrics'][h]['rmse'] for f in valid_folds
                           if len(f['horizon_metrics']) > h]
                if day_rmses:
                    horizon_stats[f'day_{h+1}'] = {
                        'mean_rmse': np.mean(day_rmses),
                        'std_rmse': np.std(day_rmses),
                        'min_rmse': np.min(day_rmses),
                        'max_rmse': np.max(day_rmses)
                    }

            horizon_analysis['horizon_stats'] = horizon_stats

        return avg_weighted_rmse, horizon_analysis

    def _get_hydroelectric_weights(self, horizon):
        """
        Pesos operacionais para contexto hidroelétrico
        Replica a lógica do optimizer para consistência
        """
        if horizon == 1:
            return np.array([1.0])

        # Peso exponencial decrescente: dias iniciais são mais críticos
        weights = np.exp(-0.15 * np.arange(horizon))

        # Manter pesos normalizados (soma = 1) para média ponderada correta
        weights = weights / weights.sum()

        return weights

    def walk_forward_validation(self, X, y, model_creator, params, window_size=100):
        """
        Validação walk-forward para séries temporais
        NOTA: Operar apenas em dados de otimização (sem dados de teste)

        Parameters:
        -----------
        X, y : arrays
            Dados de entrada e target (já sem dados de teste)
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
                    batch_size=Config.VALIDATION_SPECIFIC['walk_forward_batch_size'],
                    epochs=Config.VALIDATION_SPECIFIC['walk_forward_epochs'],
                    verbose=0
                )

                y_pred = model.predict(X_test_point, verbose=0)

                # Compatibilidade com múltiplos outputs
                if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
                    # Para walk-forward, usar apenas a primeira predição (mais próxima)
                    rmse = np.sqrt(mean_squared_error(y_test_point, y_pred[:, 0:1]))
                else:
                    rmse = np.sqrt(mean_squared_error(y_test_point, y_pred))

                rmse_scores.append(rmse)

            except Exception as e:
                rmse_scores.append(float('inf'))

        return np.mean(rmse_scores)

    def evaluate_final_model(self, model, X_test, y_test):
        """
        Avalia modelo final nos dados de teste não vistos

        Parameters:
        -----------
        model : trained model
            Modelo já treinado com melhores hiperparâmetros
        X_test, y_test : arrays
            Dados de teste separados no início

        Returns:
        --------
        float: RMSE no conjunto de teste
        """
        y_pred = model.predict(X_test, verbose=0)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        return test_rmse

    def evaluate_final_model_horizon_aware(self, model, X_test, y_test, lstm_model=None):
        """
        Avaliação final com análise detalhada por horizonte para contexto hidroelétrico

        Parameters:
        -----------
        model : trained model
            Modelo já treinado com melhores hiperparâmetros
        X_test, y_test : arrays
            Dados de teste separados no início
        lstm_model : LSTMModel
            Instância do modelo LSTM para acessar o scaler (opcional)

        Returns:
        --------
        dict: Análise completa da performance no teste final
        """
        from config.config import Config

        y_pred = model.predict(X_test, verbose=0)
        horizon = getattr(Config, 'PREDICTION', {}).get('horizon', 1)

        # Análise básica
        test_rmse_overall = np.sqrt(mean_squared_error(y_test, y_pred))

        if horizon == 1:
            # Single-step: análise simples
            return {
                'test_rmse_overall': test_rmse_overall,
                'test_rmse_weighted': test_rmse_overall,
                'horizon': 1,
                'horizon_metrics': [{'day': 1, 'rmse': test_rmse_overall}],
                'operational_analysis': {
                    'short_term_rmse': test_rmse_overall,
                    'medium_term_rmse': test_rmse_overall,
                    'long_term_rmse': test_rmse_overall,
                    'operational_score': test_rmse_overall
                }
            }

        # Multi-step: análise detalhada por horizonte
        horizon_metrics = []
        horizon_rmses = []

        print(f"\\n🎯 AVALIAÇÃO FINAL - ANÁLISE POR HORIZONTE (horizon={horizon}):")
        print(f"{'='*60}")

        for h in range(horizon):
            rmse_h = np.sqrt(mean_squared_error(y_test[:, h], y_pred[:, h]))
            mae_h = mean_absolute_error(y_test[:, h], y_pred[:, h])
            try:
                r2_h = r2_score(y_test[:, h], y_pred[:, h])
            except:
                r2_h = 0.0

            horizon_metrics.append({
                'day': h+1,
                'rmse': rmse_h,
                'mae': mae_h,
                'r2': r2_h
            })
            horizon_rmses.append(rmse_h)

            print(f"  Dia {h+1:2d} (t+{h+1}): RMSE={rmse_h:.3f}, MAE={mae_h:.3f}, R²={r2_h:.3f}")

        # RMSE ponderado usando pesos hidroelétricos
        weights = self._get_hydroelectric_weights(horizon)
        test_rmse_weighted = np.average(horizon_rmses, weights=weights)

        # Análise operacional hidroelétrica
        operational_analysis = self._calculate_operational_metrics(horizon_rmses, horizon)

        # Análise de degradação
        degradation_analysis = {
            'rmse_day1': horizon_rmses[0],
            'rmse_last': horizon_rmses[-1],
            'degradation_factor': horizon_rmses[-1] / horizon_rmses[0],
            'avg_degradation_per_day': (horizon_rmses[-1] - horizon_rmses[0]) / (horizon - 1)
        }

        print(f"\\n📊 RESUMO DA AVALIAÇÃO FINAL:")
        print(f"  RMSE Geral (não-ponderado): {test_rmse_overall:.4f}")
        print(f"  RMSE Ponderado (hidroelétrico): {test_rmse_weighted:.4f}")
        print(f"\\n🏭 ANÁLISE OPERACIONAL:")
        print(f"  Score Operacional: {operational_analysis['operational_score']:.4f}")
        print(f"  Curto Prazo (D1-3): {operational_analysis['short_term_rmse']:.4f}")
        print(f"  Médio Prazo (D4-7): {operational_analysis['medium_term_rmse']:.4f}")
        print(f"  Longo Prazo (D8+): {operational_analysis['long_term_rmse']:.4f}")
        print(f"\\n📉 DEGRADAÇÃO:")
        print(f"  Fator de degradação: {degradation_analysis['degradation_factor']:.1f}x")
        print(f"  Taxa média: +{degradation_analysis['avg_degradation_per_day']:.3f} RMSE/dia")
        print(f"{'='*60}")

        return {
            'test_rmse_overall': test_rmse_overall,
            'test_rmse_weighted': test_rmse_weighted,
            'horizon': horizon,
            'horizon_metrics': horizon_metrics,
            'operational_analysis': operational_analysis,
            'degradation_analysis': degradation_analysis,
            'weights_used': weights.tolist()
        }

    def _calculate_operational_metrics(self, horizon_rmses, horizon):
        """
        Calcula métricas operacionais para contexto hidroelétrico
        """
        # Definir períodos operacionais
        short_term_days = min(3, horizon)
        medium_term_days = min(7, horizon)

        # RMSE por período operacional
        short_term_rmse = np.mean(horizon_rmses[:short_term_days])

        if horizon > 3:
            medium_term_rmse = np.mean(horizon_rmses[3:medium_term_days])
        else:
            medium_term_rmse = short_term_rmse

        if horizon > 7:
            long_term_rmse = np.mean(horizon_rmses[7:])
        else:
            long_term_rmse = medium_term_rmse

        # Score operacional composto
        operational_score = (0.50 * short_term_rmse +
                           0.35 * medium_term_rmse +
                           0.15 * long_term_rmse)

        return {
            'operational_score': operational_score,
            'short_term_rmse': short_term_rmse,
            'medium_term_rmse': medium_term_rmse,
            'long_term_rmse': long_term_rmse
        }