import numpy as np
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from ..utils.validation import TimeSeriesValidator
from config.config import Config


class OptunaOptimizer:
    def __init__(self, lstm_model):
        self.lstm_model = lstm_model
        self.validator = TimeSeriesValidator(
            n_splits=Config.VALIDATION['cv_splits'],
            test_size=Config.VALIDATION['test_size']
        )
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None
        self.X_optim = None  # Dados para otimização (sem teste)
        self.y_optim = None  # Dados para otimização (sem teste)
        self.X_test = None   # Dados de teste final
        self.y_test = None   # Dados de teste final
        self.horizon = Config.PREDICTION.get('horizon', 1)
        self.trial_logs = []  # Histórico de trials para análise

    def _log_detailed_metrics(self, trial_num, y_true, y_pred, params, rmse_overall):
        """
        Log detalhado das métricas por horizonte durante otimização
        """
        print(f"\n{'='*60}")
        print(f"🔍 TRIAL {trial_num} - ANÁLISE DETALHADA")
        print(f"{'='*60}")

        if self.horizon == 1:
            print(f"📊 Single-step Prediction:")
            print(f"   RMSE: {rmse_overall:.4f}")
            trial_log = {
                'trial': trial_num,
                'params': params,
                'rmse_overall': rmse_overall,
                'horizon': 1
            }
        else:
            print(f"📊 Multi-step Prediction (horizon={self.horizon}):")
            print(f"   RMSE Geral: {rmse_overall:.4f}")

            # Calcular métricas por horizonte
            horizon_metrics = []
            print(f"\n📈 MÉTRICAS POR HORIZONTE:")

            for h in range(self.horizon):
                if len(y_true.shape) > 1 and y_true.shape[1] > 1:
                    # Multi-step: calcular por coluna
                    rmse_h = np.sqrt(mean_squared_error(y_true[:, h], y_pred[:, h]))
                    mae_h = mean_absolute_error(y_true[:, h], y_pred[:, h])
                    try:
                        r2_h = r2_score(y_true[:, h], y_pred[:, h])
                    except:
                        r2_h = 0.0
                else:
                    # Fallback para single-step
                    rmse_h = rmse_overall
                    mae_h = mean_absolute_error(y_true.flatten(), y_pred.flatten())
                    r2_h = r2_score(y_true.flatten(), y_pred.flatten())

                horizon_metrics.append({
                    'day': h+1,
                    'rmse': rmse_h,
                    'mae': mae_h,
                    'r2': r2_h
                })

                print(f"   Step {h+1:2d} (t+{h+1}): RMSE={rmse_h:.3f}, MAE={mae_h:.3f}, R²={r2_h:.3f}")

                if h >= 15:  # Limitar output para horizontes muito grandes
                    remaining = self.horizon - h - 1
                    if remaining > 0:
                        print(f"   ... (+{remaining} steps)")
                    break

            # Análise de degradação
            if len(horizon_metrics) > 1:
                degradation = horizon_metrics[-1]['rmse'] / horizon_metrics[0]['rmse']
                print(f"\n📉 DEGRADAÇÃO:")
                print(f"   RMSE aumenta {degradation:.1f}x do step 1 ao step {len(horizon_metrics)}")

                avg_degradation = (horizon_metrics[-1]['rmse'] - horizon_metrics[0]['rmse']) / (len(horizon_metrics) - 1)
                print(f"   Taxa média: +{avg_degradation:.3f} RMSE por step")

            trial_log = {
                'trial': trial_num,
                'params': params,
                'rmse_overall': rmse_overall,
                'horizon': self.horizon,
                'horizon_metrics': horizon_metrics,
                'degradation': degradation if len(horizon_metrics) > 1 else 1.0
            }

        # Salvar log para análise posterior
        self.trial_logs.append(trial_log)

        print(f"{'='*60}\n")

    def _print_optimization_progress(self, study, trial_num, total_trials):
        """
        Mostra progresso da otimização com estatísticas e análise operacional
        """
        if trial_num > 0:
            best_value = study.best_value
            current_best_trial = study.best_trial.number

            print(f"\\n🎯 PROGRESSO DA OTIMIZAÇÃO [{trial_num}/{total_trials}]:")
            print(f"   Melhor RMSE até agora: {best_value:.4f} (Trial {current_best_trial})")

            # Análise de tendência geral
            if len(self.trial_logs) >= 3:
                recent_rmses = [log['rmse_overall'] for log in self.trial_logs[-3:]]
                trend = "📈" if recent_rmses[-1] < recent_rmses[0] else "📉"
                improvement = abs(recent_rmses[-1] - recent_rmses[0])
                print(f"   Tendência recente: {trend} {recent_rmses[-3]:.4f} → {recent_rmses[-1]:.4f} (Δ{improvement:.4f})")

            # Análise operacional específica para multi-step
            if self.horizon > 1 and len(self.trial_logs) >= 2:
                self._show_operational_progress_insights(trial_num, total_trials)

            # Estimativa de tempo restante e qualidade
            if trial_num >= 3:
                self._show_optimization_insights(trial_num, total_trials)

    def _show_operational_progress_insights(self, trial_num, total_trials):
        """
        Insights operacionais específicos para otimização hidroelétrica
        """
        # Análise baseada nas métricas individuais por horizonte (valores corretos)
        if len(self.trial_logs) >= 2 and self.horizon > 1:
            recent_trials = self.trial_logs[-3:]

            # Análise de consistência da degradação
            consistent_degradations = []
            for trial in recent_trials:
                if 'horizon_metrics' in trial and len(trial['horizon_metrics']) > 1:
                    day1_rmse = trial['horizon_metrics'][0]['rmse']
                    day_last_rmse = trial['horizon_metrics'][-1]['rmse']
                    degradation = day_last_rmse / day1_rmse
                    consistent_degradations.append(degradation)

            if len(consistent_degradations) >= 2:
                avg_degradation = np.mean(consistent_degradations)
                degradation_icon = "🟢" if avg_degradation < 1.3 else "🟡" if avg_degradation < 1.5 else "🔴"
                print(f"   Degradação média: {degradation_icon} {avg_degradation:.1f}x (últimos {len(consistent_degradations)} trials)")

                # Análise da estabilidade
                degradation_std = np.std(consistent_degradations)
                stability_icon = "🎯" if degradation_std < 0.1 else "⚡"
                print(f"   Estabilidade: {stability_icon} Variação de ±{degradation_std:.2f}x")

    def _show_optimization_insights(self, trial_num, total_trials):
        """
        Insights e estatísticas avançadas da otimização
        """
        # Análise de convergência
        recent_best_values = []
        for i in range(max(0, len(self.trial_logs)-5), len(self.trial_logs)):
            if i < len(self.trial_logs):
                recent_best_values.append(min([log['rmse_overall'] for log in self.trial_logs[:i+1]]))

        if len(recent_best_values) >= 2:
            improvement_rate = (recent_best_values[0] - recent_best_values[-1]) / len(recent_best_values)
            if improvement_rate > 0.001:
                convergence_status = "🚀 Convergindo rapidamente"
            elif improvement_rate > 0.0001:
                convergence_status = "📈 Melhorando gradualmente"
            else:
                convergence_status = "🎯 Próximo do ótimo"

            print(f"   Status: {convergence_status}")

        # Progresso estimado
        progress_pct = (trial_num / total_trials) * 100
        if progress_pct >= 50:
            trials_remaining = total_trials - trial_num
            print(f"   Progresso: {progress_pct:.0f}% • {trials_remaining} trials restantes")

            # Sugestão baseada na performance
            if len(self.trial_logs) >= 5:
                stability = np.std([log['rmse_overall'] for log in self.trial_logs[-5:]])
                if stability < 0.01:
                    print(f"   💡 Dica: Performance estabilizada, considere aumentar epochs para refinamento")

    def _print_optimization_summary(self):
        """
        Resumo final da otimização com análise de todos os trials
        """
        print(f"\n{'='*70}")
        print(f"📋 RESUMO FINAL DA OTIMIZAÇÃO")
        print(f"{'='*70}")

        if not self.trial_logs:
            print("Nenhum log de trial disponível.")
            return

        # Estatísticas gerais
        rmse_values = [log['rmse_overall'] for log in self.trial_logs]
        best_trial_idx = rmse_values.index(min(rmse_values))
        worst_trial_idx = rmse_values.index(max(rmse_values))

        print(f"📊 ESTATÍSTICAS GERAIS:")
        print(f"   Total de trials: {len(self.trial_logs)}")
        print(f"   Melhor RMSE: {min(rmse_values):.4f} (Trial {best_trial_idx})")
        print(f"   Pior RMSE: {max(rmse_values):.4f} (Trial {worst_trial_idx})")
        print(f"   RMSE médio: {np.mean(rmse_values):.4f}")
        print(f"   Desvio padrão: {np.std(rmse_values):.4f}")

        # Análise para multi-step
        if self.horizon > 1:
            print(f"\n📈 ANÁLISE MULTI-STEP (horizon={self.horizon}):")

            # Encontrar trials com melhor/pior degradação
            trials_with_degradation = [log for log in self.trial_logs if 'degradation' in log]
            if trials_with_degradation:
                degradations = [log['degradation'] for log in trials_with_degradation]
                best_degradation_idx = degradations.index(min(degradations))
                worst_degradation_idx = degradations.index(max(degradations))

                print(f"   Melhor degradação: {min(degradations):.1f}x (Trial {trials_with_degradation[best_degradation_idx]['trial']})")
                print(f"   Pior degradação: {max(degradations):.1f}x (Trial {trials_with_degradation[worst_degradation_idx]['trial']})")
                print(f"   Degradação média: {np.mean(degradations):.1f}x")

                # Top 3 trials
                print(f"\n🏆 TOP 3 TRIALS:")
                sorted_trials = sorted(self.trial_logs, key=lambda x: x['rmse_overall'])
                for i, trial in enumerate(sorted_trials[:3]):
                    params = trial['params']
                    degradation = trial.get('degradation', 'N/A')
                    print(f"   #{i+1}: Trial {trial['trial']} - RMSE: {trial['rmse_overall']:.4f}, Degradação: {degradation:.1f}x")
                    print(f"        Params: LSTM={params['lstm_units']}, dropout={params['dropout_rate']:.3f}, lr={params['learning_rate']:.4f}")

        print(f"{'='*70}\n")

    def _get_hydroelectric_weights(self, horizon):
        """
        Pesos operacionais para contexto hidroelétrico

        Prioriza predições de curto prazo (dias 1-3) que são críticas
        para decisões operacionais imediatas de usinas hidroelétricas
        """
        if horizon == 1:
            return np.array([1.0])

        # Peso exponencial decrescente: steps iniciais são mais críticos
        # Fator 0.15 garante que o step 10 ainda tem ~20% do peso do step 1
        weights = np.exp(-0.15 * np.arange(horizon))

        # Manter pesos normalizados (soma = 1) para média ponderada correta
        weights = weights / weights.sum()

        return weights

    def _proper_denormalize_multistep(self, y_scaled, horizon):
        """
        Desnormalização correta para predições multi-step
        Replica a lógica do lstm_model.py para consistência
        """
        if horizon == 1:
            if len(y_scaled.shape) == 1:
                return self.lstm_model.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
            else:
                return self.lstm_model.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
        else:
            # Multi-step: desnormalizar cada horizonte separadamente
            n_samples = y_scaled.shape[0]
            y_denorm = np.zeros((n_samples, horizon))

            for h in range(horizon):
                y_denorm[:, h] = self.lstm_model.scaler_y.inverse_transform(
                    y_scaled[:, h].reshape(-1, 1)
                ).flatten()

            return y_denorm

    def _calculate_hydroelectric_objective(self, y_true_scaled, y_pred_scaled):
        """
        Calcula função objetivo ponderada para otimização hidroelétrica

        Args:
            y_true_scaled: Valores reais escalados
            y_pred_scaled: Valores preditos escalados

        Returns:
            float: RMSE ponderado otimizado para operações hidroelétricas
        """
        horizon = self.horizon

        # Calcular RMSE ponderado diretamente nos dados escalados
        # (a escala relativa é preservada, priorizando horizontes críticos)
        weights = self._get_hydroelectric_weights(horizon)

        if horizon == 1:
            # Single-step: RMSE simples nos dados escalados
            return np.sqrt(mean_squared_error(y_true_scaled, y_pred_scaled))
        else:
            # Multi-step: RMSE por horizonte nos dados escalados
            horizon_rmses = []
            for h in range(horizon):
                rmse_h = np.sqrt(mean_squared_error(y_true_scaled[:, h], y_pred_scaled[:, h]))
                horizon_rmses.append(rmse_h)

            # Média ponderada: Σ(rmse_h * weight_h) onde Σ(weight_h) = 1
            weighted_rmse = np.average(horizon_rmses, weights=weights)

            # Log para debugging durante otimização (valores escalados)
            unweighted_rmse = np.sqrt(mean_squared_error(y_true_scaled, y_pred_scaled))
            print(f"    DEBUG: RMSE ponderado (escalado): {weighted_rmse:.4f}, RMSE geral (escalado): {unweighted_rmse:.4f}")

            return weighted_rmse


    def objective(self, trial):
        """Objective function for Optuna optimization."""
        try:
            # Suggest hyperparameters
            lstm_units = trial.suggest_int('lstm_units',
                                         Config.LSTM_PARAMS_BOUNDS['units_min'],
                                         Config.LSTM_PARAMS_BOUNDS['units_max'])
            dropout_rate = trial.suggest_float('dropout_rate',
                                             Config.LSTM_PARAMS_BOUNDS['dropout_min'],
                                             Config.LSTM_PARAMS_BOUNDS['dropout_max'])
            learning_rate = trial.suggest_float('learning_rate',
                                              Config.LSTM_PARAMS_BOUNDS['lr_min'],
                                              Config.LSTM_PARAMS_BOUNDS['lr_max'], log=True)
            batch_size = trial.suggest_categorical('batch_size', Config.LSTM_PARAMS_BOUNDS['batch_sizes'])

            # Convert to individual format for compatibility
            individual = [lstm_units, dropout_rate, learning_rate,
                         Config.LSTM_PARAMS_BOUNDS['batch_sizes'].index(batch_size)]

            # Show optimization weights for multi-step (first trial only)
            if trial.number == 0 and self.horizon > 1:
                weights = self._get_hydroelectric_weights(self.horizon)
                print(f"\\n  🏭 OTIMIZAÇÃO HIDROELÉTRICA (horizon={self.horizon}):")
                print(f"     Pesos operacionais por step: {[f'{w:.2f}' for w in weights]}")
                print(f"     → Steps 1-3 recebem {weights[:3].sum()/weights.sum()*100:.0f}% do peso total")
                print(f"     → Priorizando predições de curto prazo\\n")

            print(f"  Trial {trial.number}: LSTM={lstm_units}, dropout={dropout_rate:.3f}, lr={learning_rate:.4f}, batch={batch_size}")

            # Use validation method from config
            validation_method = Config.VALIDATION['method']

            if validation_method == 'time_series_cv':
                # Time series cross-validation (apenas dados de otimização)
                rmse = self.validator.time_series_cross_validate(
                    self.X_optim, self.y_optim,
                    self.lstm_model.create_model,
                    individual
                )
            elif validation_method == 'walk_forward':
                # Walk-forward validation (apenas dados de otimização)
                rmse = self.validator.walk_forward_validation(
                    self.X_optim, self.y_optim,
                    self.lstm_model.create_model,
                    individual,
                    window_size=Config.VALIDATION['walk_forward_window']
                )
            else:
                # Simple temporal split validation
                model = self.lstm_model.create_model(individual)

                model.fit(
                    self.X_train, self.y_train,
                    batch_size=batch_size,
                    epochs=Config.TRAINING['epochs'],
                    validation_data=(self.X_val, self.y_val),
                    verbose=0,
                    callbacks=[tf.keras.callbacks.EarlyStopping(
                        patience=Config.CALLBACKS['early_stopping_patience'], restore_best_weights=True
                    )]
                )

                # Predict and calculate RMSE with hydroelectric weighting
                y_pred = model(self.X_val, training=False).numpy()

                # Use weighted RMSE for multi-step or regular RMSE for single-step
                if self.horizon == 1:
                    rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))
                else:
                    # Calculate weighted RMSE using proper denormalization
                    rmse = self._calculate_hydroelectric_objective(self.y_val, y_pred)

                # Log detalhado das métricas
                params_dict = {
                    'lstm_units': lstm_units,
                    'dropout_rate': dropout_rate,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size
                }
                self._log_detailed_metrics(trial.number, self.y_val, y_pred, params_dict, rmse)

            print(f"  → RMSE: {rmse:.4f}")
            return rmse

        except Exception as e:
            print(f"  → Error in evaluation: {e}")
            return float('inf')

    def optimize(self, X, y, n_trials=20):
        """Optimize hyperparameters using Optuna."""
        # Primeiro: separar dados de teste (20%) de dados de otimização (80%)
        self.X_optim, self.X_test, self.y_optim, self.y_test = self.validator.split_for_optimization(X, y)

        print(f"Data split: Optimization={len(self.X_optim)}, Test={len(self.X_test)}")

        # Preparar dados de validação dependendo do método
        validation_method = Config.VALIDATION['method']

        if validation_method == 'temporal_split':
            # Use structured temporal validation apenas nos dados de otimização
            X_train, X_val, y_train, y_val = self.validator.temporal_split(self.X_optim, self.y_optim)
            self.X_train = X_train
            self.X_val = X_val
            self.y_train = y_train
            self.y_val = y_val

            print(f"Temporal split: Train={len(X_train)}, Val={len(X_val)}")
        elif validation_method == 'walk_forward':
            # Walk-forward validation operates on optimization data only
            print(f"Walk-forward validation: Using window size {Config.VALIDATION['walk_forward_window']}")
        elif validation_method == 'time_series_cv':
            # Time series CV operates on optimization data only
            print(f"Time series CV: Using {Config.VALIDATION['cv_splits']} splits")

        print("Starting optimization with Optuna...")
        print(f"Number of trials: {n_trials}")
        print(f"Validation method: {validation_method}")

        # Create study and optimize
        self.study = optuna.create_study(direction='minimize')

        # Callback para progresso
        def progress_callback(study, trial):
            self._print_optimization_progress(study, trial.number, n_trials)

        self.study.optimize(self.objective, n_trials=n_trials, callbacks=[progress_callback])

        # Get best parameters
        best_params = self.study.best_params
        best_value = self.study.best_value

        # Resumo final da otimização
        self._print_optimization_summary()

        print("Best parameters found:")
        print(f"   LSTM Units: {best_params['lstm_units']}")
        print(f"   Dropout Rate: {best_params['dropout_rate']:.3f}")
        print(f"   Learning Rate: {best_params['learning_rate']:.4f}")
        print(f"   Batch Size: {best_params['batch_size']}")
        print(f"   RMSE: {best_value:.4f}")

        # Convert to individual format for compatibility with existing code
        batch_idx = Config.LSTM_PARAMS_BOUNDS['batch_sizes'].index(best_params['batch_size'])
        best_individual = [
            best_params['lstm_units'],
            best_params['dropout_rate'],
            best_params['learning_rate'],
            batch_idx
        ]

        return best_individual, best_params

    def evaluate_on_test_data(self, best_individual):
        """
        Treina modelo final com melhores hiperparâmetros e avalia em dados de teste

        Parameters:
        -----------
        best_individual : list
            Melhores hiperparâmetros encontrados pela otimização

        Returns:
        --------
        float: RMSE no conjunto de teste não visto
        """
        print("\nTraining final model on all optimization data...")

        # Criar modelo final com melhores hiperparâmetros
        final_model = self.lstm_model.create_model(best_individual)

        # Treinar com todos os dados de otimização (sem validação = sem Early Stopping)
        batch_size = Config.LSTM_PARAMS_BOUNDS['batch_sizes'][int(best_individual[3])]

        # Treinar modelo final sem Early Stopping pois não há dados de validação
        final_model.fit(
            self.X_optim, self.y_optim,
            batch_size=batch_size,
            epochs=Config.TRAINING['final_epochs'],
            verbose=1,
            callbacks=[]  # Explicitamente sem callbacks para evitar Early Stopping warning
        )

        # Avaliar no conjunto de teste
        test_rmse = self.validator.evaluate_final_model(final_model, self.X_test, self.y_test)

        # Salvar modelo final para visualizações
        self.final_model = final_model

        print(f"\nFinal Test Results:")
        print(f"   Test RMSE: {test_rmse:.4f}")
        print(f"   Test samples: {len(self.X_test)}")

        return test_rmse