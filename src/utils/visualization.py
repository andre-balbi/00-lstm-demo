import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from config.config import Config

plt.style.use('dark_background')
sns.set_style("darkgrid")


class Visualizer:
    def __init__(self, lstm_model):
        self.lstm_model = lstm_model
        self.horizon = getattr(Config, 'PREDICTION', {}).get('horizon', 1)

    def _detect_prediction_format(self, y_pred):
        """
        Detecta automaticamente o formato das predições

        Returns:
            tuple: (is_multistep, horizon, n_samples)
        """
        if len(y_pred.shape) == 1:
            return False, 1, len(y_pred)
        elif len(y_pred.shape) == 2:
            if y_pred.shape[1] == 1:
                return False, 1, y_pred.shape[0]
            else:
                return True, y_pred.shape[1], y_pred.shape[0]
        else:
            raise ValueError(f"Formato de predição não suportado: {y_pred.shape}")

    def _prepare_data_for_visualization(self, y_true, y_pred):
        """
        Prepara dados para visualização, lidando com diferentes formatos

        Returns:
            tuple: (y_true_orig, y_pred_orig, is_multistep, horizon)
        """
        is_multistep, horizon, n_samples = self._detect_prediction_format(y_pred)

        if hasattr(self.lstm_model, 'scaler_y') and self.lstm_model.scaler_y is not None:
            if is_multistep:
                # Multi-step: shape (n_samples, horizon)
                y_true_orig = self.lstm_model.scaler_y.inverse_transform(y_true.reshape(-1, 1)).reshape(n_samples, horizon)
                y_pred_orig = self.lstm_model.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).reshape(n_samples, horizon)
            else:
                # Single-step: shape (n_samples,)
                y_true_orig = self.lstm_model.scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()
                y_pred_orig = self.lstm_model.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        else:
            y_true_orig = y_true
            y_pred_orig = y_pred

        return y_true_orig, y_pred_orig, is_multistep, horizon

    def plot_predictions(self, X, y_true, y_pred=None, title="Previsões vs Realidade", show_samples=100):
        """
        Plota predições vs valores reais - GENÉRICO para qualquer horizonte
        """
        if y_pred is None:
            y_pred = self.lstm_model.predict(X)

        y_true_orig, y_pred_orig, is_multistep, horizon = self._prepare_data_for_visualization(y_true, y_pred)

        print(f"\n📊 VISUALIZAÇÃO: {'Multi-step' if is_multistep else 'Single-step'} (horizon={horizon})")

        if is_multistep:
            self._plot_multistep_predictions(y_true_orig, y_pred_orig, horizon, title, show_samples)
        else:
            self._plot_singlestep_predictions(y_true_orig, y_pred_orig, title, show_samples)

    def _plot_singlestep_predictions(self, y_true, y_pred, title, show_samples):
        """
        Plot para predições de 1 passo (horizon=1)
        """
        plt.figure(figsize=(15, 6))

        # Plot temporal
        plt.subplot(1, 2, 1)
        samples_to_show = min(show_samples, len(y_true))
        plt.plot(y_true[-samples_to_show:], label='Real', color='cyan', linewidth=2)
        plt.plot(y_pred[-samples_to_show:], label='Predito', color='orange', linewidth=2, alpha=0.8)
        plt.title(f"{title}\nPredição de 1 step (Últimos {samples_to_show} pontos)")
        plt.xlabel("Tempo")
        plt.ylabel("Vazão")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Scatter plot
        plt.subplot(1, 2, 2)
        plt.scatter(y_true, y_pred, alpha=0.6, color='lightblue')
        plt.plot([y_true.min(), y_true.max()],
                 [y_true.min(), y_true.max()],
                 'r--', linewidth=2, label='Linha Perfeita')
        plt.xlabel("Vazão Real")
        plt.ylabel("Vazão Predita")
        plt.title("Correlação Real vs Predito")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _plot_multistep_predictions(self, y_true, y_pred, horizon, title, show_samples):
        """
        Plot para predições multi-step (horizon>1) - GENÉRICO
        """
        if horizon <= 4:
            rows, cols = 2, 2
        elif horizon <= 6:
            rows, cols = 2, 3
        elif horizon <= 9:
            rows, cols = 3, 3
        elif horizon <= 12:
            rows, cols = 3, 4
        else:
            rows, cols = 4, 4

        fig = plt.figure(figsize=(5*cols, 4*rows))

        samples_to_show = min(show_samples, len(y_true))
        colors = plt.cm.viridis(np.linspace(0, 1, horizon))

        plots_needed = min(horizon, rows*cols-2)

        for h in range(plots_needed):
            plt.subplot(rows, cols, h+1)
            plt.plot(y_true[-samples_to_show:, h], label=f'Real S{h+1}',
                    color='cyan', linewidth=2, alpha=0.8)
            plt.plot(y_pred[-samples_to_show:, h], label=f'Pred S{h+1}',
                    color=colors[h], linewidth=2, alpha=0.8)
            plt.title(f"Step {h+1} (t+{h+1})")
            plt.xlabel("Tempo")
            plt.ylabel("Vazão")
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)

        if plots_needed < rows*cols-1:
            plt.subplot(rows, cols, plots_needed+1)
            rmse_by_horizon = [np.sqrt(np.mean((y_true[:, h] - y_pred[:, h])**2)) for h in range(horizon)]
            bars = plt.bar(range(1, horizon+1), rmse_by_horizon, color=colors, alpha=0.7)
            plt.title("RMSE por Step")
            plt.xlabel("Step à Frente")
            plt.ylabel("RMSE")
            plt.grid(True, alpha=0.3)

            for bar, rmse in zip(bars, rmse_by_horizon):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(rmse_by_horizon),
                        f'{rmse:.2f}', ha='center', va='bottom', fontsize=8)

        if plots_needed < rows*cols:
            plt.subplot(rows, cols, plots_needed+2)
            plt.scatter(y_true.flatten(), y_pred.flatten(), alpha=0.4, color='lightblue', s=1)
            min_val, max_val = y_true.min(), y_true.max()
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Linha Perfeita')
            plt.xlabel("Vazão Real")
            plt.ylabel("Vazão Predita")
            plt.title(f"Correlação Geral\n(Todos os {horizon} steps)")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.suptitle(f"{title} - Predição Multi-step: {horizon} steps consecutivos", fontsize=16, y=0.98)
        plt.tight_layout()
        plt.show()

        print(f"\n📈 ESTATÍSTICAS POR STEP:")
        for h in range(horizon):
            rmse_h = np.sqrt(np.mean((y_true[:, h] - y_pred[:, h])**2))
            mae_h = np.mean(np.abs(y_true[:, h] - y_pred[:, h]))
            corr_h = np.corrcoef(y_true[:, h], y_pred[:, h])[0,1]
            print(f"  Step {h+1}: RMSE={rmse_h:.3f}, MAE={mae_h:.3f}, Corr={corr_h:.3f}")

    def plot_horizon_degradation_analysis(self, X, y_true, y_pred=None):
        """
        Análise específica da degradação por horizonte (só para multi-step)
        """
        if y_pred is None:
            y_pred = self.lstm_model.predict(X)

        y_true_orig, y_pred_orig, is_multistep, horizon = self._prepare_data_for_visualization(y_true, y_pred)

        if not is_multistep:
            print("⚠️ Análise de degradação só disponível para predições multi-step (horizon>1)")
            return

        metrics_by_horizon = []
        for h in range(horizon):
            rmse = np.sqrt(np.mean((y_true_orig[:, h] - y_pred_orig[:, h])**2))
            mae = np.mean(np.abs(y_true_orig[:, h] - y_pred_orig[:, h]))
            corr = np.corrcoef(y_true_orig[:, h], y_pred_orig[:, h])[0,1]
            metrics_by_horizon.append({'rmse': rmse, 'mae': mae, 'corr': corr})

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        steps = range(1, horizon+1)
        rmses = [m['rmse'] for m in metrics_by_horizon]
        maes = [m['mae'] for m in metrics_by_horizon]
        corrs = [m['corr'] for m in metrics_by_horizon]

        axes[0,0].plot(steps, rmses, 'o-', color='red', linewidth=2, markersize=6)
        axes[0,0].set_title('Degradação RMSE por Step')
        axes[0,0].set_xlabel('Step à Frente')
        axes[0,0].set_ylabel('RMSE')
        axes[0,0].grid(True, alpha=0.3)
        for i, rmse in enumerate(rmses):
            axes[0,0].text(i+1, rmse + 0.02*max(rmses), f'{rmse:.2f}', ha='center')

        axes[0,1].plot(steps, maes, 'o-', color='orange', linewidth=2, markersize=6)
        axes[0,1].set_title('Degradação MAE por Step')
        axes[0,1].set_xlabel('Step à Frente')
        axes[0,1].set_ylabel('MAE')
        axes[0,1].grid(True, alpha=0.3)

        axes[1,0].plot(steps, corrs, 'o-', color='green', linewidth=2, markersize=6)
        axes[1,0].set_title('Correlação por Step')
        axes[1,0].set_xlabel('Step à Frente')
        axes[1,0].set_ylabel('Correlação')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].set_ylim(0, 1)

        errors_step1 = y_true_orig[:, 0] - y_pred_orig[:, 0]
        errors_last = y_true_orig[:, -1] - y_pred_orig[:, -1]

        axes[1,1].hist(errors_step1, alpha=0.7, bins=20, label=f'Step 1', color='cyan')
        axes[1,1].hist(errors_last, alpha=0.7, bins=20, label=f'Step {horizon}', color='red')
        axes[1,1].set_title('Distribuição de Erros: Primeiro vs Último Step')
        axes[1,1].set_xlabel('Erro (Real - Predito)')
        axes[1,1].set_ylabel('Frequência')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)

        plt.suptitle(f'Análise de Degradação - Horizonte {horizon} Steps', fontsize=16)
        plt.tight_layout()
        plt.show()

        print(f"\n📊 RELATÓRIO DE DEGRADAÇÃO (horizon={horizon}):")
        print(f"  • RMSE aumenta {rmses[-1]/rmses[0]:.1f}x do step 1 ao step {horizon}")
        print(f"  • MAE aumenta {maes[-1]/maes[0]:.1f}x do step 1 ao step {horizon}")
        print(f"  • Correlação diminui de {corrs[0]:.3f} para {corrs[-1]:.3f}")

        degradation_rate = (rmses[-1] - rmses[0]) / (horizon - 1)
        print(f"  • Taxa de degradação RMSE: +{degradation_rate:.3f} por step")

    def quick_plot(self, X, y_true, y_pred=None, samples=50):
        """
        Plot rápido e simples - detecta automaticamente o formato
        """
        if y_pred is None:
            y_pred = self.lstm_model.predict(X)

        y_true_orig, y_pred_orig, is_multistep, horizon = self._prepare_data_for_visualization(y_true, y_pred)

        plt.figure(figsize=(12, 4))

        if is_multistep:
            plt.subplot(1, 2, 1)
            plt.plot(y_true_orig[:samples, 0], label='Real S1', color='cyan', linewidth=2)
            plt.plot(y_pred_orig[:samples, 0], label='Pred S1', color='orange', linewidth=2)
            plt.title(f'Step 1 (t+1)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.subplot(1, 2, 2)
            plt.plot(y_true_orig[:samples, -1], label=f'Real S{horizon}', color='cyan', linewidth=2)
            plt.plot(y_pred_orig[:samples, -1], label=f'Pred S{horizon}', color='red', linewidth=2)
            plt.title(f'Step {horizon} (t+{horizon})')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.suptitle(f'Quick View: Multi-step Prediction (horizon={horizon})')
        else:
            plt.plot(y_true_orig[:samples], label='Real', color='cyan', linewidth=2)
            plt.plot(y_pred_orig[:samples], label='Predito', color='orange', linewidth=2)
            plt.title('Single-step Prediction')
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
