from src.models.lstm_model import LSTMModel
from src.optuna_optimizer.optimizer import OptunaOptimizer
from src.data.synthetic_data import generate_synthetic_data
from src.utils.visualization import Visualizer
from config.config import Config


class LSTMOptunaOptimizer:
    def __init__(self, sequence_length=None, n_features=None):
        self.sequence_length = sequence_length or Config.SEQUENCE_LENGTH
        self.n_features = n_features or Config.N_FEATURES

        self.lstm_model = LSTMModel(self.sequence_length, self.n_features)
        self.optuna_optimizer = OptunaOptimizer(self.lstm_model)
        self.visualizer = Visualizer(self.lstm_model)
        self.best_params = None

    def prepare_sequences(self, data, target_col='vazao'):
        return self.lstm_model.prepare_sequences(data, target_col)

    def optimize_with_optuna(self, X, y, n_trials=None):
        n_trials = n_trials or Config.OPTUNA['n_trials']

        self.best_params, self.best_optuna_params = self.optuna_optimizer.optimize(X, y, n_trials)
        return self.best_params

    def evaluate_final_model(self):
        """
        Avalia o modelo final nos dados de teste não vistos durante otimização
        """
        if self.best_params is None:
            raise ValueError("Execute optimize_with_optuna() primeiro!")

        print("\nAvaliando modelo final nos dados de teste não vistos...")
        test_rmse = self.optuna_optimizer.evaluate_on_test_data(self.best_params)

        return test_rmse

    def predict(self, X):
        return self.lstm_model.predict(X)

    def plot_predictions(self, X, y_true, title="Previsões vs Realidade"):
        """
        Plota predições usando o modelo final treinado no otimizador
        """
        if hasattr(self.optuna_optimizer, 'final_model'):
            # Usar modelo final do otimizador se disponível
            y_pred = self.optuna_optimizer.final_model.predict(X, verbose=0)
            self.visualizer.plot_predictions(X, y_true, y_pred, title)
        else:
            # Fallback para método original (predições calculadas internamente)
            self.visualizer.plot_predictions(X, y_true, title=title)


if __name__ == "__main__":
    print("Sistema LSTM + Optuna para Modelagem Chuva-Vazão")
    print("=" * 50)

    data = generate_synthetic_data(n_samples=Config.N_SAMPLES)
    features_data = data.drop(['data'], axis=1)

    optimizer = LSTMOptunaOptimizer()

    print("\nPreparando sequências temporais...")
    X, y = optimizer.prepare_sequences(features_data, target_col='vazao')
    print(f"Formato das sequências: X={X.shape}, y={y.shape}")

    print("\nExecutando otimização híbrida LSTM + Optuna...")
    best_individual = optimizer.optimize_with_optuna(X, y)

    print("\nAvaliando modelo final nos dados de teste não vistos...")
    test_rmse = optimizer.evaluate_final_model()

    print(f"\n{'='*50}")
    print(f"RESULTADO FINAL:")
    print(f"   Validation RMSE (durante otimização): {optimizer.optuna_optimizer.study.best_value:.4f}")
    print(f"   Test RMSE (dados não vistos): {test_rmse:.4f}")
    print(f"   Diferença: {abs(test_rmse - optimizer.optuna_optimizer.study.best_value):.4f}")
    print(f"{'='*50}")

    # Gerar visualizações com dados de teste
    print("\nGerando visualizações...")
    X_test_viz = optimizer.optuna_optimizer.X_test
    y_test_viz = optimizer.optuna_optimizer.y_test
    optimizer.plot_predictions(X_test_viz, y_test_viz, title="Previsão Chuva-Vazão - Dados de Teste")

    print("\nOtimização concluída !")