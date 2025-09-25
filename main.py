from src.models.lstm_model import LSTMModel
from src.genetic_algorithm.optimizer import GeneticOptimizer
from src.data.synthetic_data import generate_synthetic_data
from src.utils.visualization import Visualizer
from config.config import Config


class LSTMGeneticOptimizer:
    def __init__(self, sequence_length=None, n_features=None):
        self.sequence_length = sequence_length or Config.SEQUENCE_LENGTH
        self.n_features = n_features or Config.N_FEATURES

        self.lstm_model = LSTMModel(self.sequence_length, self.n_features)
        self.genetic_optimizer = GeneticOptimizer(self.lstm_model)
        self.visualizer = Visualizer(self.lstm_model)
        self.best_params = None

    def prepare_sequences(self, data, target_col='vazao'):
        return self.lstm_model.prepare_sequences(data, target_col)

    def optimize_with_genetic_algorithm(self, X, y, population_size=None, generations=None):
        population_size = population_size or Config.GENETIC_ALGORITHM['population_size']
        generations = generations or Config.GENETIC_ALGORITHM['generations']

        self.best_params = self.genetic_optimizer.optimize(X, y, population_size, generations)
        return self.best_params

    def train_final_model(self, X, y, epochs=None):
        epochs = epochs or Config.TRAINING['epochs']

        print("Treinando modelo final com parâmetros otimizados...")
        history, metrics = self.lstm_model.train(X, y, self.best_params, epochs)

        print(f"Métricas do modelo final:")
        print(f"   === TESTE (nunca visto) ===")
        print(f"   RMSE: {metrics['rmse']:.2f}")
        print(f"   MAE: {metrics['mae']:.2f}")
        print(f"   R²: {metrics['r2']:.3f}")
        print(f"   === VALIDAÇÃO ===")
        print(f"   RMSE: {metrics['val_rmse']:.2f}")
        print(f"   MAE: {metrics['val_mae']:.2f}")
        print(f"   R²: {metrics['val_r2']:.3f}")

        return history, metrics

    def predict(self, X):
        return self.lstm_model.predict(X)

    def plot_predictions(self, X, y_true, title="Previsões vs Realidade"):
        self.visualizer.plot_predictions(X, y_true, title)


if __name__ == "__main__":
    print("Sistema LSTM + Algoritmos Genéticos para Modelagem Chuva-Vazão")
    print("=" * 60)

    data = generate_synthetic_data(n_samples=Config.N_SAMPLES)
    features_data = data.drop(['data'], axis=1)

    optimizer = LSTMGeneticOptimizer()

    print("\nPreparando sequências temporais...")
    X, y = optimizer.prepare_sequences(features_data, target_col='vazao')
    print(f"Formato das sequências: X={X.shape}, y={y.shape}")

    print("\nExecutando otimização híbrida LSTM + AG...")
    best_individual = optimizer.optimize_with_genetic_algorithm(X, y)

    print("\nTreinando modelo final...")
    history, metrics = optimizer.train_final_model(X, y)

    X_test = X[-100:]
    y_test = y[-100:]
    optimizer.plot_predictions(X_test, y_test, title="Previsão Chuva-Vazão Otimizada")

    print("\nOtimização concluída com sucesso!")
    print("O modelo híbrido LSTM + AG está pronto para previsões!")