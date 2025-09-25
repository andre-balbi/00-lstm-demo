import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('dark_background')
sns.set_style("darkgrid")


class Visualizer:
    def __init__(self, lstm_model):
        self.lstm_model = lstm_model

    def plot_predictions(self, X, y_true, title="Previsões vs Realidade"):
        y_pred = self.lstm_model.predict(X)
        y_true_orig = self.lstm_model.scaler_y.inverse_transform(y_true.reshape(-1, 1)).flatten()

        plt.figure(figsize=(15, 6))

        plt.subplot(1, 2, 1)
        plt.plot(y_true_orig[-100:], label='Real', color='cyan', linewidth=2)
        plt.plot(y_pred[-100:], label='Predito', color='orange', linewidth=2, alpha=0.8)
        plt.title(f"{title}\n(Últimos 100 pontos)")
        plt.xlabel("Tempo")
        plt.ylabel("Vazão")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.scatter(y_true_orig, y_pred, alpha=0.6, color='lightblue')
        plt.plot([y_true_orig.min(), y_true_orig.max()],
                 [y_true_orig.min(), y_true_orig.max()],
                 'r--', linewidth=2, label='Linha Perfeita')
        plt.xlabel("Vazão Real")
        plt.ylabel("Vazão Predita")
        plt.title("Correlação Real vs Predito")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()