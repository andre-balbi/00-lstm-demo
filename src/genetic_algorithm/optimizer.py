import numpy as np
import random
from deap import base, creator, tools, algorithms
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from ..utils.validation import TimeSeriesValidator
from config.config import Config


class GeneticOptimizer:
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
        self.setup_genetic_algorithm()

    def setup_genetic_algorithm(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()

        self.toolbox.register("attr_lstm_units", random.randint, 32, 128)
        self.toolbox.register("attr_dropout", random.uniform, 0.1, 0.5)
        self.toolbox.register("attr_lr", random.uniform, 0.001, 0.01)
        self.toolbox.register("attr_batch_idx", random.randint, 0, 2)

        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                             (self.toolbox.attr_lstm_units,
                              self.toolbox.attr_dropout,
                              self.toolbox.attr_lr,
                              self.toolbox.attr_batch_idx), n=1)

        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate_individual(self, individual):
        try:
            print(f"  Avaliando indivíduo: LSTM={int(individual[0])}, dropout={individual[1]:.3f}, lr={individual[2]:.4f}, batch={[16,32,64][int(individual[3])]}")

            # Usar validação baseada no método configurado
            validation_method = Config.VALIDATION['method']

            if validation_method == 'time_series_cv':
                # Validação cruzada temporal
                rmse = self.validator.time_series_cross_validate(
                    self.X_full, self.y_full,
                    self.lstm_model.create_model,
                    individual
                )
            else:
                # Validação simples (temporal_split)
                model = self.lstm_model.create_model(individual)
                batch_sizes = [16, 32, 64]
                batch_size = batch_sizes[int(individual[3])]

                model.fit(
                    self.X_train, self.y_train,
                    batch_size=batch_size,
                    epochs=Config.TRAINING['epochs'],
                    validation_data=(self.X_val, self.y_val),
                    verbose=0,
                    callbacks=[tf.keras.callbacks.EarlyStopping(
                        patience=10, restore_best_weights=True
                    )]
                )

                # Use direct model call to avoid tf.function retracing
                y_pred = model(self.X_val, training=False).numpy()
                rmse = np.sqrt(mean_squared_error(self.y_val, y_pred))

            print(f"  → RMSE: {rmse:.4f}")
            return (rmse,)

        except Exception as e:
            print(f"  → Erro na avaliação: {e}")
            return (float('inf'),)

    def optimize(self, X, y, population_size=20, generations=10):
        # Armazenar dados completos para validação cruzada
        self.X_full = X
        self.y_full = y

        # Divisão temporal dos dados
        validation_method = Config.VALIDATION['method']

        if validation_method in ['temporal_split', 'walk_forward']:
            # Usar validação temporal estruturada
            X_train, X_val, _, y_train, y_val, _ = self.validator.temporal_split(X, y)
            self.X_train = X_train
            self.X_val = X_val
            self.y_train = y_train
            self.y_val = y_val

            print(f"Divisão temporal: Train={len(X_train)}, Val={len(X_val)}")

        print("Iniciando otimização com Algoritmo Genético...")
        print(f"População: {population_size} indivíduos, {generations} gerações")
        print(f"Método de validação: {validation_method}")

        population = self.toolbox.population(n=population_size)
        print(f"Avaliando população inicial ({population_size} indivíduos)...")

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)

        hall_of_fame = tools.HallOfFame(1)

        algorithms.eaSimple(
            population, self.toolbox,
            cxpb=0.7,
            mutpb=0.2,
            ngen=generations,
            stats=stats,
            halloffame=hall_of_fame,
            verbose=True
        )

        best_params = hall_of_fame[0]
        print("Melhores parâmetros encontrados:")
        print(f"   LSTM Units: {int(best_params[0])}")
        print(f"   Dropout Rate: {best_params[1]:.3f}")
        print(f"   Learning Rate: {best_params[2]:.4f}")
        print(f"   Batch Size: {[16, 32, 64][int(best_params[3])]}")
        print(f"   RMSE: {hall_of_fame[0].fitness.values[0]:.4f}")

        return best_params