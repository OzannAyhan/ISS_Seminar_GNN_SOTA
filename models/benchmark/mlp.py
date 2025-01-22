
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
class MLPModelOptimized:



    def __init__(self, ratings):
        self.ratings = ratings
        self.X, self.y = self.prepare_data()
        self.X = self.scale_data(self.X)
        self.X_train, self.y_train, self.X_test, self.y_test = self.split_data()

    def prepare_data(self):
        X = self.ratings.drop(columns=['Unnamed: 0', 'MovieID', 'UserID', 'Rating', 'Date', 'movie_graph_id']).values
        y = np.array(self.ratings['Rating'])
        return X, y
    
    def split_data(self, test_size=0.1):
        # Determine the number of samples for the test set
        num_test_samples = int(len(self.X) * test_size)

        # Split the data
        X_test, y_test = self.X[-num_test_samples:], self.y[-num_test_samples:]
        X_train, y_train = self.X[:-num_test_samples], self.y[:-num_test_samples]

        return X_train, y_train, X_test, y_test   
    
    def scale_data(self, X):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled

    def objective(self, trial):
        # Configuration for MLPRegressor hyperparameters
        hidden_layer_sizes_option = trial.suggest_categorical('hidden_layer_sizes_option', ['50', '100', '50_50', '100_100'])
        hidden_layer_sizes = self.format_hidden_layers(hidden_layer_sizes_option)
        activation = trial.suggest_categorical('activation', ['tanh', 'relu'])
        solver = trial.suggest_categorical('solver', ['sgd', 'adam'])
        alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
        learning_rate_init = trial.suggest_float('learning_rate_init', 1e-6, 1e-2, log=True)  # Adjusted range


        # Cross-validation to evaluate model performance
        kf = KFold(n_splits=2, shuffle=True, random_state=888)
        mse_scores = []

        for train_index, test_index in kf.split(self.X_train):
            X_train_fold, X_test_fold = self.X_train[train_index], self.X_train[test_index]
            y_train_fold, y_test_fold = self.y_train[train_index], self.y_train[test_index]

            model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver,
                                alpha=alpha, learning_rate_init=learning_rate_init, max_iter=2000)
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_test_fold)
            mse = mean_squared_error(y_test_fold, y_pred)

            if not np.isfinite(mse):  # Check for non-finite MSE values
                return np.inf  # Return a large value to discard this trial

            mse_scores.append(mse)

        average_mse = np.mean(mse_scores)
        return average_mse


    def format_hidden_layers(self, option):
        if '_' in option:
            return tuple(map(int, option.split('_')))
        return (int(option),)
    
    def tune_hyperparameters(self, n_trials=10):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        self.best_params = study.best_trial.params
        self.best_params['hidden_layer_sizes'] = self.format_hidden_layers(self.best_params['hidden_layer_sizes_option'])
        del self.best_params['hidden_layer_sizes_option']
        print(f"Best parameters: {self.best_params}")

    def train_and_evaluate_model(self):
         # Now, train the model on the entire training dataset with the best parameters found
        self.model = MLPRegressor(
            hidden_layer_sizes=self.best_params['hidden_layer_sizes'],
            activation=self.best_params['activation'],
            solver=self.best_params['solver'],
            alpha=self.best_params['alpha'],
            learning_rate_init=self.best_params['learning_rate_init'],
            max_iter=2000
        )
        self.model.fit(self.X_train, self.y_train)

        # After training, evaluate the model on both the training set and test set
        y_train_pred = self.model.predict(self.X_train)
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(self.y_train, y_train_pred)

        y_test_pred = self.model.predict(self.X_test)
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)

        print(f"Training MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
        print(f"Test MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}")

    def display_predictions(self, num_examples=10):
        """Displays the actual vs. predicted ratings for a number of examples from the test set."""
        if self.model is None:
            raise Exception
        
        # Make predictions on the test data
        y_pred_test = self.model.predict(self.X_test)
        
        # Display the actual vs. predicted ratings
        print("Actual vs. Predicted Ratings:")
        for actual, predicted in zip(self.y_test[:num_examples], y_pred_test[:num_examples]):
            print(f"Actual: {actual}, Predicted: {predicted:.4f}")

