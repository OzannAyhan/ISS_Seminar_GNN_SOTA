import numpy as np
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge

class LRModel:

    def __init__(self, ratings):
        self.ratings = ratings
        self.X, self.y = self.extract_features(ratings)
        self.X_train_val, self.y_train_val, self.X_test, self.y_test = self.split_data()
        self.model = None  # Initialize the model attribute

  
    def extract_features(self, ratings):
        X = ratings.drop(columns=['Unnamed: 0', 'MovieID', 'UserID', 'Rating', 'Date', 'movie_graph_id']).values
        y = np.array(ratings['Rating'])
        return X, y

    def split_data(self, test_size=0.1):
        # Determine the number of samples for the test set
        num_test_samples = int(len(self.X) * test_size)

        # Split the data
        X_test, y_test = self.X[-num_test_samples:], self.y[-num_test_samples:]
        X_train_val, y_train_val = self.X[:-num_test_samples], self.y[:-num_test_samples]

        return X_train_val, y_train_val, X_test, y_test    
    
    def tune_hyperparameters(self, n_trials=10):
        def objective(trial):
            # Hyperparameters search space
            alpha = trial.suggest_loguniform('alpha', 0.0001, 10)
            model = Ridge(alpha=alpha)

            # Manual 2-fold cross-validation
            kf = KFold(n_splits=2, shuffle=True, random_state=888)
            rmse_scores = []

            for train_index, val_index in kf.split(self.X_train_val):
                X_train, X_val = self.X_train_val[train_index], self.X_train_val[val_index]
                y_train, y_val = self.y_train_val[train_index], self.y_train_val[val_index]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                rmse_scores.append(rmse)

            return np.mean(rmse_scores)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        self.best_alpha = study.best_params['alpha']
        print("Best alpha:", self.best_alpha)
        self.model = Ridge(alpha=self.best_alpha)  # Use the best alpha found

   
    def train_and_evaluate(self):
        if self.model is None:
            raise Exception("Hyperparameters not tuned yet. Please run tune_hyperparameters() first.")
        # Since self.model is already set to the best Ridge model, you can train it directly
        self.model.fit(self.X_train_val, self.y_train_val)
        y_pred_test = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred_test)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred_test)
        print(f"Final Evaluation on Test Set - MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    def print_actual_vs_predicted(self, dataset='test'):
        if self.model is None:
            raise Exception("Model not trained yet. Please run train_and_evaluate() first.")
 
        if dataset == 'train':
            X, y = self.X_train_val, self.y_train_val
            data_label = "Training"
        elif dataset == 'test':
            X, y = self.X_test, self.y_test
            data_label = "Test"
        else:
            raise ValueError("Invalid dataset specified. Choose 'train' or 'test'.")

        # Making predictions
        y_pred = self.model.predict(X)

        print(f"{data_label} Data - Actual vs. Predicted Ratings:")
        for actual, predicted in zip(y[:10], y_pred[:10]):  # Adjust the slice as needed
            print(f"Actual: {actual}, Predicted: {predicted:.4f}")
