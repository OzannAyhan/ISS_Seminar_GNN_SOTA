import xgboost as xgb
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

class XGBModel:
    def __init__(self, ratings, objective='reg:squarederror'):
        self.ratings = ratings
        self.extract_features()
        # Default parameters 
        self.params = {
            'objective': objective,
            'colsample_bytree': 0.3,
            'learning_rate': 0.1,
            'max_depth': 5,
            'alpha': 10,
            'n_estimators': 100
        }
        self.X_train, self.y_train, self.X_test, self.y_test = self.split_data()
        self.model = None  # This will hold the trained model

    def extract_features(self):
        X = self.ratings.drop(columns=['Unnamed: 0', 'MovieID', 'UserID', 'Rating', 'Date', 'movie_graph_id']).values
        self.X = X  # Note the use of self.X to store the features matrix
        self.y = np.array(self.ratings['Rating'])  # Store the target variable

    def split_data(self, test_size=0.1):
        # Determine the number of samples for the test set
        num_test_samples = int(len(self.X) * test_size)

        # Split the data
        X_test, y_test = self.X[-num_test_samples:], self.y[-num_test_samples:]
        X_train, y_train = self.X[:-num_test_samples], self.y[:-num_test_samples]

        return X_train, y_train, X_test, y_test   
    
    def objective(self, trial):
        # Define the hyperparameter search space
        param = {
            'objective': self.params['objective'],
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'alpha': trial.suggest_float('alpha', 0, 100),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        }

        kf = KFold(n_splits=2, shuffle=True, random_state=888)
        rmse_scores = []

        for train_index, test_index in kf.split(self.X_train):
            X_train, X_test = self.X_train[train_index], self.X_train[test_index]
            y_train, y_test = self.y_train[train_index], self.y_train[test_index]

            model = xgb.XGBRegressor(**param)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            rmse_scores.append(rmse)

        

        # Store fold results in the trial's user attributes
        trial.set_user_attr("fold_rmse_scores", rmse_scores)

        return np.mean(rmse_scores)

    def tune_hyperparameters(self, n_trials=100):
        self.X_train, self.y_train, self.X_test, self.y_test = self.split_data()

        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)
        self.params.update(study.best_params)

        # Print the fold results for the best trial
        best_trial = study.best_trial
        print(f"Best trial: {best_trial.number}")
        print(f"Fold RMSE scores: {best_trial.user_attrs['fold_rmse_scores']}")
        print(f"Best parameters: {study.best_params}")

    def train_model(self):
        start_time = time.time()
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(self.X_train, self.y_train)
        end_time = time.time()
        print(f"Model trained. Time taken: {end_time - start_time:.2f} seconds")

        # Directly evaluate on training data after training
        self.evaluate_training_data()
        self.evaluate_test_data()


    def evaluate_training_data(self):
        """Evaluate the model on the training data and print metrics."""
        if not self.model:
            raise Exception("Model is not trained. Call train_model() first.")
        y_pred_train = self.model.predict(self.X_train)
        mse = mean_squared_error(self.y_train, y_pred_train)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_train, y_pred_train)
        print(f"Training MSE: {mse:.4f}, Training RMSE: {rmse:.4f}, Training MAE: {mae:.4f}")

    def evaluate_test_data(self):
        """Evaluate the model on the test data, print metrics, and display predicted vs actual."""
        y_pred_test = self.predict_test_data()
        mse = mean_squared_error(self.y_test, y_pred_test)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred_test)
        print(f"Test MSE: {mse:.4f}, Test RMSE: {rmse:.4f}, Test MAE: {mae:.4f}")


    def predict_test_data(self):
        if not self.model:
            raise Exception("Model is not trained. Call train_model() first.")
        return self.model.predict(self.X_test)

    def print_actual_vs_predicted(self):
          """Prints the actual vs. predicted ratings for the first few entries of the test set."""
          if not self.model:
              raise Exception("Model is not trained. Call train_model() first.")
          y_pred_test = self.predict_test_data()  # Get predictions
          print("Actual vs. Predicted Ratings:")
          for actual, predicted in zip(self.y_test[:10], y_pred_test[:10]):  # Adjust the slice as needed
              print(f"Actual: {actual}, Predicted: {predicted:.4f}")
