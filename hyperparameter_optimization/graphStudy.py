#Ali Jaabous
#GraphStudy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import optuna

import models

class GraphStudy:
    def __init__(self, train_data_1, train_data_2, val_data_1, val_data_2, train_data_final, data, flag, EPOCHS, LR):
        self.train_data_1 = train_data_1
        self.train_data_2 = train_data_2
        self.val_data_1 = val_data_1
        self.val_data_2 = val_data_2
        self.data = data
        self.train_data_final = train_data_final
        self.flag = flag
        self.EPOCHS = EPOCHS
        self.LR = LR

    def objective(self, trial):
        hidden_channels_encoder = trial.suggest_int('hidden_channels_encoder', 32, 256)
        latent_space_dim = trial.suggest_int('latent_space_dim', 32, 256)
        hidden_channels_predictor = trial.suggest_int('hidden_channels_predictor', 32, 256)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.9)

        if self.flag == 1:
            model_1 = models.GCN(self.train_data_1, hidden_channels_encoder, latent_space_dim, hidden_channels_predictor, n_classes=1, dropout_rate=dropout_rate)
            model_2 = models.GCN(self.train_data_2, hidden_channels_encoder, latent_space_dim, hidden_channels_predictor, n_classes=1, dropout_rate=dropout_rate)
        elif self.flag == 2:
            model_1 = models.GraphSAGE(self.train_data_1, hidden_channels_encoder, latent_space_dim, hidden_channels_predictor, n_classes=1, dropout_rate=dropout_rate)
            model_2 = models.GraphSAGE(self.train_data_2, hidden_channels_encoder, latent_space_dim, hidden_channels_predictor, n_classes=1, dropout_rate=dropout_rate)
        elif self.flag == 3:
            model_1 = models.GAT(self.train_data_1, hidden_channels_encoder, latent_space_dim, hidden_channels_predictor, n_classes=1, dropout_rate=dropout_rate)
            model_2 = models.GAT(self.train_data_2, hidden_channels_encoder, latent_space_dim, hidden_channels_predictor, n_classes=1, dropout_rate=dropout_rate)

        model_1 = model_1.double()
        model_2 = model_2.double()

        optimizer_1 = optim.Adam(model_1.parameters(), lr=self.LR)
        optimizer_2 = optim.Adam(model_2.parameters(), lr=self.LR)
        criterion = nn.HuberLoss()

        avg_mae = 0

        for data, val_data, model, optimizer in [(self.train_data_1, self.val_data_1, model_1, optimizer_1), (self.train_data_2, self.val_data_2, model_2, optimizer_2)]:
            model.train()
            for epoch in range(self.EPOCHS):
                optimizer.zero_grad()
                output = model(data.x_dict, data.edge_index_dict, data['user', 'rating', 'movie'].edge_label_index, data)
                preds = output.squeeze()
                labels = data['user', 'rating', 'movie'].edge_label
                loss = criterion(preds.double(), labels)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                output = model(val_data.x_dict, val_data.edge_index_dict, val_data['user', 'rating', 'movie'].edge_label_index, val_data)
            preds = output.squeeze()
            labels = val_data['user', 'rating', 'movie'].edge_label
            mae = F.l1_loss(preds.double(), labels)
            avg_mae += mae

        avg_mae /= 2
        return avg_mae

    def run_study(self, n_trials=5):
        study = optuna.create_study(direction='minimize')
        study.optimize(self.objective, n_trials=n_trials)

        best_params = study.best_params

        hidden_channels_encoder = best_params['hidden_channels_encoder']
        latent_space_dim = best_params['latent_space_dim']
        hidden_channels_predictor = best_params['hidden_channels_predictor']
        dropout_rate = best_params['dropout_rate']

        if self.flag == 1:
            best_model = models.GCN(self.train_data_final, hidden_channels_encoder, latent_space_dim, hidden_channels_predictor, n_classes=1, dropout_rate=dropout_rate)
        elif self.flag == 2:
            best_model = models.GraphSAGE(self.train_data_final, hidden_channels_encoder, latent_space_dim, hidden_channels_predictor, n_classes=1, dropout_rate=dropout_rate)
        elif self.flag == 3:
            best_model = models.GAT(self.train_data_final, hidden_channels_encoder, latent_space_dim, hidden_channels_predictor, n_classes=1, dropout_rate=dropout_rate)

        best_model = best_model.double()

        optimizer = optim.Adam(best_model.parameters(), lr=self.LR)
        criterion = nn.HuberLoss()

        torch.manual_seed(888)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(888)

        best_model.train()
        for epoch in range(self.EPOCHS):
            optimizer.zero_grad()
            output = best_model(self.train_data_final.x_dict, self.train_data_final.edge_index_dict, self.train_data_final['user', 'rating', 'movie'].edge_label_index, self.data)
            preds = output.squeeze()
            labels = self.train_data_final['user', 'rating', 'movie'].edge_label
            loss = criterion(preds.double(), labels)
            loss.backward()
            optimizer.step()

        return study, best_model
