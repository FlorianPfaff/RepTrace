"""PyTorch Lightning models used by optional RepTrace classifiers."""

from __future__ import annotations

import pytorch_lightning as pl
import torch
from torch import nn, optim


class MLPClassifierTorch(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        learning_rate: float = 1e-3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, output_dim)
        self.learning_rate = learning_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.dropout(x)
        x = self.relu(self.layer_2(x))
        x = self.dropout(x)
        x = self.layer_3(x)
        return x

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            predictions = self.relu(self.layer_1(x))
            predictions = self.relu(self.layer_2(predictions))
            predictions = self.layer_3(predictions)
        return torch.argmax(predictions, dim=1).numpy()

    def training_step(self, batch, _batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
