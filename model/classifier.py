import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int, hidden_dims: list = None, dropout_prob: float = 0.0):
        super().__init__()

        layers = []
        current_dim = embedding_dim

        if hidden_dims:
            for h_dim in hidden_dims:
                layers.append(nn.Linear(current_dim, h_dim))
                layers.append(nn.ReLU())
                if dropout_prob > 0:
                    layers.append(nn.Dropout(dropout_prob))
                current_dim = h_dim

        layers.append(nn.Linear(current_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)