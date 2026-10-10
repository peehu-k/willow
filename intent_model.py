"""IntentNet: the deep neural network that classifies sentence embeddings into intents.

Defined in PyTorch for training (train.py). The trained weights are exported to model.npz and
run in NumPy at serving time (chatbot.py), which gives identical predictions without PyTorch.
"""
import torch.nn as nn


class IntentNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)
