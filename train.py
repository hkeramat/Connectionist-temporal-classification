import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from data import sequences, sequence_labels

ctc_loss = nn.CTCLoss()

# Reproducibility check
torch.manual_seed(785)
torch.cuda.manual_seed(785)


# Defining the CTC model with BLSTM

class CTCModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CTCModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, proj_size=0)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

        # Initializing peephole connections and forget gate biases (doesn't help the model that much)
        for layer in range(num_layers):
            for direction in range(2):  
                layer_params = getattr(self.lstm, f'bias_ih_l{layer}')
                if direction == 1:
                    layer_params = getattr(self.lstm, f'bias_ih_l{layer}_reverse')
                
                layer_params.data[hidden_size:2*hidden_size].fill_(1.0)
                peephole_weights = torch.zeros(3 * hidden_size)
                setattr(self.lstm, f'peephole_ih_l{layer}_{direction}', nn.Parameter(peephole_weights))
                setattr(self.lstm, f'peephole_hh_l{layer}_{direction}', nn.Parameter(peephole_weights))

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

# Converting sequences and sequence_labels to PyTorch tensors
sequences_tensor = torch.tensor(sequences, dtype=torch.float32).permute(0, 2, 1)
sequence_labels_tensor = torch.tensor(sequence_labels, dtype=torch.long)

# Creating a dataset and data loader
dataset = TensorDataset(sequences_tensor, sequence_labels_tensor)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model parameters
input_size = 28  # Digit sequence dimensions
hidden_size = 128 # Had to change these compared to the paper
num_layers = 2
num_classes = 10  # Number of digits (0-9) + blank label

# Instantiating training the model
model = CTCModel(input_size, hidden_size, num_layers, num_classes + 1)  # +1 for blank label
criterion = nn.CTCLoss(blank=num_classes, reduction='mean', zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 30
for epoch in range(num_epochs):
    for sequences, labels in dataloader:
        optimizer.zero_grad()

        # Forward pass
        logits = model(sequences)

        input_lengths = torch.full((batch_size,), logits.size(1), dtype=torch.long)
        label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)

        logits = logits.log_softmax(2).permute(1, 0, 2)
        ctc_loss = criterion(logits, labels, input_lengths, label_lengths)

        # Backward pass and optimization
        ctc_loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {ctc_loss.item():.4f}")

print("Training completed.")