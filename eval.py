
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


from train import model

ctc_loss = nn.CTCLoss()

# Reproducibility check
torch.manual_seed(785)
torch.cuda.manual_seed(785)

# Generating test data 
test_sequences, test_sequence_labels = create_sequences(test_images, test_labels, num_sequences=20, sequence_length=5)
test_sequences_tensor = torch.tensor(test_sequences, dtype=torch.float32).permute(0, 2, 1)

# Evaluation
model.eval()

for i in range(len(test_sequences)):
    sequence = test_sequences_tensor[i].unsqueeze(0)
    
    # Forward pass
    with torch.no_grad():
        logits = model(sequence)
    
    # Decoding the CTC output
    output = logits.permute(1, 0, 2).log_softmax(2).argmax(2).squeeze(0)
    
    # Converting output to a list of integers
    output_list = output.tolist()
    
    # CTC decoding
    predicted_sequence = []
    previous_digit = -1
    for digit in output_list:
        if digit != previous_digit:
            if digit != [10]:  # blank label index
                predicted_sequence.append(str(digit[0]))
            previous_digit = digit
    
    predicted_sequence = ''.join(predicted_sequence)
    
    # Converting the ground truth label to a string
    ground_truth_sequence = ''.join(str(digit) for digit in test_sequence_labels[i])
    
    # Display the input image
    plt.imshow(test_sequences[i], cmap='gray')
    plt.axis('off')
    plt.title(f"Predicted: {predicted_sequence}\nGround Truth: {ground_truth_sequence}")
    plt.show()