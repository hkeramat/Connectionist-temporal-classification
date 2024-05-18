import numpy as np
import os

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


# Reproducibility check
np.random.seed(785)


from tensorflow.keras.datasets import mnist

# Loading MNIST data and creating a handwritten digit sequence 
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

def create_sequences(images, labels, num_sequences=20000, sequence_length=5):
    # Initialize arrays to hold sequences and sequence labels
    height, width = images[0].shape
    total_width = width * sequence_length
    all_sequences = np.zeros((num_sequences, height, total_width), dtype=np.uint8)
    all_labels = np.zeros((num_sequences, sequence_length), dtype=int)

    for i in range(num_sequences):
        idx = np.random.choice(len(images), sequence_length)
        sequence_images = images[idx]
        sequence_labels = labels[idx]

        # Creating one sequence
        new_sequence = np.hstack(sequence_images)
        
        # Storing the sequence and the labels
        all_sequences[i] = new_sequence
        all_labels[i] = sequence_labels

    return all_sequences, all_labels

# Generating 20,000 sequence data 
sequences, sequence_labels = create_sequences(train_images, train_labels, 20000, 5)

# Visualization 
plt.imshow(sequences[0], cmap='gray')
plt.title('Labels: {}'.format(sequence_labels[0]))
plt.show()
print("Sequences shape:", sequences.shape)
print("sequence labels shape:", sequence_labels.shape)
