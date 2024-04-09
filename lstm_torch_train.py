import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
import nltk
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from lstm_architecture import LSTMNN 
from utils import input_sequences, word_to_idx

nltk.download('punkt')

# Convert input_sequences to PyTorch tensors
input_seq_tensor = [torch.tensor(seq) for seq in input_sequences]

# Pad the input sequences
padded_seq = pad_sequence(input_seq_tensor, batch_first=True)

# Define the model parameters
vocab_size = len(word_to_idx)
input_size = vocab_size
hidden_size = 248
output_size = vocab_size
num_layers = 2
batch_size = 32  # Adjusted batch size
learning_rate = 0.001
num_epochs = 100

# Initialize the model
model = LSTMNN(input_size, hidden_size, num_layers, output_size, batch_size)

# Create DataLoader with Inputs and Targets
dataset = TensorDataset(padded_seq[:, :-1], padded_seq[:, 1:])
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

from tqdm import tqdm

# Train the model
# Train the model
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    total_batches = 0
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
    for inputs, targets in progress_bar:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.transpose(1, 2), targets)  # Transpose outputs for CrossEntropyLoss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        total_batches += inputs.size(0)
        progress_bar.set_postfix({'loss': total_loss / total_batches})
    print(f'Train Loss: {total_loss / total_batches:.4f}', flush=True)

    print('Epoch [{}/{}], Train Loss: {:.4f}'.format(epoch+1, num_epochs, total_loss / total_batches), flush=True)

    # Validation
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total_batches = 0
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs.transpose(1, 2), targets)
            total_loss += loss.item() * inputs.size(0)
            total_batches += inputs.size(0)
    print('Epoch [{}/{}], Validation Loss: {:.4f}'.format(epoch+1, num_epochs, total_loss / total_batches), flush=True)

    # Save the model after each epoch
    torch.save(model.state_dict(), f'E:\\myvenv\\lstm_model\\lstm_model_epoch_{epoch+1}.pth')
# Save the model
torch.save(model.state_dict(), r'E:\myvenv\lstm_modellstm_model.pth')

