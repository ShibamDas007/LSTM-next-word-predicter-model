import torch
import torch.nn as nn
from nltk.tokenize import word_tokenize
import nltk
from torch.nn.utils.rnn import pad_sequence
import sys
import time

# Define the LSTM architecture
class LSTMNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super(LSTMNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        embedded = self.embedding(x)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(embedded, (h0, c0))
        out = self.linear(out)
        return out

# Read data
with open(r'E:\myvenv\data\data_3.txt', 'r', encoding='utf-8') as file:
    data = file.read()

# Tokenize words
tokenized_words = [word_tokenize(line) for line in data.split('\n')]

# Build vocabulary
word_to_index = {}
for tokens in tokenized_words:
    for word in tokens:
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)
# Define idx_to_word mapping
idx_to_word = {idx: word for word, idx in word_to_index.items()}

# Load the trained model
model = LSTMNN(input_size=len(word_to_index), hidden_size=248, num_layers=2, output_size=len(word_to_index), batch_size=32)
model.load_state_dict(torch.load(r'E:\myvenv\lstm_model.pth', map_location='cpu'))
model.eval()

# Function to generate a full sentence given a prompt
def generate_full_sentence(prompt, model, word_to_idx, idx_to_word, max_length=50):
    tokenized_prompt = word_tokenize(prompt)
    sequence = [word_to_idx.get(word, 0) for word in tokenized_prompt]  # Convert words to indices
    input_tensor = torch.tensor(sequence).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        generated_sentence = tokenized_prompt  # Initialize generated sentence with prompt
        for _ in range(max_length):
            output = model(input_tensor)
            _, predicted = torch.max(output[:, -1, :], 1)  # Take the last output and get the predicted word
            next_word_idx = predicted.item()
            if next_word_idx == 0:  # End of sequence token
                break
            generated_sentence.append(idx_to_word[next_word_idx])  # Append next word to generated sentence
            input_tensor = torch.cat((input_tensor, torch.tensor([[next_word_idx]])), dim=1)  # Append next word index
    generated_text = ' '.join(generated_sentence)
    return generated_text

def animate_text(text, delay=0.1):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()  # Add a newline at the end

# Inference
prompt = "To Sherlock Holmes"
generated_text = generate_full_sentence(prompt, model, word_to_index, idx_to_word)
print(f"{generated_text}.")
animate_text(generated_text)