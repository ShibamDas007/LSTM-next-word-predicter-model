from nltk.tokenize import word_tokenize

# Read data
with open(r'E:\myvenv\data\data_3.txt', 'r', encoding='utf-8') as file:
    data = file.read()

# Tokenize words
tokenized_words = [word_tokenize(line) for line in data.split('\n')]

# Build vocabulary
word_to_idx = {}
for tokens in tokenized_words:
    for word in tokens:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

# Generate input sequences
input_sequences = []
for sentence in data.split('\n'):
    tokens = word_tokenize(sentence)
    sequence = [word_to_idx[word] for word in tokens]
    for i in range(1, len(sequence) + 1):
        input_sequences.append(sequence[:i])
