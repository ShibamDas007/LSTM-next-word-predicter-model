# LSTM-next-word-predicter-model
---

# LSTM Text Generation

This repository contains code for training and using an LSTM neural network for text generation. The trained model can generate coherent sentences given a prompt.

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.x
- PyTorch
- NLTK

You can install the required Python packages using pip:

```
pip install torch nltk
```

### Installation

1. Clone the repository:

```
git clone https://github.com/ShibamDas007/LSTM-next-word-predicter-model.git
```

2. Navigate to the project directory:

```
cd lstm-text-generation
```

3. Download NLTK data:

```
python -m nltk.downloader punkt
```

## Usage

### Training

To train the LSTM model, prepare your training data in a text file and run:

```
python train.py --data_path /path/to/training/data.txt
```

### Inference

To generate text using the trained model, specify the path to the saved model and provide a prompt:

```
python generate.py --model_path /path/to/saved/model.pth --prompt "Your prompt text here"
```

## File Structure

- `lstm_torch_train.py`: Script for training the LSTM model.
- `run.py`: Script for generating text using the trained model.
- `lstm_architecture.py`: Definition of the LSTM neural network architecture.
- `utils.py`: Utility functions for data preprocessing and text generation.
- `data/`: Directory for storing training data.
- `models/`: Directory for saving trained models.

## Model Architecture

The LSTM model architecture consists of an embedding layer, followed by one or more LSTM layers, and a linear layer for output. The model is defined in lstm_model.py with configurable parameters such as input size, hidden size, number of layers, and batch size.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to add more details or sections as needed, such as troubleshooting tips, additional usage examples, or links to external resources.
