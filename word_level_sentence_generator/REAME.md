# RNN-based Word Level Sentence Generator
TokenLevelRNN is a PyTorch-based token-level language model that generates text word-by-word using a multi-layer vanilla RNN. It learns to predict the next token given a sequence of previous tokens, helping it create coherent sentences over time.
The model includes:
* An embedding layer to represent tokens in a continuous space
* A multi-layer RNN with dropout for better generalization
* A linear layer to predict the next token from hidden states
* It supports training with randomly sampled sequences (batches) and flexible text generation starting from any initial phrase. The generation uses softmax sampling with adjustable temperature to control creativity. Utilities for encoding/decoding and sentence formatting are also built-in.

The implementation is designed for easy experimentation with sequence length, batch size, model depth, and sampling strategies!
