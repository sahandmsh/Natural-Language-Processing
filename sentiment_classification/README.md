# Text Classification
Text Classification is the process of categorizing text into different classes and is one of the fundamental problems in natural language processing (NLP). Text classification applications include sentiment analysis, spam detection, categorization, etc.

Implementing text classification requires several steps:
1) Preprocessing text data: Text dataset needs to be preprocessed first (e.g., tokenizing text, removing stop words, ...)
2) Feature extraction - word vectors: The next step is to convert the preprocessed text into numerical features that can be used to develop/train the text classifier model.
3) Classifier model selection and training: Choosing an appropriate classifier is crucial for achieving high accuracy. The model should generalize well to avoid underfitting or overfitting.
4) Implementation: The trained model is used in action to classify new text data (e.g., identifying the sentiment of tweets or reviews).

In this project, I'll provide implementations of different text classifiers.

# Tweet Sentiment Classification Overview:
1) **Pretrained Word Embeddings:** Pretrained word embeddings (GloVe) are used to process the tweets and extract meaningful features
   for sentiment analysis.
2) **Data Splitting:** The dataset is split into train, validation, and test sets to evaluate the model’s performance effectively.
3) **1D Convolution Layer:** A 1D convolution layer is employed to capture dependencies between words within each tweet, allowing
   the model to learn important contextual information.
4) **Loss Function:** Cross-entropy loss is used as the objective function to measure the model’s performance during training.
5) **Sliding Window Average for Validation Loss:** A sliding window average method monitors the validation loss at every epoch
   and halts the training if the validation loss stops improving, thus preventing overfitting.
