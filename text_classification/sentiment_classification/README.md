# Tweet Sentiment Classification Overview:
1) **Pretrained Word Embeddings:** Pretrained word embeddings (GloVe) are used to process the tweets and extract meaningful features
   for sentiment analysis.
2) **Data Splitting:** The dataset is split into train, validation, and test sets to evaluate the model’s performance effectively.
3) **1D Convolution Layer:** A 1D convolution layer is employed to capture dependencies between words within each tweet, allowing
   the model to learn important contextual information.
4) **Loss Function:** Cross-entropy loss is used as the objective function to measure the model’s performance during training.
5) **Sliding Window Average for Validation Loss:** A sliding window average method monitors the validation loss at every epoch
   and halts the training if the validation loss stops improving, thus preventing overfitting.
