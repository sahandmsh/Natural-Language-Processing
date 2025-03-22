{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "655abc9e-cc1b-4035-9587-5ab049cea880",
   "metadata": {},
   "source": [
    "Import libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 276,
   "id": "c0483182-a2ac-4df8-9cff-28fa76848ffc",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "from sklearn.model_selection import train_test_split\n",
    "from torch.nn.utils.rnn import pad_sequence\n",
    "from torchmetrics.classification import MulticlassAccuracy\n",
    "from torchtext.data.utils import get_tokenizer\n",
    "from torchtext.vocab import GloVe"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c5796885-3a9b-4e76-bde0-1c7ec9619f46",
   "metadata": {},
   "source": [
    "Create tokenizer object (to convert tweets to tokens). Get the pre-trained word embeddings from GloVe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 264,
   "id": "48e0d690-cdc3-44d2-b93b-89a21dd12a76",
   "metadata": {},
   "outputs": [],
   "source": [
    "tokenizer = get_tokenizer('basic_english')\n",
    "glove = GloVe(name='6B', dim=100)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f010fcaa-b103-42fe-91d8-c619679bff36",
   "metadata": {},
   "source": [
    "Function to clean out tweets dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 265,
   "id": "d32f90e3-81cf-4a59-9f06-33a6f7b8742f",
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_out_data(original_df, desired_columns_list):\n",
    "    \"\"\" Reduces data columns to desired column list\n",
    "        Modifies polarity values (negative sentiment: 0, positive sentiment: 1)\n",
    "    Args:\n",
    "        original_df: pandas dataframe\n",
    "        desired_columns_list: list containing the desired columns to keep\n",
    "    \"\"\"\n",
    "    df = original_df.copy()\n",
    "    df = df[desired_columns_list]\n",
    "    df.loc[:, 'polarity'] = df['polarity'].replace({0: 0, 4: 1})\n",
    "    return df"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "387bd95b-f9af-4d1f-9b7d-8e9eb8990983",
   "metadata": {},
   "source": [
    "Load the dataset. Create train (70%), validation (15%), and test (15%) datasets\n",
    "\n",
    "Dataset is a part of Stanford Sentiment140 dataset (http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 266,
   "id": "67c6557d",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# Load the dataset and split into train validation test\n",
    "dataset_df = pd.read_csv(\"trainingandtestdata/training.1600000.processed.noemoticon.csv\", encoding='latin1', header=None)\n",
    "dataset_df = dataset_df.sample(n=20000)\n",
    "dataset_df.columns = ['polarity', 'id', 'date', 'query', 'user', 'tweet']\n",
    "\n",
    "train_df, validation_test_df = train_test_split(dataset_df, test_size=0.3)\n",
    "validation_df, test_df = train_test_split(validation_test_df, test_size=0.5)\n",
    "\n",
    "train_df = clean_out_data(train_df, ['tweet', 'polarity'])\n",
    "validation_df = clean_out_data(validation_df, ['tweet', 'polarity'])\n",
    "test_df = clean_out_data(test_df, ['tweet', 'polarity'])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a414cb76-af47-44a7-933c-2a874dd9e65a",
   "metadata": {},
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 267,
   "id": "50c3e57c-9899-4414-a2fa-327723ad85d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "def process_tweets(df):\n",
    "    \"\"\" Tokenizes tweets then stack them as tensors\n",
    "    Args:\n",
    "        df: pandas dataframe\n",
    "    Returns:\n",
    "        tuple (list, tensor): a list of stacked word embeddings, corresponding labels tensor\n",
    "    \"\"\"\n",
    "    tweets_list, labels_list = [], []\n",
    "    for idx, row in df.iterrows():\n",
    "        tweet, label = row['tweet'], row['polarity']\n",
    "        tokens = tokenizer(tweet)\n",
    "        tweet_embeddings = [glove[word] for word in tokens if word in glove.stoi]\n",
    "        if tweet_embeddings and label in [0, 1]:\n",
    "            tweets_list.append(torch.stack(tweet_embeddings))\n",
    "            labels_list.append(label)\n",
    "    return tweets_list, torch.tensor(labels_list, dtype=torch.int64)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5c59670f-91d0-4c44-8b6c-3c7e4b17a271",
   "metadata": {},
   "source": [
    "Process tweets by Tokenizing them each tweet then stacking them as tensors\n",
    "\n",
    "Pad the processed tweets and adding zeros so that all tensors in each data group (train, test, validation) are of the same size."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 268,
   "id": "c867ca1e-fe99-438a-9699-d859f051bca7",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_data, train_data_labels = process_tweets(train_df)\n",
    "validation_data, validation_data_labels = process_tweets(validation_df)\n",
    "test_data, test_data_labels = process_tweets(test_df)\n",
    "\n",
    "# Padding train, test, validation datasets by adding zeros\n",
    "train_data = pad_sequence(train_data, batch_first = True, padding_value= 0)\n",
    "validation_data = pad_sequence(validation_data, batch_first = True, padding_value= 0)\n",
    "test_data = pad_sequence(test_data, batch_first = True, padding_value= 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 269,
   "id": "0753c9c4-62fd-46dc-b206-80e88c144f61",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Tweet_SentimentClassifier(nn.Module):\n",
    "    \"\"\" Class with the 1D convolutional NN structure, and includes train and predict methods\n",
    "    \"\"\"\n",
    "    def __init__(self, input_size, hidden_size, output_size, kernel_size=5):\n",
    "        nn.Module.__init__(self)\n",
    "        self.conv = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=kernel_size, padding=kernel_size//2)\n",
    "        self.linear = nn.Linear(input_size, output_size)\n",
    "        self.dropout = nn.Dropout(0.5)\n",
    "        self.relu = nn.ReLU()  # ReLU activation function\n",
    "\n",
    "    \n",
    "    def forward(self, input):\n",
    "        \"\"\" NN forward pass\n",
    "        Args:\n",
    "            input: tensor of shape (batch_size, input_size)\n",
    "        Returns:\n",
    "            output: tensor of shape (batch_size, output_size)\n",
    "        \"\"\"\n",
    "        input = input.permute(0, 2, 1)  # (batch_size, embedding_size, seq_len)\n",
    "        conv_out = self.conv(input)\n",
    "        conv_out = self.relu(conv_out)\n",
    "        pooled_out, _ = torch.max(conv_out, dim=2)  # max pooling\n",
    "        pooled_out = self.dropout(pooled_out) # dropout to avoid overfitting\n",
    "        output = self.linear(pooled_out)\n",
    "        return output\n",
    "\n",
    "\n",
    "    def _moving_average(self,losses, window):\n",
    "        \"\"\" Compute a simple moving average over the loss values \"\"\"\n",
    "        return np.convolve(losses, np.ones(window)/window, mode='valid')\n",
    "\n",
    "        \n",
    "    def train_model(self, train_input_tensor, train_labels,validation_input_tensor, validation_labels, loss_function, optimizer, epochs, validation_loss_smoothing_window, epochs_without_improvement_threshold):\n",
    "        \"\"\" Trains the model\n",
    "        Args:\n",
    "            input_tensors: input tensor of shape (batch_size, input_size)\n",
    "            labels: label tensor of shape (batch_size, output_size)\n",
    "            validation_input_tensor: validation data tensor of shape (batch_size, input_size)\n",
    "            validation_labels: validation label tensor of shape (batch_size, output_size)\n",
    "            loss_function: loss function (e.g., cross entropy loss)\n",
    "            optimizer: the optimizer (e.g., Adam)\n",
    "            epochs: number of epochs\n",
    "            validation_loss_smoothing_window: validation loss smoothing window\n",
    "            epochs_without_improvement_threshold: Maximum allowed consequent epochs without validation loss improvement\n",
    "        Returns:\n",
    "            None\n",
    "        \"\"\"\n",
    "        # Line objects for train and validation loss\n",
    "        train_line, = ax.plot([], [], label='Train Loss', color='b')\n",
    "        val_line, = ax.plot([], [], label='Validation Loss', color='r')\n",
    "        ax.legend()\n",
    "\n",
    "        \n",
    "        best_validation_loss = float('inf')\n",
    "        epochs_without_improvement = 0\n",
    "        train_loss_list = []\n",
    "        validation_loss_list = []\n",
    "        for epoch in range(epochs):\n",
    "            self.train()\n",
    "            output = self(train_input_tensor)\n",
    "            loss = loss_function(output, train_labels)\n",
    "            train_loss_list.append(loss.item())\n",
    "            optimizer.zero_grad()\n",
    "            loss.backward()\n",
    "            optimizer.step()\n",
    "\n",
    "            _, validation_loss = self.predict(validation_input_tensor, validation_labels)\n",
    "            validation_loss_list.append(validation_loss.item())\n",
    "\n",
    "            # Monitoring smoothed validation loss and potentially stop the training to avoid overfitting\n",
    "            smoothed_validation_loss = self._moving_average(validation_loss_list, validation_loss_smoothing_window)\n",
    "            if epoch<validation_loss_smoothing_window:\n",
    "                pass\n",
    "            elif smoothed_validation_loss[-1] < best_validation_loss:\n",
    "                best_validation_loss = smoothed_validation_loss[-1]\n",
    "                epochs_without_improvement = 0  # Reset the counter when validation loss improves\n",
    "            else:\n",
    "                epochs_without_improvement += 1\n",
    "            \n",
    "            # Stop if smoothed validation loss starts increasing while training loss is still decreasing\n",
    "            if epochs_without_improvement > epochs_without_improvement_threshold:\n",
    "                print(f\"Stopping at epoch {epoch + 1} due to no improvement in smoothed validation loss.\")\n",
    "                break\n",
    "\n",
    "            if (epoch+1)%10 == 0:\n",
    "                print(f\"Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item():.6f}, Validation Loss: {validation_loss.item():.6f}\")\n",
    "        \n",
    "        # Plot the train and validation loss\n",
    "        plt.plot(range(1, len(train_loss_list) + 1), train_loss_list, color='red', label=\"train loss\")\n",
    "        plt.plot(range(1, len(validation_loss_list) + 1), validation_loss_list, color='blue', label=\"validation loss\")\n",
    "        plt.xlabel(\"epoch\")\n",
    "        plt.ylabel(\"loss\")\n",
    "        plt.legend()\n",
    "        plt.show()\n",
    "\n",
    "    \n",
    "    def predict(self, input_tensor, actual_labels):\n",
    "        \"\"\" Predicts labels\n",
    "        Args:\n",
    "            input_tensor: input tensor of shape (batch_size, input_size)\n",
    "        Returns:\n",
    "            tuple(predicted_labels, loss): predicted labels of shape (batch_size, output_size), loss\n",
    "        \"\"\"\n",
    "        self.eval()\n",
    "        with torch.no_grad():\n",
    "            predictions = self(input_tensor).squeeze()\n",
    "            loss = loss_function(predictions, actual_labels)\n",
    "            predicted_labels = torch.argmax(predictions, dim = 1)\n",
    "        return predicted_labels, loss"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d1e78502-78ca-447f-bec8-cb79aea52d4e",
   "metadata": {},
   "source": [
    "Define the tweet classifier object, loss function, and optimizer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 273,
   "id": "7db3c860-e658-4621-a998-3063050ff12f",
   "metadata": {},
   "outputs": [],
   "source": [
    "tweet_classifier = Tweet_SentimentClassifier(input_size = 100, hidden_size = 100, output_size = 2)\n",
    "loss_function = nn.CrossEntropyLoss()\n",
    "optimizer = optim.Adam(tweet_classifier.parameters(), lr = 0.001, weight_decay = 0.001)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0e282790-ca46-4699-95f9-6f817454d8c4",
   "metadata": {},
   "source": [
    "Train the model and track the train loss and validation loss to avoid overfitting"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 274,
   "id": "aceb5a64-b34c-43f9-89c2-67aaa5a03b53",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [10/500], Training Loss: 0.666011, Validation Loss: 0.655122\n",
      "Epoch [20/500], Training Loss: 0.620021, Validation Loss: 0.621124\n",
      "Epoch [30/500], Training Loss: 0.590449, Validation Loss: 0.591847\n",
      "Epoch [40/500], Training Loss: 0.559723, Validation Loss: 0.568076\n",
      "Epoch [50/500], Training Loss: 0.535091, Validation Loss: 0.550279\n",
      "Epoch [60/500], Training Loss: 0.514181, Validation Loss: 0.537924\n",
      "Epoch [70/500], Training Loss: 0.493877, Validation Loss: 0.528112\n",
      "Epoch [80/500], Training Loss: 0.477881, Validation Loss: 0.521075\n",
      "Epoch [90/500], Training Loss: 0.462070, Validation Loss: 0.515595\n",
      "Epoch [100/500], Training Loss: 0.447814, Validation Loss: 0.511545\n",
      "Epoch [110/500], Training Loss: 0.434128, Validation Loss: 0.508603\n",
      "Epoch [120/500], Training Loss: 0.416183, Validation Loss: 0.505922\n",
      "Epoch [130/500], Training Loss: 0.400803, Validation Loss: 0.505074\n",
      "Epoch [140/500], Training Loss: 0.390168, Validation Loss: 0.503191\n",
      "Epoch [150/500], Training Loss: 0.379966, Validation Loss: 0.503596\n",
      "Epoch [160/500], Training Loss: 0.366351, Validation Loss: 0.503184\n",
      "Stopping at epoch 163 due to no improvement in smoothed validation loss.\n"
     ]
    },
    {
     "data": {
      "image/png": "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+02uEYsoxpQ7p4iv6W/L6v353Ejn57v8+dxI5+e7/PncnD4/jvycO3fOfI97fR2gQODPdYX8+dxI5+e7/PncSOfnu/z53Hzp/LyiFYaIiIhIfFIAJCIiIgFHAVA8YO5Rv379/HKpvT+fG+n8fJc/nxvp/HyXP5+bL52fcoBEREQk4GgESERERAKOAiAREREJOAqAREREJOAoABIREZGAowAojo0aNQo5c+ZEkiRJUL58eaxZswa+iM1my5Yta6pjZ8iQwfRu2759e4R9Ll++jA4dOiBdunRInjw5GjVqdFOjW1/w3nvvmQrgr732mt+c28GDB/Hcc8+Z42cF9aJFi2Lt2rVhj3MtBKuxs18eH69RowZ27twJX3D9+nX06dMHuXLlMseeJ08eU/ndc32HL53fb7/9hnr16plKtvw5nD59eoTHo3MuJ0+eRLNmzUwRutSpU6NNmzY4f/48vPncrl27hh49epifzWTJkpl9WrRogX///dcnzi06/+88vfTSS2afYcOG+cT5/RaNc2Nj8vr165siiPx/yO8MdnPw1s9RBUBxaPLkyejSpYtZDrh+/XoUL14ctWrVwtGjR+FrlixZYn5wV61ahQULFpgPK7YYYWsSt86dO2PmzJn4/vvvzf784GrYsCF8ye+//45PP/0UxYoVi3C/L5/bqVOnUKlSJSRKlAhz5szBli1b8OGHHyJNmjRh+wwePBjDhw/HmDFjsHr1avPhxZ9VfmB5u/fffx+jR4/GyJEjzQcwb/N8RowY4ZPnx98pflbwj6eoROdc+AX6119/md/VWbNmmS+vdu3awZvP7eLFi+ZzksEsL6dOnWr+yOIXqidvPbfo/L9zmzZtmvksjapdg7ee34U7nNvu3btRuXJl055q8eLFpiUV/1/yj3+v/Ry9xx6mEg3lypVzdejQIez29evXXVmyZHENGjTI5euOHj1qGs4tWbLE3D59+rQrUaJEru+//z5sn61bt5p92JjWF7CRLpvmLliwwFW1alVXp06d/OLcevTo4apcufItHw8NDXVlypTJNWTIkLD7eM5sUvzdd9+5vN3jjz/uat26dYT7GjZs6GrWrJnPnx9/xqZNmxZ2OzrnwibRfN7vv/8ets+cOXNcQUFBroMHD7q89dyismbNGrPf3r17fercbnd+Bw4ccGXNmtW1efNmV44cOVwfffRR2GO+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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "tweet_classifier.train_model(train_data, train_data_labels, validation_data, validation_data_labels, loss_function, optimizer, 500, 10, 10)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fe90f184-8f76-4134-bba9-9b4653991437",
   "metadata": {},
   "source": [
    "Evaluate the trained model using test data by calculating accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 280,
   "id": "e1da55e7-dee9-4983-90aa-e3c14bcccf12",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Evaluating test data:\n",
      "Accuracy: 0.7519570589065552\n",
      "Checking model accuracy on train data (overfitting avoided by early stop):\n",
      "Accuracy: 0.8830654621124268\n"
     ]
    }
   ],
   "source": [
    "calculate_accuracy = MulticlassAccuracy(num_classes = 2)\n",
    "print(\"Evaluating test data:\")\n",
    "predicted_test_labels,_ = tweet_classifier.predict(test_data, test_data_labels)\n",
    "print(f\"Accuracy: {calculate_accuracy(predicted_test_labels, test_data_labels)}\")\n",
    "\n",
    "print(\"Checking model accuracy on train data (overfitting avoided by early stop):\")\n",
    "predicted_train_labels,_ = tweet_classifier.predict(train_data, train_data_labels)\n",
    "print(f\"Accuracy: {calculate_accuracy(predicted_train_labels, train_data_labels)}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
