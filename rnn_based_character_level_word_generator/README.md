# RNN-based Character Level Word Generator
CharecterLevelRNN is a simple implementation of a character‐level language model using an RNN. Given a vocabulary (including \<s\> and \</s\> tokens), the model follows below workflow:

The developed model converts each input character into a vector, feeds these vectors through an RNN to capture context, and then predicts the next character. The model is trained by comparing its predicted next characters to the actual next characters.

### Key functionalities:

* encode/decode methods are resposible for converting char to its index in the vocabulary/vocabulary index to char.
* train_model method trains the model based on a dataset of words, where each word is wrapped with start (<s>) and end (</s>) tokens.
* generate_word generates one character at a time given a word prefix. This method recognizes generated character sequences' start and end by looking for \<s\> and \</s\> tokens.
