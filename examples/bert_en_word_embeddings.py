import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from sklearn.metrics.pairwise import cosine_similarity

#pip install pytorch-pretrained-bert
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#1 Sentence Input:
#text = "Here is the sentence I want embeddings for."
text = "After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank."
marked_text = "[CLS] " + text + " [SEP]"
print (marked_text)

#We’ve imported a BERT-specific tokenizer, let’s take a look at the output:
tokenized_text = tokenizer.tokenize(marked_text)
print (tokenized_text)

list(tokenizer.vocab.keys())[5000:5020]

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

for tup in zip(tokenized_text, indexed_tokens):
  print (tup)

segments_ids = [1] * len(tokenized_text)
print (segments_ids)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased')

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

# Predict hidden states features for each layer
with torch.no_grad():
    encoded_layers, _ = model(tokens_tensor, segments_tensors)

print ("Number of layers:", len(encoded_layers))
layer_i = 0

print ("Number of batches:", len(encoded_layers[layer_i]))
batch_i = 0

print ("Number of tokens:", len(encoded_layers[layer_i][batch_i]))
token_i = 0

print ("Number of hidden units:", len(encoded_layers[layer_i][batch_i][token_i]))


# For the 5th token in our sentence, select its feature values from layer 5.
token_i = 5
layer_i = 5
vec = encoded_layers[layer_i][batch_i][token_i]

# Plot the values as a histogram to show their distribution.
plt.figure(figsize=(10,10))
plt.hist(vec, bins=200)
plt.show()

# Convert the hidden state embeddings into single token vectors

# Holds the list of 12 layer embeddings for each token
# Will have the shape: [# tokens, # layers, # features]
token_embeddings = []

# For each token in the sentence...
for token_i in range(len(tokenized_text)):

    # Holds 12 layers of hidden states for each token
    hidden_layers = []

    # For each of the 12 layers...
    for layer_i in range(len(encoded_layers)):
        # Lookup the vector for `token_i` in `layer_i`
        vec = encoded_layers[layer_i][batch_i][token_i]

        hidden_layers.append(vec)

    token_embeddings.append(hidden_layers)

print('------------------------------------------------------------')

# Sanity check the dimensions:
print("Number of tokens in sequence:", len(token_embeddings))
print("Number of layers per token:", len(token_embeddings[0]))

concatenated_last_4_layers = [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0) for layer in token_embeddings] # [number_of_tokens, 3072]

summed_last_4_layers = [torch.sum(torch.stack(layer)[-4:], 0) for layer in token_embeddings] # [number_of_tokens, 768]

sentence_embedding = torch.mean(encoded_layers[11], 1)

print ("Our final sentence embedding vector of shape:"), sentence_embedding[0].shape[0]

print (text)
for i,x in enumerate(tokenized_text):
  print (i,x)

print ("First fifteen values of 'bank' as in 'bank robber':")
print (summed_last_4_layers[10][:15])

print ("First fifteen values of 'bank' as in 'bank vault':")
print(summed_last_4_layers[6][:15])

print ("First fifteen values of 'bank' as in 'river bank':")
print(summed_last_4_layers[19][:15])

# Compare "bank" as in "bank robber" to "bank" as in "river bank"
different_bank = cosine_similarity(summed_last_4_layers[10].reshape(1,-1), summed_last_4_layers[19].reshape(1,-1))[0][0]

# Compare "bank" as in "bank robber" to "bank" as in "bank vault"
same_bank = cosine_similarity(summed_last_4_layers[10].reshape(1,-1), summed_last_4_layers[6].reshape(1,-1))[0][0]

print ("Similarity of 'bank' as in 'bank robber' to 'bank' as in 'bank vault':",  same_bank)

print ("Similarity of 'bank' as in 'bank robber' to 'bank' as in 'river bank':",  different_bank)




