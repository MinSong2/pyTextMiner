
from transformers import BertTokenizer, BertModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

#import pytorch_pretrained_bert as ppb
import torch

#bert multi-lingual model
#https://github.com/google-research/bert/blob/master/multilingual.md

def get_pretrained_model(pretrained_type):
    if pretrained_type == 'etri':
        # use etri tokenizer
        from py_bert.tokenization_korbert import BertTokenizer
        tokenizer_path = 'D:\\python_workspace\\pyTextMiner\\bert_models\\vocab_mecab.list'
        tokenizer = BertTokenizer.from_pretrained(
            tokenizer_path, do_lower_case=False)
        vocab = tokenizer.vocab
    elif pretrained_type == 'skt':
        # use gluonnlp tokenizer
        import gluonnlp as nlp
        vocab_path = './pretrained_model/skt/vocab.json'
        tokenizer_path = './pretrained_model/skt/tokenizer.model'
        vocab = nlp.vocab.BERTVocab.from_json(open(vocab_path, 'rt').read())
        tokenizer = nlp.data.BERTSPTokenizer(
            path=tokenizer_path, vocab=vocab, lower=False)
        vocab = tokenizer.vocab.token_to_idx
    else:
        TypeError('Invalid pretrained model type')
    return tokenizer, vocab

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

#tokenizer, vocab = get_pretrained_model('kobert')

import pyTextMiner as ptm

text = "예로부터 말이 많은 사람은 말로 망하고 밤중에 말 타고 토끼를 데리고 도망치는 경우가 많다."
mecab_path='C:\\mecab\\mecab-ko-dic'
mecab = ptm.tokenizer.MeCab(mecab_path)
pos = mecab.inst.pos(text)
pos_text = ''
for word_pos in pos:
    pos_text += word_pos[0] + '/' + word_pos[1] + ' '

pos_text = pos_text.strip()
print(pos_text)

marked_text = "[CLS] " + text + " [SEP]"

# Tokenize our sentence with the BERT tokenizer.
tokenized_text = tokenizer.tokenize(marked_text)

# Print out the tokens.
print (tokenized_text)

print(list(tokenizer.vocab.keys())[5000:5020])

# Map the token strings to their vocabulary indeces.
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

# Display the words with their indeces.
for tup in zip(tokenized_text, indexed_tokens):
    print('{:<12} {:>6,}'.format(tup[0], tup[1]))


#segment ID
# Mark each of the 22 tokens as belonging to sentence "1".
segments_ids = [1] * len(tokenized_text)

print (segments_ids)

'''
3. Extracting Embeddings
3.1. Running BERT on our text
Next we need to convert our data to torch tensors and call the BERT model. 
The BERT PyTorch interface requires that the data be in torch tensors rather than Python lists, 
so we convert the lists here - this does not change the shape or the data.
'''
# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])


device = torch.device("cpu")

# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-multilingual-cased',output_hidden_states = True,)

#model = BertModel.from_pretrained('D:\\python_workspace\\pyTextMiner\\bert_models\\pytorch_model.bin',
#                                  output_hidden_states = True,)

model.to(device)

# Put the model in "evaluation" mode, meaning feed-forward operation.
model.eval()

# Run the text through BERT, and collect all of the hidden states produced
# from all 12 layers.
with torch.no_grad():
    outputs = model(tokens_tensor, segments_tensors)

    # Evaluating the model will return a different number of objects based on
    # how it's  configured in the `from_pretrained` call earlier. In this case,
    # becase we set `output_hidden_states = True`, the third item will be the
    # hidden states from all layers. See the documentation for more details:
    # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
    hidden_states = outputs[2]


print ("Number of layers:", len(hidden_states), "  (initial embeddings + 12 BERT layers)")
layer_i = 0

print ("Number of batches:", len(hidden_states[layer_i]))
batch_i = 0

print ("Number of tokens:", len(hidden_states[layer_i][batch_i]))
token_i = 0

print ("Number of hidden units:", len(hidden_states[layer_i][batch_i][token_i]))

# For the 5th token in our sentence, select its feature values from layer 5.
token_i = 5
layer_i = 5
vec = hidden_states[layer_i][batch_i][token_i]

# Plot the values as a histogram to show their distribution.
plt.figure(figsize=(10,10))
plt.hist(vec, bins=200)
plt.show()


# `hidden_states` is a Python list.
print('      Type of hidden_states: ', type(hidden_states))

# Each layer in the list is a torch tensor.
print('Tensor shape for each layer: ', hidden_states[0].size())

#Let’s combine the layers to make this one whole big tensor.
# Concatenate the tensors for all layers. We use `stack` here to
# create a new dimension in the tensor.
token_embeddings = torch.stack(hidden_states, dim=0)
print(str(token_embeddings.size()))

#Let’s get rid of the “batches” dimension since we don’t need it.
# Remove dimension 1, the "batches".
token_embeddings = torch.squeeze(token_embeddings, dim=1)
print(str(token_embeddings.size()))

#Finally, we can switch around the “layers” and “tokens” dimensions with permute.
# Swap dimensions 0 and 1.
token_embeddings = token_embeddings.permute(1,0,2)
print(str(token_embeddings.size()))


# Stores the token vectors, with shape [22 x 3,072]
token_vecs_cat = []

# `token_embeddings` is a [22 x 12 x 768] tensor.

# For each token in the sentence...
for token in token_embeddings:
    # `token` is a [12 x 768] tensor

    # Concatenate the vectors (that is, append them together) from the last
    # four layers.
    # Each layer vector is 768 values, so `cat_vec` is length 3,072.
    cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)

    # Use `cat_vec` to represent `token`.
    token_vecs_cat.append(cat_vec)

print('Shape is: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))


#As an alternative method, let’s try creating the word vectors by summing together the last four layers.
# Stores the token vectors, with shape [22 x 768]
token_vecs_sum = []

# `token_embeddings` is a [22 x 12 x 768] tensor.

# For each token in the sentence...
for token in token_embeddings:
    # `token` is a [12 x 768] tensor

    # Sum the vectors from the last four layers.
    sum_vec = torch.sum(token[-4:], dim=0)

    # Use `sum_vec` to represent `token`.
    token_vecs_sum.append(sum_vec)

print('Shape is: %d x %d' % (len(token_vecs_sum), len(token_vecs_sum[0])))


# `token_vecs` is a tensor with shape [22 x 768]
token_vecs = hidden_states[-2][0]

# Calculate the average of all 22 token vectors.
sentence_embedding = torch.mean(token_vecs, dim=0)
print ("Our final sentence embedding vector of shape:", sentence_embedding.size())


for i, token_str in enumerate(tokenized_text):
  print (i, token_str)

'''
They are at 3, 9, and 15.

For this analysis, we’ll use the word vectors that we created by summing the last four layers.

We can try printing out their vectors to compare them.
'''
print('First 10 vector values for each instance of "말".')
print('')
print("말 (언어) ", str(token_vecs_sum[3][:10]))
print("말 (언어)  ", str(token_vecs_sum[9][:10]))
print("말 (동물)   ", str(token_vecs_sum[15][:10]))

'''
We can see that the values differ, but let’s calculate the cosine similarity between the vectors to make a more precise comparison.
'''

# Calculate the cosine similarity between the word
diff_word = 1 - cosine(token_vecs_sum[3], token_vecs_sum[15])

# Calculate the cosine similarity between the word
same_word = 1 - cosine(token_vecs_sum[3], token_vecs_sum[9])

print('Vector similarity for  *similar*  meanings:  %.2f' % same_word)
print('Vector similarity for *different* meanings:  %.2f' % diff_word)