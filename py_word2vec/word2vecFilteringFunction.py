import os

from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

from tqdm import tqdm
import numpy as np
from keras.preprocessing.text import Tokenizer

class word2vecFilteringFunction:
    def __init__(self):

        self.maxlen = 100 # Make all sequences 100 words long
        self.embedding_dim = 100  # We now use larger embeddings
        self.max_words = 10000 # We will only consider the 10K most used words in this dataset

        self.tokenizer = Tokenizer(num_words=self.max_words)

        self.labels = []

        # deep learning model
        self.model = Sequential()

    def fetch(self):

        imdb_dir = 'D:\\My Documents\\teaching\\nlp_deeplearning\\aclImdb_v1\\aclImdb'  # Data directory
        train_dir = os.path.join(imdb_dir, 'train')  # Get the path of the train set

        # Setup empty lists to fill
        labels = []
        texts = []

        # First go through the negatives, then through the positives
        for label_type in ['neg', 'pos']:
            # Get the sub path
            dir_name = os.path.join(train_dir, label_type)
            print('loading ', label_type)
            # Loop over all files in path
            for fname in tqdm(os.listdir(dir_name)):
                # Only consider text files
                if fname[-4:] == '.txt':
                    # Read the text file and put it in the list
                    f = open(os.path.join(dir_name, fname))
                    texts.append(f.read())
                    f.close()
                    # Attach the corresponding label
                    if label_type == 'neg':
                        labels.append(0)
                    else:
                        labels.append(1)

        self.labels = labels;

        # print mean value
        print('label mean value' + str(np.mean(labels)))

        #print sample text and label
        print('Label',labels[24002])
        print(texts[24002])

        return texts

    def tokenize(self, texts):

        self.tokenizer.fit_on_texts(texts) # Generate tokens by counting frequency
        sequences = self.tokenizer.texts_to_sequences(texts) # Turn text into sequence of numbers
        word_index = self.tokenizer.word_index
        print('Token for "the"',word_index['the'])
        print('Token for "Movie"',word_index['movie'])
        print('Token for "generator"',word_index['generator'])

        # Display the first 10 words of the sequence tokenized
        print (str(sequences[24002][:10]))

        return sequences


    def createEmbeddings(self):
        glove_dir = '../embeddings' # This is the folder with the dataset

        print('Loading word vectors')
        embeddings_index = {} # We create a dictionary of word -> embedding
        f = open(os.path.join(glove_dir, 'glove.6B.100d.txt')) # Open file

        # In the dataset, each line represents a new word embedding
        # The line starts with the word and the embedding values follow
        for line in tqdm(f):
            values = line.split()
            word = values[0] # The first value is the word, the rest are the values of the embedding
            embedding = np.asarray(values[1:], dtype='float32') # Load embedding
            embeddings_index[word] = embedding # Add embedding to our embedding dictionary
        f.close()

        print('Found %s word vectors.' % len(embeddings_index))

        # Create a matrix of all embeddings
        all_embs = np.stack(embeddings_index.values())
        emb_mean = all_embs.mean() # Calculate mean
        emb_std = all_embs.std() # Calculate standard deviation

        #filter input data by word embedding

        word_index = self.tokenizer.word_index
        nb_words = min(self.max_words, len(word_index)) # How many words are there actually

        # Create a random matrix with the same mean and std as the embeddings
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, self.embedding_dim))

        # The vectors need to be in the same position as their index.
        # Meaning a word with token 1 needs to be in the second row (rows start with zero) and so on

        # Loop over all words in the word index
        for word, i in word_index.items():
            # If we are above the amount of words we want to use we do nothing
            if i >= self.max_words:
                continue
            # Get the embedding vector for the word
            embedding_vector = embeddings_index.get(word)
            # If there is an embedding vector, put it in the embedding matrix
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix


    def buildModel(self, filtering=True):

        if (filtering):
            embedding_matrix = self.createEmbeddings();
            self.model.add(Embedding(self.max_words, self.embedding_dim, input_length=self.maxlen, weights = [embedding_matrix], trainable = False))
        else:
            self.model.add(Embedding(self.max_words, self.embedding_dim, input_length=self.maxlen))


        self.model.add(Flatten())
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.summary()

    def train(self, sequences):
        #training and validation data

        data = pad_sequences(sequences, maxlen=self.maxlen)
        print(data.shape) # We have 25K, 100 word sequences now

        labels = np.asarray(self.labels)

        # Shuffle data
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]

        training_samples = 20000  # We will be training on 10K samples
        validation_samples = 5000  # We will be validating on 10000 samples

        # Split data
        x_train = data[:training_samples]
        y_train = labels[:training_samples]
        x_val = data[training_samples: training_samples + validation_samples]
        y_val = labels[training_samples: training_samples + validation_samples]

        #
        self.model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['acc'])

        history = self.model.fit(x_train, y_train,
                            epochs=10,
                            batch_size=32,
                            validation_data=(x_val, y_val))

    def saveModel(self, model_file):
        self.model.save(model_file, overwrite=True, include_optimizer=True)


    def loadModel(self, model_file):
        self.model = load_model(model_file)

    def test(self):
        # Demo on a positive text
        my_text = 'I love dogs. Dogs are the best. They are lovely, cuddly animals that only want the best for humans.'

        seq = self.tokenizer.texts_to_sequences([my_text])
        print('raw seq:', seq)
        seq = pad_sequences(seq, maxlen=self.maxlen)
        print('padded seq:', seq)
        prediction = self.model.predict(seq)
        print('positivity:', prediction)


if __name__ == '__main__':

    word2vec_filter = word2vecFilteringFunction()

    model_file = 'sentiment_model.model'
    # fetch data
    texts = word2vec_filter.fetch()

    # tokenize data
    sequences = word2vec_filter.tokenize(texts)

    mode = 'test' # train or test

    if (mode == 'train'):
        filtering_mode = True
        #create deel learning model for sentiment analysis
        word2vec_filter.buildModel(filtering_mode)

        #train the classifier
        word2vec_filter.train(sequences)

        word2vec_filter.saveModel(model_file)

    elif (mode == 'test'):
        #load
        word2vec_filter.loadModel(model_file)

        word2vec_filter.test()