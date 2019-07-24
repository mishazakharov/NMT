#!/usr/bin/env python3
import os
import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.initializers import Constant


def load(filename):
    """ Load dataset

    Args:
        filename(str): a name of the file we are downloading from

    Returns:
        return(str): downloaded str
    """
    return pickle.load(open(filename,'rb'))

def tokenization(text):
    """ This function allows to vectorize a text corpus.

    To be exact, it translates all vocabulary in numbers

    Args:
        text(str): language-vocabulary we want to "vectroize"

    Returns:
        tokenizer(class): tokenizer
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)

    return tokenizer

def length(lines):
    """ This function finds the length of the longest sequence in a vocabulary.

    Args:
        lines(str): list of phrases - our vocabulary

    Returns:
        returns(int): a length of the longest sentence in a vocabulary
    """

    return max(len(line.split()) for line in lines)

def encode(tokenizer,size,lines):
    """ This function encodes to integers and paddes to the maximum phrase len

    i.e. it takes a sentence and transform it to integers with respect to a
    language-tokenizer we created and the length of the longest sentence in 
    a vocabulary on that particular language.

    Args:
        tokenizer(class): tokenizer for a language-vocabulary
        size(int): the lenght of the longest SENTENCE in a vocabulary
        lines(str): the lines we want to encode

    Returns:
        x(list/np.ndarray): encoded data
    """
    x = tokenizer.texts_to_sequences(lines)
    # pad missing with zeros
    x = pad_sequences(x,maxlen=size,padding='post')

    return x

def encode_output(sequences,vocab_size):
    """ This function translates from integers to words from our vocabulary

    Args:
        sequences(list/np.ndarray): encoded "phrases"
        vocab_size(int): the size of language vocabulary

    Returns:
        y(np.ndarray): encoded vector
    """
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence,num_classes=vocab_size)
        ylist.append(encoded)
    y = np.array(ylist)
    y = y.reshape(sequences.shape[0],sequences.shape[1],vocab_size)

    return y

def build_model(source_vocab,target_vocab,
        source_timesteps,target_timesteps,n_units,embedding_layer):
    """ This function builds a model, which is a seq2seq architecture of rnns.

    Args:
        source_vocab(int): number of words in a source language
        target_vocab(int): number of words in a target language
        source_timesteps(int): length of the longest sentence in
                               a source language
        target_timesteps(int): length of the longest sentence in
                               a target language
        n_units(int): number of neurons in a layer
        embedding_layer(class): pre-trained GloVe english embeddings

    Returns:
        model(class): finished model
    """
    model = Sequential()
    model.add(embedding_layer)
    model.add(LSTM(n_units))
    # We have to do this for our hidden state of lstm to match the input dim
    # of a decoder.
    model.add(RepeatVector(target_timesteps))
    model.add(LSTM(n_units,return_sequences=True))
    model.add(TimeDistributed(Dense(target_vocab,activation='softmax')))

    return model

# Loading datasets:
dataset = load('./data/english-russian-reduced.pkl')
train = load('./data/english-russian-train.pkl')
test = load('./data/english-russian-test.pkl')
# Prepare english vocabulary
eng_tokenizer = tokenization(dataset[:,0])
eng_vocabulary_size = len(eng_tokenizer.word_index) + 1
eng_max_sentence_length = length(dataset[:,0])
# Prepare russian vocabulary
rus_tokenizer = tokenization(dataset[:,1])
rus_vocabulary_size = len(rus_tokenizer.word_index) + 1
rus_max_sentence_length = length(dataset[:,1])

if __name__ == '__main__':
    # Prepare the training data
    X_train = encode(rus_tokenizer,rus_max_sentence_length,train[:,1])
    y_train = encode(eng_tokenizer,eng_max_sentence_length,train[:,0])
    X_train = encode_output(X_train,rus_vocabulary_size)
    # Prepare the test data
    X_test = encode(rus_tokenizer,rus_max_sentence_length,test[:,1])
    y_test = encode(eng_tokenizer,eng_max_sentence_length,test[:,0])
    X_test = encode_output(X_test,rus_vocabulary_size)

    print('Preparing pre-trained embeddings')
    # Defining pre-trained embeddings!
    BASE_DIR = '' 
    GLOVE_DIR = os.path.join(BASE_DIR, 'glove')
    embeddings_index = dict()
    with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    print('Prepare embedding matrix')
    # Preparing embeddings matrix for our frozen embedding layer
    EMBEDDING_DIM = 100
    MAX_NUM_WORDS = eng_max_sentence_length
    word_index = eng_tokenizer.word_index
    num_words = min(MAX_NUM_WORDS, len(word_index)) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embeddin index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # Load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as te keep the embeddins fixed
    embedding_layer = Embedding(eng_vocabulary_size,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=eng_max_sentence_length,
                                trainable=False
                                )
    #print(X_train.shape,y_train.shape,'xtrain,ytrain')
    #print(X_test.shape,y_test.shape,'xtest,ytest')

    # defining the NMT model
    model = build_model(eng_vocabulary_size,rus_vocabulary_size,
                       eng_max_sentence_length,rus_max_sentence_length,
                       100,
                       embedding_layer)
    print(model.summary())
    model.compile(optimizer='adam',loss='categorical_crossentropy')
    # fitting
    saving_path = './models/seq2seq.h5'
    n_epochs = int(input('Type the number of epoch for learning:')) 
    history = model.fit(y_train,X_train,epochs=n_epochs,batch_size=64,
                        validation_data=(y_test,X_test))
    model.save(saving_path)
    print('This model has been successfully saved!')
