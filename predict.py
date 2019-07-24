#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from model import eng_tokenizer,encode,rus_tokenizer,eng_max_sentence_length


def mapping(number,tokenizer):
    """ This function maps a word to a given number(output of seq2seq)

    Args:
        number(int): outputted integer
        tokenizer(class): tokenizer of a language we want to map a word in

    Returns:
        vocab_word(str): word that corresponding to this number
    """
    for vocab_word,vocab_number in tokenizer.word_index.items():
        if number == vocab_number:
            return vocab_word
    
    return None

def to_word(prediction,tokenizer):
    """ This function transforms outputted SEQUENCE of integers to a sentence.

    Args:
        prediction(np.ndarray/list): vector of integers representing outputted
        sentence
        tokenizer(class): language tokenizer to map words correctly

    Returns:
        returns(str): a final translated sentence
    """
    target = list()
    for i in prediction:
        word = mapping(i,tokenizer)
        if word is None:
            break
        target.append(word)

    return ' '.join(target)

# Hiding tensorflow warnings!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Avoiding the bag!
with CustomObjectScope({'GlorotUniform':glorot_uniform()}):
    model = load_model('./models/seq2seq.h5')
    
# Source is an input from a user
source = input('Type here:')
# Data preparation
source = encode(eng_tokenizer,eng_max_sentence_length,source)
prediction = model.predict(source,verbose=0)[0]
# Have to take the highest probability for a single word!
prediction = [np.argmax(vector) for vector in prediction]
#print(prediction)
# Getting everything right!
prediction = to_word(prediction,rus_tokenizer)
print('Translation: %s' % prediction)
