#!/usr/bin/env python3
from pickle import dump
import numpy as np
import pickle

def load(filename):
    """ Load clean dataset

    Args:
        filename(str): a name of the file we are downloading from.

    Returns:
        return(str): downloaded str
    """
    return pickle.load(open(filename,'rb'))

def save_splitted(data,filename):
    """ Save transformed data

    Args:
        data(str): transformed data we want to save with pickle
        filename(str): where we want to save our str

    Returns:
        None
    """
    dump(data,open(filename,'wb'))
    print('Saved: {}'.format(filename))

    return None


data = load('./data/english-russian.pkl')
# Use only 50000 of sentences!
data = data[:20000,:]
# shuffle them randomly
np.random.shuffle(data)
# splitting 
train,test = data[:16000],data[16000:]
# saving
save_splitted(train,'./data/english-russian-train.pkl')
save_splitted(test,'./data/english-russian-test.pkl')
save_splitted(data,'./data/english-russian-reduced.pkl')
