#!/usr/bin/env python3
import string
import re
from pickle import dump
from unicodedata import normalize 
import numpy as np

file_name = './data/rus-eng/rus.txt'

def load_doc(file_name):
    """ This function load the data.

    In particular, there will be english-russian sentence seperated by a tab!

    Args:
        file_name(str): path to download data from 

    Returns:
        text(str): downloaded big string of data
    """
    doc = open(file_name,mode='rt',encoding='utf-8')
    text = doc.read()
    doc.close()
    
    return text

def pairs(doc):
    """ This function split data by line adn then by phrase

    So if we get something like this ['...'(eng),'...'(rus)]

    Args:
        doc(str): downloaded string-document

    Returns:
        pairx(list): list of pairs
    """
    lines = doc.strip().split('\n')
    pairx = [line.split('\t') for line in lines]

    return pairx

def clear_data(column):
    """ Clear data from any punctuation and conver it to a lower case!

    Args:
        column(np.ndarray): just a numpy vector with sentences.

    Returns:
        column(np.ndarray): transformed vector.
    """ 
    column = [i.translate(str.maketrans('','',string.punctuation)) for i in column]
    for i in range(len(column)):
        column[i] = column[i].lower()

    return column 

def save(data,name):
    """ Save our prepared and cleaned text using pickle API to a file!

    Args:
        data(np.ndarray/str): data we want to save
        name(str): name of the file we want to save to
    """
    dump(data,open(name,'wb'))
    print('Saved {}'.format(name))
    
    return None

text = load_doc(file_name)
eng_rus = pairs(text)
eng_rus = np.array(eng_rus)

eng_rus[:,0] = clear_data(eng_rus[:,0])
eng_rus[:,1] = clear_data(eng_rus[:,1])
# saving our prepared text!
save(eng_rus,'./data/english-russian.pkl')
