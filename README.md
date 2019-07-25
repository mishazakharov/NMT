# Neural Machine Translation

<h2>About</h2>

This is a *SMALL* neural machine translation application that can translate from Enlgish to Russian.

To solve the problem of neural machine translation, I used seq2seq recurrent neural network architecture and pre-trained word embeddings([GloVe](https://nlp.stanford.edu/projects/glove/)). This project implements everything from downloading / preparing data to launching an application on a user's sentence. The network has been trained in 20,000 pairs of phrases over 30 epochs.

The purpose of this project is not to make as optimized and computationally effective translation application as possible.
The main goal of the project is to get some experience in building translation-applications and understand seq2seq architecure
of recurrent neural network better.

**Learning and having FUN!**

<h2>Table of Content</h2>

1. [prepare.py](https://github.com/mishazakharov/NMT/blob/master/prepare.py)
    * This file contains code for data preparation, getting rid of punctuation etc...
    
2. [split.py](https://github.com/mishazakharov/NMT/blob/master/split.py)
    * This file contains code for reducing and splitting main dataset.
    
3. [model.py](https://github.com/mishazakharov/NMT/blob/master/model.py)
    * This file contains code for loading prepared data, loading pre-trained word embeddings, preparing embedding matrix,
      compiling and training a model. You can choose the number of epochs yourself. Mine was trained only within 30 epochs!
      
4. [predict.py](https://github.com/mishazakharov/NMT/blob/master/predict.py)
    * This file containcs code for preparing user's sentence and running our trained model on it.
    
5. [models](https://github.com/mishazakharov/NMT/tree/master/models)
    * This folder contains my pre-trained network.
    
6. [data](https://github.com/mishazakharov/NMT/tree/master/data)
    * This folder contains the data I've trained my seq2seq on and compressed *Pickle* files you need to run predict.py.
    
7. [glove](https://github.com/mishazakharov/NMT/tree/master/glove)
    * This folder contains a HTML file that redirects to the download site for pre-trained embeddings.
    
    
<h2>Seq2Seq</h2>


    

