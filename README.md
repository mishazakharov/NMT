# Neural Machine Translation

<h2>About</h2>

This is a *SMALL* neural machine translation application that can translate from Enlgish to Russian.

To solve the problem of neural machine translation, I used seq2seq recurrent neural network architecture and pre-trained word embeddings([GloVe](https://nlp.stanford.edu/projects/glove/)). This project implements everything from downloading / preparing data to launching an application on a user's sentence. The network has been trained in 20,000 pairs of phrases over 30 epochs.

![image](https://miro.medium.com/max/1400/1*3lj8AGqfwEE5KCTJ-dXTvg.png)

> The purpose of this project is not to make as optimized and computationally effective translation application as possible.

> The main goal of the project is to get some experience in building translation-applications and understand seq2seq      architecure
of recurrent neural network better.

**Learning and having FUN!**

<h2>Table of Content</h2>

1. [prepare.py](https://github.com/mishazakharov/NMT/blob/master/prepare.py)
    * This file contains code for data preparation, getting rid of punctuation etc...
    [
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


1. General ideas about architecture.
![image](https://miro.medium.com/max/1400/1*1JcHGUU7rFgtXC_mydUA_Q.jpeg)

2. Links that can help you figure it out and which I used: [1](https://towardsdatascience.com/nlp-sequence-to-sequence-networks-part-2-seq2seq-model-encoderdecoder-model-6c22e29fd7e1),[2](https://towardsdatascience.com/understanding-encoder-decoder-sequence-to-sequence-model-679e04af4346),[3](https://machinelearningmastery.com/introduction-neural-machine-translation/),[4](https://machinelearningmastery.com/develop-neural-machine-translation-system-keras/),[5](https://keras.io/examples/pretrained_word_embeddings/),[6](https://www.youtube.com/watch?v=XXtpJxZBa2c)

<h2>Installation</h2>

```
git clone https://github.com/mishazakharov/NMT
cd NMT
python3 prepare.py
```

In case you want to train network yourself:

```
python3 model.py
```

Before you run any files you need to do:

```
./NMT/glove/glove.html
unzip ./NMT/data/data.zip
```

<h2>Contact</h2>

If you want to work on this together or just feeling social, feel free to contact me [here](https://vk.com/rtyyu).
And I am also available at this(misha_zakharov96@mail.ru) and this(vorahkazmisha@gmail.com) email addresses!
**Feel free** to give me any advice. :+1:





