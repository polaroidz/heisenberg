# -*- coding: utf-8 -*-
import numpy as np
import unicodedata
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Embedding
from sklearn.preprocessing import OneHotEncoder

__all__ = ['get_model', 'str2vec', 'vec2str', 'v2k', 'k2v', 'clear_string']

def clear_string(data):
    return ''.join(x for x in unicodedata.normalize('NFKD', data)
                     if x not in ['$', '%', '&', '/']).lower()

mode = 'test'

conversas = open('data/scripts.txt')
conversas = conversas.readlines()
conversas = map(lambda x: x.split(":")[-1].strip(), conversas)
conversas = " ".join(conversas)
conversas = clear_string(conversas)

boc = set(conversas)
boc = sorted(boc)

classes = len(boc)
epochs = 30
batch_size = 64
sent_size = 30
batchs = 5000

k2v = dict([(k, v) for k,v in enumerate(boc)])
v2k = dict([(v, k) for k,v in enumerate(boc)])

encoder = OneHotEncoder()
encoder = encoder.fit(np.arange(classes).reshape(-1, 1))

def vec2str(vec):
    out = ""
    
    for v in vec:
        max_ = max(v)
        idx = list(v).index(max_)
        out += k2v[idx]
    
    return out
    
def str2vec(str_):
    str_ = map(v2k.get, str_)
    str_ = np.array(list(str_)).reshape(-1,1)
    str_ = encoder.transform(str_).toarray()
    return str_

def get_batch(n, size=batch_size):
    batch = conversas[n*size:n*size+size]
    X = map(v2k.get, batch[:-1])
    X = np.array(list(X))
    y = str2vec(batch[sent_size:])
    
    X_new = np.zeros((size - sent_size, sent_size))
    
    for i in range(0, size - sent_size):
        X_new[i] = X[i:i+sent_size]
    
    return (X_new, y)

def batch_generator(model):
    i = 0
    while True:
        yield get_batch(i)
        i += 1
        if i == batchs:
            i = 0
            model.save_weights("heisenberg_checkpoint.h5")

def get_model(pretrained=False):
    model = Sequential()

    model.add(Embedding(classes, 128))
    model.add(Bidirectional(LSTM(128, activation='tanh')))
    model.add(Dense(classes))
    model.add(Activation('softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    if pretrained:
        model.load_weights("./heisenberg_checkpoint.h5")

    return model

if __name__ == '__main__':
    if mode == 'train':
        model.fit_generator(batch_generator(model), steps_per_epoch=batchs, epochs=epochs)
        
        model.save_weights("heisenberg_final.h5")
    else:
        model.load_weights("heisenberg_checkpoint.h5")
        
        pred_sent = "hi "    

        for i in range(50): 
            X = map(v2k.get, pred_sent[-sent_size:])
            X = np.array(list(X)).reshape(1, -1)
            
            y = model.predict(X)
            pred_sent += vec2str(y)
        
        print(pred_sent)
