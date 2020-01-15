from __future__ import division
import numpy as np
import os 
from sklearn.model_selection import train_test_split
from nltk.tokenize import SpaceTokenizer
from keras.models import Model
from keras import layers
from keras.layers import Input, Dense, MaxPooling2D, Flatten, Dropout
from keras.layers import SimpleRNN, LSTM, GRU
import sys
import zipfile
import logging
import imageio
import glob

wd = os.getcwd()

#Load images
def load_images(filepath):
    png = []
    for im_path in sorted(glob.glob(filepath)):
        im = imageio.imread(im_path)
        png.append(im)
    
    return np.array(png)

png = load_images(wd+"/train/png/*.png")

#Load svg-files
def load(path):
    strings = []
    for seq_path in sorted(glob.glob(path)):
        proxy = open(seq_path).read()
        proxy = proxy.replace('\n', " ")
        strings.append(proxy)
    return strings

svg = load(wd+"/train/svg/*.svg")

#Process svg-files
tk = SpaceTokenizer()

#Tokenization of sequences
def tokenize(liste):
    tokens = []
    for sequence in svg:
        l = tk.tokenize(sequence)
        tokens.append(l)
    return tokens
tokens = tokenize(svg)

#Determine maximum length of tokenized sequence
def max_length(tokenized_seq_list):
    maxi = 0
    for file in tokens:
        length = len(file)
        if length > maxi:
            maxi = length
        else:
            continue
    return maxi

max_length = max_length(tokens)

#Dictionary of all tokens
def dictionary_tokens_encode(tokenized_seq_list):
    dic = {}
    counter = 0
    for tokenlist in tokenized_seq_list:
        for token in tokenlist:
            if token in dic:
                continue
            else:
                dic[token] = counter
                counter +=1
    return dic
            
token_dic_en = dictionary_tokens_encode(tokens)

#Dictionary of all tokens
def dictionary_tokens_decode(tokenized_seq_list):
    dicti = {}
    counter = 0
    for tokenlist in tokenized_seq_list:
        for token in tokenlist:
            if token in dicti.values():
                continue
            else:
                dicti[counter] = token
                counter +=1
    return dicti
            
token_dic_de = dictionary_tokens_decode(tokens)
print(token_dic_de)

#Make array out of list
def create_array(tokenized_seq_list, dic):
    z = []
    aux =[]
    for tokenlist in tokens:
        for token in tokenlist:
            aux.append(dic[token])
        while len(aux) < 66:
            aux.append(0)
        n = np.array(aux)
        z.append(aux)
        aux = []
    return z

tokenized_seq = create_array(tokens, token_dic_en)

#One-hot-encode tokenized sequences
def one_hot_encode(tokenized_sequence, max_length):
    a = []
    #print(l.shape)
    for item in tokenized_sequence:
        #print(item)
        l = np.zeros((max_length,58))
        for index,value in enumerate(item):
            l[index,value] = 1
        a.append(l)
    return np.array(a)

svg_f = one_hot_encode(tokenized_seq,max_length)

#Train/test-split
X_train, X_val, y_train, y_val = train_test_split(png, svg_f, test_size= 0.33, random_state=0)

###################### - - - - - - - - - Model 

def model():
    num_tokens = 58
    hidden_size = 150
    img_shape = (64,64,4)
    sequence_shape = (None,57)
    ###########---------------------------------Convnet
    #Image Input
    image_input = layers.Input(shape=(img_shape), name="image_input")
    #Conv-Layer
    c1 = layers.Conv2D(256, kernel_size=(3,3), activation='relu', strides=(1, 1), padding='same')(image_input)
    #Maxpooling
    m1 = layers.MaxPooling2D((3,3))(c1)
    #Conv2D layer
    c2 = layers.Conv2D(hidden_size, kernel_size=(3,3), activation='relu', padding='same', strides=(2, 2))(m1)
    #Maxpooling
    m2 = layers.MaxPooling2D((2,2))(c2)
    #Conv2D layer
    c3 = layers.Conv2D(hidden_size, kernel_size=(3,3), activation='relu', padding='same', strides=(2, 2))(m2)
    #Maxpooling
    m3 = layers.MaxPooling2D((2,2))(c3)
    f = layers.Flatten()(m3)
    #Dense layer
    d1 = layers.Dense(hidden_size, activation="relu")(f)

    ###########---------------------------------Sequential part
    #Sequence input
    input_seq = Input(shape=(None, num_tokens))
    #LSTM
    lstm = LSTM(hidden_size, return_sequences=True, name="lstm1")(input_seq)
    # Concatenate inputs
    decoder1 = layers.add([lstm,d1])
    d2 = Dense(hidden_size, activation='relu')(decoder1)
    d3 = Dense(hidden_size, activation='relu')(d2)
    outputs = Dense(num_tokens, activation='softmax')(d3)
    
    train = Model(inputs=[image_input, input_seq], outputs=outputs)

    train.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    return train

train = model()

#Train model
num_epochs = 15
batch_size = 25
train.fit([X_train[:15000], y_train[:15000]], y_train[:15000], 
           batch_size=batch_size,
           epochs=num_epochs,
           verbose=2)

X_test = load_images(wd+"/test/png/*.png")

#Predict model
def predict(images):
    a = []
    b = []
    for image in images:
        #Start is always the same
        start = y_train[0][0].reshape(1,1,58)        
        proxy = image.reshape(1,64,64,4)
        for seq in range(66):
            pred = train.predict([proxy,start])
            p = pred[:,-1:,:]            
            index = p.argmax(axis=0)
            t = np.zeros((1,1,58))
            t[0][0][index] = 1
            start = np.concatenate((start,t), axis=1)
        a.append(start)
        g=np.array(a).reshape(67,58)
        a= []
        b.append(g)
    return np.array(b)

prediciton = predict(X_test)

def decode(predictions, dic):
    svgs = []
    svg = ''
    for example in predictions:
        index = example.argmax(axis=1)
        for item in index:
            proxy = int(item)
            svg += dic[proxy]
        svgs.append(svg)
        svg = ''
    return svgs

result = decode(prediciton, token_dic_de)

for index, pred in enumerate(result):
    file = open(wd+'/result/'+str(48000+index)+'.svg',"w")
    file.write(pred)
    file.close()

