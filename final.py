from __future__ import division
import numpy as np
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

#Load images
png = []
for im_path in glob.glob("/Users/aliciahorsch/Anaconda/DL challenge/train/png/*.png"):
    im = imageio.imread(im_path)
    png.append(im)
png=np.array(png)
np.save("png.npy", png)
#X = np.load("png.npy")

#Load svg-files
def load(path, zipped=True):
    js = range(48000)
    strings = []
    if zipped:
        with zipfile.ZipFile(path) as z:
            for j in js:
                strings.append(z.open("{}.svg".format(j)).read())
    else:
        for j in js:
                strings.append(open("{}/{}.svg".format(path, j)).read())
                
    return strings

svg = load("/Users/aliciahorsch/Anaconda/DL Challenge/train/svg/", zipped = False)

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

#Dictionary of all tokens
def dictionary_tokens(tokenized_seq_list):
    dic = {}
    counter = 1
    for tokenlist in tokens:
        for token in tokenlist:
            if token in dic:
                continue
            else:
                dic[token] = counter
                counter +=1
    return dic
            
token_dic = dictionary_tokens(tokens)

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

tokenized_seq = create_array(tokens, token_dic)

#One-hot-encode tokenized sequences
def one_hot_encode(tokenized_sequence, max_length):
    a = []
    #print(l.shape)
    for item in z:
        #print(item)
        l = np.zeros((max_length,57))
        for index,value in enumerate(item):
            l[index,value] = 1
        a.append(l)
    return np.array(a)
svg_f = one_hot_encode(z,max_length)
np.save("svg.npy", svg_f)
#y = np.load("svg.npy")

#Train/test-split
X_train, X_val, y_train, y_val = train_test_split(png, svg_f, test_size= 0.33, random_state=0)

########################################## --------- Model
#Parameters for model
num_tokens = 57
hidden_size = 100

def model():
    num_tokens = 57
    hidden_size = 100
    img_shape = (64,64,4)
    sequence_shape = (None,57)
    ###########---------------------------------Convnet
    #Image Input
    image_input = layers.Input(shape=(img_shape), name="image_input")
    #Conv-Layer
    c1 = layers.Conv2D(512, kernel_size=(3,3), activation='relu', strides=(1, 1), padding='same')(image_input)
    #Maxpooling
    m1 = layers.MaxPooling2D((3,3))(c1)
    #Conv2D layer
    c2 = layers.Conv2D(256, kernel_size=(3,3), activation='relu', padding='same', strides=(2, 2))(m1)
    #Maxpooling
    m2 = layers.MaxPooling2D((2,2))(c2)
    #Conv2D layer
    c3 = layers.Conv2D(hidden_size, kernel_size=(3,3), activation='relu', padding='same', strides=(2, 2))(m1)
    #Maxpooling
    m3 = layers.MaxPooling2D((2,2))(c3)
    f = layers.Flatten()(m3)
    #Dense layer
    d1 = layers.Dense(hidden_size, activation="relu")(f)

    ###########---------------------------------Sequential part
    #Sequence input
    input_seq = Input(shape=(None, 57))
    #LSTM
    lstm = LSTM(hidden_size, return_sequences=True, name="lstm1")(input_seq)
    # Concatenate inputs
    decoder1 = layers.add([lstm,d1])
    outputs = Dense(num_tokens, activation='softmax')(decoder1)
    
    train = Model(inputs=[image_input, input_seq], outputs=outputs)

    train.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])
    return train

train = model()
print(train.summary())

#Train model
num_epochs = 10
batch_size = 25
train.fit([X_t, y_t], y_t, 
           batch_size=batch_size,
           epochs=num_epochs,
           verbose=2)

#Predict model
def predict(images):
    a = []
    b = []
    for image in images:
        #Start is always the same
        start = y_t[0][0].reshape(1,1,57)
        proxy = image.reshape(1,64,64,4)
        #print(proxy.shape)

        for seq in range(65):
            pred = train.predict([proxy,start])
            p = pred[:,-1:,:]
            index = p.argmax(axis=1)

            t = np.zeros((1,1,57))
            t[0][0][index] = 1

            start = np.concatenate((start,t), axis=1)
            #print(start.shape)

        a.append(start)    

        g=np.array(a).reshape(66,57)

        index = g.argmax(axis=1)
        #print(index)

        #print(g.shape)
        a= []
        b.append(g)
    return np.array(b)

l = predict(X_v)

def decode(predictions, dic):
    svgs = []
    for example in predictions:
        index = example.argmax(axis=1)
        for item in index:
            svg = ''
            svg+=dic[item]
        svgs.append(svg)
        svg = ''
    return svgs


