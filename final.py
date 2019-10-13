from __future__ import division
import numpy as np
from nltk.tokenize import SpaceTokenizer
import sys
import zipfile
import logging
import imageio
import glob
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

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
test = tk.tokenize(svg[677])

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

