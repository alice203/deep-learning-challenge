#!/usr/bin/env python
# coding: utf-8

# In[2]:


from __future__ import division
import numpy as np
import sys
import zipfile
import logging
import imageio
import glob
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Loading png

# In[2]:


png = []

for im_path in glob.glob("/Users/aliciahorsch/Anaconda/DL challenge/train/png/*.png"):
    im = imageio.imread(im_path)
    png.append(im)


# In[3]:


plt.imshow(png[0])

#4th channel is transparency, 1-3 is RGB


# In[5]:


png=np.array(png)
print(png.shape)


# In[7]:


np.save("png.npy", png)


# In[8]:


print(png.shape)


# In[9]:


X = np.load("png.npy")


# In[12]:


print(X.shape)


# Load svg file

# In[14]:


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


# In[15]:


svg = load("/Users/aliciahorsch/Anaconda/DL Challenge/train/svg/", zipped = False)


# Process svg to array

# In[16]:


from nltk.tokenize import SpaceTokenizer

tk = SpaceTokenizer()

test = tk.tokenize(svg[677])
print(test)


# In[17]:


tokens = []
for sequence in svg:
    l = tk.tokenize(sequence)
    tokens.append(l)

print(len(tokens))
print(tokens[:1])


# In[18]:


maxi = 0
for file in tokens:
    length = len(file)
    if length > maxi:
        maxi = length
    else:
        continue
print(maxi)


# In[19]:


dic = {}
counter = 1

for tokenlist in tokens:
    for token in tokenlist:
        if token in dic:
            continue
        else:
            dic[token] = counter
            counter +=1
            
print(dic)
print(len(dic))


# In[20]:


r = np.zeros((len(tokens),maxi,len(dic)))
print(r.shape)
l = np.zeros((maxi,len(dic)))
print(l.shape)


# In[21]:


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


# In[22]:


print(len(z))
print(z[0])


# In[33]:


a = []

print(l.shape)
for item in z:
    #print(item)
    l = np.zeros((maxi,57))
    for index,value in enumerate(item):
        l[index,value] = 1
        
    a.append(l)

        


# In[34]:


f = np.array(a)
print(f.shape)


# In[35]:


np.save("svg.npy", f)


# In[36]:


y = np.load("svg.npy")


# In[37]:


print(y.shape)


# In[40]:


print(y[0][18])

