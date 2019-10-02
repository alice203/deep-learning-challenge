#!/usr/bin/env python
# coding: utf-8

# Load pictures, make them into arrays

# In[9]:


import imageio
import glob

im0 = imageio.imread("/Users/aliciahorsch/Anaconda/DL challenge/0.png")
print(im0.shape)
print(im0[:5,:10,])


# In[11]:


png = []

for im_path in glob.glob("/Users/aliciahorsch/Anaconda/DL challenge/train/png/*.png"):
    im = imageio.imread(im_path)
    png.append(im)


# In[12]:


print(png[0].shape)


# In[24]:


#print(png[0])


# In[37]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.imshow(png[0])
plt.imshow(png[0][:,:,1])
#4th channel is transparency, 1-3 is RGB


# Load svg, make them into arrays

# In[34]:


import xml.etree.ElementTree as ET

al = []
svg = []

for svg_path in glob.glob("/Users/aliciahorsch/Anaconda/DL challenge/train/svg/*.svg"):
    tree = ET.parse('0.svg')
    root = tree.getroot()
    for child in root:
        if "defs" in child.tag:
            continue
        else:
            a = child.tag
            b = a[28:]
            svg.append([b, child.attrib])
    al.append(svg)
    svg = []


# In[6]:


#need tag (ellipse)
#and attributes of shapes


# In[35]:


print(al[0])
print()
print(al[0][0][0])

