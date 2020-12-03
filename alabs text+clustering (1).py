#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


#X.shape


# In[4]:


#print(X)


# In[5]:


terms = vectorizer.get_feature_names()
terms


# In[12]:


Y=pd.DataFrame(X.toarray())
Y.columns=terms


# In[34]:


Y


# In[28]:


X.fillna(0)


# In[3]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

documents = ["This little kitty came to play when I was eating at a restaurant.",
             "Merley has the best squooshy kitten belly.",
             "Google Translate app is incredible.",
             "If you open 100 tab in google you get a smiley face.",
             "Best cat photo I've ever taken.",
             "Climbing ninja cat.",
             "Impressed with google map feedback.",
             "Key promoter extension for Google Chrome."]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

true_k = 4
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)


# In[5]:


type(documents)


# In[8]:


model.cluster_centers_.argsort()[:, ::-1]


# In[9]:


print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print


# In[10]:


print("\n")
print("Prediction")

Y = vectorizer.transform(["chrome browser to open."])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["My kitty is hungry."])
prediction = model.predict(Y)
print(prediction)


# In[11]:


Y = vectorizer.transform(["chrome browser to open."])
prediction = model.predict(Y)
print(prediction)


# In[12]:


Y = vectorizer.transform(["smile face cat climb"])
prediction = model.predict(Y)
print(prediction)


# In[ ]:





# In[ ]:




