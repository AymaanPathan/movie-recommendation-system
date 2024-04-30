#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


movies = pd.read_csv('dataset.csv')


# In[4]:


movies


# In[5]:


movies.head()


# In[6]:


movies.columns #taking out the columns


# In[7]:


movies.info


# In[8]:


movies['tags'] = movies['genre'] + movies['overview'] ## combining two columns for understand the keywords and implement it in future algorithm


# In[9]:


movies.head()


# In[10]:


new_df = movies[['id','title','genre','overview','tags']]


# In[11]:


new_df


# In[12]:


new_df = new_df.drop(columns=['genre','overview'])


# In[13]:


new_df.head()


# In[14]:


from sklearn.feature_extraction.text import CountVectorizer
#CountVectorizer is a tool that helps us turn words in text into numbers so computers can understand them. I


# In[15]:


## stop words for we ignore like [is ,the and other word] we just focuses on genres like stuff


# In[16]:


cv = CountVectorizer(max_features=1000,stop_words='english')


# In[17]:


cv


# In[18]:


vec = cv.fit_transform(new_df['tags'].values.astype('U')).toarray() 
#We use .astype('U') to ensure that the data is represented as Unicode strings ('U').
#If the data is not explicitly converted to Unicode strings, there might be issues with encoding or data representation, 


# In[19]:


vec


# In[20]:


vec.shape


# In[21]:


from sklearn.metrics.pairwise import cosine_similarity


# In[22]:


sim = cosine_similarity(vec)


# In[23]:


sim


# In[47]:


dist = sorted(list(enumerate(sim[0])), reverse=True, key=lambda vec: vec[1])


# In[48]:


dist


# In[58]:


for i in dist[0:5]: #top 5 movies are close to [The Shawshank Redemption] based on keys
    print(new_df.iloc[i[0]].title)


# In[ ]:


# function that recommed movies to user
def recommed(movies):
    index = new_df[new_df['title']==movies].index[0]
    distance = sorted(list(enumerate(sim[index])),reverse=True,key=lambda vec:vec[1]) # It creates pairs of indices and similarity scores for each item in the dataset, where sim[index] represents the similarity scores of the target item (indexed by index).
    for i in distance[0:5]:
        print(new_df.iloc[i[0]].title)


# In[37]:


recommed("Schindler's List") #provide related movies that user search


# In[29]:


movies


# In[ ]:




