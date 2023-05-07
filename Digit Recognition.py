#!/usr/bin/env python
# coding: utf-8

# # Fatching the dataset

# In[1]:


# fetching dataset
from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

mnist = fetch_openml('mnist_784')


# In[17]:


x, y = mnist['data'], mnist['target']


# In[2]:


mnist


# In[31]:


x


# In[18]:


x.shape


# In[32]:


y


# In[19]:


y.shape


# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


import matplotlib
import matplotlib.pyplot as plt


# In[22]:


some_digit = x.to_numpy()[36001]
some_digit_image = some_digit.reshape(28, 28)  # let's reshape to plot it


# In[23]:


plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')

plt.show()


# In[24]:


y[36001]


# In[33]:


x_train, x_test = x[:60000], x[6000:70000]


# In[34]:


y_train, y_test = y[:60000], y[6000:70000]


# In[ ]:





# # Creating a 2-detector

# In[48]:


# Creating a 2-detector
y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)
y_train_2 = (y_train == '2')
y_test_2 = (y_test == '2')


# In[49]:


y_train_2 


# In[ ]:





# In[50]:


from sklearn.linear_model import LogisticRegression


# In[51]:


clf = LogisticRegression(tol=0.1)


# In[ ]:


clf.fit(x_train, y_train_2)


# In[ ]:


example = clf.predict([some_digit])
print(example)


# In[ ]:


# Cross Validation
a = cross_val_score(clf, x_train, y_train_2, cv=3, scoring="accuracy")
print(a.mean())


# In[ ]:





# In[ ]:





# In[ ]:




