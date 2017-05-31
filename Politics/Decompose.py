
# coding: utf-8

# In[1]:

import time
import cPickle as pickle

starttime = time.time()
# read the cotensor
cotensor = pickle.load( open( "cotensor_5k_no10.p", "rb" ) )
endtime = time.time()
print 'reading done! time:', endtime - starttime


# In[3]:

# Decompose and get the embeddings
import tf_glove


# In[4]:

# play with the parameters here

model = tf_glove.GloVeModel(embedding_size=100, cooccurrence_cap=1000, scaling_factor=0.75, learning_rate=0.08, cooccur=cotensor)


# In[ ]:

starttime = time.time()
model.fit_to_corpus('5k')
model.train(num_epochs=100)
model.flush_embeddings('5k')
endtime = time.time()
print 'training done! time:', endtime - starttime


# In[ ]:



