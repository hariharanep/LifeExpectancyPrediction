#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
def training_validation_testing(data, labels):
  positive_samples = np.where(labels == 1)[0]
  negative_samples = np.where(labels == -1)[0]
  zero_samples = np.where(labels == 0)[0]

  train_samples = list(positive_samples[0:int(np.floor(len(positive_samples)/3))]) + list(negative_samples[0:int(np.floor(len(negative_samples)/3))]) + list(zero_samples[0:int(np.floor(len(zero_samples)/3))])  

  validation_samples = list(positive_samples[int(np.floor(len(positive_samples)/3)):int(np.floor(2*len(positive_samples)/3))]) + list(negative_samples[int(np.floor(len(negative_samples)/3)):int(np.floor(2*len(negative_samples)/3))]) + list(zero_samples[int(np.floor(len(zero_samples)/3)):int(np.floor(2*len(zero_samples)/3))])
  

  test_samples = list(positive_samples[int(np.floor(2*len(positive_samples)/3)):len(positive_samples)]) + list(negative_samples[int(np.floor(2*len(negative_samples)/3)):len(negative_samples)]) + list(zero_samples[int(np.floor(2*len(zero_samples)/3)):len(zero_samples)])

  k_list = list(range(1, int(np.sqrt(len(data))) + 1))
  errors = []
  best_err = 1.1
  best_k = 0
  for k in k_list:
    alg = KNeighborsClassifier(n_neighbors=k,algorithm='brute')
    alg.fit(data[train_samples],labels[train_samples])
    labels_pred = alg.predict(data[validation_samples])
    err = np.mean(labels[validation_samples] != np.array([labels_pred]).T)
    errors.append(err)
    print("k=" + str(k) + ", validation set error=" + str(err))
    if err < best_err:
      best_err = err
      best_k = k
  
  print("best_k=" + str(best_k))

  alg = KNeighborsClassifier(n_neighbors=best_k,algorithm='brute')
  alg.fit(data[train_samples],labels[train_samples])
  labels_pred = alg.predict(data[test_samples])
  err = np.mean(labels[test_samples] != np.array([labels_pred]).T)
  print("final best_k test set error=" + str(err))

  return k_list, errors

def bootstrapping(B, data, labels):
  k_list = list(range(1, int(np.sqrt(len(data))) + 1))
  n, d = data.shape  
  errors = []
  best_err = 1000000
  best_k = 0
  for k_to_choose in k_list:
    z = np.zeros((B, 1))
    for i in range(0, B):
      mu = []
      S = set()
      j = 0
      while (j < n and len(S) <= 0.25*len(data)):
        k = np.random.randint(0, n)
        mu.append(k)
        S.add(k)
        j = j + 1
      T = np.array(set(range(0, n))) - np.array(S)

      mu = list(S)
      
      data_train = data[mu]
      labels_train = labels[mu]
      alg = KNeighborsClassifier(n_neighbors=k_to_choose,algorithm='brute')
      alg.fit(data_train, labels_train)
      z[i] = 0
      temp = list(T)
      #print(temp)
      #print(len(temp))
      labels_pred = alg.predict(data[temp])
      z[i]= np.mean(labels[temp] != np.array([labels_pred]).T)
  
    total_err = (sum(z)/len(z))
    errors.append(total_err)
    print("k=" + str(k_to_choose) + ", avg error of all bootstraps=" + str(total_err))
    if total_err < best_err:
      best_err = total_err
      best_k = k_to_choose
  

  
  print("best_k=" + str(best_k))
  return k_list, errors


# In[ ]:


