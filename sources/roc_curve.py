#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as pp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
def k_nearest_neighbors(data, labels):
  positive_samples = np.where(labels == 1)[0]
  negative_samples = np.where(labels == -1)[0]
  zero_samples = np.where(labels == 0)[0]

  train_samples = list(positive_samples[0:int(np.floor(len(positive_samples)/2))]) + list(negative_samples[0:int(np.floor(len(negative_samples)/2))]) + list(zero_samples[0:int(np.floor(len(zero_samples)/2))])

  test_samples = list(positive_samples[int(np.floor(len(positive_samples)/2)):len(positive_samples)]) + list(negative_samples[int(np.floor(len(negative_samples)/2)):len(negative_samples)]) + list(zero_samples[int(np.floor(len(zero_samples)/2)):len(zero_samples)])
  
  data_train = data[train_samples]
  labels_train = labels[train_samples]
  data_test = data[test_samples]
  labels_test = labels[test_samples]

  for i in range(0, 3):
    temp = 0
    if i == 0:
      temp = -1
    elif i == 1:
      temp = 0
    elif i == 2:
      temp = 1

    x_values = []
    y_values = []
    for k in range(1, int(np.sqrt(len(data))) + 1):
      alg = KNeighborsClassifier(n_neighbors = k, algorithm = 'brute')
      alg.fit(data_train, labels_train)
      labels_pred = alg.predict(data_test)
      tru_pos = 0
      false_neg = 0
      tru_neg = 0
      false_pos = 0
    
      if i == 0:
        tru_pos = len(np.where(np.logical_and(labels_test == -1, labels_pred == -1))[0])
        false_neg = len(np.where(np.logical_and(labels_test == -1, labels_pred != -1))[0])
        tru_neg = len(np.where(np.logical_and(labels_test != -1, labels_pred != -1))[0])
        false_pos = len(np.where(np.logical_and(labels_test != -1, labels_pred == -1))[0])
      elif i == 1:
        tru_pos = len(np.where(np.logical_and(labels_test == 0, labels_pred == 0))[0])
        false_neg = len(np.where(np.logical_and(labels_test == 0, labels_pred != 0))[0])
        tru_neg = len(np.where(np.logical_and(labels_test != 0, labels_pred != 0))[0])
        false_pos = len(np.where(np.logical_and(labels_test != 0, labels_pred == 0))[0])
      elif i == 2:
        tru_pos = len(np.where(np.logical_and(labels_test == 1, labels_pred == 1))[0])
        false_neg = len(np.where(np.logical_and(labels_test == 1, labels_pred != 1))[0])
        tru_neg = len(np.where(np.logical_and(labels_test != 1, labels_pred != 1))[0])
        false_pos = len(np.where(np.logical_and(labels_test != 1, labels_pred == 1))[0])

      sensitivity = tru_pos/(tru_pos + false_neg)
      specificity = tru_neg/(tru_neg + false_pos)

      x_values.append(specificity)
      y_values.append(sensitivity)
    
    pp.figure(i)
    pp.title('Receiver Operating Characteristic For Classification Label ' + str(temp))
    pp.plot([0, 1], [1, 0],'r--')
    pp.xlim([0, 1])
    pp.ylim([0, 1])
    pp.ylabel('Sensitivity')
    pp.xlabel('Specificity')  
    pp.plot(x_values, y_values, 'b')
    
  pp.show()
  


def naive_bayes(data, labels):
  positive_samples = np.where(labels == 1)[0]
  negative_samples = np.where(labels == -1)[0]
  zero_samples = np.where(labels == 0)[0]

  train_samples = list(positive_samples[0:int(np.floor(len(positive_samples)/2))]) + list(negative_samples[0:int(np.floor(len(negative_samples)/2))]) + list(zero_samples[0:int(np.floor(len(zero_samples)/2))])

  test_samples = list(positive_samples[int(np.floor(len(positive_samples)/2)):len(positive_samples)]) + list(negative_samples[int(np.floor(len(negative_samples)/2)):len(negative_samples)]) + list(zero_samples[int(np.floor(len(zero_samples)/2)):len(zero_samples)])

  data_train = data[train_samples]
  labels_train = labels[train_samples]
  data_test = data[test_samples]
  labels_test = labels[test_samples]

  for i in range(0, 3):
    x_values = []
    y_values = []
    temp = 0
    if i == 0:
      temp = -1
    elif i == 1:
      temp = 0
    elif i == 2:
      temp = 1 
    for k in range(0, 4):
      gnb = 0
      if k == 0:
        gnb = GaussianNB()
      elif k == 1:
        gnb = MultinomialNB()
      elif k == 2:
        gnb = ComplementNB()
      elif k == 3:
        gnb = BernoulliNB()


      labels_pred = gnb.fit(data_train, labels_train).predict(data_test)
      tru_pos = 0
      false_neg = 0
      tru_neg = 0
      false_pos = 0

      if i == 0:
        tru_pos = len(np.where(np.logical_and(labels_test == -1, labels_pred == -1))[0])
        false_neg = len(np.where(np.logical_and(labels_test == -1, labels_pred != -1))[0])
        tru_neg = len(np.where(np.logical_and(labels_test != -1, labels_pred != -1))[0])
        false_pos = len(np.where(np.logical_and(labels_test != -1, labels_pred == -1))[0])
      elif i == 1:
        tru_pos = len(np.where(np.logical_and(labels_test == 0, labels_pred == 0))[0])
        false_neg = len(np.where(np.logical_and(labels_test == 0, labels_pred != 0))[0])
        tru_neg = len(np.where(np.logical_and(labels_test != 0, labels_pred != 0))[0])
        false_pos = len(np.where(np.logical_and(labels_test != 0, labels_pred == 0))[0])
      elif i == 2:
        tru_pos = len(np.where(np.logical_and(labels_test == 1, labels_pred == 1))[0])
        false_neg = len(np.where(np.logical_and(labels_test == 1, labels_pred != 1))[0])
        tru_neg = len(np.where(np.logical_and(labels_test != 1, labels_pred != 1))[0])
        false_pos = len(np.where(np.logical_and(labels_test != 1, labels_pred == 1))[0])

      sensitivity = tru_pos/(tru_pos + false_neg)
      specificity = tru_neg/(tru_neg + false_pos)

      x_values.append(specificity)
      y_values.append(sensitivity)
  
    pp.figure(i)
    pp.title('Receiver Operating Characteristic For Classification Label ' + str(temp))
    pp.plot([0, 1], [1, 0],'r--')
    pp.xlim([0, 1])
    pp.ylim([0, 1])
    pp.ylabel('Sensitivity')
    pp.xlabel('Specificity')  
    pp.plot(x_values, y_values, 'b')


  pp.show()


# In[ ]:

