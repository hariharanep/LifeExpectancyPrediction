#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as pp
import k_nearest_neighbors
import naive_bayes
def generate_explained_variance_graph(data):
  vars = []
  for i in range(0, len(data[0])):
    vars.append(np.var(data[:, i]))
  
  vars.sort(reverse = True)
  vars = vars/sum(vars)
  
  temp = list(range(1, len(data[0]) + 1))

  pp.figure()
  pp.title("Share of Total Explained Variance in Data for All Amounts of Components")
  pp.plot(temp, vars) 
  pp.xlabel('Number of Components')
  pp.xticks(temp, temp)
  pp.ylabel('Share of Total Explained Variance In Data')
  pp.show()

def one_feature_classification_results(data, labels, feature):
  reduced_data = data[:, feature].reshape(-1, 1)
  k_list, errors = k_nearest_neighbors.training_validation_testing(reduced_data, labels)
  naive_bayes.training_validation_testing(reduced_data, labels)
# In[ ]:

