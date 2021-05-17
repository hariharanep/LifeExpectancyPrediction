#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import csv
def run(filename):
  fp = open(filename, "r")
  reader = csv.reader(fp, delimiter = ",")
  data = list(reader)
  fp.close()
  
  real_data = np.array(data)
  column_names = real_data[0, :]
  real_data = real_data[1:]

  reduced_data = np.delete(real_data, np.where(real_data == '')[0], axis = 0)
  unique_countries = np.unique(reduced_data[:, 0])

  for I in range(0, len(reduced_data)):
    to_change = reduced_data[I, 0]
    reduced_data[I, 0] = np.where(unique_countries == to_change)[0][0]

  unique_status = np.unique(reduced_data[:, 2])

  for I in range(0, len(reduced_data)):
    to_change = reduced_data[I, 2]
    reduced_data[I, 2] = np.where(unique_status == to_change)[0][0]

  reduced_data = reduced_data.astype(np.float)
  
  new_col = np.zeros((len(reduced_data), 1))
  new_col_name = np.array(["Life Expectancy Classifier"])
  reduced_data = np.append(reduced_data, new_col, axis=1)
  column_names = np.append(column_names, new_col_name, axis = 0)

  upper_quartile = np.quantile(reduced_data[:, 3], 0.75)
  lower_quartile = np.quantile(reduced_data[:, 3], 0.25)

  for I in range(0, len(reduced_data)):
    if reduced_data[I, 3] >= upper_quartile:
      reduced_data[I, 22] = 1
    elif reduced_data[I, 3] > lower_quartile and reduced_data[I, 3] < upper_quartile:
      reduced_data[I, 22] = 0
    else:
      reduced_data[I, 22] = -1

  reduced_data = np.delete(reduced_data, 3, 1)
  column_names = np.delete(column_names, 3, 0)
  classifications = reduced_data[:, 21]
  reduced_data = np.delete(reduced_data, 21, 1)
  column_names = column_names[0: len(column_names) - 1]
  

  return reduced_data, column_names, classifications

# In[ ]:

