#!/usr/bin/env python
# coding: utf-8

# In[1]:

import matplotlib.pyplot as pp
def graph(parameters, errors):
  pp.figure()
  pp.title("Hyperparameters vs Errors Plot")
  pp.plot(parameters, errors, 'o')
  pp.xlabel('hyperparameters')
  pp.ylabel('errors')
  pp.show() # This command will open the figure, and wait

# In[ ]:

