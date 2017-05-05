#!/usr/bin/python

"""
Multilinear approximators for multimodal approximation problems.
"""

__author__    = 'David Qiu'
__email__     = 'david@davidqiu.com'
__website__   = 'www.davidqiu.com'
__copyright__ = 'Copyright (C) 2017, David Qiu. All rights reserved.'


import numpy as np
from scipy import stats

import pdb


class TMultilinearApproximators(object):
  def __init__(self, init_range_min, init_range_max, n_approximators=20):
    assert(init_range_max > init_range_min)
    super(TMultilinearApproximators, self).__init__()
    
    self._N_APPROXIMATORS = n_approximators
    
    self._approximators = []
    self._X_train_list = []
    self._Y_train_list = []
    self._rvalue_list = [] # correlation coefficient
    self._pvalue_list = [] # two-sided p-value for a hypothesis test whose null hypothesis is that the slope is zero
    self._stderr_list = [] # standard error of the estimated gradient
    for i in range(self._N_APPROXIMATORS):
      w0 = float(init_range_min + np.random.sample() * (init_range_max - init_range_min)) # intercept
      w1 = float(0.0) # slope
      self._approximators.append([w0, w1])
      self._X_train_list.append([])
      self._Y_train_list.append([])
      self._rvalue_list.append(0.0)
      self._pvalue_list.append(0.0)
      self._stderr_list.append(0.0)
  
  def predictByApproximator(self, approximatorIndex, x):
    w = self._approximators[approximatorIndex]
    y = w[0] + w[1] * x
    return y
  
  def predict(self, x):
    is_batch = hasattr(x, '__iter__')
    
    if is_batch:
      X = x
      Y_list = [[(self.predictByApproximator(i, X[j])) for j in range(len(X))] for i in range(self._N_APPROXIMATORS)]
      return Y_list
    else:
      y_list = [(self.predictByApproximator(i, x)) for i in range(self._N_APPROXIMATORS)]
      return y_list
  
  def update(self, append_X_train_list, append_Y_train_list):
    assert(len(append_X_train_list) == self._N_APPROXIMATORS)
    assert(len(append_Y_train_list) == self._N_APPROXIMATORS)
    
    for i in range(self._N_APPROXIMATORS):
      assert(len(append_X_train_list[i]) == len(append_Y_train_list[i]))
      for j in range(len(append_X_train_list[i])):
        self._X_train_list[i].append(append_X_train_list[i][j])
        self._Y_train_list[i].append(append_Y_train_list[i][j])
      
      if len(self._X_train_list[i]) > 0:
        assert(len(self._X_train_list[i]) == len(self._Y_train_list[i]))
        slope, intercept, r_value, p_value, std_err = stats.linregress(self._X_train_list[i], self._Y_train_list[i])
        self._approximators[i][0] = intercept
        self._approximators[i][1] = slope
        self._rvalue_list[i] = r_value
        self._pvalue_list[i] = p_value
        self._stderr_list[i] = std_err
    
    return self._approximators, self._rvalue_list, self._pvalue_list, self._stderr_list
  
  def getApproximators(self):
    return self._approximators
  
  def getApproximator(self, approximatorIndex):
    return self._approximators[approximatorIndex]
  
  def get_X_train_list(self):
    return self._X_train_list
  
  def get_X_train(self, approximatorIndex):
    return self._X_train_list[approximatorIndex]
  
  def get_Y_train_list(self):
    return self._Y_train_list
  
  def get_Y_train(self, approximatorIndex):
    return self._Y_train_list[approximatorIndex]
  
  def get_rvalue_list(self):
    return self._rvalue_list
  
  def get_rvalue(self, approximatorIndex):
    return self._rvalue_list[approximatorIndex]
  
  def get_pvalue_list(self):
    return self._pvalue_list
  
  def get_pvalue(self, approximatorIndex):
    return self._pvalue_list[approximatorIndex]
  
  def get_stderr_list(self):
    return self._stderr_list
  
  def get_stderr(self, approximatorIndex):
    return self._stderr_list[approximatorIndex]


if __name__ == '__main__':
  import matplotlib.pyplot as plt
  
  n_approximators = 3
  n_samples = 16
  mtl = TMultilinearApproximators(0.0, 1.0, n_approximators=n_approximators)
  
  X_train_list = [(np.random.random(n_samples)) for i in range(n_approximators)]
  Y_train_list = [(np.random.random(n_samples)) for i in range(n_approximators)]
  mtl.update(X_train_list, Y_train_list)
  
  X = [0.0, 1.0]
  Y_list = mtl.predict(X)
  color_list = ['r', 'y', 'b']
  for i in range(n_approximators):
    print X, Y_list[i], color_list[i] + '-'
    plt.plot(X, Y_list[i], color_list[i] + '-')
    print X_train_list[i], Y_train_list[i], color_list[i] + 'x'
    plt.scatter(X_train_list[i], Y_train_list[i], color=color_list[i], marker='x')
  plt.show()

