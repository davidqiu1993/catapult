#!/usr/bin/python

"""
Multilinear approximators for multimodal approximation problems.
"""

__author__    = 'David Qiu'
__email__     = 'david@davidqiu.com'
__website__   = 'www.davidqiu.com'
__copyright__ = 'Copyright (C) 2017, David Qiu. All rights reserved.'


import math
import numpy as np
from sklearn import linear_model

import pdb


class TMultilinearApproximators(object):
  def __init__(self, dim_input=1, n_approximators=20):
    assert(dim_input > 0)
    assert(n_approximators > 0)
    
    super(TMultilinearApproximators, self).__init__()
    
    self._DIM_INPUT = dim_input
    self._N_APPROXIMATORS = n_approximators
    
    self._approximators = [] # [(coefficients, intercept), ...]
    self._X_train_list = [] # [[x, ...], ...] where x = vector([x1, x2, ...])
    self._Y_train_list = [] # [[y, ...], ...] where y = scalar(y)
    for approximatorIndex in range(self._N_APPROXIMATORS):
      coefficients = [0.0 for i in range(self._DIM_INPUT)]
      intercept = 0.0
      self._approximators.append((coefficients, intercept))
      self._X_train_list.append([])
      self._Y_train_list.append([])
  
  def _train(self, approximatorIndex):
    assert(len(self._X_train_list[approximatorIndex]) > 0)
    assert(len(self._X_train_list[approximatorIndex]) == len(self._Y_train_list[approximatorIndex]))
    
    reg = linear_model.LinearRegression()
    reg.fit(self._X_train_list[approximatorIndex], self._Y_train_list[approximatorIndex])
    
    coefficients = reg.coef_.tolist()
    intercept = float(reg.intercept_)
    self._approximators[approximatorIndex] = (coefficients, intercept)
    
    return self._approximators[approximatorIndex]
  
  def _trainAll(self):
    for approximatorIndex in range(self._N_APPROXIMATORS):
      self._train(approximatorIndex)
    
    return self._approximators
  
  def update(self, approximatorIndex, append_X_train, append_Y_train):
    assert(0 <= approximatorIndex and approximatorIndex < self._N_APPROXIMATORS)
    assert(len(append_X_train) > 0)
    assert(len(append_X_train) == len(append_Y_train))
    for i in range(len(append_X_train)):
      assert(len(append_X_train[i]) == self._DIM_INPUT)
    
    self._X_train_list[approximatorIndex] += append_X_train
    self._Y_train_list[approximatorIndex] += append_Y_train
    
    return self._train(approximatorIndex)
  
  def updateAll(self, append_X_train_list, append_Y_train_list):
    for approximatorIndex in range(self._N_APPROXIMATORS):
      self.update(approximatorIndex, append_X_train_list[approximatorIndex], append_Y_train_list[approximatorIndex])
    
    return self._approximators
  
  def flush(self, approximatorIndex):
    self._X_train_list[approximatorIndex] = []
    self._Y_train_list[approximatorIndex] = []
  
  def flushAll(self):
    for approximatorIndex in range(self._N_APPROXIMATORS):
      self.flush(approximatorIndex)
  
  def predict(self, approximatorIndex, x):
    assert(len(x) == self._DIM_INPUT)
    
    w, b = self._approximators[approximatorIndex]
    
    y_h = float(np.dot(np.array(w), np.array(x)) + b)
    
    return y_h
  
  def predictAll(self, x):
    y_h_list = []
    for approximatorIndex in range(self._N_APPROXIMATORS):
      y_h = self.predict(approximatorIndex, x)
      y_h_list.append(y_h)
    
    return y_h_list
  
  def normalVectorsAngle(self, approximatorIndex_1, approximatorIndex_2):
    """
    @return angle in [0.0, pi]
    """
    assert(0 <= approximatorIndex_1 and approximatorIndex_1 < self._N_APPROXIMATORS)
    assert(0 <= approximatorIndex_2 and approximatorIndex_2 < self._N_APPROXIMATORS)
    
    coefficients_1, intercept_1 = self._approximators[approximatorIndex_1]
    coefficients_2, intercept_2 = self._approximators[approximatorIndex_2]
    
    nv_1 = np.array([-1] + coefficients_1)
    nv_2 = np.array([-1] + coefficients_2)
    cosTheta = np.dot(nv_1, nv_2) / (np.linalg.norm(nv_1) * np.linalg.norm(nv_2))
    theta = float(np.arccos(cosTheta))
    
    return theta
  
  def hyperplanesAngle(self, approximatorIndex_1, approximatorIndex_2):
    """
    return angle in [0.0, 0.5*pi]
    """
    nv_angle = self.normalVectorsAngle(approximatorIndex_1, approximatorIndex_2)
    hp_angle = float(min(nv_angle, math.pi - nv_angle))
    
    return hp_angle
  
  def countApproximators(self):
    return self._N_APPROXIMATORS
  
  @property
  def N(self):
    return self.countApproximators()
  
  def __len__(self):
    return self.countApproximators()
  
  def getApproximator(self, approximatorIndex):
    return self._approximators[approximatorIndex]
  
  def __getitem__(self, approximatorIndex):
    return self.getApproximator(approximatorIndex)
  
  def getAllApproximators(self):
    return self._approximators
  
  def getX(self, approximatorIndex):
    return self._X_train_list[approximatorIndex]
  
  def getY(self, approximatorIndex):
    return self._Y_train_list[approximatorIndex]
  
  def getAllX(self):
    return self._X_train_list
  
  def getAllY(self):
    return self._Y_train_list



if __name__ == '__main__':
  import matplotlib.pyplot as plt
  
  n_approximators = 3
  n_samples = 5
  mtl = TMultilinearApproximators(dim_input=1, n_approximators=n_approximators)
  
  X_train_list = [[[np.random.random()] for j in range(n_samples)] for i in range(n_approximators)]
  Y_train_list = [[ np.random.random()  for j in range(n_samples)] for i in range(n_approximators)]
  mtl.updateAll(X_train_list, Y_train_list)
  
  X_train = [[np.random.random()] for i in range(n_samples)]
  Y_train = [ np.random.random()  for i in range(n_samples)]
  mtl.update(0, X_train, Y_train)
  
  X = [[0.0], [1.0]]
  y_list_list = [mtl.predictAll(X[i]) for i in range(len(X))]
  Y_list = [[y_list_list[i][j] for i in range(len(X))] for j in range(len(mtl))]
  
  color_list = ['r', 'y', 'b']
  
  print('X = {}, Y = {}, marker = {}'.format(X_train, Y_train, color_list[0] + 'x'))
  plt.plot(X_train, Y_train, color_list[0] + 'x')
  
  for i in range(n_approximators):
    print('X = {}, Y = {}, marker = {}'.format(X_train_list[i], Y_train_list[i], color_list[i] + 'x'))
    plt.plot(X_train_list[i], Y_train_list[i], color_list[i] + 'x')
    print('X_pred = {}, Y_pred = {}, marker = {}'.format(X, Y_list[i], color_list[i] + '-'))
    plt.plot(X, Y_list[i], color_list[i] + '-')
  
  nv_angle = 180 * mtl.normalVectorsAngle(0, 1) / math.pi
  hp_angle = 180 * mtl.hyperplanesAngle(0, 1) / math.pi
  print('line_red, line_yellow: nv_angle = {}, hp_angle = {}'.format(nv_angle, hp_angle))
  
  plt.show()


