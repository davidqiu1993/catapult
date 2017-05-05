#!/usr/bin/python

"""
The neural network model simulation catapult controller library for advanced 
motion controlling.
"""

__author__    = 'David Qiu'
__email__     = 'david@davidqiu.com'
__website__   = 'www.davidqiu.com'
__copyright__ = 'Copyright (C) 2017, David Qiu. All rights reserved.'


import os
import sys
import time
import datetime
import yaml
import math
import numpy as np
import matplotlib.pyplot as plt

import pdb

from libCatapultDatasetSimNN import TCatapultDatasetSimNN

sys.path.append(os.path.join(os.path.dirname(__file__), '../lib/base'))
from base_ml_dnn import TNNRegression
from base_util import LoadYAML, SaveYAML, Rand, FRange1



class TCatapultSimNN1D(object):
  """
  Catapult controller.
  
  @constant MOTION The throwing motion trajectory.
  @constant POS_MIN The minimum position.
  @constant POS_MAX The maximum position.
  @constant POS_INIT The initial position.
  @constant DURATION The duration.
  """
  
  def __init__(self, modelName, dirpath_data):
    """
    Initialize a catapult controller.
    """
    super(TCatapultSimNN1D, self).__init__()
    
    if modelName == 'simNN_001':
      self.MOTION   = 'linear'
      self.POS_MIN  = 0.00 * math.pi
      self.POS_MAX  = 0.75 * math.pi
      self.POS_INIT = 0.00 * math.pi
      self.DURATION = 0.10
    else:
      assert(False)
    self.modelName = modelName
    
    self._dirpath_data       = os.path.abspath(dirpath_data)
    self._dirpath_refdataset = os.path.abspath(os.path.join(self._dirpath_data, 'catapult_' + self.modelName + '_referenceDataset'));
    self._dirpath_refmodel   = os.path.abspath(os.path.join(self._dirpath_data, 'catapult_' + self.modelName + '_referenceModel'));
    if self._dirpath_refmodel[-1] != '/':
      self._dirpath_refmodel += '/'
    
    self._dataset_dynamics = self._load_reference_dynamics_dataset()
    self._model_dynamics = self._load_reference_dynamics_model()
  
  def _load_reference_dynamics_dataset(self):
    dataset = TCatapultDatasetSimNN(abs_dirpath=self._dirpath_refdataset, auto_init=False)
    dataset.load_dataset()
    return dataset
  
  def _load_reference_dynamics_model(self):
    model_dynamics = TNNRegression()
    
    options = {
      #'AdaDelta_rho':         0.5, # 0.5, 0.9
      'dropout':              False,
      'dropout_ratio':        0.01,
      'loss_stddev_stop':     1.0e-4,
      'loss_stddev_stop_err': 1.0e-6,
      'batchsize':            64,
      #'num_check_stop':       50,
      #'loss_maf_alpha':       0.4,
      'num_max_update':       20000,
      'gpu':                  -1,
      'verbose':              True,
      'n_units':              [1, 200, 200, 1]
    }
    model_dynamics.Load({'options':options})
    
    hasModelLoaded = False
    try:
      model_dynamics.Load(LoadYAML(self._dirpath_refmodel + 'nn_model.yaml'), self._dirpath_refmodel)
      hasModelLoaded = True
    except IOError:
      hasModelLoaded = False
    
    model_dynamics.Init()
    
    if not hasModelLoaded:
      X_train_dynamics = []
      Y_train_dynamics = []
      for entry in self._dataset_dynamics:
        assert(entry['motion'] == self.MOTION)
        assert(entry['action']['pos_init'] == self.POS_INIT)
        assert(entry['action']['duration'] == self.DURATION)
        X_train_dynamics.append([entry['action']['pos_target']])
        Y_train_dynamics.append([entry['result']['loc_land']])
      model_dynamics.UpdateBatch(X_train_dynamics, Y_train_dynamics)
      SaveYAML(model_dynamics.Save(self._dirpath_refmodel), self._dirpath_refmodel + 'nn_model.yaml')
    
    return model_dynamics
  
  def show_dynamics_model(self, plot_density=100):
    X_train_plot = []
    Y_train_plot = []
    for entry in self._dataset_dynamics:
      X_train_plot.append(entry['action']['pos_target'])
      Y_train_plot.append(entry['result']['loc_land'])
    
    X_plot = []
    Y_plot = []
    errs_plot = []
    for i in range(plot_density + 1):
      x = 0.0 + i * (self.POS_MAX - self.POS_MIN) / plot_density
      
      prediction = self._model_dynamics.Predict([x], with_var=True)
      y = float(prediction.Y.ravel()[0])
      err = float(np.sqrt(np.diag(prediction.Var))[0])
      
      X_plot.append(x)
      Y_plot.append(y)
      errs_plot.append(err)
    
    plt.figure(1)
    plt.clf()
    plt.plot(X_train_plot, Y_train_plot, 'bx')
    plt.errorbar(X_plot, Y_plot, errs_plot, color='r', linestyle='-')
    plt.show()
  
  def throw_linear(self, pos_target):
    """
    Throw from initial position to target position in linear motion.

    @param pos_target The target position of the catapult.
    @return The landing location of the thrown object.
    """
    prediction   = self._model_dynamics.Predict([pos_target], with_var=True)
    loc_land_h   = float(prediction.Y.ravel()[0])
    loc_land_err = float(np.sqrt(np.diag(prediction.Var))[0])
    
    return loc_land_h



if __name__ == '__main__':
  catapult = TCatapultSimNN1D('simNN_001', '../data/')
  loc_land = catapult.throw_linear(catapult.POS_MAX)
  print('pos_target = {}, loc_land = {}'.format(catapult.POS_MAX, loc_land))
  catapult.show_dynamics_model()


