#!/usr/bin/python

"""
Simulation catapult linear motion control learning and planning, for landing 
location controlling task considering the generalization of the desired landing 
location. Only the the target position parameter of the linear motion is 
optimized to achieve better performance, and the other parameters are fixed.
"""

__author__    = 'David Qiu'
__email__     = 'david@davidqiu.com'
__website__   = 'www.davidqiu.com'
__copyright__ = 'Copyright (C) 2017, David Qiu. All rights reserved.'


from libCatapultSim import TCatapultSim
from libCatapultDatasetSim import TCatapultDatasetSim

import os
import time
import datetime
import yaml
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

import pdb

sys.path.append(os.path.join(os.path.dirname(__file__), '../lib/cma'))
import cma

sys.path.append(os.path.join(os.path.dirname(__file__), '../lib/DeltaDNN'))
from base_ml_dnn import TNNRegression
from base_util import LoadYAML, SaveYAML

try:
  input = raw_input
except NameError:
  pass



class TCatapultLPLinearSim(object):
  """
  Simulation catapult learning and planning agent in linear motion control.
  """
  
  def __init__(self, catapult, abs_dirpath_data, abs_dirpath_model):
    super(TCatapultLPLinearSim, self).__init__()
    
    self._FIXED_POS_INIT = 0.0
    self._FIXED_DURATION = 0.10

    self._SHOULD_LOAD_MODEL = False
    self._SHOULD_SAVE_MODEL = False

    self.catapult = catapult
    
    self._abs_dirpath_data = abs_dirpath_data
    self._abs_dirpath_model = os.path.join(abs_dirpath_model, './')

    loader_dataset = TCatapultDatasetSim(abs_dirpath=self._abs_dirpath_data, auto_init=False)
    loader_dataset.load_dataset()
    self._dataset = []
    for entry in loader_dataset:
      is_valid = True
      if entry['action']['pos_init'] != self._FIXED_POS_INIT: is_valid = False
      if entry['action']['duration'] != self._FIXED_DURATION: is_valid = False
      if is_valid:
        self._dataset.append(entry)
  
  def _create_model(self, input_dim, output_dim, hiddens=[128, 128], max_updates=20000, should_load_model=False, prefix_info='catapult'):
    model = TNNRegression()
    
    options = {
      #'AdaDelta_rho':         0.5, # 0.5, 0.9
      'dropout':              True,
      'dropout_ratio':        0.01,
      'loss_stddev_stop':     1.0e-4,
      'loss_stddev_stop_err': 1.0e-6,
      #'batchsize':            5, # 5, 10
      #'num_check_stop':       50,
      #'loss_maf_alpha':       0.4,
      'num_max_update':       max_updates,
      'gpu':                  -1,
      'verbose':              True,
      'n_units':              [input_dim] + hiddens + [output_dim]
    }
    model.Load({'options': options})
    
    if should_load_model:
      self._load_model(model, prefix_info)
    
    model.Init()
    
    return model

  def _train_model(self, model, x_train, y_train, batch_train=True):
    if batch_train:
      model.UpdateBatch(x_train, y_train)
    else:
      for x, y, n in zip(x_train, y_train, range(len(x_train))):
        model.Update(x, y, not_learn=((n+1) % min(10, len(x_train)) != 0))
  
  def _save_model(self, model, prefix_info):
    print('{} save mode. (dirpath={})'.format(prefix_info, self._abs_dirpath_model))
    SaveYAML(model.Save(self._abs_dirpath_model), self._abs_dirpath_model + 'nn_model.yaml')
  
  def _load_model(self, model, prefix_info):
    print('{} load mode. (dirpath={})'.format(prefix_info, self._abs_dirpath_model))
    model.Load(LoadYAML(self._abs_dirpath_model + 'nn_model.yaml'), self._abs_dirpath_model)
  
  def _estimate_model_quality(self, model, x_train, y_train, x_valid, y_valid, should_plot=True):
    assert(len(x_valid) > 0)
    assert(len(y_valid) > 0)
    assert(len(x_valid[0]) == 1)
    assert(len(y_valid[0]) == 1)
    
    y_hypo    = []
    err_hypo  = []
    grad_hypo = []
    for i in range(len(x_valid)):
      prediction = model.Predict(x_valid[i], x_var=0.0**2, with_var=True, with_grad=True)
      h_i    = prediction.Y.ravel() # hypothesis
      err_i  = np.sqrt(np.diag(prediction.Var))
      grad_i = prediction.Grad.ravel()
      
      y_hypo.append(h_i)
      err_hypo.append(err_i)
      grad_hypo.append(grad_i)
    
    y_diff = []
    y_stderr = []
    for i in range(len(y_valid)):
      y_diff_i   = y_valid[i] - y_hypo[i]
      y_stderr_i = np.linalg.norm(y_diff_i)
      
      y_diff.append(y_diff_i)
      y_stderr.append(y_stderr_i)
    
    err_diff = []
    err_stderr = []
    for i in range(len(y_stderr)):
      err_diff_i   = y_stderr[i] - err_hypo[i]
      err_stderr_i = np.linalg.norm(err_diff_i)
      
      err_diff.append(err_diff_i)
      err_stderr.append(err_stderr_i)
    
    acc_stderr_y   = 0.0
    acc_stderr_err = 0.0
    for y_stderr_i in y_stderr:
      acc_stderr_y += y_stderr_i
    for err_stderr_i in err_stderr:
      acc_stderr_err += err_stderr_i
    ave_stderr_y   = acc_stderr_y / len(y_stderr)
    ave_stderr_err = acc_stderr_err / len(err_stderr)
    
    if should_plot:
      plot_x_train  = [(x[0]) for x in x_train]
      plot_y_train  = [(y[0]) for y in y_train]
      plot_x_valid  = [(x[0]) for x in x_valid]
      plot_y_valid  = [(y[0]) for y in y_valid]
      plot_y_hypo   = [(y[0]) for y in y_hypo]
      plot_err_hypo = [(e[0]) for e in err_hypo]
      plot_y_diff   = [(y[0]) for y in y_diff]
      plot_y_stderr = [(e)    for e in y_stderr]
      
      plt.figure(1)
      plt.clf()
      
      plt.subplot(211)
      plt.plot(plot_x_train, plot_y_train, 'ro')
      plt.plot(plot_x_valid, plot_y_valid, 'g--',
               plot_x_valid, plot_y_hypo,  'b-')
      plt.ylabel('y')
      plt.grid(True)
      
      plt.subplot(212)
      plt.plot(plot_x_valid, plot_y_diff,   'ro',
               plot_x_valid, plot_y_stderr, 'g--',
               plot_x_valid, plot_err_hypo, 'b-')
      plt.ylabel('err')
      plt.grid(True)
      
      plt.show()
    
    return ave_stderr_y, ave_stderr_err

  def _run_model_based(self):
    prefix = 'catapult/model_based'
    prefix_info = prefix + ':'
    
    should_load_model = False
    should_load_model_input = input('load model without training (y/N)?> ').strip().lower()
    if should_load_model_input in ['y']:
      should_load_model = True
    
    model = self._create_model(1, 1, hiddens=[128, 128], max_updates=10000, should_load_model=should_load_model, prefix_info=prefix_info)
    
    x_train = []
    y_train = []
    for entry in self._dataset:
      x_train.append([entry['action']['pos_target']])
      y_train.append([entry['result']['loc_land']])
    if not should_load_model:
      self._train_model(model, x_train, y_train, batch_train=True)
      
    should_estimate_model_quality_input = input('estimate model quality (Y/n)?> ').strip().lower()
    if should_estimate_model_quality_input in ['', 'y']:
      ave_stderr_y, ave_stderr_err = self._estimate_model_quality(model, x_train, y_train, x_train, y_train, should_plot=True)
      print('{} ave_stderr_y = {}, ave_stderr_err = {}'.format(prefix_info, ave_stderr_y, ave_stderr_err))
    
    should_save_model_input = input('save model (Y/n)?> ').strip().lower()
    if should_save_model_input in ['', 'y']:
      self._save_model(model, prefix_info)
    

  def _run_model_free(self):
    pass

  def _run_hybrid(self):
    pass

  def getOperations(self):
    operation_dict = {
      'mb': self._run_model_based,
      'mf': self._run_model_free,
      'hybrid': self._run_hybrid
    }
    return operation_dict
  
  def run(self, operation):
    operation_dict = self.getOperations()
    
    if operation in operation_dict:
      op_func = operation_dict[operation]
      return op_func()
    
    raise ValueError('Invalid operation.', operation)



if __name__ == '__main__':
  dirpath_sim = '../../ode/simpleode/catapult'
  catapult_name = 'sim_001_01'

  catapult = TCatapultSim(dirpath_sim)
  
  abs_dirpath_data = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/catapult_' + catapult_name))
  abs_dirpath_model = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/model_' + catapult_name))
  agent = TCatapultLPLinearSim(catapult, abs_dirpath_data, abs_dirpath_model)
  
  operation = 'mb'
  if len(sys.argv) >= 2:
    if len(sys.argv) == 2 and (sys.argv[1] in agent.getOperations()):
      operation = sys.argv[1]
    else:
      print('usage: ./run_001_linear.py <operation>')
      quit()
  
  agent.run(operation)

