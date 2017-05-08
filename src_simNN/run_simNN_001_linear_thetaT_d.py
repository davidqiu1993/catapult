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


import os
import time
import datetime
import yaml
import sys
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import optimize as sp_optimize

import pdb

from libCatapultDatasetSimNN import TCatapultDatasetSimNN
from libCatapultSimNN import TCatapultSimNN1D
from libLoggerSimNN import TLoggerSimNN

sys.path.append(os.path.join(os.path.dirname(__file__), '../lib/cma'))
import cma

sys.path.append(os.path.join(os.path.dirname(__file__), '../lib/base'))
from base_ml_dnn import TNNRegression
from base_util import LoadYAML, SaveYAML

sys.path.append(os.path.join(os.path.dirname(__file__), '../lib/multilinear'))
from libMultilinearApproximators import TMultilinearApproximators

try:
  input = raw_input
except NameError:
  pass

#logger = TLoggerSimNN(fp_log='run_simNN_001_linear_thetaT_d.log')
logger = TLoggerSimNN(fp_log=None)



class TCatapultLPLinearSimNN(object):
  """
  Neural network model simulation catapult learning and planning agent in 
  linear motion control.
  """
  
  def __init__(self, catapult, abs_data_dirpath, timestamp=None):
    super(TCatapultLPLinearSimNN, self).__init__()
    
    self._BENCHMARK_EPISODES = 100
    self._BENCHMARK_INIT_SAMPLES_N = 5
    
    self._CONFIG_NN_MAX_UPDATE = 5000
    self._CONFIG_NN_BATCHSIZE = 64
    self._CONFIG_NN_VERBOSE = False
    self._CONFIG_MTL_APPROXIMATORS_N = 20
    self._CONFIG_MTL_LOCAL_SAMPLES_N = 20
    self._CONFIG_MTL_LOCAL_SAMPLES_VAR_RATIO = 0.05
    self._CONFIG_MTL_REINIT_THRESHOLD = 6.0 / 180.0 * math.pi # in rad
    
    self._CONFIG_CMAES_POPSIZE = 20
    self._CONFIG_CMAES_VERBOSE = False
    self._CONFIG_GD_VERBOSE = False
    self._CONFIG_MSGD_STSIZE = 20
    self._CONFIG_MSGD_VERBOSE = False
    self._CONFIG_NNPOLICYGD_VERBOSE = False
    self._CONFIG_MSGD_NN_VERBOSE = False
    self._CONFIG_MTLPOLICYGD_VERBOSE = False
    self._CONFIG_MTL_NN_VERBOSE = False
    
    self._CONFIG_GENERATE_NNPOLICY_SAMPLES_VERBOSE = False
    
    self._CONFIG_HYBRID_POLICY_EXP_SAMPLES_N = 12
    self._CONFIG_HYBRID_POLICY_NEW_SAMPLES_N = 4
    
    self._CONFIG_EVALUATION_PLOT_DENSITY = 100
    
    self.catapult = catapult
    
    dirpath_prefix = 'catapult_' + self.catapult.modelName + '_'
    self._dirpath_data        = os.path.abspath(abs_data_dirpath)
    self._dirpath_dataset     = os.path.abspath(os.path.join(self._dirpath_data, dirpath_prefix + 'dataset'))
    self._dirpath_log         = os.path.abspath(os.path.join(self._dirpath_data, dirpath_prefix + 'log'))
    self._dirpath_model       = os.path.abspath(os.path.join(self._dirpath_data, dirpath_prefix + 'model'))
    self._dirpath_refdataset  = os.path.abspath(os.path.join(self._dirpath_data, dirpath_prefix + 'referenceDataset'))
    self._dirpath_refmodel    = os.path.abspath(os.path.join(self._dirpath_data, dirpath_prefix + 'referenceModel'))
    if self._dirpath_refmodel[-1] != '/':
      self._dirpath_refmodel += '/'
    
    self._FIXED_MOTION   = self.catapult.MOTION
    self._FIXED_POS_INIT = self.catapult.POS_INIT
    self._FIXED_DURATION = self.catapult.DURATION
    self._POS_TARGET_MIN = self.catapult.POS_MIN + 0.001 * math.pi
    self._POS_TARGET_MAX = self.catapult.POS_MAX
    
    if self.catapult.modelName == 'simNN_001':
      self._ESTIMATED_LOC_LAND_MIN = 0.0
      self._ESTIMATED_LOC_LAND_MAX = 26.0
    else:
      assert(False)
    
    self._timestamp = timestamp
    if self._timestamp is None:
      self._timestamp = self._getTimestamp()
    
    self._ref_model_dynamics = self.catapult._model_dynamics
  
  def _getTimestamp(self):
    timestamp_str = '{:%Y%m%d_%H%M%S_%f}'.format(datetime.datetime.now())
    return timestamp_str
  
  def _createNNDynamics(self):
    model_dynamics = TNNRegression()
    
    options = {
      #'AdaDelta_rho':         0.5, # 0.5, 0.9
      #'dropout':              True,
      #'dropout_ratio':        0.01,
      'loss_stddev_stop':     1.0e-4,
      'loss_stddev_stop_err': 1.0e-6,
      'batchsize':            self._CONFIG_NN_BATCHSIZE,
      #'num_check_stop':       50,
      #'loss_maf_alpha':       0.4,
      'num_max_update':       self._CONFIG_NN_MAX_UPDATE,
      'gpu':                  -1,
      'verbose':              self._CONFIG_NN_VERBOSE,
      'n_units':              [1, 200, 200, 1]
    }
    model_dynamics.Load({'options': options})
    
    model_dynamics.Init()
    
    return model_dynamics
  
  def _createNNPolicy(self):
    model_policy = TNNRegression()
    
    options = {
      #'AdaDelta_rho':         0.5, # 0.5, 0.9
      #'dropout':              True,
      #'dropout_ratio':        0.01,
      'loss_stddev_stop':     1.0e-4,
      'loss_stddev_stop_err': 1.0e-6,
      'batchsize':            self._CONFIG_NN_BATCHSIZE,
      #'num_check_stop':       50,
      #'loss_maf_alpha':       0.4,
      'num_max_update':       self._CONFIG_NN_MAX_UPDATE,
      'gpu':                  -1,
      'verbose':              self._CONFIG_NN_VERBOSE,
      'n_units':              [1, 200, 200, 1]
    }
    model_policy.Load({'options': options})
    
    model_policy.Init()
    
    return model_policy
  
  def _createMTLPolicy(self, model_dynamics):
    prefix_info = 'catapult/createMTLPolicy:'
    
    mtl_policy = TMultilinearApproximators(dim_input=1, n_approximators=self._CONFIG_MTL_APPROXIMATORS_N)
    mtl_policy_inactive = []
    
    for approximatorIndex in range(len(mtl_policy)):
      pos_target_sample = self._POS_TARGET_MIN + (self._POS_TARGET_MAX - self._POS_TARGET_MIN) * np.random.sample()
      self._updateLocalMTLPolicyForApproximator(mtl_policy, approximatorIndex, model_dynamics, pos_target_sample)
      mtl_policy_inactive.append(int(0.5 * len(mtl_policy)))
      logger.log('{} train multilinear policy approximator with local samples (approximator: {}/{}, mean: {})'.format(
        prefix_info, approximatorIndex+1, len(mtl_policy), pos_target_sample))
    
    return mtl_policy, mtl_policy_inactive
  
  def _trainNNDynamics(self, model_dynamics, samples, flush=False, not_learn=False):
    """
    sample = (pos_target, loc_land)
    """
    if flush:
      model_dynamics.DataX = np.array([], np.float32)
      model_dynamics.DataY = np.array([], np.float32)
    
    X_train_dynamics = []
    Y_train_dynamics = []
    for sample in samples:
      pos_target, loc_land = sample
      X_train_dynamics.append([pos_target])
      Y_train_dynamics.append([loc_land])
    
    model_dynamics.UpdateBatch(X_train_dynamics, Y_train_dynamics, not_learn=not_learn)
  
  def _trainNNPolicy(self, model_policy, samples, flush=False, not_learn=False):
    """
    sample = (loc_land, pos_target)
    """
    if flush:
      model_policy.DataX = np.array([], np.float32)
      model_policy.DataY = np.array([], np.float32)
    
    X_train_policy = []
    Y_train_policy = []
    for sample in samples:
      loc_land, pos_target = sample
      X_train_policy.append([loc_land])
      Y_train_policy.append([pos_target])
    
    model_policy.UpdateBatch(X_train_policy, Y_train_policy, not_learn=not_learn)
  
  def _trainMTLPolicyForApproximator(self, mtl_policy, approximatorIndex, samples, flush=False):
    """
    sample = (loc_land, pos_target)
    """
    if flush:
      mtl_policy.flush(approximatorIndex)
    
    X_train_policy = []
    Y_train_policy = []
    for sample in samples:
      loc_land, pos_target = sample
      X_train_policy.append([loc_land])
      Y_train_policy.append(pos_target)
    
    mtl_policy.update(approximatorIndex, X_train_policy, Y_train_policy)
  
  def _trainMTLPolicy(self, mtl_policy, samples_list, flush=False):
    """
    sample = (loc_land, pos_target)
    """
    for approximatorIndex in range(len(samples_list)):
      self._trainMTLPolicyForApproximator(mtl_policy, approximatorIndex, samples, flush=flush)
  
  def _updateLocalMTLPolicyForApproximator(self, mtl_policy, approximatorIndex, model_dynamics, pos_target):
    samples = []
    
    for i in range(self._CONFIG_MTL_LOCAL_SAMPLES_N):
      y = np.random.normal(pos_target, self._CONFIG_MTL_LOCAL_SAMPLES_VAR_RATIO * (self._POS_TARGET_MAX - self._POS_TARGET_MIN))
      y = self._fixRange(y, self._POS_TARGET_MIN, self._POS_TARGET_MAX)
      
      prediction = model_dynamics.Predict([y], with_var=True)
      loc_land_h   = (prediction.Y.ravel())[0]
      loc_land_err = (np.sqrt(np.diag(prediction.Var)))[0]
      
      x = loc_land_h
      
      samples.append((x, y))
    
    self._trainMTLPolicyForApproximator(mtl_policy, approximatorIndex, samples, flush=True)
  
  def _savetrial(self, trialResults):
    filepath_trialResults = os.path.abspath(os.path.join(self._dirpath_log, 'trial_' + self._timestamp + '.yaml'))
    with open(filepath_trialResults, 'w') as yamlfile:
      yaml.dump(trialResults, yamlfile, default_flow_style=False)
  
  def _savefig(self, plt, postfix_file):
    plt.savefig(os.path.join(self._dirpath_log, 'trial_' + self._timestamp + '_' + postfix_file + '.svg'), format='svg')
    plt.savefig(os.path.join(self._dirpath_log, 'trial_' + self._timestamp + '_' + postfix_file + '.png'), format='png')
  
  def _evaluateTrialResults(self, trialResults):
    postfix_file = 'results'
    postfix_file_learning_curve = 'learning_curve'
    
    self._savetrial(trialResults)
    
    samples_episode = []
    samples_desired_loc_land = []
    samples_pos_target = []
    samples_loc_land = []
    samples_reward = []
    for i in range(len(trialResults)):
      entry = trialResults[i]
      desired_loc_land = entry['desired_loc_land']
      pos_target       = entry['pos_target']
      loc_land         = entry['loc_land']
      reward           = - abs(desired_loc_land - loc_land)
      samples_episode.append(i+1)
      samples_desired_loc_land.append(desired_loc_land)
      samples_pos_target.append(pos_target)
      samples_loc_land.append(loc_land)
      samples_reward.append(reward)
    
    X_plot_dynamics_true = []
    Y_plot_dynamics_true = []
    for i in range(self._CONFIG_EVALUATION_PLOT_DENSITY + 1):
      x = self._POS_TARGET_MIN + i * (self._POS_TARGET_MAX - self._POS_TARGET_MIN) / self._CONFIG_EVALUATION_PLOT_DENSITY
      
      prediction = self._ref_model_dynamics.Predict([x])
      y = (prediction.Y.ravel())[0]
      
      X_plot_dynamics_true.append(x)
      Y_plot_dynamics_true.append(y)
    
    # Plot episodic data
    plt.figure(1)
    plt.clf()
    
    plt.subplot(211)
    plt.xlabel('d*')
    plt.ylabel('thetaT')
    plt.plot(Y_plot_dynamics_true, X_plot_dynamics_true, 'b-')
    plt.plot(samples_desired_loc_land, samples_pos_target, 'ro')
    
    plt.subplot(212)
    plt.xlabel('d*')
    plt.ylabel('d')
    plt.plot(Y_plot_dynamics_true, Y_plot_dynamics_true, 'b-')
    plt.plot(samples_desired_loc_land, samples_loc_land, 'ro')
    
    self._savefig(plt, postfix_file)
    
    # Plot learning curve
    plt.figure(1)
    plt.clf()
    
    plt.xlabel('episode')
    plt.ylabel('reward')
    axis_x_min = 0
    axis_x_max = self._BENCHMARK_EPISODES + 1
    axis_y_min = - (self._ESTIMATED_LOC_LAND_MAX - self._ESTIMATED_LOC_LAND_MIN) * 1.01
    axis_y_max = + (self._ESTIMATED_LOC_LAND_MAX - self._ESTIMATED_LOC_LAND_MIN) * 0.01
    plt.axis((axis_x_min, axis_x_max, axis_y_min, axis_y_max))
    plt.plot(samples_episode, samples_reward, 'r-')
    
    self._savefig(plt, postfix_file_learning_curve)
  
  def _evaluateNNDynamics(self, model_dynamics, X_train_dynamics, Y_train_dynamics):
    postfix_file = 'NNDynamics'
    
    plt.figure(1)
    plt.clf()
    
    # Plot true dynamics
    X_plot_dynamics_true = []
    Y_plot_dynamics_true = []
    for i in range(self._CONFIG_EVALUATION_PLOT_DENSITY + 1):
      x = self._POS_TARGET_MIN + i * (self._POS_TARGET_MAX - self._POS_TARGET_MIN) / self._CONFIG_EVALUATION_PLOT_DENSITY
      
      prediction = self._ref_model_dynamics.Predict([x])
      y = (prediction.Y.ravel())[0]
      
      X_plot_dynamics_true.append(x)
      Y_plot_dynamics_true.append(y)
    
    plt.plot(X_plot_dynamics_true, Y_plot_dynamics_true, 'b-')
    
    # Plot training data
    plt.plot(X_train_dynamics, Y_train_dynamics, 'ro')
    
    # Plot dynamics model
    X_plot_dynamics = []
    Y_plot_dynamics = []
    errs_plot_dynamics = []
    for i in range(self._CONFIG_EVALUATION_PLOT_DENSITY + 1):
      x = self._POS_TARGET_MIN + i * (self._POS_TARGET_MAX - self._POS_TARGET_MIN) / self._CONFIG_EVALUATION_PLOT_DENSITY
      
      prediction = model_dynamics.Predict([x], with_var=True)
      y = (prediction.Y.ravel())[0]
      err = (np.sqrt(np.diag(prediction.Var)))[0]
      
      X_plot_dynamics.append(x)
      Y_plot_dynamics.append(y)
      errs_plot_dynamics.append(err)
    
    plt.errorbar(X_plot_dynamics, Y_plot_dynamics, errs_plot_dynamics, color='r', linestyle='-')
    
    self._savefig(plt, postfix_file)
  
  def _evaluateNNPolicy(self, model_dynamics, model_policy, X_train_policy, Y_train_policy):
    postfix_file = 'NNPolicy'
    
    plt.figure(1)
    plt.clf()
    
    # Plot flipped true dynamics
    X_plot_dynamics_true = []
    Y_plot_dynamics_true = []
    for i in range(self._CONFIG_EVALUATION_PLOT_DENSITY + 1):
      x = self._POS_TARGET_MIN + i * (self._POS_TARGET_MAX - self._POS_TARGET_MIN) / self._CONFIG_EVALUATION_PLOT_DENSITY
      
      prediction = self._ref_model_dynamics.Predict([x])
      y = (prediction.Y.ravel())[0]
      
      X_plot_dynamics_true.append(x)
      Y_plot_dynamics_true.append(y)
    
    plt.plot(Y_plot_dynamics_true, X_plot_dynamics_true, 'b-')
    
    # Plot flipped dynamics model
    X_plot_dynamics = []
    Y_plot_dynamics = []
    for i in range(self._CONFIG_EVALUATION_PLOT_DENSITY + 1):
      x = self._POS_TARGET_MIN + i * (self._POS_TARGET_MAX - self._POS_TARGET_MIN) / self._CONFIG_EVALUATION_PLOT_DENSITY
      
      prediction = model_dynamics.Predict([x])
      y = (prediction.Y.ravel())[0]
      
      X_plot_dynamics.append(x)
      Y_plot_dynamics.append(y)
    
    plt.plot(Y_plot_dynamics, X_plot_dynamics, 'r-')
    
    # Plot training data
    plt.plot(X_train_policy, Y_train_policy, 'yo')
    
    # Plot policy network
    X_plot_policy = []
    Y_plot_policy = []
    errs_plot_policy = []
    for i in range(self._CONFIG_EVALUATION_PLOT_DENSITY + 1):
      x = self._ESTIMATED_LOC_LAND_MIN + i * (self._ESTIMATED_LOC_LAND_MAX - self._ESTIMATED_LOC_LAND_MIN) / self._CONFIG_EVALUATION_PLOT_DENSITY
      
      prediction = model_policy.Predict([x], with_var=True)
      y = (prediction.Y.ravel())[0]
      err = (np.sqrt(np.diag(prediction.Var)))[0]
      
      X_plot_policy.append(x)
      Y_plot_policy.append(y)
      errs_plot_policy.append(err)
    
    plt.errorbar(X_plot_policy, Y_plot_policy, errs_plot_policy, color='r', linestyle='-')
    
    self._savefig(plt, postfix_file)
  
  def _evaluateMTLPolicy(self, model_dynamics, mtl_policy):
    postfix_file = 'MTLPolicy'
    
    plt.figure(1)
    plt.clf()
    
    axis_x_min = - (self._ESTIMATED_LOC_LAND_MAX - self._ESTIMATED_LOC_LAND_MIN) * 0.01
    axis_x_max = + (self._ESTIMATED_LOC_LAND_MAX - self._ESTIMATED_LOC_LAND_MIN) * 1.01
    axis_y_min = self.catapult.POS_MIN
    axis_y_max = self.catapult.POS_MAX
    plt.axis((axis_x_min, axis_x_max, axis_y_min, axis_y_max))
    
    # Plot flipped true dynamics
    X_plot_dynamics_true = []
    Y_plot_dynamics_true = []
    for i in range(self._CONFIG_EVALUATION_PLOT_DENSITY + 1):
      x = self._POS_TARGET_MIN + i * (self._POS_TARGET_MAX - self._POS_TARGET_MIN) / self._CONFIG_EVALUATION_PLOT_DENSITY
      
      prediction = self._ref_model_dynamics.Predict([x])
      y = (prediction.Y.ravel())[0]
      
      X_plot_dynamics_true.append(x)
      Y_plot_dynamics_true.append(y)
    
    plt.plot(Y_plot_dynamics_true, X_plot_dynamics_true, 'b-')
    
    # Plot flipped dynamics model
    X_plot_dynamics = []
    Y_plot_dynamics = []
    for i in range(self._CONFIG_EVALUATION_PLOT_DENSITY + 1):
      x = self._POS_TARGET_MIN + i * (self._POS_TARGET_MAX - self._POS_TARGET_MIN) / self._CONFIG_EVALUATION_PLOT_DENSITY
      
      prediction = model_dynamics.Predict([x])
      y = (prediction.Y.ravel())[0]
      
      X_plot_dynamics.append(x)
      Y_plot_dynamics.append(y)
    
    plt.plot(Y_plot_dynamics, X_plot_dynamics, 'r-')
    
    # Plot policy approximators
    X_plot_policy = [[self._ESTIMATED_LOC_LAND_MIN], [self._ESTIMATED_LOC_LAND_MAX]]
    y_plot_policy_list_list = [mtl_policy.predictAll(X_plot_policy[i]) for i in range(len(X_plot_policy))]
    Y_plot_policy_list = [[y_plot_policy_list_list[i][j] for i in range(len(X_plot_policy))] for j in range(len(mtl_policy))]
    for approximatorIndex in range(len(Y_plot_policy_list)):
      plt.plot(X_plot_policy, Y_plot_policy_list[approximatorIndex], 'y-')
    
    self._savefig(plt, postfix_file)
  
  def _fixRange(self, val, val_min, val_max):
    return max(val_min, min(val, val_max))
  
  def _launchModule_generateTestSamples(self):
    """
    sample = desired_loc_land
    """
    prefix_info = 'catapult/generateTestSamples:'
    
    samples_desired_loc_land = self._ESTIMATED_LOC_LAND_MIN + (self._ESTIMATED_LOC_LAND_MAX - self._ESTIMATED_LOC_LAND_MIN) * np.random.sample(self._BENCHMARK_EPISODES)
    logger.log('{} samples generated (samples: {})'.format(prefix_info, len(samples_desired_loc_land)))
    
    return samples_desired_loc_land
  
  def _launchModule_collectInitialDynamicsSamples(self):
    """
    sample = (pos_target, loc_land)
    """
    prefix_info = 'catapult/collectInitialSamples:'
    
    samples = []
    for i in range(self._BENCHMARK_INIT_SAMPLES_N):
      pos_target = float(self._POS_TARGET_MIN + np.random.sample() * (self._POS_TARGET_MAX - self._POS_TARGET_MIN))
      loc_land = self.catapult.throw_linear(pos_target)
      samples.append((pos_target, loc_land))
      logger.log('{} collect sample from true dynamics (pos_target: {}, loc_land: {})'.format(prefix_info, pos_target, loc_land))
    
    return samples
  
  def _launchModule_solveForAction_MB_CMAES(self, desired_loc_land, model_dynamics, options={}):
    """
    CMA-ES
    options['init_pos_target']: initial guess of pos_target
    options['init_var']:        initial standard deviation for CMA-ES search
    """
    prefix_info = 'catapult/solveForAction_MB_CMAES:'
    
    option_init_pos_target = options.get('init_pos_target', self._POS_TARGET_MIN + (self._POS_TARGET_MAX - self._POS_TARGET_MIN) / 2)
    option_init_var        = options.get('init_var', (self._POS_TARGET_MAX - self._POS_TARGET_MIN) / 2)
    
    init_guess = [option_init_pos_target, option_init_pos_target]
    init_var   = option_init_var
    
    self._launchModule_solveForAction_MB_CMAES_iteration = 0
    
    def f_loss(x):
      self._launchModule_solveForAction_MB_CMAES_iteration += 1
      pos_target, x_1 = x
      if self._CONFIG_CMAES_VERBOSE:
        logger.log('{} sample by CMA-ES (iteration = {}, desired_loc_land = {}, pos_target = {})'.format(
          prefix_info, self._launchModule_solveForAction_MB_CMAES_iteration, desired_loc_land, pos_target))
      prediction = model_dynamics.Predict([pos_target], with_var=True)
      loc_land_h   = (prediction.Y.ravel())[0]
      loc_land_err = (np.sqrt(np.diag(prediction.Var)))[0]
      loss = 0.5 * (desired_loc_land - loc_land_h)**2
      if self._CONFIG_CMAES_VERBOSE:
        logger.log('{} loss = {}, loc_land_h = {}, loc_land_err = {}'.format(prefix_info, loss, loc_land_h, loc_land_err))
        logger.log('')
      return loss
    
    has_finished_this_round = False
    while not has_finished_this_round:
      try:
        self._launchModule_solveForAction_MB_CMAES_iteration = 0
        res = cma.fmin(f_loss, init_guess, init_var,
                       bounds=[[self._POS_TARGET_MIN, self._POS_TARGET_MIN], [self._POS_TARGET_MAX, self._POS_TARGET_MAX]],
                       popsize=self._CONFIG_CMAES_POPSIZE, tolx=0.0001, verb_disp=False, verb_log=0)
        has_finished_this_round = True
      except:
        has_finished_this_round = False
    
    optimal_pos_target = res[0][0]
    
    return optimal_pos_target, self._launchModule_solveForAction_MB_CMAES_iteration
  
  def _launchModule_solveForAction_MB_GD(self, desired_loc_land, model_dynamics, options={}):
    """
    Gradient Descent
    options['init_pos_target']: initial guess of pos_target
    options['algorithm']: optimization algorithm for gradient descent
    - 'nms': Nelder-Mead simplex
    - 'dls': direction with line search
    """
    prefix_info = 'catapult/solveForAction_MB_GD:'
    
    option_init_pos_target = options.get('init_pos_target', self._POS_TARGET_MIN + (self._POS_TARGET_MAX - self._POS_TARGET_MIN) / 2)
    option_algorithm       = options.get('algorithm', 'nms')
    
    tolx = 0.0001
    
    init_guess = np.array([option_init_pos_target])
    self._launchModule_solveForAction_MB_GD_iteration = 0
    
    def f_loss(x):
      pos_target = x[0]
      self._launchModule_solveForAction_MB_GD_iteration += 1
      prediction = model_dynamics.Predict([pos_target], with_var=True)
      loc_land_h   = float((prediction.Y.ravel())[0])
      loc_land_err = float((np.sqrt(np.diag(prediction.Var)))[0])
      loss = 0.5 * (desired_loc_land - loc_land_h)**2
      return loss
    
    if option_algorithm == 'nms':
      res = sp_optimize.fmin(f_loss, init_guess, xtol=tolx, disp=0)
      optimal_pos_target = self._fixRange(res[0], self._POS_TARGET_MIN, self._POS_TARGET_MAX)
    
    elif option_algorithm == 'dls':
      def f_loss_grad(x):
        pos_target = x[0]
        self._launchModule_solveForAction_MB_GD_iteration += 1
        prediction = model_dynamics.Predict([pos_target], x_var=0.0**2, with_var=True, with_grad=True)
        loc_land_h    = float((prediction.Y.ravel())[0])
        loc_land_err  = float((np.sqrt(np.diag(prediction.Var)))[0])
        loc_land_grad = float(prediction.Grad.ravel()[0])
        loss_grad = loc_land_grad * (loc_land_h - desired_loc_land)
        return np.array([loss_grad])
      
      x = init_guess
      x_prev = x + tolx + 1
      
      iteration = 0
      has_diverge = False
      while np.linalg.norm(x - x_prev) >= tolx and not has_diverge:
        iteration += 1
        direction = - f_loss_grad(x)
        res = sp_optimize.line_search(f_loss, f_loss_grad, x, direction)
        if res[0] is None:
          has_diverge = True
          alpha = 1.0
          while not ((x + alpha * direction) < self._POS_TARGET_MIN or (x + alpha * direction) > self._POS_TARGET_MAX):
            alpha = alpha * 2
        else:
          has_diverge = False
          alpha = res[0]
        x_prev = x
        x = x_prev + alpha * direction
        if self._CONFIG_GD_VERBOSE:
          logger.log('{} iter = {} ({}), alpha = {}, direction = {}, x_next = {}'.format(prefix_info, iteration, self._launchModule_solveForAction_MB_GD_iteration, alpha, direction, x))
      
      optimal_pos_target = self._fixRange(x[0], self._POS_TARGET_MIN, self._POS_TARGET_MAX)
    
    else:
      assert(False)
    
    return optimal_pos_target, self._launchModule_solveForAction_MB_GD_iteration
  
  def _launchModule_solveForAction_MB_MSGD(self, desired_loc_land, model_dynamics, options={}):
    """
    Multistart Gradient Descent
    """
    prefix_info = 'catapult/solveForAction_MB_MSGD:'
    
    init_pos_target_list = self._POS_TARGET_MIN + (self._POS_TARGET_MAX - self._POS_TARGET_MIN) * np.random.sample(self._CONFIG_MSGD_STSIZE)
    self._launchModule_solveForAction_MB_MSGD_iteration = 0
    best_pos_target = None
    best_loss = None
    for i_start in range(len(init_pos_target_list)):
      init_pos_target = init_pos_target_list[i_start]
      options = {
        'init_pos_target': float(init_pos_target)
      }
      optimal_pos_target, n_iter = self._launchModule_solveForAction_MB_GD(desired_loc_land, model_dynamics, options)
      self._launchModule_solveForAction_MB_MSGD_iteration += n_iter
      
      prediction = model_dynamics.Predict([optimal_pos_target], with_var=True)
      loc_land_h   = float((prediction.Y.ravel())[0])
      loc_land_err = float((np.sqrt(np.diag(prediction.Var)))[0])
      loss = 0.5 * (desired_loc_land - loc_land_h)**2
      if best_pos_target is None:
        best_pos_target = optimal_pos_target
        best_loss = loss
      elif loss < best_loss:
        best_pos_target = optimal_pos_target
        best_loss = loss
      
      if self._CONFIG_MSGD_VERBOSE:
        logger.log('{} optimize action candidate by gradient descent (candidate: {}/{}, desired_loc_land: {}, loc_land_h: {}, loss: {}, pos_target: {} ({}), iter: {}/{})'.format(
          prefix_info, i_start+1, self._CONFIG_MSGD_STSIZE, desired_loc_land, loc_land_h, loss, optimal_pos_target, init_pos_target, n_iter, self._launchModule_solveForAction_MB_MSGD_iteration))
    
    return best_pos_target, self._launchModule_solveForAction_MB_MSGD_iteration
  
  def _launchModule_solveForAction_Hybrid_NNPolicyGD(self, desired_loc_land, model_dynamics, model_policy, options={}):
    """
    Hybrid action optimizer with gradient descent starting from initial guess 
    suggested by policy network.
    """
    prefix_info = 'catapult/solveForAction_Hybrid_NNPolicyGD:'
    
    self._launchModule_solveForAction_Hybrid_NNPolicyGD_iteration = 0
    
    # Predict initial guess by policy network
    prediction = model_policy.Predict([desired_loc_land], with_var=True)
    pos_target_h   = (prediction.Y.ravel())[0]
    pos_target_err = (np.sqrt(np.diag(prediction.Var)))[0]
    
    # Optimize action by gradient descent from initial guess
    options = {
      'init_pos_target': pos_target_h
    }
    optimal_pos_target, n_iter = self._launchModule_solveForAction_MB_GD(desired_loc_land, model_dynamics, options)
    self._launchModule_solveForAction_Hybrid_NNPolicyGD_iteration += n_iter
    if self._CONFIG_NNPOLICYGD_VERBOSE:
      logger.log('{} optimize action by NNPolicy gradient descent (init_pos_target: {}, optimal_pos_target: {}, n_iter: {})'.format(
        prefix_info, pos_target_h, optimal_pos_target, n_iter))
    
    return optimal_pos_target, self._launchModule_solveForAction_Hybrid_NNPolicyGD_iteration
  
  def _launchModule_solveForAction_Hybrid_MSGD_NN(self, desired_loc_land, model_dynamics, model_policy, options={}):
    """
    Hybrid action optimizer with multistart gradient descent and gradient 
    descent starting from initial guess suggested by policy network.
    """
    prefix_info = 'catapult/solveForAction_Hybrid_MSGD_NN:'
    
    self._launchModule_solveForAction_Hybrid_MSGD_NN_iteration = 0
    
    # Optimize action by gradient descent starting from initial guess suggested by policy network
    pos_target_candidate_policy_gd, n_iter = self._launchModule_solveForAction_Hybrid_NNPolicyGD(desired_loc_land, model_dynamics, model_policy)
    self._launchModule_solveForAction_Hybrid_MSGD_NN_iteration += n_iter
    prediction = model_dynamics.Predict([pos_target_candidate_policy_gd], with_var=True)
    loc_land_h_policy_gd   = (prediction.Y.ravel())[0]
    loc_land_err_policy_gd = (np.sqrt(np.diag(prediction.Var)))[0]
    if self._CONFIG_MSGD_NN_VERBOSE:
      logger.log('{} optimize action by gradient descent starting from initial guess suggested by policy network. (pos_target: {}, loc_land_h: {}, loc_land_err: {}, n_iter: {}/{})'.format(
        prefix_info, pos_target_candidate_policy_gd, loc_land_h_policy_gd, loc_land_err_policy_gd, n_iter, self._launchModule_solveForAction_Hybrid_MSGD_NN_iteration))
    
    # Optimize action by multistart gradient descent
    pos_target_candidate_msgd, n_iter = self._launchModule_solveForAction_MB_MSGD(desired_loc_land, model_dynamics)
    self._launchModule_solveForAction_Hybrid_MSGD_NN_iteration += n_iter
    prediction = model_dynamics.Predict([pos_target_candidate_msgd], with_var=True)
    loc_land_h_msgd   = (prediction.Y.ravel())[0]
    loc_land_err_msgd = (np.sqrt(np.diag(prediction.Var)))[0]
    if self._CONFIG_MSGD_NN_VERBOSE:
      logger.log('{} optimize action by multistart gradient descent. (pos_target: {}, loc_land_h: {}, loc_land_err: {}, n_iter: {}/{})'.format(
        prefix_info, pos_target_candidate_msgd, loc_land_h_msgd, loc_land_err_msgd, n_iter, self._launchModule_solveForAction_Hybrid_MSGD_NN_iteration))
    
    # Evaluate action candidates
    best_pos_target = pos_target_candidate_policy_gd
    if abs(desired_loc_land - loc_land_h_msgd) < abs(desired_loc_land - loc_land_h_policy_gd):
      best_pos_target = pos_target_candidate_msgd
    
    return best_pos_target, self._launchModule_solveForAction_Hybrid_MSGD_NN_iteration
  
  def _launchModule_solveForAction_Hybrid_MTLPolicyGD(self, desired_loc_land, model_dynamics, mtl_policy, options={}):
    """
    Hybrid action optimizer with gradient descent starting from initial guesses 
    suggested by multilinear policy approximators and select the best candidate.
    """
    prefix_info = 'catapult/solveForAction_Hybrid_MTLPolicyGD:'
    
    self._launchModule_solveForAction_Hybrid_MTLPolicyGD_iteration = 0
    
    # Predict initial guesses by multilinear policy approximators
    pos_target_h_candidates = mtl_policy.predictAll([desired_loc_land])
    pos_target_h_candidates = [self._fixRange(pos_target_h_candidates[i], self._POS_TARGET_MIN, self._POS_TARGET_MAX) for i in range(len(pos_target_h_candidates))]
    
    # For each candidate
    best_candidate = None
    best_loss = None
    best_pos_target = None
    best_iterations = None
    for candidate in range(len(pos_target_h_candidates)):
      pos_target_h = pos_target_h_candidates[candidate]
      
      # Optimize action by gradient descent from initial guess
      options = {
        'init_pos_target': pos_target_h
      }
      optimal_pos_target, n_iter = self._launchModule_solveForAction_MB_GD(desired_loc_land, model_dynamics, options)
      self._launchModule_solveForAction_Hybrid_MTLPolicyGD_iteration += n_iter
      if self._CONFIG_MTLPOLICYGD_VERBOSE:
        logger.log('{} optimize action candidate by multilinear policy gradient descent (init_pos_target: {}, optimal_pos_target: {}, n_iter: {}/{})'.format(
          prefix_info, pos_target_h, optimal_pos_target, n_iter, self._launchModule_solveForAction_Hybrid_MTLPolicyGD_iteration))
      
      # Evaluate action candidate
      prediction = model_dynamics.Predict([optimal_pos_target], with_var=True)
      loc_land_h   = (prediction.Y.ravel())[0]
      loc_land_err = (np.sqrt(np.diag(prediction.Var)))[0]
      loss = abs(desired_loc_land - loc_land_h)
      if (best_candidate is None) or (loss < best_loss) or (best_loss == loss and n_iter < best_iterations):
        best_candidate = candidate
        best_loss = loss
        best_pos_target = optimal_pos_target
        best_iterations = n_iter
      
    return best_pos_target, best_candidate, self._launchModule_solveForAction_Hybrid_MTLPolicyGD_iteration
  
  def _launchModule_solveForAction_Hybrid_MTL_NN(self, desired_loc_land, model_dynamics, mtl_policy, model_policy, options={}):
    """
    Hybrid action optimizer with MTLPolicy and NNPolicy gradient descent.
    """
    prefix_info = 'catapult/solveForAction_Hybrid_MTL_NN:'
    
    self._launchModule_solveForAction_Hybrid_MTL_NN_iteration = 0
    
    # Optimize action by NNPolicy gradient descent
    pos_target_candidate_policy_gd, n_iter = self._launchModule_solveForAction_Hybrid_NNPolicyGD(desired_loc_land, model_dynamics, model_policy)
    self._launchModule_solveForAction_Hybrid_MTL_NN_iteration += n_iter
    prediction = model_dynamics.Predict([pos_target_candidate_policy_gd], with_var=True)
    loc_land_h_policy_gd   = (prediction.Y.ravel())[0]
    loc_land_err_policy_gd = (np.sqrt(np.diag(prediction.Var)))[0]
    if self._CONFIG_MTL_NN_VERBOSE:
      logger.log('{} optimize action by NNPolicy gradient descent (pos_target: {}, loc_land_h: {}, loc_land_err: {}, n_iter: {}/{})'.format(
        prefix_info, pos_target_candidate_policy_gd, loc_land_h_policy_gd, loc_land_err_policy_gd, n_iter, self._launchModule_solveForAction_Hybrid_MTL_NN_iteration))
    
    # Optimize action by MTLPolicy gradient descent
    pos_target_candidate_mtl, best_candidate_mtl, n_iter = self._launchModule_solveForAction_Hybrid_MTLPolicyGD(desired_loc_land, model_dynamics, mtl_policy)
    self._launchModule_solveForAction_Hybrid_MTL_NN_iteration += n_iter
    prediction = model_dynamics.Predict([pos_target_candidate_mtl], with_var=True)
    loc_land_h_mtl   = (prediction.Y.ravel())[0]
    loc_land_err_mtl = (np.sqrt(np.diag(prediction.Var)))[0]
    if self._CONFIG_MTL_NN_VERBOSE:
      logger.log('{} optimize action by MTLPolicy gradient descent. (pos_target: {}, loc_land_h: {}, loc_land_err: {}, n_iter: {}/{})'.format(
        prefix_info, pos_target_candidate_mtl, loc_land_h_mtl, loc_land_err_mtl, n_iter, self._launchModule_solveForAction_Hybrid_MTL_NN_iteration))
    
    # Evaluate action candidates
    best_pos_target = pos_target_candidate_policy_gd
    if abs(desired_loc_land - loc_land_h_mtl) < abs(desired_loc_land - loc_land_h_policy_gd):
      best_pos_target = pos_target_candidate_mtl
    
    return best_pos_target, best_candidate_mtl, self._launchModule_solveForAction_Hybrid_MTL_NN_iteration
  
  def _launchModule_generateNNPolicyTrainingSamples(self, Y_train_dynamics, model_dynamics, model_policy):
    """
    Generate policy network training samples from dynamics model considering 
    past experiences.
    """
    prefix_info = 'catapult/generateNNPolicySamples:'
    
    X_train_policy = []
    Y_train_policy = []
    
    exp_samples_n = min(self._CONFIG_HYBRID_POLICY_EXP_SAMPLES_N, len(Y_train_dynamics))
    new_samples_n = (self._CONFIG_HYBRID_POLICY_NEW_SAMPLES_N + self._CONFIG_HYBRID_POLICY_EXP_SAMPLES_N) - exp_samples_n
    
    X_train_policy = X_train_policy + random.sample(Y_train_dynamics, exp_samples_n)
    for i in range(new_samples_n):
      x = float(self._ESTIMATED_LOC_LAND_MIN + (self._ESTIMATED_LOC_LAND_MAX - self._ESTIMATED_LOC_LAND_MIN) * np.random.sample())
      X_train_policy.append(x)
    
    total_iterations = 0
    for i in range(len(X_train_policy)):
      desired_loc_land = X_train_policy[i]
      optimal_pos_target, n_iter = self._launchModule_solveForAction_Hybrid_NNPolicyGD(desired_loc_land, model_dynamics, model_policy)
      total_iterations += n_iter
      Y_train_policy.append(optimal_pos_target)
      if self._CONFIG_GENERATE_NNPOLICY_SAMPLES_VERBOSE:
        logger.log('{} generate policy network training sample (sample: {}/{}, desired_loc_land: {}, optimal_pos_target: {}, n_iter: {}/{})'.format(
          prefix_info, i+1, len(X_train_policy), desired_loc_land, optimal_pos_target, n_iter, total_iterations))
    
    samples = []
    for i in range(len(X_train_policy)):
      samples.append((X_train_policy[i], Y_train_policy[i]))
    
    return samples, total_iterations
  
  def _launchModule_reinitializeMTLPolicyApproximators(self, mtl_policy, mtl_policy_inactive, model_dynamics):
    """
    Reinitialize similar and inactive approximators in multilinear policy
    """
    prefix_info = 'catapult/reinitializeMTLPolicyApproximators:'
    
    # Reinitialize similar approximators
    #TODO: how to find intersection subspace
    
    # Reinitialize inactive approximators
    for approximatorIndex in range(len(mtl_policy)):
      mtl_policy_inactive[approximatorIndex] += 1
      
      if mtl_policy_inactive[approximatorIndex] > len(mtl_policy):
        pos_target_sample = self._POS_TARGET_MIN + (self._POS_TARGET_MAX - self._POS_TARGET_MIN) * np.random.sample()
        self._updateLocalMTLPolicyForApproximator(mtl_policy, approximatorIndex, model_dynamics, pos_target_sample)
        mtl_policy_inactive[approximatorIndex] = int(0.5 * len(mtl_policy))
        logger.log('{} reinitialize multilinear policy approximator with local samples (approximator: {}/{}, mean: {})'.format(
          prefix_info, approximatorIndex+1, len(mtl_policy), pos_target_sample))
  
  def launchApproach_MB_CMAES(self):
    """
    Model-based, CMA-ES(action)
    """
    prefix_info = 'catapult/MB_CMAES:'
    
    X_train_dynamics = []
    Y_train_dynamics = []
    trialResults = []
    
    # Train dynamics model with initial random samples
    logger.log('{} collect initial random samples'.format(prefix_info))
    model_dynamics = self._createNNDynamics()
    samples_dynamics = self._launchModule_collectInitialDynamicsSamples()
    for sample in samples_dynamics:
      pos_target, loc_land = sample
      X_train_dynamics.append(pos_target)
      Y_train_dynamics.append(loc_land)
    logger.log('{} train dynamics model with initial random samples'.format(prefix_info))
    self._trainNNDynamics(model_dynamics, samples_dynamics)
    
    # For each episode
    logger.log('{} generate test samples'.format(prefix_info))
    samples_desired_loc_land = self._launchModule_generateTestSamples()
    for episode in range(len(samples_desired_loc_land)):
      desired_loc_land = samples_desired_loc_land[episode]
      logger.log('{} episodic test (episode: {}, desired_loc_land: {})'.format(prefix_info, episode+1, desired_loc_land))
      
      # Optimize action by CMA-ES
      optimal_pos_target, n_iter = self._launchModule_solveForAction_MB_CMAES(desired_loc_land, model_dynamics)
      logger.log('{} optimize action by CMA-ES (desired_loc_land: {}, optimal_pos_target: {}, n_iter: {})'.format(
        prefix_info, desired_loc_land, optimal_pos_target, n_iter))
      
      # Test in true dynamics
      pos_target = optimal_pos_target
      loc_land = self.catapult.throw_linear(pos_target)
      logger.log('{} test in true dynamics (desired_loc_land: {}, loc_land: {}, pos_target: {})'.format(prefix_info, desired_loc_land, loc_land, pos_target))
      
      # Train dynamics model
      X_train_dynamics.append(pos_target)
      Y_train_dynamics.append(loc_land)
      samples = [(pos_target, loc_land)]
      logger.log('{} train dynamics model (samples: {})'.format(prefix_info, samples))
      self._trainNNDynamics(model_dynamics, samples)
      
      # Add trial result entry
      entry = {
        'approach':           str('Model-based, CMA-ES(action)'),
        'desired_loc_land':   float(desired_loc_land),
        'pos_target':         float(pos_target),
        'loc_land':           float(loc_land),
        'preopt_samples':     int(self._BENCHMARK_INIT_SAMPLES_N),
        'preopt_simulations': int(0),
        'samples':            int(episode),
        'simulations':        int(n_iter)
      }
      trialResults.append(entry)
      logger.log('{} add trial result entry >>>'.format(prefix_info))
      logger.log(entry)
      
      logger.log('')
    
    # Evaluate
    self._evaluateNNDynamics(model_dynamics, X_train_dynamics, Y_train_dynamics)
    self._evaluateTrialResults(trialResults)

  def launchApproach_MB_MSGD(self):
    """
    Model-based, Multistart-GD(action)
    """
    prefix_info = 'catapult/MB_MSGD:'
    
    X_train_dynamics = []
    Y_train_dynamics = []
    trialResults = []
    
    # Train dynamics model with initial random samples
    logger.log('{} collect initial random samples'.format(prefix_info))
    model_dynamics = self._createNNDynamics()
    samples_dynamics = self._launchModule_collectInitialDynamicsSamples()
    for sample in samples_dynamics:
      pos_target, loc_land = sample
      X_train_dynamics.append(pos_target)
      Y_train_dynamics.append(loc_land)
    logger.log('{} train dynamics model with initial random samples'.format(prefix_info))
    self._trainNNDynamics(model_dynamics, samples_dynamics)
    
    # For each episode
    logger.log('{} generate test samples'.format(prefix_info))
    samples_desired_loc_land = self._launchModule_generateTestSamples()
    for episode in range(len(samples_desired_loc_land)):
      desired_loc_land = samples_desired_loc_land[episode]
      logger.log('{} episodic test (episode: {}, desired_loc_land: {})'.format(prefix_info, episode+1, desired_loc_land))
      
      # Optimize action by multistart gradient descent
      optimal_pos_target, n_iter = self._launchModule_solveForAction_MB_MSGD(desired_loc_land, model_dynamics)
      logger.log('{} optimize action by multistart gradient descent (desired_loc_land: {}, optimal_pos_target: {}, n_iter: {})'.format(
        prefix_info, desired_loc_land, optimal_pos_target, n_iter))
      
      # Test in true dynamics
      pos_target = optimal_pos_target
      loc_land = self.catapult.throw_linear(pos_target)
      logger.log('{} test in true dynamics (desired_loc_land: {}, loc_land: {}, pos_target: {})'.format(prefix_info, desired_loc_land, loc_land, pos_target))
      
      # Train dynamics model
      X_train_dynamics.append(pos_target)
      Y_train_dynamics.append(loc_land)
      samples = [(pos_target, loc_land)]
      logger.log('{} train dynamics model (samples: {})'.format(prefix_info, samples))
      self._trainNNDynamics(model_dynamics, samples)
      
      # Add trial result entry
      entry = {
        'approach':           str('Model-based, Multistart-GD(action)'),
        'desired_loc_land':   float(desired_loc_land),
        'pos_target':         float(pos_target),
        'loc_land':           float(loc_land),
        'preopt_samples':     int(self._BENCHMARK_INIT_SAMPLES_N),
        'preopt_simulations': int(0),
        'samples':            int(episode),
        'simulations':        int(n_iter)
      }
      trialResults.append(entry)
      logger.log('{} add trial result entry >>>'.format(prefix_info))
      logger.log(entry)
      
      logger.log('')
    
    # Evaluate
    self._evaluateNNDynamics(model_dynamics, X_train_dynamics, Y_train_dynamics)
    self._evaluateTrialResults(trialResults)

  def launchApproach_Hybrid_MSGD_NN(self):
    """
    Hybrid, MSGD-MB, NN-MF
    - {(d*, Multistart-GD(action -> d))} -> NNPolicy,
    - action <- GD(NNPolicy(d*) -> d) + Multistart-GD(action -> d)
    """
    prefix_info = 'catapult/Hybrid_MSGD_NN:'
    
    X_train_dynamics = []
    Y_train_dynamics = []
    X_train_policy = []
    Y_train_policy = []
    trialResults = []
    
    # Train dynamics model with initial random samples
    logger.log('{} collect initial random samples'.format(prefix_info))
    model_dynamics = self._createNNDynamics()
    samples_dynamics = self._launchModule_collectInitialDynamicsSamples()
    for sample in samples_dynamics:
      pos_target, loc_land = sample
      X_train_dynamics.append(pos_target)
      Y_train_dynamics.append(loc_land)
    logger.log('{} train dynamics model with initial random samples'.format(prefix_info))
    self._trainNNDynamics(model_dynamics, samples_dynamics)
    
    # Create policy network
    model_policy = self._createNNPolicy()
    
    # For each episode
    logger.log('{} generate test samples'.format(prefix_info))
    samples_desired_loc_land = self._launchModule_generateTestSamples()
    for episode in range(len(samples_desired_loc_land)):
      desired_loc_land = samples_desired_loc_land[episode]
      logger.log('{} episodic test (episode: {}, desired_loc_land: {})'.format(prefix_info, episode+1, desired_loc_land))
      
      result_preopt_simulations = 0
      result_simulations = 0
      
      # Generate policy network training samples
      X_train_policy = []
      Y_train_policy = []
      samples, n_iter = self._launchModule_generateNNPolicyTrainingSamples(Y_train_dynamics, model_dynamics, model_policy)
      for sample in samples:
        x, y = sample
        X_train_policy.append(x)
        Y_train_policy.append(y)
      result_preopt_simulations += n_iter
      logger.log('{} generated policy network training samples with model-based optimization (samples: {}, n_iter: {})'.format(
        prefix_info, len(samples), n_iter))
      
      # Train policy network with generate training samples
      logger.log('{} train policy network with generate training samples'.format(prefix_info))
      self._trainNNPolicy(model_policy, samples, flush=True)
      
      # Optimize action by hybrid NNPolicy and MSGD
      optimal_pos_target, n_iter = self._launchModule_solveForAction_Hybrid_MSGD_NN(desired_loc_land, model_dynamics, model_policy)
      result_simulations += n_iter
      logger.log('{} optimize action by hybrid NNPolicy and MSGD (desired_loc_land: {}, optimal_pos_target: {}, n_iter: {})'.format(
        prefix_info, desired_loc_land, optimal_pos_target, n_iter))
      
      # Test in true dynamics
      pos_target = optimal_pos_target
      loc_land = self.catapult.throw_linear(pos_target)
      logger.log('{} test in true dynamics (desired_loc_land: {}, loc_land: {}, pos_target: {})'.format(prefix_info, desired_loc_land, loc_land, pos_target))
      
      # Train dynamics model
      X_train_dynamics.append(pos_target)
      Y_train_dynamics.append(loc_land)
      samples = [(pos_target, loc_land)]
      logger.log('{} train dynamics model (samples: {})'.format(prefix_info, samples))
      self._trainNNDynamics(model_dynamics, samples)
      
      # Add trial result entry
      entry = {
        'approach':           str('Hybrid, MSGD-MB, NN-MF'),
        'desired_loc_land':   float(desired_loc_land),
        'pos_target':         float(pos_target),
        'loc_land':           float(loc_land),
        'preopt_samples':     int(self._BENCHMARK_INIT_SAMPLES_N),
        'preopt_simulations': int(result_preopt_simulations),
        'samples':            int(episode),
        'simulations':        int(result_simulations)
      }
      trialResults.append(entry)
      logger.log('{} add trial result entry >>>'.format(prefix_info))
      logger.log(entry)
      
      logger.log('')
    
    # Evaluate
    self._evaluateNNDynamics(model_dynamics, X_train_dynamics, Y_train_dynamics)
    self._evaluateNNPolicy(model_dynamics, model_policy, X_train_policy, Y_train_policy)
    self._evaluateTrialResults(trialResults)

  def launchApproach_Hybrid_MTL(self):
    """
    Hybrid, MTL-MF
    - Initialize multilinear policy with Gaussian random samples in action space
    - Update the best fit approximator at each episode with local samples
    - Reinitialize similar or inactive policy approximators
    """
    prefix_info = 'catapult/Hybrid_MTL:'
    
    X_train_dynamics = []
    Y_train_dynamics = []
    trialResults = []
    
    # Train dynamics model with initial random samples
    logger.log('{} collect initial random samples'.format(prefix_info))
    model_dynamics = self._createNNDynamics()
    samples_dynamics = self._launchModule_collectInitialDynamicsSamples()
    for sample in samples_dynamics:
      pos_target, loc_land = sample
      X_train_dynamics.append(pos_target)
      Y_train_dynamics.append(loc_land)
    logger.log('{} train dynamics model with initial random samples'.format(prefix_info))
    self._trainNNDynamics(model_dynamics, samples_dynamics)
    
    # Train multilinear policy approximators with local samples
    mtl_policy, mtl_policy_inactive = self._createMTLPolicy(model_dynamics)
    
    # For each episode
    logger.log('{} generate test samples'.format(prefix_info))
    samples_desired_loc_land = self._launchModule_generateTestSamples()
    for episode in range(len(samples_desired_loc_land)):
      desired_loc_land = samples_desired_loc_land[episode]
      logger.log('{} episodic test (episode: {}, desired_loc_land: {})'.format(prefix_info, episode+1, desired_loc_land))
      
      result_simulations = 0
      
      # Optimize action by MTLPolicy gradient descent
      optimal_pos_target, approximatorIndex, n_iter = self._launchModule_solveForAction_Hybrid_MTLPolicyGD(desired_loc_land, model_dynamics, mtl_policy)
      mtl_policy_inactive[approximatorIndex] = 0
      result_simulations += n_iter
      logger.log('{} optimize action by MTLPolicy gradient descent (desired_loc_land: {}, optimal_pos_target: {}, approximator: {}, n_iter: {})'.format(
        prefix_info, desired_loc_land, optimal_pos_target, approximatorIndex, n_iter))
      
      # Update multilinear policy approximator with local samples
      self._updateLocalMTLPolicyForApproximator(mtl_policy, approximatorIndex, model_dynamics, optimal_pos_target)
      logger.log('{} update multilinear policy approximator with local samples (approximator: {}, mean: {})'.format(
        prefix_info, approximatorIndex, optimal_pos_target))
      
      # Reinitialize similar and inactive approximators
      self._launchModule_reinitializeMTLPolicyApproximators(mtl_policy, mtl_policy_inactive, model_dynamics)
      
      # Test in true dynamics
      pos_target = optimal_pos_target
      loc_land = self.catapult.throw_linear(pos_target)
      logger.log('{} test in true dynamics (desired_loc_land: {}, loc_land: {}, pos_target: {})'.format(prefix_info, desired_loc_land, loc_land, pos_target))
      
      # Train dynamics model
      X_train_dynamics.append(pos_target)
      Y_train_dynamics.append(loc_land)
      samples = [(pos_target, loc_land)]
      logger.log('{} train dynamics model (samples: {})'.format(prefix_info, samples))
      self._trainNNDynamics(model_dynamics, samples)
      
      # Add trial result entry
      entry = {
        'approach':           str('Hybrid, MTL-MF'),
        'desired_loc_land':   float(desired_loc_land),
        'pos_target':         float(pos_target),
        'loc_land':           float(loc_land),
        'preopt_samples':     int(self._BENCHMARK_INIT_SAMPLES_N),
        'preopt_simulations': int(0),
        'samples':            int(episode),
        'simulations':        int(result_simulations)
      }
      trialResults.append(entry)
      logger.log('{} add trial result entry >>>'.format(prefix_info))
      logger.log(entry)
      
      logger.log('')
    
    # Evaluate
    self._evaluateNNDynamics(model_dynamics, X_train_dynamics, Y_train_dynamics)
    self._evaluateMTLPolicy(model_dynamics, mtl_policy)
    self._evaluateTrialResults(trialResults)

  def launchApproach_Hybrid_MTL_NN(self):
    """
    Hybrid, MTL-MF, NN-MF
    """
    prefix_info = 'catapult/Hybrid_MTL_NN:'
    
    X_train_dynamics = []
    Y_train_dynamics = []
    X_train_policyNN = []
    Y_train_policyNN = []
    trialResults = []
    
    # Train dynamics model with initial random samples
    logger.log('{} collect initial random samples'.format(prefix_info))
    model_dynamics = self._createNNDynamics()
    samples_dynamics = self._launchModule_collectInitialDynamicsSamples()
    for sample in samples_dynamics:
      pos_target, loc_land = sample
      X_train_dynamics.append(pos_target)
      Y_train_dynamics.append(loc_land)
    logger.log('{} train dynamics model with initial random samples'.format(prefix_info))
    self._trainNNDynamics(model_dynamics, samples_dynamics)
    
    # Train multilinear policy approximators with local samples
    mtl_policy, mtl_policy_inactive = self._createMTLPolicy(model_dynamics)
    
    # Create policy network
    model_policy = self._createNNPolicy()
    
    # For each episode
    logger.log('{} generate test samples'.format(prefix_info))
    samples_desired_loc_land = self._launchModule_generateTestSamples()
    for episode in range(len(samples_desired_loc_land)):
      desired_loc_land = samples_desired_loc_land[episode]
      logger.log('{} episodic test (episode: {}, desired_loc_land: {})'.format(prefix_info, episode+1, desired_loc_land))
      
      result_preopt_simulations = 0
      result_simulations = 0
      
      # Generate policy network training samples
      X_train_policyNN = []
      Y_train_policyNN = []
      samples, n_iter = self._launchModule_generateNNPolicyTrainingSamples(Y_train_dynamics, model_dynamics, model_policy)
      for sample in samples:
        x, y = sample
        X_train_policyNN.append(x)
        Y_train_policyNN.append(y)
      result_preopt_simulations += n_iter
      logger.log('{} generated policy network training samples with model-based optimization (samples: {}, n_iter: {})'.format(
        prefix_info, len(samples), n_iter))
      
      # Train policy network with generate training samples
      logger.log('{} train policy network with generate training samples'.format(prefix_info))
      self._trainNNPolicy(model_policy, samples, flush=True)
      
      # Optimize action by hybrid NNPolicy and MTLPolicy gradient descent
      optimal_pos_target, approximatorIndex, n_iter = self._launchModule_solveForAction_Hybrid_MTL_NN(desired_loc_land, model_dynamics, mtl_policy, model_policy)
      mtl_policy_inactive[approximatorIndex] = 0
      result_simulations += n_iter
      logger.log('{} optimize action by hybrid NNPolicy and MTLPolicy gradient descent (desired_loc_land: {}, optimal_pos_target: {}, n_iter: {})'.format(
        prefix_info, desired_loc_land, optimal_pos_target, n_iter))
      
      # Update multilinear policy approximator with local samples
      self._updateLocalMTLPolicyForApproximator(mtl_policy, approximatorIndex, model_dynamics, optimal_pos_target)
      logger.log('{} update multilinear policy approximator with local samples (approximator: {}, mean: {})'.format(
        prefix_info, approximatorIndex, optimal_pos_target))
      
      # Reinitialize similar and inactive approximators
      self._launchModule_reinitializeMTLPolicyApproximators(mtl_policy, mtl_policy_inactive, model_dynamics)
      
      # Test in true dynamics
      pos_target = optimal_pos_target
      loc_land = self.catapult.throw_linear(pos_target)
      logger.log('{} test in true dynamics (desired_loc_land: {}, loc_land: {}, pos_target: {})'.format(prefix_info, desired_loc_land, loc_land, pos_target))
      
      # Train dynamics model
      X_train_dynamics.append(pos_target)
      Y_train_dynamics.append(loc_land)
      samples = [(pos_target, loc_land)]
      logger.log('{} train dynamics model (samples: {})'.format(prefix_info, samples))
      self._trainNNDynamics(model_dynamics, samples)
      
      # Add trial result entry
      entry = {
        'approach':           str('Hybrid, MTL-MF, NN-MF'),
        'desired_loc_land':   float(desired_loc_land),
        'pos_target':         float(pos_target),
        'loc_land':           float(loc_land),
        'preopt_samples':     int(self._BENCHMARK_INIT_SAMPLES_N),
        'preopt_simulations': int(result_preopt_simulations),
        'samples':            int(episode),
        'simulations':        int(result_simulations)
      }
      trialResults.append(entry)
      logger.log('{} add trial result entry >>>'.format(prefix_info))
      logger.log(entry)
      
      logger.log('')
    
    # Evaluate
    self._evaluateNNDynamics(model_dynamics, X_train_dynamics, Y_train_dynamics)
    self._evaluateMTLPolicy(model_dynamics, mtl_policy)
    self._evaluateNNPolicy(model_dynamics, model_policy, X_train_policyNN, Y_train_policyNN)
    self._evaluateTrialResults(trialResults)


if __name__ == '__main__':
  approach = None
  trials = None
  if len(sys.argv) == 3:
    approach = str(sys.argv[1]).strip().lower()
    rounds = int(sys.argv[2])
  else:
    print('usage: ./run_simNN_001_linear_thetaT_d <approach> <rounds>')
    quit()
  
  abs_dirpath_data = os.path.abspath('../data')
  
  catapult = TCatapultSimNN1D('simNN_001', abs_dirpath_data)
  
  for i_round in range(rounds):
    logger.log('ROUND: {}/{}'.format(i_round+1, rounds))
    agent = TCatapultLPLinearSimNN(catapult, abs_dirpath_data)
    
    if approach == 'mb_cmaes':
      agent.launchApproach_MB_CMAES()
    elif approach == 'mb_msgd':
      agent.launchApproach_MB_MSGD()
    elif approach == 'hybrid_msgd_nn':
      agent.launchApproach_Hybrid_MSGD_NN()
    elif approach == 'hybrid_mtl':
      agent.launchApproach_Hybrid_MTL()
    elif approach == 'hybrid_mtl_nn':
      agent.launchApproach_Hybrid_MTL_NN()
    else:
      print('ERROR: invalid approach')
      quit()


