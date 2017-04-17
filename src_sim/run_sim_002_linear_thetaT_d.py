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


from libLoggerSim import TLoggerSim
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

logger = TLoggerSim(fp_log='run_sim_002_linear_thetaT_d.log')



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
  
  def _create_model(self, input_dim, output_dim, hiddens=[200, 200], max_updates=20000, should_load_model=False, prefix_info='catapult'):
    model = TNNRegression()
    
    options = {
      #'AdaDelta_rho':         0.5, # 0.5, 0.9
      #'dropout':              True,
      #'dropout_ratio':        0.01,
      'loss_stddev_stop':     1.0e-4,
      'loss_stddev_stop_err': 1.0e-6,
      'batchsize':            100,
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
    logger.log('{} save mode. (dirpath={})'.format(prefix_info, self._abs_dirpath_model))
    SaveYAML(model.Save(self._abs_dirpath_model), self._abs_dirpath_model + 'nn_model.yaml')
  
  def _load_model(self, model, prefix_info):
    logger.log('{} load mode. (dirpath={})'.format(prefix_info, self._abs_dirpath_model))
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
    prefix = 'catapult_sim/model_based'
    prefix_info = prefix + ':'
    
    # Create model (and train/load)
    should_train_model = True
    should_train_model_input = input('{} train model rather than load from model file (Y/n)?> '.format(prefix_info)).strip().lower()
    if should_train_model_input in ['', 'y']:
      should_train_model = True
    else:
      should_train_model = False
    should_load_model = (not should_train_model)
    model = self._create_model(1, 1, hiddens=[200, 200], max_updates=40000, should_load_model=should_load_model, prefix_info=prefix_info)
    
    # Train model
    x_train = []
    y_train = []
    for entry in self._dataset:
      x_train.append([entry['action']['pos_target']])
      y_train.append([entry['result']['loc_land']])
    if not should_load_model:
      self._train_model(model, x_train, y_train, batch_train=True)
    
    # Estimate model quality
    should_estimate_model_quality_input = input('{} estimate model quality (Y/n)?> '.format(prefix_info)).strip().lower()
    if should_estimate_model_quality_input in ['', 'y']:
      ave_stderr_y, ave_stderr_err = self._estimate_model_quality(model, x_train, y_train, x_train, y_train, should_plot=True)
      logger.log('{} ave_stderr_y = {}, ave_stderr_err = {}'.format(prefix_info, ave_stderr_y, ave_stderr_err))
    
    # Save model
    if not should_load_model:
      should_save_model_input = input('{} save model (Y/n)?> '.format(prefix_info)).strip().lower()
      if should_save_model_input in ['', 'y']:
        self._save_model(model, prefix_info)
    
    # Query desired landing location
    desired_loc_land_input = input('{} desired_loc_land = '.format(prefix_info)).strip().lower()
    desired_loc_land = float(desired_loc_land_input)
    
    # Optimize parameters with CMA-ES
    init_guess = [0.1, 0.1]
    init_var   = 1.0
    self._run_model_based_iteration = 0
    def f(x):
      logger.log('{} optimization with CMA-ES. (iteration = {}, desired_loc_land = {})'.format(prefix_info, self._run_model_based_iteration, desired_loc_land))
      self._run_model_based_iteration += 1
      pos_target, x_1 = x
      logger.log('{} sample from CMA-ES. (pos_target = {})'.format(prefix_info, pos_target))
      prediction = model.Predict([pos_target], x_var=0.0**2, with_var=True, with_grad=True)
      loc_land_h    = prediction.Y.ravel()
      loc_land_err  = np.sqrt(np.diag(prediction.Var))
      loc_land_grad = prediction.Grad.ravel()
      loss = 0.5 * np.sqrt(desired_loc_land - loc_land_h)
      logger.log('{} loss = {}, loc_land_h = {}, loc_land_err = {}'.format(prefix_info, loss, loc_land_h, loc_land_err))
      logger.log('')
      return loss
    res = cma.fmin(f, init_guess, init_var,
                   bounds=[[self.catapult.POS_MIN, self.catapult.POS_MIN], [self.catapult.POS_MAX, self.catapult.POS_MAX]], 
                   popsize=20, tolx=0.0001, verb_disp=False, verb_log=0)
    logger.log('{} result = {}'.format(prefix_info, res))
    logger.log('{} optimal solution found. (pos_target = {}, pos_init === {}, duration === {})'.format(prefix_info, res[0][0], self._FIXED_POS_INIT, self._FIXED_DURATION))
    optimal_pos_target = res[0][0]
    logger.log('')
    
    # Test in true dynamics
    logger.log('{} test in true dynamics. (pos_init = {}, pos_target = {}, duration = {})'.format(prefix_info, self._FIXED_POS_INIT, optimal_pos_target, self._FIXED_DURATION))
    loc_land = catapult.throw_linear(self._FIXED_POS_INIT, optimal_pos_target, self._FIXED_DURATION)
    logger.log('{} loc_land = {}, desired_loc_land = {}'.format(prefix_info, loc_land, desired_loc_land))
  
  def _penalize_action(self, pos_init, pos_target, duration):
    prefix = 'catapult_sim/penalize_action'
    prefix_info = prefix + ':'
    
    corrected_pos_init   = pos_init
    corrected_pos_target = pos_target
    corrected_duration   = duration
    
    penalty = 0
    penalty_factor = 10
    
    min_pos_diff = 0.1 * math.pi
    
    if pos_init < self.catapult.POS_MIN:
      cur_penalty = np.abs(pos_init - self.catapult.POS_MIN) * penalty_factor
      penalty += cur_penalty
      logger.log('{} penalty = {} ({})'.format(prefix_info, cur_penalty, 'pos_init < self.catapult.POS_MIN'))
      corrected_pos_init = self.catapult.POS_MIN
    if pos_init > self.catapult.POS_MID:
      cur_penalty = np.abs(pos_init - self.catapult.POS_MID) * penalty_factor
      penalty += cur_penalty
      logger.log('{} penalty = {} ({})'.format(prefix_info, cur_penalty, 'pos_init > self.catapult.POS_MID'))
      corrected_pos_init = self.catapult.POS_MID
    
    if pos_target < (corrected_pos_init + min_pos_diff):
      cur_penalty = np.abs(pos_target - (corrected_pos_init + min_pos_diff)) * penalty_factor
      penalty += cur_penalty
      logger.log('{} penalty = {} ({})'.format(prefix_info, cur_penalty, 'pos_target <= (corrected_pos_init + min_pos_diff)'))
      corrected_pos_target = (corrected_pos_init + min_pos_diff)
    if pos_target > self.catapult.POS_MAX:
      cur_penalty = np.abs(pos_target - self.catapult.POS_MAX) * penalty_factor
      penalty += cur_penalty
      logger.log('{} penalty = {} ({})'.format(prefix_info, cur_penalty, 'pos_target > self.catapult.POS_MAX'))
      corrected_pos_target = self.catapult.POS_MAX
    
    if duration < self.catapult.DURATION_MIN:
      cur_penalty = np.abs(duration - self.catapult.DURATION_MIN) * penalty_factor
      penalty += cur_penalty
      logger.log('{} penalty = {} ({})'.format(prefix_info, cur_penalty, 'duration < self.catapult.DURATION_MIN'))
      corrected_duration = self.catapult.DURATION_MIN
    if duration > 0.6:
      cur_penalty = np.abs(duration - 0.6) * penalty_factor
      penalty += cur_penalty
      logger.log('{} penalty = {} ({})'.format(prefix_info, cur_penalty, 'duration > 0.6'))
      corrected_duration = 0.6
    
    return corrected_pos_init, corrected_pos_target, corrected_duration, penalty
  
  def _launch_test(self, dataset, pos_init, pos_target, duration):
    loc_land = catapult.throw_linear(pos_init, pos_target, duration)
    entry = dataset.new_entry_linear_sim(float(pos_init), float(pos_target), float(duration), float(loc_land))
    dataset.append(entry)
    return entry
  
  def _run_model_free(self):
    prefix = 'catapult_sim/model_free'
    prefix_info = prefix + ':'
    
    # initialize dataset
    saver_dataset = TCatapultDatasetSim(abs_dirpath=self._abs_dirpath_data, auto_init=False)
    saver_dataset.init_yaml()
    
    # define policy function with parameters
    def policy_func(desired_loc_land, params):
      x = desired_loc_land
      p = params
      y = 0
      for i in range(4):
        y += p[i] * x**i
      return y
    
    # specify desired landing location sample points
    #desired_loc_land_samples = [(0 + 2.5*i) for i in range(11)] # full sampling
    desired_loc_land_samples = [0.0, 2.5, 5.0, 7.5, 10.0, 17.5, 20.0, 22.5, 25.0] # ignore 12.5, 15.0 (keep 17.5)
    N = len(desired_loc_land_samples)

    # define loss function
    self._run_model_free_iteration = 0
    def loss_func(params, desired_loc_land_samples, N):
      logger.log('{} optimize policy parameters with CMA-ES. (iteration = {}, N = {})'.format(prefix_info, self._run_model_free_iteration, N))
      self._run_model_free_iteration += 1
      logger.log('{} sample from CMA-ES. (params = {})'.format(prefix_info, params))
      pos_target_hypos = [policy_func(desired_loc_land_samples[i], params) for i in range(N)]
      logger.log('{} generate hypo target positions. (hypos = {})'.format(prefix_info, pos_target_hypos))
      sum_loss = 0
      sum_penalty = 0
      for i in range(N):
        pos_init, pos_target, duration, penalty = self._penalize_action(self._FIXED_POS_INIT, pos_target_hypos[i], self._FIXED_DURATION)
        logger.log('{} test target position hypotheses suggested by current policy parameters. (sample = {}/{})'.format(prefix_info, i+1, N))
        logger.log('{} launch test. (pos_init = {}, pos_target = {}, duration = {})'.format(prefix_info, pos_init, pos_target, duration))
        entry = self._launch_test(saver_dataset, pos_init, pos_target, duration)
        loc_land_i = float(entry['result']['loc_land'])
        logger.log('{} loc_land = {}, desired_loc_land = {}'.format(prefix_info, loc_land_i, desired_loc_land_samples[i]))
        sum_penalty += penalty
        sum_loss += (desired_loc_land_samples[i] - loc_land_i)**2
      ave_loss = sum_loss / (2 * N)
      ave_penalty = sum_penalty / (2 * N)
      final_loss = ave_loss + ave_penalty
      logger.log('{} loss = {}, penalty = {}'.format(prefix_info, final_loss, ave_penalty))
      logger.log('')
      return final_loss

    # optimize policy parameters with CMA-ES
    #init_guess = [0.38544, 0.10898, -0.00605, 0.00015] # for duration = 0.10, pos_init = 0.0, full sampling
    #init_var   = 0.00100 # for duration = 0.10, pos_init = 0.0, full sampling
    init_guess = [3.84310955e-01, 1.07871025e-01, -5.57357436e-03, 1.17338141e-04] # for duration = 0.10, pos_init = 0.0, ignore 12.5, 15.0 (keep 17.5)
    init_var   = 0.00020 # for duration = 0.10, pos_init = 0.0, ignore 12.5, 15.0 (keep 17.5)
    res = cma.fmin(loss_func, init_guess, init_var, args=(desired_loc_land_samples, N), 
                   popsize=20, tolx=10e-6, verb_disp=False, verb_log=0)
    optimal_params = res[0]
    #optimal_params = [3.84177005e-01,   1.07917544e-01,  -5.55294093e-03,   1.14927976e-04]
    logger.log('{} result = {}'.format(prefix_info, res))
    logger.log('{} optimal solution found. (params = {})'.format(prefix_info, optimal_params))
    logger.log('')

    # Query desired landing location
    desired_loc_land_input = input('{} desired_loc_land = '.format(prefix_info)).strip().lower()
    desired_loc_land = float(desired_loc_land_input)

    # test policy parameters in true dynamics
    logger.log('{} apply optimal policy parameters. (params = {})'.format(prefix_info, optimal_params))
    pos_target_hypo = policy_func(desired_loc_land, optimal_params)
    logger.log('{} predict action by parameterized policy. (desired_loc_land = {}, pos_target_hypo = {})'.format(prefix_info, desired_loc_land, pos_target_hypo))
    logger.log('{} test in true dynamics. (pos_init = {}, pos_target = {}, duration = {})'.format(prefix_info, self._FIXED_POS_INIT, pos_target_hypo, self._FIXED_DURATION))
    loc_land = catapult.throw_linear(self._FIXED_POS_INIT, pos_target_hypo, self._FIXED_DURATION)
    logger.log('{} loc_land = {}, desired_loc_land = {}'.format(prefix_info, loc_land, desired_loc_land))

  def _run_hybrid(self):
    prefix = 'catapult_sim/model_hybrid'
    prefix_info = prefix + ':'
    
    # Create model (and train/load)
    should_train_model = True
    should_train_model_input = input('{} train model rather than load from model file (Y/n)?> '.format(prefix_info)).strip().lower()
    if should_train_model_input in ['', 'y']:
      should_train_model = True
    else:
      should_train_model = False
    should_load_model = (not should_train_model)
    model = self._create_model(1, 1, hiddens=[200, 200], max_updates=40000, should_load_model=should_load_model, prefix_info=prefix_info)
    
    # Train model
    x_train = []
    y_train = []
    for entry in self._dataset:
      x_train.append([entry['action']['pos_target']])
      y_train.append([entry['result']['loc_land']])
    if not should_load_model:
      self._train_model(model, x_train, y_train, batch_train=True)
    
    # Estimate model quality
    should_estimate_model_quality_input = input('{} estimate model quality (Y/n)?> '.format(prefix_info)).strip().lower()
    if should_estimate_model_quality_input in ['', 'y']:
      ave_stderr_y, ave_stderr_err = self._estimate_model_quality(model, x_train, y_train, x_train, y_train, should_plot=True)
      logger.log('{} ave_stderr_y = {}, ave_stderr_err = {}'.format(prefix_info, ave_stderr_y, ave_stderr_err))
    
    # Save model
    if not should_load_model:
      should_save_model_input = input('{} save model (Y/n)?> '.format(prefix_info)).strip().lower()
      if should_save_model_input in ['', 'y']:
        self._save_model(model, prefix_info)
    
    # define policy function with parameters
    def policy_func(desired_loc_land, params):
      x = desired_loc_land
      p = params
      y = 0
      for i in range(4):
        y += p[i] * x**i
      return y

    # specify desired landing location sample points
    #desired_loc_land_samples = [(0 + 2.5*i) for i in range(11)] # full sampling
    desired_loc_land_samples = [0.0, 2.5, 5.0, 7.5, 10.0, 17.5, 20.0, 22.5, 25.0] # ignore 12.5, 15.0 (keep 17.5)
    N = len(desired_loc_land_samples)

    # define loss function
    self._run_hybrid_mf_policy_iteration = 0
    def mf_policy_loss_func(params, desired_loc_land_samples, N):
      logger.log('{} optimize policy parameters with CMA-ES. (iteration = {}, N = {})'.format(prefix_info, self._run_hybrid_mf_policy_iteration, N))
      self._run_hybrid_mf_policy_iteration += 1
      logger.log('{} sample from CMA-ES. (params = {})'.format(prefix_info, params))
      pos_target_hypos = [policy_func(desired_loc_land_samples[i], params) for i in range(N)]
      logger.log('{} generate hypo target positions. (hypos = {})'.format(prefix_info, pos_target_hypos))
      sum_loss = 0
      sum_penalty = 0
      for i in range(N):
        pos_init, pos_target, duration, penalty = self._penalize_action(self._FIXED_POS_INIT, pos_target_hypos[i], self._FIXED_DURATION)
        logger.log('{} check with model the hypotheses suggested by current policy parameters. (sample = {}/{})'.format(prefix_info, i+1, N))
        logger.log('{} predict by model. (pos_init = {}, pos_target = {}, duration = {})'.format(prefix_info, pos_init, pos_target, duration))
        prediction = model.Predict([pos_target], x_var=0.0**2, with_var=True, with_grad=True)
        loc_land_h    = prediction.Y.ravel()
        loc_land_err  = np.sqrt(np.diag(prediction.Var))
        loc_land_grad = prediction.Grad.ravel()
        loc_land_i = float(loc_land_h)
        logger.log('{} loc_land = {}, desired_loc_land = {}'.format(prefix_info, loc_land_i, desired_loc_land_samples[i]))
        sum_penalty += penalty
        sum_loss += (desired_loc_land_samples[i] - loc_land_i)**2
      ave_loss = sum_loss / (2 * N)
      ave_penalty = sum_penalty / (2 * N)
      final_loss = ave_loss + ave_penalty
      logger.log('{} loss = {}, penalty = {}'.format(prefix_info, final_loss, ave_penalty))
      logger.log('')
      return final_loss

    # optimize policy parameters with CMA-ES
    #mf_policy_init_guess = [0.38544, 0.10898, -0.00605, 0.00015] # for duration = 0.10, pos_init = 0.0, full sampling
    #mf_policy_init_var   = 0.00100 # for duration = 0.10, pos_init = 0.0, full sampling
    mf_policy_init_guess = [3.84310955e-01, 1.07871025e-01, -5.57357436e-03, 1.17338141e-04] # for duration = 0.10, pos_init = 0.0, ignore 12.5, 15.0 (keep 17.5)
    mf_policy_init_var   = 0.00100 # for duration = 0.10, pos_init = 0.0, ignore 12.5, 15.0 (keep 17.5)
    mf_policy_res = cma.fmin(mf_policy_loss_func, mf_policy_init_guess, mf_policy_init_var, args=(desired_loc_land_samples, N), 
                             popsize=20, tolx=10e-6, verb_disp=False, verb_log=0)
    mf_policy_optimal_params = mf_policy_res[0]
    logger.log('{} result = {}'.format(prefix_info, mf_policy_res))
    logger.log('{} optimal solution found. (params = {})'.format(prefix_info, mf_policy_optimal_params))
    logger.log('')

    # Query desired landing location
    desired_loc_land_input = input('{} desired_loc_land = '.format(prefix_info)).strip().lower()
    desired_loc_land = float(desired_loc_land_input)

    # predict optimal initial action from policy optimized by model
    optimal_inital_action = policy_func(desired_loc_land, mf_policy_optimal_params)

    # Optimize action with model by CMA-ES
    mb_action_init_guess = [optimal_inital_action, 0.1]
    mb_action_init_var   = 0.5
    self._run_hybrid_mb_action_iteration = 0
    def mb_action_loss_func(x):
      logger.log('{} optimization with model by CMA-ES. (iteration = {}, desired_loc_land = {})'.format(prefix_info, self._run_hybrid_mb_action_iteration, desired_loc_land))
      self._run_hybrid_mb_action_iteration += 1
      pos_target, x_1 = x
      logger.log('{} sample from model by CMA-ES. (pos_target = {})'.format(prefix_info, pos_target))
      prediction = model.Predict([pos_target], x_var=0.0**2, with_var=True, with_grad=True)
      loc_land_h    = prediction.Y.ravel()
      loc_land_err  = np.sqrt(np.diag(prediction.Var))
      loc_land_grad = prediction.Grad.ravel()
      loss = 0.5 * np.sqrt(desired_loc_land - loc_land_h)
      logger.log('{} loss = {}, loc_land_h = {}, loc_land_err = {}'.format(prefix_info, loss, loc_land_h, loc_land_err))
      logger.log('')
      return loss
    mb_action_res = cma.fmin(mb_action_loss_func, mb_action_init_guess, mb_action_init_var,
                             bounds=[[self.catapult.POS_MIN, self.catapult.POS_MIN], [self.catapult.POS_MAX, self.catapult.POS_MAX]], 
                             popsize=20, tolx=0.0001, verb_disp=False, verb_log=0)
    mb_action_optimal_pos_target = mb_action_res[0][0]
    logger.log('{} result = {}'.format(prefix_info, mb_action_res))
    logger.log('{} optimal solution found. (pos_target = {}, pos_init === {}, duration === {})'.format(prefix_info, mb_action_optimal_pos_target, self._FIXED_POS_INIT, self._FIXED_DURATION))
    logger.log('')
    
    # Test in true dynamics
    logger.log('{} test in true dynamics. (pos_init = {}, pos_target = {}, duration = {})'.format(prefix_info, self._FIXED_POS_INIT, mb_action_optimal_pos_target, self._FIXED_DURATION))
    loc_land = catapult.throw_linear(self._FIXED_POS_INIT, mb_action_optimal_pos_target, self._FIXED_DURATION)
    logger.log('{} loc_land = {}, desired_loc_land = {}'.format(prefix_info, loc_land, desired_loc_land))

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
      logger.log('usage: ./run_sim_002_linear_thetaT_d.py <operation>')
      quit()
  
  agent.run(operation)


