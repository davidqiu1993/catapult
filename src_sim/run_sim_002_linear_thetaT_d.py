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

#logger = TLoggerSim(fp_log='run_sim_002_linear_thetaT_d.log')
logger = TLoggerSim(fp_log=None)



class TCatapultLPLinearSim(object):
  """
  Simulation catapult learning and planning agent in linear motion control.
  """
  
  def __init__(self, catapult, abs_dirpath_data, abs_dirpath_model, abs_dirpath_log):
    super(TCatapultLPLinearSim, self).__init__()
    
    self.catapult = catapult
    
    self._FIXED_POS_INIT = self.catapult.POS_MIN
    self._FIXED_DURATION = self.catapult.DURATION_MIN
    
    self._SHOULD_LOAD_MODEL = False
    self._SHOULD_SAVE_MODEL = False
    
    self._abs_dirpath_data = abs_dirpath_data
    self._abs_dirpath_model = os.path.join(abs_dirpath_model, './')
    self._abs_dirpath_log = abs_dirpath_log

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

  def _train_model(self, model, x_train, y_train):
    model.UpdateBatch(x_train, y_train)
  
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
      
      plt.plot(plot_x_train, plot_y_train, 'bx')
      #plt.plot(plot_x_valid, plot_y_valid, 'b--')
      #plt.plot(plot_x_valid, plot_y_hypo,  'r-')
      plt.errorbar(plot_x_valid, plot_y_hypo, plot_err_hypo, color='r', linestyle='-')
      plt.ylabel('y')
      plt.grid(True)
      
      plt.show()
    
    return float(ave_stderr_y), float(ave_stderr_err)

  def _estimate_test_results(self, test_results, should_save=True):
    prefix = 'estimate_test_results'
    prefix_info = prefix + ':'
    
    if should_save:
      filename_test_results = 'test_' + '{:%Y%m%d_%H%M%S_%f}'.format(datetime.datetime.now()) + '.yaml'
      filepath_test_results = os.path.abspath(os.path.join(self._abs_dirpath_log, filename_test_results))
      with open(filepath_test_results, 'w') as yaml_file:
        yaml.dump(test_results, yaml_file, default_flow_style=False)
      logger.log('{} save test results. (path={})'.format(prefix_info, filepath_test_results))
    
    samples_actual_loc_land = []
    samples_actual_pos_target = []
    for entry in self._dataset:
      samples_actual_loc_land.append(float(entry['result']['loc_land']))
      samples_actual_pos_target.append(float(entry['action']['pos_target']))
    
    samples_test_desired_loc_land = []
    samples_test_pos_target = []
    samples_test_loc_land = []
    for entry in test_results:
      approach              = str(entry['approach'])
      desired_loc_land      = float(entry['desired_loc_land'])
      loc_land              = float(entry['loc_land'])
      pos_target            = float(entry['pos_target'])
      n_preopt_samples      = int(entry['preopt_samples'])
      n_samples             = int(entry['samples'])
      n_preopt_simulations  = int(entry['preopt_simulations'])
      n_simulations         = int(entry['simulations'])
      samples_test_desired_loc_land.append(desired_loc_land)
      samples_test_pos_target.append(pos_target)
      samples_test_loc_land.append(loc_land)
    
    plt.figure()
    
    plt.subplot(211)
    plt.xlabel('d*')
    plt.ylabel('thetaT')
    plt.plot(samples_actual_loc_land, samples_actual_pos_target, 'bx')
    plt.plot(samples_test_desired_loc_land, samples_test_pos_target, 'ro')
    
    plt.subplot(212)
    plt.xlabel('d*')
    plt.ylabel('d')
    plt.plot(samples_actual_loc_land, samples_actual_loc_land, 'b-')
    plt.plot(samples_test_desired_loc_land, samples_test_loc_land, 'ro')
    
    plt.show()

  def _run_model_based(self):
    """
    model-based:
      - offline
      - NN(dynamics)
      - CMA-ES(action)
    """
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
    model = self._create_model(1, 1, hiddens=[200, 200], max_updates=10000, should_load_model=should_load_model, prefix_info=prefix_info)
    
    # Train model
    x_train = []
    y_train = []
    for entry in self._dataset:
      x_train.append([entry['action']['pos_target']])
      y_train.append([entry['result']['loc_land']])
    if not should_load_model:
      self._train_model(model, x_train, y_train)
    
    # Estimate model quality
    x_valid = [x_train[i * 2] for i in range(int(len(x_train) / 2))]
    y_valid = [y_train[i * 2] for i in range(int(len(y_train) / 2))]
    should_estimate_model_quality_input = input('{} estimate model quality (Y/n)?> '.format(prefix_info)).strip().lower()
    if should_estimate_model_quality_input in ['', 'y']:
      ave_stderr_y, ave_stderr_err = self._estimate_model_quality(model, x_train, y_train, x_valid, y_valid, should_plot=True)
      logger.log('{} ave_stderr_y = {}, ave_stderr_err = {}'.format(prefix_info, ave_stderr_y, ave_stderr_err))
    
    # Save model
    if not should_load_model:
      should_save_model_input = input('{} save model (Y/n)?> '.format(prefix_info)).strip().lower()
      if should_save_model_input in ['', 'y']:
        self._save_model(model, prefix_info)
    
    # Test desired landing locations
    test_sample_desired_loc_land = [(0.0 + 2.5*i) for i in range(42)]
    test_results = []
    for i in range(len(test_sample_desired_loc_land)):
      has_finished_this_round = False
      while not has_finished_this_round:
        try:
          # Query desired landing location
          desired_loc_land = test_sample_desired_loc_land[i]
          
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
            loc_land_h    = (prediction.Y.ravel())[0]
            loc_land_err  = (np.sqrt(np.diag(prediction.Var)))[0]
            loc_land_grad = (prediction.Grad.ravel())[0]
            loss = 0.5 * (desired_loc_land - loc_land_h)**2
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

          # Add to test results
          entry = {
            'approach': 'model-based, offline, NN(dynamics), CMA-ES(action)',
            'desired_loc_land': float(desired_loc_land),
            'loc_land': float(loc_land),
            'pos_target': float(optimal_pos_target),
            'preopt_samples': int(len(self._dataset)),
            'samples': int(0),
            'preopt_simulations': int(0),
            'simulations': int(self._run_model_based_iteration)
          }
          test_results.append(entry)
          logger.log('{} test result added to temperary result dataset >>> '.format(prefix_info))
          logger.log(entry)
          
          has_finished_this_round = True
        
        except:
          has_finished_this_round = False

    # Estimate test results
    self._estimate_test_results(test_results)
  
  def _run_model_based_online(self):
    """
    Model-based:
      - online
      - NN(dynamics)
      - CMA-ES(action)
    """
    prefix = 'catapult_sim/model_based_online'
    prefix_info = prefix + ':'
    
    # Create model
    model = self._create_model(1, 1, hiddens=[200, 200], max_updates=10000, should_load_model=False, prefix_info=prefix_info)
    
    # Load validation dataset
    x_valid = []
    y_valid = []
    for i in range(len(self._dataset)):
      if i % 3 == 0:
        entry = self._dataset[i]
        x_valid.append([entry['action']['pos_target']])
        y_valid.append([entry['result']['loc_land']])
    
    # Online dataset
    x_train = []
    y_train = []
    
    # Test desired landing locations
    test_sample_desired_loc_land = [float(0.0 + np.random.sample() * 100.0) for i in range(100)]
    test_results = []
    for i in range(len(test_sample_desired_loc_land)):
      has_finished_this_round = False
      while not has_finished_this_round:
        try:
          # Query desired landing location
          desired_loc_land = test_sample_desired_loc_land[i]
          
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
            loc_land_h    = (prediction.Y.ravel())[0]
            loc_land_err  = (np.sqrt(np.diag(prediction.Var)))[0]
            loc_land_grad = (prediction.Grad.ravel())[0]
            loss = 0.5 * (desired_loc_land - loc_land_h)**2
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
          
          # Update model samples
          x_train.append([optimal_pos_target])
          y_train.append([loc_land])
          model.Update([optimal_pos_target], [loc_land], not_learn=False)
          
          # Estimate model quality
          ave_stderr_y, ave_stderr_err = self._estimate_model_quality(model, x_train, y_train, x_valid, y_valid, should_plot=False)
          logger.log('{} ave_stderr_y = {}, ave_stderr_err = {}'.format(prefix_info, ave_stderr_y, ave_stderr_err))
          time.sleep(2.0)
          
          # Add to test results
          entry = {
            'approach': 'model-based, online, NN(dynamics), CMA-ES(action)',
            'desired_loc_land': float(desired_loc_land),
            'loc_land': float(loc_land),
            'pos_target': float(optimal_pos_target),
            'preopt_samples': int(0),
            'samples': int(i),
            'preopt_simulations': int(0),
            'simulations': int(self._run_model_based_iteration),
            'model_ave_stderr_y': ave_stderr_y,
            'model_ave_stderr_err': ave_stderr_err
          }
          test_results.append(entry)
          logger.log('{} test result added to temperary result dataset >>> '.format(prefix_info))
          logger.log(entry)
          
          has_finished_this_round = True
        
        except:
          has_finished_this_round = False

    # Estimate test results
    self._estimate_test_results(test_results, should_save=True)
  
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
    """
    model-free:
      - online
      - CMA-ES(policy; cubic)
      - policy(action)
    """
    prefix = 'catapult_sim/model_free'
    prefix_info = prefix + ':'
    
    # query operation
    logger.log('{} model-free operations:'.format(prefix_info))
    logger.log('  - (1) launch policy parameters optimization with CMA-ES in complete sample space')
    logger.log('  - (2) launch policy parameters optimization with CMA-ES in touched sample space')
    logger.log('  - (3) load pre-optimized policy parameters in complete sample space')
    logger.log('  - (4) load pre-optimized policy parameters in touched sample space')
    mf_operation_input = input('{} please model-free approach operation (1/2/..)> '.format(prefix_info)).strip().lower()
    mf_operation = str(int(mf_operation_input))
    assert(mf_operation in ['1', '2', '3', '4'])
    
    # define policy function with parameters
    def policy_func(desired_loc_land, params):
      x = desired_loc_land
      p = params
      y = 0
      for i in range(4):
        y += p[i] * x**i
      return y
    
    # specify desired landing location sample points
    if mf_operation == '1' or mf_operation == '3':
      desired_loc_land_samples = [(0 + 2.5*i) for i in range(11)] # full sampling
    elif mf_operation == '2' or mf_operation == '4':
      desired_loc_land_samples = [0.0, 2.5, 5.0, 7.5, 10.0, 17.5, 20.0, 22.5, 25.0] # ignore 12.5, 15.0 (keep 17.5)
    else:
      assert(False)
    N = len(desired_loc_land_samples)
    
    # select operation
    logger.log('{} selected operation: {}'.format(prefix_info, mf_operation))
    if mf_operation == '1' or mf_operation == '2':
      saver_dataset = TCatapultDatasetSim(abs_dirpath=self._abs_dirpath_data, auto_init=False)
      saver_dataset.init_yaml()
      
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
      if mf_operation == '1':
        init_guess = [0.38544, 0.10898, -0.00605, 0.00015] # for duration = 0.10, pos_init = 0.0, full sampling
        init_var   = 0.00100 # for duration = 0.10, pos_init = 0.0, full sampling
      elif mf_operation == '2':
        init_guess = [3.84310955e-01, 1.07871025e-01, -5.57357436e-03, 1.17338141e-04] # for duration = 0.10, pos_init = 0.0, ignore 12.5, 15.0 (keep 17.5)
        init_var   = 0.00020 # for duration = 0.10, pos_init = 0.0, ignore 12.5, 15.0 (keep 17.5)
      else:
        assert(False)
      res = cma.fmin(loss_func, init_guess, init_var, args=(desired_loc_land_samples, N), 
                    popsize=20, tolx=10e-6, verb_disp=False, verb_log=0)
      optimal_params = res[0]
      logger.log('{} result = {}'.format(prefix_info, res))
      logger.log('{} optimal solution found. (params = {})'.format(prefix_info, optimal_params))
    
    elif mf_operation == '3':
      logger.log('{} load pre-optimized policy parameters in complete sample space'.format(prefix_info))
      optimal_params = [3.84207069e-01, 1.08050257e-01, -5.44212265e-03, 1.20992580e-04]
      self._run_model_free_iteration = 577
      logger.log('{} iterations = {}, optimal_params = {}'.format(prefix_info, self._run_model_free_iteration, optimal_params))
    
    elif mf_operation == '4':
      logger.log('{} load pre-optimized policy parameters in touched sample space'.format(prefix_info))
      optimal_params = [3.84177005e-01, 1.07917544e-01, -5.55294093e-03, 1.14927976e-04]
      self._run_model_free_iteration = 961
      logger.log('{} iterations = {}, optimal_params = {}'.format(prefix_info, self._run_model_free_iteration, optimal_params))
    
    else:
      assert(False)
    
    logger.log('')
    
    # Test desired landing locations
    test_sample_desired_loc_land = [(0.0 + 1.25*i) for i in range(21)]
    test_results = []
    for i in range(len(test_sample_desired_loc_land)):
      # Query desired landing location
      desired_loc_land = test_sample_desired_loc_land[i]
      
      # test policy parameters in true dynamics
      logger.log('{} apply optimal policy parameters. (params = {})'.format(prefix_info, optimal_params))
      pos_target_hypo = policy_func(desired_loc_land, optimal_params)
      logger.log('{} predict action by parameterized policy. (desired_loc_land = {}, pos_target_hypo = {})'.format(prefix_info, desired_loc_land, pos_target_hypo))
      logger.log('{} test in true dynamics. (pos_init = {}, pos_target = {}, duration = {})'.format(prefix_info, self._FIXED_POS_INIT, pos_target_hypo, self._FIXED_DURATION))
      loc_land = catapult.throw_linear(self._FIXED_POS_INIT, pos_target_hypo, self._FIXED_DURATION)
      logger.log('{} loc_land = {}, desired_loc_land = {}'.format(prefix_info, loc_land, desired_loc_land))
      
      # Add to test results
      entry = {
        'approach': 'model-free, online, CMA-ES(policy; cubic), policy(action)',
        'desired_loc_land': float(desired_loc_land),
        'loc_land': float(loc_land),
        'pos_target': float(pos_target_hypo),
        'preopt_samples': int(self._run_model_free_iteration * N),
        'samples': int(0),
        'preopt_simulations': int(0),
        'simulations': int(0)
      }
      test_results.append(entry)
      logger.log('{} test result added to temperary result dataset >>> '.format(prefix_info))
      logger.log(entry)
    
    # Estimate test results
    self._estimate_test_results(test_results)
  
  def _run_model_free_nn(self):
    """
    model-free:
      - online
      - NN(policy; deterministic)
      - policy(action)
    """
    prefix = 'catapult_sim/model_free_nn'
    prefix_info = prefix + ':'
    
    epsilon       = 0.05#0.70
    epsilon_decay = 1.00#0.90
    
    # define policy network
    model_policy = self._create_model(1, 1, hiddens=[200, 200], max_updates=10000, should_load_model=False, prefix_info=prefix_info)
    
    # Load validation dataset
    x_valid = []
    y_valid = []
    for i in range(len(self._dataset)):
      if i % 3 == 0:
        entry = self._dataset[i]
        x_valid.append([entry['result']['loc_land']])
        y_valid.append([entry['action']['pos_target']])
    
    # Online dataset
    x_train = []
    y_train = []
    
    # Test desired landing locations
    test_sample_desired_loc_land = [float(0.0 + np.random.sample() * 100.0) for i in range(100)]
    test_results = []
    for i in range(len(test_sample_desired_loc_land)):
      epsilon = epsilon * epsilon_decay
      has_finished_this_round = False
      while not has_finished_this_round:
        try:
          # Query desired landing location
          desired_loc_land = test_sample_desired_loc_land[i]
          
          # Predict from policy network
          prediction = model_policy.Predict([desired_loc_land], x_var=0.0**2, with_var=True, with_grad=True)
          pos_target_h    = (prediction.Y.ravel())[0]
          pos_target_err  = (np.sqrt(np.diag(prediction.Var)))[0]
          pos_target_grad = (prediction.Grad.ravel())[0]
          logger.log('{} Predict from policy network. (desired_loc_land = {}, pos_target_h = {}, pos_target_err = {})'.format(prefix_info, desired_loc_land, pos_target_h, pos_target_err))
          
          # Fix action
          if pos_target_h < self.catapult.POS_MIN: pos_target_h = self.catapult.POS_MIN
          if pos_target_h > self.catapult.POS_MAX: pos_target_h = self.catapult.POS_MAX
          logger.log('{} Fix action. (pos_target_h = {})'.format(prefix_info, pos_target_h))
          
          # Add randomness to action
          pos_target_h = pos_target_h + float((self.catapult.POS_MAX - self.catapult.POS_MIN) * (np.random.sample() - 0.5) * 2.0 * epsilon)
          if pos_target_h < self.catapult.POS_MIN: pos_target_h = self.catapult.POS_MIN
          if pos_target_h > self.catapult.POS_MAX: pos_target_h = self.catapult.POS_MAX
          logger.log('{} Randomize action. (pos_target_h = {})'.format(prefix_info, pos_target_h))
          
          # Test in true dynamics
          logger.log('{} test in true dynamics. (pos_init = {}, pos_target = {}, duration = {})'.format(prefix_info, self._FIXED_POS_INIT, pos_target_h, self._FIXED_DURATION))
          loc_land = catapult.throw_linear(self._FIXED_POS_INIT, pos_target_h, self._FIXED_DURATION)
          logger.log('{} loc_land = {}, desired_loc_land = {}'.format(prefix_info, loc_land, desired_loc_land))
          
          # Update model samples
          logger.log('{} train policy network samples. (loc_land = {}, pos_target = {})'.format(prefix_info, loc_land, pos_target_h))
          x_train.append([loc_land])
          y_train.append([pos_target_h])
          model_policy.Update([loc_land], [pos_target_h], not_learn=False)
          
          # Estimate model quality
          ave_stderr_y, ave_stderr_err = self._estimate_model_quality(model_policy, x_train, y_train, x_valid, y_valid, should_plot=False)
          logger.log('{} ave_stderr_y = {}, ave_stderr_err = {}'.format(prefix_info, ave_stderr_y, ave_stderr_err))
          time.sleep(2.0)
          
          # Add to test results
          entry = {
            'approach': 'model-free, online, NN(policy), policy(action)',
            'desired_loc_land': float(desired_loc_land),
            'loc_land': float(loc_land),
            'pos_target': float(pos_target_h),
            'preopt_samples': int(0),
            'samples': int(i),
            'preopt_simulations': int(0),
            'simulations': int(0),
            'policy_nn_ave_stderr_y': float(ave_stderr_y),
            'policy_nn_ave_stderr_err': float(ave_stderr_err),
            'epsilon': epsilon
          }
          test_results.append(entry)
          logger.log('{} test result added to temperary result dataset >>> '.format(prefix_info))
          logger.log(entry)
          
          has_finished_this_round = True
        
        except:
          has_finished_this_round = False
    
    # Estimate final model quality
    ave_stderr_y, ave_stderr_err = self._estimate_model_quality(model_policy, x_train, y_train, x_valid, y_valid, should_plot=False)
    logger.log('{} ave_stderr_y = {}, ave_stderr_err = {}'.format(prefix_info, ave_stderr_y, ave_stderr_err))
    
    # Estimate test results
    self._estimate_test_results(test_results, should_save=True)
  
  def _run_model_free_nn_offline(self):
    """
    model-free:
      - offline
      - NN(policy; deterministic)
      - policy(action)
    """
    prefix = 'catapult_sim/model_free_nn_offline'
    prefix_info = prefix + ':'
    
    # Create policy network (and train/load)
    should_train_model = True
    should_train_model_input = input('{} train policy network rather than load from model file (Y/n)?> '.format(prefix_info)).strip().lower()
    if should_train_model_input in ['', 'y']:
      should_train_model = True
    else:
      should_train_model = False
    should_load_model = (not should_train_model)
    model_policy = self._create_model(1, 1, hiddens=[200, 200], max_updates=10000, should_load_model=should_load_model, prefix_info=prefix_info)
    
    # Train policy network
    x_train = []
    y_train = []
    for i in range(len(self._dataset)):
      entry = self._dataset[i]
      x_train.append([entry['result']['loc_land']])
      y_train.append([entry['action']['pos_target']])
    if should_train_model:
      self._train_model(model_policy, x_train, y_train)
    
    # Estimate model quality
    x_valid = []
    y_valid = []
    for i in range(len(self._dataset)):
      if i % 3 == 0:
        entry = self._dataset[i]
        x_valid.append([entry['result']['loc_land']])
        y_valid.append([entry['action']['pos_target']])
    should_estimate_model_quality_input = input('{} estimate policy network quality (Y/n)?> '.format(prefix_info)).strip().lower()
    if should_estimate_model_quality_input in ['', 'y']:
      ave_stderr_y, ave_stderr_err = self._estimate_model_quality(model_policy, x_train, y_train, x_valid, y_valid, should_plot=True)
      logger.log('{} ave_stderr_y = {}, ave_stderr_err = {}'.format(prefix_info, ave_stderr_y, ave_stderr_err))
    
    # Save policy network
    if not should_load_model:
      should_save_model_input = input('{} save policy network (Y/n)?> '.format(prefix_info)).strip().lower()
      if should_save_model_input in ['', 'y']:
        self._save_model(model_policy, prefix_info)
    
    # Test desired landing locations
    test_sample_desired_loc_land = [(0.0 + 2.5*i) for i in range(42)]
    test_results = []
    for i in range(len(test_sample_desired_loc_land)):
      has_finished_this_round = False
      while not has_finished_this_round:
        try:
          # Query desired landing location
          desired_loc_land = test_sample_desired_loc_land[i]
          
          # Predict from policy network
          prediction = model_policy.Predict([desired_loc_land], x_var=0.0**2, with_var=True, with_grad=True)
          pos_target_h    = (prediction.Y.ravel())[0]
          pos_target_err  = (np.sqrt(np.diag(prediction.Var)))[0]
          pos_target_grad = (prediction.Grad.ravel())[0]
          logger.log('{} Predict from policy network. (desired_loc_land = {}, pos_target_h = {}, pos_target_err = {})'.format(prefix_info, desired_loc_land, pos_target_h, pos_target_err))
          
          # Fix action
          if pos_target_h < self.catapult.POS_MIN: pos_target_h = self.catapult.POS_MIN
          if pos_target_h > self.catapult.POS_MAX: pos_target_h = self.catapult.POS_MAX
          logger.log('{} Fix action. (pos_target_h = {})'.format(prefix_info, pos_target_h))
          
          # Test in true dynamics
          logger.log('{} test in true dynamics. (pos_init = {}, pos_target = {}, duration = {})'.format(prefix_info, self._FIXED_POS_INIT, pos_target_h, self._FIXED_DURATION))
          loc_land = catapult.throw_linear(self._FIXED_POS_INIT, pos_target_h, self._FIXED_DURATION)
          logger.log('{} loc_land = {}, desired_loc_land = {}'.format(prefix_info, loc_land, desired_loc_land))
          
          # Add to test results
          entry = {
            'approach': 'model-free, offline, NN(policy), policy(action)',
            'desired_loc_land': float(desired_loc_land),
            'loc_land': float(loc_land),
            'pos_target': float(pos_target_h),
            'preopt_samples': int(len(x_train)),
            'samples': int(0),
            'preopt_simulations': int(0),
            'simulations': int(0)
          }
          test_results.append(entry)
          logger.log('{} test result added to temperary result dataset >>> '.format(prefix_info))
          logger.log(entry)
          
          has_finished_this_round = True
        
        except:
          has_finished_this_round = False

    # Estimate test results
    self._estimate_test_results(test_results, should_save=True)
  
  def _run_hybrid(self):
    prefix = 'catapult_sim/model_hybrid'
    prefix_info = prefix + ':'
    
    logger.log('{} ALERT: Discarded version of hybrid approach.'.format(prefix_info))
    assert(False)
    
    # query operation
    logger.log('{} hyprid approach operations:'.format(prefix_info))
    logger.log('  - (1) launch policy parameters optimization with CMA-ES in complete sample space')
    logger.log('  - (2) launch policy parameters optimization with CMA-ES in touched sample space')
    hybrid_operation_input = input('{} please model-free approach operation (1/2)> '.format(prefix_info)).strip().lower()
    hybrid_operation = str(int(hybrid_operation_input))
    assert(hybrid_operation in ['1', '2'])
    
    # create model (and train/load)
    should_train_model = True
    should_train_model_input = input('{} train model rather than load from model file (Y/n)?> '.format(prefix_info)).strip().lower()
    if should_train_model_input in ['', 'y']:
      should_train_model = True
    else:
      should_train_model = False
    should_load_model = (not should_train_model)
    model = self._create_model(1, 1, hiddens=[200, 200], max_updates=40000, should_load_model=should_load_model, prefix_info=prefix_info)
    
    # train model
    x_train = []
    y_train = []
    for entry in self._dataset:
      x_train.append([entry['action']['pos_target']])
      y_train.append([entry['result']['loc_land']])
    if not should_load_model:
      self._train_model(model, x_train, y_train)
    
    # estimate model quality
    x_valid = [x_train[i * 2] for i in range(int(len(x_train) / 2))]
    y_valid = [y_train[i * 2] for i in range(int(len(y_train) / 2))]
    should_estimate_model_quality_input = input('{} estimate model quality (Y/n)?> '.format(prefix_info)).strip().lower()
    if should_estimate_model_quality_input in ['', 'y']:
      ave_stderr_y, ave_stderr_err = self._estimate_model_quality(model, x_train, y_train, x_valid, y_valid, should_plot=True)
      logger.log('{} ave_stderr_y = {}, ave_stderr_err = {}'.format(prefix_info, ave_stderr_y, ave_stderr_err))
    
    # save model
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
    if hybrid_operation == '1':
      desired_loc_land_samples = [(0 + 2.5*i) for i in range(11)] # full sampling
    elif hybrid_operation == '2':
      desired_loc_land_samples = [0.0, 2.5, 5.0, 7.5, 10.0, 17.5, 20.0, 22.5, 25.0] # ignore 12.5, 15.0 (keep 17.5)
    else:
      assert(False)
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
        loc_land_h    = (prediction.Y.ravel())[0]
        loc_land_err  = (np.sqrt(np.diag(prediction.Var)))[0]
        loc_land_grad = (prediction.Grad.ravel())[0]
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
    mf_policy_start_policy_params_optimization_input = input('{} start policy parameters optimization with model (Y)?> '.format(prefix_info))
    if hybrid_operation == '1':
      mf_policy_init_guess = [0.38544, 0.10898, -0.00605, 0.00015] # for duration = 0.10, pos_init = 0.0, full sampling
      mf_policy_init_var   = 0.00100 # for duration = 0.10, pos_init = 0.0, full sampling
    elif hybrid_operation == '2':
      mf_policy_init_guess = [3.84310955e-01, 1.07871025e-01, -5.57357436e-03, 1.17338141e-04] # for duration = 0.10, pos_init = 0.0, ignore 12.5, 15.0 (keep 17.5)
      mf_policy_init_var   = 0.00100 # for duration = 0.10, pos_init = 0.0, ignore 12.5, 15.0 (keep 17.5)
    else:
      assert(False)
    mf_policy_res = cma.fmin(mf_policy_loss_func, mf_policy_init_guess, mf_policy_init_var, args=(desired_loc_land_samples, N), 
                             popsize=20, tolx=10e-4, verb_disp=False, verb_log=0)
    mf_policy_optimal_params = mf_policy_res[0]
    #mf_policy_optimal_params = [  3.08515264e-01,   1.16457655e-01,  -5.83116752e-03,   1.19370654e-04]
    logger.log('{} result = {}'.format(prefix_info, mf_policy_res))
    logger.log('{} optimal solution found. (params = {})'.format(prefix_info, mf_policy_optimal_params))
    logger.log('')
    
    # test desired landing locations
    test_sample_desired_loc_land = [(0.0 + 1.25*i) for i in range(21)]
    test_results = []
    for i in range(len(test_sample_desired_loc_land)):
      # query desired landing location
      desired_loc_land = test_sample_desired_loc_land[i]
      
      # predict optimal initial action from policy optimized by model
      optimal_inital_action = policy_func(desired_loc_land, mf_policy_optimal_params) # pos_target
      
      # evaluate initial action
      eval_prediction = model.Predict([optimal_inital_action], x_var=0.0**2, with_var=True, with_grad=True)
      eval_loc_land_h = (eval_prediction.Y.ravel())[0]
      
      # optimize action with model by CMA-ES
      mb_action_init_guess = [optimal_inital_action, 0.1]
      mb_action_init_var   = 0.75 * math.pi * abs(desired_loc_land - eval_loc_land_h) / 25.0
      self._run_hybrid_mb_action_iteration = 0
      def mb_action_loss_func(x):
        logger.log('{} optimization with model by CMA-ES. (iteration = {}, desired_loc_land = {})'.format(prefix_info, self._run_hybrid_mb_action_iteration, desired_loc_land))
        self._run_hybrid_mb_action_iteration += 1
        pos_target, x_1 = x
        logger.log('{} sample from model by CMA-ES. (pos_target = {})'.format(prefix_info, pos_target))
        prediction = model.Predict([pos_target], x_var=0.0**2, with_var=True, with_grad=True)
        loc_land_h    = (prediction.Y.ravel())[0]
        loc_land_err  = (np.sqrt(np.diag(prediction.Var)))[0]
        loc_land_grad = (prediction.Grad.ravel())[0]
        loss = 0.5 * (desired_loc_land - loc_land_h)**2
        logger.log('{} loss = {}, loc_land_h = {}, loc_land_err = {}'.format(prefix_info, loss, loc_land_h, loc_land_err))
        logger.log('')
        return loss
      mb_has_finished_action_optimization = False
      while not mb_has_finished_action_optimization:
        try:
          mb_action_res = cma.fmin(mb_action_loss_func, mb_action_init_guess, mb_action_init_var,
                                  bounds=[[self.catapult.POS_MIN, self.catapult.POS_MIN], [self.catapult.POS_MAX, self.catapult.POS_MAX]], 
                                  popsize=20, tolx=0.0001, verb_disp=False, verb_log=0)
          mb_has_finished_action_optimization = True
        except:
          mb_has_finished_action_optimization = False
      mb_action_optimal_pos_target = mb_action_res[0][0]
      logger.log('{} result = {}'.format(prefix_info, mb_action_res))
      logger.log('{} optimal solution found. (pos_target = {}, pos_init === {}, duration === {})'.format(prefix_info, mb_action_optimal_pos_target, self._FIXED_POS_INIT, self._FIXED_DURATION))
      logger.log('')
      
      # test in true dynamics
      logger.log('{} test in true dynamics. (pos_init = {}, pos_target = {}, duration = {})'.format(prefix_info, self._FIXED_POS_INIT, mb_action_optimal_pos_target, self._FIXED_DURATION))
      loc_land = catapult.throw_linear(self._FIXED_POS_INIT, mb_action_optimal_pos_target, self._FIXED_DURATION)
      logger.log('{} loc_land = {}, desired_loc_land = {}'.format(prefix_info, loc_land, desired_loc_land))
      
      # Add to test results
      entry = {
        'approach': 'hybrid, CMA-ES(policy; model), CMA-ES(action; model, policy(init_action))',
        'desired_loc_land': float(desired_loc_land),
        'loc_land': float(loc_land),
        'pos_target': float(mb_action_optimal_pos_target),
        'preopt_samples': int(len(self._dataset)),
        'samples': int(0),
        'preopt_simulations': int(self._run_hybrid_mf_policy_iteration * N),
        'simulations': int(self._run_hybrid_mb_action_iteration)
      }
      test_results.append(entry)
      logger.log('{} test result added to temperary result dataset >>> '.format(prefix_info))
      logger.log(entry)
    
    # Estimate test results
    self._estimate_test_results(test_results)
  
  def _run_hybrid_nn2_offline(self):
    """
    hybrid:
      - offline
      - NN(policy; deterministic)
      - NN(dynamics)
      - policy(init_action)
      - CMA-ES(action; init_action, var~err)
      - [model-based if err>threshold]
    """
    prefix = 'catapult_sim/hybrid_nn2_offline'
    prefix_info = prefix + ':'
    
    MIN_LOC_LAND = -1
    MAX_LOC_LAND = 105
    THRESHOLD_ACTION_REPLAN_ERR = 0.05 * (MAX_LOC_LAND - MIN_LOC_LAND)
    
    # Create policy network
    model_policy = self._create_model(1, 1, hiddens=[200, 200], max_updates=10000, should_load_model=False, prefix_info=prefix_info)
    
    # Train policy network
    x_train_policy = []
    y_train_policy = []
    for i in range(len(self._dataset)):
      entry = self._dataset[i]
      x_train_policy.append([entry['result']['loc_land']])
      y_train_policy.append([entry['action']['pos_target']])
    input('{} start to train policy network (Y)?> '.format(prefix_info))
    self._train_model(model_policy, x_train_policy, y_train_policy)
    
    # Estimate policy network quality
    x_valid_policy = []
    y_valid_policy = []
    for i in range(len(self._dataset)):
      if i % 3 == 0:
        entry = self._dataset[i]
        x_valid_policy.append([entry['result']['loc_land']])
        y_valid_policy.append([entry['action']['pos_target']])
    should_estimate_policy_network_quality_input = input('{} estimate policy network quality (Y/n)?> '.format(prefix_info)).strip().lower()
    if should_estimate_policy_network_quality_input in ['', 'y']:
      policy_ave_stderr_y, policy_ave_stderr_err = self._estimate_model_quality(model_policy, x_train_policy, y_train_policy, x_valid_policy, y_valid_policy, should_plot=True)
      logger.log('{} policy_ave_stderr_y = {}, policy_ave_stderr_err = {}'.format(prefix_info, policy_ave_stderr_y, policy_ave_stderr_err))
    
    # Create dynamics model
    model_dynamics = self._create_model(1, 1, hiddens=[200, 200], max_updates=10000, should_load_model=False, prefix_info=prefix_info)
    
    # Train dynamics model
    x_train_dynamics = []
    y_train_dynamics = []
    for i in range(len(self._dataset)):
      entry = self._dataset[i]
      x_train_dynamics.append([entry['action']['pos_target']])
      y_train_dynamics.append([entry['result']['loc_land']])
    input('{} start to train dynamics model (Y)?> '.format(prefix_info))
    self._train_model(model_dynamics, x_train_dynamics, y_train_dynamics)
    
    # Estimate dynamics model quality
    x_valid_dynamics = []
    y_valid_dynamics = []
    for i in range(len(self._dataset)):
      if i % 3 == 0:
        entry = self._dataset[i]
        x_valid_dynamics.append([entry['action']['pos_target']])
        y_valid_dynamics.append([entry['result']['loc_land']])
    should_estimate_dynamics_model_quality_input = input('{} estimate dynamics model quality (Y/n)?> '.format(prefix_info)).strip().lower()
    if should_estimate_dynamics_model_quality_input in ['', 'y']:
      dynamics_ave_stderr_y, dynamics_ave_stderr_err = self._estimate_model_quality(model_dynamics, x_train_dynamics, y_train_dynamics, x_valid_dynamics, y_valid_dynamics, should_plot=True)
      logger.log('{} dynamics_ave_stderr_y = {}, dynamics_ave_stderr_err = {}'.format(prefix_info, dynamics_ave_stderr_y, dynamics_ave_stderr_err))
    
    # Test in real dynamics
    test_sample_desired_loc_land = [(0.0 + 2.5*i) for i in range(42)]
    test_results = []
    for i in range(len(test_sample_desired_loc_land)):
      # Query desired landing location
      desired_loc_land = test_sample_desired_loc_land[i]
      
      # Predict initial action from policy network
      prediction_policy = model_policy.Predict([desired_loc_land], x_var=0.0**2, with_var=True, with_grad=True)
      pos_target_h    = (prediction_policy.Y.ravel())[0]
      pos_target_err  = (np.sqrt(np.diag(prediction_policy.Var)))[0]
      pos_target_grad = (prediction_policy.Grad.ravel())[0]
      logger.log('{} Predict initial action from policy network. (desired_loc_land = {}, pos_target_h = {}, pos_target_err = {})'.format(prefix_info, desired_loc_land, pos_target_h, pos_target_err))
      
      # Evaluate initial action
      eval_prediction = model_dynamics.Predict([pos_target_h], x_var=0.0**2, with_var=True, with_grad=True)
      eval_loc_land_h    = (eval_prediction.Y.ravel())[0]
      eval_loc_land_err  = (np.sqrt(np.diag(eval_prediction.Var)))[0]
      eval_loc_land_grad = (eval_prediction.Grad.ravel())[0]
      logger.log('{} Evaluate initial action with dynamics model. (desired_loc_land = {}, eval_loc_land_h = {})'.format(prefix_info, desired_loc_land, eval_loc_land_h))
      
      # Optimize parameters with CMA-ES and initial guess
      init_guess = [pos_target_h, 0.1]
      init_var   = (self.catapult.POS_MAX - self.catapult.POS_MIN) * abs(desired_loc_land - eval_loc_land_h) / (MAX_LOC_LAND - MIN_LOC_LAND)
      logger.log('{} Optimize action by CMA-ES with model. (init_guess: {}, init_var: {})'.format(prefix_info, pos_target_h, init_var))
      self._run_model_based_iteration = 0
      def f(x):
        self._run_model_based_iteration += 1
        pos_target, x_1 = x
        logger.log('{} sample from CMA-ES. (iteration = {}, desired_loc_land = {}, pos_target = {})'.format(prefix_info, self._run_model_based_iteration, desired_loc_land, pos_target))
        prediction = model_dynamics.Predict([pos_target], x_var=0.0**2, with_var=True, with_grad=True)
        loc_land_h    = (prediction.Y.ravel())[0]
        loc_land_err  = (np.sqrt(np.diag(prediction.Var)))[0]
        loc_land_grad = (prediction.Grad.ravel())[0]
        loss = 0.5 * (desired_loc_land - loc_land_h)**2
        logger.log('{} loss = {}, loc_land_h = {}, loc_land_err = {}'.format(prefix_info, loss, loc_land_h, loc_land_err))
        logger.log('')
        return loss
      
      has_finished_this_round = False
      while not has_finished_this_round:
        try:
          self._run_model_based_iteration = 0
          res = cma.fmin(f, init_guess, init_var,
                         bounds=[[self.catapult.POS_MIN, self.catapult.POS_MIN], [self.catapult.POS_MAX, self.catapult.POS_MAX]], 
                         popsize=20, tolx=0.0001, verb_disp=False, verb_log=0)
          has_finished_this_round = True
        except:
          has_finished_this_round = False
      
      optimal_pos_target = res[0][0]
      optimal_pos_target_cma = optimal_pos_target
      iter_cma = self._run_model_based_iteration
      logger.log('{} result = {}'.format(prefix_info, res))
      logger.log('{} optimal solution found. (pos_target = {}, pos_init === {}, duration === {})'.format(prefix_info, optimal_pos_target, self._FIXED_POS_INIT, self._FIXED_DURATION))
      logger.log('')
      
      # Evaluate optimal action after CMA-ES converged
      eval_opt_prediction = model_dynamics.Predict([optimal_pos_target], x_var=0.0**2, with_var=True, with_grad=True)
      eval_opt_loc_land_h    = (eval_opt_prediction.Y.ravel())[0]
      eval_opt_loc_land_err  = (np.sqrt(np.diag(eval_opt_prediction.Var)))[0]
      eval_opt_loc_land_grad = (eval_opt_prediction.Grad.ravel())[0]
      should_replan = (abs(eval_opt_loc_land_h - desired_loc_land) > THRESHOLD_ACTION_REPLAN_ERR)
      logger.log('{} Evaluate optimal action after CMA-ES converged. (desired_loc_land = {}, eval_opt_loc_land_h = {}, should_replan = {})'.format(prefix_info, desired_loc_land, eval_opt_loc_land_h, should_replan))
      
      # Replan action
      optimal_pos_target_replan = 0
      iter_replan = 0
      if should_replan:
        replan_init_guess = [0.1, 0.1]
        replan_init_var   = 1.0
        has_finished_this_round = False
        while not has_finished_this_round:
          try:
            self._run_model_based_iteration = 0
            res = cma.fmin(f, replan_init_guess, replan_init_var,
                           bounds=[[self.catapult.POS_MIN, self.catapult.POS_MIN], [self.catapult.POS_MAX, self.catapult.POS_MAX]], 
                           popsize=20, tolx=0.0001, verb_disp=False, verb_log=0)
            has_finished_this_round = True
          except:
            has_finished_this_round = False
        
        optimal_pos_target = res[0][0]
        optimal_pos_target_replan = optimal_pos_target
        iter_replan = self._run_model_based_iteration
        logger.log('{} result = {}'.format(prefix_info, res))
        logger.log('{} optimal solution found after replan. (pos_target = {}, pos_init === {}, duration === {})'.format(prefix_info, optimal_pos_target, self._FIXED_POS_INIT, self._FIXED_DURATION))
        logger.log('')
      
      # Test in true dynamics
      logger.log('{} test in true dynamics. (pos_init = {}, pos_target = {}, duration = {})'.format(prefix_info, self._FIXED_POS_INIT, optimal_pos_target, self._FIXED_DURATION))
      loc_land = catapult.throw_linear(self._FIXED_POS_INIT, optimal_pos_target, self._FIXED_DURATION)
      logger.log('{} loc_land = {}, desired_loc_land = {}'.format(prefix_info, loc_land, desired_loc_land))

      # Add to test results
      entry = {
        'approach': 'offline, NN(policy; deterministic), NN(dynamics), policy(init_action), CMA-ES(action; init_action, var~err), [model-based(action) if err>threshold]',
        'desired_loc_land': float(desired_loc_land),
        'loc_land': float(loc_land),
        'pos_target': float(optimal_pos_target),
        'preopt_samples': int(len(self._dataset)),
        'samples': int(0),
        'preopt_simulations': int(0),
        'simulations': int(iter_cma + iter_replan),
        'iter_cma': int(iter_cma),
        'optimal_pos_target_cma': float(optimal_pos_target_cma),
        'should_replan': bool(should_replan),
        'iter_replan': int(iter_replan),
        'optimal_pos_target_replan': float(optimal_pos_target_replan)
      }
      test_results.append(entry)
      logger.log('{} test result added to temperary result dataset >>> '.format(prefix_info))
      logger.log(entry)

    # Estimate test results
    self._estimate_test_results(test_results)
  
  def _run_hybrid_nn2_online(self):
    """
    hybrid:
      - online
      - NN(dynamics)
      - NN(policy; deterministic, NN(dynamics))
      - policy(init_action)
      - CMA-ES(action; init_action, var~err)
      - [model-based if err>threshold]
    """
    prefix = 'catapult_sim/hybrid_nn2_online'
    prefix_info = prefix + ':'
    
    MIN_LOC_LAND = -1
    MAX_LOC_LAND = 105
    THRESHOLD_ACTION_REPLAN_ERR = 0.05 * (MAX_LOC_LAND - MIN_LOC_LAND)
    
    POLICY_NETWORK_SAMPLES = 100
    
    # Load policy network validation dataset
    x_valid_policy = []
    y_valid_policy = []
    for i in range(len(self._dataset)):
      if i % 3 == 0:
        entry = self._dataset[i]
        x_valid_policy.append([entry['result']['loc_land']])
        y_valid_policy.append([entry['action']['pos_target']])
    
    # Create dynamics model
    model_dynamics = self._create_model(1, 1, hiddens=[200, 200], max_updates=10000, should_load_model=False, prefix_info=prefix_info)
    
    # Define dynamics model online training dataset
    x_train_dynamics = []
    y_train_dynamics = []
    
    # Load dynamics model validation dataset
    x_valid_dynamics = []
    y_valid_dynamics = []
    for i in range(len(self._dataset)):
      if i % 3 == 0:
        entry = self._dataset[i]
        x_valid_dynamics.append([entry['action']['pos_target']])
        y_valid_dynamics.append([entry['result']['loc_land']])
    
    # Test in real dynamics
    test_sample_desired_loc_land = [float(0.0 + np.random.sample() * 100.0) for i in range(100)]
    test_results = []
    for i in range(len(test_sample_desired_loc_land)):
      # Query desired landing location
      desired_loc_land = test_sample_desired_loc_land[i]
      
      # Create policy network
      model_policy = self._create_model(1, 1, hiddens=[200, 200], max_updates=2000, should_load_model=False, prefix_info=prefix_info) # re-create at each iteration
      
      # Sample policy network online training dataset from dynamics model
      x_train_policy = [] # re-sample from dynamics model at each iteration
      y_train_policy = [[float(self.catapult.POS_MIN + np.random.sample() * (self.catapult.POS_MAX - self.catapult.POS_MIN))] for i in range(POLICY_NETWORK_SAMPLES)] # re-sample from dynamics model at each iteration
      for i_policy_sample in range(len(y_train_policy)):
        y_train_policy_sample = y_train_policy[i_policy_sample]
        prediction_policy_sample_x = model_dynamics.Predict([y_train_policy_sample], x_var=0.0**2, with_var=True, with_grad=True)
        prediction_policy_sample_x_h    = (prediction_policy_sample_x.Y.ravel())[0]
        prediction_policy_sample_x_err  = (np.sqrt(np.diag(prediction_policy_sample_x.Var)))[0]
        prediction_policy_sample_x_grad = (prediction_policy_sample_x.Grad.ravel())[0]
        x_train_policy.append([prediction_policy_sample_x_h])
      
      # Train policy network
      logger.log('{} Train new policy network with samples from dynamics model. (samples: {})'.format(prefix_info, POLICY_NETWORK_SAMPLES))
      self._train_model(model_policy, x_train_policy, y_train_policy)
      
      # Estimate policy network quality
      policy_ave_stderr_y, policy_ave_stderr_err = self._estimate_model_quality(model_policy, x_train_policy, y_train_policy, x_valid_policy, y_valid_policy, should_plot=False)
      logger.log('{} policy_ave_stderr_y = {}, policy_ave_stderr_err = {}'.format(prefix_info, policy_ave_stderr_y, policy_ave_stderr_err))
      
      # Estimate dynamics model quality
      dynamics_ave_stderr_y, dynamics_ave_stderr_err = self._estimate_model_quality(model_dynamics, x_train_dynamics, y_train_dynamics, x_valid_dynamics, y_valid_dynamics, should_plot=False)
      logger.log('{} dynamics_ave_stderr_y = {}, dynamics_ave_stderr_err = {}'.format(prefix_info, dynamics_ave_stderr_y, dynamics_ave_stderr_err))
      
      # Predict initial action from policy network
      prediction_policy = model_policy.Predict([desired_loc_land], x_var=0.0**2, with_var=True, with_grad=True)
      pos_target_h    = (prediction_policy.Y.ravel())[0]
      pos_target_err  = (np.sqrt(np.diag(prediction_policy.Var)))[0]
      pos_target_grad = (prediction_policy.Grad.ravel())[0]
      logger.log('{} Predict initial action from policy network. (desired_loc_land = {}, pos_target_h = {}, pos_target_err = {})'.format(prefix_info, desired_loc_land, pos_target_h, pos_target_err))
      
      # Fix initial action
      original_pos_target_h = pos_target_h
      if pos_target_h < self.catapult.POS_MIN: pos_target_h = self.catapult.POS_MIN
      if pos_target_h > self.catapult.POS_MAX: pos_target_h = self.catapult.POS_MAX
      logger.log('{} Fix initial action. (pos_target_h = {} ({}))'.format(prefix_info, pos_target_h, original_pos_target_h))
      
      # Evaluate initial action
      eval_prediction = model_dynamics.Predict([pos_target_h], x_var=0.0**2, with_var=True, with_grad=True)
      eval_loc_land_h    = (eval_prediction.Y.ravel())[0]
      eval_loc_land_err  = (np.sqrt(np.diag(eval_prediction.Var)))[0]
      eval_loc_land_grad = (eval_prediction.Grad.ravel())[0]
      logger.log('{} Evaluate initial action with dynamics model. (desired_loc_land = {}, eval_loc_land_h = {})'.format(prefix_info, desired_loc_land, eval_loc_land_h))
      
      # Optimize parameters with CMA-ES and initial guess
      init_guess = [pos_target_h, 0.1]
      init_var   = (self.catapult.POS_MAX - self.catapult.POS_MIN) * min(abs(desired_loc_land - eval_loc_land_h), (MAX_LOC_LAND - MIN_LOC_LAND)) / (MAX_LOC_LAND - MIN_LOC_LAND)
      logger.log('{} Optimize action by CMA-ES with model. (init_guess: {}, init_var: {})'.format(prefix_info, pos_target_h, init_var))
      self._run_model_based_iteration = 0
      def f(x):
        self._run_model_based_iteration += 1
        pos_target, x_1 = x
        logger.log('{} sample from CMA-ES. (iteration = {}, desired_loc_land = {}, pos_target = {})'.format(prefix_info, self._run_model_based_iteration, desired_loc_land, pos_target))
        prediction = model_dynamics.Predict([pos_target], x_var=0.0**2, with_var=True, with_grad=True)
        loc_land_h    = (prediction.Y.ravel())[0]
        loc_land_err  = (np.sqrt(np.diag(prediction.Var)))[0]
        loc_land_grad = (prediction.Grad.ravel())[0]
        loss = 0.5 * (desired_loc_land - loc_land_h)**2
        logger.log('{} loss = {}, loc_land_h = {}, loc_land_err = {}'.format(prefix_info, loss, loc_land_h, loc_land_err))
        logger.log('')
        return loss
      
      has_finished_this_round = False
      while not has_finished_this_round:
        try:
          self._run_model_based_iteration = 0
          res = cma.fmin(f, init_guess, init_var,
                         bounds=[[self.catapult.POS_MIN + 0.01 * math.pi, self.catapult.POS_MIN + 0.01 * math.pi], [self.catapult.POS_MAX, self.catapult.POS_MAX]], 
                         popsize=20, tolx=0.0001, verb_disp=False, verb_log=0)
          has_finished_this_round = True
        except:
          has_finished_this_round = False
      
      optimal_pos_target = res[0][0]
      optimal_pos_target_cma = optimal_pos_target
      iter_cma = self._run_model_based_iteration
      logger.log('{} result = {}'.format(prefix_info, res))
      logger.log('{} optimal solution found. (pos_target = {}, pos_init === {}, duration === {})'.format(prefix_info, optimal_pos_target, self._FIXED_POS_INIT, self._FIXED_DURATION))
      logger.log('')
      
      # Evaluate optimal action after CMA-ES converged
      eval_opt_prediction = model_dynamics.Predict([optimal_pos_target], x_var=0.0**2, with_var=True, with_grad=True)
      eval_opt_loc_land_h    = (eval_opt_prediction.Y.ravel())[0]
      eval_opt_loc_land_err  = (np.sqrt(np.diag(eval_opt_prediction.Var)))[0]
      eval_opt_loc_land_grad = (eval_opt_prediction.Grad.ravel())[0]
      should_replan = (abs(eval_opt_loc_land_h - desired_loc_land) > THRESHOLD_ACTION_REPLAN_ERR)
      logger.log('{} Evaluate optimal action after CMA-ES converged. (desired_loc_land = {}, eval_opt_loc_land_h = {}, should_replan = {})'.format(prefix_info, desired_loc_land, eval_opt_loc_land_h, should_replan))
      
      # Replan action
      optimal_pos_target_replan = 0
      iter_replan = 0
      if should_replan:
        replan_init_guess = [0.1, 0.1]
        replan_init_var   = 1.0
        has_finished_this_round = False
        while not has_finished_this_round:
          try:
            self._run_model_based_iteration = 0
            res = cma.fmin(f, replan_init_guess, replan_init_var,
                           bounds=[[self.catapult.POS_MIN + 0.01 * math.pi, self.catapult.POS_MIN + 0.01 * math.pi], [self.catapult.POS_MAX, self.catapult.POS_MAX]], 
                           popsize=20, tolx=0.0001, verb_disp=False, verb_log=0)
            has_finished_this_round = True
          except:
            has_finished_this_round = False
        
        optimal_pos_target = res[0][0]
        optimal_pos_target_replan = optimal_pos_target
        iter_replan = self._run_model_based_iteration
        logger.log('{} result = {}'.format(prefix_info, res))
        logger.log('{} optimal solution found after replan. (pos_target = {}, pos_init === {}, duration === {})'.format(prefix_info, optimal_pos_target, self._FIXED_POS_INIT, self._FIXED_DURATION))
        logger.log('')
      
      # Test in true dynamics
      logger.log('{} test in true dynamics. (pos_init = {}, pos_target = {}, duration = {})'.format(prefix_info, self._FIXED_POS_INIT, optimal_pos_target, self._FIXED_DURATION))
      loc_land = catapult.throw_linear(self._FIXED_POS_INIT, optimal_pos_target, self._FIXED_DURATION)
      logger.log('{} loc_land = {}, desired_loc_land = {}'.format(prefix_info, loc_land, desired_loc_land))
      
      # Add to test results
      entry = {
        'approach': 'hybrid, online, NN(dynamics), NN(policy; deterministic, NN(dynamics)), policy(init_action), CMA-ES(action; init_action, var~err), [model-based if err>threshold]',
        'desired_loc_land': float(desired_loc_land),
        'loc_land': float(loc_land),
        'pos_target': float(optimal_pos_target),
        'preopt_samples': int(0),
        'samples': int(len(x_train_dynamics)),
        'preopt_simulations': int(0),
        'simulations': int(iter_cma + iter_replan),
        'iter_cma': int(iter_cma),
        'optimal_pos_target_cma': float(optimal_pos_target_cma),
        'should_replan': bool(should_replan),
        'iter_replan': int(iter_replan),
        'optimal_pos_target_replan': float(optimal_pos_target_replan),
        'dynamics_ave_stderr_y': float(dynamics_ave_stderr_y),
        'dynamics_ave_stderr_err': float(dynamics_ave_stderr_err),
        'policy_ave_stderr_y': float(policy_ave_stderr_y),
        'policy_ave_stderr_err': float(policy_ave_stderr_err)
      }
      test_results.append(entry)
      logger.log('{} test result added to temperary result dataset >>> '.format(prefix_info))
      logger.log(entry)
      
      # Update dynamics model online training dataset and train dynamics model
      x_train_dynamics.append([optimal_pos_target])
      y_train_dynamics.append([loc_land])
      model_dynamics.Update([optimal_pos_target], [loc_land], not_learn=False)
    
    # Estimate test results
    self._estimate_test_results(test_results)
  
  def getOperations(self):
    operation_dict = {
      'mb': self._run_model_based,
      'mb_online': self._run_model_based_online,
      'mf': self._run_model_free,
      'mf_nn': self._run_model_free_nn,
      'mf_nn_offline': self._run_model_free_nn_offline,
      'hybrid': self._run_hybrid,
      'hybrid_nn2_offline': self._run_hybrid_nn2_offline,
      'hybrid_nn2_online': self._run_hybrid_nn2_online
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
  catapult_name = 'sim_002_01'

  catapult = TCatapultSim(dirpath_sim)
  
  abs_dirpath_data = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/catapult_' + catapult_name))
  abs_dirpath_model = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/model_' + catapult_name))
  abs_dirpath_log = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/log_' + catapult_name))
  agent = TCatapultLPLinearSim(catapult, abs_dirpath_data, abs_dirpath_model, abs_dirpath_log)
  
  operation = 'mb'
  if len(sys.argv) >= 2:
    if len(sys.argv) == 2 and (sys.argv[1] in agent.getOperations()):
      operation = sys.argv[1]
    else:
      logger.log('usage: ./run_sim_002_linear_thetaT_d.py <operation>')
      quit()
  
  agent.run(operation)


