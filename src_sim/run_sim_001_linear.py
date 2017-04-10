#!/usr/bin/python

"""
Simulation catapult linear motion control learning and planning.
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

import pdb

sys.path.append(os.path.join(os.path.dirname(__file__), '../lib/cma'))
import cma



class TCatapultLPLinearSim(object):
  """
  Simulation catapult learning and planning agent in linear motion control.
  """
  
  def __init__(self, catapult, abs_dirpath_data=None):
    super(TCatapultLPLinearSim, self).__init__()
    
    self.catapult = catapult
    
    self._abs_dirpath_data = abs_dirpath_data
  
  def _launch_test(self, dataset, pos_init, pos_target, duration, prefix='catapult_sim'):
    prefix_info = prefix + ':'
    
    print prefix_info, 'launch test. (pos_init = {}, pos_target = {}, duration = {})'.format(pos_init, pos_target, duration)
    
    loc_land = catapult.throw_linear(pos_init, pos_target, duration)
    
    print prefix_info, 'loc_land = {}'.format(loc_land)
    
    entry = dataset.new_entry_linear_sim(float(pos_init), float(pos_target), float(duration), float(loc_land))
    dataset.append(entry)
    
    return entry
  
  def _run_data_collection(self):
    prefix = 'catapult_sim/data_collection'
    prefix_info = prefix + ':'
    
    dataset = TCatapultDatasetSim(abs_dirpath=self._abs_dirpath_data)
    
    feature_dict = {
      'pos_init': np.array([0.1, 0.2, 0.3, 0.4]) * math.pi,
      'pos_target': np.array([0.5, 0.6, 0.7, 0.8, 0.9]) * math.pi,
      'duration': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    }
    
    feature_space = []
    for feature in feature_dict:
      if len(feature_space) == 0:
        for feature_i in feature_dict[feature]:
          feature_space.append({feature: feature_i})
      else:
        new_feature_space = []
        for feature_i in feature_dict[feature]:
          for feature_space_i in feature_space:
            new_feature_space_i = feature_space_i.copy() # deep copy
            new_feature_space_i[feature] = feature_i
            new_feature_space.append(new_feature_space_i)
        feature_space = new_feature_space
    
    n_samples = 1
    count_feature = 0
    for feature_comb in feature_space:
      count_feature += 1
      for i in range(n_samples):
        print prefix_info, 'samples: {}/{}, feature: {}/{}'.format(i+1, n_samples, count_feature, len(feature_space))
        self._launch_test(dataset, feature_comb['pos_init'], feature_comb['pos_target'], feature_comb['duration'], prefix=prefix)
        print ''
    
    print 'datafile:', dataset.append_filepath
  
  def _check_action(self, pos_init, pos_target, duration):
    if pos_init < self.catapult.POS_MIN: return False
    if pos_init > self.catapult.POS_MID: return False
    if pos_target < pos_init: return False
    if pos_target > self.catapult.POS_MAX: return False
    if duration < self.catapult.DURATION_MIN: return False
    if duration > 0.60: return False
    return True
  
  def _correct_action(self, pos_init, pos_target, duration):
    if pos_init < self.catapult.POS_MIN: pos_init = self.catapult.POS_MIN
    if pos_init > self.catapult.POS_MID: pos_init = self.catapult.POS_MID
    if pos_target < pos_init: pos_target = pos_init
    if pos_target > self.catapult.POS_MAX: pos_target = self.catapult.POS_MAX
    if duration < self.catapult.DURATION_MIN: duration = self.catapult.DURATION_MIN
    if duration > 0.60: duration = 0.60
    duration = np.round(duration, 2)
    return pos_init, pos_target, duration
  
  def _penalize_action(self, pos_init, pos_target, duration):
    prefix = 'catapult_sim/penalize_action'
    prefix_info = prefix + ':'
    
    corrected_pos_init   = pos_init
    corrected_pos_target = pos_target
    corrected_duration   = duration
    
    penalty = 0
    penalty_factor = 1
    
    min_pos_diff = 0.1 * math.pi
    
    if pos_init < self.catapult.POS_MIN:
      cur_penalty = np.abs(pos_init - self.catapult.POS_MIN) * penalty_factor
      penalty += cur_penalty
      print prefix_info, 'penalty = {} ({})'.format(cur_penalty, 'pos_init < self.catapult.POS_MIN')
      corrected_pos_init = self.catapult.POS_MIN
    if pos_init > self.catapult.POS_MID:
      cur_penalty = np.abs(pos_init - self.catapult.POS_MID) * penalty_factor
      penalty += cur_penalty
      print prefix_info, 'penalty = {} ({})'.format(cur_penalty, 'pos_init > self.catapult.POS_MID')
      corrected_pos_init = self.catapult.POS_MID
    
    if pos_target < (corrected_pos_init + min_pos_diff):
      cur_penalty = np.abs(pos_target - (corrected_pos_init + min_pos_diff)) * penalty_factor
      penalty += cur_penalty
      print prefix_info, 'penalty = {} ({})'.format(cur_penalty, 'pos_target <= (corrected_pos_init + min_pos_diff)')
      corrected_pos_target = (corrected_pos_init + min_pos_diff)
    if pos_target > self.catapult.POS_MAX:
      cur_penalty = np.abs(pos_target - self.catapult.POS_MAX) * penalty_factor
      penalty += cur_penalty
      print prefix_info, 'penalty = {} ({})'.format(cur_penalty, 'pos_target > self.catapult.POS_MAX')
      corrected_pos_target = self.catapult.POS_MAX
    
    if duration < self.catapult.DURATION_MIN:
      cur_penalty = np.abs(duration - self.catapult.DURATION_MIN) * penalty_factor
      penalty += cur_penalty
      print prefix_info, 'penalty = {} ({})'.format(cur_penalty, 'duration < self.catapult.DURATION_MIN')
      corrected_duration = self.catapult.DURATION_MIN
    if duration > 0.6:
      cur_penalty = np.abs(duration - 0.6) * penalty_factor
      penalty += cur_penalty
      print prefix_info, 'penalty = {} ({})'.format(cur_penalty, 'duration > 0.6')
      corrected_duration = 0.6
    
    return corrected_pos_init, corrected_pos_target, corrected_duration, penalty
  
  def _run_cma_throw_farther(self):
    prefix = 'catapult_sim/cma_throw_farther'
    prefix_info = prefix + ':'
    
    dataset = TCatapultDatasetSim(abs_dirpath=self._abs_dirpath_data)
    
    self._run_cma_throw_farther_INIT_GUESS = [0.2 * math.pi, 0.6 * math.pi, 0.3]
    self._run_cma_throw_farther_INIT_VAR   = 0.5
    self._run_cma_throw_farther_CONSTRAIN_ACTION = 'penalize' # {'check', 'correct', 'penalize'}
    
    self._run_cma_throw_farther_count_test = 0
    
    def f(x):
      self._run_cma_throw_farther_count_test += 1
      print prefix_info, 'optimizes with CMA-ES. (test = {})'.format(self._run_cma_throw_farther_count_test)
      
      pos_init, pos_target, duration = x
      print prefix_info, 'sample from CMA-ES. (pos_init = {}, pos_target = {}, duration = {})'.format(pos_init, pos_target, duration)
      
      if self._run_cma_throw_farther_CONSTRAIN_ACTION == 'check':
        is_action_checked = self._check_action(pos_init, pos_target, duration)
        entry = None
        if is_action_checked and pos_init != pos_target:
          entry = self._launch_test(dataset, pos_init, pos_target, duration, prefix=prefix)
        loss = 1 if entry is None else -float(entry['result']['loc_land'])
        print prefix_info, 'loss = {}'.format(loss)
        
      elif self._run_cma_throw_farther_CONSTRAIN_ACTION == 'correct':
        pos_init, pos_target, duration = self._correct_action(pos_init, pos_target, duration)
        entry = None
        if pos_init != pos_target:
          entry = self._launch_test(dataset, pos_init, pos_target, duration, prefix=prefix)
        loss = 1 if entry is None else -float(entry['result']['loc_land'])
        print prefix_info, 'loss = {}'.format(loss)
      
      elif self._run_cma_throw_farther_CONSTRAIN_ACTION == 'penalize':
        pos_init, pos_target, duration, penalty = self._penalize_action(pos_init, pos_target, duration)
        entry = self._launch_test(dataset, pos_init, pos_target, duration, prefix=prefix)
        loss_raw = 1 if entry is None else -float(entry['result']['loc_land'])
        loss = loss_raw + penalty
        print prefix_info, 'loss = {}, penalty = {}'.format(loss, penalty)
      
      print ''
      
      return loss
    
    res = cma.fmin(f, self._run_cma_throw_farther_INIT_GUESS, self._run_cma_throw_farther_INIT_VAR, 
                   popsize=20, tolx=0.001, verb_disp=False, verb_log=0)
    print prefix_info, 'result =', res
    print prefix_info, 'optimal solution found. (pos_init = {}, pos_target = {}, duration = {})'.format(res[0][0], res[0][1], res[0][2])
  
  def _run_cma_throw_farther_pos(self):
    prefix = 'catapult_sim/cma_throw_farther_pos'
    prefix_info = prefix + ':'
    
    dataset = TCatapultDatasetSim(abs_dirpath=self._abs_dirpath_data)
    
    self._run_cma_throw_farther_pos_DURATION   = self.catapult.DURATION_MIN
    self._run_cma_throw_farther_pos_INIT_GUESS = [0.2 * math.pi, 0.6 * math.pi]
    self._run_cma_throw_farther_pos_INIT_VAR   = 1.0
    self._run_cma_throw_farther_pos_CONSTRAIN_ACTION = 'penalize' # {'check', 'correct', 'penalize'}
    
    self._run_cma_throw_farther_pos_count_test = 0
    
    def f(x):
      self._run_cma_throw_farther_pos_count_test += 1
      print prefix_info, 'optimizes with CMA-ES. (test = {})'.format(self._run_cma_throw_farther_pos_count_test)
      
      pos_init, pos_target = x
      duration = self._run_cma_throw_farther_pos_DURATION
      print prefix_info, 'sample from CMA-ES. (pos_init = {}, pos_target = {})'.format(pos_init, pos_target)
      
      if self._run_cma_throw_farther_pos_CONSTRAIN_ACTION == 'check':
        is_action_checked = self._check_action(pos_init, pos_target, duration)
        entry = None
        if is_action_checked and pos_init != pos_target:
          entry = self._launch_test(dataset, pos_init, pos_target, duration, prefix=prefix)
        loss = 1 if entry is None else -float(entry['result']['loc_land'])
        print prefix_info, 'loss = {}'.format(loss)
        
      elif self._run_cma_throw_farther_pos_CONSTRAIN_ACTION == 'correct':
        pos_init, pos_target, duration = self._correct_action(pos_init, pos_target, duration)
        entry = None
        if pos_init != pos_target:
          entry = self._launch_test(dataset, pos_init, pos_target, duration, prefix=prefix)
        loss = 1 if entry is None else -float(entry['result']['loc_land'])
        print prefix_info, 'loss = {}'.format(loss)
      
      elif self._run_cma_throw_farther_pos_CONSTRAIN_ACTION == 'penalize':
        pos_init, pos_target, duration, penalty = self._penalize_action(pos_init, pos_target, duration)
        entry = self._launch_test(dataset, pos_init, pos_target, duration, prefix=prefix)
        loss_raw = 1 if entry is None else -float(entry['result']['loc_land'])
        loss = loss_raw + penalty
        print prefix_info, 'loss = {}, penalty = {}'.format(loss, penalty)
      
      print ''
      
      return loss
    
    res = cma.fmin(f, self._run_cma_throw_farther_pos_INIT_GUESS, self._run_cma_throw_farther_pos_INIT_VAR, 
                   popsize=20, tolx=0.001, verb_disp=False, verb_log=0)
    print prefix_info, 'result =', res
    print prefix_info, 'optimal solution found. (pos_init = {}, pos_target = {}, duration === {})'.format(res[0][0], res[0][1], self._run_cma_throw_farther_pos_DURATION)
  
  def _run_cma_ctrl_loc_land_pos(self, target_loc_land):
    prefix = 'catapult_sim/cma_ctrl_loc_land_pos'
    prefix_info = prefix + ':'

    dataset = TCatapultDatasetSim(abs_dirpath=self._abs_dirpath_data)
    
    self._run_cma_ctrl_loc_land_pos_DURATION   = self.catapult.DURATION_MIN
    self._run_cma_ctrl_loc_land_pos_INIT_GUESS = [0.2 * math.pi, 0.6 * math.pi]
    self._run_cma_ctrl_loc_land_pos_INIT_VAR   = 1.0
    self._run_cma_ctrl_loc_land_pos_CONSTRAIN_ACTION = 'penalize' # {'check', 'correct', 'penalize'}
    
    self._run_cma_ctrl_loc_land_pos_count_test = 0
    
    def f(x):
      self._run_cma_ctrl_loc_land_pos_count_test += 1
      print prefix_info, 'optimizes with CMA-ES. (test = {})'.format(self._run_cma_ctrl_loc_land_pos_count_test)
      
      max_loss = np.abs(target_loc_land) + 1

      pos_init, pos_target = x
      duration = self._run_cma_ctrl_loc_land_pos_DURATION
      print prefix_info, 'sample from CMA-ES. (pos_init = {}, pos_target = {})'.format(pos_init, pos_target)
      
      if self._run_cma_ctrl_loc_land_pos_CONSTRAIN_ACTION == 'check':
        is_action_checked = self._check_action(pos_init, pos_target, duration)
        entry = None
        if is_action_checked and pos_init != pos_target:
          entry = self._launch_test(dataset, pos_init, pos_target, duration, prefix=prefix)
        loss = max_loss if entry is None else np.abs(target_loc_land - float(entry['result']['loc_land']))
        print prefix_info, 'loss = {}'.format(loss)
        
      elif self._run_cma_ctrl_loc_land_pos_CONSTRAIN_ACTION == 'correct':
        pos_init, pos_target, duration = self._correct_action(pos_init, pos_target, duration)
        entry = None
        if pos_init != pos_target:
          entry = self._launch_test(dataset, pos_init, pos_target, duration, prefix=prefix)
        loss = max_loss if entry is None else np.abs(target_loc_land - float(entry['result']['loc_land']))
        print prefix_info, 'loss = {}'.format(loss)
      
      elif self._run_cma_ctrl_loc_land_pos_CONSTRAIN_ACTION == 'penalize':
        pos_init, pos_target, duration, penalty = self._penalize_action(pos_init, pos_target, duration)
        entry = self._launch_test(dataset, pos_init, pos_target, duration, prefix=prefix)
        loss_raw = max_loss if entry is None else np.abs(target_loc_land - float(entry['result']['loc_land']))
        loss = loss_raw + penalty
        print prefix_info, 'loss = {}, penalty = {}'.format(loss, penalty)
      
      print ''
      
      return loss
    
    res = cma.fmin(f, self._run_cma_ctrl_loc_land_pos_INIT_GUESS, self._run_cma_ctrl_loc_land_pos_INIT_VAR, 
                   popsize=20, tolx=0.001, verb_disp=False, verb_log=0)
    print prefix_info, 'result =', res
    print prefix_info, 'optimal solution found. (pos_init = {}, pos_target = {}, duration === {})'.format(res[0][0], res[0][1], self._run_cma_ctrl_loc_land_pos_DURATION)

  def _run_same_throw(self):
    prefix = 'catapult_sim/same_throw'
    prefix_info = prefix + ':'
    
    pos_init = 0.1 * math.pi
    pos_target = 0.6 * math.pi
    duration = 0.10
    print prefix_info, 'pos_init = {}, pos_target = {}, duration = {}'.format(pos_init, pos_target, duration)
    
    loc_land = self.catapult.throw_linear(pos_init, pos_target, duration)
    print prefix_info, 'land_loc = {}'.format(loc_land)
  
  def _run_check_dataset(self):
    prefix = 'catapult_sim/check_dataset'
    prefix_info = prefix + ':'
    
    dataset = TCatapultDatasetSim(abs_dirpath=self._abs_dirpath_data, auto_init=False)
    dataset.load_dataset()
    
    count_invalid_entries = 0
    count_suspicious_entries = 0
    for entry in dataset:
      is_invalid = False
      is_suspicious = False
      
      if not (entry['motion'] == 'linear'): is_invalid = True
      
      if not (type(entry['action']) is dict): is_invalid = True
      if entry['action']['pos_init'] is None: is_invalid = True
      elif entry['action']['pos_init'] < self.catapult.POS_MIN: is_invalid = True
      elif entry['action']['pos_init'] > self.catapult.POS_MAX: is_invalid = True
      if entry['action']['pos_target'] is None: is_invalid = True
      elif entry['action']['pos_target'] < self.catapult.POS_MIN: is_invalid = True
      elif entry['action']['pos_target'] > self.catapult.POS_MAX: is_invalid = True
      if entry['action']['pos_init'] >= entry['action']['pos_target']: is_invalid = True
      if entry['action']['duration'] is None: is_invalid = True
      elif entry['action']['duration'] < self.catapult.DURATION_MIN: is_invalid = True
      
      if not (type(entry['result']) is dict): is_invalid = True
      if entry['result']['loc_land'] is None: is_invalid = True
      
      if is_invalid:
        count_invalid_entries += 1
        print prefix_info, 'invalid entry found >>> '
        print entry
        print ''
      
      elif is_suspicious:
        count_suspicious_entries += 1
        print prefix_info, 'suspicious entry found >>> '
        print entry
        print ''
    
    print prefix_info, 'entries    = {}'.format(dataset.size)
    print prefix_info, 'suspicious = {}/{}'.format(count_suspicious_entries, dataset.size)
    print prefix_info, 'invalid    = {}/{}'.format(count_invalid_entries, dataset.size)
  
  def getOperations(self):
    operation_dict = {
      'data_collection': self._run_data_collection,
      'cma_throw_farther': self._run_cma_throw_farther,
      'cma_throw_farther_pos': self._run_cma_throw_farther_pos,
      'cma_ctrl_loc_land_pos': self._run_cma_ctrl_loc_land_pos,
      'check_dataset': self._run_check_dataset,
      'same_throw': self._run_same_throw
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
  catapult_name = 'catapult_sim_001'

  catapult = TCatapultSim(dirpath_sim)
  
  abs_dirpath_data = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/' + catapult_name))
  agent = TCatapultLPLinearSim(catapult, abs_dirpath_data=abs_dirpath_data)
  
  operation = 'check_dataset'
  if len(sys.argv) >= 2:
    if len(sys.argv) == 2 and (sys.argv[1] in agent.getOperations()):
      operation = sys.argv[1]
    else:
      print 'usage: ./run_001_linear.py <operation>'
      quit()
  
  agent.run(operation)


