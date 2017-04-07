#!/usr/bin/python

"""
Catapult linear motion control learning and planning.
"""

__author__    = 'David Qiu'
__email__     = 'david@davidqiu.com'
__website__   = 'www.davidqiu.com'
__copyright__ = 'Copyright (C) 2017, David Qiu. All rights reserved.'


from catapult import *

import os
import time
import datetime
import yaml
import sys
import numpy as np

import pdb

sys.path.append(os.path.join(os.path.dirname(__file__), '../lib/cma'))
import cma



class TCatapultLPLinear(object):
  """
  Catapult learning and planning agent in linear motion control.
  """
  
  def __init__(self, catapult, abs_dirpath_data=None):
    super(TCatapultLPLinear, self).__init__()
    
    self.catapult = catapult
    
    self._abs_dirpath_data = abs_dirpath_data
    
    self.reset()
  
  def reset(self):
    self.catapult.move(self.catapult.POS_LOAD)
  
  def throw(self, pos_init, pos_target, duration):
    self.catapult.move(pos_init, duration=1.0, interval=0.01, wait=False)
    time.sleep(1.0)
    pos_init_actual = self.catapult.position
    
    self.catapult.move(pos_target, duration=duration, interval=0.01, wait=False, motion=self.catapult.MOTION_LINEAR)
    time.sleep(1.0)
    pos_target_actual = self.catapult.position
    
    self.reset()
    
    return pos_init_actual, pos_target_actual
  
  def _launch_test(self, dataset, face_init, pos_init, pos_target, duration, check_thrown=False, prefix='catapult'):
    prefix_info = prefix + ':'
    
    captured = False
    while not captured:
      print prefix_info, 'face_init = {}, pos_init = {}, pos_target = {}, duration = {}'.format(face_init, pos_init, pos_target, duration)
      input_ready = raw_input(prefix_info + ' ready (Y)?> ')
      pos_init_actual, pos_target_actual = self.throw(pos_init, pos_target, duration)
      print prefix_info, 'pos_init_actual = {}, pos_target_actual = {}'.format(pos_init_actual, pos_target_actual)
      
      input_captured = raw_input(prefix_info + ' captured (Y/n)?> ')
      if input_captured == '' or input_captured == 'y' or input_captured == 'Y':
        captured = True
      else:
        captured = False
      
      thrown = True
      if check_thrown and not captured:
        input_thrown = raw_input(prefix_info + ' thrown (Y/n)?> ')
        if input_thrown == '' or input_thrown == 'y' or input_thrown == 'Y':
          thrown = True
        else:
          thrown = False
        
        if not thrown:
          print prefix_info, 'not thrown.'
          return None
      
      if captured:
        confirmed = False
        while not confirmed:
          input_loc_land = raw_input(prefix_info + ' loc_land = ')
          loc_land = int(input_loc_land)
          input_loc_stop = raw_input(prefix_info + ' loc_stop = ')
          loc_stop = int(input_loc_stop)
          input_face_stop = raw_input(prefix_info + ' face_stop = ')
          face_stop = str(input_face_stop)
          
          input_confirmed = raw_input(prefix_info + ' confirmed (Y/n)?> ')
          if input_confirmed == '' or input_confirmed == 'y' or input_confirmed == 'Y':
            confirmed = True
          else:
            confirmed = False
          
          if not confirmed:
            input_captured = raw_input(prefix_info + ' not confirmed but captured (Y/n)?> ')
            if input_captured == '' or input_captured == 'y' or input_captured == 'Y':
              captured = True
            else:
              captured = False
              break
    
    entry = dataset.new_entry_linear(face_init, pos_init, pos_init_actual, pos_target, pos_target_actual, duration, loc_land, loc_stop, face_stop)
    dataset.append(entry)
    
    return entry
  
  def _run_data_collection(self):
    prefix = 'catapult/data_collection'
    prefix_info = prefix + ':'
    
    if self._abs_dirpath_data is None:
      dataset = TCatapultDataset()
    else:
      dataset = TCatapultDataset(abs_dirpath=self._abs_dirpath_data)
    
    feature_dict = {
      'face_init': ['1'],
      'pos_init': [0],#[50, 100, 200],
      'pos_target': [480],#[500],
      'duration': [0.10]#[0.2, 0.15, 0.10, 0.05]
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
    
    n_samples = 3
    count_feature = 0
    for feature_comb in feature_space:
      count_feature += 1
      for i in range(n_samples):
        print prefix_info, 'samples: {}/{}, feature: {}/{}'.format(i+1, n_samples, count_feature, len(feature_space))
        self._launch_test(dataset, feature_comb['face_init'], feature_comb['pos_init'], feature_comb['pos_target'], feature_comb['duration'], prefix=prefix)
        print ''
    
    print 'datafile:', dataset.append_filepath
  
  def _check_action(self, pos_init, pos_target, duration):
    if pos_init < self.catapult.POS_MIN: return False
    if pos_init > self.catapult.POS_MID: return False
    if pos_target < pos_init: return False
    if pos_target > self.catapult.POS_MAX: return False
    if duration < 0.01: return False
    if duration > 2.00: return False
    return True
  
  def _correct_action(self, pos_init, pos_target, duration):
    if pos_init < self.catapult.POS_MIN: pos_init = self.catapult.POS_MIN
    if pos_init > self.catapult.POS_MID: pos_init = self.catapult.POS_MID
    if pos_target < pos_init: pos_target = pos_init
    if pos_target > self.catapult.POS_MAX: pos_target = self.catapult.POS_MAX
    if duration < 0.01: duration = 0.01
    if duration > 2.00: duration = 2.00
    duration = np.round(duration, 2)
    return pos_init, pos_target, duration
  
  def _penalize_action(self, pos_init, pos_target, duration):
    prefix = 'catapult/penalize_action'
    prefix_info = prefix + ':'
    
    corrected_pos_init   = pos_init
    corrected_pos_target = pos_target
    corrected_duration   = duration
    
    penalty = 0
    penalty_factor = 1
    
    min_pos_diff = 20
    
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
    
    if pos_target <= (corrected_pos_init + min_pos_diff):
      cur_penalty = np.abs(pos_target - (corrected_pos_init + min_pos_diff)) * penalty_factor
      penalty += cur_penalty
      print prefix_info, 'penalty = {} ({})'.format(cur_penalty, 'pos_target <= (corrected_pos_init + min_pos_diff)')
      corrected_pos_target = (corrected_pos_init + min_pos_diff)
    if pos_target > self.catapult.POS_MAX:
      cur_penalty = np.abs(pos_target - self.catapult.POS_MAX) * penalty_factor
      penalty += cur_penalty
      print prefix_info, 'penalty = {} ({})'.format(cur_penalty, 'pos_target > self.catapult.POS_MAX')
      corrected_pos_target = self.catapult.POS_MAX
    
    if duration < 0.01:
      cur_penalty = np.abs(duration - 0.01) * 1000 * penalty_factor
      penalty += cur_penalty
      print prefix_info, 'penalty = {} ({})'.format(cur_penalty, 'duration < 0.01')
      corrected_duration = 0.01
    if duration > 0.6:
      cur_penalty = np.abs(duration - 0.6) * 1000 * penalty_factor
      penalty += cur_penalty
      print prefix_info, 'penalty = {} ({})'.format(cur_penalty, 'duration > 0.6')
      corrected_duration = 0.6
    
    return corrected_pos_init, corrected_pos_target, corrected_duration, penalty
  
  def _run_cma_throw_farther(self):
    prefix = 'catapult/cma_throw_farther'
    prefix_info = prefix + ':'
    
    if self._abs_dirpath_data is None:
      dataset = TCatapultDataset()
    else:
      dataset = TCatapultDataset(abs_dirpath=self._abs_dirpath_data)
    
    self._run_cma_throw_farther_INIT_GUESS = [200.0, 480.0, 0.3 * 1000.]
    self._run_cma_throw_farther_INIT_VAR   = 300.0
    self._run_cma_throw_farther_CONSTRAIN_ACTION = 'penalize' # {'check', 'correct', 'penalize'}
    
    self._run_cma_throw_farther_count_test = 0
    
    def f(x):
      self._run_cma_throw_farther_count_test += 1
      print prefix_info, 'optimizes with CMA-ES. (test = {})'.format(self._run_cma_throw_farther_count_test)
      
      pos_init, pos_target, duration_scaled = x
      duration = np.round(duration_scaled / 1000., 2)
      print prefix_info, 'sample from CMA-ES. (pos_init = {}, pos_target = {}, duration = {} ({}))'.format(pos_init, pos_target, duration, duration_scaled)
      
      if self._run_cma_throw_farther_CONSTRAIN_ACTION == 'check':
        is_action_checked = self._check_action(pos_init, pos_target, duration)
        entry = None
        if is_action_checked and pos_init != pos_target:
          entry = self._launch_test(dataset, '1', int(pos_init), int(pos_target), float(duration), check_thrown=True, prefix=prefix)
        loss = 0 if entry is None else -float(entry['result']['loc_land'])
        print prefix_info, 'loss = {}'.format(loss)
        
      elif self._run_cma_throw_farther_CONSTRAIN_ACTION == 'correct':
        pos_init, pos_target, duration = self._correct_action(pos_init, pos_target, duration)
        entry = None
        if pos_init != pos_target:
          entry = self._launch_test(dataset, '1', int(pos_init), int(pos_target), float(duration), check_thrown=True, prefix=prefix)
        loss = 0 if entry is None else -float(entry['result']['loc_land'])
        print prefix_info, 'loss = {}'.format(loss)
      
      elif self._run_cma_throw_farther_CONSTRAIN_ACTION == 'penalize':
        pos_init, pos_target, duration, penalty = self._penalize_action(pos_init, pos_target, duration)
        entry = self._launch_test(dataset, '1', int(pos_init), int(pos_target), float(duration), check_thrown=True, prefix=prefix)
        loss_raw = 0 if entry is None else -float(entry['result']['loc_land'])
        loss = loss_raw + penalty
        print prefix_info, 'loss = {}, penalty = {}'.format(loss, penalty)
      
      print ''
      
      return loss
    
    res = cma.fmin(f, self._run_cma_throw_farther_INIT_GUESS, self._run_cma_throw_farther_INIT_VAR, popsize=10, tolx=5.0, verb_disp=False, verb_log=0)
    print prefix_info, 'result =', res
    print prefix_info, 'optimal solution found. (pos_init = {}, pos_target = {}, duration = {} ({}))'.format(res[0][0], res[0][1], round(res[0][2] / 1000., 2), res[0][2])
  
  def _run_same_throw(self):
    prefix = 'catapult/same_throw'
    prefix_info = prefix + ':'
    
    face_init = '1'
    pos_init = 200
    pos_target = 400
    duration = 0.01
    print prefix_info, 'face_init = {}, pos_init = {}, pos_target = {}, duration = {}'.format(face_init, pos_init, pos_target, duration)
    
    input_ready = raw_input(prefix_info + ' ready (Y)?> ')
    pos_init_actual, pos_target_actual = self.throw(pos_init, pos_target, duration)
    print prefix_info, 'pos_init_actual = {}, pos_target_actual = {}'.format(pos_init_actual, pos_target_actual)
    print prefix_info, 'expected_land_loc = ~{}'.format(1200)
  
  def _run_check_dataset(self):
    prefix = 'catapult/check_dataset'
    prefix_info = prefix + ':'
    
    VALID_FACE_INIT           = ['1', '2', '3', '4']
    VALID_FACE_STOP           = ['1', '2', '3', '4', 'side']
    VALID_POS_DIFF_THRESHOLD  = 10
    
    if self._abs_dirpath_data is None:
      dataset = TCatapultDataset(auto_init=False)
    else:
      dataset = TCatapultDataset(abs_dirpath=self._abs_dirpath_data, auto_init=False)
    dataset.load_dataset()
    
    count_invalid_entries = 0
    count_suspicious_entries = 0
    for entry in dataset:
      is_invalid = False
      is_suspicious = False
      
      if not (entry['motion'] == 'linear'): is_invalid = True
      
      if not (type(entry['action']) is dict): is_invalid = True
      if entry['action']['face_init'] is None: is_invalid = True
      if entry['action']['face_init'] not in VALID_FACE_INIT: is_invalid = True
      if entry['action']['pos_init'] is None: is_invalid = True
      if entry['action']['pos_init_actual'] is None: is_invalid = True
      if np.abs(entry['action']['pos_init'] - entry['action']['pos_init_actual']) >= VALID_POS_DIFF_THRESHOLD: is_suspicious = True
      if entry['action']['pos_target'] is None: is_invalid = True
      if entry['action']['pos_target_actual'] is None: is_invalid = True
      if np.abs(entry['action']['pos_target'] - entry['action']['pos_target_actual']) >= VALID_POS_DIFF_THRESHOLD: is_suspicious = True
      if np.abs(entry['action']['pos_init']   - entry['action']['pos_init_actual'])   == 0 and \
         np.abs(entry['action']['pos_target'] - entry['action']['pos_target_actual']) == 0: is_suspicious = True
      if entry['action']['duration'] is None: is_invalid = True
      if entry['action']['duration'] < 0: is_invalid = True
      
      if not (type(entry['result']) is dict): is_invalid = True
      if entry['result']['face_stop'] is None: is_invalid = True
      if entry['result']['face_stop'] not in VALID_FACE_STOP: is_invalid = True
      if entry['result']['loc_land'] is None: is_invalid = True
      if entry['result']['loc_stop'] is None: is_invalid = True
      
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
  catapult_name = 'catapult_001'
  pos_base = 2300
  
  catapult = TCatapult(reset=False, _pos_base=pos_base)
  
  abs_dirpath_data = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/' + catapult_name))
  agent = TCatapultLPLinear(catapult, abs_dirpath_data=abs_dirpath_data)
  
  operation = 'check_dataset'
  if len(sys.argv) >= 2:
    if len(sys.argv) == 2 and (sys.argv[1] in agent.getOperations()):
      operation = sys.argv[1]
    else:
      print 'usage: ./run_001_linear.py <operation>'
      quit()
  
  agent.run(operation)


