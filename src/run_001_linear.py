#!/usr/bin/python

"""
Catapult linear motion control learning and planning.
"""

__author__    = 'David Qiu'
__email__     = 'david@davidqiu.com'
__website__   = 'www.davidqiu.com'
__copyright__ = 'Copyright (C) 2017, David Qiu. All rights reserved.'


from catapult import *

import time
import datetime
import yaml
import sys

import pdb



class TCatapultLPLinear(object):
  """
  Catapult learning and planning agent in linear motion control.
  """
  
  def __init__(self, catapult):
    super(TCatapultLPLinear, self).__init__()
    
    self.catapult = catapult
    
    self.reset()
  
  def reset(self):
    self.catapult.move(self.catapult.POS_LOAD)
  
  def throw(self, pos_init, pos_target, duration):
    self.catapult.move(pos_init, duration=1.0, interval=0.01, wait=False)
    time.sleep(1.0)
    pos_init_actural = self.catapult.position
    
    self.catapult.move(pos_target, duration=duration, interval=0.01, wait=False, motion=self.catapult.MOTION_LINEAR)
    time.sleep(1.0)
    pos_target_actural = self.catapult.position
    
    self.reset()
    
    return pos_init_actural, pos_target_actural
  
  def _run_data_collection(self):
    prefix = 'catapult/data_collection'
    prefix_info = prefix + ':'
    
    dirpath_data = '../data/'
    filename_save = 'catapult_' + '{:%Y%m%d_%H%M%S_%f}'.format(datetime.datetime.now()) + '.yaml'
    filepath_save = dirpath_data + filename_save
    
    yaml_file = open(filepath_save, 'w')
    dataset = []
    
    def new_entry(motion, face_init, pos_init, pos_init_actural, pos_target, pos_target_actural, duration, loc_land, loc_stop, face_stop):
      entry = {
        'motion': motion,
        'action': {
          'face_init': face_init,
          'pos_init': pos_init,
          'pos_init_actural': pos_init_actural,
          'pos_target': pos_target,
          'pos_target_actural': pos_target_actural,
          'duration': duration
        },
        'result': {
          'loc_land': loc_land,
          'loc_stop': loc_stop,
          'face_stop': face_stop
        }
      }
      
      return entry
    
    def launch_test(face_init, pos_init, pos_target, duration):
      captured = False
      
      while not captured:
        print prefix_info, 'face_init = {}, pos_init = {}, pos_target = {}, duration = {}'.format(face_init, pos_init, pos_target, duration)
        input_ready = raw_input(prefix_info + ' ready (Y)?> ')
        pos_init_actural, pos_target_actural = self.throw(pos_init, pos_target, duration)
        
        input_captured = raw_input(prefix_info + ' captured (Y/n)?> ')
        if input_captured == '' or input_captured == 'y' or input_captured == 'Y':
          captured = True
        else:
          captured = False
        
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
      
      dataset.append(new_entry('linear', face_init, pos_init, pos_init_actural, pos_target, pos_target_actural, duration, loc_land, loc_stop, face_stop))
      yaml.dump([dataset[-1]], yaml_file, default_flow_style=False)
      print prefix_info, 'new entry added to dataset.'
      print ''
    
    feature_dict = {
      'face_init': ['1'],
      'pos_init': [0],#[50, 100, 200],
      'pos_target': [480],#[500],
      'duration': [0.05]#[0.2, 0.15, 0.10, 0.05]
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
    
    for feature_comb in feature_space:
      for i in range(1):
        launch_test(feature_comb['face_init'], feature_comb['pos_init'], feature_comb['pos_target'], feature_comb['duration']);
    
    print 'datafile:', filepath_save
  
  def _run_cma_throw_farther(self):
    pass
  
  def getOperations(self):
    operation_dict = {
      'data_collection': self._run_data_collection,
      'cma_throw_farther': self._run_cma_throw_farther
    }
    return operation_dict
  
  def run(self, operation):
    operation_dict = self.getOperations()
    
    if operation in operation_dict:
      op_func = operation_dict[operation]
      return op_func()
    
    raise ValueError('Invalid operation.', operation)



if __name__ == '__main__':
  catapult = TCatapult(reset=False)
  agent = TCatapultLPLinear(catapult)
  
  operation = 'data_collection'
  if len(sys.argv) >= 2:
    if len(sys.argv) == 2 and (sys.argv[1] in agent.getOperations()):
      operation = sys.argv[1]
    else:
      print 'usage: ./run_001_linear.py <operation>'
      quit()
  
  agent.run(operation)


