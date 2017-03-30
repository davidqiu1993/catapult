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
    
    
    launch_test('1', 200, 480, 1.0)
    launch_test('1', 300, 480, 0.5)
    
    
    print 'datafile:', filepath_save
    
    
  
  def run(self, operation):
    if operation == 'data_collection':
      return self._run_data_collection()
    
    raise ValueError('Invalid operation.', operation)



if __name__ == '__main__':
  catapult = TCatapult(reset=False)
  agent = TCatapultLPLinear(catapult)
  agent.run('data_collection')


