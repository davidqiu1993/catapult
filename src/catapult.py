#!/usr/bin/python

"""
The catapult controller library for advanced motion controlling.
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

sys.path.append(os.path.join(os.path.dirname(__file__), '../lib/finger'))
from dynamixel_lib import *



class TCatapult(object):
  """
  Catapult controller.
  
  @constant POS_MIN The minimum position.
  @constant POS_MAX The maximum position.
  @constant POS_MID The middle position.
  @constant POS_INIT The initial position.
  @constant POS_LOAD The object loading position.
  @constant MOTION_LINEAR The linear motion control.
  @constant MOTION_CUSTOM The customized motion control.
  """
  
  def __init__(self, reset=True):
    """
    Initialize a catapult controller.
    
    @param reset A boolean indicating if resets the position of the catapult to 
                 its initial position. (default: True)
    """
    super(TCatapult, self).__init__()
    
    self._POS_BASE = 2300
    self.POS_MIN = 0
    self.POS_MAX = 840
    self.POS_MID = 420
    self.POS_INIT = self.POS_MIN
    self.POS_LOAD = 180
    
    self.MOTION_LINEAR = 'linear'
    self.MOTION_CUSTOM = 'custom'
    
    self._dxl = TDynamixel1()
    self._dxl.Setup()
    
    if reset:
      self._move(self.POS_INIT, duration=1.0, interval=0.01, wait=False, motion=self.MOTION_LINEAR, motion_func=None)
  
  def getPosition(self):
    """
    Get the current position of the catapult.
    
    @return An integer indicating the current position of the catapult. (range: 
            POS_MIN ~ POS_MAX)
    """
    ctrl_pos = None
    retry_count = 0
    while ctrl_pos is None:
      ctrl_pos = self._dxl.Position()
      if ctrl_pos is None:
        print '[Catapult] Warning: Failed to get position. Retry after 0.001 sec. ({})'.format(retry_count)
        time.sleep(0.001)
        retry_count += 1
    
    pos = min(self.POS_MAX, max(self._POS_BASE - ctrl_pos, self.POS_MIN))
    
    return pos
  
  @property
  def position(self):
    """
    (Property)
    The current position of the catapult.
    """
    return self.getPosition()
  
  def _move_linear(self, position, duration, interval, wait):
    """
    (Internal Method)
    Move the catapult to a desired position in linear motion.
    """
    ctrl_steps = int(duration / interval)
    
    ctrl_pos_original = None
    retry_count_ctrl_pos_original = 0
    while ctrl_pos_original is None:
      ctrl_pos_original = float(self._dxl.Position())
      if ctrl_pos_original is None:
        print '[Catapult] Warning: Failed to get position. Retry after 0.001 sec. ({})'.format(retry_count_ctrl_pos_original)
        time.sleep(0.001)
        retry_count_ctrl_pos_original += 1
    ctrl_pos_target = float(self._POS_BASE) - float(position)
    ctrl_pos_interval = (float(ctrl_pos_target) - float(ctrl_pos_original)) / float(ctrl_steps)
    
    for i in range(ctrl_steps):
      ctrl_pos = ctrl_pos_original + (i + 1) * ctrl_pos_interval
      self._dxl.MoveTo(int(ctrl_pos), wait=False)
      time.sleep(interval)
    
    self._dxl.MoveTo(int(ctrl_pos_target), wait=False)
  
  def _move(self, position, duration=1.0, interval=0.01, wait=False, motion=None, motion_func=None):
    """
    (Internal Method)
    Move the catapult to a desired position.
    
    @param position The desired position.
    @param duration The time duration to move the catapult. (default: 1.0)
    @param interval The time interval for each control step. (default: 0.01)
    @param wait A boolean indicating if waits at each control step for the 
                catapult until it reaches the intermediate step position. 
                Note that this feature will introduce unsmooth control motion. 
                (default: False)
    @param motion The motion control. Note that all available motion control 
                  are defined as constant properties in this class, where 
                  `None` indicates using the `linear` motion control and 
                  `custom` indicates using a customized motion control, which 
                  is defined by the customized motion control function. 
                  (default: None)
    @param motion_func The customized motion control function, which activates 
                       only if the motion control is `custom`.
    @return The actual position of the catapult after the motion control 
            finishes.
    """
    assert(self.POS_MIN <= position and position <= self.POS_MAX)
    
    if motion is None:
      motion = self.MOTION_LINEAR
    
    if motion == self.MOTION_LINEAR:
      self._move_linear(position, duration, interval, wait)
    elif motion == self.MOTION_CUSTOM:
      pass
    else:
      raise ValueError('Invalid motion control.', motion)
    
    return self.getPosition()
  
  def move(self, position, duration=1.0, interval=0.01, wait=False, motion=None, motion_func=None):
    """
    Move the catapult to a desired position.
    
    @param position The desired position.
    @param duration The time duration to move the catapult. (default: 1.0)
    @param interval The time interval for each control step. (default: 0.01)
    @param wait A boolean indicating if waits at each control step for the 
                catapult until it reaches the intermediate step position. 
                Note that this feature will introduce unsmooth control motion. 
                (default: False)
    @param motion The motion control. Note that all available motion control 
                  are defined as constant properties in this class, where 
                  `None` indicates using the `linear` motion control and 
                  `custom` indicates using a customized motion control, which 
                  is defined by the customized motion control function. 
                  (default: None)
    @param motion_func The customized motion control function, which activates 
                       only if the motion control is `custom`.
    @return The actual position of the catapult after the motion control 
            finishes.
    """
    res = self._move(position=position, duration=duration, interval=interval, wait=wait, 
                     motion=motion, motion_func=motion_func)
    
    return res



class TCatapultDataset(object):
  """
  Catapult controller.
  
  @property dirpath The dataset directory path.
  @property size The size of the dataset.
  @property append_filename The appended dataset file name.
  @property append_filepath The appended dataset file path.
  @property append_size The size of the appended dataset.
  """
  
  def __init__(self, abs_dirpath=None, verbose=True, auto_init=True):
    """
    Initialize a catapult dataset controller.
    
    @param abs_dirpath The absolute dataset directory path. `None` indicates 
                       using the default dataset directory path. (default: None)
    @param verbose A boolean indicating if prints verbose logs.
    @param auto_init A boolean indicating if automatically load data from the 
                     dataset directory and initialize the appended dataset YAML 
                     file. Note that the appended dataset YAML file will still 
                     be initialized automatically if new entry is appended.
    """
    super(TCatapultDataset, self).__init__()
    
    self._dataset_dirpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/'))
    if abs_dirpath is not None:
      self._dataset_dirpath = abs_dirpath
    
    self._verbose = verbose
    self._prefix = 'catapult/dataset'
    self._prefix_info = self._prefix + ':'
    
    self._dataset = []
    self._dataset_append = []
    
    self._dataset_append_yaml_filename = None
    self._dataset_append_yaml_filepath = None
    self._dataset_append_yaml_file = None
    
    if auto_init:
      self.load_dataset()
      self.init_yaml()
  
  def __getitem__(self, key):
    return self.fetch(key)

  def __iter__(self):
    return self._dataset.__iter__()
  
  def __contains__(self, key):
    return self._dataset.__contains__(key)
  
  @property
  def dirpath(self):
    return self._dataset_dirpath
  
  @property
  def size(self):
    return len(self._dataset)
  
  @property
  def append_filename(self):
    return self._dataset_append_yaml_filename
  
  @property
  def append_filepath(self):
    return self._dataset_append_yaml_filepath
  
  @property
  def append_size(self):
    return len(self._dataset_append)  
  
  def load_dataset(self):
    """
    Load the data from the dataset directory. Note the the original data will 
    be flushed before loading data from the dataset directory.
    """
    self._dataset = []
    
    dataset_filepaths = []
    for dirname, dirnames, filenames in os.walk(self._dataset_dirpath):
      dataset_filepaths = [(os.path.abspath(os.path.join(dirname, filename))) for filename in filenames]
      break
    
    for dataset_filepath in dataset_filepaths:
      if self._verbose:
        print self._prefix_info, 'load data from file (path={})'.format(dataset_filepath)
      with open(dataset_filepath, 'r') as cur_yaml_file:
        cur_dataset = yaml.load(cur_yaml_file)
        if cur_dataset is not None:
          for entry in cur_dataset:
            self._dataset.append(entry)
  
  def init_yaml(self):
    """
    Initialize the appended dataset YAML file. The file will be named according 
    to the current system date and time.
    """
    self._dataset_append_yaml_filename = 'catapult_' + '{:%Y%m%d_%H%M%S_%f}'.format(datetime.datetime.now()) + '.yaml'
    self._dataset_append_yaml_filepath = os.path.abspath(os.path.join(self._dataset_dirpath, self._dataset_append_yaml_filename))
    self._dataset_append_yaml_file = open(self._dataset_append_yaml_filepath, 'w')
  
  def new_entry(self, motion, action={}, result={}):
    """
    Create a new entry.
    
    @param motion The motion control. Note that custom motion is not supported.
    @param action The action record. (default: {})
    @param result The result record. (default: {})
    @return The entry created.
    """
    assert(motion != 'custom')
    
    entry = {
      'motion': motion,
      'action': action,
      'result': result
    }
    
    return entry
  
  def new_entry_linear(self, face_init, pos_init, pos_init_actual, pos_target, pos_target_actual, duration, loc_land, loc_stop, face_stop):
    """
    Create a new entry for linear motion.
    
    @param face_init The initial face.
    @param pos_init The initial catapult position.
    @param pos_init_actual The actual initial catapult position.
    @param pos_target The catapult target position.
    @param pos_target_actual The actual catapult target position.
    @param duration The time duration of the linear motion.
    @param loc_land The landing location.
    @param loc_stop The stopping location.
    @param face_stop The stopping face.
    @return The entry created.
    """
    motion = 'linear'
    action = {
      'face_init':          face_init,
      'pos_init':           pos_init,
      'pos_init_actual':   pos_init_actual,
      'pos_target':         pos_target,
      'pos_target_actual': pos_target_actual,
      'duration':           duration
    }
    result = {
      'loc_land':           loc_land,
      'loc_stop':           loc_stop,
      'face_stop':          face_stop
    }
    entry = self.new_entry('linear', action=action, result=result)
    
    return entry
  
  def append(self, entry):
    """
    Append a new entry to the dataset.
    
    @param entry The entry to append to the dataset.
    """
    assert(type(entry) is dict)
    assert(entry['motion'] is not None)
    assert(type(entry['action']) is dict)
    assert(type(entry['result']) is dict)
    
    if self._dataset_append_yaml_file is None:
      self.init_yaml()
    
    self._dataset.append(entry)
    self._dataset_append.append(entry)
    
    yaml.dump([self._dataset_append[-1]], self._dataset_append_yaml_file, default_flow_style=False)
    if self._verbose:
      print self._prefix_info, 'new entry added to dataset.'
  
  def fetch(self, entry_index):
    """
    Fetch an entry by its index.
    
    @param entry_index The index of the entry.
    @return The entry fetched from the dataset. Note that error will be thrown 
            if the entry is not found.
    """
    return self._dataset[entry_index]


