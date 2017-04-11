#!/usr/local/bin/python3

"""
The simulation catapult dataset library. (for Python 3)
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

from libCatapultDataset import TCatapultDataset



class TCatapultDatasetSim(TCatapultDataset):
  """
  Simulation catapult dataset controller.
  
  @property dirpath The dataset directory path.
  @property size The size of the dataset.
  @property append_filename The appended dataset file name.
  @property append_filepath The appended dataset file path.
  @property append_size The size of the appended dataset.
  """
  
  def __init__(self, abs_dirpath=None, verbose=True, auto_init=True):
    """
    Initialize a simulation catapult dataset controller.
    
    @param abs_dirpath The absolute dataset directory path. `None` indicates 
                       using the default dataset directory path. (default: None)
    @param verbose A boolean indicating if prints verbose logs.
    @param auto_init A boolean indicating if automatically load data from the 
                     dataset directory and initialize the appended dataset YAML 
                     file. Note that the appended dataset YAML file will still 
                     be initialized automatically if new entry is appended.
    """
    super(TCatapultDatasetSim, self).__init__(abs_dirpath=abs_dirpath, verbose=verbose, auto_init=False)
    
    self._prefix = 'catapult_sim/dataset'
    self._prefix_info = self._prefix + ':'

    if auto_init:
      self.load_dataset()
      self.init_yaml()
  
  def new_entry_linear(self, *argv):
    raise Exception('This method is discarded. Use \"new_entry_linear_sim\" instead.')

  def new_entry_linear_sim(self, pos_init, pos_target, duration, loc_land):
    """
    Create a new entry for linear motion.
    
    @param pos_init The initial catapult position.
    @param pos_target The catapult target position.
    @param duration The time duration of the linear motion.
    @param loc_land The landing location.
    @return The entry created.
    """
    motion = 'linear'
    action = {
      'pos_init':   pos_init,
      'pos_target': pos_target,
      'duration':   duration
    }
    result = {
      'loc_land':   loc_land
    }
    entry = self.new_entry('linear', action=action, result=result)
    
    return entry


