#!/usr/bin/python

"""
The simulation catapult controller library for advanced motion controlling.
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
import math



class TCatapultSim(object):
  """
  Catapult controller.
  
  @constant POS_MIN The minimum position.
  @constant POS_MAX The maximum position.
  @constant POS_MID The middle position.
  @constant POS_INIT The initial position.
  @constant MOTION_LINEAR The linear motion control.
  @constant MOTION_CUSTOM The customized motion control.
  """
  
  def __init__(self, dirpath_sim):
    """
    Initialize a catapult controller.
    
    @param dirpath_sim The directory path of the simulator for data exchange.
    """
    super(TCatapultSim, self).__init__()
    
    self._dirpath_sim = dirpath_sim;
    
    self.POS_MIN  = 0.0 * math.pi
    self.POS_MAX  = 1.0 * math.pi
    self.POS_MID  = 0.5 * math.pi
    self.POS_INIT = 0.0 * math.pi
    
    self.MOTION_LINEAR = 'linear'
    self.MOTION_CUSTOM = 'custom'
  
  #TODO
  
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


