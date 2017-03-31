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
    
    ctrl_pos_original = float(self._dxl.Position())
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


