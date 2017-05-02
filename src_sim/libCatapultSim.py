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
  @constant DURATION_MIN The minimum duration.
  @constant MOTION_LINEAR The linear motion control.
  @constant MOTION_CUSTOM The customized motion control.
  """
  
  def __init__(self, dirpath_sim, catapult_model):
    """
    Initialize a catapult controller.
    
    @param dirpath_sim The directory path of the simulator for data exchange.
    @param catapult_model The model of the catapult.
    """
    super(TCatapultSim, self).__init__()
    
    self._dirpath_sim = os.path.abspath(dirpath_sim);
    self._filepath_msg_simulator = os.path.abspath(os.path.join(self._dirpath_sim, 'msg_simulator.msg'))
    self._filepath_msg_controller = os.path.abspath(os.path.join(self._dirpath_sim, 'msg_controller.msg'))
    
    if catapult_model == '001_02':
      self.POS_MIN  = 0.00 * math.pi
      self.POS_MAX  = 0.80 * math.pi
      self.POS_MID  = 0.50 * math.pi
      self.POS_INIT = 0.00 * math.pi
      self.DURATION_MIN = 0.10
    elif catapult_model == '002_02':
      self.POS_MIN  = 0.00 * math.pi
      self.POS_MAX  = 0.95 * math.pi
      self.POS_MID  = 0.50 * math.pi
      self.POS_INIT = 0.00 * math.pi
      self.DURATION_MIN = 0.50
    else:
      assert(False)
    self.model = catapult_model
    
    self.MOTION_LINEAR = 'linear'
    self.MOTION_CUSTOM = 'custom'
  
  def throw_linear(self, pos_init, pos_target, duration):
    """
    Throw from initial position to target position in linear motion. Note that 
    this is a sychronized method, which means it will wait for the result from 
    the catapult simulator.

    @param pos_init The initial position of the catapult.
    @param pos_target The target position of the catapult.
    @param duration The time interval in between.
    @return The landing location of the thrown object.
    """
    WAIT_TIME = 0.10

    assert(self.POS_MIN <= pos_init and pos_init <= self.POS_MAX)
    assert(self.POS_MIN <= pos_target and pos_target <= self.POS_MAX)
    assert(pos_init < pos_target)
    assert(self.DURATION_MIN <= duration)

    with open(self._filepath_msg_controller, 'w') as msg_controller:
      msg_controller.write(str(pos_init)   + ' ' + 
                           str(pos_target) + ' ' + 
                           str(duration)   + '\n')

    hasRead = False
    loc_land = None
    while not hasRead:
      line_last = None

      if os.path.isfile(self._filepath_msg_simulator):
        with open(self._filepath_msg_simulator, 'r') as msg_simulator:
          line_incoming = msg_simulator.readline()
          while line_incoming != '':
            line_last = line_incoming
            line_incoming = msg_simulator.readline()

      if line_last is not None:
        loc_land = float(line_last)
        hasRead = True
      else:
        time.sleep(WAIT_TIME)

    assert(hasRead)

    with open(self._filepath_msg_simulator, 'w') as msg_simulator_flush:
      msg_simulator_flush.write('')

    return loc_land


