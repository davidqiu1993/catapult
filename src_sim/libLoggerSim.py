#!/usr/bin/python

"""
The universial logger library for simulation environment.
"""

__author__    = 'David Qiu'
__email__     = 'david@davidqiu.com'
__website__   = 'www.davidqiu.com'
__copyright__ = 'Copyright (C) 2017, David Qiu. All rights reserved.'


import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from libLogger import TLogger



class TLoggerSim(TLogger):
  """
  Universial logger for simulation environment.
  """
  
  def __init__(self, fp_log=None):
    """
    Initialize a universial logger for simulation environment.
    
    @param fp_log The file path to output the log. Note that `None` indicates 
                  not writing output log file.
    """
    super(TLoggerSim, self).__init__(fp_log=fp_log)


