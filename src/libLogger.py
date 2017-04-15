#!/usr/bin/python

"""
The universial logger library.
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



class TLogger(object):
  """
  Universial logger.
  """
  
  def __init__(self, fp_log=None):
    """
    Initialize a universial logger.
    
    @param fp_log The file path to output the log. Note that `None` indicates 
                  not writing output log file.
    """
    self._filepath_log = fp_log
  
  def log(self, msg):
    """
    Add a message to the logs.
    """
    msg_str = str(msg)
    
    print(msg_str)
    
    if self._filepath_log is not None:
      with open(self._filepath_log, 'a') as log_file:
        log_file.write(msg_str + '\n')


