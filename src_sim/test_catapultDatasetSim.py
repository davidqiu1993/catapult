#!/usr/bin/python

"""
Testing the simulation catapult dataset library.
"""

__author__    = 'David Qiu'
__email__     = 'david@davidqiu.com'
__website__   = 'www.davidqiu.com'
__copyright__ = 'Copyright (C) 2017, David Qiu. All rights reserved.'


from libCatapultDatasetSim import TCatapultDatasetSim

import os
import time


if __name__ == '__main__':
  prefix = 'test_catapultDatasetSim'
  prefix_info = prefix + ':'

  print(prefix_info, 'Initialize a simulation catapult dataset controller.')
  abs_dirpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/' + 'catapult_sim_test'))
  dataset = TCatapultDatasetSim(abs_dirpath=abs_dirpath)
  
  print(prefix_info, 'size =', dataset.size)

  print(prefix_info, 'Append entry.')
  entry = dataset.new_entry_linear_sim(0.1, 0.3, 0.1, 3.123456)
  dataset.append(entry)

  print(prefix_info, 'size =', dataset.size)

  print(prefix_info, 'data >>> ')
  for entry in dataset:
    print(entry)

  print(prefix_info, 'file =', dataset.append_filepath)

  print(prefix_info, 'Simulation catapult dataset controller testing finished.')


