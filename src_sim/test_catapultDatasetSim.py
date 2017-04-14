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

  print('Initialize a simulation catapult dataset controller.'.format(prefix_info))
  abs_dirpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/' + 'catapult_sim_test'))
  dataset = TCatapultDatasetSim(abs_dirpath=abs_dirpath)
  
  print('{} size = {}'.format(prefix_info, dataset.size))

  print('{} Append entry.'.format(prefix_info))
  entry = dataset.new_entry_linear_sim(0.1, 0.3, 0.1, 3.123456)
  dataset.append(entry)

  print('{} size = {}'.format(prefix_info, dataset.size))

  print('{} data >>> '.format(prefix_info))
  for entry in dataset:
    print(entry)

  print('{} file = {}'.format(prefix_info, dataset.append_filepath))

  print('{} Simulation catapult dataset controller testing finished.'.format(prefix_info))


