#!/usr/bin/python

"""
Testing the simulation catapult controller library.
"""

__author__    = 'David Qiu'
__email__     = 'david@davidqiu.com'
__website__   = 'www.davidqiu.com'
__copyright__ = 'Copyright (C) 2017, David Qiu. All rights reserved.'


from libCatapultSim import TCatapultSim

import os
import sys
import time


if __name__ == '__main__':
  if len(sys.argv) != 2:
    print('Usage: ./test_catapultSim.py <dirpath_sim>')
    quit()

  dirpath_sim = sys.argv[1]

  prefix = 'test_catapultSim'
  prefix_info = prefix + ':'

  print('{} dirpath_sim = {}'.format(prefix_info, os.path.abspath(dirpath_sim)))

  print('{} Initialize a simulation catapult controller.'.format(prefix_info))
  catapult = TCatapultSim(dirpath_sim)
  
  print('{} Throw with linear motion. (please manage msg files manually or with simulator)'.format(prefix_info))
  loc_land = catapult.throw_linear(catapult.POS_MIN, catapult.POS_MID, 0.1)

  print('{} loc_land = {}'.format(prefix_info, loc_land))

  print('{} Simulation catapult controller testing finished.'.format(prefix_info))


