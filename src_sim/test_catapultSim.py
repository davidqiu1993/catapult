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
    print 'Usage: ./test_catapultSim.py <dirpath_sim>'
    quit()

  dirpath_sim = sys.argv[1]

  prefix = 'test_catapultSim'
  prefix_info = prefix + ':'

  print prefix_info, 'dirpath_sim =', os.path.abspath(dirpath_sim)

  print prefix_info, 'Initialize a simulation catapult controller.'
  catapult = TCatapultSim(dirpath_sim)
  
  print prefix_info, 'Throw with linear motion. (please manage msg files manually or with simulator)'
  loc_land = catapult.throw_linear(catapult.POS_MIN, catapult.POS_MID, 0.1)

  print prefix_info, 'loc_land =', loc_land

  print prefix_info, 'Simulation catapult controller testing finished.'


