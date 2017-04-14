#!/usr/bin/python

"""
Testing the catapult controller library for advanced motion controlling.
"""

__author__    = 'David Qiu'
__email__     = 'david@davidqiu.com'
__website__   = 'www.davidqiu.com'
__copyright__ = 'Copyright (C) 2017, David Qiu. All rights reserved.'


from libCatapult import TCatapult

import time
import sys


if __name__ == '__main__':
  pos_base = None
  if len(sys.argv) == 2:
    pos_base = int(sys.argv[1])
  
  print('Initialize a catapult controller.')
  if pos_base is None:
    catapult = TCatapult()
  else:
    catapult = TCatapult(_pos_base=pos_base)
  
  time.sleep(0.5)
  
  assert(catapult.position == catapult.getPosition())
  print('Actual initial position: {}'.format(catapult.position))
  
  time.sleep(0.5)
  
  print('Move to middle position.')
  pos = catapult.move(catapult.POS_MID)
  print('Current actual position: {}'.format(catapult.position))
  
  time.sleep(0.5)
  
  print('Move to minimum position.')
  catapult.move(catapult.POS_MIN)
  print('Current actual position: {}'.format(catapult.position))
  
  time.sleep(0.5)
  
  print('Move to maximum position.')
  catapult.move(catapult.POS_MAX)
  print('Current actual position: {}'.format(catapult.position))
  
  time.sleep(0.5)
  
  print('Move to initial position.')
  catapult.move(catapult.POS_INIT)
  print('Current actual position: {}'.format(catapult.position))
  
  time.sleep(0.5)
  
  print('Catapult controller testing finished.')


