#!/usr/bin/python

"""
Testing the catapult controller library for advanced motion controlling.
"""

__author__    = 'David Qiu'
__email__     = 'david@davidqiu.com'
__website__   = 'www.davidqiu.com'
__copyright__ = 'Copyright (C) 2017, David Qiu. All rights reserved.'


from catapult import *

import time


if __name__ == '__main__':
  print 'Initialize a catapult controller.'
  catapult = TCatapult()
  
  time.sleep(0.5)
  
  assert(catapult.position == catapult.getPosition())
  print 'Actual initial position:', catapult.position
  
  time.sleep(0.5)
  
  print 'Move to middle position.'
  pos = catapult.move(catapult.POS_MID)
  print 'Current actual position:', catapult.position
  
  time.sleep(0.5)
  
  print 'Move to minimum position.'
  catapult.move(catapult.POS_MIN)
  print 'Current actual position:', catapult.position
  
  time.sleep(0.5)
  
  print 'Move to maximum position.'
  catapult.move(catapult.POS_MAX)
  print 'Current actual position:', catapult.position
  
  time.sleep(0.5)
  
  print 'Move to initial position.'
  catapult.move(catapult.POS_INIT)
  print 'Current actual position:', catapult.position
  
  time.sleep(0.5)
  
  print 'Catapult controller testing finished.'


