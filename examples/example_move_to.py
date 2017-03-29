#!/usr/bin/python
#Move Dynamixel to the initial position, and to a target

import sys
sys.path.append('../lib/finger')
from dynamixel_lib import *
import time

dxl= TDynamixel1()
dxl.Setup()

#Move to initial position
p_start= 2300
dxl.MoveTo(p_start)
time.sleep(1.0)  #wait 1 sec
print 'Current position=',dxl.Position()

#Move to a target position
p_trg= p_start-500
dxl.MoveTo(p_trg)
time.sleep(1.0)  #wait 0.1 sec
print 'Current position=',dxl.Position()
