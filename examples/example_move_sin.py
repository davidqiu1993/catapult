#!/usr/bin/python
#Control Dynamixel to follow a sin curve

import sys
sys.path.append('../lib/finger')
from dynamixel_lib import *
import time
import math
import numpy as np

dxl= TDynamixel1()
dxl.Setup()

#Move to initial position
p_start= 2300
dxl.MoveTo(p_start)
time.sleep(1.0)  #wait 1 sec
print 'Current position=',dxl.Position()

#Move to a target position
for t in np.mgrid[0:2*math.pi:0.1]:
  p_trg= p_start - 600*(0.5-0.5*math.cos(t))
  print p_trg
  dxl.MoveTo(int(p_trg), wait=False)
  time.sleep(0.01)
  #print 'Current position=',dxl.Position()
