#!/usr/bin/python

"""
CMA-ES boundaries constraints handling testing.
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
import numpy as np
import matplotlib.pyplot as plt

import pdb

sys.path.append(os.path.join(os.path.dirname(__file__), '../lib/cma'))
import cma



if __name__ == '__main__':
  data_dir_path = '../data_test/'
  yaml_file_name = 'test_001_' + '{:%Y%m%d_%H%M%S_%f}'.format(datetime.datetime.now()) + '.yaml'
  yaml_file_path = os.path.abspath(os.path.join(data_dir_path, yaml_file_name))
  
  args = { 'count_iter': 0, 'dataset': [], 'init_guess': [1, 0, 0], 'popsize': 10, 'sigma': 3.0 }
  def f(x, args):
    sample_x = float(x[0])
    
    # loss
    if   sample_x < 0.0: loss = - (0.0)
    elif sample_x > 5.0: loss = - (5.0)
    else:                loss = - sample_x
    
    # panelty
    if   sample_x < 0.0: loss += np.abs(sample_x - 0.0) * 10.
    elif sample_x > 5.0: loss += np.abs(sample_x - 5.0) * 10.
    
    entry = {
      't': args['count_iter'],
      'x': sample_x,
      'loss': loss
    }
    args['dataset'].append(entry)
    
    print args['count_iter'], sample_x, loss
    
    args['count_iter'] += 1
    
    return loss
  
  res = cma.fmin(f, args['init_guess'], args['sigma'], [args], tolx=0.01, popsize=args['popsize'], verb_disp=False, verb_log=0)
  print res
  
  args['solution'] = res[0].tolist()
  
  """
  with open(yaml_file_path, 'w') as yaml_file:
    yaml.dump(args, yaml_file, default_flow_style=False)
  """
  
  plt.figure(1)
  
  plt.subplot(211)
  plt.plot([args['dataset'][i]['t'] for i in range(len(args['dataset']))], 
            [args['dataset'][i]['x'] for i in range(len(args['dataset']))])
  
  plt.subplot(212)
  plt.plot([args['dataset'][i]['t'] for i in range(len(args['dataset']))], 
            [args['dataset'][i]['loss'] for i in range(len(args['dataset']))])
  
  plt.show()


