#!/usr/bin/python

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt

import pdb

sys.path.append(os.path.join(os.path.dirname(__file__), '../lib/base'))
from base_dpl4 import TGraphDynDomain, TDynNode, TCompSpaceDef, REWARD_KEY, PROB_KEY, TLocalLinear, TLocalQuad
from base_dpl4 import TGraphEpisodeDB
from base_dpl4 import TModelManager
from base_dpl4 import TGraphDynPlanLearn, SSA, Vec



class GraphDDPTest(object):
  def __init__(self):
    super(GraphDDPTest, self).__init__()  
  
  def launch_test(self):
    """
    x2 = (x1 - 2)^2
    x3 = (x2 - 3)^2
    R  = -x3
    """
    
    dirpath_log = '/tmp/test_020_GraphDDP'
    
    # define domain
    domain = TGraphDynDomain()
    SP = TCompSpaceDef
    domain.SpaceDefs = {
      'x1': SP('action', 1, min=[-5.0], max=[5.0]),
      'x2': SP('state', 1),
      'x3': SP('state', 1),
      REWARD_KEY: SP('state', 1)
    }
    
    # define model
    def FdF_f1(x, with_grad=False):
      y = [(x[0] - 2)**2]
      if with_grad:
        grad = [[2 * (x[0] - 2)]]
        return y, grad
      else:
        return y
    
    def FdF_f2(x, with_grad=False):
      y = [(x[0] - 3)**2]
      if with_grad:
        grad = [[2 * (x[0] - 3)]]
        return y, grad
      else:
        return y
    
    def FdF_R(x, with_grad=False):
      r = [- x[0]]
      if with_grad:
        grad = [[-1]]
        return r, grad
      else:
        return r
    
    domain.Models = {
      'f1': [['x1'], ['x2'], TLocalLinear(1, 1, FdF=FdF_f1)],
      'f2': [['x2'], ['x3'], TLocalLinear(1, 1, FdF=FdF_f2)],
      'R':  [['x3'], [REWARD_KEY], TLocalLinear(1, 1, FdF=FdF_R)],
      'P1': [[], [PROB_KEY], TLocalLinear(0, 1, lambda x:[1.0], lambda x:[0.0])]
    }
    
    # define graph
    domain.Graph = {
      'n1': TDynNode(None, 'P1', ('f1', 'n2')),
      'n2': TDynNode('n1', 'P1', ('f2', 'n3')),
      'n3': TDynNode('n2', 'P1', ('R',  'n3r')),
      'n3r': TDynNode('n2')
    }
    
    # define dynamic planning and learning agent
    dpl_options = {}
    dpl_options['ddp_sol'] = { 'f_reward_ucb': 0.0 }
    dpl_options['base_dir'] = dirpath_log
    dpl = TGraphDynPlanLearn(domain)
    dpl.Load({ 'options': dpl_options })
    dpl.Init()
    
    # run
    for episode in range(1):
      dpl.NewEpisode()
      
      xs0 = {}
      res = dpl.Plan('n1', xs0)
      xs = xs0
      x1 = xs['x1'].X
      x2 = FdF_f1(x1)
      x3 = FdF_f2(x2)
      R  = FdF_R(x3)
      print('xs={}'.format(xs))
      print('x1={}, x2={}, x3={}, R={}'.format(x1, x2, x3, R))
      
      dpl.EndEpisode()



if __name__ == '__main__':
  tester = GraphDDPTest()
  tester.launch_test()


