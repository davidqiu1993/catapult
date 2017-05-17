#!/usr/bin/python

import os
import sys
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt

import pdb

sys.path.append(os.path.join(os.path.dirname(__file__), '../lib/base'))
from base_dpl5 import TGraphDynDomain, TDynNode, TCompSpaceDef, REWARD_KEY, PROB_KEY, TLocalLinear, TLocalQuad
from base_dpl5 import TGraphEpisodeDB
from base_dpl5 import TModelManager
from base_dpl5 import TGraphDynPlanLearn, SSA, Vec



def sampleActionSSA(space_def):
  return SSA([(space_def.Min[d] + (space_def.Max[d] - space_def.Min[d]) * np.random.sample()) for d in range(space_def.D)])



def SSAVal(ssa):
  return Vec(ssa.X).ravel()



class GraphDDPTest(object):
  def __init__(self):
    super(GraphDDPTest, self).__init__()  
  
  def launch_test(self):
    """
    x2 = (x1 - 2)^2
    x3 = (x2 - 3)^2
    R  = -x3
    """
    
    dirpath_log = '/tmp/test_021_GraphDDP/'
    
    
    # define actual dynamics
    def actual_FdF_f1(x, with_grad=False):
      y = [(x[0] - 2)**2]
      if with_grad:
        grad = [[2 * (x[0] - 2)]]
        return y, grad
      else:
        return y
    
    def actual_FdF_f2(x, with_grad=False):
      y = [(x[0] - 3)**2]
      if with_grad:
        grad = [[2 * (x[0] - 3)]]
        return y, grad
      else:
        return y
    
    
    # define reward function
    def FdF_R(x, with_grad=False):
      r = [- x[0]]
      if with_grad:
        grad = [[-1]]
        return r, grad
      else:
        return r
    
    
    # define domain
    domain = TGraphDynDomain()
    
    # define spaces
    SP = TCompSpaceDef
    domain.SpaceDefs = {
      'x1': SP('action', 1, min=[-5.0], max=[5.0]),
      'x2': SP('state', 1),
      'x3': SP('state', 1),
      REWARD_KEY: SP('state', 1)
    }
    
    # define dynamics models
    domain.Models = {
      'f1': [['x1'], ['x2'], None],
      'f2': [['x2'], ['x3'], None],
      'R':  [['x3'], [REWARD_KEY], TLocalLinear(1, 1, FdF_R)],
      'P1': [[], [PROB_KEY], TLocalLinear(0, 1, lambda x:[1.0], lambda x:[0.0])]
    }
    
    # define graph
    domain.Graph = {
      'n1': TDynNode(None, 'P1', ('f1', 'n2')),
      'n2': TDynNode('n1', 'P1', ('f2', 'n3')),
      'n3': TDynNode('n2', 'P1', ('R',  'n3r')),
      'n3r': TDynNode('n2')
    }
    
    # define model manager
    mm_options = {
      'base_dir': dirpath_log + 'models/'
    }
    mm = TModelManager(domain.SpaceDefs, domain.Models)
    mm.Load({ 'options': mm_options })
    
    # define database
    db = TGraphEpisodeDB()
    
    # define dynamic planning and learning agent
    dpl_options = {
      'ddp_sol': { 'f_reward_ucb': 0.0 },
      'base_dir': dirpath_log
    }
    dpl = TGraphDynPlanLearn(domain, database=db, model_manager=mm, use_policy=False)
    dpl.Load({ 'options': dpl_options })
    
    # initialize
    dpl.MM.Init()
    dpl.Init()
    
    
    # inital samples
    n_init_samples = 3
    for i_sample in range(n_init_samples):
      xs = {}
      xs['x1'] = sampleActionSSA(domain.SpaceDefs['x1'])
      
      xs['x2'] = SSA(actual_FdF_f1(SSAVal(xs['x1'])))
      dpl.MM.Update('f1', xs, xs)
      
      xs['x3'] = SSA(actual_FdF_f2(SSAVal(xs['x2'])))
      dpl.MM.Update('f2', xs, xs)
    pdb.set_trace()
    
    # run
    n_episodes = 10
    for episode in range(n_episodes):
      dpl.NewEpisode()
      
      # plan
      xs0 = {}
      res = dpl.Plan('n1', xs0)
      if res.ResCode <= 0: quit()
      xs = res.XS
      
      # execute
      idb_prev = dpl.DB.AddToSeq(parent=None, name='n1', xs=xs)
      
      xs['x2'] = SSA(actual_FdF_f1(SSAVal(xs['x1'])))
      idb_prev = dpl.DB.AddToSeq(parent=idb_prev, name='n2', xs=xs)
      dpl.MM.Update('f1', xs, xs)
      
      xs['x3'] = SSA(actual_FdF_f2(SSAVal(xs['x2'])))
      idb_prev = dpl.DB.AddToSeq(parent=idb_prev, name='n3', xs=xs)
      dpl.MM.Update('f2', xs, xs)
      
      R = FdF_R(SSAVal(xs['x3']))
      
      print('actual: xs={}, R={}'.format(xs, R))
      
      dpl.EndEpisode()



if __name__ == '__main__':
  tester = GraphDDPTest()
  tester.launch_test()

