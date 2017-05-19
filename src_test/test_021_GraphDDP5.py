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
from base_dpl5 import TGraphDynPlanLearn, SSA, Vec, CopyXSSA



def sampleActionSSA(space_def):
  return SSA([(space_def.Min[d] + (space_def.Max[d] - space_def.Min[d]) * np.random.sample()) for d in range(space_def.D)])



def SSAVal(ssa):
  return Vec(ssa.X).ravel()



class GraphDDPTest(object):
  def __init__(self):
    super(GraphDDPTest, self).__init__()  
  
  def launch_test(self):
    """
    x2 = (x1 * a1 - 2)^2
    x3 = (x2 - 3)^2
    R  = -x3
    """
    
    dirpath_log = '/tmp/test_021_GraphDDP/'
    
    
    # define actual dynamics
    def actual_FdF_f1(x, with_grad=False):
      y = [(x[0] * x[1] - 2)**2]
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
      'x1': SP('state', 1), # default: 1.0
      'a1': SP('action', 1, min=[-5.0], max=[5.0]),
      'x2': SP('state', 1),
      'x3': SP('state', 1),
      REWARD_KEY: SP('state', 1)
    }
    
    # define dynamics models
    domain.Models = {
      'f1': [['x1', 'a1'], ['x2'], None],
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
      'base_dir': dirpath_log + 'models/',
      'dnn_options': {
        'verbose': False
      }
    }
    mm = TModelManager(domain.SpaceDefs, domain.Models)
    mm.Load({ 'options': mm_options })
    
    # define database
    db = TGraphEpisodeDB()
    
    # define dynamic planning and learning agent
    dpl_options = {
      'use_policy': True,
      'policy_verbose': True,
      'policy_manager_options': {
        'dnn_options': { 'verbose': False }
      },
      'ddp_sol': {
        'f_reward_ucb': 0.0,
        'verbose': False
      },
      'base_dir': dirpath_log
    }
    dpl = TGraphDynPlanLearn(domain, database=db, model_manager=mm)
    dpl.Load({ 'options': dpl_options })
    
    # initialize
    dpl.MM.Init()
    dpl.Init()
    
    
    # inital samples
    n_init_samples = 3
    for i_sample in range(n_init_samples):
      print('update models with initial samples (sample: {}/{})'.format(i_sample+1, n_init_samples))
      
      xs = {}
      xs['x1'] = SSA([0.8])
      xs['a1'] = sampleActionSSA(domain.SpaceDefs['a1'])
      
      xs['x2'] = SSA(actual_FdF_f1([ SSAVal(xs['x1'])[0], SSAVal(xs['a1'])[0] ]))
      dpl.MM.Update('f1', xs, xs)
      
      xs['x3'] = SSA(actual_FdF_f2(SSAVal(xs['x2'])))
      dpl.MM.Update('f2', xs, xs)
    
    print('')
    
    
    # run
    n_episodes = 20
    for episode in range(n_episodes):
      print('episode: {}/{}'.format(episode+1, n_episodes))
      dpl.NewEpisode()
      
      # plan
      xs0 = { 'x1': SSA([1.0]) }
      print('GraphDDP planning begins (xs0: {})'.format(xs0))
      res = dpl.Plan('n1', xs0)
      if res.ResCode <= 0: quit()
      xs = res.XS
      print('GraphDDP planning finished (xs: {})'.format(xs))
      
      # execute
      print('execution and models update begin (xs: {})'.format(xs))
      
      idb_prev = dpl.DB.AddToSeq(parent=None, name='n1', xs=xs)
      print('execution phase (name: {}, xs: {})'.format('n1', xs))
      
      xs['x2'] = SSA(actual_FdF_f1([ SSAVal(xs['x1'])[0], SSAVal(xs['a1'])[0] ]))
      idb_prev = dpl.DB.AddToSeq(parent=idb_prev, name='n2', xs=xs)
      print('execution phase (name: {}, xs: {})'.format('n2', xs))
      dpl.MM.Update('f1', xs, xs)
      
      xs['x3'] = SSA(actual_FdF_f2(SSAVal(xs['x2'])))
      idb_prev = dpl.DB.AddToSeq(parent=idb_prev, name='n3', xs=xs)
      print('execution phase (name: {}, xs: {})'.format('n3', xs))
      dpl.MM.Update('f2', xs, xs)
      
      R = FdF_R(SSAVal(xs['x3']))
      print('execution phase (name: {}, R: {})'.format('.r', R))
      
      print('execution and models update finished (xs: {}, R: {})'.format(xs, R))
      
      dpl.EndEpisode()
      
      print('')



if __name__ == '__main__':
  tester = GraphDDPTest()
  tester.launch_test()


