#!/usr/bin/python
'''
Dynamic programming and learning over graph-dynamical system.
'''
from base_util import *
from base_ml import *
from base_ml_dnn import TNNRegression
from base_opt2 import TDiscOptProb

from base_dpl4 import TGraphDynDomain, TDynNode, TCompSpaceDef, REWARD_KEY, PROB_KEY, TLocalLinear, TLocalQuad
from base_dpl4 import TGraphEpisodeDB
from base_dpl4 import TModelManager
from base_dpl4 import SSA, Vec
from base_dpl4 import TGraphDynPlanLearn as TGraphDynPlanLearnCore



class TGraphDynPlanLearn(object):
  def __init__(self, domain, database=None, model_manager=None):
    super(TGraphDynPlanLearn, self).__init__()
    
    self._core = TGraphDynPlanLearnCore(domain, database, model_manager)
  
  @property
  def DB(self):
    return self._core.DB
  
  @property
  def MM(self):
    return self._core.MM
  
  def Save(self):
    return self._core.Save()
  
  def Load(self, data):
    return self._core.Load(data)

  def Init(self):
    return self._core.Init()
  
  def GetDDPSol(self, logfp=None):
    return self._core.GetDDPSol(logfp=logfp)
  
  def GetPTree(self, n_start, xs_start=None, max_visits=None):
    return self._core.GetPTree(n_start, xs_start=xs_start, max_visits=max_visits)
  
  def NewEpisode(self):
    return self._core.NewEpisode()
  
  def EndEpisode(self):
    return self._core.EndEpisode()
  
  def Plan(self, n_start, xs):
    return self._core.Plan(n_start, xs)


