#!/usr/bin/python
'''
Dynamic programming and learning over graph-dynamical system.
'''

import random

import pdb

from base_util import *
from base_ml import *
from base_ml_dnn import TNNRegression
from base_opt2 import TDiscOptProb

from base_dpl4 import TGraphDynDomain, TDynNode, TCompSpaceDef, REWARD_KEY, PROB_KEY, TLocalLinear, TLocalQuad
from base_dpl4 import TGraphEpisodeDB
from base_dpl4 import TModelManager
from base_dpl4 import SSA, Vec
from base_dpl4 import TGraphDynPlanLearn as TGraphDynPlanLearnCore



class TPolicyManager(object):
  @staticmethod
  def DefaultOptions():
    Options= {}
    Options['type']= 'dnn'
    
    Options['dnn_hidden_units']= [200,200,200]  #Number of hidden units.
    Options['dnn_options']= {}  #Options for DNN. 'n_units' option is ignored. e.g. Options['dnn_options']['dropout_ratio']= 0.01

    Options['base_dir']= '/tmp/dpl/policies/'  #Base directory.  Last '/' matters.
    return Options
  
  def CheckOptions(self,options):
    res= True
    defaults= self.DefaultOptions()
    for key,val in options.iteritems():
      if not key in defaults:
        print('Invalid option: %s'%(key))
        res= False
    return res

  '''Generate an instance.
      space_defs: the same as TGraphDynDomain.SpaceDefs.
      models: the same as TGraphDynDomain.Models.  F may be None; in this case F is learned.
        A reference of models is stored in this class, and modified. '''
  def __init__(self, space_defs, models):
    self.Options= {}
    
    self.Load(data={'options':self.DefaultOptions()})

    self.SpaceDefs= space_defs
    
    self.Models = {}
    for key in models:
      In,Out,F = models[key]
      actions = []
      states = []
      for key_in in In:
        if self.SpaceDefs[key_in].Is('state'):
          states.append(key_in)
        elif self.SpaceDefs[key_in].Is('action'):
          actions.append(key_in)
      if len(actions) > 0 and len(states) > 0:
        self.Models[key] = [states, actions, None]
    
    self.Learning= set()  #Memorizing learning models (keys in self.Models).

  #Save into data (dict):  {'options':{options}, 'params':{parameters}}
  #base_dir: used to store data into external data file(s); None for a default value.
  def Save(self, base_dir=None):
    if base_dir is None:  base_dir= self.Options['base_dir']
    data= {}
    data['options']= self.Options

    for key in self.Learning:
      In,Out,F= self.Models[key]
      prefix,path= self.GetFilePrefixPath(base_dir,key)
      SaveYAML(F.Save(prefix), path)
    return data

  #Load from data (dict):  {'options':{options}, 'params':{parameters}}
  #base_dir: where external data file(s) are stored; if None, data is not load.
  def Load(self, data, base_dir=None):
    if data!=None and 'options' in data:
      assert(self.CheckOptions(data['options']))
      InsertDict(self.Options, data['options'])
    
    if base_dir is None:  return

    self.CreateModels()
    for key in self.Learning:
      In,Out,F= self.Models[key]
      prefix,path= self.GetFilePrefixPath(base_dir,key)
      if os.path.exists(path):
        F.Load(LoadYAML(path), prefix)

  #Initialize planner/learner.  Should be executed before execution.
  def Init(self):
    self.CreateModels()
    for key in self.Learning:
      In,Out,F= self.Models[key]
      F.Options['base_dir']= self.Options['base_dir']
      F.Init()

  #Create learning models of self.Models[key] for key in keys.
  #If keys is None, learning models are created for all self.Models whose F is None and len(Out)>0.
  def CreateModels(self, keys=None):
    if keys is None:  keys= [key for key,(In,Out,F) in self.Models.iteritems() if F is None and len(Out)>0]
    if len(keys)==0:  return
    
    if self.Options['type']=='dnn':
      for key in keys:
        In,Out,F= self.Models[key]
        dim_in= sum(DimsXSSA(self.SpaceDefs,In))
        dim_out= sum(DimsXSSA(self.SpaceDefs,Out))
        options= copy.deepcopy(self.Options['dnn_options'])
        options['base_dir']= self.Options['base_dir']
        options['n_units']= [dim_in] + list(self.Options['dnn_hidden_units']) + [dim_out]
        options['name']= key
        model= TNNRegression()
        model.Load(data={'options':options})
        self.Models[key][2]= model
        self.Learning.update({key})
  
  def hasKey(self, key):
    if key in self.Models:
      return True
    else:
      return False
  
  #Return file prefix and path to save model data.
  #  key: a name of model (a key of self.Models).
  def GetFilePrefixPath(self, base_dir, key):
    if self.Options['type']=='dnn':  file_suffix= 'nn.yaml'
    prefix= '{base}{model}_'.format(base=base_dir, model=key)
    path= prefix+file_suffix
    return prefix,path

  '''Update a model.
    key: a name of model (a key of self.Models).
    xs: input XSSA.
    ys: output XSSA.
    not_learn: option for TFunctionApprox (if True, just storing samples; i.e. training is not performed). '''
  def Update(self, key, xs, ys, not_learn=False):
    if key not in self.Learning:  return
    In,Out,F= self.Models[key]
    x_in,cov_in,dims_in= SerializeXSSA(self.SpaceDefs, xs, In)
    x_out,cov_out,dims_out= SerializeXSSA(self.SpaceDefs, ys, Out)
    F.Update(x_in, x_out, not_learn=not_learn)
  
  '''Predict a set of initial guesses points'''
  def Predict(self, key, xs):
    In,Out,F= self.Models[key]
    
    x_in,cov_in,dims_in= SerializeXSSA(self.SpaceDefs, xs, In)
    
    pred = F.Predict(x_in, with_var=True)
    a_h   = pred.Y.ravel()
    a_err = pred.Var
    
    ys = {}
    MapToXSSA(self.SpaceDefs, a_h, a_err, self.Models[key][1], ys)
    
    return [ys]
  
  #Dump data for plot into files.
  #file_prefix: prefix of the file names; {key} is replaced by the model name (a key of self.Models).
  #  {file_prefix}_est.dat, {file_prefix}_smp.dat are generated.
  def PlotModel(self,key,f_reduce,f_repair,file_prefix='/tmp/f{key}'):
    if key not in self.Models:
      raise Exception('PlotModel: Model not found: {key}'.format(key=key))
    In,Out,F= self.Models[key]
    if sum(DimsXSSA(self.SpaceDefs,In))==0 or sum(DimsXSSA(self.SpaceDefs,Out))==0:
      raise Exception('PlotModel: Model In/Out is zero: In={In}, Out={Out}'.format(In=In,Out=Out))
    DumpPlot(F, f_reduce=f_reduce, f_repair=f_repair, file_prefix=file_prefix.format(key=key))



class TGraphDynPlanLearn(object):
  def __init__(self, domain, database=None, model_manager=None, use_policy=False):
    super(TGraphDynPlanLearn, self).__init__()
    
    self._options = {
      'policy_training_samples': 10
    }
    
    self._domain = domain
    self._database = database
    self._model_manager = model_manager
    
    self._core = TGraphDynPlanLearnCore(self._domain, self._database, self._model_manager)
    
    self._use_policy = use_policy
    self._policy_manager = None
    if self._use_policy:
      self._policy_manager = TPolicyManager(self._domain.SpaceDefs, self._domain.Models)
  
  @property
  def DB(self):
    return self._core.DB
  
  @property
  def MM(self):
    return self._core.MM
  
  def PM(self):
    return self._policy_manager
  
  def Save(self):
    return self._core.Save()
  
  def Load(self, data):
    return self._core.Load(data)

  def Init(self):
    if self._use_policy:
      self._policy_manager.Init()
    
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
    """
    plan from n_start with initial xs
    """
    
    if self._use_policy:
      next_models = self._domain.Graph[n_start].Fd
      next_policies = [key for key in next_models if self._policy_manager.hasKey(key)]
      
      for key_policy in next_policies:
        state_keys  = self._policy_manager.Models[key_policy][0]
        action_keys = self._policy_manager.Models[key_policy][1]
        
        n_samples = self._options['policy_training_samples']
        
        randidx = self.DB.SearchIf(lambda eps: True)
        random.shuffle(randidx)
        n_samples = min(n_samples, len(randidx))
        randidx = [randidx[i] for i in range(n_samples)]
        
        for idx in randidx:
          eps = self.DB.GetEpisode(idx)
          pdb.set_trace()
    
    
    
    
    
    
    
    
    
    return self._core.Plan(n_start, xs)


