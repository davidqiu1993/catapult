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
from base_dpl4 import SSA, Vec, DimsXSSA, CopyXSSA, SerializeXSSA, MapToXSSA
from base_dpl4 import TGraphDynUtil
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
        options['name']= 'pi_' + key
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
  
  def Flush(self, key):
    In,Out,F = self.Models[key]
    F.DataX  = np.array([], np.float32)
    F.DataY  = np.array([], np.float32)
  
  '''
  Predict a set of initial guesses points. Note that it returns a list of 
  possible initial guesses the content of which is determined by the policy 
  policy approximators used.
  
  @param policy_approximators Policy approximators used for prediction.
          - 'all': all policy approximators
          - 'dnn': neural network policy
  '''
  def Predict(self, key, xs, policy_approximators='dnn'):
    if policy_approximators == 'all':
      ys_list = [] + self.Predict(key, xs, policy_approximators='dnn')
      return ys_list
    
    elif policy_approximators == 'dnn':
      In,Out,F= self.Models[key]
      
      x_in,cov_in,dims_in= SerializeXSSA(self.SpaceDefs, xs, In)
      
      pred = F.Predict(x_in, with_var=True)
      a_h   = pred.Y.ravel()
      a_err = pred.Var
      
      ys = {}
      MapToXSSA(self.SpaceDefs, a_h, a_err, self.Models[key][1], ys)
      
      return [ys]
    
    else:
      assert(False)
  
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



class TGraphDynPlanLearn(TGraphDynUtil):
  @staticmethod
  def DefaultOptions():
    options = TGraphDynPlanLearnCore.DefaultOptions()
    options['use_policy'] = False
    options['policy_verbose'] = True
    options['policy_manager_options'] = {}
    options['policy_training_samples'] = 10
    return options
  
  def __init__(self, domain, database=None, model_manager=None):
    self._options = {
      'use_policy': False,
      'policy_verbose': True,
      'policy_manager_options': {},
      'policy_training_samples': 10
    }
    
    self._domain = domain
    self._database = database
    self._model_manager = model_manager
    self._policy_manager = None
    
    self._core = TGraphDynPlanLearnCore(self._domain, self._database, self._model_manager)
  
  @property
  def d(self):
    return self._core.d
  
  @property
  def DB(self):
    return self._core.DB
  
  @property
  def MM(self):
    return self._core.MM
  
  @property
  def PM(self):
    return self._policy_manager
  
  @property
  def Options(self):
    return self._core.Options;
  
  def Save(self):
    return self._core.Save()
  
  def Load(self, data):
    core_data = {}
    
    if 'options' in data:
      options = data['options']
      core_data['options'] = {}
      for key in options:
        if key in self._options:
          self._options[key] = options[key]
        else:
          core_data['options'][key] = options[key]
      
      for key in data:
        if key != 'options':
          core_data[key] = data[key]
    
    else:
      core_data = data
    
    return self._core.Load(core_data)
  
  def Init(self):
    if self._options['use_policy']:
      self._policy_manager = TPolicyManager(self._domain.SpaceDefs, self._domain.Models)
      self._policy_manager.Load({ 'options': self._options['policy_manager_options'] })
      self._policy_manager.Init()
    
    return self._core.Init()
  
  def RandActions(self, actions):
    return self._core.RandActions(actions)
  
  def RandSelections(self, selections):
    return self._core.RandSelections(selections)
  
  def ActionNoise(self, actions, var):
    return self._core.ActionNoise(actions, var)
  
  def Value(self, ptree, with_grad=False):
    return self._core.Value(ptree, with_grad)
  
  def Forward(self, key, xs, with_grad=False):
    return self._core.Forward(key, xs, with_grad)
  
  def ForwardP(self, key, xs, with_grad=False):
    return self._core.ForwardP(key, xs, with_grad)
  
  def ForwardTree(self, ptree, with_grad=False):
    return self._core.ForwardTree(ptree, with_grad)
  
  def BackwardTree(self, ptree):
    return self._core.BackwardTree(ptree)
  
  def GetPTree(self, n_start, xs_start=None, max_visits=3):
    return self._core.GetPTree(n_start, xs_start, max_visits)
  
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
    """
    TODO: Use policy prediction at each stage in both policy update 
    optimization and real action optimization (now only at n_start)
    """
    
    xs = CopyXSSA(xs)
    
    next_models = self._domain.Graph[n_start].Fd
    next_policies = None
    if self._policy_manager is not None:
      next_policies = [key for key in next_models if self._policy_manager.hasKey(key)]
    
    # update policy networks
    if self._policy_manager is not None:
      """
      train the following policy network(s) with samples optimized from MB 
      method at some visited points started from initial points suggested by 
      the policy networks.
      """
      for key_policy in next_policies:
        state_keys  = self._policy_manager.Models[key_policy][0]
        action_keys = self._policy_manager.Models[key_policy][1]
        
        n_samples = self._options['policy_training_samples']
        
        # select visited episodes
        def shoudSelectEpisode(eps, n_start, key_policy):
          if eps.R is None: return False
          
          eps_node_cur = None
          for eps_node in eps.Seq:
            if eps_node.Name == n_start:
              eps_node_cur = eps_node
              break
          if eps_node_cur is None: return False
          
          eps_node_next = None
          for eps_node in eps.Seq:
            if eps_node.Parent == eps.Seq.index(eps_node_cur):
              eps_node_next = eps_node
              break
          if eps_node_next is None: return False
          
          dyn_node = self._domain.Graph[n_start]
          Fd_name = dyn_node.Fd[dyn_node.Next.index(eps_node_next.Name)]
          
          if Fd_name == key_policy:
            return True
          else:
            return False
        
        randidx = self.DB.SearchIf(lambda eps: shoudSelectEpisode(eps, n_start, key_policy))
        random.shuffle(randidx)
        n_samples = min(n_samples, len(randidx))
        randidx = [randidx[i] for i in range(n_samples)]
        
        # optimize visited nodes with GraphDDP and train policy network
        def retriveEpisodeNode(eps, n_start):
          for eps_node in eps.Seq:
            if eps_node.Name == n_start:
              return eps_node
          return None
        
        self._policy_manager.Flush(key_policy)
        for i_idx in range(len(randidx)):
          idx = randidx[i_idx]
          eps = self.DB.GetEpisode(idx)
          eps_node = retriveEpisodeNode(eps, n_start)
          if self._options['policy_verbose']:
            print('select visited node for policy training (R: {}, node: {})'.format(eps.R, eps_node))
          
          # optimize with policy guided GraphDDP
          init_xs_policy = CopyXSSA(eps_node.XS)
          
          init_xs_policy_action = (self._policy_manager.Predict(key_policy, init_xs_policy, policy_approximators='dnn'))[0]
          for key in init_xs_policy_action:
            init_xs_policy[key] = SSA(init_xs_policy_action[key].X)
          
          ptree = self.GetPTree(n_start, xs, max_visits=self.Options['ddp_sol']['max_visits'])
          
          rand_xs_policy_actions = self.RandActions([key for key in ptree.Actions if key not in init_xs_policy])
          for key in rand_xs_policy_actions:
            init_xs_policy[key] = rand_xs_policy_actions[key]
          
          rand_xs_policy_selections = self.RandSelections([key for key in ptree.Selections if key not in init_xs_policy])
          for key in rand_xs_policy_selections:
            init_xs_policy[key] = rand_xs_policy_selections[key]
          
          res_policy = self._core.Plan(n_start, init_xs_policy)
          if self._options['policy_verbose']:
            print('optimize with policy guided GraphDDP (xs: {}, value: {})'.format(res_policy.XS, res_policy.PTree.Value()))
          
          # optimize the multistart GraphDDP
          init_xs_multistart = CopyXSSA(eps_node.XS)
          for action_key in action_keys:
            if action_key in init_xs_multistart:
              del init_xs_multistart[action_key]
          res_multistart = self._core.Plan(n_start, init_xs_multistart)
          if self._options['policy_verbose']:
            print('optimize with multistart GraphDDP (xs: {}, value: {})'.format(res_multistart.XS, res_multistart.PTree.Value()))
          
          # compare and update policy
          xs_policy = None
          value = None
          if res_multistart.PTree.Value() > res_policy.PTree.Value():
            xs_policy = res_multistart.XS
            value = res_multistart.PTree.Value()
          else:
            xs_policy = res_policy.XS
            value = res_policy.PTree.Value()
          if self._options['policy_verbose']:
            print('update policy network (key: {}, xs: {}, value: {})'.format(key_policy, xs_policy, value))
          self._policy_manager.Update(key_policy, xs_policy, xs_policy, not_learn=(i_idx < len(randidx) - 1))
    
    # predict with policy network
    shouldUsePolicy = True
    actual_res_policy = None
    if self._policy_manager is None:
      shouldUsePolicy = False
    else:
      # check if any initial action is given
      '''
      for key_policy in next_policies:
        action_keys = self._policy_manager.Models[key_policy][1]
        for action_key in action_keys:
          if action_key in xs:
            shouldUsePolicy = False
            break
        if shouldUsePolicy is False:
          break
      '''
      pass
      
      if self._options['policy_verbose']:
        print('check policy usage condition (shouldUsePolicy: {})'.format(shouldUsePolicy))
      
      # optimize actual action by policy guided GraphDDP
      if shouldUsePolicy:
        actual_xs_list = []
        
        # optimize for best action at each bifurcation
        for key_policy in next_policies:
          candidate_ys_list    = self._policy_manager.Predict(key_policy, xs, policy_approximators='all')
          candidate_xs_list    = []
          candidate_value_list = []
          best_candidate_idx   = None
          
          # optimize each action candidates
          for i_candidate_ys in range(len(candidate_ys_list)):
            candidate_ys = candidate_ys_list[i_candidate_ys]
            candidate_init_xs = CopyXSSA(xs)
            for key in candidate_ys:
              candidate_init_xs[key] = SSA(candidate_ys[key].X)
            
            # optimize action candidate with GraphDDP
            res = self._core.Plan(n_start, candidate_init_xs)
            candidate_xs    = res.XS
            candidate_value = res.PTree.Value()
            candidate_xs_list.append(candidate_xs)
            candidate_value_list.append(candidate_value)
            
            # update best candidate
            if best_candidate_idx is None or candidate_value > candidate_value_list[best_candidate_idx]:
              best_candidate_idx = i_candidate_ys
          
          # append best candidate at this bifurcation to action-value list
          actual_xs_list.append((key_policy, candidate_xs_list[best_candidate_idx], candidate_value_list[best_candidate_idx]))
          if self._options['policy_verbose']:
            print('predict bifurcation action by policy (key_policy: {}, xs: {}, value: {})'.format(
              key_policy, candidate_xs_list[best_candidate_idx], candidate_value_list[best_candidate_idx]))
        
        # construct actual action initial guess optimized by policy guided GraphDDP
        sorted(actual_xs_list, key=lambda x: x[2])
        actual_init_xs_policy = CopyXSSA(xs)
        for i in range(len(actual_xs_list)):
          key_policy_i, actual_xs_i, value_i = actual_xs_list[i]
          action_keys = self._policy_manager.Models[key_policy_i][1]
          for action_key in action_keys:
            actual_init_xs_policy[action_key] = actual_xs_i[action_key]
        if self._options['policy_verbose']:
          print('construct complete initial guess by policies (xs0: {})'.format(actual_init_xs_policy))
        
        # optimize initial guess for actual action
        actual_res_policy = self._core.Plan(n_start, actual_init_xs_policy)
        if self._options['policy_verbose']:
          print('optimize action by policy guided GraphDDP (xs0: {}, xs: {}, value: {})'.format(
            actual_init_xs_policy, actual_res_policy.XS, actual_res_policy.PTree.Value()))
    
    # optimize actual action by multistart GraphDDP
    actual_res_multistart = self._core.Plan(n_start, xs)
    if self._policy_manager is not None and self._options['policy_verbose']:
      print('optimize action by multistart GraphDDP (xs0: {}, xs: {}, value: {})'.format(
        xs, actual_res_multistart.XS, actual_res_multistart.PTree.Value()))
    
    # compare and return best planning result
    best_res = actual_res_multistart
    
    if shouldUsePolicy and actual_res_policy.PTree.Value() > best_res.PTree.Value():
      best_res = actual_res_policy
    
    if self._policy_manager is not None and self._options['policy_verbose']:
      print('select best action (xs: {}, value: {})'.format(best_res.XS, best_res.PTree.Value()))
    
    return best_res


