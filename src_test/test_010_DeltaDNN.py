#!/usr/bin/python

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt

import pdb

sys.path.append(os.path.join(os.path.dirname(__file__), '../lib/DeltaDNN'))
from base_ml_dnn import TNNRegression
from base_util import LoadYAML, SaveYAML, Rand, FRange1



class DeltaDNNTest(object):
  def __init__(self, dp_model_load=None, dp_model_save=None, fp_dataset=None, fp_result=None, plot=False):
    super(DeltaDNNTest, self).__init__()
    
    self.dirpath_model_load = dp_model_load # Model loading directory, with tailing '/'
    self.dirpath_model_save = dp_model_save # Model saving directory, with tailing '/'
    self.filepath_dataset   = fp_dataset # Training and validation dataset, with extension '.yaml'
    self.filepath_result    = fp_result # Training and validation result, with extension '.yaml'
    
    self.plot = plot # Only one-dimensional data can be plotted
    
    self._func_dict = self._defineTestFunctions()
    
    self._model = None
    
  def _defineTestFunctions(self):
    func_dict = {
      'linear':       lambda x: 0.5 * x,
      'quadratic':    lambda x: 2.0 * x**2,
      'sin':          lambda x: 1.2 + np.sin(3 * x),
      'step':         lambda x: np.array([(0.0 if xi < 1.0 else 4.0) for xi in x]),
      'step_slope':   lambda x: np.array([(4.0 - xi if xi > 0.0 else 0.0) for xi in x]),
      'rect_bulge':   lambda x: np.array([(4.0 if 0.0 < xi and xi < 2.5 else 0.0) for xi in x]),
      'double_step':  lambda x: np.array([(0.0 if xi < 0.0 else (2.0 if xi < 2.5 else 4.0)) for xi in x])
    }
    
    return func_dict
    
  def _generateSamples(self, func_name, bound, n, noise=None, uniform=False):
    assert(func_name in self._func_dict)
    
    bound = np.array(bound)
    dim = len(bound)
    assert(dim > 0)
    assert(bound.shape[1] == 2)
    for bound_i in bound:
      assert(bound_i[0] < bound_i[1])
    
    assert(n > 0)
    
    if noise is None:
      noise = [0.0 for d in range(dim)]
    noise = np.array(noise)
    assert(noise.shape[0] == dim)
    
    f = self._func_dict[func_name]
    
    data_x = []
    if uniform:
      data_x_dim = [[(xi) for xi in FRange1(*(bound[d]), num_div=n)] for d in range(dim)]
      for d in range(len(data_x_dim)):
        if len(data_x) == 0:
          for i in range(len(data_x_dim[d])):
            data_x.append(np.array([data_x_dim[d][i]]))
        else:
          new_data_x = []
          for i in range(len(data_x_dim[d])):
            for j in range(len(data_x)):
              new_xi = data_x[j].copy()
              new_xi = np.append(new_xi, [data_x_dim[d][i]], axis=0)
              new_data_x.append(new_xi)
          data_x = new_data_x
    else:
      data_x = [np.array([(Rand(*bound[d])) for d in range(dim)]) for i in range(n**dim)]
    
    data_y = [(f(x) + [(noise[d] * Rand()) for d in range(dim)]) for x in data_x] # [-0.5, 0.5) ~ Rand()
    
    return data_x, data_y
  
  def _createModel(self, input_dim, output_dim, hiddens=[128, 128], max_updates=20000):
    model = TNNRegression()
    
    options = {
      #'AdaDelta_rho':         0.5, # 0.5, 0.9
      'dropout':              True,
      'dropout_ratio':        0.01,
      'loss_stddev_stop':     1.0e-4,
      'loss_stddev_stop_err': 1.0e-6,
      #'batchsize':            5, # 5, 10
      #'num_check_stop':       50,
      #'loss_maf_alpha':       0.4,
      'num_max_update':       max_updates,
      'gpu':                  -1,
      'verbose':              True,
      'n_units':              [input_dim] + hiddens + [output_dim]
    }
    model.Load({'options':options})
    
    if self.dirpath_model_load is not None:
      model.Load(LoadYAML(self.dirpath_model_load + 'nn_model.yaml'), self.dirpath_model_load)
    
    model.Init()
    
    if self.dirpath_model_save is not None:
      SaveYAML(model.Save(self.dirpath_model_save), self.dirpath_model_save + 'nn_model.yaml')
    
    return model
  
  def _trainModel(self, x_train, y_train, batch_train=True):
    if batch_train:
      self._model.UpdateBatch(x_train, y_train)
    else:
      for x, y, n in zip(x_train, y_train, range(len(x_train))):
        self._model.Update(x, y, not_learn=((n+1) % min(10, len(x_train)) != 0))
    
    if self.dirpath_model_save is not None:
      SaveYAML(self._model.Save(self.dirpath_model_save), self.dirpath_model_save + 'nn_model.yaml')
  
  def test_same_dim(self, func_name, bound, n_train=100, n_valid=25, noise=None, hiddens=[128, 128], max_updates=20000):
    dim = len(bound)
    self._model = self._createModel(dim, dim, hiddens=hiddens, max_updates=max_updates)
    
    x_train, y_train = self._generateSamples(func_name, bound, n_train, noise=noise, uniform=False)
    x_valid, y_valid = self._generateSamples(func_name, bound, n_valid, noise=None,  uniform=True )
    
    if self.filepath_dataset is not None:
      print 'TODO: Save dataset...'
    
    if self.dirpath_model_load is None:
      self._trainModel(x_train, y_train, batch_train=True)
    
    y_hypo    = []
    err_hypo  = []
    grad_hypo = []
    for i in range(len(x_valid)):
      prediction = self._model.Predict(x_valid[i], x_var=0.0**2, with_var=True, with_grad=True)
      h_i    = prediction.Y.ravel() # hypothesis
      err_i  = np.sqrt(np.diag(prediction.Var))
      grad_i = prediction.Grad.ravel()
      
      y_hypo.append(h_i)
      err_hypo.append(err_i)
      grad_hypo.append(grad_i)
    
    y_diff = []
    y_stderr = []
    for i in range(len(y_valid)):
      y_diff_i   = y_valid[i] - y_hypo[i]
      y_stderr_i = np.linalg.norm(y_diff_i)
      
      y_diff.append(y_diff_i)
      y_stderr.append(y_stderr_i)
    
    err_diff = []
    err_stderr = []
    for i in range(len(y_stderr)):
      err_diff_i   = y_stderr[i] - err_hypo[i]
      err_stderr_i = np.linalg.norm(err_diff_i)
      
      err_diff.append(err_diff_i)
      err_stderr.append(err_stderr_i)
    
    acc_stderr_y   = 0.0
    acc_stderr_err = 0.0
    for y_stderr_i in y_stderr:
      acc_stderr_y += y_stderr_i
    for err_stderr_i in err_stderr:
      acc_stderr_err += err_stderr_i
    ave_stderr_y   = acc_stderr_y / len(y_stderr)
    ave_stderr_err = acc_stderr_err / len(err_stderr)
    
    if self.filepath_result is not None:
      print 'TODO: Save result...'
    
    if self.plot and dim == 1:
      plot_x_train  = [(x[0]) for x in x_train]
      plot_y_train  = [(y[0]) for y in y_train]
      plot_x_valid  = [(x[0]) for x in x_valid]
      plot_y_valid  = [(y[0]) for y in y_valid]
      plot_y_hypo   = [(y[0]) for y in y_hypo]
      plot_err_hypo = [(e[0]) for e in err_hypo]
      plot_y_diff   = [(y[0]) for y in y_diff]
      plot_y_stderr = [(e)    for e in y_stderr]
      
      plt.figure(1)
      plt.clf()
      
      plt.subplot(211)
      if self.dirpath_model_load is None:
        plt.plot(plot_x_train, plot_y_train, 'ro')
      plt.plot(plot_x_valid, plot_y_valid, 'g--',
               plot_x_valid, plot_y_hypo,  'b-')
      plt.ylabel('y')
      plt.grid(True)
      
      plt.subplot(212)
      plt.plot(plot_x_valid, plot_y_diff,   'ro',
               plot_x_valid, plot_y_stderr, 'g--',
               plot_x_valid, plot_err_hypo, 'b-')
      plt.ylabel('err')
      plt.grid(True)
      
      plt.show()
    
    return ave_stderr_y, ave_stderr_err

  def test_structures(self, bound, n_train=1000, n_valid=64, fp_result=None):
    n_samples = 8
    n_hidden_unit = 16
    
    dim = len(bound)
    
    feature_dict = {
      'func': ['linear', 'double_step'], #[(func) for func in self._func_dict],
      'max_updates': [12000], #[(i * 5000) for i in range(1, 5)],
      'noise_magitude': [0.0, 0.20], #[(i * 0.05) for i in range(5)],
      'hiddens_depth': [(i) for i in range(1, 9)],
      'hiddens_width': [(i) for i in range(1, 9)]
    }
    
    feature_space = []
    for feature in feature_dict:
      if len(feature_space) == 0:
        for feature_i in feature_dict[feature]:
          feature_space.append({feature: feature_i})
      else:
        new_feature_space = []
        for feature_i in feature_dict[feature]:
          for feature_space_i in feature_space:
            new_feature_space_i = feature_space_i.copy() # deep copy
            new_feature_space_i[feature] = feature_i
            new_feature_space.append(new_feature_space_i)
        feature_space = new_feature_space
    
    result = []
    for i_feature_comb in range(len(feature_space)):
      feature_comb = feature_space[i_feature_comb]
      
      func = feature_comb['func']
      max_updates = feature_comb['max_updates']
      noise_magitude = feature_comb['noise_magitude']
      hiddens_depth = feature_comb['hiddens_depth']
      hiddens_width = feature_comb['hiddens_width']
      
      noise = [noise_magitude for d in range(dim)]
      hiddens = [(hiddens_width * n_hidden_unit) for layer in range(hiddens_depth)]
      
      x_acc_stderr_y = 0.0
      x_acc_stderr_err = 0.0
      for i_sample in range(n_samples):
        ave_stderr_y, ave_stderr_err = self.test_same_dim(func, bound, n_train=n_train, n_valid=n_valid, noise=noise, hiddens=hiddens, max_updates=max_updates)
        x_acc_stderr_y += ave_stderr_y
        x_acc_stderr_err += ave_stderr_err
        print 'progress (sample): ' + str(i_sample + 1) + '/' + str(n_samples) + ', ' + str(i_feature_comb) + '/' + str(len(feature_space))
      x_ave_stderr_y   = x_acc_stderr_y / n_samples
      x_ave_stderr_err = x_acc_stderr_err / n_samples
      
      result_i = {
        'feature_comb': feature_comb,
        'ave_stderr_y': x_ave_stderr_y,
        'ave_stderr_err': x_ave_stderr_err
      }
      
      print 'progress (feature): ' + str(i_feature_comb + 1) + '/' + str(len(feature_space))
      print result_i
      
      result.append(result_i)
      
      if fp_result is not None:
        SaveYAML(result, fp_result)
    
    return result
    

if __name__ == '__main__':
  tester = DeltaDNNTest(dp_model_load=None, #'/tmp/dnn/', 
                        dp_model_save=None, #'/tmp/dnn/', 
                        fp_dataset=None, #'/tmp/dnn/nn_dataset.yaml', 
                        fp_result=None, #'/tmp/dnn/nn_result.yaml', 
                        plot=True)
  
  if True:
    ave_stderr_y, ave_stderr_err = tester.test_same_dim('step', [[-3.0, 5.0]], noise=[0.2], max_updates=10000)
    print 'ave_stderr_y =', ave_stderr_y
    print 'ave_stderr_err =', ave_stderr_err
  
  if False:
    result = tester.test_structures([[-5.0, 5.0]], fp_result='david_dnn_result_2_tmp.yaml')
    
    for result_i in result:
      print result_i['feature_comb']
      print 'ave_stderr_y =', result_i['ave_stderr_y']
      print 'ave_stderr_err =', result_i['ave_stderr_err']
    
    SaveYAML(result, 'david_dnn_result_2.yaml')


