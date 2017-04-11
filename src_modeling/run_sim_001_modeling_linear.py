#!/usr/local/bin/python3

"""
Simulation catapult linear motion modeling.
"""

__author__    = 'David Qiu'
__email__     = 'david@davidqiu.com'
__website__   = 'www.davidqiu.com'
__copyright__ = 'Copyright (C) 2017, David Qiu. All rights reserved.'


from libCatapultDatasetSim import TCatapultDatasetSim

import os
import time
import datetime
import yaml
import sys
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop, SGD

import pdb

sys.path.append(os.path.join(os.path.dirname(__file__), '../lib/cma'))
import cma



class TCatapultModelLinearSim(object):
  def __init__(self, abs_dirpath_data=None):
    super(TCatapultModelLinearSim, self).__init__()

    self._dataset = []
    loader_dataset = TCatapultDatasetSim(abs_dirpath=abs_dirpath_data, auto_init=False)
    loader_dataset.load_dataset()
    for entry in loader_dataset:
      if entry['motion'] == 'linear':
      #if entry['motion'] == 'linear' and np.round(entry['action']['duration'], 2) == 0.05:
        self._dataset.append(entry)

    self._model = self._create_model()

  def _create_model(self):
    dropout_rate = 0.01

    model = Sequential()
    model.add(Dense(64, input_dim=3, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=RMSprop(lr=0.01))
    #model.compile(loss='mse', optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True))
    return model

  def _train_model_minibatch(self, batch_size=64, nb_epoch=8, cheat=False, cheat_threshold=50):
    X = []
    Y = []
    batch_indices = random.sample([(i) for i in range(len(self._dataset))], min(batch_size, len(self._dataset)))
    for index in batch_indices:
      entry = self._dataset[index]
      X_i = [entry['action']['pos_init'], 
             entry['action']['pos_target'], 
             entry['action']['duration']]
      Y_i = [entry['result']['loc_land']]
      X.append(X_i)
      Y.append(Y_i)
      if cheat and entry['result']['loc_land'] > cheat_threshold:
        for i_cheat in range(2):
          X.append(X_i)
          Y.append(Y_i)

    self._model.fit(X, Y, batch_size=len(batch_indices), nb_epoch=nb_epoch, verbose=False)

  def _train_model_batch(self, nb_epoch=8, cheat=False, cheat_threshold=50):
    X = []
    Y = []
    for entry in self._dataset:
      X_i = [entry['action']['pos_init'], 
             entry['action']['pos_target'], 
             entry['action']['duration']]
      Y_i = [entry['result']['loc_land']]
      X.append(X_i)
      Y.append(Y_i)
      if cheat and entry['result']['loc_land'] > cheat_threshold:
        for i_cheat in range(2):
          X.append(X_i)
          Y.append(Y_i)

    self._model.fit(X, Y, batch_size=len(X), nb_epoch=nb_epoch)

  def _validate(self, validation_dataset, cheat_threshold=50):
    X_valid = []
    Y_valid = []
    for entry in validation_dataset:
      X_i = [entry['action']['pos_init'], 
             entry['action']['pos_target'], 
             entry['action']['duration']]
      Y_i = [entry['result']['loc_land']]
      X_valid.append(X_i)
      Y_valid.append(Y_i)

    Y_predict = self._model.predict(X_valid)

    acc_err_loc_land = 0
    acc_err_loc_land_cheat = 0
    count_cheat = 0
    for i in range(len(X_valid)):
      err_loc_land = np.abs(Y_valid[i][0] - Y_predict[i][0])
      print('{} {} {} => {} / {} ~ {}'.format(
        np.round(X_valid[i][0], 6),   np.round(X_valid[i][1], 6),   np.round(X_valid[i][2], 6),
        np.round(Y_valid[i][0], 2),
        np.round(Y_predict[i][0], 2),
        np.round(err_loc_land)
      ))

      acc_err_loc_land += err_loc_land

      if Y_valid[i][0] > cheat_threshold:
        acc_err_loc_land_cheat += err_loc_land
        count_cheat += 1

    ave_err_loc_land = acc_err_loc_land / len(X_valid)
    ave_err_loc_land_cheat = acc_err_loc_land_cheat / float(count_cheat)

    print('acc_err_loc_land =', np.round(ave_err_loc_land, 2))
    print('acc_err_loc_land_cheat =', np.round(ave_err_loc_land_cheat, 2))

  def _plot_prediction_errors(self, validation_dataset):
    LOC_MIN = -1
    LOC_MAX = 125

    UPPER_BOUND_FACTOR = 1.10

    n_samples = len(validation_dataset)
    plt.figure(num=1, figsize=(8, 6), dpi=120, facecolor='w', edgecolor='k')

    X_valid = []
    Y_valid = []
    for entry in validation_dataset:
      X_i = [entry['action']['pos_init'], 
             entry['action']['pos_target'], 
             entry['action']['duration']]
      Y_i = [entry['result']['loc_land']]
      X_valid.append(X_i)
      Y_valid.append(Y_i)

    Y_predict = self._model.predict(X_valid)

    plt.title('Prediction Errors (b: ori, r: pred)')
    #plt.axis([LOC_MIN, LOC_MAX * UPPER_BOUND_FACTOR, LOC_MIN, LOC_MAX * UPPER_BOUND_FACTOR])
    plt.xlabel('distance')
    plt.ylabel('loc_land')
    loc_land_original   = [(Y_valid[i][0])   for i in range(n_samples)]
    loc_land_prediction = [(Y_predict[i][0]) for i in range(n_samples)]
    plt.scatter(loc_land_original, loc_land_original,   color='b')
    plt.scatter(loc_land_original, loc_land_prediction, color='r')

    plt.show()

  def _plot_prediction_errors_pos(self, validation_dataset, duration=0.05):
    LOC_MIN = -1
    LOC_MAX = 125

    UPPER_BOUND_FACTOR = 1.10

    dataset = []
    for entry in validation_dataset:
      if np.round(entry['action']['duration'], 2) == duration:
        dataset.append(entry)

    n_samples = len(dataset)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X_valid = []
    Y_valid = []
    for entry in dataset:
      X_i = [entry['action']['pos_init'], 
             entry['action']['pos_target'], 
             entry['action']['duration']]
      Y_i = [entry['result']['loc_land']]
      X_valid.append(X_i)
      Y_valid.append(Y_i)

    Y_predict = self._model.predict(X_valid)

    #plt.title('Prediction Errors (b: ori, r: pred)')
    ax.set_xlabel('pos_init')
    ax.set_ylabel('pos_target')
    ax.set_zlabel('loc_land')

    pos_init            = [(X_valid[i][0])   for i in range(n_samples)]
    pos_target          = [(X_valid[i][1])   for i in range(n_samples)]
    loc_land_original   = [(Y_valid[i][0])   for i in range(n_samples)]
    loc_land_prediction = [(Y_predict[i][0]) for i in range(n_samples)]

    plt.scatter(pos_init, pos_target, zs=loc_land_original,   c='b', marker='o', s=3.0)
    plt.scatter(pos_init, pos_target, zs=loc_land_prediction, c='r', marker='^', s=3.0)

    plt.show()

  def _penalize_action(self, pos_init, pos_target, duration):
    prefix = 'penalize_action'
    prefix_info = prefix + ':'
    
    POS_MIN = 0.00 * math.pi
    POS_MID = 0.50 * math.pi
    POS_MAX = 0.75 * math.pi
    DURATION_MIN = 0.01
    DURATION_MAX = 0.60

    corrected_pos_init   = pos_init
    corrected_pos_target = pos_target
    corrected_duration   = duration
    
    penalty = 0
    penalty_factor = 1
    
    min_pos_diff = 0.1 * math.pi
    
    if pos_init < POS_MIN:
      cur_penalty = np.abs(pos_init - POS_MIN) * penalty_factor
      penalty += cur_penalty
      print(prefix_info, 'penalty = {} ({})'.format(cur_penalty, 'pos_init < POS_MIN'))
      corrected_pos_init = POS_MIN
    if pos_init > POS_MID:
      cur_penalty = np.abs(pos_init - POS_MID) * penalty_factor
      penalty += cur_penalty
      print(prefix_info, 'penalty = {} ({})'.format(cur_penalty, 'pos_init > POS_MID'))
      corrected_pos_init = POS_MID
    
    if pos_target < (corrected_pos_init + min_pos_diff):
      cur_penalty = np.abs(pos_target - (corrected_pos_init + min_pos_diff)) * penalty_factor
      penalty += cur_penalty
      print(prefix_info, 'penalty = {} ({})'.format(cur_penalty, 'pos_target < (corrected_pos_init + min_pos_diff)'))
      corrected_pos_target = (corrected_pos_init + min_pos_diff)
    if pos_target > POS_MAX:
      cur_penalty = np.abs(pos_target - POS_MAX) * penalty_factor
      penalty += cur_penalty
      print(prefix_info, 'penalty = {} ({})'.format(cur_penalty, 'pos_target > POS_MAX'))
      corrected_pos_target = POS_MAX
    
    if duration < DURATION_MIN:
      cur_penalty = np.abs(duration - DURATION_MIN) * penalty_factor
      penalty += cur_penalty
      print(prefix_info, 'penalty = {} ({})'.format(cur_penalty, 'duration < DURATION_MIN'))
      corrected_duration = DURATION_MIN
    if duration > DURATION_MAX:
      cur_penalty = np.abs(duration - DURATION_MAX) * penalty_factor
      penalty += cur_penalty
      print(prefix_info, 'penalty = {} ({})'.format(cur_penalty, 'duration > DURATION_MAX'))
      corrected_duration = DURATION_MAX
    
    return corrected_pos_init, corrected_pos_target, corrected_duration, penalty

  def _optimize_cma_throw_farther(self):
    args = {
      'count_iter': 0
    }

    def f(x, args):
      print('CMA-ES iteration. (iteration = {})'.format(args['count_iter']))
      args['count_iter'] += 1

      pos_init, pos_target, duration = x
      print('CMA-ES sample. (pos_init = {}, pos_target = {}, duration = {})'.format(pos_init, pos_target, duration))
      pos_init, pos_target, duration, penalty = self._penalize_action(pos_init, pos_target, duration)
      print('Test sample. (pos_init = {}, pos_target = {}, duration = {})'.format(pos_init, pos_target, duration))

      predictions = self._model.predict([[pos_init, pos_target, duration], [pos_init, pos_target, duration]])
      y_predict = predictions[0]
      loc_land_predict = y_predict[0]

      loss = - loc_land_predict + penalty
      print('loss = {}, loc_land_predict = {}, penalty = {}'.format(np.round(loss, 2), np.round(loc_land_predict, 2), np.round(penalty, 2)))
      print('')

      return loss

    res = cma.fmin(f, [0.1 * math.pi, 0.6 * math.pi, 0.3], 0.5, [args], 
                   popsize=20, tolx=0.001, verb_disp=False, verb_log=0)

    optimal_action = {
      'pos_init': res[0][0],
      'pos_target': res[0][1],
      'duration': res[0][2]
    }

    return optimal_action

  def run(self):
    training_method   = 'batch' # 'batch', 'minibatch'
    batch_rounds      = 512
    minibatch_size    = 64
    minibatch_rounds  = int(len(self._dataset) / minibatch_size) * int(np.sqrt(len(self._dataset))) * 2 + 1
    minibatch_epochs  = 8

    ready_input = input('train with ' + training_method + ', ready (Y)?> ')
    if training_method == 'batch':
      self._train_model_batch(nb_epoch=batch_rounds, cheat=False, cheat_threshold=50)
    elif training_method == 'minibatch':
      for i in range(minibatch_rounds):
        self._train_model_minibatch(batch_size=minibatch_size, nb_epoch=minibatch_epochs, cheat=False, cheat_threshold=50)
        print('minibatch: {}/{}'.format(i+1, minibatch_rounds))
    else:
      assert(False)
    
    self._validate(self._dataset)

    while True:
      print('options:')
      print('  (1) plotting prediction errors')
      print('  (2) optimization with CMA-ES')
      print('  (3) plotting prediction errors w.r.t. pos')
      print('  (q) quit (default)')
      option_input = input('option (Q/1/2/...)?> ')
      option = option_input.lower()
      if option == '1':
        self._plot_prediction_errors(self._dataset)
      elif option == '2':
        res = self._optimize_cma_throw_farther()
        print(res)
      elif option == '3':
        self._plot_prediction_errors_pos(self._dataset, duration=0.05)
      else: # (q)
        print('Exit the program.')
        quit()



if __name__ == '__main__':
  catapult_name = 'catapult_sim_001'
  abs_dirpath_data = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/' + catapult_name))

  catapultModel = TCatapultModelLinearSim(abs_dirpath_data=abs_dirpath_data)
  catapultModel.run()


