#!/usr/bin/python

"""
Catapult linear motion modeling.
"""

__author__    = 'David Qiu'
__email__     = 'david@davidqiu.com'
__website__   = 'www.davidqiu.com'
__copyright__ = 'Copyright (C) 2017, David Qiu. All rights reserved.'


from CatapultDataset import *

import time
import datetime
import yaml
import sys
import numpy as np
import random

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop, SGD

import pdb

sys.path.append(os.path.join(os.path.dirname(__file__), '../lib/cma'))
import cma



class TCatapultModelLinear(object):
  def __init__(self, abs_dirpath_data=None):
    super(TCatapultModelLinear, self).__init__()

    self._dataset = []
    loader_dataset = None
    if abs_dirpath_data is None:
      loader_dataset = TCatapultDataset(auto_init=False)
    else:
      loader_dataset = TCatapultDataset(abs_dirpath=abs_dirpath_data, auto_init=False)
    loader_dataset.load_dataset()
    for entry in loader_dataset:
      if entry['motion'] == 'linear':
        self._dataset.append(entry)


    self._model = self._create_model()

  def _create_model(self):
    model = Sequential()
    model.add(Dense(128, input_dim=3, activation='relu'))
    model.add(Dropout(0.05))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.05))
    #model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.01))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=RMSprop(lr=0.01))
    #model.compile(loss='mse', optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True))
    return model

  def _train_model_minibatch(self, batch_size=64, nb_epoch=8, cheat=False, cheat_threshold=750):
    X = []
    Y = []
    batch_indices = random.sample([(i) for i in range(len(self._dataset))], min(batch_size, len(self._dataset)))
    for index in batch_indices:
      entry = self._dataset[index]
      X_i = [entry['action']['pos_init_actual'], 
             entry['action']['pos_target_actual'], 
             entry['action']['duration'] * 1000.]
      Y_i = [entry['result']['loc_land']]
      X.append(X_i)
      Y.append(Y_i)
      if cheat and entry['result']['loc_land'] > cheat_threshold:
        for i_cheat in range(2):
          X.append(X_i)
          Y.append(Y_i)

    self._model.fit(X, Y, batch_size=len(batch_indices), nb_epoch=nb_epoch, verbose=False)

  def _train_model_batch(self, nb_epoch=8, cheat=False, cheat_threshold=750):
    X = []
    Y = []
    for entry in self._dataset:
      X_i = [entry['action']['pos_init_actual'], 
             entry['action']['pos_target_actual'], 
             entry['action']['duration'] * 1000.]
      Y_i = [entry['result']['loc_land']]
      X.append(X_i)
      Y.append(Y_i)
      if cheat and entry['result']['loc_land'] > cheat_threshold:
        for i_cheat in range(2):
          X.append(X_i)
          Y.append(Y_i)

    self._model.fit(X, Y, batch_size=len(X), nb_epoch=nb_epoch)

  def _validate(self, validation_dataset, cheat_threshold=750):
    X_valid = []
    Y_valid = []
    for entry in validation_dataset:
      X_i = [entry['action']['pos_init_actual'], 
             entry['action']['pos_target_actual'], 
             entry['action']['duration'] * 1000.]
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
        np.round(X_valid[i][0], 0),   np.round(X_valid[i][1], 0),   np.round(X_valid[i][2] / 1000., 2),
        np.round(Y_valid[i][0], 0),
        np.round(Y_predict[i][0], 0),
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

  def _plot_predictions(self):
    pass

  def _penalize_action(self, pos_init, pos_target, duration):
    prefix = 'penalize_action'
    prefix_info = prefix + ':'
    
    POS_MIN = 0
    POS_MID = 420
    POS_MAX = 840
    DURATION_MIN = 0.01
    DURATION_MAX = 0.60

    corrected_pos_init   = pos_init
    corrected_pos_target = pos_target
    corrected_duration   = duration
    
    penalty = 0
    penalty_factor = 1
    
    min_pos_diff = 20
    
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
    
    if pos_target <= (corrected_pos_init + min_pos_diff):
      cur_penalty = np.abs(pos_target - (corrected_pos_init + min_pos_diff)) * penalty_factor
      penalty += cur_penalty
      print(prefix_info, 'penalty = {} ({})'.format(cur_penalty, 'pos_target <= (corrected_pos_init + min_pos_diff)'))
      corrected_pos_target = (corrected_pos_init + min_pos_diff)
    if pos_target > POS_MAX:
      cur_penalty = np.abs(pos_target - POS_MAX) * penalty_factor
      penalty += cur_penalty
      print(prefix_info, 'penalty = {} ({})'.format(cur_penalty, 'pos_target > POS_MAX'))
      corrected_pos_target = POS_MAX
    
    if duration < DURATION_MIN:
      cur_penalty = np.abs(duration - DURATION_MIN) * 1000 * penalty_factor
      penalty += cur_penalty
      print(prefix_info, 'penalty = {} ({})'.format(cur_penalty, 'duration < DURATION_MIN'))
      corrected_duration = DURATION_MIN
    if duration > DURATION_MAX:
      cur_penalty = np.abs(duration - DURATION_MAX) * 1000 * penalty_factor
      penalty += cur_penalty
      print(prefix_info, 'penalty = {} ({})'.format(cur_penalty, 'duration > DURATION_MAX'))
      corrected_duration = DURATION_MAX
    
    return corrected_pos_init, corrected_pos_target, corrected_duration, penalty

  def _optimize_cma(self):
    args = {
      'count_iter': 0
    }

    def f(x, args):
      print('CMA-ES iteration. (iteration = {})'.format(args['count_iter']))
      args['count_iter'] += 1

      pos_init, pos_target, duration_scaled = x
      pos_init = int(pos_init)
      pos_target = int(pos_target)
      duration = np.round(duration_scaled / 1000., 2)
      print('CMA-ES sample. (pos_init = {}, pos_target = {}, duration = {})'.format(pos_init, pos_target, duration))
      pos_init, pos_target, duration, penalty = self._penalize_action(pos_init, pos_target, duration)
      print('Test sample. (pos_init = {}, pos_target = {}, duration = {})'.format(pos_init, pos_target, duration))

      predictions = self._model.predict([[pos_init, pos_target, duration * 1000.], [pos_init, pos_target, duration * 1000.]])
      y_predict = predictions[0]
      loc_land_predict = y_predict[0]

      loss = - loc_land_predict + penalty
      print('loss = {}, loc_land_predict = {}, penalty = {}'.format(np.round(loss, 2), np.round(loc_land_predict, 2), np.round(penalty, 2)))
      print('')

      return loss

    res = cma.fmin(f, [200, 400, 0.3 * 1000], 300., [args], popsize=20, tolx=1.0, verb_disp=False, verb_log=0)

    optimal_action = {
      'pos_init': int(np.round(res[0][0])),
      'pos_target': int(res[0][1]),
      'duration': np.round(res[0][2] / 1000., 2)
    }

    return optimal_action

  def run(self):
    training_method   = 'minibatch' # 'batch', 'minibatch'
    batch_rounds      = 128
    minibatch_size    = 64
    minibatch_rounds  = int(len(self._dataset) / minibatch_size) * int(np.sqrt(len(self._dataset))) * 2 + 1
    minibatch_epochs  = 8

    if training_method == 'batch':
      self._train_model_batch(nb_epoch=batch_rounds, cheat=True)
    elif training_method == 'minibatch':
      for i in range(minibatch_rounds):
        self._train_model_minibatch(batch_size=minibatch_size, nb_epoch=minibatch_epochs, cheat=False)
        print('minibatch: {}/{}'.format(i+1, minibatch_rounds))
    else:
      assert(False)
    
    self._validate(self._dataset)

    should_optimize_input = input('optimize (Y/n)?> ')
    if (should_optimize_input == '' or should_optimize_input == 'Y' or should_optimize_input == 'y'):
      print(self._optimize_cma())


if __name__ == '__main__':
  catapult_name = 'catapult_001'
  abs_dirpath_data = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/' + catapult_name))

  catapultModel = TCatapultModelLinear(abs_dirpath_data=abs_dirpath_data)
  catapultModel.run()


