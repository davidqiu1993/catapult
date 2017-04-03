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



class TCatapultModelLinear(object):
  def __init__(self):
    super(TCatapultModelLinear, self).__init__()

    self._dataset = []
    loader_dataset = TCatapultDataset(auto_init=False)
    loader_dataset.load_dataset()
    for entry in loader_dataset:
      if entry['motion'] == 'linear':
        self._dataset.append(entry)

    self._model = self._create_model()

  def _create_model(self):
    model = Sequential()
    model.add(Dense(32, input_dim=3, activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer=RMSprop(lr=0.05))
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

    self._model.fit(X, Y, nb_epoch=nb_epoch)

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

    self._model.fit(X, Y, nb_epoch=nb_epoch)

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

  def run(self):
    self._train_model_batch(nb_epoch=256, cheat=True)
    #for i in range(64):
    #  self._train_model_minibatch(batch_size=32, cheat=False)
    self._validate(self._dataset)


if __name__ == '__main__':
  catapultModel = TCatapultModelLinear()
  catapultModel.run()


