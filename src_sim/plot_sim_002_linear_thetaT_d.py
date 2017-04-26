#!/usr/bin/python

"""
Simulation catapult optimization test dataset plotting.
"""

__author__    = 'David Qiu'
__email__     = 'david@davidqiu.com'
__website__   = 'www.davidqiu.com'
__copyright__ = 'Copyright (C) 2017, David Qiu. All rights reserved.'


from libCatapultDatasetSim import TCatapultDatasetSim

import sys
import yaml
import matplotlib.pyplot as plt
import math

import pdb

try:
  input = raw_input
except NameError:
  pass



def _estimate_test_results(test_results, dataset):
  prefix = 'estimate_test_results'
  prefix_info = prefix + ':'
  
  SHOULD_SAVE_TEST_RESULTS = True
  
  samples_actual_loc_land = []
  samples_actual_pos_target = []
  for entry in dataset:
    samples_actual_loc_land.append(float(entry['result']['loc_land']))
    samples_actual_pos_target.append(float(entry['action']['pos_target']))
  
  samples_test_desired_loc_land = []
  samples_test_pos_target = []
  samples_test_loc_land = []
  for entry in test_results:
    approach              = str(entry['approach'])
    desired_loc_land      = float(entry['desired_loc_land'])
    loc_land              = float(entry['loc_land'])
    pos_target            = float(entry['pos_target'])
    n_preopt_samples      = int(entry['preopt_samples'])
    n_samples             = int(entry['samples'])
    n_preopt_simulations  = int(entry['preopt_simulations'])
    n_simulations         = int(entry['simulations'])
    samples_test_desired_loc_land.append(desired_loc_land)
    samples_test_pos_target.append(pos_target)
    samples_test_loc_land.append(loc_land)
  
  plt.figure()
  
  plt.subplot(211)
  plt.xlabel('d*')
  plt.ylabel('thetaT')
  plt.plot(samples_actual_loc_land, samples_actual_pos_target, 'bx')
  plt.plot(samples_test_desired_loc_land, samples_test_pos_target, 'ro')
  
  plt.subplot(212)
  plt.xlabel('d*')
  plt.ylabel('d')
  plt.plot(samples_actual_loc_land, samples_actual_loc_land, 'b-')
  plt.plot(samples_test_desired_loc_land, samples_test_loc_land, 'ro')
  
  plt.show()



def _estimate_online_learning(test_results, dataset):
  pass



def _getOperations():
  operations_list = [
    {
      'code': '1',
      'desc': 'general test result estimation',
      'func': _estimate_test_results
    },
    {
      'code': '2',
      'desc': 'online learning test result estimation',
      'func': _estimate_online_learning
    }
  ]
  
  return operations_list



if __name__ == '__main__':
  if len(sys.argv) != 3:
    print('usage: ./plot_sim_002_linear_thetaT_d.py <test_results_yaml> <dataset_dir>')
    quit()
  
  # Load test results
  with open(sys.argv[1], 'r') as yaml_file:
    test_results = yaml.load(yaml_file)
  assert(len(test_results) > 0)
  print('')
  
  # Load reference dataset
  loader_dataset = TCatapultDatasetSim(abs_dirpath=sys.argv[2], auto_init=False)
  loader_dataset.load_dataset()
  
  # Print test result entry structure
  print('test result entry structure:')
  entry_items = []
  max_entry_item_len = 0
  for o in test_results[0]:
    entry_items.append(o)
    if len(o) > max_entry_item_len:
      max_entry_item_len = len(o)
  entry_items.sort()
  print('  - approach: {}'.format(test_results[0]['approach']))
  for i in range(len(entry_items)):
    o = entry_items[i]
    if o != 'approach':
      print(('  - {0: <' + str(max_entry_item_len + 1) + '} {1}').format(o + ':', type(test_results[0][o])))
  print('')
  
  # Get operations list
  operations_list = _getOperations()
  assert(len(operations_list) > 0)
  
  # Query operation
  print('operations:')
  for i in range(len(operations_list)):
    print('  - ({}) {}'.format(operations_list[i]['code'], operations_list[i]['desc']))
  operation_code_input = input('select operation (1/..)> ').strip().lower()
  operation_code = str(operation_code_input)
  print('')
  
  # Select operation
  operation_index = 0 # default operation index
  for i in range(len(operations_list)):
    if operation_code == operations_list[i]['code']:
      operation_index = i
  op_func = operations_list[operation_index]['func']
  op_func(test_results, loader_dataset)
  
  # Quit
  quit()


