#!/usr/bin/python

"""
Simulation catapult optimization test dataset plotting.
"""

__author__    = 'David Qiu'
__email__     = 'david@davidqiu.com'
__website__   = 'www.davidqiu.com'
__copyright__ = 'Copyright (C) 2017, David Qiu. All rights reserved.'


from libCatapultDatasetSim import TCatapultDatasetSim

import os
import sys
import yaml
import matplotlib.pyplot as plt
import math

import pdb

try:
  input = raw_input
except NameError:
  pass


def _select_test_results_files(test_results_dir, max_select=None):
  yaml_filenames = []
  yaml_filenames_selected = []
  
  for dirname, dirnames, filenames in os.walk(test_results_dir):
    for filename in filenames:
      if '.yaml' in filename:
        yaml_filenames.append(filename)
  
  should_end_select = False
  while not should_end_select:
    if len(yaml_filenames) <= 0:
      break;
    if max_select is not None and len(yaml_filenames_selected) >= max_select:
      break;
    
    print('test results files available:')
    for yaml_filename in yaml_filenames:
      print ('  - {}'.format(yaml_filename))
    
    should_reselect = True
    while should_reselect:
      file_selected_input = input('select test results file (empty to end): ').strip()
      file_selected = str(file_selected_input)
      
      if file_selected == '':
        should_end_select = True
        should_reselect = False
      elif file_selected not in yaml_filenames:
        print('file not found in available list')
        should_reselect = True
      else:
        yaml_filenames.remove(file_selected)
        yaml_filenames_selected.append(file_selected)
        should_reselect = False
    
    if max_select is None:
      print('test results files selected ({}):'.format(len(yaml_filenames_selected)))
    else:
      print('test results files selected ({}/{}):'.format(len(yaml_filenames_selected), max_select))
    for yaml_filename in yaml_filenames_selected:
      print ('  - {}'.format(yaml_filename))
  
  return yaml_filenames_selected


def _load_test_results_files(test_results_dir, filenames):
  test_results_list = []
  for filename in filenames:
    with open(os.path.join(test_results_dir, filename), 'r') as yaml_file:
      test_results = yaml.load(yaml_file)
      test_results_list.append(test_results)
  return test_results_list


def _load_reference_dataset(dataset_dir):
  loader_dataset = TCatapultDatasetSim(abs_dirpath=dataset_dir, auto_init=False)
  loader_dataset.load_dataset()
  return loader_dataset


def _load_for_analysis(test_results_dir, dataset_dir, multiple_test_results=False):
  dataset = _load_reference_dataset(dataset_dir)
  
  test_results_filenames = []
  if multiple_test_results:
    test_results_filenames = _select_test_results_files(test_results_dir, max_select=None)
  else:
    test_results_filenames = _select_test_results_files(test_results_dir, max_select=1)
  test_results_list = _load_test_results_files(test_results_dir, test_results_filenames)
  
  if multiple_test_results:
    return test_results_list, dataset
  else:
    return test_results_list[0], dataset


def _evaluate_test_results(test_results_dir, dataset_dir):
  prefix = 'evaluate_test_results'
  prefix_info = prefix + ':'
  
  test_results, dataset = _load_for_analysis(test_results_dir, dataset_dir)
  
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


def _evaluate_online_learning(test_results_dir, dataset_dir):
  prefix = 'evaluate_online_learning'
  prefix_info = prefix + ':'
  
  test_results, dataset = _load_for_analysis(test_results_dir, dataset_dir)
  
  # print test samples
  print('test results: {}'.format(len(test_results)))
  print('')
  
  # print test result entry structure
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
  
  # add customized plotting keys
  list_seqs = []
  list_seqkeys = []
  append_seqkey = None
  while not append_seqkey == '':
    append_seqkey_input = input('{} add key (empty to end): '.format(prefix_info)).strip().lower()
    append_seqkey = str(append_seqkey_input)
    if len(append_seqkey) > 0:
      list_seqkeys.append(append_seqkey)
      list_seqs.append([])
  
  # construct episodic plotting data
  seq_episode = []
  seq_rewards = []
  for i in range(len(test_results)):
    entry = test_results[i]

    # episode sequence
    cur_episode = int(i)
    seq_episode.append(cur_episode)

    # rewards sequence
    cur_rewards = float(- abs(entry['desired_loc_land'] - entry['loc_land']))
    seq_rewards.append(cur_rewards)
    
    # other sequences
    for i_seq in range(len(list_seqkeys)):
      cur_value = float(entry[list_seqkeys[i_seq]])
      list_seqs[i_seq].append(cur_value)
  
  # figure configuraiton
  n_subplots = 1 + len(list_seqkeys)
  plt.figure()

  # plot: episode - rewards
  plt.subplot(n_subplots * 100 + 10 + 1)
  plt.xlabel('episode')
  plt.ylabel('rewards')
  plt.plot(seq_episode, seq_rewards, 'b-')
  
  # plot: others
  for i_seq in range(len(list_seqkeys)):
    plt.subplot(n_subplots * 100 + 10 + i_seq + 2)
    plt.xlabel('episode')
    plt.ylabel(list_seqkeys[i_seq])
    plt.plot(seq_episode, list_seqs[i_seq], 'b-')
  
  # show plot
  plt.show()


def _evaluate_online_learning_multitest(test_results_dir, dataset_dir):
  prefix = 'evaluate_online_learning_multitest'
  prefix_info = prefix + ':'
  
  test_results_list, dataset = _load_for_analysis(test_results_dir, dataset_dir, multiple_test_results=True)
  
  # Check loaded test results
  if len(test_results_list) < 2:
    print('{} please select at least two test results files'.format(prefix_info))
    return
  for test_results in test_results_list:
    if len(test_results) < 2:
      print('{} there must be at least two episodes at each online learning process'.format(prefix_info))
      return
    if len(test_results) != len(test_results_list[0]):
      print('{} all online learning processes must have the same episodes'.format(prefix_info))
      return
    if test_results[0]['approach'] != test_results_list[0][0]['approach']:
      print('{} all online learning processes must use the same approach'.format(prefix_info))
      return
  
  # print test samples
  print('test results: {}'.format(len(test_results_list[0])))
  print('')
  
  # print test result entry structure
  print('test result entry structure:')
  entry_items = []
  max_entry_item_len = 0
  for o in test_results_list[0][0]:
    entry_items.append(o)
    if len(o) > max_entry_item_len:
      max_entry_item_len = len(o)
  entry_items.sort()
  print('  - approach: {}'.format(test_results_list[0][0]['approach']))
  for i in range(len(entry_items)):
    o = entry_items[i]
    if o != 'approach':
      print(('  - {0: <' + str(max_entry_item_len + 1) + '} {1}').format(o + ':', type(test_results_list[0][0][o])))
  print('')
  
  # add customized plotting keys
  list_seqkeys = []
  append_seqkey = None
  while not append_seqkey == '':
    append_seqkey_input = input('{} add key (empty to end): '.format(prefix_info)).strip().lower()
    append_seqkey = str(append_seqkey_input)
    if len(append_seqkey) > 0:
      list_seqkeys.append(append_seqkey)
  
  # construct episodic data
  seq_episode_mtlist = []
  seq_rewards_mtlist = []
  list_seqs_mtlist = []
  for test_results in test_results_list:
    seq_episode = []
    seq_rewards = []
    list_seqs = []
    for seqkey in list_seqkeys:
      list_seqs.append([])
    for i in range(len(test_results)):
      entry = test_results[i]

      # episode sequence
      cur_episode = int(i)
      seq_episode.append(cur_episode)

      # rewards sequence
      cur_rewards = float(- abs(entry['desired_loc_land'] - entry['loc_land']))
      seq_rewards.append(cur_rewards)
      
      # other sequences
      for i_seq in range(len(list_seqkeys)):
        cur_value = float(entry[list_seqkeys[i_seq]])
        list_seqs[i_seq].append(cur_value)
    
    seq_episode_mtlist.append(seq_episode)
    seq_rewards_mtlist.append(seq_rewards)
    list_seqs_mtlist.append(list_seqs)
  
  # construct plotting data
  mt_count = len(seq_episode_mtlist)
  seq_episode = seq_episode_mtlist[0]
  seq_rewards_mean = []
  seq_rewards_center = []
  seq_rewards_err = []
  list_seqs_mean = [[] for j in range(len(list_seqs_mtlist[0]))]
  list_seqs_center = [[] for j in range(len(list_seqs_mtlist[0]))]
  list_seqs_err = [[] for j in range(len(list_seqs_mtlist[0]))]  
  for i in range(len(seq_episode)):
    reward_acc = 0
    reward_min = seq_rewards_mtlist[0][0]
    reward_max = seq_rewards_mtlist[0][0]
    
    seqs_acc = [0                             for i_seq in range(len(list_seqs_mtlist[0]))]
    seqs_min = [list_seqs_mtlist[0][i_seq][0] for i_seq in range(len(list_seqs_mtlist[0]))]
    seqs_max = [list_seqs_mtlist[0][i_seq][0] for i_seq in range(len(list_seqs_mtlist[0]))]
    
    for i_mt in range(mt_count):
      reward_acc += seq_rewards_mtlist[i_mt][i]
      if seq_rewards_mtlist[i_mt][i] < reward_min: reward_min = seq_rewards_mtlist[i_mt][i]
      if seq_rewards_mtlist[i_mt][i] > reward_max: reward_max = seq_rewards_mtlist[i_mt][i]
      
      for i_seq in range(len(list_seqs_mtlist[0])):
        seqs_acc[i_seq] += list_seqs_mtlist[i_mt][i_seq][i]
        if list_seqs_mtlist[i_mt][i_seq][i] < seqs_min[i_seq]: seqs_min[i_seq] = list_seqs_mtlist[i_mt][i_seq][i]
        if list_seqs_mtlist[i_mt][i_seq][i] > seqs_max[i_seq]: seqs_max[i_seq] = list_seqs_mtlist[i_mt][i_seq][i]
    
    seq_rewards_mean.append(reward_acc / mt_count)
    seq_rewards_center.append((reward_max + reward_min) / 2)
    seq_rewards_err.append((reward_max - reward_min) / 2)
    
    for i_seq in range(len(list_seqs_mtlist[0])):
      list_seqs_mean[i_seq].append(seqs_acc[i_seq] / mt_count)
      list_seqs_center[i_seq].append((seqs_max[i_seq] + seqs_min[i_seq]) / 2)
      list_seqs_err[i_seq].append((seqs_max[i_seq] - seqs_min[i_seq]) / 2)
  
  # figure configuraiton
  n_subplots = 1 + len(list_seqkeys)
  plt.figure()
  
  # plot: episode - rewards
  plt.subplot(n_subplots * 100 + 10 + 1)
  plt.xlabel('episode')
  plt.ylabel('rewards')
  plt.plot(seq_episode, seq_rewards_mean, 'b-')
  plt.errorbar(seq_episode, seq_rewards_center, seq_rewards_err, color='b', linestyle='None')
  
  # plot: others
  for i_seq in range(len(list_seqkeys)):
    plt.subplot(n_subplots * 100 + 10 + i_seq + 2)
    plt.xlabel('episode')
    plt.ylabel(list_seqkeys[i_seq])
    plt.plot(seq_episode, list_seqs_mean[i_seq], 'b-')
    plt.errorbar(seq_episode, list_seqs_center[i_seq], list_seqs_err[i_seq], color='b', linestyle='None')
  
  # show plot
  plt.show()


def _quit(test_results_dir, dataset_dir):
  quit()


def _getOperations():
  operations_list = [
    {
      'code': 'q',
      'desc': 'quit',
      'func': _quit
    },
    {
      'code': '1',
      'desc': 'general test result evaluation',
      'func': _evaluate_test_results
    },
    {
      'code': '2',
      'desc': 'online learning test result evaluation',
      'func': _evaluate_online_learning
    },
    {
      'code': '3',
      'desc': 'online learning test result evaluation (multitest)',
      'func': _evaluate_online_learning_multitest
    }
  ]
  
  return operations_list


def _select_operation(test_results_dir, dataset_dir):
  operations_list = _getOperations()
  assert(len(operations_list) > 0)
  
  print('operations:')
  for i in range(len(operations_list)):
    print('  - ({}) {}'.format(operations_list[i]['code'], operations_list[i]['desc']))
  operation_code_input = input('select operation (Q)> ').strip().lower()
  operation_code = str(operation_code_input)
  print('')
  
  operation_index = 0 # default operation index
  for i in range(len(operations_list)):
    if operation_code == operations_list[i]['code']:
      operation_index = i
  op_func = operations_list[operation_index]['func']
  op_func(test_results_dir, dataset_dir)


if __name__ == '__main__':
  dataset_dir = None
  test_results_dir = None
  
  if len(sys.argv) == 3:
    dataset_dir = os.path.abspath(sys.argv[1])
    test_results_dir = os.path.abspath(sys.argv[2])
  else:
    print('usage: ./plot_sim_002_linear_thetaT_d.py <dataset_dir> <test_results_dir>')
    quit()
  
  while True:
    _select_operation(test_results_dir, dataset_dir)


