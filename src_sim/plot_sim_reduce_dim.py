#!/usr/bin/python

"""
Simulation catapult dataset plotting with dimensional complexity reduced.
"""

__author__    = 'David Qiu'
__email__     = 'david@davidqiu.com'
__website__   = 'www.davidqiu.com'
__copyright__ = 'Copyright (C) 2017, David Qiu. All rights reserved.'


import sys
import yaml
import matplotlib.pyplot as plt
import math

import pdb


def _run_plot(dataset, x_tag, y_tag, fixed_action_values, fixed_result_values, prefix_info):
  reduced_dataset = []
  for entry in dataset:
    should_select = True
    for tag in fixed_action_values:
      if fixed_action_values[tag] != entry['action'][tag]: should_select = False
    for tag in fixed_result_values:
      if fixed_result_values[tag] != entry['result'][tag]: should_select = False
    if should_select:
      reduced_dataset.append(entry)
  
  n_samples = len(dataset)
  n_selected_samples = len(reduced_dataset)
  print('{} selected samples = {}/{}'.format(prefix_info, n_selected_samples, n_samples))
  
  plt.figure(num=1, figsize=(8, 6), dpi=120, facecolor='w', edgecolor='k')
  plt.title('Correlation between {} and {}'.format(x_tag, y_tag))
  plt.xlabel(x_tag)
  plt.ylabel(y_tag)
  
  X = []
  Y = []
  for entry in reduced_dataset:
    X.append(entry['action'][x_tag])
    Y.append(entry['result'][y_tag])
  plt.scatter(X, Y)
  
  plt.show()


if __name__ == '__main__':
  if len(sys.argv) != 2:
    print('usage: plot_sim_reduce_dim.py <datafile>')
    quit()
  
  filepath = sys.argv[1]
  
  prefix = 'catapult/plot_sim_reduce_dim'
  prefix_info = prefix + ':'
  
  # Load dataset
  dataset = []
  with open(filepath, 'r') as yaml_file:
    dataset = yaml.load(yaml_file)
  
  # Dataset check
  n_samples = len(dataset)
  assert(n_samples > 1)
  motion = dataset[0]['motion']
  action_tags = [(tag) for tag in dataset[0]['action']]
  result_tags = [(tag) for tag in dataset[0]['result']]
  for entry in dataset:
    assert(motion == entry['motion'])
    for tag in entry['action']:
      assert(tag in action_tags)
    for tag in entry['result']:
      assert(tag in result_tags)
  print('{} motion = {}, samples = {}'.format(prefix_info, motion, n_samples))
  
  # Reduce dimensionality
  try:
    input = raw_input
  except NameError:
    pass
  
  x_tag = None
  print('{} Please indicate action tag for x-axis:'.format(prefix_info))
  for i in range(len(action_tags)):
    print('  ({}) {}'.format(i+1, action_tags[i]))
  x_tag_index_input = input('x_tag_index = ').strip().lower()
  x_tag_index = int(x_tag_index_input)
  x_tag = action_tags[x_tag_index - 1]
  print('{} x_tag = {}'.format(prefix_info, x_tag))
  print('')
  
  y_tag = None
  print('{} Please indicate result tag for y-axis:'.format(prefix_info))
  for i in range(len(result_tags)):
    print('  ({}) {}'.format(i+1, result_tags[i]))
  y_tag_index_input = input('y_tag_index = ').strip().lower()
  y_tag_index = int(y_tag_index_input)
  y_tag = result_tags[y_tag_index - 1]
  print('{} y_tag = {}'.format(prefix_info, y_tag))
  print('')
  
  if len(action_tags) > 1 or len(result_tags > 1):
    print('{} Please indicate fixed values:'.format(prefix_info))
  
  fixed_action_values = {}
  for tag in action_tags:
    if tag != x_tag:
      value_input = input('{} = '.format(tag))
      fixed_action_values[tag] = float(value_input)
  
  fixed_result_values = {}
  for tag in result_tags:
    if tag != y_tag:
      value_input = input('{} = '.format(tag))
      fixed_result_values[tag] = float(value_input)
  
  print('')
  
  # Execute plotting
  _run_plot(dataset, x_tag, y_tag, fixed_action_values, fixed_result_values, prefix_info)


