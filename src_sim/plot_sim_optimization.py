#!/usr/bin/python

"""
Simulation catapult optimization test dataset plotting.
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

POS_INIT_MIN    = - 0.10 * math.pi
POS_INIT_MAX    = + 0.50 * math.pi
POS_TARGET_MIN  = - 0.10 * math.pi
POS_TARGET_MAX  = + 0.75 * math.pi
DURATION_MIN    = - 0.1
DURATION_MAX    = + 0.6
LOC_LAND_MIN    = - 1.0
LOC_LAND_MAX    = + 125.0

UPPER_BOUND_FACTOR = 1.10


def _run_throw_farther(dataset, prefix_info):
  n_samples = len(dataset)
  plt.figure(num=1, figsize=(6, 8), dpi=120, facecolor='w', edgecolor='k')
  
  # Landing locations
  plt.subplot(411)
  plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.50, wspace=0.35)
  plt.title('Landing Locations Maximization')
  plt.axis([-1, n_samples, LOC_LAND_MIN, LOC_LAND_MAX * UPPER_BOUND_FACTOR])
  plt.xlabel('Landing Locations (episode, loc_land)')
  loc_land_sequence = [entry['result']['loc_land'] for entry in dataset]
  plt.plot([(i) for i in range(n_samples)], loc_land_sequence, 'b')
  
  # Initial positions
  plt.subplot(412)
  plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.50, wspace=0.35)
  plt.axis([-1, n_samples, POS_INIT_MIN, POS_INIT_MAX * UPPER_BOUND_FACTOR])
  plt.xlabel('Initial Positions (episode, pos_init)')
  pos_init_sequence = [entry['action']['pos_init'] for entry in dataset]
  plt.plot([(i) for i in range(n_samples)], pos_init_sequence, 'r')
  
  # Target positions
  plt.subplot(413)
  plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.50, wspace=0.35)
  plt.axis([-1, n_samples, POS_TARGET_MIN, POS_TARGET_MAX * UPPER_BOUND_FACTOR])
  plt.xlabel('Target Positions (episode, pos_target)')
  pos_target_sequence = [entry['action']['pos_target'] for entry in dataset]
  plt.plot([(i) for i in range(n_samples)], pos_target_sequence, 'r')
  
  # Durations
  plt.subplot(414)
  plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.50, wspace=0.35)
  plt.axis([-1, n_samples, DURATION_MIN, DURATION_MAX * UPPER_BOUND_FACTOR])
  plt.xlabel('Motion Durations (episode, duration)')
  duration_sequence = [entry['action']['duration'] for entry in dataset]
  plt.plot([(i) for i in range(n_samples)], duration_sequence, 'r')
  
  # Display
  plt.show()


def getOptions():
  option_dict = {
    'loc_land': _run_throw_farther
  }
  return option_dict


if __name__ == '__main__':
  if len(sys.argv) != 3 and len(sys.argv) != 2:
    print 'usage: plot_sim_optimization.py <datafile> <option>'
    quit()
  
  option = 'loc_land'
  if len(sys.argv) == 3:
    option = sys.argv[2]
  option_dict = getOptions()
  assert(option in option_dict)
  option_func = option_dict[option]
  
  filepath = sys.argv[1]
  
  prefix = 'catapult/plot_sim_optimization'
  prefix_info = prefix + ':'
  
  # Load dataset
  dataset = []
  with open(filepath, 'r') as yaml_file:
    dataset = yaml.load(yaml_file)
  
  # Dataset check
  n_samples = len(dataset)
  assert(n_samples > 1)
  print prefix_info, 'episodes =', n_samples
  
  # Execute option
  option_func(dataset, prefix_info)


