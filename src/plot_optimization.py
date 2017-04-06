#!/usr/bin/python

"""
Catapult optimization test dataset plotting.
"""

__author__    = 'David Qiu'
__email__     = 'david@davidqiu.com'
__website__   = 'www.davidqiu.com'
__copyright__ = 'Copyright (C) 2017, David Qiu. All rights reserved.'


from catapult import *

import sys
import yaml
import matplotlib.pyplot as plt

import pdb

POS_INIT_MIN = 0
POS_INIT_MAX = 420
POS_TARGET_MIN = 0
POS_TARGET_MAX = 840
DURATION_MIN = 0.0
DURATION_MAX = 0.6
LOC_MIN = 0
LOC_MAX = 2000


def _run_default(dataset, prefix_info):
  n_samples = len(dataset)
  plt.figure(num=1, figsize=(6, 8), dpi=120, facecolor='w', edgecolor='k')
  
  # Stopping faces distribution
  plt.subplot(311)
  plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.50, wspace=0.35)
  plt.title('Stopping Faces Distribution')
  plt.axis([0, 6, 0, n_samples + 1])
  plt.xlabel('Face')
  plt.ylabel('Frequency')
  face_stop_distribution = [(entry['result']['face_stop']) for entry in dataset]
  face_stop_distribution_plot = []
  for face in face_stop_distribution:
    face_plot = {'1': 1, '2': 2, '3': 3, '4': 4, 'side': 5}[face]
    face_stop_distribution_plot.append(face_plot)
  plt.hist(face_stop_distribution_plot, bins=[(0.25 + i*0.5) for i in range(11)])
  
  # Landing locations optimization
  plt.subplot(312)
  plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.50, wspace=0.35)
  plt.title('Landing Locations Optimization')
  plt.axis([-1, n_samples, LOC_MIN, LOC_MAX * 1.01])
  plt.xlabel('Episode')
  plt.ylabel('Location')
  loc_land_optimization = [entry['result']['loc_land'] for entry in dataset]
  plt.plot([(i) for i in range(n_samples)], loc_land_optimization)
  
  # Stopping locations optimization
  plt.subplot(313)
  plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.50, wspace=0.35)
  plt.title('Stopping Locations Optimization')
  plt.axis([-1, n_samples, LOC_MIN, LOC_MAX * 1.01])
  plt.xlabel('Episode')
  plt.ylabel('Location')
  loc_stop_optimization = [entry['result']['loc_stop'] for entry in dataset]
  plt.plot([(i) for i in range(n_samples)], loc_stop_optimization)
  
  # Display
  plt.show()


def _run_throw_farther(dataset, prefix_info):
  n_samples = len(dataset)
  plt.figure(num=1, figsize=(6, 8), dpi=120, facecolor='w', edgecolor='k')
  
  # Landing locations
  plt.subplot(411)
  plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.50, wspace=0.35)
  plt.title('Landing Locations Maximization')
  plt.axis([-1, n_samples, LOC_MIN, LOC_MAX * 1.01])
  plt.xlabel('Landing Locations (episode, loc_land)')
  loc_land_sequence = [entry['result']['loc_land'] for entry in dataset]
  plt.plot([(i) for i in range(n_samples)], loc_land_sequence, 'b')
  
  # Initial positions
  plt.subplot(412)
  plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.50, wspace=0.35)
  plt.axis([-1, n_samples, POS_INIT_MIN, POS_INIT_MAX * 1.01])
  plt.xlabel('Initial Positions (episode, pos_init)')
  pos_init_sequence = [entry['action']['pos_init'] for entry in dataset]
  pos_init_actual_sequence = [entry['action']['pos_init_actual'] for entry in dataset]
  plt.plot([(i) for i in range(n_samples)], pos_init_sequence, 'r')
  plt.plot([(i) for i in range(n_samples)], pos_init_actual_sequence, 'b')
  
  # Target positions
  plt.subplot(413)
  plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.50, wspace=0.35)
  plt.axis([-1, n_samples, POS_TARGET_MIN, POS_TARGET_MAX * 1.01])
  plt.xlabel('Target Positions (episode, pos_target)')
  pos_target_sequence = [entry['action']['pos_target'] for entry in dataset]
  pos_target_actual_sequence = [entry['action']['pos_target_actual'] for entry in dataset]
  plt.plot([(i) for i in range(n_samples)], pos_target_sequence, 'r')
  plt.plot([(i) for i in range(n_samples)], pos_target_actual_sequence, 'b')
  
  # Durations
  plt.subplot(414)
  plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.50, wspace=0.35)
  plt.axis([-1, n_samples, DURATION_MIN, DURATION_MAX * 1.01])
  plt.xlabel('Motion Durations (episode, duration)')
  duration_sequence = [entry['action']['duration'] for entry in dataset]
  plt.plot([(i) for i in range(n_samples)], duration_sequence, 'r')
  
  # Display
  plt.show()


def getOptions():
  option_dict = {
    'default': _run_default,
    'throw_farther': _run_throw_farther
  }
  return option_dict


if __name__ == '__main__':
  if len(sys.argv) != 3 and len(sys.argv) != 2:
    print 'usage: plot_optimization.py <datafile> <option>'
    quit()
  
  option = 'default'
  if len(sys.argv) == 3:
    option = sys.argv[2]
  option_dict = getOptions()
  assert(option in option_dict)
  option_func = option_dict[option]
  
  filepath = sys.argv[1]
  
  prefix = 'catapult/plot_optimization'
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


