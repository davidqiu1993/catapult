#!/usr/bin/python

"""
Catapult consistency test dataset plotting.
"""

__author__    = 'David Qiu'
__email__     = 'david@davidqiu.com'
__website__   = 'www.davidqiu.com'
__copyright__ = 'Copyright (C) 2017, David Qiu. All rights reserved.'


import sys
import yaml
import matplotlib.pyplot as plt

import pdb



if __name__ == '__main__':
  if len(sys.argv) != 2:
    print('usage: plot_consistency.py datafile')
    quit()
  
  filepath = sys.argv[1]
  
  prefix = 'catapult/plot_consistency'
  prefix_info = prefix + ':'
  
  with open(filepath, 'r') as yaml_file:
    dataset = yaml.load(yaml_file)
    
    # Dataset check
    assert(len(dataset) > 1)
    check_action = None
    for entry in dataset:
      if check_action is None:
        check_action = entry['action']
      else:
        for k in check_action:
          if '_actual' not in k:
            assert(check_action[k] == entry['action'][k])
    n_samples = len(dataset)
    print('{} samples = {}'.format(prefix_info, n_samples))
    for k in check_action:
      if '_actual' not in k:
        print('{} {} = {}'.format(prefix_info, k, check_action[k]))
    
    # Configurations
    LOC_MIN = 0
    LOC_MAX = 2000
    LOC_LAND_ACCURACY = 10
    LOC_STOP_ACCURACY = 10
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
    
    # Landing locations distribution
    plt.subplot(312)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.50, wspace=0.35)
    loc_land_plot_min = float(LOC_MIN) - float(LOC_LAND_ACCURACY) / 2.
    loc_land_plot_max = float(LOC_MAX) + float(LOC_LAND_ACCURACY) / 2.
    loc_land_plot_steps = int((float(LOC_MAX) - float(LOC_MIN)) / float(LOC_LAND_ACCURACY) + 2.)
    plt.title('Landing Locations Distribution')
    plt.axis([loc_land_plot_min, loc_land_plot_max, 0, n_samples + 1])
    plt.xlabel('Location')
    plt.ylabel('Frequency')
    loc_land_distribution = [entry['result']['loc_land'] for entry in dataset]
    plt.hist(loc_land_distribution, bins=[(loc_land_plot_min + i*LOC_LAND_ACCURACY) for i in range(loc_land_plot_steps)])
    
    # Stopping locations distribution
    plt.subplot(313)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.50, wspace=0.35)
    loc_stop_plot_min = float(LOC_MIN) - float(LOC_STOP_ACCURACY) / 2.
    loc_stop_plot_max = float(LOC_MAX) + float(LOC_STOP_ACCURACY) / 2.
    loc_stop_plot_steps = int((float(LOC_MAX) - float(LOC_MIN)) / float(LOC_STOP_ACCURACY) + 2.)
    plt.title('Stopping Locations Distribution')
    plt.axis([loc_stop_plot_min, loc_stop_plot_max, 0, n_samples + 1])
    plt.xlabel('Location')
    plt.ylabel('Frequency')
    loc_stop_distribution = [entry['result']['loc_stop'] for entry in dataset]
    plt.hist(loc_stop_distribution, bins=[(loc_stop_plot_min + i*LOC_STOP_ACCURACY) for i in range(loc_stop_plot_steps)])
    
    # Display
    plt.show()


