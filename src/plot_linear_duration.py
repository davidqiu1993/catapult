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
    print 'usage: plot_consistency.py datafile'
    quit()
  
  filepath = sys.argv[1]
  
  prefix = 'catapult/plot_linear_duration'
  prefix_info = prefix + ':'
  
  with open(filepath, 'r') as yaml_file:
    dataset = yaml.load(yaml_file)
    
    # Dataset check
    check_face_init = None
    check_pos_init = None
    check_pos_target = None
    for entry in dataset:
      if check_pos_init is None:
        check_face_init = entry['action']['face_init']
        check_pos_init = entry['action']['pos_init']
        check_pos_target = entry['action']['pos_target']
      else:
        assert(entry['action']['face_init'] == check_face_init)
        assert(entry['action']['pos_init'] == check_pos_init)
        assert(entry['action']['pos_target'] == check_pos_target)
    n_samples = len(dataset)
    print prefix_info, 'samples =', n_samples
    print prefix_info, 'face_init =', check_face_init
    print prefix_info, 'pos_init =', check_pos_init
    print prefix_info, 'pos_target =', check_pos_target
    
    # Configurations
    plt.figure(num=1, figsize=(6, 8), dpi=120, facecolor='w', edgecolor='k')
    
    # Landing locations
    plt.subplot(211)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.50, wspace=0.35)
    plt.title('Landing Locations')
    plt.xlabel('t')
    plt.ylabel('location')
    for entry in dataset:
      plt.scatter(entry['action']['duration'], entry['result']['loc_land'])
    
    # Stopping locations
    plt.subplot(212)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.50, wspace=0.35)
    plt.title('Stopping Locations')
    plt.xlabel('t')
    plt.ylabel('location')
    for entry in dataset:
      plt.scatter(entry['action']['duration'], entry['result']['loc_stop'])
    
    # Display
    plt.show()


