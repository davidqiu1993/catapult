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



if __name__ == '__main__':
  if len(sys.argv) != 2:
    print 'usage: plot_optimization.py datafile'
    quit()
  
  filepath = sys.argv[1]
  
  prefix = 'catapult/plot_optimization'
  prefix_info = prefix + ':'
  
  with open(filepath, 'r') as yaml_file:
    dataset = yaml.load(yaml_file)
    
    # Dataset check
    n_samples = len(dataset)
    assert(n_samples > 1)
    print prefix_info, 'episodes =', n_samples
    
    # Configurations
    LOC_MIN = 0
    LOC_MAX = 2000
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


