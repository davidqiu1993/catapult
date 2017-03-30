#!/usr/bin/python

"""
Catapult linear motion control plotting.
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
    print 'usage: plot_001_linear.py datafile'
    quit()
  
  filepath = sys.argv[1]
  
  with open(filepath, 'r') as yaml_file:
    dataset = yaml.load(yaml_file)
    
    # Face distributions
    face_distributions = [(entry['result']['face_stop']) for entry in dataset]
    face_distributions_plot = []
    for face in face_distributions:
      face_plot = {'1': 1, '2': 2, '3': 3, '4': 4, 'side': 5}[face]
      face_distributions_plot.append(face_plot)
    plt.hist(face_distributions_plot, bins=[(0.25 + i*0.5) for i in range(11)])
    
    # Show
    plt.show()


