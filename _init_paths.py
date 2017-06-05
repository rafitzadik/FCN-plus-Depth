# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 22:26:12 2017

@author: rafi
"""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
caffe_path = '/home/rafi/nv-caffe/caffe/python/'
add_path(caffe_path)

# Add lib to PYTHONPATH
lib_path = '/home/rafi/nv-caffe/caffe/build/lib/'
add_path(lib_path)