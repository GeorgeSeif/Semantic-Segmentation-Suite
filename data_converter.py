from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time
from PIL import Image
from scipy import misc
import fnmatch
import cv2
import re
import utils
import ast

 # python data_converter.py \
 # --input_dir $datasets/ADE20K_2016_07_26/images/training \
 # --output_dir $datasets/ade20k_sss \
 # --filter_categories $datasets/ADE20K_2016_07_26/indoor-categories.txt 

parser = argparse.ArgumentParser()

# required together:
parser.add_argument("--input_dir", required=True, help="Source Input Path")
parser.add_argument("--input_match_exp", required=False, help="Source Input expression to match files")

parser.add_argument("--filter_categories", required=False, help="Path to file with valid categories")
parser.add_argument("--replace_colors", required=False, help="Path to file with GT color replacements. See replace-colors.txt")

# Place to output A/B images
parser.add_argument("--output_dir", required=True, help="where to put output files")

a = parser.parse_args()

def main():

    filtered_dirs = utils.getFilteredDirs(a)
    num_dirs = 1 if filtered_dirs is None else len(filtered_dirs)

    print("Got %d image directories, finding images..." % num_dirs)

    paths = utils.get_image_paths(a.input_dir, a.input_match_exp, require_rgb=False, filtered_dirs=filtered_dirs)
    
    print("Processing %d images" % len(paths))


    print("DONE")

main()