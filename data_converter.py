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
import old_utils as utils
import ast
from shutil import copyfile

## 6 categories below, three commands. Don't forget class_dict.csv!

 # python data_converter.py --mode train \
 # --input_dir  ~/Desktop/ADE20K_2016_07_26/images/training \
 # --input_match_exp "*.jpg" \
 # --label_dir ~/Desktop/ADE20K_2016_07_26/images/training \
 # --label_match_exp "*_seg.png" \
 # --output_dir ~/Desktop/ADE20K_2016_07_26/ade20k_floor_wall \
 # --replace_colors ~/Desktop/ADE20K_2016_07_26/replace-wall-floor.txt --min_colors=3

 # python data_converter.py --mode val \
 # --input_dir $datasets/ADE20K_2016_07_26/images/validation \
 # --input_match_exp "*.jpg" \
 # --label_dir $datasets/ADE20K_2016_07_26/images/validation \
 # --label_match_exp "*_seg.png" \
 # --output_dir $datasets/ade20k_sss \
 # --filter_categories $datasets/ADE20K_2016_07_26/indoor-categories.txt \
 # --replace_colors $datasets/ADE20K_2016_07_26/replace-all-colors.txt

 # python data_converter.py --mode test \
 # --input_dir $datasets/ADE20K_2016_07_26/images/validation \
 # --input_match_exp "*.jpg" \
 # --label_dir $datasets/ADE20K_2016_07_26/images/validation \
 # --label_match_exp "*_seg.png" \
 # --output_dir $datasets/ade20k_sss \
 # --filter_categories $datasets/ADE20K_2016_07_26/indoor-categories.txt \
 # --replace_colors $datasets/ADE20K_2016_07_26/replace-all-colors.txt

## 2 categories, just floor below, three commands. Don't forget class_dict.csv!

 # python data_converter.py --mode train \
 # --input_dir $datasets/ADE20K_2016_07_26/images/training \
 # --input_match_exp "*.jpg" \
 # --label_dir $datasets/ADE20K_2016_07_26/images/training \
 # --label_match_exp "*_seg.png" \
 # --output_dir $datasets/ade20k_floors_sss \
 # --replace_colors $datasets/ADE20K_2016_07_26/just-floor.txt --min_colors=2

 # python data_converter.py --mode val \
 # --input_dir $datasets/ADE20K_2016_07_26/images/validation \
 # --input_match_exp "*.jpg" \
 # --label_dir $datasets/ADE20K_2016_07_26/images/validation \
 # --label_match_exp "*_seg.png" \
 # --output_dir $datasets/ade20k_floors_sss \
 # --replace_colors $datasets/ADE20K_2016_07_26/just-floor.txt --min_colors=2

 # python data_converter.py --mode test \
 # --input_dir $datasets/ADE20K_2016_07_26/images/validation \
 # --input_match_exp "*.jpg" \
 # --label_dir $datasets/ADE20K_2016_07_26/images/validation \
 # --label_match_exp "*_seg.png" \
 # --output_dir $datasets/ade20k_floors_sss \
 # --replace_colors $datasets/ADE20K_2016_07_26/just-floor.txt --min_colors=2

 # python data_converter.py --mode train --match_hue 1 --min_area 0.1  \
 # --input_dir $datasets/matterport/area_1/data/rgb \
 # --input_match_exp "*.png" \
 # --label_dir $datasets/matterport/area_1/data/semantic_pretty \
 # --label_match_exp "*.png" \
 # --output_dir $datasets/matterport-floor-walls/area_1 \
 # --replace_colors $datasets/matterport/replace-floor-wall-hsv.txt --min_colors=2

parser = argparse.ArgumentParser()

# required together:
parser.add_argument("--input_dir", required=True, help="Source Input Path")
parser.add_argument("--input_match_exp", required=False, help="Source Input expression to match files")
parser.add_argument("--output_dir", required=True, help="where to put output files")

parser.add_argument("--mode", required=True, choices=["train", "test", "val"])
parser.add_argument("--match_hue", type=utils.str2bool, nargs='?', const=True, default=False, help="Match hue only")

parser.add_argument("--label_dir", required=True, help="Label Input Path")
parser.add_argument("--label_match_exp", required=False, help="Label Input expression to match files")

parser.add_argument("--filter_categories", required=False, help="Path to file with valid categories")
parser.add_argument("--replace_colors", required=False, help="Path to file with GT color replacements. See replace-colors.txt")
parser.add_argument("--min_colors", type=int, required=False, default=3, help="minimum number of labels to be valid")
parser.add_argument("--min_area", type=float, default=0.0, help="Minimum area to count as a valid label")


a = parser.parse_args()

matches = []
replacements = []

def getColor(input): 
    if not input.startswith('['):
        input = '[' + input

    if not input.endswith(']'):
        input = input + ']'

    return ast.literal_eval(input)

def replaceColors(im, min_colors=2, hue_only=False, min_label_area=0.0):

    h,w = im.shape[:2]

    im = im[:,:,:3]

    if hue_only:
        image = Image.fromarray(np.uint8(im))
        hue = np.asarray(image.convert('HSV'))[:,:,0]
    else:
        red, green, blue = im[:,:,0], im[:,:,1], im[:,:,2]

    # return hue

    # print(hue[int(h-1),int(w/2)])
    # return hue

    default = None
    total_mask = np.zeros([h,w],dtype=np.uint8)

    num_elements = 1
    lastZeroCount = 0
    total_pixels = h * w

    for i in range(0, len(matches)):
        if matches[i] == "*":
            default = replacements[i]
        else:
            for j in range(0, len(matches[i])):
                color_to_replace = matches[i][j]

                if hue_only:
                    mask = (hue == color_to_replace[0])
                else:
                    mask = (red == color_to_replace[0]) & (green == color_to_replace[1])
                im[mask] = replacements[i] #codes for below
                total_mask[mask] = 255
                nzCount = cv2.countNonZero(total_mask)
                coverage = float(nzCount) / float(total_pixels)
                if nzCount > lastZeroCount and min_label_area == 0 or coverage >= min_label_area:
                    num_elements = num_elements + 1
                lastZeroCount = nzCount
    
    if num_elements < min_colors:
        # print("Got %d colors which is less than allowed %d" % (num_elements, min_colors))
        return None

    if not default is None:
        im[total_mask != 255] = default

    return im

def main():

    filtered_dirs = utils.getFilteredDirs(a)
    num_dirs = 1 if filtered_dirs is None else len(filtered_dirs)

    print("Got %d image directories, finding images..." % num_dirs)

    src_dir = os.path.join(a.output_dir, a.mode)
    if not os.path.exists(src_dir):
        os.makedirs(src_dir)

    src_paths = utils.get_image_paths(a.input_dir, a.input_match_exp, require_rgb=False, filtered_dirs=filtered_dirs)

    label_dir = os.path.join(a.output_dir, a.mode + "_labels")
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    label_paths = utils.get_image_paths(a.label_dir, a.label_match_exp, require_rgb=False, filtered_dirs=filtered_dirs)

    num_src = len(src_paths)
    num_labels = len(label_paths)

    if (num_src != num_labels):
        raise Exception('Found different number of source images than labels (%d vs %d)' % (num_src, num_labels))
    
    print("Processing %d images" % num_src)

    num_dest = 0

    if not a.replace_colors is None:
        if not os.path.isfile(a.replace_colors): 
            print("Error: replace_colors file %s does not exist" % a.replace_colors)
            return

        with open(a.replace_colors) as f:
            content = f.readlines()
            content = [x.strip() for x in content] 

        #/b/banquet_hall 38

        for line in content:
            line = re.sub(r'\s+', '', line) # Remove spaces
            data_search = re.search('(.+):(.+)//', line, re.IGNORECASE)
            if data_search:
                if data_search.group(1).startswith('*'):
                    to_replace = data_search.group(1)
                else:
                    to_replace = data_search.group(1).split('],[')
                    to_replace = [getColor(x) for x in to_replace]
                matches.append(to_replace)
                replace_with = data_search.group(2)
                replacements.append(ast.literal_eval(replace_with.strip()))

    for i in range(num_src):
        src_path = src_paths[i]
        src_name = os.path.basename(src_path)
        dst_path = os.path.join(src_dir, src_name)

        label_path = label_paths[i]
        label_name = os.path.basename(label_path)
        dst_label_path = os.path.join(label_dir, label_name)
        if not a.replace_colors is None:
            image = misc.imread(label_path)
            image = replaceColors(image, a.min_colors, a.match_hue, a.min_area)
            if image is None:
                # print("Skipping %s" % label_name)
                continue
            misc.imsave(dst_label_path, image)
        else:
            copyfile(label_path, dst_label_path)

        copyfile(src_path, dst_path)
        num_dest = num_dest + 1
        print("Processed %s and %s" % (src_name, label_name))

    print("DONE, processed %d images" % num_dest)

main()