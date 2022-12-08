# This script is a modified version of the blog below
# https://www.section.io/engineering-education/bodypix-for-body-segmentation/
# How to fix the saved_model not found error
# pip install --upgrade tensorflow-estimator==2.3.0


# Import libraries
import os
import cv2
from tf_bodypix.api import load_model, download_model, BodyPixModelPaths
import numpy as np
import matplotlib.pyplot as plt
from utils import extract_partwise_feature, get_body_part_features_stats
import pickle


# with open('c0.txt', 'rb') as f:
#     a = pickle.load(f)



# Source image
# image = cv2.imread("img_1000.jpg")

# Load body-pix model
# List of models that worked (actually all the models work, something was stopping the model to make predictions earlier)
# MOBILENET_FLOAT_50_STRIDE_16, RESNET50_FLOAT_STRIDE_16
# bp_model = load_model(download_model(BodyPixModelPaths.RESNET50_FLOAT_STRIDE_16))

train_root = 'imgs/train'
class_folders = os.listdir(train_root)
for this_class_dir in class_folders:
    this_class_path = os.path.join(train_root, this_class_dir)
    list_of_images = os.listdir(this_class_path)
    this_stat = get_body_part_features_stats(list_of_images, root=this_class_path, class_name=this_class_dir)
    print(f'For class {this_class_dir} obtained feature stats are: {this_stat}')
