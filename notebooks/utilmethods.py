import torch
import random
import numpy as np
import os

def check_torch_mps_device():
  if torch.backends.mps.is_available():
      mps_device = torch.device("mps")
      x = torch.ones(1, device=mps_device)
      print (x)
  else:
      print ("MPS device not found.")

def make_torch_deterministic(seed):
  torch.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)
  torch.use_deterministic_algorithms(True)


def create_output_folders(config, mtcnn_config):
    def create_if_not_found(path):
        if not os.path.exists(path):
            os.mkdir(path)
    create_if_not_found(mtcnn_config.FEATURES_FOLDER_FULL)
    for label in config.class_dict:
        create_if_not_found(os.path.join(mtcnn_config.FEATURES_FOLDER_FULL, f'c{label}'))

