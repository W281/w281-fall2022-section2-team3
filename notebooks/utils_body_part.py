import os
import pickle
import cv2
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tf_bodypix.api import load_model, download_model, BodyPixModelPaths

def get_body_part_features_stats(image_files, root, class_name):
    # Feature availability accumulators
    feature_stats = {'total_images': len(image_files), 'no_arm': 0, 'left_arm': 0, 'right_arm': 0}
    # Part of body that needs to be segmented out
    parts_of_interest = [
        'left_lower_arm_back',  # (255, 115, 75)
        'right_lower_arm_front',  # (255, 140, 56)
    ]
    out_feat = []
    pickle_dict = dict()
    bp_model = load_model(download_model(BodyPixModelPaths.RESNET50_FLOAT_STRIDE_16))

    for this_image_file in image_files:
        this_image_path = os.path.join(root, this_image_file)
        this_image = cv2.imread(this_image_path)

        prediction = bp_model.predict_single(this_image)
        mask = prediction.get_mask(threshold=0.5).numpy().astype(np.uint8)
        colored_mask = prediction.get_colored_part_mask(mask, part_names=parts_of_interest)
        body_parts = [
            ['left_lower_arm_back', (255, 115, 75)],
            ['right_lower_arm_front', (255, 140, 56)]
        ]
        channel_1 = colored_mask[:, :, 1]
        this_bp_features = list()
        feature_flag = False
        for this_part in body_parts:
            channel_color_code = this_part[1][1]
            this_coord = np.argwhere(channel_1 == channel_color_code)
            # Check if this body part was detected
            if len(this_coord) > 0:
                x_coord, y_coord = this_coord[:, 1], -this_coord[:, 0]
                # Estimate center
                center_x, center_y = np.mean(x_coord), np.mean(y_coord)
                # Estimate orientation
                z = np.polyfit(x_coord, y_coord, 1)
                orientation_in_rad = np.arctan(z[0])
                this_bp_features += [center_x, center_y, orientation_in_rad]
                if this_part[0] == 'left_lower_arm_back':
                    feature_flag = True
                    feature_stats['left_arm'] += 1
                elif this_part[0] == 'right_lower_arm_front':
                    feature_flag = True
                    feature_stats['right_arm'] += 1
            else:
                this_bp_features += [None, None, None]
        if feature_flag is False:
            feature_stats['no_arm'] += 1
        pickle_dict[this_image_file] = this_bp_features
        out_feat.append(np.array(this_bp_features))
    # Might have to use vstack or hstack depending on how features are indexed for later use
    with open(class_name+'.txt', 'wb') as f:
        pickle.dump(pickle_dict, f)
    return feature_stats
