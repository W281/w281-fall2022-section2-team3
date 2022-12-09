import numpy as np
import cv2 as cv
from skimage.feature import hog
import models
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision import transforms
from PIL import Image
import torch
from pathlib import Path
import enums
import transformers
import customdataset
from torch.utils.data import DataLoader
import pandas as pd
from tf_bodypix.api import load_model, download_model, BodyPixModelPaths
import math
import copy


class FeatureExtractor:
    def __init__(self, config, face_config, pose_config, tqdm):
        self.config = config
        self.tqdm = tqdm
        self.face_config = face_config
        self.pose_config = pose_config

    def get_pixel_features(self, images):
        enumerator = images if self.tqdm is None else self.tqdm(images, unit='images', desc='Building Pixel Features')
        out_feat = []
        for img in enumerator:
            out_feat.append(img.flatten()[np.newaxis, :])
        return np.vstack(out_feat)

    def get_hog_features(self, images):
        out_feat = []
        hogs = []
        enumerator = images if self.tqdm is None else self.tqdm(images, unit='images', desc='Building HOG Features')
        for img in enumerator:
            fd, hog_image = hog(img, orientations=4, pixels_per_cell=(8, 8), 
                                cells_per_block=(2, 2), visualize=True)
            out_feat.append(fd[np.newaxis, :])
            hogs.append(hog_image)
        return np.vstack(out_feat), hogs

    def get_canny_features(self, images):
        out_feat = []
        cannies = []
        enumerator = images if self.tqdm is None else self.tqdm(images, unit='images', desc='Building Edge Features')
        for img in enumerator:
            if np.max(img) > 1:
                img = img.astype(np.uint8)
            else:
                img = (img*255).astype(np.uint8)

            edges = cv.Canny(img, 50, 200, 1)
            out_feat.append(edges.flatten()[np.newaxis, :])
            cannies.append(edges)
        return np.vstack(out_feat), cannies

    def get_cnn_features(self, images, device='cpu', model=None):
        out_feat = []
        model = models.ResNet152().to(device) if model is None else model
        enumerator = images if self.tqdm is None else self.tqdm(images, unit='images', desc='Building ResNet152 Features')
        for img in enumerator:
            # convert the grayscale to RGB images
            cur_rgb = np.stack([img, img, img], axis=2)
            if np.max(cur_rgb) > 1:
                cur_rgb = cur_rgb.astype(np.uint8)
            else:
                cur_rgb = (cur_rgb*255).astype(np.uint8)

            # preprocess the image to prepare it for input to CNN
            preprocess = transforms.Compose([
                  transforms.Resize(256),
                  transforms.CenterCrop(224),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            out_im = preprocess(Image.fromarray(cur_rgb)).to(device)
            cnn_image = model(out_im.unsqueeze(0).to('cpu')).squeeze().detach().numpy()
            out_feat.append(cnn_image)
        return np.stack(out_feat, axis=0)


    def get_PCA(self, X_list, n_components):
        pca_list = []
        xpca_list = []

        enumerator = enumerate(X_list) if self.tqdm is None else self.tqdm(enumerate(X_list), unit='images', desc=f'Doing PCA({n_components})', total=len(X_list))
        for i, X in enumerator:
            pca = PCA(n_components=n_components[i], svd_solver="randomized", whiten=True).fit(X)
            X_pca = pca.transform(X)
            pca_list.append(pca)
            xpca_list.append(X_pca)
        return pca_list, xpca_list

    #Arms' location and anle with BodyPix:
    def get_body_part_features(self, base_folder, labels, image_files):
        # Part of body that needs to be segmented out
        parts_of_interest = [
            'left_lower_arm_back',  # (255, 115, 75)
            'right_lower_arm_front',  # (255, 140, 56)
        ]
        out_feat = []
        bp_model = load_model(download_model(BodyPixModelPaths.RESNET50_FLOAT_STRIDE_16))
        enumerator = enumerate(images) if self.tqdm is None else self.tqdm(enumerate(image_files), unit='images', desc='Building Body-Part Features', total=len(image_files))
        for i, image_file in enumerator:
            img = cv.imread(f'{base_folder}/c{labels[i]}/{image_file}')
            prediction = bp_model.predict_single(img)
            mask = prediction.get_mask(threshold=0.5).numpy().astype(np.uint8)
            colored_mask = prediction.get_colored_part_mask(mask, part_names=parts_of_interest)
            body_parts = [
                ['left_lower_arm_back', (255, 115, 75)],
                ['right_lower_arm_front', (255, 140, 56)]
            ]
            channel_1 = colored_mask[:, :, 1]
            this_bp_features = list()
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
                else:
                    this_bp_features += [None, None, None]
            out_feat.append(this_bp_features)
        # Might have to use vstack or hstack depending on how features are indexed for later use
        return np.vstack(out_feat)

    def detect_eyes(self, base_folder, labels, image_files, count=None):
        # create a list to store number of eyes detected
        n_eyes_detected = []
        enumerator = enumerate(image_files) if self.tqdm is None else self.tqdm(enumerate(image_files), unit='images', desc=f'Getting eye counts', total=len(image_files))
        for i, image_file in enumerator:
            if count is not None and i > count:
                break
            # turn the image to gray image
            # print(f'Opening {base_folder}/c{labels[i]}/{image_file}')
            image = cv.imread(f'{base_folder}/c{labels[i]}/{image_file}')
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            eye_cascade = cv.CascadeClassifier(self.face_config.EYE_HAAR_CASCADES)
            eyes = eye_cascade.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors = 5)

            # save the number of eyes detcted
            n_eyes_detected.append(len(eyes))

        return n_eyes_detected

    def get_tsne(self, X_list, n_components=2, perplexity=50):
        xtsne_list = []
        enumerator = X_list if self.tqdm is None else self.tqdm(X_list, unit='images', desc=f'Doing tSNE({n_components})')

        for X in enumerator:
            tsne = TSNE(n_components=n_components, random_state=self.config.SEED, 
                perplexity=perplexity, init='pca', learning_rate='auto')
            X_tsne = tsne.fit_transform(np.array(X))
            xtsne_list.append(X_tsne)
        return xtsne_list

    def save_feature_vectors(self, output_base, filenames, labels, vectors):
        [pixel_features, hog_features, cnn_features, canny_features, pose_features, body_parts_features] = vectors
        enumerator = enumerate(filenames)
        if self.tqdm is not None:
            enumerator = self.tqdm(enumerator, unit='images', desc=f'Saving feature vectors', total=len(filenames))

        for i, filename in enumerator:
            cur_label = labels[i]
            cur_folder = f'{output_base}/c{cur_label}'
            Path(cur_folder).mkdir(parents=True, exist_ok=True)
            cur_features = [pixel_features[i], hog_features[i], cnn_features[i], canny_features[i], pose_features[i], body_parts_features[i]]
            torch.save(cur_features, f'{cur_folder}/{filename}.pt')

    def load_feature_vectors(self, output_base, filenames, labels, features=None):
        enumerator = enumerate(filenames)
        if features is None:
            features = set(list(enums.FeatureType))

        if self.tqdm is not None:
            enumerator = self.tqdm(enumerator, unit='images', desc=f'Loading feature vectors', total=len(filenames))
        pixel_features = []
        hog_features = []
        cnn_features = []
        canny_features = []
        pose_features = []
        keypoints_features = []
        body_parts_features = []
        right_hand_angles = []
        left_hand_angles = []
        for i, filename in enumerator:
            cur_label = labels[i]
            cur_folder = f'{output_base}/c{cur_label}'
            [cur_pixels, cur_hogs, cur_cnn, cur_canny, cur_pose, cur_body_parts] = torch.load(f'{cur_folder}/{filename}.pt')
            _, keypoints_feature, (rh_angle, lh_angle) = self._keypoint_offsets_for(cur_label, filename)
            right_hand_angles.append(rh_angle)
            left_hand_angles.append(lh_angle)
            keypoints_feature.append(rh_angle/360)
            keypoints_feature.append(lh_angle/360)
            keypoints_features.append(keypoints_feature)
            pixel_features.append(cur_pixels)
            hog_features.append(cur_hogs)
            cnn_features.append(cur_cnn)
            canny_features.append(cur_canny)
            pose_features.append(cur_pose)
            body_parts_features.append(cur_body_parts)


        list_of_features = []
        if enums.FeatureType.PIXEL in features:
            list_of_features.append(np.array(pixel_features))
        if enums.FeatureType.HOG in features:
            list_of_features.append(np.array(hog_features))
        if enums.FeatureType.CNN in features:
            list_of_features.append(np.array(cnn_features))
        if enums.FeatureType.CANNY in features:
            list_of_features.append(np.array(canny_features))
        if enums.FeatureType.POSE in features:
            list_of_features.append(np.array(pose_features))
        if enums.FeatureType.KEYPOINTS in features:
            list_of_features.append(np.array(keypoints_features))
        if enums.FeatureType.BODY_PARTS in features:
            list_of_features.append(np.array(body_parts_features))
        list_of_features.append(np.array(right_hand_angles))
        list_of_features.append(np.array(left_hand_angles))

        return list_of_features

    def load_data_for_label(self, label, image_types, shuffle, sample_type, count_per_label=None, image_transformers=None, pbar=None):
        default_t = {
            enums.ImageTypes.FACE: transforms.Compose([transforms.ToTensor(),
                                         transforms.Grayscale(),
                                         transformers.PyTorchImageToNPArray()]),
            enums.ImageTypes.ORIGINAL: transforms.Compose([transforms.ToTensor(),
                                       transforms.Grayscale(),
                                       transformers.PyTorchImageToNPArray()]),
            enums.ImageTypes.FACE_ANNOTATED: transforms.Compose([transforms.ToTensor(),
                                                   transformers.PyTorchImageToNPArray()]),
            enums.ImageTypes.POSE: transforms.Compose([transforms.ToTensor(),
                                        transforms.Grayscale(),
                                        transformers.PyTorchImageToNPArray()])
        }
        t = default_t if image_transformers is None else image_transformers

        dataset = customdataset.MainDataset(self.config, 
                                    self.face_config,
                                    self.pose_config,
                                    sample_type=sample_type,
                                    image_types=image_types,
                                    transformers=t,
                                    labels=[label])
        dataloader = DataLoader(dataset, num_workers=0, batch_size=1, 
                        shuffle=shuffle, collate_fn=dataset.get_image_from)
        stack = []
        for batch_idx, samples in enumerate(dataloader):
            if count_per_label is not None and batch_idx >= count_per_label:
                break
            cur_image_map, cur_label, cur_filename = copy.deepcopy(samples)
            row = [cur_filename, cur_label]
            if image_types is not None:
                for cur_type in image_types:
                    row.append(cur_image_map[cur_type])
            if pbar is not None:
              pbar.update(1)
            stack.append(row)
        return stack

    def load_data(self, image_types, shuffle, sample_type, labels=None, count_per_label=None, image_transformers=None, include_feature_vectors=False, ignore_large_features=False):
        labels = self.config.class_dict.keys() if labels is None else labels
        dataset = customdataset.MainDataset(self.config, 
                                            self.face_config,
                                            self.pose_config,
                                            sample_type=sample_type,
                                            labels=labels)
        total_sample_count = len(dataset)
        dataset = None
        if count_per_label is not None:
            total_sample_count = len(labels) * count_per_label

        stack = []

        if self.tqdm is not None:
            pbar = self.tqdm(total=total_sample_count, position=0, leave=True, unit='samples', desc=f'Loading {total_sample_count} samples')

        for label in labels:
            cur_stack = self.load_data_for_label(label, image_types, shuffle, sample_type, count_per_label=count_per_label, image_transformers=image_transformers, pbar=pbar)
            stack.extend(cur_stack)

        col_names = [enums.DataColumn.FILENAME.value, 
                     enums.DataColumn.LABEL.value]
        if image_types is not None:
            col_names.extend([cur_type.name.lower() for cur_type in image_types])

        df = pd.DataFrame(stack, columns=col_names)

        if pbar is not None:
            pbar.close()

        # if include_feature_vectors:
        #     vectors = self.load_feature_vectors(
        #         self.config.FEATURE_VECTORS_FOLDER, 
        #         df[enums.DataColumn.FILENAME.value],
        #         df[enums.DataColumn.LABEL.value])
        #     [pixel_features, hog_features, cnn_features, canny_features, pose_features, keypoints_features, body_parts_features] = vectors
        #     enums.DataColumn.POSE_KEYPOINTS_VECTOR.value
        #     if ignore_large_features:
        #         df[enums.DataColumn.HOG_VECTOR.value] = hog_features.tolist()
        #         df[enums.DataColumn.POSE_KEYPOINTS_VECTOR.value] = keypoints_features.tolist()
        #     else:
        #         df[enums.DataColumn.PIXEL_VECTOR.value] = pixel_features.tolist()
        #         df[enums.DataColumn.HOG_VECTOR.value] = hog_features.tolist()
        #         df[enums.DataColumn.CNN_VECTOR.value] = cnn_features.tolist()
        #         df[enums.DataColumn.CANNY_VECTOR.value] = canny_features.tolist()
        #         df[enums.DataColumn.POSE_VECTOR.value] = pose_features.tolist()
        #         df[enums.DataColumn.POSE_KEYPOINTS_VECTOR.value] = keypoints_features.tolist()
        #         df[enums.DataColumn.BODY_PARTS__VECTOR.value] = body_parts_features.tolist()
        return df

    def _get_hand_angles(self, keypoints):
        def _getAnglePy(shoulder_x, shoulder_y, elbow_x, elbow_y, wrist_x, wrist_y):
            ang = math.degrees(math.atan2(wrist_y-elbow_y, wrist_x-elbow_x) - math.atan2(shoulder_y-elbow_y, shoulder_x-elbow_x))
            ang = ang + 360 if ang < 0 else ang
            return ang

        right_shoulder = keypoints[self.pose_config.KEYPOINT_DICT['right_shoulder']]
        right_elbow = keypoints[self.pose_config.KEYPOINT_DICT['right_elbow']]
        right_wrist = keypoints[self.pose_config.KEYPOINT_DICT['right_wrist']]
        rh_ang = _getAnglePy(right_shoulder[0], right_shoulder[1], right_elbow[0], right_elbow[1], right_wrist[0], right_wrist[1])

        left_shoulder = keypoints[self.pose_config.KEYPOINT_DICT['left_shoulder']]
        left_elbow = keypoints[self.pose_config.KEYPOINT_DICT['left_elbow']]
        left_wrist = keypoints[self.pose_config.KEYPOINT_DICT['left_wrist']]
        lh_ang = _getAnglePy(left_shoulder[0], left_shoulder[1], left_elbow[0], left_elbow[1], left_wrist[0], left_wrist[1])
        return (rh_ang, lh_ang)

    def _keypoint_offsets_for(self, label, filename, verbose=False):
        poses_folder = f'{self.pose_config.FEATURES_FOLDER}/c{label}'
        keypoints_file_path = f'{poses_folder}/{filename.replace(".jpg", "_keypoints.pt")}'
        
        keypoints = torch.load(keypoints_file_path) # A numpy array with shape [1, 1, 17, 3] 
                                                    # representing the keypoint coordinates and scores
                                                    # returned from the MoveNet model.
        # Apply threshold filter and convert to a map with needed keypoints
        if not keypoints.shape[0] == 1:
            raise ValueError('{filename} has {keypoints.shape[0]} poses. Should have just one pose.')
        
        # Ignore the first two dimensions
        keypoints = keypoints[0][0]

        feature_vector = []
        (nose_x, nose_y, nose_score) = keypoints[self.pose_config.KEYPOINT_DICT['nose']]

        # Add the nose keypoint first. Offsets are zero for nose.
        final_keypoints = [[nose_x, nose_y, nose_score, 0, 0]]
        (rh_angle, lh_angle) = self._get_hand_angles(keypoints)

        keypoints[self.pose_config.KEYPOINT_DICT['nose']]
        # Iterate through remaining keypoints, skipping nose.
        for i in range(1, len(self.pose_config.KEYPOINT_DICT)):
            cur_keypoint = keypoints[i] # has 3 values - x, y , score
            new_keypoints = [cur_keypoint[0], cur_keypoint[1], cur_keypoint[2], cur_keypoint[0], cur_keypoint[1]]
            new_keypoints[3] = new_keypoints[3] - nose_x
            new_keypoints[4] = new_keypoints[4] - nose_y
            final_keypoints.append(new_keypoints)
            if i in self.pose_config.KEYPOINTS_INCLUDED_IN_FEATURES:
                feature_vector.extend([new_keypoints[3], new_keypoints[4]])
        if verbose:
            print(f'{filename} keypoints: {final_keypoints}')
        return final_keypoints, feature_vector, (rh_angle, lh_angle)
        
    def keypoints_relative_to_nose(self, sample_type=enums.SampleType.TRAIN_TEST_VALIDATION):
        dataset = customdataset.MainDataset(self.config, self.face_config, self.pose_config,
                                            sample_type=sample_type, should_load_images=False)

        dataloader = DataLoader(dataset, num_workers=0, batch_size=1,
                                shuffle=True, collate_fn=dataset.get_image_from)
        rows = []
        enumerator = enumerate(dataloader) if self.tqdm is None else self.tqdm(enumerate(dataloader), total=len(dataset))
        for i, sample in enumerator:
            (_, label, filename) = copy.deepcopy(sample)
            # Load the keypoints file
            keypoints, feature_vector, _ = self._keypoint_offsets_for(label, filename)
            row = [filename, label, feature_vector]
            row.extend([keypoint[0] for keypoint in keypoints])
            row.extend([keypoint[1] for keypoint in keypoints])
            row.extend([keypoint[2] for keypoint in keypoints])
            row.extend([keypoint[3] for keypoint in keypoints])
            row.extend([keypoint[4] for keypoint in keypoints])
            rows.append(row)
            
        col_names = [enums.DataColumn.FILENAME.value, enums.DataColumn.LABEL.value, 'pose_feature_vector']
        col_names.extend([f'{name}_x' for name in self.pose_config.SORTED_KEYPOINT_NAMES])
        col_names.extend([f'{name}_y' for name in self.pose_config.SORTED_KEYPOINT_NAMES])
        col_names.extend([f'{name}_score' for name in self.pose_config.SORTED_KEYPOINT_NAMES])
        col_names.extend([f'{name}_offset_x' for name in self.pose_config.SORTED_KEYPOINT_NAMES])
        col_names.extend([f'{name}_offset_y' for name in self.pose_config.SORTED_KEYPOINT_NAMES])
                                     
        df = pd.DataFrame(rows, columns=col_names)
        return df

    def _keypoints_for(self, label, filename):
        poses_folder = f'{self.pose_config.FEATURES_FOLDER}/c{label}'
        keypoints_file_path = f'{poses_folder}/{filename.replace(".jpg", "_keypoints.pt")}'
        
        keypoints = torch.load(keypoints_file_path) # A numpy array with shape [1, 1, 17, 3] 
                                                    # representing the keypoint coordinates and scores
                                                    # returned from the MoveNet model.
        # Apply threshold filter and convert to a map with needed keypoints
        if not keypoints.shape[0] == 1:
            raise ValueError('{filename} has {keypoints.shape[0]} poses. Should have just one pose.')
        
        # Ignore the first two dimensions
        keypoints = keypoints[0][0]

        # Add the nose keypoint first. Offsets are zero for nose.
        final_keypoints = []

        # Iterate through remaining keypoints, skipping nose.
        for i in range(0, len(self.pose_config.KEYPOINT_DICT)):
            cur_keypoint = keypoints[i] # has 3 values - x, y , score
            new_keypoints = [cur_keypoint[0], cur_keypoint[1], cur_keypoint[2]]
            final_keypoints.append(new_keypoints)
        return final_keypoints

    def load_keypoints(self, sample_type=enums.SampleType.TRAIN_TEST_VALIDATION, count=None):
        dataset = customdataset.MainDataset(self.config, self.face_config, self.pose_config,
                                            sample_type=sample_type, should_load_images=False)

        dataloader = DataLoader(dataset, num_workers=0, batch_size=1,
                                shuffle=True, collate_fn=dataset.get_image_from)
        rows = []
        enumerator = enumerate(dataloader) if self.tqdm is None else self.tqdm(enumerate(dataloader), total=len(dataset))
        for i, sample in enumerator:
            (_, label, filename) = copy.deepcopy(sample)
            if count is not None and i > count:
                break
            file_keypoints = self._keypoints_for(label, filename)
            row = [filename, label]
            row.extend([keypoint[0] for keypoint in file_keypoints])
            row.extend([keypoint[1] for keypoint in file_keypoints])
            row.extend([keypoint[2] for keypoint in file_keypoints])
            rows.append(row)

        col_names = [enums.DataColumn.FILENAME.value, enums.DataColumn.LABEL.value]
        col_names.extend([f'{name}_x' for name in self.pose_config.SORTED_KEYPOINT_NAMES])
        col_names.extend([f'{name}_y' for name in self.pose_config.SORTED_KEYPOINT_NAMES])
        col_names.extend([f'{name}_score' for name in self.pose_config.SORTED_KEYPOINT_NAMES])
                                     
        df = pd.DataFrame(rows, columns=col_names)
        return df