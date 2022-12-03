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
            # (image*255).astype(np.uint8)
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


    def get_PCA(self, X_list, n_components=2):
        pca_list = []
        xpca_list = []
        enumerator = X_list if self.tqdm is None else self.tqdm(X_list, unit='images', desc=f'Doing PCA({n_components})')

        for X in enumerator:
            pca = PCA(n_components=n_components, svd_solver="randomized", whiten=True).fit(X)
            X_pca = pca.transform(X)
            pca_list.append(pca)
            xpca_list.append(X_pca)
        return pca_list, xpca_list

    def get_tsne(self, X_list, n_components=2, perplexity=50):
        xtsne_list = []
        enumerator = X_list if self.tqdm is None else self.tqdm(X_list, unit='images', desc=f'Doing tSNE({n_components})')

        for X in enumerator:
            tsne = TSNE(n_components=n_components, random_state=self.config.SEED, 
                perplexity=perplexity, init='pca', learning_rate='auto')
            X_tsne = tsne.fit_transform(X)
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

    def load_feature_vectors(self, output_base, filenames, labels):
        enumerator = enumerate(filenames)
        if self.tqdm is not None:
            enumerator = self.tqdm(enumerator, unit='images', desc=f'Loading feature vectors', total=len(filenames))
        pixel_features = []
        hog_features = []
        cnn_features = []
        canny_features = []
        pose_features = []
        body_parts_features = []
        for i, filename in enumerator:
            cur_label = labels[i]
            cur_folder = f'{output_base}/c{cur_label}'
            [cur_pixels, cur_hogs, cur_cnn, cur_canny, cur_pose, cur_body_parts] = torch.load(f'{cur_folder}/{filename}.pt')
            pixel_features.append(cur_pixels)
            hog_features.append(cur_hogs)
            cnn_features.append(cur_cnn)
            canny_features.append(cur_canny)
            pose_features.append(cur_pose)
            body_parts_features.append(cur_body_parts)
        return [np.array(pixel_features), 
                np.array(hog_features),
                np.array(cnn_features),
                np.array(canny_features),
                np.array(pose_features),
                np.array(body_parts_features)]

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
            cur_image_map, cur_label, cur_filename = samples
            row = [cur_filename, cur_label]
            for cur_type in image_types:
                row.append(cur_image_map[cur_type])
            stack.append(row)
            if pbar is not None:
              pbar.update(1)
        return stack

    def load_data(self, image_types, shuffle, sample_type, labels=None, count_per_label=None, image_transformers=None, include_feature_vectors=False):
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
                     enums.DataColumn.LABEL.value, 
                     *[cur_type.name.lower() for cur_type in image_types]]
        df = pd.DataFrame(stack, columns=col_names)

        if pbar is not None:
            pbar.close()

        if include_feature_vectors:
            vectors = self.load_feature_vectors(
                self.config.FEATURE_VECTORS_FOLDER, 
                df[enums.DataColumn.FILENAME.value],
                df[enums.DataColumn.LABEL.value])
            [pixel_features, hog_features, cnn_features, canny_features, pose_features] = vectors
            df[enums.DataColumn.PIXEL_VECTOR.value] = pixel_features.tolist()
            df[enums.DataColumn.HOG_VECTOR.value] = hog_features.tolist()
            df[enums.DataColumn.CNN_VECTOR.value] = cnn_features.tolist()
            df[enums.DataColumn.CANNY_VECTOR.value] = canny_features.tolist()
            df[enums.DataColumn.POSE_VECTOR.value] = pose_features.tolist()
        return df


