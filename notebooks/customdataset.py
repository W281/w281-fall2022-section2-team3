import os
import pandas as pd
import csv
import glob
from torch.utils.data import Dataset
from PIL import Image
import enums

class CustomDataset(Dataset):
    def get_image_from(self, sample):
        return sample[0]

#TODO: All functionality in OriginalDataset is also in MainDataset. So we can remove
#      OriginalDataset altogether after migrating existing code to MainDataset.
class OriginalDataset(CustomDataset):
    def __init__(self, config, transform=None, target_transform=None, label=None):
        if not os.path.exists(config.ANNOTATION_FILE):
            OriginalDataset.__create_annotation_file(config)
        self.img_labels = pd.read_csv(config.ANNOTATION_FILE)
        if label is not None:
            self.img_labels = self.img_labels[self.img_labels['class'] == label]
        self.config = config
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels.iloc[idx, 1]
        filename = self.img_labels.iloc[idx, 0]
        image_path = os.path.join(f'{self.config.TRAIN_DATA}/c{label}', filename)

        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, filename

    def __create_annotation_file(config):
        annotation_file = config.ANNOTATION_FILE
        base_folder = config.TRAIN_DATA
        with open(annotation_file, 'w', newline='') as csvfile:
            annotation_writer = csv.writer(csvfile, delimiter=',')
            annotation_writer.writerow(config.ANNOTATION_FILE_COLS)
            for cur_class in range(len(config.class_dict)):
                class_folder = f'{base_folder}/c{cur_class}'
                files = list(glob.glob(class_folder + '/*.jpg'))
                for file in files:
                    annotation_writer.writerow([os.path.basename(file), cur_class, 0, enums.SampleType.IGNORED])

class MainDataset(CustomDataset):
    def __init__(self, config, face_config, pose_config, 
                 sample_type=enums.SampleType.ALL, image_types=None,
                 transformers={}, labels=None, should_load_images=True):
        self.sample_type = sample_type
        self.config = config
        self.face_config = face_config
        self.pose_config = pose_config
        self.transformers = transformers
        self.image_types = image_types
        self.should_load_images = should_load_images
        self.labels = None if labels is None or len(labels) == 0 else labels
        self.__load_data()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        def load_image(base_path, filename, transformer):
            image_path = os.path.join(base_path, f'c{label}', filename)
            if self.should_load_images and os.path.exists(image_path):
                image = Image.open(image_path)
                if transformer:
                    image = transformer(image)
                return image
            else:
                return None

        images = {}
        label = self.data.iloc[idx]['class']
        if 'target' in self.transformers:
            label = self.transformers['target'](label)

        filename = self.data.iloc[idx]['filename']

        if self.image_types is not None:
            for image_type in self.image_types:
                cur_transformer = self.transformers.get(image_type)
                cur_image = None
                if image_type is enums.ImageTypes.ORIGINAL:
                    cur_image = load_image(self.config.TRAIN_DATA, filename, cur_transformer)
                elif image_type is enums.ImageTypes.FACE_ANNOTATED:
                    cur_filename = filename.replace('.jpg', '_annotated.png')
                    cur_image = load_image(self.face_config.FEATURES_FOLDER, cur_filename, cur_transformer)
                elif image_type is enums.ImageTypes.POSE:
                    cur_filename = filename.replace('.jpg', '_pose_cropped.png')
                    cur_image = load_image(self.pose_config.FEATURES_FOLDER, cur_filename, cur_transformer)
                elif image_type is enums.ImageTypes.POSE_ANNOTATED:
                    cur_filename = filename.replace('.jpg', '_annotated.png')
                    cur_image = load_image(self.pose_config.FEATURES_FOLDER, cur_filename, cur_transformer)
                elif image_type is enums.ImageTypes.FACE:
                    face_filename = filename.replace('.jpg', '_0.png')
                    cur_image = load_image(self.face_config.FEATURES_FOLDER, face_filename, cur_transformer)
                else:
                    raise ValueError('Unknown image type', image_type)
                if cur_image is not None:
                    images[image_type] = cur_image

        return images, label, filename

    def __load_data(self):
        pd_data = pd.read_csv(self.config.ANNOTATION_FILE)
        if self.labels is not None:
            pd_data = pd_data[pd_data['class'].isin(self.labels)]
        if self.sample_type is enums.SampleType.TRAIN:
            pd_data = pd_data[pd_data['sample_type'] == enums.SampleType.TRAIN]
        elif self.sample_type is enums.SampleType.VALIDATION:
            pd_data = pd_data[pd_data['sample_type'] == enums.SampleType.VALIDATION]
        elif self.sample_type is enums.SampleType.TEST:
            pd_data = pd_data[pd_data['sample_type'] == enums.SampleType.TEST]
        elif self.sample_type is enums.SampleType.IGNORED:
            pd_data = pd_data[pd_data['sample_type'] == enums.SampleType.IGNORED]
        elif self.sample_type is enums.SampleType.ALL:
            pass
        elif self.sample_type is enums.SampleType.TRAIN_TEST_VALIDATION:
            pd_data = pd_data[(pd_data['sample_type'] == enums.SampleType.TRAIN) | 
            (pd_data['sample_type'] == enums.SampleType.TEST) | 
            (pd_data['sample_type'] == enums.SampleType.VALIDATION)] 
        elif self.sample_type is enums.SampleType.TRAIN_VALIDATION:
            pd_data = pd_data[(pd_data['sample_type'] == enums.SampleType.TRAIN) | 
            (pd_data['sample_type'] == enums.SampleType.VALIDATION)] 
        elif self.sample_type is enums.SampleType.WITH_JUST_ONE_FACE:
            pd_data = pd_data[pd_data['num_faces'] == 1]
        elif self.sample_type is enums.SampleType.WITH_NO_FACE:
            pd_data = pd_data[pd_data['num_faces'] == 0]
        elif self.sample_type is enums.SampleType.WITH_MORE_THAN_ONE_FACE:
            pd_data = pd_data[pd_data['num_faces'] > 1]
        else:
            raise ValueError('Unknown sample type {sample_type}')

        self.data = pd_data

    def get_image_from(self, sample):
        return sample[0]


