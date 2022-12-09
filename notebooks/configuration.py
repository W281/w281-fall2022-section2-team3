class PoseConfig:
    def __init__(self, config):
        self.FEATURES_FOLDER = f'{config.OUTPUT_FOLDER}/poses'
        self.SUMMARY_NAME = 'pose_extraction_summary.csv'
        self.SAVED_MODEL_FOLDER = f'{config.INPUT_FOLDER}/movenet_models/movenet_singlepose_thunder_4'
        # self.input_size = 192 # if model is movenet_lightning
        self.INPUT_SIZE = 256 # if model is movenet_thunder
        self.MIN_CROP_KEYPOINT_SCORE = 0.2
        self.KEYPOINT_THRESHOLD = 0.11
        self.KEYPOINT_DICT = {
            'nose': 0,
            'left_eye': 1,
            'right_eye': 2,
            'left_ear': 3,
            'right_ear': 4,
            'left_shoulder': 5,
            'right_shoulder': 6,
            'left_elbow': 7,
            'right_elbow': 8,
            'left_wrist': 9,
            'right_wrist': 10,
            'left_hip': 11,
            'right_hip': 12,
            'left_knee': 13,
            'right_knee': 14,
            'left_ankle': 15,
            'right_ankle': 16
        }
        self.KEYPOINT_IDX_NAME = {v: k for k, v in self.KEYPOINT_DICT.items()}
        self.SORTED_KEYPOINT_NAMES = [k for k, v in sorted(self.KEYPOINT_DICT.items(), key=lambda item: item[1])]
        self.BASE_KEYPOINT = 0
        self.KEYPOINTS_INCLUDED_IN_FEATURES = set(range(1, 13)) # The feature vector is based on all keypoints
                                                                # except nose, left_knee, right_knee,
                                                                # left_ankle and right_ankle
        # Maps bones to a matplotlib color name.
        self.KEYPOINT_EDGE_INDS_TO_COLOR = {
            (0, 1): 'm',
            (0, 2): 'c',
            (1, 3): 'm',
            (2, 4): 'c',
            (0, 5): 'm',
            (0, 6): 'c',
            (5, 7): 'm',
            (7, 9): 'm',
            (6, 8): 'c',
            (8, 10): 'c',
            (5, 6): 'y',
            (5, 11): 'm',
            (6, 12): 'c',
            (11, 12): 'y',
            (11, 13): 'm',
            (13, 15): 'm',
            (12, 14): 'c',
            (14, 16): 'c'
        }

class FaceConfig:
    def __init__(self, config):
        self.FEATURES_FOLDER = f'{config.OUTPUT_FOLDER}/faces'
        self.FACE_SUMMARY_NAME = 'face_extraction_summary.csv'
        self.FACE_SUMMARY_FILEPATH = f'{self.FEATURES_FOLDER}/{self.FACE_SUMMARY_NAME}'
        self.EYE_HAAR_CASCADES = f'{config.INPUT_FOLDER}/haarcascade_eye.xml'

class Configuration:
    def __init__(self):
        self.INPUT_FOLDER = '/Users/rasentha/mids/w281/project/w281-fall2022-section2-team3/input'
        self.OUTPUT_FOLDER = '/Users/rasentha/mids/w281/project/w281-fall2022-section2-team3/output'

        self.TRAIN_DATA = f'{self.INPUT_FOLDER}/state-farm-distracted-driver-detection/imgs/train'
        self.DLIB_MODELS_FOLDER = f'{self.INPUT_FOLDER}/dlib_models'

        self.ANNOTATION_FILE = f'{self.TRAIN_DATA}/annotation.csv'
        self.ANNOTATION_FILE_COLS = ['filename', 'class', 'num_faces', 'sample_type']
        self.FEATURE_VECTORS_FOLDER = f'{self.TRAIN_DATA}/feature_vectors'
        self.IMAGES_BASE = self.TRAIN_DATA
        self.SEED = 42
        self.SAVED_MODELS_FOLDER = f'{self.OUTPUT_FOLDER}/saved_models'
        self.class_dict = {0 : "Safe Driving",
                           1 : "Texting(right)",
                           2 : "Phone Call(right)",
                           3 : "Texting(left)",
                           4 : "Phone Call(left)",
                           5 : "Fiddling With Console",
                           6 : "Drinking",
                           7 : "Reaching Back",
                           8 : "Fixing Looks",
                           9 : "Conversing"}

        self.included_labels = set([0, # Safe Driving
                                    1, # Texting (right)
                                    2, # Phone Call (right)"
                                    9 # Conversing
                                  ])

