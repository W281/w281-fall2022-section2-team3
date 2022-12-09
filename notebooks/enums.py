from enum import Enum, IntEnum, unique

@unique
class ImageTypes(Enum):
    ORIGINAL = 1
    FACE = 2
    FACE_ANNOTATED = 3
    POSE = 4
    POSE_ANNOTATED = 5

@unique
class SampleType(IntEnum):
    TRAIN = 1
    VALIDATION = 2
    TEST = 3
    IGNORED = 4
    ALL = 5
    TRAIN_TEST_VALIDATION = 6
    WITH_JUST_ONE_FACE = 7
    WITH_NO_FACE = 8
    WITH_MORE_THAN_ONE_FACE = 9
    TRAIN_VALIDATION = 10

@unique
class FeatureType(Enum):
    PIXEL = 'Pixel'
    HOG = 'Hog'
    CNN = 'CNN'
    CANNY = 'Canny'
    POSE = 'Pose'
    KEYPOINTS = 'Keypoints'
    BODY_PARTS = 'Bodyparts'

@unique
class DataColumn(Enum):
    FILENAME = 'filename'
    LABEL = 'label'
    ORIGINAL = 'original'
    FACE = 'face'
    FACE_ANNOTATED = 'face_annotated'
    POSE = 'pose'
    POSE_ANNOTATED = 'pose_annotated'
    PIXEL_VECTOR = 'pixel_vector'
    HOG_VECTOR = 'hog_vector'
    CNN_VECTOR = 'cnn_vector'
    CANNY_VECTOR = 'canny_vector'
    POSE_VECTOR = 'pose_vector'
    POSE_KEYPOINTS_VECTOR = 'pose_keypoints_vector'
    BODY_PARTS__VECTOR = 'body_parts_vector'

@unique
class DataScalers(Enum):
    POSE_KEYPOINTS_VECTOR_SCALER = 'scaler_pose_keypoints_vector.pkl',
    POSE_KEYPOINTS_VECTOR_PCA_SCALER = 'scaler_pose_keypoints_vector_pca_14.pkl'
