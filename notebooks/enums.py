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

