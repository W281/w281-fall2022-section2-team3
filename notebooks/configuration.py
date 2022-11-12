class Configuration:
    def __init__(self):
        self.INPUT_FOLDER = '../input'
        self.OUTPUT_FOLDER = '../output'
        self.TRAIN_DATA = f'{self.INPUT_FOLDER}/state-farm-distracted-driver-detection/imgs/train'
        self.TEST_DATA = f'{self.INPUT_FOLDER}/state-farm-distracted-driver-detection/imgs/test'
        self.TRAIN_DATA_FULL = f'{self.INPUT_FOLDER}/state-farm-distracted-driver-detection/imgs/train_orig'
        self.DLIB_MODELS_FOLDER = f'{self.INPUT_FOLDER}/dlib_models'

        # IMAGES_BASE = '/Users/rasentha/mids/w281/project/test_images'
        self.IMAGES_BASE = self.TRAIN_DATA_FULL
        
        self.class_dict = {0 : "Safe Driving",
                           1 : "Texting (right)",
                           2 : "Phone Call (right)",
                           3 : "Texting(left)",
                           4 : "Phone Call(left)",
                           5 : "Fiddling With Console",
                           6 : "Drinking",
                           7 : "Reaching Back",
                           8 : "Fixing Looks",
                           9 : "Conversing"}
