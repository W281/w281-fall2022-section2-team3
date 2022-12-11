import torchvision.models as models
import configuration

from hyperopt import STATUS_OK,rand, tpe, Trials, fmin, hp
from hyperopt.early_stop import no_progress_loss
from torch import nn
from sklearn.model_selection import cross_val_score
import feature_helpers
import numpy as np
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, validation_curve
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform
from torch import nn
from torchsummary import summary
from torchvision import transforms

from tensorflow.keras.layers import ZeroPadding2D, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.layers import GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import coremltools

# Pull config
config = configuration.Configuration()

class ResNet152(nn.Module):
    def __init__(self, progress=True):
        super(ResNet152, self).__init__()
        
        # load the pretrained model
        self.model = models.resnet152(weights=models.ResNet152_Weights.DEFAULT, progress=progress)

        # select till the last layer
        # Dropping output layer (the ImageNet classifier)
        self.model = nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, x):
      x = self.model(x)
      return x


class LogisticRegression(nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.model = nn.Linear(input_dim, output_dim)

     def forward(self, x):
         outputs = torch.sigmoid(self.model(x))
         return outputs
    
# Optimizer definition
def optimize(opt_dict, best_dict, X_train, y_train, max_evals = 50, scoring_fn = 'neg_log_loss', random_state = config.SEED):
    """
        Runs hyperopt for all the models in opt_dict. Adds the best hyperparameter set for each model.
        Returns dictionary of best hyperparameter set.
    """
    # Define TPE algorithm for all optimizers
    tpe_algo = tpe.suggest

    # Iterate over opt_dict
    for k,v in opt_dict.items():
        ## Attributes
        model_name = k
        model = v['model']
        params = v['params']

        ## objective function definition
        def f(params):
            loss = None
            # try:
            m = model(random_state = random_state, **params)
            loss = -cross_val_score(m, X_train, y_train, scoring = scoring_fn).mean()
            # except: AttributeError

            return {'loss': loss, 'status': STATUS_OK}

        ## Define trial space
        trials = Trials()

        print(f"Optimizing {k} model...")

        ## optimize
        best = fmin(
            fn = f,
            space = params,
            algo = tpe_algo,
            max_evals = max_evals,
            trials = trials,
            early_stop_fn=no_progress_loss(1e-10)
        )

        best_dict[model_name] = best
        print(f"Best hyperparameter set for {model_name} is {best}")
        print("\n")

    return best_dict

class ModelRunner:
    def __init__(self, config, face_config, pose_config, tqdm):
        self.config = config
        self.face_config = face_config
        self.pose_config = pose_config
        self.tqdm = tqdm

    def run_knn(self, features, y, train_idx, n_samples):
        def calc_accuracy(pca, y_train, idx, model_name):
            knn = KNeighborsClassifier(n_neighbors = 3)
            knn.fit(pca, y_train)
            filename = f'{self.config.SAVED_MODELS_FOLDER}/knn_{model_name}.pkl'
            with open(filename, 'wb') as file:
                pickle.dump(knn, file)

            accuracy = 0

            for i in idx:
                label = knn.predict([pca[i,:]])
                if label[0] == y_train[i]:
                    accuracy +=1
            return accuracy

        feature_extractor = feature_helpers.FeatureExtractor(self.config, self.face_config, self.pose_config, self.tqdm)
        y_train = y[train_idx].tolist()
        idx = list(np.random.choice(np.arange(len(y_train)), n_samples, replace=False))

        # Get the principal components
        [pca_model], [X_keypoints_pca] = feature_extractor.get_PCA([features], n_components=[14])
        filename = f'{self.config.SAVED_MODELS_FOLDER}/pca_for_knn_keypoints_pca.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(pca_model, file)

        X_keypoints_pca = X_keypoints_pca[train_idx]

        [X_keypoints_tsne] = feature_extractor.get_tsne([features], n_components=2)
        X_keypoints_tsne = X_keypoints_tsne[train_idx]

        print(f'Accuracies from {n_samples} training samples:')
        print('Keypoints PCA Accuracy: ', calc_accuracy(X_keypoints_pca, y_train, idx, 'keypoints_pca')/n_samples)
        print('Keypoints tSNE Accuracy: ', calc_accuracy(X_keypoints_tsne, y_train, idx, 'keypoints_tsne')/n_samples)


    def load_VGG16(self, weights_path=None, no_top=True):

        input_shape = (224, 224, 3)

        #Instantiate an empty model
        img_input = Input(shape=input_shape)   # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
        x = GlobalAveragePooling2D()(x)
        vmodel = Model(img_input, x, name='vgg16')
        if weights_path is not None:
            vmodel.load_weights(weights_path)
            print("Weights have been loaded.")

        return vmodel

    # Ref: https://coremltools.readme.io/docs/introductory-quickstart
    def convert_tf_saved_model(self, in_model_file, out_model_file):
        # Define the input type as image, 
        # set pre-processing parameters to normalize the image 
        # to have its values in the interval [-1,1] 
        # as expected by the mobilenet model
        image_input = coremltools.ImageType(shape=(1, 224, 224, 3,),
                                   bias=[-1,-1,-1], scale=1/255.)

        # set class labels
        class_labels = list(self.config.class_dict.values())
        
        classifier_config = coremltools.ClassifierConfig(class_labels)

        model = load_model(in_model_file)
        
        print("[INFO] converting model")
        coreml_model = coremltools.convert(
            model, 
            inputs=[image_input], 
            classifier_config=classifier_config,
        )    

        print("[INFO] saving model")
        coreml_model.save(out_model_file)