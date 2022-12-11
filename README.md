# **W281 Fall 2022 Final Report Driver Behavior Detection**
## by Hoon Kim, Kai Ying, Ram Senthamarai, Dmitry Baron
The main objective of our project is to design classification algorithm to classify ten types of driver behavior via **identification of the differences in driver’s face and posture** when performing each of these actions:
* Safe Driving
* Texting (right)
* Phone Call (right)
* Texting(left)
* Phone Call(left)
* Fiddling With Console
* Drinking
* Reaching Back
* Fixing Looks
* Conversing.

## Code Organization
This repository has two main notebooks under [notebooks](./notebooks) folder:
* [EDA](./notebooks/Run_eda.ipynb) with code for generating all the features required for the main notebook below.
* [Main Notebook](./notebooks/W281_Fall_2022_Final_Report_Driver_Behavior_Detection.ipynb) covers feature analysis, feature selection, model training and model evaluation.

Each of these notebooks list out the prerequisites needed.

Apart from these notebooks, we also have a few helper classes and methods defined in a few python files, also located under notebooks folder.

## Setting up Environment
Some of the main dependencies for this project:
* MTCNN model
* DLIB model
* BodyPix model
* Pytorch's facenet library
* ScikitLearn library
* pytorch and TorchVisions libraries
* Tensorflow library

Please use the following commands to create a new Conda environment for the project:
```
create -n w281_proj python=3.8
conda activate w281_proj
# SYSTEM_VERSION_COMPAT=0 pip install tensorflow-macos tensorflow-metal # Needed only on MacOS else just install tensorflow.
conda install -c pytorch torchvision
conda install jupyterlab
conda install matplotlib pandas tabulate numpy nltk scikit-learn
conda install seaborn pillow lxml protobuf
conda install imageio
pip install tensorflow-io

pip install scikit-build
pip install cmake
# opencv installation with conda install fails.
# conda install -c conda-forge opencv tf_slim
# Instead, download opencv-python-4.6.0.66 tar file from 
# https://pypi.org/project/opencv-python/ and use setup.py in it.
python setup.py install 

conda install scikit-image
pip install torch-summary
conda install setuptools
pip install coremltools
pip install mtcnn
pip install dlib

pip install tf-bodypix
pip install tfjs-graph-converter

pip install facenet-pytorch
pip install -U openmim
mim install mmcv-full
conda install -c conda-forge ipywidge
conda install -c conda-forge hyperopt
```

## Running Notebooks
In order to run the notebooks, checkout this repo, change in to the base folder and start jupyter notebook. Open the W281_Fall_2022_Final_Report_Driver_Behavior_Detection.ipynb notebook.

## iOS App
The iOS app is located under [ios_app/StopDistractedDriving](ios_app/StopDistractedDriving) folder. Please refer to [README.md](ios_app/StopDistractedDriving/README.md) for more information on the app.
