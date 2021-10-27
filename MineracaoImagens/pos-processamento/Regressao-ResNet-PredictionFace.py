# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 14:56:26 2021

@author: gabriel
"""

"""

Regressão - Estimar idade a partir da foto
Fonte: https://github.com/amaiya/ktrain/blob/master/examples/vision/utk_faces_age_prediction-resnet50.ipynb
"""

#!pip install ktrain

"""
%reload_ext autoreload
%autoreload 2
%matplotlib inline
"""

import os
os.environ['DISABLE_V2_BEHAVIOR'] = '1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0" 
import ktrain
from ktrain import vision as vis


"""
Image Regression: Age Prediction
In this example, we will build a model that predicts the age of a person given the person's photo.

Download the Dataset
From this blog post by Abhik Jha, we see that there are several face datasets with age annotations from which to choose:

UTK Face Dataset
IMDb-Wiki Face Dataset
Appa Real Face Dataset
In this notebook, we use the UTK Face Dataset. Download the data from http://aicip.eecs.utk.edu/wiki/UTKFace and extrct each of the three zip files to the same folder.

STEP 1: Load and Preprocess the Dataset
The target age attribute in this dataset is encoded in the filename. More specifically, filenames are of the form:

 [age]_[gender]_[race]_[date&time].jpg
where

[age] is an integer from 0 to 116, indicating the age
[gender] is either 0 (male) or 1 (female)
[race] is an integer from 0 to 4, denoting White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).
[date&time] is in the format of yyyymmddHHMMSSFFF, showing the date and time an image was collected to UTKFace
We are only interested in extracting the age for use as a numerical target. Let us first construct a regular expression to extract the age from the filename. Then, we can supply the pattern to images_from_fname to load and preprocess the dataset. Supplying is_regression=True is important here, as it tells ktrain that the integer targets representing age should be treated as a numerical target, as oppposed to a class label.

"""

#!pip install gdown

#!gdown https://drive.google.com/uc?id=0BxYys69jI14kYVM3aVhKS1VhRUk

#!tar -xzvf UTKFace.tar.gz


# build a regular expression that extracts the age from file name
PATTERN = r'([^/]+)_\d+_\d+_\d+.jpg.chip.jpg$'
import re
p = re.compile(PATTERN)
r = p.search('UTKFace/30_1_2_20170116191309887.jpg.chip.jpg')
print("Extracted Age:%s" % (int(r.group(1))))


#Set DATADIR to the folder where you extracted all the images.

DATADIR='UTKFace'
data_aug = vis.get_data_aug(horizontal_flip=False)
(train_data, val_data, preproc) = vis.images_from_fname(DATADIR, pattern = PATTERN, data_aug = data_aug, 
                                                        is_regression=True, random_state=42)


"""
From the warnings above, we see that a few filenames in the dataset are constructed incorrectly. For instance, the first filename incorrectly has two consecutive underscore characters after the age attribute. Although the age attribute appears to be intact despite the errors and we could modify the regular expression to process these files, we will ignore them in this demonstration.
STEP 2: Create a Model and Wrap in Learner
We use the image_regression_model function to create a ResNet50 model. By default, the model freezes all layers except the final randomly-initialized dense layer.

"""

vis.print_image_regression_models()

model = vis.image_regression_model('pretrained_resnet50', train_data, val_data)

# wrap model and data in Learner object
learner = ktrain.get_learner(model=model, train_data=train_data, val_data=val_data, 
                             workers=8, use_multiprocessing=False, batch_size=64)


"""
STEP 3: Estimate Learning Rate
We will select a learning rate associated with falling loss from the plot displayed.
"""

"""
From the plot above, we choose a learning rate of 1e-4.

STEP 4: Train Model
We will begin by training the model for 3 epochs using a 1cycle learning rate policy.
"""
learner.fit_onecycle(1e-4, 1)

"""
After only 5 epochs, our validation MAE is 6.57. That is, on average, our age predictions are off about 6 1/2 years. Since it does not appear that we are overfitting yet, we could try training further for further improvement, but we will stop here for now.

Make Predictions
Let's make predictions on individual photos. We could either randomly select from the entire image directory or select just from the validation images.
"""

# get a Predictor instance that wraps model and Preprocessor object
predictor = ktrain.get_predictor(learner.model, preproc)

## get some random file names of images
#!!ls {DATADIR} | sort -R |head -10

# how to get validation filepaths
val_data.filenames[10:100]


def show_prediction(fname):
    fname = DATADIR+'/'+fname
    predicted = round(predictor.predict_filename(fname)[0])
    actual = int(p.search(fname).group(1))
    vis.show_image(fname)
    print('predicted:%s | actual: %s' % (predicted, actual))
    

show_prediction('55_1_3_20170109142400325.jpg.chip.jpg')

#!pip install git+https://github.com/amaiya/eli5@tfkeras_0_10_1

predictor.explain( DATADIR + '/55_1_3_20170109142400325.jpg.chip.jpg')