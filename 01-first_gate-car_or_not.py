#!/usr/bin/env python
# coding: utf-8

# ### data0 - all car images, including whole and damaged

# In[1]:


import urllib
from IPython.display import Image, display, clear_output
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import seaborn as sns
# get_ipython().magic(u'matplotlib inline')

import json
import pickle as pk
from sklearn.metrics import classification_report, confusion_matrix


# In[2]:


import os
import h5py
import numpy as np
import pandas as pd

from keras.utils.data_utils import get_file
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from keras import optimizers
from keras.callbacks import ModelCheckpoint, History
from tensorflow.python.client import device_lib




CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# from Keras GitHub
def get_predictions(preds, top=5):
    global CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        fpath = get_file('imagenet_class_index.json',
                         CLASS_INDEX_PATH,
                         cache_subdir='models')
        CLASS_INDEX = json.load(open(fpath))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results


def prepare_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def get_car_categories1():
    d = defaultdict(float)
    img_list = os.listdir('data0')
    for i, img_path in enumerate(img_list):
        img = prepare_image('data0/'+img_path)
        out = vgg16.predict(img)
        top = get_predictions(out, top=5)
        for j in top[0]:
            d[j[0:2]] += j[2]
        if i % 50 == 0:
            print i, '/', len(img_list), 'complete'
    return Counter(d)


def get_car_categories2(cat_list):
    img_list = os.listdir('data0')
    num = 0
    bad_list = []
    for i, img_path in enumerate(img_list):
        img = prepare_image('data0/'+img_path)
        out = vgg16.predict(img)
        top = get_predictions(out, top=5)
        for j in top[0]:
            if j[0:2] in cat_list:
                num += 1
                break # breaks out of for loop if one of top 50 categories is found
            else:
                pass
            bad_list.append(img_path) # appends to "bad list" if none of the 50 are found
        if i % 100 == 0:
            print i, '/', len(img_list), 'complete'
    bad_list = [k for k, v in Counter(bad_list).iteritems() if v == 5]
    return num, bad_list


def view_images(img_dir, img_list):
    for img in img_list:
        clear_output()
        display(Image(img_dir+img))
        num = raw_input("c to continue, q to quit")
        if num == 'c':
            pass
        else:
            return 'Finished for now.'

def car_categories_gate(image_path, cat_list):
    # urllib.urlretrieve(image_path, 'save.jpg') # or other way to upload image
    img = prepare_image(image_path)
    out = vgg16.predict(img)
    top = get_predictions(out, top=5)
    print "Predictions for ", image_path, ": ", top
    print "Validating that ", image_path, " is a picture of a car..."
    for j in top[0]:
        if j[0:2] in cat_list:
            print "Matches from category list for ", image_path, ": ", j[0:2]
            return "YES"
    return "NO"

vgg16 = VGG16(weights='imagenet')
vgg16.save('vgg16.h5')

cat_counter = get_car_categories1()
with open('cat_counter.pk', 'wb') as f:
    pk.dump(cat_counter,f,-1)
with open('cat_counter.pk', 'rb') as f:
    cat_counter = pk.load(f)
cat_list = [k for k, v in cat_counter.most_common()[:50]]

get_available_gpus()
print car_categories_gate('cat.jpg', cat_list)
print car_categories_gate('whole-car.jpg', cat_list)
print car_categories_gate('damaged_car.jpg', cat_list)
print car_categories_gate('damaged_car_2.jpg', cat_list)
print car_categories_gate('damaged_car_3.jpg', cat_list)
