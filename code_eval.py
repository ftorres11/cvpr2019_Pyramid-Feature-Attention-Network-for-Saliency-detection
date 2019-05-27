# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import pdb

from keras.layers import Input
from model import VGG16
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ========================================================================
# Functions
# ========================================================================
def padding(x):
    h, w, c = x.shape
    size = max(h,w)
    paddingh = (size-h)//2
    paddingw = (size-w)//2
    temp_x = np.zeros((size, size, c))
    temp_x[paddingh:h+paddingh, paddingw:w+paddingw, :] = x
    return temp_x


def load_image(path):
    x = cv2.imread(path)
    sh = x.shape
    x = np.array(x, dtype=np.float32)
    x = x[..., ::-1]
    # Zero-center by mean pixel
    x[..., 0] -= 103.939
    x[..., 1] -= 116.779
    x[..., 2] -= 123.68
    x = padding(x)
    x = cv2.resize(x, target_size, interpolation=cv2.INTER_LINEAR)
    x = np.expand_dims(x, 0)
    return x, sh


def cut(predict, shape):
    h, w, c = shape
    size = max(h, w)
    predict = cv2.resize(predict, (size, size))
    paddingh = (size - h)//2
    paddingw = (size - w)//2
    return predict[paddingh:h + paddingh, paddingw:w + paddingw]


def sigmoid(x):
    return 1/(1+np.exp(-x))


def getres(predict, shape):
    predict = sigmoid(predict)*255
    predict = np.array(predict, dtype=np.uint8)
    predict = np.squeeze(predict)
    predict = cut(predict, shape)
    return predict


def laplace_edge(x):
    laplace = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    edge = cv2.filter2D(x/255., -1, laplace)
    edge = np.maximum(np.tanh(edge), 0)
    edge = edge*255
    edge = np.array(edge, dtype=np.uint8)
    return edge


def str_find(char, target):
    indexes = []
    for idx, elem in char:
        if elem == target:
            indexes.append(idx)
    if len(indexes) == 0:
        indexes.append(-1)
    return indexes


# ========================================================================
# Main Code
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = 'model/PFA_00050.h5'
target_size = (256, 256)

target_store = 'Trials/{}'.format(model_name.replace('model/')\
                .replace('.h5'))

target_sal = os.path.join(target_store, 'Salient_Maps')
target_conf = os.path.join(target_store, 'Confidence')

if not os.path.exists(target_store):
    os.makedirs(target_sal)
    os.makedirs(target_conf)

droput = False
with_CPFE = True
with_CA = True
with_SA = True

if target_size[0] % 32 != 0 or target_size[1] % 32 != 0:
    raise ValueError('Image height and weight must be a multiple of 32')

model_input = Input(shape=(target_size[0], target_size[1], 3))
model = VGG16(model_input, dropout=droput, with_CPFE=with_CPFE,
           with_CA=with_CA, with_SA=with_SA)
model.load_weights(model_name, by_name=True)

for layer in model.layers:
    layer.trainable = False

# ========================================================================
# Preparing the evaluation
val_lines = open('test_pair.txt', 'r').readlines()
for idx, line_idx in enumerate(val_lines):
    raw_route, mask_route = line_idx.split(' ')
    raw_route = raw_route.strip()
    mask_route = mask_route.strip()
    raw_img, raw_shape = load_image(raw_route)
    mask_im, mask_shpe = load_image(mask_route)

    raw_img = np.array(raw_img, dtype=np.float32)
    sa = model.predict(raw_img)
    sa = getres(sa, raw_shape)
   
    pdb.set_trace()
