# -*- coding: utf-8 -*-
import os
import pdb
import cv2
import json
import numpy as np
import scipy as sc
from PIL import Image

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
    predict = cv2.resize(predict, shape[::-1])
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
    for idx, elem in enumerate(char):
        if elem == target:
            indexes.append(idx)
    if len(indexes) == 0:
        indexes.append(-1)
    return indexes


def pr_im(pred, normalized):
    thresh = range(0, 256)
    normalized.astype(int)
    tot_elem = pred.shape[0]*pred.shape[1]
    prec = []
    rec = []
    fval = []
    wfval = []
    beta_sq = 0.3
    for x in thresh:
        # Getting those that pass the threshold
        booled_pos = (pred >= x)*1  # Positives
        booled_neg = (pred < x)*1  #Negatives

        positives = (booled_pos == normalized)
        negatives = (booled_neg == (normalized == 0)*1)

        # Cases
        tp_x, tp_y = np.where(positives == True)  #TP
        fp_x, fp_y = np.where((booled_pos-normalized) == 1)  #FP

        tn_x, tn_y = np.where(negatives==True)  #TN
        fn_x, fn_y = np.where(booled_neg-((normalized == 0)*1)==1)  #FN

        TP = len(tp_x)
        FP = len(fp_x)
        TN = len(tn_x)
        FN = len(fn_x)

        precision = float(TP)/float(TP+FP)
        recall = float(TP)/float(TP+FN)
        prec.append(precision)
        rec.append(recall)

        fval.append(2*((precision*recall)/(precision+recall)))
        wfval.append((1+beta_sq)*((precision*recall)/\
                    ((beta_sq*precision)+recall)))

    prec.reverse()
    rec.reverse()
    max_f = np.max(fval)
    max_fb = np.max(wfval)
    return prec, rec, max_f, max_fb


# ========================================================================
# Main Code
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name = 'model/PFA_00050.h5'
target_size = (256, 256)

target_store = 'Trials/{}'.format(model_name.replace('model/', '')\
                .replace('.h5', ''))

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
val_lines = open('../datasets/test_pair.txt', 'r').readlines()
precision = 0
recall = 0
fval = 0
wfval = 0
for idx, line_idx in enumerate(val_lines):
    raw_route, mask_route = line_idx.split(' ')
    raw_route = raw_route.strip()
    mask_route = mask_route.strip()
    raw_img, raw_shape = load_image(raw_route)
    mask_im = Image.open(mask_route)
    mask_np = np.array(mask_im)/255
    mask_sh = mask_np.shape

    raw_img = np.array(raw_img, dtype=np.float32)
    sa = model.predict(raw_img)
    segm = sigmoid(sa)
    segm = np.squeeze(segm)*255
    sa = getres(sa, mask_sh[:2])
    
    sl_idx = str_find(raw_route, '/')
    img_name = raw_route[sl_idx[-1]+1::]
    dest_sal = os.path.join(target_sal, img_name)
    dest_conf = os.path.join(target_conf, img_name)

    if idx % 1000 == 0:
        cv2.imwrite(dest_sal, segm)  #Segmentation
        cv2.imwrite(dest_conf, sa)  # Sa?

    prec, rec, f_it, wf_it = pr_im(sa, mask_np)

    precision += np.asarray(prec)
    recall += np.asarray(rec)

    fval += f_it
    wfval += wf_it


precision = precision.astype(float)/float(idx+1)
recall = recall.astype(float)/float(idx+1)
avg_f = fval/float(idx+1)
avgwf = wfval/float(idx+1)
plt.figure()
plt.plot(recall, precision, 'r-')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid()
plt.title('PR Curve, f-value {:.3}, weighted f-value {:.3}'.format(avg_f, avgwf))
plt.savefig(os.path.join(target_store,'pr-curve.pdf'))


data_sto = {}
data_sto['precision'] = precision.tolist()
data_sto['recall'] = recall.tolist()
data_sto['f_value'] = avg_f
data_sto['wf_value'] = avgwf

with open(os.path.join(target_store, 'metrics.json'), 'w') as outfile:
    json.dump(data_sto, outfile)

