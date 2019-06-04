# -*- coding: utf-8 -*-
import os
import math
import argparse
from utils import *
from edge_hold_loss import *

import tensorflow as tf
from keras.layers import Input
from keras import callbacks, optimizers

from model import VGG16
from data import getTrainGenerator


# ========================================================================
# Environmental variables
# ========================================================================
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# ========================================================================
# Functions
# ========================================================================
def lr_scheduler(epoch):
    drop = 0.5
    epoch_drop = epochs/8.
    lr = base_lr * math.pow(drop, math.floor((1+epoch)/epoch_drop))
    print('lr: %f' % lr)
    return lr


# ========================================================================
# Main Routine
# ========================================================================
if __name__ == '__main__':
    # Parsing arguments
    parser = argparse.ArgumentParser(description=
                      'Train model your dataset')
    parser.add_argument('--train_file', default='train_pair.txt',help = 
                      'your train file', type=str)
    parser.add_argument('--model_weights', default='model/vgg16_no_top.h5',
                      help='your model weights', type=str)

    args = parser.parse_args()
    model_name = args.model_weights
    train_path = args.train_file
    
    # Model parameters
    target_size = (256,256)
    batch_size = 15
    base_lr = 1e-2
    epochs = 50
    
    # Dataloading routine
    f = open(train_path, 'r')
    trainlist = f.readlines()
    f.close()
     
    # Training routine definitions
    steps_per_epoch = len(trainlist)/batch_size
    optimizer = optimizers.SGD(lr=base_lr, momentum=0.9, decay=0)
    '''optimizer = optimizers.Adam(lr=base_lr)'''  # Optional optimizer
    loss = EdgeHoldLoss

    # Storing and metrics 
    metrics = [acc,pre,rec]
    dropout = True
    with_CPFE = True
    with_CA = True
    with_SA = True
    log = './COCO.csv'
    tb_log = './tensorboard-logs/COCO'
    model_save = 'model/COCO_'
    model_save_period = 5
    if target_size[0 ] % 32 != 0 or target_size[1] % 32 != 0:
        raise ValueError('{} {}'.format('Image height and weight must',
                                        'be a multiple of 32'))

    # Training data generator and or shuffler
    traingen = getTrainGenerator(train_path, target_size, batch_size,
                                 israndom=False)

    # Model definition and options
    model_input = Input(shape=(target_size[0],target_size[1],3))
    model = VGG16(model_input,dropout=dropout, with_CPFE=with_CPFE, 
                  with_CA=with_CA, with_SA=with_SA)
    model.load_weights(model_name,by_name=True)

    # Tensorflow & Tensorboard options
    tb = callbacks.TensorBoard(log_dir=tb_log)
    lr_decay = callbacks.LearningRateScheduler(schedule=lr_scheduler)
    es = callbacks.EarlyStopping(monitor='loss', patience=3, verbose=0,
                                 mode='auto')
    modelcheck = callbacks.ModelCheckpoint(model_save+'{epoch:05d}.h5',
                                 monitor='loss', verbose=1,
                                 save_best_only=False,
                                 save_weights_only=True, mode='auto',
                                 period=model_save_period)

    callbacks = [lr_decay,modelcheck,tb]

    # Training routine
    model.compile(optimizer=optimizer,loss=loss,metrics=metrics)
    model.fit_generator(traingen, steps_per_epoch=steps_per_epoch,
                        epochs=epochs,verbose=1,callbacks=callbacks)
