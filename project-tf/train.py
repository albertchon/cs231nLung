import os
import sys

import tensorflow as tf

import numpy as np
from model import LungSystem
from os.path import join as pjoin
import random


import logging
starting=True
logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("best_val_loss", float('inf'), "current best validation loss")
tf.app.flags.DEFINE_string("model", 'simplecnn', "Type of model to use: linear or cnn or simplecnn")
tf.app.flags.DEFINE_string("features", 'pixels', "Type of features to use: pixels or hog")
tf.app.flags.DEFINE_integer("epochs", 15, "number of epochs")
tf.app.flags.DEFINE_float("learning_rate", 0.0003, "Learning rate.")
tf.app.flags.DEFINE_float("leak", 0.01, "Leakiness")
tf.app.flags.DEFINE_float("dropout", 0.0, "dropout prob")
tf.app.flags.DEFINE_integer("num_slices", 64, "number of CT slices for each patient")
tf.app.flags.DEFINE_integer("image_height", 128, "height of each slice in pixels")
tf.app.flags.DEFINE_integer("image_width", 128, "width of each slice in pixels")
tf.app.flags.DEFINE_integer("conv1_filters", 64, "number of conv filters")
tf.app.flags.DEFINE_integer("conv2_filters", 32, "number of conv filters")
tf.app.flags.DEFINE_integer("aff_size", 256, "affine layer size")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_float("train_size", 0.8, "Size of train set")
tf.app.flags.DEFINE_float("val_size", 0.2, "Size of val set")

FLAGS = tf.app.flags.FLAGS
train_dir = './weights/%s-%s/' % (FLAGS.model, FLAGS.features)

def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model

def main(_):
    INPUT_FOLDER = './npy3d-data/'
    patients = os.listdir(INPUT_FOLDER)
    patient_images = {}
    labels = {}

    #storing images
    for p in range(len(patients)):
        path = INPUT_FOLDER + patients[p]
        if patients[p][0] == '1':
            image = np.load(path)
            patient_images[patients[p][:-4]] = image

    #storing labels
    lines = [line for line in open('stage1_solution.csv')]
    for i in range(1, len(lines)):
        data = lines[i].split(',')
        labels['1-'+data[0]] = int(data[1])
    '''
    lines = [line for line in open('stage2_solution.csv')]
    for i in range(1, len(lines)):
        data = lines[i].split(',')
        labels['2-'+data[0]] = int(data[1])
    '''

    x = np.zeros((len(patient_images), FLAGS.num_slices, FLAGS.image_height, FLAGS.image_width), dtype=np.float32)
    y = np.zeros(len(patient_images), dtype=np.int32)
    
    patient_keys = list(patient_images.keys())
    for i in range(len(patient_images)):
        patient = patient_keys[i]
        x[i] = patient_images[patient]
        y[i] = labels[patient]
    num_total = x.shape[0]
    #num_test = int(np.round(num_total * FLAGS.test_size))
    num_val = int(np.round(num_total * FLAGS.val_size))
    num_train = int(np.round(num_total * FLAGS.train_size))
    indices = list(range(num_total))
    random.seed(4783)
    random.shuffle(indices)
    #test_indices = indices[:num_test]
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]
    print('num_total: %s' % num_total)
    #print('num_test: %s' % num_test)
    print('num_val: %s' % num_val)
    print('num_train: %s' % num_train)
    print('indices: %s' % indices)
    #print('test_indices: %s' % test_indices)
    print('val_indices: %s' % val_indices)
    print('train_indices: %s' % train_indices)
    x_train = x[train_indices]
    y_train = y[train_indices]
    if starting:
        indices = list(range(num_train))
        random.shuffle(indices)
        x_train = x_train[indices[:128]]
        y_train = y_train[indices[:128]]
    x_val = x[val_indices]
    y_val = y[val_indices]
    dataset = (x_train, y_train, x_val, y_val)

    lung_model = LungSystem(FLAGS)

    with tf.Session() as sess:
        initialize_model(sess, lung_model, train_dir)
        lung_model.train(sess, dataset, train_dir)
        

        # qa.evaluate_answer(sess, dataset, vocab, FLAGS.evaluate, log=True)

if __name__ == "__main__":
    tf.app.run()


