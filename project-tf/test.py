import os
import sys

import tensorflow as tf

import numpy as np
from model2 import LungSystem
from os.path import join as pjoin
import random


import logging
starting=False
logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_bool("save_best_train_loss", False, "whether save based on tr or val loss")
tf.app.flags.DEFINE_float("best_gm", float('-inf'), "current best GM between sens and spec")
tf.app.flags.DEFINE_float("best_val_loss", float('inf'), "current best val loss")
tf.app.flags.DEFINE_float("best_train_loss", float('inf'), "current best train loss")
tf.app.flags.DEFINE_string("model", 'googlenet', "Type of model to use: linear or cnn or simplecnn")
tf.app.flags.DEFINE_string("features", 'pixels', "Type of features to use: pixels or hog")
tf.app.flags.DEFINE_integer("epochs", 99999, "number of epochs")
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate.")
tf.app.flags.DEFINE_float("leak", 0.01, "Leakiness")
tf.app.flags.DEFINE_float("dropout", 0.5, "dropout prob")
tf.app.flags.DEFINE_float("reg_scale", 0.0001, "Size of val set")
tf.app.flags.DEFINE_integer("num_slices", 64, "number of CT slices for each patient")
tf.app.flags.DEFINE_integer("image_height", 128, "height of each slice in pixels")
tf.app.flags.DEFINE_integer("image_width", 128, "width of each slice in pixels")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_float("train_size", 0.6, "Size of train set")
tf.app.flags.DEFINE_float("val_size", 0.2, "Size of val set")
tf.app.flags.DEFINE_float("test_size", 0.2, "Size of val set")


FLAGS = tf.app.flags.FLAGS
train_dir = './weights/%s-%s/' % (FLAGS.model, FLAGS.features)

def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Unable to read parameters from %s" % train_dir)
        sys.exit()
    return model

def main(_):
    INPUT_FOLDER = './npy3d-data/'
    patients = os.listdir(INPUT_FOLDER)
    patients.sort()
    patient_images = {}
    labels = {}

    #storing images
    for p in range(len(patients)):
        path = INPUT_FOLDER + patients[p]
        #if patients[p][0] == '1':
        image = np.load(path)
        patient_images[patients[p][:-4]] = image

    #storing labels
    lines = [line for line in open('stage1_solution.csv')]
    for i in range(1, len(lines)):
        data = lines[i].split(',')
        labels['1-'+data[0]] = int(data[1])
    
    lines = [line for line in open('stage2_solution.csv')]
    for i in range(1, len(lines)):
        data = lines[i].split(',')
        labels['2-'+data[0]] = int(data[1])
    

    x = np.zeros((len(patient_images), FLAGS.num_slices, FLAGS.image_height, FLAGS.image_width), dtype=np.float32)
    y = np.zeros(len(patient_images), dtype=np.int32)
    
    patient_keys = sorted(list(patient_images.keys()))
    for i in range(len(patient_images)):
        patient = patient_keys[i]
        x[i] = patient_images[patient]
        y[i] = labels[patient]
    num_total = x.shape[0]
    num_test = int(np.round(num_total * FLAGS.test_size))
    num_val = int(np.round(num_total * FLAGS.val_size))
    num_train = int(np.round(num_total * FLAGS.train_size))
    indices = list(range(num_total))
    random.seed(4783)
    random.shuffle(indices)
    test_indices = indices[:num_test]
    val_indices = indices[num_test:num_val+num_test]
    train_indices = indices[num_test+num_val:]
    print('num_total: %s' % num_total)
    print('num_test: %s' % num_test)
    print('num_val: %s' % num_val)
    print('num_train: %s' % num_train)
    print('indices: %s' % indices)
    print('test_indices: %s' % test_indices)
    print('val_indices: %s' % val_indices)
    print('train_indices: %s' % train_indices)
    x_train = x[train_indices]
    mean = np.mean(x_train, axis=0)
    x_train -= mean
    y_train = y[train_indices]
    if starting:
        indices = list(range(num_train))
        x_train = x_train[:128]
        y_train = y_train[:128]
    x_val = x[val_indices]
    x_val -= mean
    y_val = y[val_indices]
    x_test = x[test_indices]
    x_test -= mean
    y_test = y[test_indices]
    lung_model = LungSystem(FLAGS)

    with tf.Session() as sess:
        initialize_model(sess, lung_model, train_dir)
        logging.info("Evaluating model on stage 2...") 
        #test_loss = lung_model.test(sess, x_test, y_test)[0]
        test_accuracy, test_sens, test_spec = lung_model.accuracy(sess, x_test, y_test)
        #logging.info("Test loss: %s" % (test_loss))
        logging.info("Test: accuracy = %s, sensitivity = %s, specificity = %s" % (test_accuracy, 
                test_sens, test_spec))

if __name__ == "__main__":
    tf.app.run()
