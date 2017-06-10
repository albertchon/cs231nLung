from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import sys
import numpy as np
from model import LungSystem
from os.path import join as pjoin
import random

import logging

logging.basicConfig(level=logging.INFO)

STAGE = 2
tf.app.flags.DEFINE_float("best_val_hm", -1.0, "current best validation HM between sensitivity and specificity")
tf.app.flags.DEFINE_string("model", 'cnn', "Type of model to use: linear or cnn")
tf.app.flags.DEFINE_string("features", 'hog', "Type of features to use: pixels or hog")
tf.app.flags.DEFINE_integer("epochs", 10, "number of epochs")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_integer("num_slices", 64, "number of CT slices for each patient")
tf.app.flags.DEFINE_integer("image_height", 128, "height of each slice in pixels")
tf.app.flags.DEFINE_integer("image_width", 128, "width of each slice in pixels")
tf.app.flags.DEFINE_integer("conv1_filters", 64, "number of conv filters")
tf.app.flags.DEFINE_integer("conv2_filters", 32, "number of conv filters")
tf.app.flags.DEFINE_integer("aff_size", 256, "affine layer size")
tf.app.flags.DEFINE_integer("batch_size", 8, "Batch size to use during training.")
tf.app.flags.DEFINE_float("train_size", 0.6, "Size of train set")
tf.app.flags.DEFINE_float("val_size", 0.2, "Size of val set")
tf.app.flags.DEFINE_float("test_size", 0.2, "Size of test set")


FLAGS = tf.app.flags.FLAGS
train_dir = './stage%s-weights/%s-%s/' % (STAGE, FLAGS.model, FLAGS.features)

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
    INPUT_FOLDER = './stage%s-%s-npy/' % (STAGE, FLAGS.features)
    patients = os.listdir(INPUT_FOLDER)
    patient_images = {}
    labels = {}

    #storing images
    for p in range(len(patients)):
        image = np.load(INPUT_FOLDER + patients[p])
        patient_images[patients[p][:-4]] = image

    #storing labels
    lines = [line for line in open('stage%s_solution.csv' % (STAGE))]
    for i in range(1, len(lines)):
        data = lines[i].split(',')
        labels[data[0]] = int(data[1])


    x = np.zeros((len(patient_images), FLAGS.image_height, FLAGS.image_width, FLAGS.num_slices), dtype=np.float32)
    y = np.zeros(len(patient_images), dtype=np.int32)
    for i in range(len(patient_images)):
        patient = patient_images.keys()[i]
        x[i] = patient_images[patient]
        y[i] = labels[patient]
    num_total = x.shape[0]
    num_test = int(np.round(num_total * FLAGS.test_size))
    num_val = int(np.round(num_total * FLAGS.val_size))
    num_train = int(np.round(num_total * FLAGS.train_size))
    indices = range(num_total)
    random.seed(231)
    random.shuffle(indices)
    test_indices = indices[:num_test]
    val_indices = indices[num_test:num_test + num_val]
    train_indices = indices[num_test + num_val:]
    print('num_total: %s' % num_total)
    print('num_test: %s' % num_test)
    print('num_val: %s' % num_val)
    print('num_train: %s' % num_train)
    print('indices: %s' % indices)
    print('test_indices: %s' % test_indices)
    print('val_indices: %s' % val_indices)
    print('train_indices: %s' % train_indices)
    x_train = x[train_indices]
    y_train = y[train_indices]
    mean = np.mean(x_train, axis=0)
    x_train -= mean
    x_test = x[test_indices] - mean
    y_test = y[test_indices]
    x_val = x[val_indices] - mean
    y_val = y[val_indices]
    dataset = (x_train, y_train, x_val, y_val)

    lung_model = LungSystem(FLAGS)

    with tf.Session() as sess:
        initialize_model(sess, lung_model, train_dir)
        logging.info("Evaluating model...")        
        train_accuracy, train_sens, train_spec = lung_model.accuracy(sess, x_train, y_train)
        train_hm = (2*train_sens*train_spec) / (train_sens + train_spec)
        logging.info("Training: accuracy = %s, sensitivity = %s, specificity = %s, HM = %s" % (train_accuracy, 
        	train_sens, train_spec, train_hm))
        val_accuracy, val_sens, val_spec = lung_model.accuracy(sess, x_val, y_val)
        val_hm = (2*val_sens*val_spec) / (val_sens + val_spec)
        logging.info("Validation: accuracy = %s, sensitivity = %s, specificity = %s, HM = %s" % (val_accuracy, 
        	val_sens, val_spec, val_hm))
        test_accuracy, test_sens, test_spec = lung_model.accuracy(sess, x_test, y_test)
        test_hm = (2*test_sens*test_spec) / (test_sens + test_spec)
        logging.info("Test: accuracy = %s, sensitivity = %s, specificity = %s, HM = %s" % (test_accuracy, 
        	test_sens, test_spec, test_hm))

if __name__ == "__main__":
    tf.app.run()
