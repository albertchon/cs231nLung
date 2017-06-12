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

tf.app.flags.DEFINE_float("best_val_loss", float('inf'), "best val loss so far")
tf.app.flags.DEFINE_string("model", 'unet', "Type of model to use: linear or cnn or simplecnn")
tf.app.flags.DEFINE_integer("epochs", 999999, "number of epochs")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("leak", 0.01, "Leakiness")
tf.app.flags.DEFINE_float("dropout", 0.2, "dropout prob")
tf.app.flags.DEFINE_integer("image_depth", 32, "width of each slice in pixels")
tf.app.flags.DEFINE_integer("image_height", 64, "height of each slice in pixels")
tf.app.flags.DEFINE_integer("image_width", 64, "width of each slice in pixels")
tf.app.flags.DEFINE_integer("batch_size", 8, "Batch size to use during training.")
tf.app.flags.DEFINE_float("train_size", 0.9, "Size of train set")
tf.app.flags.DEFINE_float("val_size", 0.1, "Size of val set")
#tf.app.flags.DEFINE_float("weight_one", 2730.5211584, "ones label multiplier")

FLAGS = tf.app.flags.FLAGS
train_dir = './weights/%s/' % (FLAGS.model)

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
    DATA_FOLDER = 'compiledvoxels/'
    #LABEL_FOLDER = 'masks/labels/'
    patient_data = os.listdir(DATA_FOLDER)
    #patient_labels = os.listdir(LABEL_FOLDER)

    data = {}
    labels = {}

    #storing images
    for p in range(len(patient_data)):
        path = DATA_FOLDER + patient_data[p]
        image = np.load(path)
        pat_id = patient_data[p][:-14]
        data[pat_id] = image
        labels[pat_id] = random.randint(0,1) # generate a random label for now

    #storing labels
    # for p in range(len(patient_labels)):
    #     path = LABEL_FOLDER + patient_labels[p]
    #     image = np.load(path)
    #     pat_id = patient_labels[p][:-16]
    #     labels[pat_id] = image
    '''
    lines = [line for line in open('stage2_solution.csv')]
    for i in range(1, len(lines)):
        data = lines[i].split(',')
        labels['2-'+data[0]] = int(data[1])
    '''

    # x = np.zeros((0, FLAGS.image_height, FLAGS.image_width), dtype=np.float32)
    # y = np.zeros((0, FLAGS.image_height, FLAGS.image_width), dtype=np.bool)
    x = []
    y = []
    
    patient_keys = sorted(list(data.keys()))
    for i in range(len(patient_keys)):
        print(i)
        patient = patient_keys[i]
        x.append(data[patient])
        y.append(labels[patient])

        # x = np.concatenate([x, data[patient]])
        # y = np.concatenate([y, labels[patient]])

    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    print(x.shape)
    print(y.shape)
    
    print('as arrayed')
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
    print('indices: %s' % indices[:10])
    #print('test_indices: %s' % test_indices)
    print('val_indices: %s' % val_indices[:10])
    print('train_indices: %s' % train_indices[:10])
    x_train = x[train_indices]
    y_train = y[train_indices]
    if starting:
        indices = list(range(num_train))
        x_train = x_train[:32]
        y_train = y_train[:32]
    x_val = x[val_indices]
    y_val = y[val_indices]
    dataset = (x_train, y_train, x_val, y_val)
    print (np.sum(x_train[10]))
    print (np.sum(x_val[10]))

    lung_model = LungSystem(FLAGS)

    with tf.Session() as sess:
        initialize_model(sess, lung_model, train_dir)
        lung_model.train(sess, dataset, train_dir)

if __name__ == "__main__":
    tf.app.run()


