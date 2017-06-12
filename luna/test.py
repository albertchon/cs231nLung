import os
import sys

import tensorflow as tf

import numpy as np
from model256 import LungSystem
from os.path import join as pjoin
import random


import logging
starting=False
logging.basicConfig(level=logging.INFO)


tf.app.flags.DEFINE_float("best_val_loss", float('inf'), "best val loss so far")
tf.app.flags.DEFINE_string("model", 'unet', "Type of model to use: linear or cnn or simplecnn")
tf.app.flags.DEFINE_integer("epochs", 999999, "number of epochs")
tf.app.flags.DEFINE_float("learning_rate", 0.0001, "Learning rate.")
tf.app.flags.DEFINE_float("leak", 0.01, "Leakiness")
tf.app.flags.DEFINE_float("dropout", 0.2, "dropout prob")
tf.app.flags.DEFINE_integer("image_height", 256, "height of each slice in pixels")
tf.app.flags.DEFINE_integer("image_width", 256, "width of each slice in pixels")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_float("train_size", 0.9, "Size of train set")
tf.app.flags.DEFINE_float("val_size", 0.1, "Size of val set")
tf.app.flags.DEFINE_float("weight_one", 2640.68737846, "ones label multiplier")
tf.app.flags.DEFINE_integer("skip", 1480, "ones label multiplier")

FLAGS = tf.app.flags.FLAGS
train_dir = './weights/%s256/' % (FLAGS.model)

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
    DATA_FOLDER = 'kaggle-lung-masks/'
    OUTPUT_FOLDER = 'kaggle-lung-unet-outputs/'
    patient_data = os.listdir(DATA_FOLDER)
    patient_data.sort()


    lung_model = LungSystem(FLAGS)

    with tf.Session() as sess:
        initialize_model(sess, lung_model, train_dir)
        for p in range(len(patient_data)):
            
            if p < FLAGS.skip or p >= FLAGS.skip+50:
                continue
            print(p)
            path = DATA_FOLDER + patient_data[p]
            x = np.load(path)
            pat_id = patient_data[p][:-4]    
            logging.info("Evaluating model on %s" % pat_id) 
            y = lung_model.predict(sess, x).astype(np.float16)
            np.save(OUTPUT_FOLDER + pat_id, y)
            

if __name__ == "__main__":
    tf.app.run()
