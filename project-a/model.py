from __future__ import absolute_import
from __future__ import print_function

import time
import logging

import math
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
import os
import random



logging.basicConfig(level=logging.INFO)

class LungSystem(object):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

        # ==== set up placeholder tokens ========
        # self.inputs_placeholder = tf.placeholder(tf.float32, shape=(None, self.FLAGS.output_size, self.FLAGS.embedding_size), name="x")
        # self.paragraph = tf.placeholder(shape=[None, self.FLAGS.output_size])

        self.images = tf.placeholder(tf.float32, shape=[None, FLAGS.image_height, FLAGS.image_width, FLAGS.num_slices])
        self.labels = tf.placeholder(tf.int32, shape=[None])
        self.is_training = tf.placeholder(tf.bool)
        # ==== assemble pieces ====
        with tf.variable_scope("lung_sys"):
            if FLAGS.model == 'linear':
                self.setup_linear_system()
            else:
                self.setup_cnn_system()
            print('setting up loss')
            self.setup_loss()   

        # ==== set up training/updating procedure ====
        # global_step = tf.Variable(0, trainable=False)
        # learning_rate = tf.train.polynomial_decay(self.FLAGS.learning_rate, global_step, dec_step, 
        #     end_learning_rate=self.FLAGS.end_learning_rate, power=1.0)
        # decay = self.FLAGS.decay
        # learning_rate = tf.maximum(self.FLAGS.min_learning_rate, tf.train.exponential_decay(self.FLAGS.learning_rate, global_step, self.FLAGS.global_step, FLAGS.decay))


        print('setting up optimizer')
        self.optimizer = tf.train.AdamOptimizer(self.FLAGS.learning_rate)

        print('computing gradients')
        self.updates = self.optimizer.minimize(self.loss)
        print('setting up saver')
        self.saver = tf.train.Saver()

    def setup_linear_system(self):
        x = tf.reshape(self.images, shape=[-1, self.FLAGS.num_slices*self.FLAGS.image_height*self.FLAGS.image_width])
        W = tf.get_variable('W', shape=[self.FLAGS.num_slices*self.FLAGS.image_height*self.FLAGS.image_width, 2], 
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', shape=[2])
        self.predictions = tf.matmul(x, W) + b

    def setup_cnn_system(self):
        """
        model goes here
        """
        c1 = tf.layers.conv2d(self.images, filters=self.FLAGS.conv1_filters, kernel_size=[3,3], padding='same', 
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        print(c1)
        b1 = tf.layers.batch_normalization(c1, training=self.is_training)
        print(b1)
        r1 = tf.nn.relu(b1)
        print(r1)
        
        m1 = tf.layers.max_pooling2d(r1, pool_size=[2,2], strides=2)
        print(m1)
        
        c2 = tf.layers.conv2d(m1, filters=self.FLAGS.conv2_filters, kernel_size=[3,3], padding='same', 
            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        print(c2)
        b2 = tf.layers.batch_normalization(c2, training=self.is_training)
        print(b2)
        r2 = tf.nn.relu(b2)
        print(r2)

        m2 = tf.layers.max_pooling2d(r2, pool_size=[2,2], strides=2)
        print(m2)

        m2 = tf.reshape(m2, shape=[-1, self.FLAGS.conv2_filters*self.FLAGS.image_height/4*self.FLAGS.image_width/4])
        print(m2)

        aff1_W = tf.get_variable('aff1_W', shape=[self.FLAGS.conv2_filters*self.FLAGS.image_width/4*self.FLAGS.image_width/4, self.FLAGS.aff_size], 
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        aff1_b = tf.get_variable('aff1_b', shape=[self.FLAGS.aff_size])

        a1 = tf.nn.relu(tf.matmul(m2, aff1_W) + aff1_b)
        print(a1)

        aff2_W = tf.get_variable('aff2_W', shape=[self.FLAGS.aff_size, 2], 
            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        aff2_b = tf.get_variable('aff2_b', shape=[2])

        self.predictions = tf.matmul(a1, aff2_W) + aff2_b
        print(self.predictions)

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            # self.start_answer (N)
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predictions, labels=self.labels))      

    def optimize(self, session, x_train, y_train):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x
        input_feed[self.images] = x_train
        input_feed[self.labels] = y_train
        input_feed[self.is_training] = True        

        # grad_norm, param_norm
        output_feed = [self.updates, self.loss]


        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, x_val, y_val):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        input_feed[self.images] = x_val
        input_feed[self.labels] = y_val
        input_feed[self.is_training] = False          

        output_feed = [self.loss]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def predict(self, session, x):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        input_feed[self.images] = x
        input_feed[self.is_training] = False          

        output_feed = [self.predictions]

        outputs = session.run(output_feed, input_feed)

        return outputs

    def accuracy(self, session, x, y):
        outputs = self.predict(session, x)
        probabilities = outputs[0]
        pred = np.argmax(probabilities, axis=1)
        TP = float(np.sum(y[y == 1] == pred[y == 1]))
        TN = float(np.sum(y[y == 0] == pred[y == 0]))
        FP = float(np.sum(y[y == 0] != pred[y == 0]))
        FN = float(np.sum(y[y == 1] != pred[y == 1])) 
        acc = (TP + TN) / (TP + TN + FP + FN)
        sens = TP / (TP + FN)
        spec = TN / (TN + FP)
        return acc, sens, spec



    def train(self, session, dataset, train_dir):
        logging.info("training:")
        x_train, y_train, x_val, y_val = dataset

        best_val_hm = self.FLAGS.best_val_hm
        train_losses = []
        val_losses = []
        for e in range(self.FLAGS.epochs):
            logging.info("="*80)
            logging.info("Epoch %s of %s" % (e+1, self.FLAGS.epochs))
            num_batches = int(math.ceil(x_train.shape[0]/self.FLAGS.batch_size))
            epoch_train_losses = []
            for i in range(num_batches):
                batch_indices = random.sample(range(len(x_train)), self.FLAGS.batch_size)
                x_train_batch = x_train[batch_indices]
                y_train_batch = y_train[batch_indices]
                _, train_loss = self.optimize(session, x_train_batch, y_train_batch)
                epoch_train_losses.append(train_loss)
                logging.info("batch %s/%s training loss: %s" % (i+1, num_batches, train_loss))
            logging.info("-"*80)
            train_losses += epoch_train_losses
            logging.info("Validating epoch %s of %s" % (e+1, self.FLAGS.epochs))
            train_accuracy, train_sens, train_spec = self.accuracy(session, x_train, y_train)
            train_hm = (2*train_sens*train_spec) / (train_sens + train_spec)
            logging.info("Training loss: %s" % (np.mean(np.asarray(epoch_train_losses))))
            logging.info("Training: accuracy = %s, sensitivity = %s, specificity = %s, HM = %s" % (train_accuracy, 
            	train_sens, train_spec, train_hm))
            val_accuracy, val_sens, val_spec = self.accuracy(session, x_val, y_val)
            val_hm = (2*val_sens*val_spec) / (val_sens + val_spec)
            val_loss = self.test(session, x_val, y_val)[0]
            val_losses.append(val_loss)
            logging.info("Validation loss: %s" % (val_loss))
            logging.info("Validation: accuracy = %s, sensitivity = %s, specificity = %s, HM = %s" % (val_accuracy, 
            	val_sens, val_spec, val_hm))
            if val_hm > best_val_hm:
                logging.info("NEW BEST VALIDATION HM: %s, SAVING!" % (val_hm))
                best_val_hm = val_hm
                self.saver.save(session, train_dir + 'model.weights')  
            logging.info("CURRENT BEST VALIDATION HM: %s" % (best_val_hm))