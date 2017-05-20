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

        self.images = tf.placeholder(tf.float32, shape=[None, FLAGS.num_slices, FLAGS.image_height, FLAGS.image_width])
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
        x = tf.reshape(self.images, [-1, self.FLAGS.num_slices, self.FLAGS.image_height, self.FLAGS.image_width, 1])

        b1 = tf.layers.batch_normalization(x, training=self.is_training)

        c1 = tf.layers.conv3d(b1, filters=16, kernel_size=[3, 3, 3], strides=[1,1,1], padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        #r1 = tf.nn.relu(c1)
        r1 = tf.maximum(c1, self.FLAGS.leak*c1)
        
        m1 = tf.layers.max_pooling3d(r1, pool_size=2, strides=2, padding='valid')

        b2 = tf.layers.batch_normalization(m1, training=self.is_training)

        c2 = tf.layers.conv3d(b2, filters=16, kernel_size=[3, 3, 3], strides=[1, 1, 1], padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer())

        #r2 = tf.nn.relu(c2)
        r2 = tf.maximum(c2, self.FLAGS.leak*c2)
        

        m2 = tf.layers.max_pooling3d(r2, pool_size=2, strides=2, padding='valid')

        c3 = tf.layers.conv3d(m2, filters=1, kernel_size=[2,2,2], strides=[1,1,1], padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer())
        c3 = tf.reshape(c3, [-1, 16 * 32 * 32])
        #r3 = tf.nn.relu(c3)
        r3 = tf.maximum(c3, self.FLAGS.leak*c3)

        aff1_W = tf.get_variable('aff1_W', shape=[16 * 32 * 32, 2048],
            initializer=tf.contrib.layers.xavier_initializer())
        aff1_b = tf.get_variable('aff1_b', shape=[2048])
        a1 = tf.matmul(r3, aff1_W) + aff1_b
        #a1 = tf.nn.relu(a1)
        a1 = tf.maximum(a1, self.FLAGS.leak*a1)
        aff2_W = tf.get_variable('aff2_W', shape=[2048, 2],
            initializer=tf.contrib.layers.xavier_initializer())
        aff2_b = tf.get_variable('aff2_b', shape=[2])

        self.predictions = tf.matmul(a1, aff2_W) + aff2_b

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

        #print(x_train.shape)
        # grad_norm, param_norm
        output_feed = [self.updates, self.loss]


        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, x, y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        batch_indices = [0]
        index = 0
        while (True):
            index += 1
            if index == x.shape[0]:
                if batch_indices[-1] != index:
                    batch_indices.append(index)
                break
            if index % self.FLAGS.batch_size == 0:
                batch_indices.append(index)
        num_minibatches = len(batch_indices) - 1
        losses = []
        for b_end in range(1, num_minibatches + 1):
            start = batch_indices[b_end-1]
            end = batch_indices[b_end]
            x_batch = x[start:end]
            y_batch = y[start:end]  

            input_feed = {}

            # fill in this feed_dictionary like:
            # input_feed['valid_x'] = valid_x

            input_feed[self.images] = x_batch
            input_feed[self.labels] = y_batch
            input_feed[self.is_training] = False          

            output_feed = [self.loss]

            outputs = session.run(output_feed, input_feed)
            losses += outputs # TODO: consider weighing last

        return [sum(losses)/float(len(losses))]

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

        batch_indices = [0]
        index = 0
        while (True):
            index += 1
            if index == x.shape[0]:
                if batch_indices[-1] != index:
                    batch_indices.append(index)
                break
            if index % self.FLAGS.batch_size == 0:
                batch_indices.append(index)
        num_minibatches = len(batch_indices) - 1
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for b_end in range(1, num_minibatches + 1):
            start = batch_indices[b_end-1]
            end = batch_indices[b_end]
            x_batch = x[start:end]
            y_batch = y[start:end]

            outputs = self.predict(session, x_batch)
            probabilities = outputs[0]
            pred = np.argmax(probabilities, axis=1)
            print(probabilities)
            TP += np.sum(y_batch[y_batch == 1] == pred[y_batch == 1])
            TN += np.sum(y_batch[y_batch == 0] == pred[y_batch == 0])
            FP += np.sum(y_batch[y_batch == 0] != pred[y_batch == 0])
            FN += np.sum(y_batch[y_batch == 1] != pred[y_batch == 1])

        acc = float(TP + TN) / (TP + TN + FP + FN)
        sens = float(TP) / (TP + FN)
        spec = float(TN) / (TN + FP)
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