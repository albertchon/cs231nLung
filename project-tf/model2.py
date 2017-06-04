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
        self.prelu_counter = 0

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
            elif FLAGS.model == 'simplecnn':
                self.setup_simplecnn()
            elif FLAGS.model == 'asu':
                self.setup_asu()                
            elif FLAGS.model == 'googlenet':
                self.setup_googlenet()
            elif FLAGS.model == 'naive':
                self.setup_naive()
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
    def _activation_summary(x):  #from https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
        """Helper to create summaries for activations.
          Creates a summary that provides a histogram of activations.
          Creates a summary that measures the sparsity of activations.
          Args:
            x: Tensor
          Returns:
            nothing
          """
        # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
        # session. This helps the clarity of presentation on tensorboard.
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
    
    def leaky_relu(self, x):
        return tf.maximum(x, self.FLAGS.leak*x)
    
    def setup_naive(self):
        pred = tf.zeros_like(self.images)
        b = tf.get_variable('b', shape=[2])
        self.predictions = pred[:,0,0,:2]*b
    
    def setup_linear_system(self):
        x = tf.reshape(self.images, shape=[-1, self.FLAGS.num_slices*self.FLAGS.image_height*self.FLAGS.image_width])
        W = tf.get_variable('W', shape=[self.FLAGS.num_slices*self.FLAGS.image_height*self.FLAGS.image_width, 2], 
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', shape=[2])
        self.predictions = tf.matmul(x, W) + b
   
    def prelu(self, _x):
        alphas = tf.get_variable('alpha'+str(self.prelu_counter), _x.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pos = tf.nn.relu(_x)
        neg = alphas * (_x - abs(_x)) * 0.5
        self.prelu_counter += 1
        return pos + neg
    
    def inception_a(self, x, filters, increase_layers=True):
        if increase_layers:
            filters //= 2
        else:
            filters //= 4
        c1 = tf.layers.conv3d(x, filters=filters, kernel_size=[1,1,1], strides=[1,1,1], padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer())
        #c1 = tf.layers.batch_normalization(c1, training=self.is_training)
        c1 = self.leaky_relu(c1)
        c3 = tf.layers.conv3d(x, filters=2*filters, kernel_size=[3,3,3], strides=[1,1,1], padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer())        
        #c3 = tf.layers.batch_normalization(c3, training=self.is_training)
        c3 = self.leaky_relu(c3)
        
        c5 = tf.layers.conv3d(x, filters=filters, kernel_size=[5,5,5], strides=[1,1,1], padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer())        
        #c5 = tf.layers.batch_normalization(c5, training=self.is_training)
        c5 = self.leaky_relu(c5)
        return tf.concat([c1, c3, c5], axis=-1)
        

    def setup_googlenet(self):
        x = tf.reshape(self.images, [-1, self.FLAGS.num_slices, self.FLAGS.image_height, self.FLAGS.image_width, 1])
        x = tf.layers.batch_normalization(x, training=self.is_training)
        #? x 64 x 128 x 128 x 1
        
        c1 = tf.layers.conv3d(x, filters=32, kernel_size=[3,3,3], strides=[2,2,2], padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer())
        #c1 = tf.layers.batch_normalization(c1, training=self.is_training)
        c1 = self.leaky_relu(c1)
        #? x 32 x 64 x 64 x 32
        m1 = tf.layers.max_pooling3d(c1, pool_size=2, strides=2, padding='valid')
        #? x 16 x 32 x 32 x 32
        c2 = tf.layers.conv3d(m1, filters=64, kernel_size=[3,3,3], strides=[1,1,1], padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer())
        #? x 16 x 32 x 32 x 64
        #c2 = tf.layers.batch_normalization(c2, training=self.is_training)
        c2 = self.leaky_relu(c2)
        m2 = tf.layers.max_pooling3d(c2, pool_size=2, strides=2, padding='valid')
        #? x 8 x 16 x 16 x 64
        
        i1 = self.inception_a(m2, 64)
        #? x 8 x 16 x 16 x 128
        #i2 = self.inception_a(i1, 128, False)
        #? x 8 x 16 x 16 x 128
        m3 = tf.layers.max_pooling3d(i1, pool_size=2, strides=2, padding='valid')
        #? x 4 x 8 x 8 x 128
        i3 = self.inception_a(m3, 128)
        #? x 4 x 8 x 8 x 256
        #i4 = self.inception_a(i3, 256, False)
        #? x 4 x 8 x 8 x 256
        #m4 = tf.layers.max_pooling3d(i3, pool_size=2, strides=2, padding='valid')  
        #? x 2 x 4 x 4 x 256
        #i5 = self.inception_a(m4, 256)
        #? x 2 x 4 x 4 x 512
        #i6 = self.inception_a(i5, 512, False)
        #? x 2 x 4 x 4 x 512
        a1 = tf.layers.average_pooling3d(i3, pool_size = [4,8,8], strides=[1,1,1])
        #? x 1 x 1 x 1 x 256
        a1 = tf.reshape(a1, [-1, 256])
        a1 = tf.layers.dropout(a1, rate=self.FLAGS.dropout, training=self.is_training)
        self.predictions = tf.layers.dense(a1, 2)
        
    
    
    def setup_asu(self):
        # http://www.public.asu.edu/~ranirudh/papers/spie2016.pdf
        x = tf.reshape(self.images, [-1, self.FLAGS.num_slices, self.FLAGS.image_height, self.FLAGS.image_width, 1])
        #? x 64 x 128 x 128 x 1
        x = tf.layers.average_pooling3d(x, pool_size = [4,4,4], strides=[4,4,4])
        #? x 16 x 32 x 32 x 1
        x = tf.layers.batch_normalization(x, training=self.is_training)
        
        c1 = tf.layers.conv3d(x, filters=64, kernel_size=[3,3,3], strides=[2,2,2], padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             activation=self.leaky_relu)
        #? x 8 x 16 x 16 x 64
        tf.summary.histogram('conv1', c1)
        m1 = tf.layers.max_pooling3d(c1, pool_size=2, strides=2, padding='valid')
        #? x 4 x 8 x 8 x 64
        
        c2 = tf.layers.conv3d(m1, filters=64, kernel_size=[3,3,3], strides=[1,1,1], padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             activation=self.leaky_relu)
        tf.summary.histogram('conv2', c2)
        #? x 4 x 8 x 8 x 64
        m2 = tf.layers.max_pooling3d(c2, pool_size=2, strides=2, padding='valid')
        #? x 2 x 4 x 4 x 64
        c3 = tf.layers.conv3d(m2, filters=128, kernel_size=[3,3,3], strides=[2,2,2], padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             activation=self.leaky_relu)
        tf.summary.histogram('conv3', c3)
        #? x 1 x 2 x 2 x 128
        c4 = tf.layers.conv3d(c3, filters=256, kernel_size=[3,3,3], strides=[1,1,1], padding='same',
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                             activation=self.leaky_relu)
        tf.summary.histogram('conv4', c4)
        #? x 1 x 2 x 2 x 256
        c4 = tf.reshape(c4, (-1, 1024))
        a1 = tf.layers.dense(c4, 512, activation=self.leaky_relu)
        tf.summary.histogram('affine1', a1)
        a2 = tf.layers.dense(a1, 256, activation=self.leaky_relu)
        tf.summary.histogram('affine2', a2)
        self.predictions = tf.layers.dense(a2,2)
        
        
        
        
        
        
        
   
    def setup_simplecnn(self):
        #? x 64 x 128 x 128 x 1
        x = tf.reshape(self.images, [-1, self.FLAGS.num_slices, self.FLAGS.image_height, self.FLAGS.image_width, 1])

        
        b1 = tf.layers.batch_normalization(x, training=self.is_training)

        
        c1 = tf.layers.conv3d(b1, filters=8, kernel_size=[3, 3, 3], strides=[1,1,1], padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=self.leaky_relu)
        #? x 64 x 128 x 128 x 8
        
        tf.summary.histogram('conv1', c1)
        m1 = tf.layers.max_pooling3d(c1, pool_size=2, strides=2, padding='valid')
        #? x 32 x 64 x 64 x 8
        b2 = tf.layers.batch_normalization(m1, training=self.is_training)
        
        
        c2 = tf.layers.conv3d(b2, filters=16, kernel_size=[3, 3, 3], strides=[1,1,1], padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer(), activation=self.leaky_relu)
        #? x 32 x 64 x 64 x 16
        tf.summary.histogram('conv2', c2)
        m2 = tf.layers.max_pooling3d(c2, pool_size=2, strides=2, padding='valid')
        #? x 16 x 32 x 32 x 16
        
        m2 = tf.reshape(m2, (-1, 16*32*32*16))
        
        a1 = tf.layers.dense(m2, 256, activation=self.leaky_relu)
        tf.summary.histogram('affine1', a1)
        a1 = tf.layers.dropout(a1, rate=self.FLAGS.dropout, training=self.is_training)
        
        self.predictions = tf.layers.dense(a1, 2)
                         

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
            self.loss = tf.losses.sparse_softmax_cross_entropy(self.labels, self.predictions)
            tf.summary.scalar('loss', self.loss)

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
        # merged = tf.summary.merge_all()
        
        output_feed = [self.updates, self.loss]
        
        

        outputs = session.run(output_feed, input_feed)
        return outputs

    def test(self, session, x, y, train_writer, e, merged, addBool):
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
    
            
            
            
            output_feed = [self.loss, merged]

            outputs = session.run(output_feed, input_feed)
            if addBool:
                train_writer.add_summary(outputs[1], e)
            losses.append(outputs[0]) # TODO: consider weighing last
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
            # print(probabilities)
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
        print('X TRAIN SHAPE: %s' % x_train.shape[0])
        print('y TRAIN SHAPE: %s' % y_train.shape[0])

        best_train_loss = self.FLAGS.best_train_loss
        best_val_loss = self.FLAGS.best_val_loss

        train_losses = []
        val_losses = []
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('summaries', session.graph) # this is for training only
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
                logging.info("batch %s/%s training loss: %s" % (i+1, num_batches , train_loss))
            logging.info("-"*80)
            train_losses += epoch_train_losses
            
            
            
            logging.info("Validating epoch %s of %s" % (e+1, self.FLAGS.epochs))
            train_accuracy, train_sens, train_spec = self.accuracy(session, x_train, y_train)
            train_hm = (2*train_sens*train_spec) / (train_sens + train_spec)
            
            
            train_loss = self.test(session, x_train, y_train, train_writer, e, merged, True)[0]
            
            logging.info("Training loss: %s" % (train_loss))
            logging.info("Training: accuracy = %s, sensitivity = %s, specificity = %s" % (train_accuracy, 
                train_sens, train_spec))

            val_accuracy, val_sens, val_spec = self.accuracy(session, x_val, y_val)
            val_loss = self.test(session, x_val, y_val, train_writer, e, merged, False)[0]
            val_losses.append(val_loss)
            logging.info("Validation loss: %s" % (val_loss))
            logging.info("Validation: accuracy = %s, sensitivity = %s, specificity = %s" % (val_accuracy, 
                val_sens, val_spec))
            if self.FLAGS.save_best_train_loss:
                if train_loss < best_train_loss:
                    logging.info("NEW BEST TRAIN LOSS: %s, SAVING!" % (train_loss))
                    best_train_loss = train_loss
                    self.saver.save(session, train_dir + 'model.weights')  
                logging.info("CURRENT BEST TRAIN LOSS: %s" % (best_train_loss))
            else:                
                if val_loss < best_val_loss:
                    logging.info("NEW BEST VAL LOSS: %s, SAVING!" % (val_loss))
                    best_val_loss = val_loss
                    self.saver.save(session, train_dir + 'model.weights')  
                logging.info("CURRENT BEST VAL LOSS: %s" % (best_val_loss))