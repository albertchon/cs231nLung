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

        self.data = tf.placeholder(tf.float32, shape=[None, FLAGS.image_height, FLAGS.image_width])
        
        self.labels = tf.placeholder(tf.int32, shape=[None, FLAGS.image_height, FLAGS.image_width])
        self.is_training = tf.placeholder(tf.bool)
        # ==== assemble pieces ====
        with tf.variable_scope("lung_sys"):
            self.setup_unet_system()
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


    
    def leaky_relu(self, x):
        return tf.maximum(x, self.FLAGS.leak*x)


    def setup_unet_system(self):
        inputs = tf.expand_dims(self.data, axis=-1) # ? * 512 * 512 * 1    
        
        conv1 = tf.layers.conv2d(inputs, 32, 3,  padding='same', activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer()) # ? * 512 * 512 * 32
        conv1 = tf.layers.dropout(conv1, rate=self.FLAGS.dropout, training=self.is_training)
        conv1 = tf.layers.conv2d(conv1, 32, 3, padding='same', activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer()) # ? * 512 * 512 * 32
        pool1 = tf.layers.max_pooling2d(conv1, 2, 2) # ? * 256 * 256 * 32

        conv2 = tf.layers.conv2d(pool1, 80, 3, padding='same', activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer()) # ? * 256 * 256 * 64
        conv2 = tf.layers.dropout(conv2, rate=self.FLAGS.dropout, training=self.is_training)
        conv2 = tf.layers.conv2d(conv2, 80, 3, padding='same', activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2) # ? * 128 * 128 * 64

        conv3 = tf.layers.conv2d(pool2, 160, 3, padding='same', activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer()) # ? * 128 * 128 * 128
        conv3 = tf.layers.dropout(conv3, rate=self.FLAGS.dropout, training=self.is_training)
        conv3 = tf.layers.conv2d(conv3, 160, 3, padding='same', activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        pool3 = tf.layers.max_pooling2d(conv3, 2, 2) # ? * 64 * 64 * 128

        #conv4 = tf.layers.conv2d(pool3, 256, 3, padding='same', activation=tf.nn.relu,
        #    kernel_initializer=tf.contrib.layers.xavier_initializer()) # ? * 64 * 64 * 512
        #conv4 = tf.layers.dropout(conv4, rate=self.FLAGS.dropout, training=self.is_training)
        #conv4 = tf.layers.conv2d(conv4, 256, 3, padding='same', activation=tf.nn.relu,
        #    kernel_initializer=tf.contrib.layers.xavier_initializer())
        #pool4 = tf.layers.max_pooling2d(conv4, 2, 2) # ? * 32 * 32 * 512

        conv5 = tf.layers.conv2d(pool3, 320, 3, padding='same', activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer()) # ? * 64 * 64 * 256
        conv5 = tf.layers.dropout(conv5, rate=self.FLAGS.dropout, training=self.is_training)
        conv5 = tf.layers.conv2d(conv5, 320, 3, padding='same', activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer())


        #resize6 = tf.image.resize_images(conv5, [64, 64]) # ? * 64 * 64 * 1024
        #up6 = tf.concat([resize6, conv4], axis=-1) # ? * 64 * 64 * 1536
        #conv6 = tf.layers.conv2d(up6, 256, 3, padding='same', activation=tf.nn.relu,
        #    kernel_initializer=tf.contrib.layers.xavier_initializer()) # ? * 64 * 64 * 512
        #conv6 = tf.layers.dropout(conv6, rate=self.FLAGS.dropout, training=self.is_training)
        #conv6 = tf.layers.conv2d(conv6, 256, 3, padding='same', activation=tf.nn.relu,
        #    kernel_initializer=tf.contrib.layers.xavier_initializer()) # ? * 64 * 64 * 512

        resize7 = tf.image.resize_images(conv5, [64, 64]) # ? * 128 * 128 * 256
        up7 = tf.concat([resize7, conv3], axis=-1) # ? * 128 * 128 * 384
        conv7 = tf.layers.conv2d(up7, 160, 3, padding='same', activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer()) # ? * 128 * 128 * 128
        conv7 = tf.layers.dropout(conv7, rate=self.FLAGS.dropout, training=self.is_training)
        conv7 = tf.layers.conv2d(conv7, 160, 3, padding='same', activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer()) # ? * 128 * 128 * 128

        resize8 = tf.image.resize_images(conv7, [128, 128]) # ? * 256 * 256 * 128
        up8 = tf.concat([resize8, conv2], axis=-1) # ? * 256 * 256 * 192
        conv8 = tf.layers.conv2d(up8, 80, 3, padding='same', activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer()) # ? * 256 * 256 * 64
        conv8 = tf.layers.dropout(conv8, rate=self.FLAGS.dropout, training=self.is_training)
        conv8 = tf.layers.conv2d(conv8, 80, 3, padding='same', activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer()) # ? * 256 * 256 * 64

        resize9 = tf.image.resize_images(conv8, [256, 256]) # ? * 512 * 512 * 64
        up9 = tf.concat([resize9, conv1], axis=-1) # ? * 512 * 512 * 96
        conv9 = tf.layers.conv2d(up9, 32, 3, padding='same', activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer()) # ? * 512 * 512 * 32
        conv9 = tf.layers.dropout(conv9, rate=self.FLAGS.dropout, training=self.is_training)
        conv9 = tf.layers.conv2d(conv9, 32, 3, padding='same', activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer()) # ? * 512 * 512 * 32

        self.predictions = tf.layers.conv2d(conv9, 2, 1, padding='same',
            kernel_initializer=tf.contrib.layers.xavier_initializer()) # ? * 512 * 512 * 2

        #self.predictions = tf.squeeze(conv10, axis=-1)
        
        
        #conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(inputs)
        #conv1 = Dropout(0.2)(conv1)
        #conv1 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv1)
        #pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        # conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool1)
        # conv2 = Dropout(0.2)(conv2)
        # conv2 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv2)
        # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        # conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool2)
        # conv3 = Dropout(0.2)(conv3)
        # conv3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv3)
        # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        # conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool3)
        # conv4 = Dropout(0.2)(conv4)
        # conv4 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv4)
        # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        # conv5 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(pool4)
        # conv5 = Dropout(0.2)(conv5)
        # conv5 = Convolution2D(1024, 3, 3, activation='relu', border_mode='same')(conv5)

        # up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
        # conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(up6)
        # conv6 = Dropout(0.2)(conv6)
        # conv6 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv6)

        # up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
        # conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up7)
        # conv7 = Dropout(0.2)(conv7)
        # conv7 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv7)

        # up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
        # conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up8)
        # conv8 = Dropout(0.2)(conv8)
        # conv8 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv8)

        # up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
        # conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up9)
        # conv9 = Dropout(0.2)(conv9)
        # conv9 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv9)

        # conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)




    # hello
    # change the loss function
    def dice_coef(self, y_true, y_pred):
        smooth = 1.
        intersection = tf.reduce_sum(y_true * y_pred)
        return 1.-(2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            # self.start_answer (N)
            #self.loss = self.dice_coef(self.labels, self.predictions)
            #labels = tf.reshape(self.labels, [-1])
            #predictions = tf.reshape(self.labels, [-1, 2])   
            weights = (self.FLAGS.weight_one-1)*tf.to_float(self.labels) + 1
            self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.predictions, weights=weights)    

    def optimize(self, session, x_train, y_train):
        
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x
        input_feed[self.data] = x_train
        input_feed[self.labels] = y_train
        input_feed[self.is_training] = True        

        #print(x_train.shape)
        # grad_norm, param_norm
        # merged = tf.summary.merge_all()
        
        output_feed = [self.updates, self.loss]
        
        

        outputs = session.run(output_feed, input_feed)
        # train_writer.add_summary(outputs[2], 3)
        return outputs

    def test(self, session, x, y, sample=1024):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        sample_indices = list(range(len(x)))
        num_samples = min(len(x), sample)
        random.shuffle(sample_indices)
        sample_indices = sample_indices[:num_samples]
        x = x[sample_indices]
        y = y[sample_indices]
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

            input_feed[self.data] = x_batch
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
        outputs = np.zeros((x.shape[0], 256, 256))        
        for b_end in range(1, num_minibatches + 1):
            start = batch_indices[b_end-1]
            end = batch_indices[b_end]
            x_batch = x[start:end]

            input_feed = {}

            # fill in this feed_dictionary like:
            # input_feed['valid_x'] = valid_x

            input_feed[self.data] = x_batch
            input_feed[self.is_training] = False          

            output_feed = [self.predictions]
            outputs[start:end] = tf.nn.softmax(session.run(output_feed, input_feed)[0]).eval(session=session).astype(np.float16)[:,:,:,1]

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
        #print('X TRAIN SHAPE: ', x_train.shape)
        #print('y TRAIN SHAPE: ', y_train.shape)

        best_val_loss = self.FLAGS.best_val_loss
        train_losses = []
        val_losses = []
        # train_writer = tf.summary.FileWriter('summaries', session.graph) # this is for training only
        for e in range(self.FLAGS.epochs):
            logging.info("="*80)
            logging.info("Epoch %s of %s" % (e+1, self.FLAGS.epochs))
            #num_batches = int(math.ceil(x_train.shape[0]/self.FLAGS.batch_size))
            epoch_train_losses = []
            indices = list(range(len(x_train)))
            random.shuffle(indices)            
            batch_indices = [0]
            index = 0
            while (True):
                index += 1
                if index == x_train.shape[0]:
                    if batch_indices[-1] != index:
                        batch_indices.append(index)
                    break
                if index % self.FLAGS.batch_size == 0:
                    batch_indices.append(index)
            num_minibatches = len(batch_indices) - 1
            
            for b_end in range(1, num_minibatches + 1):
                start = batch_indices[b_end-1]
                end = batch_indices[b_end]
                x_train_batch = x_train[indices[start:end]]
                y_train_batch = y_train[indices[start:end]]
                _, train_loss = self.optimize(session, x_train_batch, y_train_batch)
                epoch_train_losses.append(train_loss)
                logging.info("batch %s/%s training loss: %s" % (b_end, num_minibatches, train_loss))
                
                if b_end % 100 == 0:
                    logging.info("-"*80)
                    logging.info("Validating epoch %s of %s" % (e+1, self.FLAGS.epochs))
                    # train_accuracy, train_sens, train_spec = self.accuracy(session, x_train, y_train)
                    # train_hm = (2*train_sens*train_spec) / (train_sens + train_spec)


                    train_loss = self.test(session, x_train, y_train, 128)[0]
                    logging.info("Training loss: %s" % (train_loss))
                    # logging.info("Training: accuracy = %s, sensitivity = %s, specificity = %s" % (train_accuracy, 
                    #     train_sens, train_spec))
                    '''
                    # val_accuracy, val_sens, val_spec = self.accuracy(session, x_val, y_val)
                    val_loss = self.test(session, x_val, y_val, 128)[0]
                    val_losses.append(val_loss)
                    logging.info("Validation loss: %s" % (val_loss))
                    # logging.info("Validation: accuracy = %s, sensitivity = %s, specificity = %s" % (val_accuracy, 
                    #     val_sens, val_spec))
                    '''
                    
                    
            logging.info("-"*80)
            train_losses += epoch_train_losses
            
            
            
            logging.info("Validating epoch %s of %s" % (e+1, self.FLAGS.epochs))
            # train_accuracy, train_sens, train_spec = self.accuracy(session, x_train, y_train)
            # train_hm = (2*train_sens*train_spec) / (train_sens + train_spec)
            
            
            train_loss = self.test(session, x_train, y_train)[0]
            logging.info("Training loss: %s" % (train_loss))
            # logging.info("Training: accuracy = %s, sensitivity = %s, specificity = %s" % (train_accuracy, 
            #     train_sens, train_spec))
            
            # val_accuracy, val_sens, val_spec = self.accuracy(session, x_val, y_val)
            val_loss = self.test(session, x_val, y_val)[0]
            val_losses.append(val_loss)
            logging.info("Validation loss: %s" % (val_loss))
            # logging.info("Validation: accuracy = %s, sensitivity = %s, specificity = %s" % (val_accuracy, 
            #     val_sens, val_spec))
            
            if val_loss <= best_val_loss:
                logging.info("NEW BEST VALIDATION LOSS: %s, SAVING!" % (val_loss))
                best_val_loss = val_loss
                self.saver.save(session, train_dir + 'model.weights')  
            logging.info("CURRENT BEST VALIDATION LOSS: %s" % (best_val_loss))
            
            
            
            

