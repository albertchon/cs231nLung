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

        self.data = tf.placeholder(tf.float32, shape=[None, FLAGS.image_depth, FLAGS.image_height, FLAGS.image_width])
        self.labels = tf.placeholder(tf.int32, shape=[None])
        self.is_training = tf.placeholder(tf.bool)
        # ==== assemble pieces ====
        with tf.variable_scope("lung_sys"):
            self.setup_nodule_classifier()
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

    def setup_nodule_classifier(self):
        inputs = tf.expand_dims(self.data, axis=-1)
        # ? * 32 * 64 * 64 * 1
        avgpool1 = tf.layers.average_pooling3d(inputs, (2,1,1), (2,1,1)) 
        # ? * 16 * 64 * 64 * 1
        conv1 = tf.layers.conv3d(avgpool1, 32, 3, padding='same', activation=tf.nn.relu)
        # ? * 16 * 64 * 64 * 64
        maxpool1 = tf.layers.max_pooling3d(conv1, (1,2,2), (1,2,2))
        # ? * 16 * 32 * 32 * 64
        conv2 = tf.layers.conv3d(maxpool1, 64, 3, padding='same', activation=tf.nn.relu)
        # ? * 16 * 32 * 32 * 128
        maxpool2 = tf.layers.max_pooling3d(conv2, 2, 2)
        # ? * 8 * 16 * 16 * 128
        conv3 = tf.layers.conv3d(maxpool2, 64, 3, padding='same', activation=tf.nn.relu)
        # ? * 8 * 16 * 16 * 256
        maxpool3 = tf.layers.max_pooling3d(conv3, 2, 2)
        # ? * 4 * 8 * 8 * 256
        conv4 = tf.layers.conv3d(maxpool3, 128, 3, padding='same', activation=tf.nn.relu)
        # ? * 4 * 8 * 8 * 512
        maxpool4 = tf.layers.max_pooling3d(conv4, 2, 2)
        # ? * 2 * 4 * 4 * 512
        conv5 = tf.layers.conv3d(maxpool4, 64, 3, padding='same', activation=tf.nn.relu)
        # ? * 2 * 4 * 4 * 64
        maxpool5 = tf.layers.max_pooling3d(conv5, 2, 2)
        # ? * 1 * 2 * 2 * 64
        conv6 = tf.layers.conv3d(maxpool5, 1, 3, padding='same', activation=tf.nn.relu)
        # ? * 1 * 2 * 2 * 1
        reshaped = tf.reshape(conv6, [-1, 4])
        # ? * 4
        self.predictions = tf.layers.dense(reshaped, 2)
        # ? * 2

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
            #weights = (self.FLAGS.weight_one-1)*tf.to_float(self.labels) + 1
            print(self.labels.shape)
            print(self.predictions.shape)
            self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.predictions)    

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

    def test(self, session, x, y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        sample_indices = list(range(len(x)))
        num_samples = min(len(x), 128)
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
            
            x_train = x_train[indices]
            y_train = y_train[indices]
            
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
                x_train_batch = x_train[start:end]
                y_train_batch = y_train[start:end]
                _, train_loss = self.optimize(session, x_train_batch, y_train_batch)
                epoch_train_losses.append(train_loss)
                logging.info("batch %s/%s training loss: %s" % (b_end, num_minibatches, train_loss))
                
                if b_end % 100 == 0:
                    logging.info("-"*80)
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
            

