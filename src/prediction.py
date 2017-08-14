"""
Learning model uses Tensorflow LSTM and Spearman Correlation to find similarity in the pattern
"""
import logging
import os

import pandas as pd
import numpy as np
import random
import tensorflow as tf
import scipy.stats as ss

from src.CustomLogger import ColoredLogger

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logging.setLoggerClass(ColoredLogger)

class TrainingModel(object):

    def __init__(self, path_of_file=None):
        self.path_of_data = path_of_file
        self.num_periods = 20
        self.f_horizon = 1

        self.inputs = 1
        self.hidden = 100
        self.output = 1

    def generate_training_data(self, view_count_data=None):
        random.seed(111)
        rng = pd.date_range(start='2017-01-01', end='2017-01-21', freq='D')
        if not view_count_data:
            ts = pd.Series(np.random.uniform(10, 20, size=len(rng)), rng).cumsum()
        else:
            ts = pd.Series(np.array(view_count_data), rng).cumsum()

        # Normalize given time series
        TS = np.array(ts)
        num_periods = self.num_periods
        f_horizon = self.f_horizon

        x_data = TS[:(len(TS)-(len(TS) % num_periods))]
        self.x_batches = x_data.reshape(-1, 20, 1)

        y_data = TS[1:(len(TS)-(len(TS) % num_periods))+f_horizon]
        self.y_batches = y_data.reshape(-1, 20, 1)

        X_test, Y_test = self.reshape_time_series_data(TS,f_horizon,num_periods )

        if not view_count_data:
            self.Y_actual = Y_test

        return X_test, Y_test

    def prepare_predictor_model(self):
        tf.reset_default_graph()
        logging.debug("Setting up training model")

        self.X = tf.placeholder(tf.float32, [None, self.num_periods, self.inputs])
        self.Y = tf.placeholder(tf.float32, [None, self.num_periods, self.output])

        basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=self.hidden, activation=tf.nn.relu)
        rnn_output, states = tf.nn.dynamic_rnn(basic_cell, self.X, dtype=tf.float32)
        learning_rate = 0.001

        stacked_rnn_output = tf.reshape(rnn_output, [-1, self.hidden])
        stacked_outputs = tf.layers.dense(stacked_rnn_output, self.output)
        self.outputs = tf.reshape(stacked_outputs, [-1, self.num_periods, self.output])

        self.loss = tf.reduce_sum(tf.square(self.outputs - self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.training_op = optimizer.minimize(self.loss)

        self.init = tf.global_variables_initializer()
        return self.init

    def RNN_session_on_trained_data(self):
        X_test,Y_test =  self.generate_training_data()
        init = self.prepare_predictor_model()
        epochs = 1000     #number of iterations or training cycles, includes both the FeedFoward and Backpropogation
        with tf.Session() as sess:
            init.run()
            for ep in range(epochs):
                sess.run(self.training_op, feed_dict={self.X: self.x_batches, self.Y: self.y_batches})
                if ep % 100 == 0:
                    mse = self.loss.eval(feed_dict={self.X: self.x_batches, self.Y: self.y_batches})
                    logging.debug("\tEpoch %s, Mean Squared Error: %s" % (ep, mse))

            logging.debug("Training on smaller set of data completed."
                          "\n Running prediction algo. using pre-computed data")

            y_pred = sess.run(self.outputs, feed_dict={self.X: X_test})
            return y_pred

    def run_prediction_on_stream_batch(self, input_list):
        logging.debug("Running prediction algorithm on batch stream")
        X_test,Y_test = self.generate_training_data(input_list)
        y_pred = self.RNN_session_on_trained_data()
        return y_pred

    def is_video_going_to_viral(self, input_list):
        y_pred = self.run_prediction_on_stream_batch(input_list)
        y_actual = self.Y_actual
        df1 = pd.DataFrame({'actual':pd.Series(np.ravel(y_actual))})
        df2 = pd.DataFrame({'predicted':pd.Series(np.ravel(y_pred))})
        return True if ss.pearsonr(df1['actual'], df2['predicted']) >= 0.9 else False

    def reshape_time_series_data(self, series,forecast,num_periods):
        test_x_setup = series[-(num_periods + forecast):]
        testX = test_x_setup[:num_periods].reshape(-1, 20, 1)
        testY = series[-(num_periods):].reshape(-1, 20, 1)
        return testX,testY



# view_counts = [random.randint(10,100) for _ in range(21)]
# abc = TrainingModel()
# print abc.is_video_going_to_viral(view_counts)

