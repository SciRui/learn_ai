import numpy as np
import tensorflow as tf

 class DBN():
     #
     def __init__(self):
         #
         pass

    def def_inputs(self,samples_dim, labels_dim):
        #define model inputs
        with tf.name_scope("inputs"):
            self.tf_train_samples = tf.placeholder(tf.float32, [self.train_batch_size,samples_dim])
            self.tf_train_labels = tf.placeholder(tf.float32, [self.train_batch_size,labels_dim])
            self.tf_test_samples = tf.placeholder(tf.float32, [self.test_batch_size,samples_dim])

    def add_rbm(self, in_num_nodes, out_num_nodes, activation=None, name=None):
        with tf.name_scope(name):
            rbm_probability_v = tf.Variable()
            rbm_probability_h = tf.Variable()
            rbm_weights = tf.Variable()
            rbm_biases_v = tf.Variable()
            rbm_biases_h = tf.Variable()
        

    def create_model(self):
        #
        def run_model(data_flow):
            #
            h0 = sample_prob(tf.nn.sigmoid(tf.matmul(data_flow, rbm_weights) + rbm_biases_h))
            v1 = sample_prob(tf.nn.sigmoid(tf.matmul(h0, tf.transpose(rbm_weights)) + rbm_biases_v))
            h1 = tf.nn.sigmoid(tf.matmul(v1, rbm_weights) + rbm_biases_h)
            #
            w_positive_grad = tf.matmul(tf.transpose(data_flow), h0)
            w_negative_grad = tf.matmul(tf.transpose(v1), h1)
            #
            update_w = rbm_weights + alpha * (w_positive_grad - w_negative_grad) / tf.to_float(tf.shape(data_flow)[0])
            update_vb = rbm_biases_v + alpha * tf.reduce_mean(data_flow - v1, 0)
            update_hb = rbm_biases_h + alpha * tf.reduce_mean(h0 - h1, 0)
            #
            h_sample = sample_prob(tf.nn.sigmoid(tf.matmul(data_flow, rbm_weights) + rbm_biases_h))
            v_sample = sample_prob(tf.nn.sigmoid(tf.matmul(h_sample, tf.transpose(rbm_weights)) + rbm_biases_v))
            err = data_flow - v_sample
            err_sum = tf.reduce_mean(err * err)

    
    def train_model(self):
        pass

    def test_model(self):
        pass


    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random_uniform(tf.shape(probs))))