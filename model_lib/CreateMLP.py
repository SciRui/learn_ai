import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

class MLPNeuronNetwork():
    #
    def __init__(self, train_batch_size,test_batch_size,
                 init_learning_rate,decay_rate, decay_steps,
                 save_path='model/default.ckpt'):
        self.tf_train_samples = None
        self.tf_train_labels = None
        self.tf_test_samples = None
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.mlp_config = []
        #
        self.init_learning_rate = init_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        #
        self.graph_saver = None
        self.save_path = save_path

    def def_inputs(self,samples_dim, labels_dim):
        #define model inputs
        with tf.name_scope("inputs"):
            self.tf_train_samples = tf.placeholder(tf.float32, [self.train_batch_size,samples_dim])
            self.tf_train_labels = tf.placeholder(tf.float32, [self.train_batch_size,labels_dim])
            self.tf_test_samples = tf.placeholder(tf.float32, [self.test_batch_size,samples_dim])
            tf_test_labels = tf.placeholder(tf.float32, [self.test_batch_size,labels_dim])

    def add_mlp(self,in_nodes, out_nodes,activation=None, name=None):
        with tf.name_scope(name):
            weights = tf.Variable(tf.truncated_normal([in_nodes,out_nodes], stddev=0.1), name = name + "weights")
            biases = tf.Variable(tf.constant(0.1, shape=[out_nodes]), name = name + "biases")
            tmp_dict = {"in_nodes":in_nodes,"out_nodes":out_nodes,
                        "weights":weights,"biases":biases,"activation":activation}
            self.mlp_config.append(tmp_dict)
    #
    def create_model(self):
        #
        def run_model(data_flow):
            for layer_config in self.mlp_config:
                data_flow = tf.add(tf.matmul(data_flow,layer_config["weights"]),layer_config["biases"])
                if layer_config["activation"] is None:
                    continue
                else:
                    data_flow = tf.nn.relu(data_flow)
            return data_flow

        train_logits = run_model(self.tf_train_samples)
        self.model_optimizer(train_logits, self.tf_train_labels)   
        #
        with tf.name_scope("train_model"):
            self.train_prediction = tf.nn.softmax(train_logits, name = "train_prediction")

        #test
        test_logits = run_model(self.tf_test_samples)
        with tf.name_scope("test_model"):
            self.test_prediction = tf.nn.softmax(test_logits, name = "test_prediction")

        self.graph_saver =  tf.train.Saver(tf.global_variables())
    #
        # accuracy = tf.metrics.accuracy(predictions = tf.argmax(fc3, axis = 1),
        #                 labels = tf.argmax(tf_test_labels,axis = 1))[1]


    def model_optimizer(self, train_logits, train_labels):        
        #
        with tf.name_scope("loss"):
            self.train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = train_logits, 
                                                                               labels = train_labels))
            regularization = 0
            for layer_config in self.mlp_config:
                regularization += tf.nn.l2_loss(layer_config["weights"]) + tf.nn.l2_loss(layer_config["biases"])
            self.train_loss += 5e-4*regularization

        # learning rate decay
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            learning_rate=self.init_learning_rate,
            global_step=global_step * self.train_batch_size,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=True)

        # Optimizer.
        with tf.name_scope('optimizer'):
            #self.train_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(train_loss)
            #self.train_optimizer = tf.train.MomentumOptimizer(learning_rate, 0.5).minimize(train_loss)
            self.train_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.train_loss)

    def train_model(self, train_samples, train_labels, data_iterator, iteration_steps):
        #run model
        print("starting training...")
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            #tf.local_variables_initializer().run()
            model_loss = []
            model_accuracies = []
            #
            for i, samples, labels in data_iterator(train_samples, train_labels, iteration_steps=iteration_steps,
                                        chunkSize=self.train_batch_size):
                _, loss_, tmp_prediction_ = sess.run([self.train_optimizer,self.train_loss, self.train_prediction], 
                                                     feed_dict = {self.tf_train_samples:samples,self.tf_train_labels:labels})
                acc_,_ = self.accuracy(tmp_prediction_, labels)
                model_loss.append(loss_)
                model_accuracies.append(acc_)
                if i % 50 == 0:
                    print('Minibatch loss at step %d: %f' % (i, loss_))
                    print('Minibatch accuracy: %.1f%%' % acc_)

            if os.path.isdir(self.save_path.split('/')[0]):
                save_path = self.graph_saver.save(sess, self.save_path)
                print("Model saved in file: %s" % save_path)
            else:
                os.makedirs(self.save_path.split('/')[0])
                save_path = self.graph_saver.save(sess, self.save_path)
                print("Model saved in file: %s" % save_path)
            
        return model_loss, model_accuracies

    def test_model(self, test_samples, test_labels, data_iterator):
        if self.graph_saver is None:
            self.create_model()
        #
        print("starting test...")
        with tf.Session(graph=tf.get_default_graph()) as sess:
            #tf.global_variables_initializer().run()
            self.graph_saver.restore(sess, self.save_path)
            ### 测试
            model_accuracies = []
            confusionMatrices = []
            for i, samples, labels, in data_iterator(test_samples, test_labels, chunkSize=self.test_batch_size):
                tmp_predictions = sess.run(self.test_prediction, feed_dict={self.tf_test_samples: samples})

                accuracy, cm = self.accuracy(tmp_predictions, labels, need_confusion_matrix=True)
                model_accuracies.append(accuracy)
                confusionMatrices.append(cm)
                print('Test Accuracy: %.1f%%' % accuracy)
            avg_accuracy = np.average(model_accuracies)
            std_accuracy = np.std(model_accuracies)
            print(' Average  Accuracy:', avg_accuracy)
            print('Standard Deviation:', std_accuracy)
            self.print_confusion_matrix(np.add.reduce(confusionMatrices))

        return model_accuracies,avg_accuracy,std_accuracy

    def accuracy(self, predictions, labels, need_confusion_matrix=False):
        """
        计算预测的正确率与召回率
        @return: accuracy and confusionMatrix as a tuple
        """
        _predictions = np.argmax(predictions, 1)
        _labels = np.argmax(labels, 1)
        cm = confusion_matrix(_labels, _predictions) if need_confusion_matrix else None
        # == is overloaded for numpy array
        accuracy = (100.0 * np.sum(_predictions == _labels) / predictions.shape[0])
        return accuracy, cm


    def print_confusion_matrix(self, confusionMatrix):
        print('Confusion    Matrix:')
        for i, line in enumerate(confusionMatrix):
            print(line, line[i] / np.sum(line))
        a = 0
        for i, column in enumerate(np.transpose(confusionMatrix, (1, 0))):
            a += (column[i] / np.sum(column)) * (np.sum(column) / 26000)
            print(column[i] / np.sum(column), )
        print('\n', np.sum(confusionMatrix), a)
