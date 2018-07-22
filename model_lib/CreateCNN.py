import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix


class ConvolutionNeuronNetwork():
    def __init__(self, train_batch_size, test_batch_size,
                 init_learning_rate, decay_rate, decay_steps,
                 optimizeMethod='adam', save_path='model/default.ckpt',):
        #
        self.train_samples_shape = None
        self.train_labels_shape = None
        self.test_samples_shape = None
        """
        @num_hidden: 隐藏层的节点数量
        @batch_size：因为我们要节省内存，所以分批处理数据。每一批的数据量。
        """
        self.optimizeMethod = optimizeMethod
        self.init_learning_rate = init_learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        # Hyper Parameters
        self.conv_config = []
        self.fc_config = []
        self.conv_weights = []
        self.conv_biases = []
        self.fc_weights = []
        self.fc_biases = []
        # Graph Related
        self.tf_train_samples = None
        self.tf_train_labels = None
        self.tf_test_samples = None
        self.tf_test_labels = None
        #
        self.graph_saver = None
        self.save_path = save_path


    def def_inputs(self, train_samples_shape, train_labels_shape, test_samples_shape):
            # 这里只是定义图谱中的各种变量
        with tf.name_scope('inputs'):
            self.tf_train_samples = tf.placeholder(tf.float32, shape=train_samples_shape, name='tf_train_samples')
            self.tf_train_labels = tf.placeholder(tf.float32, shape=train_labels_shape, name='tf_train_labels')
            self.tf_test_samples = tf.placeholder(tf.float32, shape=test_samples_shape, name='tf_test_samples')

    def add_conv(self, filter_size, in_num_channels, out_num_channels,
                 stride=(1,1,1,1), padding="SAME", activation="relu",
                 pooling=True, pooling_scale=2, name=None):
        """
        This function does not define operations in the graph, but only store config in self.conv_layer_config
        """
        self.conv_config.append({
            'patch_size': filter_size,
            'in_depth': in_num_channels,
            'out_depth': out_num_channels,
            "stride":stride,
            "padding": padding,
            'activation': activation,
            'pooling': pooling,
            "pooling_scale":pooling_scale,
            'name': name
        })
        with tf.name_scope(name):
            weights = tf.Variable(tf.truncated_normal([filter_size[0], filter_size[1], in_num_channels, out_num_channels], stddev=0.1),
                                  name=name + '_weights')
            biases = tf.Variable(tf.constant(0.1, shape=[out_num_channels]), name=name + '_biases')
            self.conv_weights.append(weights)
            self.conv_biases.append(biases)

    def add_fc(self, in_num_nodes, out_num_nodes, activation=None, dropout=True, dropout_rate=None, dropout_seed=None, name=None):
        """
        add fc layer config to slef.fc_layer_config
        """
        self.fc_config.append({'in_num_nodes': in_num_nodes,
                               'out_num_nodes': out_num_nodes,
                               'activation': activation,
                               "dropout":dropout,
                               "dropout_rate":dropout_rate,
                               "dropout_seed":dropout_seed,
                               'name': name})
        with tf.name_scope(name):
            weights = tf.Variable(tf.truncated_normal([in_num_nodes, out_num_nodes], stddev=0.1))
            biases = tf.Variable(tf.constant(0.1, shape=[out_num_nodes]))
            self.fc_weights.append(weights)
            self.fc_biases.append(biases)

    def create_model(self):
        ##
        def run_model(data_flow, train_model = True):
            """
            @data: original inputs
            @return: logits
            """
            # Define Convolutional Layers
            for i, (weights, biases, config) in enumerate(zip(self.conv_weights, self.conv_biases, self.conv_config)):
                with tf.name_scope(config['name'] + '_model'):
                    with tf.name_scope('convolution'):
                        # default 1,1,1,1 stride and SAME padding
                        data_flow = tf.nn.conv2d(data_flow, filter=weights, strides=config["stride"], padding=config["padding"])
                        data_flow = data_flow + biases
                    if config['activation'] == 'relu':
                        data_flow = tf.nn.relu(data_flow)
                    else:
                        raise Exception('Activation Func can only be Relu right now. You passed', config['activation'])
                    if config['pooling']:
                        data_flow = tf.nn.max_pool(
                            data_flow,
                            ksize=[1, config["pooling_scale"], config["pooling_scale"], 1],
                            strides=[1, config["pooling_scale"], config["pooling_scale"], 1],
                            padding=config["padding"])

            # Define Fully Connected Layers
            for i, (weights, biases, config) in enumerate(zip(self.fc_weights, self.fc_biases, self.fc_config)):
                if i == 0:
                    shape = data_flow.get_shape().as_list()
                    data_flow = tf.reshape(data_flow, [shape[0], shape[1] * shape[2] * shape[3]])
                with tf.name_scope(config['name'] + 'model'):
                    data_flow = tf.matmul(data_flow, weights) + biases
                    if config['activation'] == 'relu':
                        data_flow = tf.nn.relu(data_flow)
                    elif config['activation'] is None:
                        pass
                    else:
                        raise Exception('Activation Func can only be Relu or None right now. You passed',
                                        config['activation'])
                
                ### Dropout
                if train_model and i == 0:
                    data_flow = tf.nn.dropout(data_flow, config["dropout_rate"], seed=config["dropout_seed"])
                ###
            return data_flow
    
        #
        train_logits = run_model(self.tf_train_samples)
        self.model_optimizer(train_logits, self.tf_train_labels)   
        with tf.name_scope('train'):
            self.train_prediction = tf.nn.softmax(train_logits, name='train_prediction')
        
        # Predictions for the training, validation, and test data.
        test_logits = run_model(self.tf_test_samples, train_model = False)
        with tf.name_scope('test'):
            self.test_prediction = tf.nn.softmax(test_logits, name='test_prediction')

        self.graph_saver =  tf.train.Saver(tf.global_variables())

    def model_optimizer(self, train_logits, train_labels):        
        #
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = train_logits, 
                                                                               labels = train_labels))
            regularization = 0
            for weights, biases in zip(self.fc_weights, self.fc_biases):
                regularization += tf.nn.l2_loss(weights) + tf.nn.l2_loss(biases)
            self.loss += 5e-4*regularization

        # learning rate decay
        global_step = tf.Variable(0)
        lr = self.init_learning_rate
        dr = self.decay_rate
        learning_rate = tf.train.exponential_decay(
            learning_rate=lr,
            global_step=global_step * self.train_batch_size,
            decay_steps=self.decay_steps,
            decay_rate=dr,
            staircase=True)

        # Optimizer.
        with tf.name_scope('optimizer'):
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
            #optimizer = tf.train.MomentumOptimizer(learning_rate, 0.5).minimize(loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def train_model(self, train_samples, train_labels, data_iterator, iteration_steps):
        #
        with tf.Session(graph=tf.get_default_graph()) as session:
            tf.global_variables_initializer().run()
            #Start Training
            model_loss = []
            model_accuracies = []
            # batch 1000
            for i, samples, labels in data_iterator(train_samples, train_labels, iteration_steps=iteration_steps,
                                                    chunkSize=self.train_batch_size):
                _, tmp_loss, tmp_predictions = session.run(
                    [self.optimizer, self.loss, self.train_prediction],
                    feed_dict={self.tf_train_samples: samples, self.tf_train_labels: labels}
                )
                acc_,_ = self.accuracy(tmp_predictions, labels)
                model_loss.append(tmp_loss)
                model_accuracies.append(acc_)
                if i % 50 == 0:
                    print('Minibatch loss at step %d: %f' % (i, tmp_loss))
                    print('Minibatch accuracy: %.1f%%' % acc_)
            
            if os.path.isdir(self.save_path.split('/')[0]):
                save_path = self.graph_saver.save(session, self.save_path)
                print("Model saved in file: %s" % save_path)
            else:
                os.makedirs(self.save_path.split('/')[0])
                save_path = self.graph_saver.save(session, self.save_path)
                print("Model saved in file: %s" % save_path)

        return model_loss, model_accuracies


    def test_model(self, test_samples, test_labels, data_iterator):
        #
        if self.graph_saver is None:
            self.create_model()
        #
        with tf.Session(graph=tf.get_default_graph()) as session:
            #tf.global_variables_initializer().run()
            self.graph_saver.restore(session, self.save_path)
            ### 测试
            model_accuracies = []
            confusionMatrices = []
            for i, samples, labels, in data_iterator(test_samples, test_labels, chunkSize=self.test_batch_size):
                tmp_predictions = session.run(self.test_prediction, feed_dict={self.tf_test_samples: samples})
                acc_, cm = self.accuracy(tmp_predictions, labels, need_confusion_matrix=True)
                model_accuracies.append(acc_)
                confusionMatrices.append(cm)
                print('Test Accuracy: %.1f%%' % acc_)
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