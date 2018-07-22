# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import ml_model.svc_water_quality as svcw
#import PreprocessData.PreprocessData_WaterQuality as predata
import PreprocessData.PreprocessData_MNIST as predata
from dl_model.CreateMLP import MLPNeuronNetwork as mlpnn
from dl_model.CreateCNN import ConvolutionNeuronNetwork as CNN

if __name__ == "__main__":
    #
    train_samples = predata.train_samples
    train_labels = predata.dl_train_labels
    test_samples = predata.test_samples
    test_labels = predata.dl_test_labels
    #
    image_height = predata.image_height
    image_width = predata.image_width
    num_labels = predata.num_labels
    num_channels = predata.num_channels
    #
    # ML,SVM-SVC
    
    # svc_model = svcw.train_model(train_samples,train_labels)
    # model_accuracy = svcw.test_model(svc_model,test_samples,test_labels)
    # # model_scopes, model_scopes_avg = svcw.create_model(predata.ml_samples, predata.ml_labels)
    # print(model_accuracy)

    #DL,MLP
    #
    #num_train_data = 57   #57   60000   73257  
    # train_batch_size = 64  #5     64      64
    # test_batch_size = 500    #10    500     500   
    # samples_dim = train_samples.shape[1]  #751   784   1024
    # labels_dim = train_labels.shape[1]    #2     10    10
    # model_save_path = "Model/mlp_mnist.ckpt"
    # #
    # #optimizer
    # iteration_steps = 2000
    # init_learning_rate = 0.001  #0.1
    # decay_steps = 100
    # decay_rate = 0.99
    # #
    # #apply model
    # mlp = mlpnn(train_batch_size,test_batch_size,
    #             init_learning_rate,decay_rate, decay_steps,
    #             save_path=model_save_path)
    # #
    # mlp.def_inputs(samples_dim,labels_dim)
    # mlp.add_mlp(samples_dim,128,activation = "relu",name="fc1")
    # mlp.add_mlp(128,128,activation = "relu",name="fc2")
    # mlp.add_mlp(128,labels_dim,name="fc3")
    # #
    # mlp.create_model()
    # #
    # print('Start Training...')
    # model_loss, train_accuracies = mlp.train_model(train_samples, train_labels,
    #                                                predata.train_data_iterator, 
    #                                                iteration_steps=iteration_steps)
    # #
    # print('Start Testing...')
    # test_accuracies, avg_accuracy, std_accuracy = mlp.test_model(test_samples, test_labels,
    #                                                              predata.test_data_iterator)

    # #DL,CNN
    #
    train_batch_size = 64  #16     64 
    test_batch_size = 500    #30    500  
    kernel_size = (3,3)   # (1,1)   (3,3)
    num_kernel = 32      #128      32
    stride = (1,1,1,1)
    model_save_path = "Model/cnn_mnist.ckpt"
    #
    pooling_scale = 2     #1(pooling = False)   2
    iteration_steps = 2000
    init_learning_rate = 0.001  #0.1
    decay_steps = 100
    decay_rate = 0.99
    #
    cnn = CNN(train_batch_size,test_batch_size,init_learning_rate,decay_rate, decay_steps, save_path = model_save_path)
    #
    cnn.def_inputs(train_samples_shape = (train_batch_size,image_height,image_width,num_channels),
                    train_labels_shape = (train_batch_size, num_labels),
                    test_samples_shape = (test_batch_size,image_height,image_width,num_channels))   
    cnn.add_conv(filter_size=kernel_size, in_num_channels=num_channels, out_num_channels=num_kernel, 
                stride=(1,1,1,1), padding="SAME", activation='relu', pooling=False, name='conv1')
    cnn.add_conv(filter_size=kernel_size, in_num_channels=num_kernel, out_num_channels=num_kernel, 
                stride=(1,1,1,1), padding="SAME", activation='relu', pooling_scale=pooling_scale, name='conv2')
    cnn.add_conv(filter_size=kernel_size, in_num_channels=num_kernel, out_num_channels=num_kernel, 
                stride=(1,1,1,1), padding="SAME", activation='relu', pooling=False, name='conv3')
    cnn.add_conv(filter_size=kernel_size, in_num_channels=num_kernel, out_num_channels=num_kernel, 
                stride=(1,1,1,1), padding="SAME", activation='relu', pooling_scale=pooling_scale, name='conv4')

    #两次pooling, 每一次缩小为 1/2,(28/4)*(28/4)*32, 32 = conv4 out_depth
    cnn.add_fc(in_num_nodes=1568, out_num_nodes=128, activation='relu', dropout_rate = 0.99, dropout_seed = 4926, name='fc1')
    cnn.add_fc(in_num_nodes=128, out_num_nodes=10, activation=None, dropout_rate = False, name='fc2')
    #
    cnn.create_model()
    #
    print('Start Training...')
    model_loss, train_accuracies = cnn.train_model(train_samples,
                                                   train_labels,
                                                   data_iterator = predata.train_data_iterator,
                                                   iteration_steps=iteration_steps)
    print('Start Testing...')
    test_accuracies, avg_accuracy, std_accuracy = cnn.test_model(test_samples,
                                                                test_labels,
                                                                data_iterator = predata.test_data_iterator)
    # #
    plt.figure()
    plt.plot(model_loss, label = "loss")
    plt.plot(train_accuracies, label = "accuracy")
    plt.xlim(0,len(train_accuracies))
    plt.ylim(0,100)
    plt.xlabel("Number of iterations")
    plt.ylabel("Percent(%)")
    plt.legend(["loss","accuracy"], loc = 'best')
    #
    plt.figure()
    plt.plot(test_accuracies)
    plt.xlim(0,len(test_accuracies))
    plt.ylim(np.min(test_accuracies),100)
    plt.xticks(range(0,len(test_accuracies)))
    plt.xlabel("Number of iterations")
    plt.ylabel("Percent(%)")
    plt.show()
