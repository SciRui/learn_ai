clc;
close all;

%%
data_set_file_path = 'E:\About_Program\Python\Projects\learn_AI\Data\sklearn_data\src_data\sklearn_boston.xlsx';
test_set_file_path = 'E:\About_Program\Python\Projects\learn_AI\Data\sklearn_data\test_sklearn_boston.xlsx';
out_result = 'E:\About_Program\Python\Projects\learn_AI\Data\sklearn_data\matlab_bpnn_rt.xlsx';
%%
trin_set = xlsread('E:\About_Program\Python\Projects\learn_AI\Data\sklearn_data\training_sklearn_boston.xlsx',...
                       'Sheet1', 'A1:N354');
valid_set = xlsread('E:\About_Program\Python\Projects\learn_AI\Data\sklearn_data\training_sklearn_boston.xlsx',...
                       'Sheet1', 'A1:N76');
test_set = xlsread('E:\About_Program\Python\Projects\learn_AI\Data\sklearn_data\training_sklearn_boston.xlsx',...
                    'Sheet1', 'A1:N76');
Dataset = [trin_set;valid_set;test_set];
X = Dataset(:,1:end-1)';
Target = Dataset(:,end)';

%% cerate bpnn
net = feedforwardnet([16,16,16]);
%
net.trainParam.lr = 0.01;
net.trainParam.goal= 0.001;
net.trainParam.epochs= 200;
net.trainFcn = 'trainbfg';
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'tansig';
net.performFcn = 'mse';
net.divideParam.trainRatio = 0.7;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.15;
% net.divideFcn = 'divideind';
% [train_indices, val_indices, test_indices] = divideind(num_samples,1: split_index1, ...\
%                                                       (split_index1 + 1):split_index2, ...\
%                                                       (split_index2 + 1):num_samples);
% net.divideParam.tainInd = train_indices;
% net.divideParam.valInd = val_indices;
% net.divideParam.testInd = test_indices;

%% train model
bpnn = train(net, X, Target);

%% validate model
% prediction_train = bpnn(X_train);
% disp(perform(net,prediction_train,y_train));
% 
% %% test model
% prediction_test = bpnn(X_test);
% 
% %% export result
% xlswrite(out_result, [y_train',prediction_train'],1);
% xlswrite(out_result, [y_test',prediction_test'],2);
