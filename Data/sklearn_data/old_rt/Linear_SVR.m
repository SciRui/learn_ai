clc;
close all;

training_set = xlsread();
X_train = ;
y_train = ;
test_set = xlsread();
X_test = ;
y_test = ;

%% train model
cv_model = fitrsvm(x_train, y_train,'KernelFunction','linear','IterationLimit',1000, ...\
                   'Epsilon',0.1,'DeltaGradientTolerance',1e-4,'KFold',10);
%% validate model(K-Fold cross validation)
[E,SE,~,~] = cvloss(cv_model,'KFold',10)

%% test model
predictions = predict(cv_model,X_test);
