clc; clear
close all
addpath(genpath(pwd));
train_path = './';
dirOutput_train = dir(fullfile(train_path,'*.mat'));
fileNames_train = sortnat({dirOutput_train.name}); 
folder_num = length(fileNames_train);%获取文件夹总数量 
d_train = zeros(1,10);
T_train = zeros(1,100);
for i =1:50
    load(char(fileNames_train(i)));
    d_train=cat(1,d_train,d);
    T_train=cat(1,T_train,T);
end
d = d_train(2:end,:);
T = T_train(2:end,:);
save('../test_10layers','d','T')