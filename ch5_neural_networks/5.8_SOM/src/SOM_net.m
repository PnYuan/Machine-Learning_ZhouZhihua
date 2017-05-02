clc
clear

%% generation of data
data = csvread('watermelon_3a.csv');
X = data(:,2:3)';
y = data(:,4);
% normalization 
X(1,:) = mapminmax(X(1,:), 0, 1);
X(2,:) = mapminmax(X(2,:), 0, 1);

%% build SOM network
%net = newsom(X,[10 1]);
net = selforgmap([10,1]);
plotsom(net.layers{1}.positions);

%% training
net.trainParam.epochs = 3000;
net = train(net,X);

%net.trainparam.epochs = 400;
%net = train(net,X);
y_pred = sim(net,X);
yc(1,:) = vec2ind(y_pred );
plotsom(net.IW{1,1},net.layers{1}.distances)

%% prediction
x_tst = [0 ;0];
r = sim(net,x_tst);
xx = vec2ind(r)

%% 网络神经元分布情况
% 查看网络拓扑学结构
plotsomtop(net);
% 查看临近神经元直接的距离情况
plotsomnd(net);
% 查看每个神经元的分类情况
plotsomhits(net,X);

