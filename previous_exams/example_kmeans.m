%% Clustering
load clusterData.mat

%% K-Means Clustering
doPlot = 1; % Turn on visualization of the algorithm in action (2D data)
k = 4;
model = clusterKmeans(X,k,doPlot);

% Use model to cluster training data
y = model.predict(model,X);
clustering2Dplot(X,y,model.W)