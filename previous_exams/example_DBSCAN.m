%% Clustering
load clusterData2.mat

%% Density-Based Clustering
radius = 1;
minPts = 3;
doPlot = 1;
model = clusterDBcluster(X,radius,minPts,doPlot);
title('Densty-Based clustering');
%print -dpng density.png