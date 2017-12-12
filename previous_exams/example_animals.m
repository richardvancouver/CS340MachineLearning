%% Animals with attributes data
load animals.mat

%% K-Means clustering
k = 5;
model = clusterKmeans(X,k,0);

for c = 1:k
    fprintf('Cluster %d: ',c);
    fprintf('%s ',animals{model.y==c});
    fprintf('\n');
end