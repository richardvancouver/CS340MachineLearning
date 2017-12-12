function [model] = decisionForest(X,y,depth,nBootstraps)

% Fit model to each boostrap sample of data
for m = 1:nBootstraps
    model.subModel{m} = decisionTree(X,y,depth);
end

model.predict = @predict;

end

function [y] = predict(model,X)

% Predict using each model
for m = 1:length(model.subModel)
    y(:,m) = model.subModel{m}.predict(model.subModel{m},X);
end

% Take the most common label
y = mode(y,2);
end