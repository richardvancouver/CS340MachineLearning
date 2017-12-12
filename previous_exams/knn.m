function [model] = knn(X,y,k)
% [model] = knn(X,y,k)
%
% Implementation of k-nearest neighbour classifer

model.X = X;
model.y = y;
model.k = k;
model.c = max(y);
model.predict = @predict;
end

function [yhat] = predict(model,Xtest)
% Write me!

xx=model.X;
[n,d]=size(xx);
[t,d]=size(Xtest);

D=xx.^2*ones(d,t)+ones(n,d)*(Xtest').^2-2*xx*Xtest';

% for each D(i,:), picking out the k smallest values, find their
% corresponding labels, deciding the label for Xtest(i) based on the majority
% rule

nk=model.k; 
ytemp=model.y;
for i=1:1:t

tt=D(:,i);


val = zeros(nk,1);
idx=zeros(nk,1);
for ii=1:nk
  [val(ii),idx(ii)] = min(tt);
  % remove for the next iteration the last smallest value:
  tt(idx(ii) ) = 1e9;%[];
end

%ytemp=model.y;

%ytemp(idx);

yhat(i)=mode(ytemp(idx));

end



end