# Load X and y variable
using JLD
data = load("basisData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])




# Data is sorted, so *randomly* split into train and validation:
n = size(X,1)
perm = randperm(n)

# sumbestSigma=0;
# folds=10
# for k in 0:(folds-1)
# validStart = Int64(n*k/folds+1) # Start of validation indices
# validEnd = Int64( (k+1)*n/folds ) # End of validation incides
# validNdx = perm[validStart:validEnd] # Indices of validation examples
# trainNdx = perm[setdiff(1:n,validNdx)] # Indices of training examples
# Xtrain = X[trainNdx,:]
# ytrain = y[trainNdx]
# Xvalid = X[validNdx,:]
# yvalid = y[validNdx]

# Find best value of RBF variance parameter,
#	training on the train set and validating on the test set
include("leastSquaresb.jl")
minErr = Inf
bestSigma = []
lambda=1e-12



for sigma in 2.0.^(-15:15)


sumve=0;



folds=10


for k in 0:(folds-1)
validStart = Int64(n*k/folds+1) # Start of validation indices
validEnd = Int64( (k+1)*n/folds ) # End of validation incides
validNdx = perm[validStart:validEnd] # Indices of validation examples
trainNdx = perm[setdiff(1:n,validNdx)] # Indices of training examples
Xtrain = X[trainNdx,:]
ytrain = y[trainNdx]
Xvalid = X[validNdx,:]
yvalid = y[validNdx]
	# Train on the training set
	model = leastSquaresRBF(Xtrain,ytrain,sigma, lambda)

	# Compute the error on the validation set
	yhat = model.predict(Xvalid)
	validError = sum((yhat - yvalid).^2)
	#@printf("With sigma = %.3f, validError = %.2f\n",sigma,validError)



sumve=sumve+validError 

end	

sumve=sumve/folds


	# Keep track of the lowest validation error
	if sumve < minErr
		minErr = sumve
		bestSigma = sigma
	end
@printf("With sigma = %.3f, validError = %.2f\n",bestSigma,minErr)

end




#end

#bestSigma=sumbestSigma/folds

# Now fit the model based on the full dataset
model = leastSquaresRBF(X,y,bestSigma, lambda)

# Report the error on the test set
t = size(Xtest,1)
yhat = model.predict(Xtest)
testError = sum((yhat - ytest).^2)/t
@printf("With best sigma of %.3f, testError = %.2f\n",bestSigma,testError)

# Plot model
using PyPlot
figure()
plot(X,y,"b.")
plot(Xtest,ytest,"g.")
Xhat = minimum(X):.1:maximum(X)
Xhat = reshape(Xhat,length(Xhat),1) # Make into an n by 1 matrix
yhat = model.predict(Xhat)
plot(Xhat,yhat,"r")
ylim((-300,400))
