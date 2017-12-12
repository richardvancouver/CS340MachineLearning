# Load data
using HDF5
using JLD

fileName = "vowel.jld"
X = load(fileName,"X")
y = load(fileName,"y")
Xtest = load(fileName,"Xtest")
ytest = load(fileName,"ytest")

#added on 171013Friday 22:00
# include("misc.jl")
# (aaa,ddd)=size(X)
# @printf("size of x: %d, %d\n", aaa,ddd)



# Fit a decision tree classifier
include("decisionTree.jl")
depth = Inf
model = decisionTree(X,y,depth)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with depth-%d decision tree: %.3f\n",depth,trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with depth-%d decision tree: %.3f\n",depth,testError)

# Fit a random tree classifier
include("decisionTree.jl")
depth = Inf#5
model = randomTree(X,y,depth)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with depth-%d random tree: %.3f\n",depth,trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with depth-%d random tree: %.3f\n",depth,testError)




# Fit a random tree classifier
include("decisionTree.jl")
depth = Inf
nTree=50
model = randomForest(X,y,depth, nTree)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with nTree-%d depth-%d random forest: %.3f\n",nTree, depth,trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with nTree-%d depth-%d random forest: %.3f\n",nTree, depth,testError)