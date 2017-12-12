# Load X and y variable
using JLD
X = load("citiesSmall.jld","X")
y = load("citiesSmall.jld","y")
n = size(X,1)

# Fit a decision tree and compute error
include("decisionTree.jl")
depth = 2
model = decisionTree(X,y,depth)

# Evaluate training error
yhat = model.predict(X)
trainError = sum(yhat .!= y)/n
@printf("Error with depth-%d decision tree: %.3f\n",depth,trainError)

sptmodel=model.splitModel
tst=sptmodel.splitVariable
@printf("splitvariable: %.2f\n",tst)
#@printf("Error with equality-rule decision stump: %.2f\n",trainError);


# Plot classifier
include("plot2Dclassifier.jl")
plot2Dclassifier(X,y,model)
