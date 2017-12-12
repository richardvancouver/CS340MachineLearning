# Load X and y variable
using JLD
X = load("citiesBig2.jld","Xtest")
y = load("citiesBig2.jld","ytest")

Xtest = load("citiesBig1.jld","X")
ytest  = load("citiesBig1.jld","y")
#Xtest = load("citiesBig2.jld","Xtest")
#ytest = load("citiesBig2.jld","ytest")



# Fit a KNN classifier
k = 1
include("knnb.jl")
model = cknn(X,y,k)

# Evaluate training error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@printf("Train Error with %d-nearest neighbours: %.3f\n",k,trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test Error with %d-nearest neighbours: %.3f\n",k,testError)


include("plot2Dclassifier.jl")
plot2Dclassifier(X,y,model)

