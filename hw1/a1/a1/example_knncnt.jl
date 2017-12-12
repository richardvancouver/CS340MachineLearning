# Load X and y variable
using JLD
Xtot = load("citiesBig2.jld","Xtest")
ytot = load("citiesBig2.jld","ytest")

szx=size(Xtot)

nn=szx[1]

X=Xtot[1:(floor(Int,szx[1]/2)),:]
y=ytot[1:(floor(Int,szx[1]/2)),:]


Xtest=Xtot[(floor(Int,szx[1]/2)+1):(floor(Int,szx[1])),:]
ytest=ytot[(floor(Int,szx[1]/2)+1):(floor(Int,szx[1])),:]

#Xtest = load("citiesBig1.jld","X")
#ytest  = load("citiesBig1.jld","y")
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

