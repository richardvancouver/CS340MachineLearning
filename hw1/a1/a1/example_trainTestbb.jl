# Load X and y variable
using JLD

using PyCall
using PyPlot


X = load("citiesSmall.jld","X")
y = load("citiesSmall.jld","y")
n = size(X,1)


Xtest = load("citiesSmall.jld","Xtest")
ytest = load("citiesSmall.jld","ytest")


Maxdepth=15
# Train a depth-2 decision tree
trerror=zeros(Maxdepth)
teserror=zeros(Maxdepth)
dd=zeros(Maxdepth)
include("decisionTree.jl")

for depth in 1:Maxdepth
	model = decisionTree(X,y,depth)

	# Evaluate the trianing error
	yhat = model.predict(X)
	trainError = sum(yhat .!= y)/n
	trerror[depth]=trainError
	dd[depth]=depth
	@printf("Train error with depth-%d decision tree: %.3f\n",depth,trainError)

	# Evaluate the test error
	#Xtest = load("citiesSmall.jld","Xtest")
	#ytest = load("citiesSmall.jld","ytest")
	t = size(Xtest,1)
	yhat = model.predict(Xtest)
	testError = sum(yhat .!= ytest)/t
	teserror[depth]=testError
	@printf("Test error with depth-%d decision tree: %.3f\n",depth,testError)



end


	figure()
	plot(dd,trerror,"b+")
	plot(dd,teserror,"ro")




