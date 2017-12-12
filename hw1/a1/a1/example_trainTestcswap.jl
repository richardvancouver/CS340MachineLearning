# Load X and y variable
using JLD

using PyCall
using PyPlot


Xtot = load("citiesSmall.jld","X")
ytot = load("citiesSmall.jld","y")
szx = size(Xtot,1)


Xtest=Xtot[1:(floor(Int,szx[1]/2)),:]
ytest=ytot[1:(floor(Int,szx[1]/2)),:]
@show(size(Xtest))

X=Xtot[(floor(Int,szx[1]/2)+1):(floor(Int,szx[1])),:]
y=ytot[(floor(Int,szx[1]/2)+1):(floor(Int,szx[1])),:]

@show(size(X))


Maxdepth=15
# Train a depth-2 decision tree
trerror=zeros(Maxdepth)
teserror=zeros(Maxdepth)
dd=zeros(Maxdepth)
include("decisionTree_infoGain.jl")

for depth in 1:Maxdepth
	model = decisionTree_infoGain(X,y,depth)

	# Evaluate the trianing error
	yhat = model.predict(X)
	sx=size(X,1)
	trainError = sum(yhat .!= y)/sx
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




