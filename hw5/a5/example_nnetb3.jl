# Load X and y variable
using JLD
using PyPlot
include("misc.jl")
data = load("basisData.jld")
(X,y) = (data["X"],data["y"])
n = size(X,1)
(X,mu,sigma) = standardizeCols(X)
#(y,mu1,sigma1) = standardizeCols(y)
@show(mu,sigma)
Xold=X
X = [ones(n,1) X]
d = 2
@show(X)
# Choose network structure and randomly initialize weights
include("NeuralNet.jl")
nHidden = [40 180 40]#[65 203 65]#[27 40 27]#[15]#[27 40 27]#[15]
nParams = NeuralNet_nParams(d,nHidden)
w = randn(nParams,1)



Xt=2*minimum(Xold):0.05:maximum(Xold)*2
Xt=[ones(length(Xt),1) Xt]
@show(Xt)

# Train with stochastic gradient
maxIter = 10000*3
stepSize = 1e-4*5
for t in 1:maxIter
    # if t > maxIter/2
    #     stepSize = stepSize * (1-2/maxIter)^2;
    # end
    wOld = w
    wOldOld= wOld
	# The stochastic gradient update:

	trials=5
	sg=0
	for kk in 1:trials
		i = rand(1:n)
		(f,g) = NeuralNet_backprop(w,X[i,:],y[i],nHidden)
		sg=sg+g
	end
	g=sg/trials


	w = w - stepSize*g + stepSize^2*(w-wOldOld)

	# Every few iterations, plot the data/model:
	if (mod(t-1,round(maxIter/50)) == 0)
		@printf("Training iteration = %d\n",t-1)
		figure(1)
		clf()
		#Xhat = -10:.05:10
		#yhat = NeuralNet_predict(w,[ones(length(Xhat),1) Xhat],nHidden)
        
  #       Xb=minimum(X):0.05:maximum(X)
		# Xbar=[ones(length(Xb),1) Xb]
		# @show(Xbar)

		Xhat=X
		yhat = NeuralNet_predict(w,Xhat,nHidden)


		yt=NeuralNet_predict(w,Xt,nHidden)


		plot(X[:,2],y,".")
		#plot(Xhat[:,2],yhat,"g-")

		plot(Xt[:,2],yt,"g-")

		sleep(.1)
	end
end


