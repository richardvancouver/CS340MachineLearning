# Load X and y variable
using JLD
data = load("outliersData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

nn=size(X)



vv=zeros(nn)

nn[1]

for ii in 1:nn[1]

	vv[ii]=1

	if ii.>400
		vv[ii]=0.1
	end
    
end






# Fit a least squares model
include("leastSquaresf.jl")
model = WeightedLeastSquares(X,y,vv)

# Evaluate training error
yhat = model.predict(X)
trainError = mean((yhat - y).^2)
@printf("Squared train Error with weighted least squares: %.3f\n",trainError)

# Evaluate test error
yhat = model.predict(Xtest)
testError = mean((yhat - ytest).^2)
@printf("Squared test Error with weighted least squares: %.3f\n",testError)

# Plot model
using PyPlot
figure()
plot(X,y,"b.")
Xhat = minimum(X):.01:maximum(X)
yhat = model.predict(Xhat)
plot(Xhat,yhat,"g")
