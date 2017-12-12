# Load X and y variable
using JLD
data = load("basisData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# (tt,nn)=size(X)
# intercept=ones(tt)

# Xnn=hcat(intercept,X)
for p in 0:10

@printf("p is %d\n", p)	

# Fit a least squares model
include("leastSquaresBasis.jl")
model = leastSquaresBasis(X,y,p)

# Evaluate training error
yhat = model.predict(X)
trainError = mean((yhat - y).^2)
@printf("Squared train Error with least squares: %.3f\n",trainError)

# Evaluate test error
# (ttb,nnb)=size(Xtest)
# intb=ones(ttb)
# Xtestnn=hcat(intb, Xtest)
yhat = model.predict(Xtest)
testError = mean((yhat - ytest).^2)
@printf("Squared test Error with least squares: %.3f\n",testError)

# Plot model
using PyPlot
#figure()
plot(X,y,"b.")
Xhat = minimum(X):.1:maximum(X)
yhat = model.predict(Xhat)
plot(Xhat,yhat,color=(0,0.1*p,0.01*p*p),label="p is $p")
legend()
xlim(-11,18)

title("polynomial fitting")

end