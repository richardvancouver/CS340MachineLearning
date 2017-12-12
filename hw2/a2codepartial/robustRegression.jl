include("misc.jl")
include("findMin.jl")

function robustRegression(X,y)

	(n,d) = size(X)

	# Initial guess
	w = zeros(d,1)

	# Function we're going to minimize (and that computes gradient)
	funObj(w) = robustRegressionObj(w,X,y)

	# This is how you compute the function and gradient:
	(f,g) = funObj(w)

	# Derivative check that the gradient code is correct:
	g2 = numGrad(funObj,w)

	if maximum(abs.(g-g2)) > 1e-4
		@printf("User and numerical derivatives differ:\n")
		@show([g g2])
	else
		@printf("User and numerical derivatives agree\n")
	end

	# Solve least squares problem
	w = findMin(funObj,w)

	# Make linear prediction function
	predict(Xhat) = Xhat*w

	# Return model
	return GenericModel(predict)
end

function robustRegressionObj(w,X,y)
	Xw = X*w
	f = sum(       log.( exp.(Xw - y) + exp.(-Xw + y)  )       )

	# (nnn1,ddd1)=size(w)#exp.(Xw - y)
	# @printf("w %d, %d\n", nnn1, ddd1)

	# (nnn,ddd)=size(exp.(Xw - y))#exp.(Xw - y)
	# @printf("exp %d, %d\n", nnn, ddd)
	g =  ( X.*exp.(Xw - y) -X.*exp.(-Xw + y)  )' * (   1./ ( exp.(Xw - y) + exp.(-Xw + y)  )    ) #* ( X'*exp.(Xw - y)' -X'*exp.(-Xw + y)'  )      #zeros(size(w))
	




	return (f,g)
end

