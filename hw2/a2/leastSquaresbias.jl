include("misc.jl")

function leastSquaresBias(X,y)

	Xb=hcat(ones(size(X)[1]),X)
	# Find regression weights minimizing squared error
	w = (Xb'*Xb)\(Xb'*y)

	# Make linear prediction function
	#predict(Xhat) = Xhat*w
	predict(Xhat) = hcat(ones(size(Xhat)[1]),Xhat)*w
	# Return model
	return GenericModel(predict)
end