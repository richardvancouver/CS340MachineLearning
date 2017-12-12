include("misc.jl")
include("polybasisb.jl")
function leastSquares(X,y,c,d)

	# Xb=hcat(ones(size(X)[1]),X)

	# for ii in 2:c
	# 	Xb=hcat(Xb,  X.^ii)
	# end
	Xb=polybasis(X,c,d)

	# Find regression weights minimizing squared error
	w = (Xb'*Xb)\(Xb'*y)

	# Make linear prediction function
	#predict(Xhat) = Xhat*w
	predict(Xhat) =polybasis(Xhat,c,d)*w
	# Return model
	return GenericModel(predict)
end