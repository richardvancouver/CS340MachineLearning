include("misc.jl")
include("polybasis.jl")
function leastSquaresBasis(X,y,c)

	# Xb=hcat(ones(size(X)[1]),X)

	# for ii in 2:c
	# 	Xb=hcat(Xb,  X.^ii)
	# end
	Xb=polybasis(X,c)

	# Find regression weights minimizing squared error
	w = (Xb'*Xb)\(Xb'*y)

	# Make linear prediction function
	#predict(Xhat) = Xhat*w
	predict(Xhat) =polybasis(Xhat,c)*w
	# Return model
	return GenericModel(predict)
end