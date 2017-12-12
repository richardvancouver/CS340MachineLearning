include("misc.jl")
#include("polybasisb.jl")
function leastSquares(X,y,v)

	# Xb=hcat(ones(size(X)[1]),X)

	# for ii in 2:c
	# 	Xb=hcat(Xb,  X.^ii)
	# end
	#Xb=polybasis(X,c,d)

	# Find regression weights minimizing squared error
	w = ((X.*v)'*X)\((X.*v)'*y)

	# Make linear prediction function
	#predict(Xhat) = Xhat*w
	predict(Xhat) = Xhat*w
	#predict(Xhat) = hcat(ones(size(Xhat)[1]),Xhat)*w
	# Return model
	return GenericModel(predict)
end