include("misc.jl")
#include("polybasisb.jl")
function WeightedLeastSquares(X,y,v)

	# Xb=hcat(ones(size(X)[1]),X)

	# for ii in 2:c
	# 	Xb=hcat(Xb,  X.^ii)
	# end
	#Xb=polybasis(X,c,d)
	
	#Xb=X.*sqrt.(v)
	#yb=y.*sqrt.(v)

	VV=diagm(v)

	# Find regression weights minimizing squared error
	#w = ((Xb)'*Xb)\((Xb)'*yb)
	w = ((X)'*VV*X)\((X)'*VV*y)

	# Make linear prediction function
	#predict(Xhat) = Xhat*w
	predict(Xhat) = Xhat*w
	#predict(Xhat) = hcat(ones(size(Xhat)[1]),Xhat)*w
	# Return model
	return GenericModel(predict)
end