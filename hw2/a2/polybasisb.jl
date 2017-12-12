include("misc.jl")

function polybasis(X,c,d)
    @printf("poly")
	Xb=hcat(ones(size(X)[1]),X)

	for ii in 2:c
		Xb=hcat(Xb,  X.^ii)
	end

	for jj in 1:d
		Xb=hcat(Xb,  sin.( (1)*jj*X))
	end	


	for jj in 1:d
		Xb=hcat(Xb,  cos.( (1)*jj*X))
	end	

	# Return Polynomials
	return Xb
end