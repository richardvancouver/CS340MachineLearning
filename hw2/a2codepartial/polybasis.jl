include("misc.jl")

function polybasis(X,c)
    #@printf("poly\n")
	Xb=hcat(ones(size(X)[1]),X)

	for ii in 2:c
		Xb=hcat(Xb,  X.^ii)
	end


	# Return Polynomials
	return Xb
end