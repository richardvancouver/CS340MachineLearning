# Load data
dataTable = readcsv("animals.csv")
X = float(real(dataTable[2:end,2:end]))
(n,d) = size(X)
@show(d)
# Standardize columns
include("misc.jl")
(X,mu,sigma) = standardizeCols(X)

# Plot matrix as image
using PyPlot
# figure(1)
# clf()
# imshow(X)

# Show scatterplot of 2 random features
# j1 = rand(1:d)
# j2 = rand(1:d)
# figure(2)
# clf()
# plot(X[:,j1],X[:,j2],".")
# for i in rand(1:n,10)
#     annotate(dataTable[i+1,1],
# 	xy=[X[i,j1],X[i,j2]],
# 	xycoords="data")
# end


include("PCA.jl")



for kk in 2:20
	pcamodel=PCA(X,kk)

	ww=pcamodel.W
	#@show(ww)
	# (ind1,ind2)=findmax(ww[1,:])
	# @show(dataTable[1,ind2+1])

	# (ind1b,ind2b)=findmax(ww[2,:])
	# @show(ind2b)
	# @show(dataTable[1,ind2b+1])

	Z = X*ww'
	#@show(Z)
	# figure(3)
	# scatter(Z[:,1],Z[:,2]);



	ZZ=pcamodel.compress(X)
	#@show(ZZ)
	# figure(4)
	# scatter(ZZ[:,1],ZZ[:,2])
	(sz1,sz2)=size(ZZ)

	# for i in 1:sz1
	# 	#@show(i)
	#     annotate(dataTable[i+1,1],
	# 	xy=[ZZ[i,1],ZZ[i,2]],
	# 	xycoords="data")
	# end


	#F-norm of standardised X
	sigmax=sum(X.*X)/(n*d)

	#the remaining variance:
	remain=abs.(X-Z*ww)

	sigmaremain=sum(remain.*remain)/(n*d)


	remainfrac=sigmaremain/sigmax
	#@printf("the remaining variance for kk=:\n",kk)
	@show(kk)
	@show(remainfrac)
	# tt=expand(ZZ)
	# @show(tt)


end