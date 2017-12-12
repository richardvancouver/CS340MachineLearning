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
figure(1)
clf()
imshow(X)

# Show scatterplot of 2 random features
j1 = rand(1:d)
@show(X[:,j1])
@show(X[:,j2])
j2 = rand(1:d)
figure(2)
clf()
plot(X[:,j1],X[:,j2],".")

# annotate(dataTable[2+1,1],
#  	xy=[X[2,j1],X[2,j2]],
#  	xycoords="data")


for i in rand(1:n,10)
	  @show(i)
	  @show(dataTable[i+1,1])
	  #
	 @show([X[i,j1],X[i,j2]]) 
	annotate("hello",
	xy=[X[i,j1],X[i,j2]],
	xycoords="data")
 #    annotate(dataTable[i+1,1],
	# xy=[X[i,j1],X[i,j2]],
	# xycoords="data")
end
