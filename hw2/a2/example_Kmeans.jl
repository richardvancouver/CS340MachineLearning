# Load data
using JLD
X = load("clusterData.jld","X")
(t,d)=size(X)
@printf("size %d, %d\n",t,d)
# K-means clustering
k = 4
include("kMeans.jl")
model = kMeans(X,k,doPlot=true)
y = model.predict(X)

include("clustering2Dplot.jl")
clustering2Dplot(X,y,model.W)



	# D = distancesSquared(X,model.W)

	# erro=0

	# for i in 1:n
	# 	erro=erro+D[i, y[i]]
	# end
	
include("kMeansc.jl")

err=KMeansError(X,y,model.W)	
@printf("Error %.3f\n",err)