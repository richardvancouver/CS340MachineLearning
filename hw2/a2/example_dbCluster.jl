# Load data
using JLD
X = load("clusterData2.jld","X")

# Density-based Clustering
radius = 3 #1
minPts = 15 #2
@printf("radius %d, minPnts %d\n",radius,minPts)
include("dbCluster.jl")
y = dbCluster(X,radius,minPts,doPlot=true)

include("clustering2Dplot.jl")
clustering2Dplot(X,y)