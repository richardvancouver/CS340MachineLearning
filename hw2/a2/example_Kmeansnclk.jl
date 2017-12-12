# Load data
using JLD
X = load("clusterData2.jld","X")
(t,d)=size(X)
@printf("size %d, %d\n",t,d)


elbows=zeros(10)

for k in 1:10


@printf("k is %d\n",k)
yy=zeros(t, 50)
ww=zeros(k, d, 50)
erary=zeros(50)
include("kMeansm.jl")
include("kMeansl.jl")

for  kk in 1:50
		# K-means clustering
		
		#include("kMeans.jl")
		model = kMeans(X,k,doPlot=false)
		y = model.predict(X)

		#include("clustering2Dplot.jl")
		#clustering2Dplot(X,y,model.W)



			# D = distancesSquared(X,model.W)

			# erro=0

			# for i in 1:n
			# 	erro=erro+D[i, y[i]]
			# end
			
		#include("kMeansc.jl")
		yy[:, kk]=y
		ww[:,:,kk]=model.W
		err=KMeansErrorl(X,y,model.W)	
		#@printf("Error %.3f\n",err)
		erary[kk]=err



end

		(aa, bb)=findmin(erary)
		elbows[k]=aa
		@printf("Minimum run is %d, error is %.3f\n",bb, aa)
		ww[:,:,bb]
		include("clustering2Dplot.jl")
		clustering2Dplot(X,yy[:,bb],ww[:,:,bb])





end

figure()
plot(1:10,elbows)