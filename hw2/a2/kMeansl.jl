include("misc.jl")
# include("clustering2Dplot.jl")

# type PartitionModel
# 	predict # Function for clustering new points
# 	y # Cluster assignments
# 	W # Prototype points
# end

# function kMeans(X,k;doPlot=false)
# # K-means clustering

# (n,d) = size(X)

# # Choos random points to initialize means
# W = zeros(k,d)
# perm = randperm(n)
# for c = 1:k
# 	W[c,:] = X[perm[c],:]
# end

# # Initialize cluster assignment vector
# y = zeros(n)
# changes = n

# while changes != 0

# 	# Compute (squared) Euclidean distance between each point and each mean
# 	D = distancesSquared(X,W)

# 	# Assign each data point to closest mean (track number of changes labels)
# 	changes = 0
# 	for i in 1:n
# 		(~,y_new) = findmin(D[i,:])
# 		changes += (y_new != y[i])
# 		y[i] = y_new
# 	end

# 	# Optionally visualize the algorithm steps
# 	if doPlot && d == 2
# 		clustering2Dplot(X,y,W)
# 		sleep(.1)
# 	end


# 	# Find mean of each cluster
# 	for c in 1:k
# 		W[c,:] = mean(X[y.==c,:],1)
# 	end

# 	# Optionally visualize the algorithm steps
# 	if doPlot && d == 2
# 		clustering2Dplot(X,y,W)
# 		sleep(.1)
# 	end

# 	@printf("Running k-means, changes = %d\n",changes)
# end



	# for i in 1:n
	# 	#(~,y_new) = findmin(D[i,:])
	# 	#changes += (y_new != y[i])
	# 	#y[i] = y_new
	# 	D[i, y[i]]
	# end

# function errof()

# 	erro=0
# 	for c in 1:k
# 		#t1=0
# 		for val in (y.==c)
# 			erro=erro+sum( (W[c,:]-X[val,:]).^2 )
# 		end

# 	end

# 	return erro
# end








# function errofb()
# 	(t,d)=size(w)
# 	erro=zeros(t)
# 	for c in 1:k
# 		#t1=0
# 		for val in (y.==c)
# 			erro[c]=erro[c]+(W[c,:]-X[val,:]).^2
# 		end


# 	end

# 	return sum(erro[:])

# end



# function KMeansError(X,y,W)
# 	(n,dd)=size(X)

# 	D = distancesSquared(X,W)

# 	erro=0
	
# 	for i in 1:n
# 		erro=erro+D[i, y[i]]
# 	end
	
# 	return erro
# end




function KMeansErrorl(X,y,W)
	(n,dd)=size(X)

	#D = distancesSquared(X,W)

	erro=0
	
	for i in 1:n

		for jj in 1:dd
		erro=erro+abs(X[i,jj]-W[y[i], jj])
		end

	end
	
	return erro
end



# function predict(Xhat)
# 	(t,d) = size(Xhat)

# 	D = distancesSquared(Xhat,W)

# 	yhat = zeros(Int64,t)
# 	for i in 1:t
# 		(~,yhat[i]) = findmin(D[i,:])
# 	end
# 	return yhat
# end

# return PartitionModel(predict,y,W)
# end



