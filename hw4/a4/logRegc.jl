include("misc.jl")
include("findMin.jl")
#include("findMinb.jl")

function logReg(X,y)

	(n,d) = size(X)

	# Initial guess
	w = zeros(d,1)

	# Function we're going to minimize (and that computes gradient)
	funObj(w) = logisticObj(w,X,y)

	# Solve least squares problem
	w = findMin(funObj,w,derivativeCheck=true)

	# Make linear prediction function
	predict(Xhat) = sign.(Xhat*w)

	# Return model
	return LinearModel(predict,w)
end

function logisticObj(w,X,y)
	yXw = y.*(X*w)
	f = sum(log.(1 + exp.(-yXw)))
	g = -X'*(y./(1+exp.(yXw)))
	return (f,g)
end

# Multi-class one-vs-all version (assumes y_i in {1,2,...,k})
function logRegOnevsAll(X,y)
	(n,d) = size(X)
	k = maximum(y)

	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)

	for c in 1:k
		yc = ones(n,1) # Treat class 'c' as +1
		yc[y .!= c] = -1 # Treat other classes as -1

		# Each binary objective has the same features but different lables
		funObj(w) = logisticObj(w,X,yc)

		W[:,c] = findMin(funObj,W[:,c],verbose=false)
	end

	# Make linear prediction function
	predict(Xhat) = mapslices(indmax,Xhat*W,2)#Xhat:n by d; W: d by k

	return LinearModel(predict,W)
end



function softmaxClassifier(X,y) #softmaxClassifier  softmax
	(n,d) = size(X)
	k = maximum(y)
    @printf("k is %d\n",k)
    @printf("d is %d\n",d)
	# Each column of 'w' will be a logistic regression classifier
	W = zeros(d,k)#zeros(d,k)
	@printf("initial W:\n")
	@show(W)
	Wr=reshape(W, d*k, 1)
	@show(Wr)


	funObj(Wr) = softmaxObjb(Wr,X,y, k)  #softmaxObjb softmaxObjd softmaxObjc originally funObj(wr)  lower capital w fucked me
	Wr = findMin(funObj,Wr,derivativeCheck=true,verbose=false) #previous findMin is questionable

	W=reshape(Wr, d, k)
	(nn1,dd1)=size(W)
	# @printf("row of w is %d\n",nn1)
	# @printf("col of w is %d\n",dd1)
 	# @show(W)
	# for tt in 1:n


	# 	for c in 1:k
	# 		# yc = ones(n,1) # Treat class 'c' as +1
	# 		# yc[y .!= c] = -1 # Treat other classes as -1

	# 		# # Each binary objective has the same features but different lables
	# 		# funObj(w) = logisticObj(w,X,yc)

	# 		# W[:,c] = findMin(funObj,W[:,c],verbose=false)
	# 		#exp(x[tt,:]*W[:,c][1,1])

	# 	end



	# end
	# Make linear prediction function
	predict(Xhat) = mapslices(indmax,Xhat*W,2)#Xhat:n by d; W: d by k

	return LinearModel(predict,W)
end




function softmaxObj(wr,X,y,k)
    (n,d) = size(X)
    W=reshape(wr,d, k)

    tmp1=X*W

    denom=sum( exp.(tmp1),2  )
    #exp.(tmp1)
    #sum( exp.(tmp1),2  )
    term2=sum(log. ( sum( exp.(tmp1),2  ) )  )


    term1=0
    for i in 1:n
	    term1=term1+sum(X[i,:].*W[:,y[i]])
	end
	term1=-term1


	f=term1+term2



	grad=zeros(d,k)

	for j in 1:d


		for c in 1:k

			left=  sum(-X[ y[:].==c, j])


			right=0
			for i in 1:n

				right=right+exp( sum( X[i,:].*W[:,c]  )  ) * X[i,j] / ( denom[i]  ) 

			end



			grad[j,c]=left+right
		end


	end





	g=reshape(grad, d*k, 1)

	return (f,g)
    # sumlogexp=0
    # for jj in 1:n

    # 	sumexp=0
    # 	for mm in 1:k
    		
    # 		sumexp=sumexp+exp(tmp[jj, mm])

    # 	end

    # 	sumlogexp=sumlogexp+log(sumexp)

    # end


	# yXw = y.*(X*w)
	# f = sum(log.(1 + exp.(-yXw)))
	# g = -X'*(y./(1+exp.(yXw)))
	# return (f,g)
end





function softmaxObjb(wr,X,y,k)
    (n,d) = size(X)
    W=reshape(wr,d, k)

    tmp1=X*W

    exw=exp.(X*W)

    denom=sum(exw,2)
    #exp.(tmp1)
    #sum( exp.(tmp1),2  )
    term2=sum(log. ( denom )  )


    term1=0
    for i in 1:n
	    term1=term1+sum(X[i,:].*W[:,y[i]])  #tmp1[i, y[i]]
	end
	term1=-term1


	f=term1+term2



	grad=zeros(d,k)

	for j in 1:d


		for c in 1:k

			left= sum(-X[ y[:].==c, j])
			right=0


			for i in 1:n

				# if y[i].==c
				# left=left-X[ i, j]
				# end


				right=right+exw[i,c] * X[i,j] / ( denom[i]  )  #exp( sum( X[i,:].*W[:,c]  )  ) 

			end



			grad[j,c]=left+right
		end


	end





	g=reshape(grad, d*k, 1)

	return (f,g)
    # sumlogexp=0
    # for jj in 1:n

    # 	sumexp=0
    # 	for mm in 1:k
    		
    # 		sumexp=sumexp+exp(tmp[jj, mm])

    # 	end

    # 	sumlogexp=sumlogexp+log(sumexp)

    # end


	# yXw = y.*(X*w)
	# f = sum(log.(1 + exp.(-yXw)))
	# g = -X'*(y./(1+exp.(yXw)))
	# return (f,g)
end




function softmaxObjc(wr,X,y,k)
    (n,d) = size(X)
    W=reshape(wr,d, k)

    tmp1=X*W

    exw=exp.(X*W)

    denom=sum(exw,2)
    #exp.(tmp1)
    #sum( exp.(tmp1),2  )
    term2=sum(log. ( denom )  )


    term1=0
    for i in 1:n
	    term1=term1+sum(X[i,:].*W[:,y[i]])  #tmp1[i, y[i]]
	end
	term1=-term1


	f=term1+term2



	grad=zeros(d,k)

	# for j in 1:d


	# 	for c in 1:k

	# 		left= sum(-X[ y[:].==c, j])
	# 		right=0


	# 		for i in 1:n

	# 			# if y[i].==c
	# 			# left=left-X[ i, j]
	# 			# end


	# 			right=right+exw[i,c] * X[i,j] / ( denom[i]  )  #exp( sum( X[i,:].*W[:,c]  )  ) 

	# 		end



	# 		grad[j,c]=left+right
	# 	end


	# end

	for c in 1:k
	    eachK = zeros(n,d)
	    for i in 1:n
	        Xi = X[i,:]
	        
	        numerator = exp(W[:,c]' * Xi) * Xi'
	        denominator = sum(exp.(Xi'* W));

	        eachK[i,:] = - Xi'* (y[i] .== c) + numerator / denominator
	    end

	    grad[:,c] = sum(eachK,1);
	end






	g=reshape(grad, d*k, 1)

	return (f,g)
    # sumlogexp=0
    # for jj in 1:n

    # 	sumexp=0
    # 	for mm in 1:k
    		
    # 		sumexp=sumexp+exp(tmp[jj, mm])

    # 	end

    # 	sumlogexp=sumlogexp+log(sumexp)

    # end


	# yXw = y.*(X*w)
	# f = sum(log.(1 + exp.(-yXw)))
	# g = -X'*(y./(1+exp.(yXw)))
	# return (f,g)
end





function softmaxObjd(wr,X,y,k)
    (n,d) = size(X)
    k=maximum(y)
    W=reshape(wr,d, k)

    # tmp1=X*W

    # exw=exp.(X*W)

    # denom=sum(exw,2)
    # #exp.(tmp1)
    # #sum( exp.(tmp1),2  )
    # term2=sum(log. ( denom )  )


 #    term1=0
 #    for i in 1:n
	#     term1=term1+sum(X[i,:].*W[:,y[i]])  #tmp1[i, y[i]]
	# end
	# term1=-term1


	# f=term1+term2

	#Compute loss
	f = 0
	for i in 1:n
	    yi = y[i]      # the k
	    Xi = X[i,:]    # 1 x 3

	    #       (3 x 1)' (1 x 3)'   sum((1 x 3) (3 x 5))
	    f = f - W[:,yi]' * Xi + log(sum(exp.(Xi'*W)))
	end


	grad=zeros(d,k)

	# for j in 1:d


	# 	for c in 1:k

	# 		left= sum(-X[ y[:].==c, j])
	# 		right=0


	# 		for i in 1:n

	# 			# if y[i].==c
	# 			# left=left-X[ i, j]
	# 			# end


	# 			right=right+exw[i,c] * X[i,j] / ( denom[i]  )  #exp( sum( X[i,:].*W[:,c]  )  ) 

	# 		end



	# 		grad[j,c]=left+right
	# 	end


	# end

	for c in 1:k
	    eachK = zeros(n,d)
	    for i in 1:n
	        Xi = X[i,:]
	        
	        numerator = exp(W[:,c]' * Xi) * Xi'
	        denominator = sum(exp.(Xi'* W))

	        eachK[i,:] = -Xi'* (y[i] .== c) + numerator / denominator
	    end 

	     (kr,kc)=size(eachK)
	      #@printf("rows of k and cols of k: %d, %d \n", kr, kc)
	    grad[:,c] = sum(eachK,1)

	    #@show(grad[:,c])
	end






	g=reshape(grad, d*k, 1)
	#@show(g)

	return (f,g)
    # sumlogexp=0
    # for jj in 1:n

    # 	sumexp=0
    # 	for mm in 1:k
    		
    # 		sumexp=sumexp+exp(tmp[jj, mm])

    # 	end

    # 	sumlogexp=sumlogexp+log(sumexp)

    # end


	# yXw = y.*(X*w)
	# f = sum(log.(1 + exp.(-yXw)))
	# g = -X'*(y./(1+exp.(yXw)))
	# return (f,g)
end





function softmaxObje(wr,X,y,k)
    (n,d) = size(X)
    k=maximum(y)
    W=reshape(wr,d, k)
    #@printf("below is reshaped W in the beginning of softmaxobje:\n")
    #@show(W)
    # tmp1=X*W

    # exw=exp.(X*W)

    # denom=sum(exw,2)
    # #exp.(tmp1)
    # #sum( exp.(tmp1),2  )
    # term2=sum(log. ( denom )  )


 #    term1=0
 #    for i in 1:n
	#     term1=term1+sum(X[i,:].*W[:,y[i]])  #tmp1[i, y[i]]
	# end
	# term1=-term1


	# f=term1+term2

	#Compute loss
	f = 0
	for i in 1:n
	    yi = y[i]      # the k
	    Xi = X[i,:]'    # 1 x 3

	    #       (3 x 1)' (1 x 3)'   sum((1 x 3) (3 x 5))
	    f = f - W[:,yi]' * Xi' + log(sum(exp.(Xi*W)))
	end


	grad=zeros(d,k)

	# for j in 1:d


	# 	for c in 1:k

	# 		left= sum(-X[ y[:].==c, j])
	# 		right=0


	# 		for i in 1:n

	# 			# if y[i].==c
	# 			# left=left-X[ i, j]
	# 			# end


	# 			right=right+exw[i,c] * X[i,j] / ( denom[i]  )  #exp( sum( X[i,:].*W[:,c]  )  ) 

	# 		end



	# 		grad[j,c]=left+right
	# 	end


	# end

	for c in 1:k
	    eachK = zeros(n,d)
	    for i in 1:n
	        Xi = X[i,:]'
	        
	        numerator = exp(W[:,c]' * Xi') * Xi
	        denominator = sum(exp.(Xi* W))
	        # @printf("xi: \n")
	        # @show(Xi)
	        # @show(numerator)
	        # @show(denominator )
	        # @show(numerator / denominator)
	        eachK[i,:] = -Xi* (y[i] .== c) + numerator / denominator
	    end 

	     (kr,kc)=size(eachK)
	      #@printf("rows of k and cols of k: %d, %d \n", kr, kc)
	    grad[:,c] = sum(eachK,1)
	    #@show(sum(eachK,1))
	    #@show(grad[:,c])
	end


	#@show(grad)



	g=reshape(grad, d*k, 1)
	#@show(g)

	return (f,g)
    # sumlogexp=0
    # for jj in 1:n

    # 	sumexp=0
    # 	for mm in 1:k
    		
    # 		sumexp=sumexp+exp(tmp[jj, mm])

    # 	end

    # 	sumlogexp=sumlogexp+log(sumexp)

    # end


	# yXw = y.*(X*w)
	# f = sum(log.(1 + exp.(-yXw)))
	# g = -X'*(y./(1+exp.(yXw)))
	# return (f,g)
end





# function softmaxClassifier(X,y)

# 	(n,d)=size(X)
# 	k=maximum(y)



# 	w=zeros(d*k)

# 	funObj(w)=softmaxObjn(X,w,y)

# 	w=findMin(funObj,w,derivativeCheck=true)

# 	W=reshape(w,d,k)

# 	predict(Xhat)=mapslices(indmax,Xhat*W,2)

# 	return LinearModel(predict,W)
# end


# function softmaxObjn(X,w,y)

# 	(n,d)=size(X)
# 	k=maximum(y)
# 	W=reshape(w,d,k)

# 	f=0.0

# 	g=zeros(d*k)

# 	for i=1:n
# 		f+=-W[:,y[i]]'*X[i,:]+log(sum(exp.(W'*X[i,:])))

# 	end


# 	for j=1:d
# 		for c=1:k
# 			g[(c-1)*d+j]+=-(y.==c)'*X[:,j]


# 			for i=1:n

# 				g[(c-1)*d+j]+=X[i,j]*exp((W[:,c])'*X[i,:])/sum(exp.(W'*X[i,:]))

# 			end
# 		end
# 	end


# 	return f,g

# end