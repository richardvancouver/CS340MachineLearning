include("misc.jl") # Includes GenericModel typedef

function naiveBayes(X,y)
	# Implementation of naive Bayes classifier for binary features

	(n,d) = size(X)

  # Compute number of classes, assuming y in {1,2,...,k}
  k = maximum(y)

  # We will store p(y(i) = c) in p_y(c)
  counts = zeros(k)
  for i in 1:n
    counts[y[i]] += 1
  end
  p_y = counts ./n  
  @show(counts)
  # We will store p(x(i,j) = 1 | y(i) = c) in p_xy(1,j,c)
  # We will store p(x(i,j) = 0 | y(i) = c) in p_xy(2,j,c)
  #p_xy = (1/2)ones(2,d,k)
  p_xy = zeros(2,d,k)
  fd=zeros(d)
  fd2=zeros(d)
	for kk in 1:k

  fd=zeros(d)
  fd2=zeros(d)
  		for dd in 1:d
  			for i in 1:n

				if y[i].==kk && X[i,dd].==1.0
				fd[dd]+= 1 
				elseif y[i].==kk && X[i,dd].!=1.0
				fd2[dd]+=1
				end	       			
			end
			
		end
		p_xy[1,:,kk]=fd/counts[kk]
		p_xy[2,:,kk]=fd2/counts[kk]
	end
  @show p_xy[2,:,1]


 # fd2=zeros(d)
#	for kk in 1:k
 # 		for dd in 1:d
  #			for i in 1:n

#				if X[y[i].==kk,dd].==1
#				fd2[dd]+= 1 
#				
#				end	       			
#			end
			
#		end
#		p_xy[1,:,kk]=fd/counts[kk]
#	end


  function predict(Xhat)
    (t,d) = size(Xhat)
    yhat = zeros(t)

    for i in 1:t
      # p_yx = p_y*prod(p_xy) for the appropriate x and y values
      p_yx = copy(p_y)
      for j in 1:d
        if Xhat[i,j] == 1
          for c in 1:k
            p_yx[c] *= p_xy[1,j,c]
          end
        else
          for c in 1:k
            p_yx[c] *= p_xy[2,j,c]
          end
        end
      (~,yhat[i]) = findmax(p_yx)
      end
    end
    return yhat
  end

	return GenericModel(predict)
end
