#type dtreeprediction
#	dtreeprediction # Function that makes predictions
	#split # Function that splits data
	#baseSplit # Set this to one stump doesn't split
#end







function test2(X)


		
		# Make a predict function
function dtreeprediction(X)

(n,d) = size(X)

yhat = zeros(n)


	for i in 1:n

		if (X[i,2].>=37.695206)

   			if (X[i, 1].>=-96.032692)
       				yhat[i]=2
    			else 
				yhat[i]=1
    			end
 


		else

   			if (X[i, 1].<=-112.548331)
       				yhat[i]=2
    			else 
				yhat[i]=1
    			end


		end



	end
	#return GenericModel(dtreeprediction)
	return yhat
end

		return GenericModel(dtreeprediction)
	
end

	
