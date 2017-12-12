include("misc.jl")
include("PCA.jl")
include("findMin.jl")

function MDS(X)
    (n,d) = size(X)

    # Compute all distances
    D = distancesSquared(X,X)
    D = sqrt.(abs.(D))
    #@show(D)
    (n1,n2)=size(D)
    #@show(n1,n2)
     ################
     knn = 2#3
     GG =fill(Inf,n1,n2) #zeros(n1,n2)    # Adjacency matrix of distances to K nearest neighbors

     #For each column of distances, get KNN
     for i in 1:n
          # Sort a column of distances from small to large
          sIndex= sortperm(D[:,i])
          sDistances=D[sIndex,i] 
          #@show(D[:,i])
          #@show(sDistances, sIndex)
          # Get the distance to the K nearest neighbors
          distances = sDistances[2:knn+1]
          neighbors = sIndex[2:knn+1]

          # Each column (~= 0) in the matrix represents the weight of edges to nodes
          GG[neighbors,i] = distances
          #@show(neighbors)
          # Make the matrix symmetrical
          GG[i,neighbors] = distances
     end
     #@show(GG)
     # Reset all distances
     D = zeros(n1,n2)

     maxD = 0

     # For each node, get the shortest distance to another node
     for i in 1:n
          for j in 1:n
               # The distance to itself is 0
               if (j != i)
                #@show(i,j,dijkstra(GG,i,j))
                    cost= dijkstra(GG,i,j) #dijkstra[G,i,j] 
                    #@show(cost)
                    if ((!isinf(cost)) & (cost > maxD) )
                         maxD = cost
                    end
                   
                    D[i,j] = cost
               end
          end
     end
         
    
    @show(D)
    
     ##################
    
     #If the distance is infinite, set it to the max
     D[isinf.(D)] = maxD


    # Initialize low-dimensional representation with PCA
    model = PCA(X,2)
    Z = model.compress(X)

    funObj(Z) = stress(Z,D)

    Z[:] = findMin(funObj,Z[:])

    return Z
end

function stress(z,D)
    n = size(D,1)
    Z = reshape(z,n,2)

    f = 0
    G = zeros(n,2)
    for i in 1:n
        for j in i+1:n
            # Objective function
            Dz = norm(Z[i,:] - Z[j,:])
            s = D[i,j] - Dz
            f = f + (1/2)s^2 #/ D[i,j]

            # Gradient
            df = s #/ D[i,j]
            dgi = (Z[i,:] - Z[j,:])/Dz
            dgj = (Z[j,:] - Z[i,:])/Dz
            G[i,:] -= df*dgi
            G[j,:] -= df*dgj
            #@show(Dz)
        end
    end
    # @show(f)
    return (f,G[:])
end

