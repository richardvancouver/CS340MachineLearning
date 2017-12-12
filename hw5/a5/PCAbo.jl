include("misc.jl")
include("findMin.jl")

function PCA(X,k)
    (n,d) = size(X)

    # Subtract mean
    mu = mean(X,1)
    X -= repmat(mu,n,1)

    (U,S,V) = svd(X)
    W = V[:,1:k]'

    compress(Xhat) = compressFunc(Xhat,W,mu)
    expand(Z) = expandFunc(Z,W,mu)

    return CompressModel(compress,expand,W)
end

function compressFunc(Xhat,W,mu)
    (t,d) = size(Xhat)
    Xcentered = Xhat - repmat(mu,t,1)
    return Xcentered*W' # Assumes W has orthogonal rows
end

function expandFunc(Z,W,mu)
    (t,k) = size(Z)
    return Z*W + repmat(mu,t,1)
end

function PCA_gradient(X,k)
    (n,d) = size(X)

    # Subtract mean
    mu = mean(X,1)
    X -= repmat(mu,n,1)

    # Initialize W and Z
    W = randn(k,d)
    Z = randn(n,k)

    R = Z*W - X
    f = sum(R.^2)
    funObjZ(z) = pcaObjZ(z,X,W)
    funObjW(w) = pcaObjW(w,X,Z)
    # funObjZ(z) = pcaObjZr(z,X,W)
    # funObjW(w) = pcaObjWr(w,X,Z)
    for iter in 1:50
        fOld = f

        # Update Z
        Z[:] = findMin(funObjZ,Z[:],verbose=false,maxIter=10)

        # Update W
        W[:] = findMin(funObjW,W[:],verbose=false,maxIter=10)

        R = Z*W - X
        f = sum(R.^2)
        @printf("Iteration %d, loss = %f\n",iter,f/length(X))

        if (fOld - f)/length(X) < 1e-2
            break
        end
    end


    # We didn't enforce that W was orthogonal so we need to optimize to find Z
    compress(Xhat) = compress_gradientDescent(Xhat,W,mu)
    expand(Z) = expandFunc(Z,W,mu)

    return CompressModel(compress,expand,W)
end

#
function robustPCA(X,k)
    (n,d) = size(X)

    # Subtract mean
    mu = mean(X,1)
    X -= repmat(mu,n,1)

    # Initialize W and Z
    W = randn(k,d)
    Z = randn(n,k)

    R = Z*W - X
    #f = sum(R.^2)

    ZW=Z*W
    su=0
    for i in 1:n

        for j in 1:d

            su=su+exp( ZW[i,j]-X[i,j])


        end

    end

    f=log(su)
    # funObjZ(z) = pcaObjZ(z,X,W)
    # funObjW(w) = pcaObjW(w,X,Z)
    funObjZ(z) = pcaObjZr(z,X,W)
    funObjW(w) = pcaObjWr(w,X,Z)
    for iter in 1:50
        fOld = f

        # Update Z
        Z[:] = findMin(funObjZ,Z[:],verbose=false,maxIter=10)

        # Update W
        W[:] = findMin(funObjW,W[:],verbose=false,maxIter=10)

        R = Z*W - X
        #f = sum(R.^2)

        ZW=Z*W
        su=0
        for i in 1:n

            for j in 1:d

                su=su+exp( ZW[i,j]-X[i,j])


            end

        end

        f=log(su)


        @printf("Iteration %d, loss = %f\n",iter,f/length(X))

        if (fOld - f)/length(X) < 1e-2
            break
        end
    end


    # We didn't enforce that W was orthogonal so we need to optimize to find Z
    compress(Xhat) = compress_gradientDescent(Xhat,W,mu)
    expand(Z) = expandFunc(Z,W,mu)

    return CompressModel(compress,expand,W)
end

function compress_gradientDescent(Xhat,W,mu)
    (t,d) = size(Xhat)
    Xcentered = Xhat - repmat(mu,t,1)
    Z = zeros(t,k)

    funObj(z) = pcaObjZr(z,Xcentered,W)
    Z[:] = findMin(funObj,Z[:],verbose=false)
    return Z
end


function pcaObjZ(z,X,W)
    # Rezie vector of parameters into matrix
    n = size(X,1)
    k = size(W,1)
    Z = reshape(z,n,k)

    # Comptue function value
    R = Z*W - X
    f = sum(R.^2)

    # Comptue derivative with respect to each residual
    dR = R

    # Multiply by W' to get elements of gradient
    G = dR*W'

    # Return function and gradient vector
    return (f,G[:])
end

function pcaObjW(w,X,Z)
    # Rezie vector of parameters into matrix
    d = size(X,2)
    k = size(Z,2)
    W = reshape(w,k,d)

    # Comptue function value
    R = Z*W - X
    f = sum(R.^2)

    # Comptue derivative with respect to each residual
    dR = R

    # Multiply by Z' to get elements of gradient
    G = Z'dR

    # Return function and gradient vector
    return (f,G[:])
end




function pcaObjZr(z,X,W)
    # Rezie vector of parameters into matrix
    n = size(X,1)
    k = size(W,1)
    Z = reshape(z,n,k)

    # Comptue function value
    R = Z*W - X
    #f = sum(R.^2)
    ZW=Z*W
    su=0
    for i in 1:n

        for j in 1:d

            su=su+exp( ZW[i,j]-X[i,j])


        end

    end

    f=log(su)





    # Comptue derivative with respect to each residual
    dR = R #fix gradient for logsum case

    # Multiply by W' to get elements of gradient
    G = dR*W' #fix gradient for logsum case

    # Return function and gradient vector
    return (f,G[:])
end

function pcaObjWr(w,X,Z)
    # Rezie vector of parameters into matrix
    d = size(X,2)
    k = size(Z,2)
    W = reshape(w,k,d)

    # Comptue function value
    R = Z*W - X
    #f = sum(R.^2)
    ZW=Z*W
    su=0
    for i in 1:n

        for j in 1:d

            su=su+exp( ZW[i,j]-X[i,j])


        end

    end

    f=log(su)
    # Comptue derivative with respect to each residual
    dR = R #fix gradient for logsum case

    # Multiply by Z' to get elements of gradient
    G = Z'dR #fix gradient for logsum case

    # Return function and gradient vector
    return (f,G[:])
end

