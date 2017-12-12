# Load data
#using JLD
#A = load("clusterData.jld","X")

using PyPlot
A=imread("/home/rui/Desktop/cs340/hw2/a2/dog.png")

#X=Imread("dog.png")

(aa,bb,cc)=size(A)
@printf("size %d, %d, %d \n",aa,bb,cc)


#convert A to X
counta=0
numofrows=aa
numofcols=bb
X=zeros(numofrows*numofcols, 3)
for r in 1:numofrows
  for c in 1:numofcols
        counta=counta+1
         X[counta, :]=A[r,c,:]
 end
end

X

# K-means clustering
b=2#2
@printf("b %d\n",b)
k = 2^b
include("kMeans3d.jl")
model = kMeans(X,k,doPlot=true)
y = model.predict(X)

WW=model.W



#convert X to B
B=zeros(aa,bb,cc)
countb=0
numofrows=aa
numofcols=bb
#X=zeros(numofrows*numofcols)
for r in 1:numofrows
  for c in 1:numofcols
        countb=countb+1
         B[r,c, :]=WW[y[countb],:]
 end
end

B

imshow(B)

# #Imsave(B,"dogb.png")
# #Imshow(B)

# #include("clustering2Dplot.jl")
# #clustering2Dplot(X,y,model.W)



# 	# D = distancesSquared(X,model.W)

# 	# erro=0

# 	# for i in 1:n
# 	# 	erro=erro+D[i, y[i]]
# 	# end
	
include("kMeansc.jl")

err=KMeansError(X,y,model.W)	
@printf("Error %.3f\n",err)