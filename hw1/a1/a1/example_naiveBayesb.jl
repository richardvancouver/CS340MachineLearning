# Load variables
include("example_BagOfWords.jl")

# Compute test error with naive Bayes
include("naiveBayesb.jl")
model = naiveBayes(X,y)
yhat = model.predict(Xtest)
testError = mean(yhat .!= ytest)
@printf("Test error with naive Bayes: %.3f\n",testError)
