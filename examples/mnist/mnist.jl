using Merlin
using MLDatasets

function create_model()
    T = Float32
    h = 1000 # hidden vector size
    g = begin
        x = Var()
        x = Linear(T,784,h)(x)
        x = relu(x)
        x = Linear(T,h,h)(x)
        x = relu(x)
        x = Linear(T,h,10)(x)
        Graph(x)
    end
    g
end

function flatten(xs::Vector{Vector{Int}})
    dest = Int[]
    for x in xs
        append!(dest, x)
    end
    dest
end

function accuracy(ys::Vector{Vector{Int}}, zs::Vector{Vector{Int}})
    y, z = flatten(ys), flatten(zs)
    @assert length(y) == length(z)
    c = count(i -> y[i] == z[i], 1:length(y))
    c / length(y)
end

function main()
    xtrain, ytrain = MNIST.traindata() # size: 28*28*60000, 60000
    xtest, ytest = MNIST.testdata() # size: 28*28*10000, 10000
    xtrain = reshape(xtrain, 784, 60000)
    xtest = reshape(xtest, 784, 10000)
    ytrain = ytrain + 1 # Change labelset: [0,1,2,...,9] -> [1,2,...,10]
    ytest = ytest + 1

    # Create mini-batch
    xtrains = [constant(Matrix{Float32}(xtrain[:,(i-1)*100+1:i*100])) for i=1:600]
    ytrains = [ytrain[(i-1)*100+1:i*100] for i = 1:600]
    xtests = [constant(Matrix{Float32}(xtest[:,(i-1)*100+1:i*100])) for i=1:100]
    ytests = [ytest[(i-1)*100+1:i*100] for i = 1:100]

    nn = create_model()
    for epoch = 1:20
        println("Epoch: $(epoch)")
        loss = fit(xtrains, ytrains, nn, crossentropy, SGD(0.01))
        println("Loss: $(loss)")
        # predict
        zs = map(xtests) do x
            out = nn(x).data
            argmax(out, 1)
        end
        acc = accuracy(ytests, zs)
        println("Test accuracy: $(acc)")
    end
end

main()
