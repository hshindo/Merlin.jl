using Merlin
using MLDatasets
using ProgressMeter

const load_model = false
const write_model = false

function setup_data(x::Array, y::Vector)
    x = reshape(x, size(x,1)*size(x,2), size(x,3))
    x = Matrix{Float32}(x)
    batchsize = 100
    xs = [Var(x[:,i:i+batchsize-1]) for i=1:batchsize:size(x,2)]
    y += 1 # Change label set: [0,1,2,...,9] -> [1,2,...,10]
    ys = [Var(y[i:i+batchsize-1]) for i=1:batchsize:length(y)]
    collect(zip(xs,ys))
end

function main()
    traindata = setup_data(MNIST.traindata()...)
    testdata = setup_data(MNIST.testdata()...)
    model = setup_model()

    opt = SGD(0.005)
    for epoch = 1:10
        println("epoch: $epoch")
        prog = Progress(length(traindata))
        loss = 0.0
        for (x,y) in shuffle(traindata)
            z = model(x, y)
            loss += sum(z.data)
            vars = gradient!(z)
            foreach(v -> opt(v.data,v.grad), vars)
            next!(prog)
        end
        loss /= length(traindata)
        println("loss: $loss")

        # predict
        ys = cat(1, map(x -> x[2].data, testdata)...)
        zs = cat(1, map(x -> model(x[1]), testdata)...)
        acc = mean(i -> ys[i] == zs[i] ? 1.0 : 0.0, 1:length(ys))
        println("test accuracy: $acc")
        println()
        #write_model && Merlin.save("mnist.h5", epoch==1 ? "w" : "r+", string(epoch), nn)
    end
end

include("model.jl")
main()
