using Merlin
using ProgressMeter
using JLD2, FileIO

include("downloader.jl")

function main()
    nepochs = 10
    training = true
    datapath = joinpath(dirname(@__FILE__), ".data")
    traindata = setup_data(get_traindata(datapath)...)
    testdata = setup_data(get_testdata(datapath)...)
    if training
        model = setup_model()
        train(traindata, testdata, model, nepochs)
        save("mnist_epoch$(nepochs).jld2", "model", model)
    else
        model = load("mnist_epoch3.jld2","model")

    end
end

function setup_data(x::Matrix{Float32}, y::Vector{Int})
    batchsize = 200
    xs = [x[:,i:i+batchsize-1] for i=1:batchsize:size(x,2)]
    y += 1 # Change label set: 0..9 -> 1..10
    ys = [y[i:i+batchsize-1] for i=1:batchsize:length(y)]
    collect(zip(xs,ys))
end

function setup_model{T}(::Type{T}, hsize::Int)
    x = Node()
    h = Linear(T,28*28,hsize)(x)
    h = relu(h)
    h = Linear(T,hsize,hsize)(h)
    h = relu(h)
    h = Linear(T,hsize,10)(h)
    Graph(input=x, output=h)
end

function train(traindata::Vector, testdata::Vector, model::Graph, nepochs::Int)
    params = getparams(model)
    opt = SGD(0.001)
    for epoch = 1:nepochs
        println("epoch: $epoch")
        prog = Progress(length(traindata))
        loss = 0.0
        for (x,y) in shuffle!(traindata)
            h = model(Var(x))
            y = softmax_crossentropy(Var(y), h)
            loss += sum(y.data)
            gradient!(y)
            foreach(p -> opt(p.data,p.grad), params)
            ProgressMeter.next!(prog)
        end
        loss /= length(traindata)
        println("Loss:\t$loss")

        # test
        golds = Int[]
        preds = Int[]
        for (x,y) in testdata
            h = model(Var(x))
            z = argmax(h.data, 1)
            append!(golds, y)
            append!(preds, z)
        end
        @assert length(golds) == length(preds)
        acc = mean(i -> golds[i] == preds[i] ? 1.0 : 0.0, 1:length(golds))
        println("test accuracy: $acc")
        println()
    end
end

main()
