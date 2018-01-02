using Merlin
using Merlin.Datasets.MNIST
using ProgressMeter
using JLD2, FileIO
using LibCUDA
setdevice(0)

const BACKEND = "CUDA:0" # or "CPU"
const NEPOCHS = 50

function main()
    trainmode = true
    datapath = joinpath(@__DIR__, ".data")
    traindata = setup_data(MNIST.traindata(datapath)...)
    testdata = setup_data(MNIST.testdata(datapath)...)
    savefile = "mnist_epoch$(NEPOCHS).jld2"
    if trainmode
        model = setup_model()
        train(traindata, testdata, model)
        save(savefile, "model", model)
    else
        model = load(savefile, "model")
        test(model, testdata)
    end
end

function setup_data(x::Matrix{Float32}, y::Vector{Int})
    batchsize = 200
    xs = [cu(x[:,i:i+batchsize-1]) for i=1:batchsize:size(x,2)]
    y += 1 # Change label set: 0..9 -> 1..10
    #ys = [y[i:i+batchsize-1] for i=1:batchsize:length(y)]
    ys = [cu(Vector{Cint}(y[i:i+batchsize-1])) for i=1:batchsize:length(y)]
    collect(zip(xs,ys))
end

function setup_model()
    T = Float32
    hsize = 1000
    x = Node()
    h = Linear(T,28*28,hsize)(x)
    h = relu(h)
    h = Linear(T,hsize,hsize)(h)
    h = relu(h)
    h = Linear(T,hsize,10)(h)
    Graph(x, h, backend=BACKEND)
end

function train(traindata::Vector, testdata::Vector, model)
    # params = getparams(model)
    opt = SGD(0.001)
    for epoch = 1:NEPOCHS
        println("epoch: $epoch")
        prog = Progress(length(traindata))
        loss = 0.0
        for (x,y) in shuffle!(traindata)
            h = model(Var(x))
            z = softmax_crossentropy(Var(y), h)
            loss += sum(Array(z.data))
            #loss += sum(z.data)
            params = gradient!(z)
            foreach(opt, params)
            ProgressMeter.next!(prog)
        end
        loss /= length(traindata)
        println("Loss:\t$loss")

        test(model, testdata)
        println()
    end
end

function test(model, data::Vector)
    golds = Int[]
    preds = Int[]
    for (x,y) in data
        h = model(Var(x))
        z = argmax(Array(h.data), 1)
        #z = argmax(h.data, 1)
        append!(golds, Array(y))
        append!(preds, z)
    end
    @assert length(golds) == length(preds)
    acc = mean(i -> golds[i] == preds[i] ? 1.0 : 0.0, 1:length(golds))
    println("test accuracy: $acc")
end

main()
