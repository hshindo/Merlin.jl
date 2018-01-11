using Merlin
using Merlin.Datasets.MNIST
using ProgressMeter
using JLD2, FileIO
using LibCUDA
setdevice(0)

const BACKEND = CUDABackend(0) # or CPUBackend()
const NEPOCHS = 50

function main()
    trainmode = true
    datapath = joinpath(@__DIR__, ".data")
    traindata = setup_data(MNIST.traindata(datapath)...)
    testdata = setup_data(MNIST.testdata(datapath)...)
    savefile = "mnist_epoch$(NEPOCHS).jld2"
    if trainmode
        # model = setup_model()
        train(traindata, testdata)
        #save(savefile, "model", model)
    else
        model = load(savefile, "model")
        test(model, testdata)
    end
end

function setup_data(x::Matrix{Float32}, y::Vector{Int})
    batchsize = 200
    xs = [x[:,i:i+batchsize-1] for i=1:batchsize:size(x,2)]
    y += 1 # Change label set: 0..9 -> 1..10
    y = Vector{Int32}(y)
    ys = [y[i:i+batchsize-1] for i=1:batchsize:length(y)]
    collect(zip(xs,ys))
end

mutable struct NN <: Graph
    l1
    l2
    lout
end

function NN()
    T = Float32
    hsize = 1000
    l1 = Linear(T, 28*28, hsize, BACKEND)
    l2 = Linear(T, hsize, hsize, BACKEND)
    lout = Linear(T, hsize, 10, BACKEND)
    NN(l1, l2, lout)
end

function (nn::NN)(x, y=nothing)
    h = relu(nn.l1(x))
    h = relu(nn.l2(h))
    h = nn.lout(h)
    if y == nothing
        h
    else
        softmax_crossentropy(y, h)
    end
end

function train(traindata::Vector, testdata::Vector)
    nn = NN()
    params = getparams(nn)
    opt = SGD(0.001)
    trainiter = BatchIterator(traindata, 1, backend=BACKEND)
    testiter = BatchIterator(testdata, 1, backend=BACKEND, shuffle=false)
    #allocator = LibCUDA.GreedyAllocator()
    #LibCUDA.setallocator(allocator)

    for epoch = 1:NEPOCHS
        println("epoch: $epoch")
        prog = Progress(length(traindata))
        loss = 0.0

        for (x,y) in trainiter
            z = nn(Var(x),Var(y))
            loss += sum(Array(z.data))
            gradient!(z)
            foreach(opt, params)
            ProgressMeter.next!(prog)
            #LibCUDA.free(allocator)
        end
        loss /= length(traindata)
        println("Loss:\t$loss")

        test(nn, testiter)
        #LibCUDA.free(allocator)
        println()
    end
end

function test(nn, iter::BatchIterator)
    golds = Int[]
    preds = Int[]
    for (x,y) in iter
        append!(golds, Array(y))
        z = nn(Var(x))
        z = argmax(Array(z.data), 1)
        append!(preds, z)
    end
    @assert length(golds) == length(preds)
    acc = mean(i -> golds[i] == preds[i] ? 1.0 : 0.0, 1:length(golds))
    println("test accuracy: $acc")
end

@time main()
