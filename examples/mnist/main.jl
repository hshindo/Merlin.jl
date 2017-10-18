using Merlin
using ProgressMeter

include("downloader.jl")
include("model.jl")

function setup_data(x::Array, y::Vector)
    x = reshape(x, size(x,1)*size(x,2), size(x,3))
    x = Matrix{Float32}(x)
    batchsize = 200
    xs = [x[:,i:i+batchsize-1] for i=1:batchsize:size(x,2)]
    y += 1 # Change label set: 0..9 -> 1..10
    ys = [y[i:i+batchsize-1] for i=1:batchsize:length(y)]
    collect(zip(xs,ys))
end

function train(model::Model, nepochs::Int)
    datapath = joinpath(dirname(@__FILE__), ".data")
    traindata = setup_data(get_traindata(datapath)...)
    testdata = setup_data(get_testdata(datapath)...)

    opt = SGD(0.001)
    for epoch = 1:nepochs
        println("epoch: $epoch")

        prog = Progress(length(traindata))
        loss = 0.0
        for i in randperm(length(traindata))
            x, y = traindata[i]
            h = model(x)
            y = softmax_crossentropy(y, h)
            loss += sum(y.data)
            params = gradient!(y)
            foreach(p -> opt(p.data,p.grad), params)
            ProgressMeter.next!(prog)
        end
        loss /= length(traindata)
        println("Loss:\t$loss")

        # test
        gold = Int[]
        pred = Int[]
        for i = 1:length(testdata)
            x, y = testdata[i]
            h = model(x)
            z = argmax(h.data, 1)
            append!(gold, y)
            append!(pred, z)
        end
        continue
        #loss = fit(train_x, train_y, model, opt)
        #println("loss: $loss")

        ys = cat(1, map(x -> vec(x.data), test_y)...)
        zs = cat(1, map(x -> vec(model(x).data), test_x)...)
        acc = mean(i -> ys[i] == zs[i] ? 1.0 : 0.0, 1:length(ys))
        println("test accuracy: $acc")
        println()
    end
    Merlin.save("mnist_epoch$(nepochs).h5", "model"=>model)
end

model = Model(Float32, 800)
#model = Merlin.load(joinpath(dirname(@__FILE__),"mnist_epoch10.h5"), "model")
train(model, 10)
