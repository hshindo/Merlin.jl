using Merlin
using Merlin.Datasets.MNIST

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
    xs = map(BACKEND, xs)
    y += 1 # Change label set: 0..9 -> 1..10
    y = Array{Int32}(y)
    ys = [y[i:i+batchsize-1] for i=1:batchsize:length(y)]
    ys = map(BACKEND, ys)
    collect(zip(xs,ys))
end

function NN()
    T = Float32
    hsize = 1000
    h = Linear(T,28*28,hsize)(Node(name="x"))
    h = relu(h)
    h = Linear(T,hsize,hsize)(h)
    h = relu(h)
    h = Linear(T,hsize,10)(h)
    BACKEND(Graph(h))
end

function train(traindata::Vector, testdata::Vector)
    nn = NN()
    opt = SGD(0.001)

    for epoch = 1:NEPOCHS
        println("epoch: $epoch")
        prog = Progress(length(traindata))
        loss = 0.0

        for (x,y) in shuffle!(traindata)
            z = nn(Var(x))
            z = softmax_crossentropy(Var(y), z)
            loss += sum(z.data)
            params = gradient!(z)
            foreach(opt, params)
            ProgressMeter.next!(prog)
        end
        loss /= length(traindata)
        println("Loss:\t$loss")

        golds = Int32[]
        preds = Int32[]
        for (x,y) in testdata
            append!(golds, Array(y))
            z = nn(Var(x))
            z = argmax(Array(z.data), 1)
            append!(preds, z)
        end
        @assert length(golds) == length(preds)
        acc = mean(i -> golds[i] == preds[i] ? 1.0 : 0.0, 1:length(golds))
        println("test accuracy: $acc")
        println()
    end
end

@time main()
