using Merlin
using MLDatasets

const save_model = false

function setup_data(x::Array, y::Vector)
    x = reshape(x, size(x,1)*size(x,2), size(x,3))
    x = Matrix{Float32}(x)
    batchsize = 100
    xs = [Var(x[:,i:i+batchsize-1]) for i=1:batchsize:size(x,2)]
    y += 1 # Change label set: 0..9 -> 1..10
    ys = [Var(y[i:i+batchsize-1]) for i=1:batchsize:length(y)]
    xs, ys
end

function main(nepochs::Int)
    train_x, train_y = setup_data(MNIST.traindata()...)
    test_x, test_y = setup_data(MNIST.testdata()...)
    model = Model(Float32, 1000)

    opt = SGD(0.0001, momentum=0.99, nesterov=true)
    for epoch = 1:nepochs
        println("epoch: $epoch")
        loss = fit(train_x, train_y, model, opt)
        println("loss: $loss")

        # predict
        ys = cat(1, map(data, test_y)...)
        zs = cat(1, map(model, test_x)...)
        acc = mean(i -> ys[i] == zs[i] ? 1.0 : 0.0, 1:length(ys))
        println("test accuracy: $acc")
        println()
        if save_model && epoch == nepochs
            Merlin.save("mnist_epoch$(nepochs).h5", "g"=>model.g)
        end
    end
end

include("model.jl")
main(10)
