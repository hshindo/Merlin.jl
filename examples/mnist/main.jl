using Merlin
using MLDatasets

const load_model = false
const write_model = false

function setup_model(input, output)
    T = Float32
    h = 1000 # hidden vector size
    x = Var()
    y = Linear(T,784,h)(x)
    y = relu(y)
    y = Linear(T,h,h)(y)
    y = relu(y)
    y = Linear(T,h,10)(y)
    compile(y, x)
end

function accuracy(ys::Vector{Int}, zs::Vector{Int})
    @assert length(y) == length(z)
    c = count(i -> y[i] == z[i], 1:length(y))
    c / length(y)
end

function setup_data(x::Array, y::Vector)
    x = reshape(x, size(x,1)*size(x,2), size(x,3))
    x = Matrix{Float32}(x)
    xs = [Var(x[:,i:i+99]) for i=1:100:size(x,2)]
    y += 1 # Change label set: [0,1,2,...,9] -> [1,2,...,10]
    ys = [Var(y[i:i+99]) for i=1:100:length(y)]
    xs, ys
end

function main()
    train_x, train_y = setup_data(MNIST.traindata()...)
    test_x, test_y = setup_data(MNIST.testdata()...)
    train_model = setup_model()

    opt = SGD(0.005)
    for epoch = 1:10
        println("epoch: $epoch")
        for i in randperm(length(train_x))

        end

        #loss = fit(xtrains, ytrains, nn, crossentropy, opt)
        #for i in randperm(length())
        #end

        println("Loss: $(loss)")
        # predict
        zs = map(xtests) do x
            out = nn(x).data
            argmax(out, 1)
        end
        acc = accuracy(cat(1,ys...), cat(1,zs...))
        println("Test accuracy: $(acc)")
        write_model && Merlin.save("mnist.h5", epoch==1 ? "w" : "r+", string(epoch), nn)
    end
end

main()
