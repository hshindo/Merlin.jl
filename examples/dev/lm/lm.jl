using Merlin
using MLDatasets

include("iddict.jl")

function model(nvocabs::Int)
    T = Float32
    x = Var()
    y = Embedding(T,nvocabs,100)(x)
    y = Linear(T,300,100)(y)
    y = relu(y)
    y = Linear(T,100,nvocabs)(y)
    Graph(y, x)
end

function train()
    train_x, train_y = PTBLM.traindata()
    test_x, test_y = PTBLM.testdata()
    dict = IdDict(["<bos>","<eos>"])
    train_x = map(x -> append!(dict,x), train_x)
    train_y = map(y -> append!(dict,y), train_y)
    test_x = map(x -> append!(dict,x), test_x)
    test_y = map(y -> append!(dict,y), test_y)

    train_x = map(train_x) do x
        x = unshift!(x, 1, 1)
        window(x, (3,), pad=2)
    end
    test_x = map(test_x) do x
        x = unshift!(x, 1, 1)
        window(x, (3,), pad=2)
    end
    train_c = sum(map(length, train_x))

    f = model(length(dict))
    opt = SGD(0.01)
    for epoch = 1:10
        println("epoch: $(epoch)")
        loss = fit(train_x, train_y, f, crossentropy, opt)
        ppl = 2^(loss/train_c)
        println("LOSS: $(loss)")
        println("PPL: $(ppl)")
        println("")
    end
end

train()
