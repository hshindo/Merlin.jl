using Merlin
using MLDatasets

include("iddict.jl")

function model(nvocabs::Int)
    T = Float32
    #filename = joinpath(Pkg.dir("JukaiNLP"), ".corpus/nyt100.h5")
    #embed = Embedding(h5read(filename,"vec"))
    embed = Embedding(T, nvocabs, 100)
    ls = [Linear(T,300,100), Linear(T,100,nvocabs)]
    f = @graph begin
        x = :x
        x = embed(x)
        x = ls[1](x)
        x = relu(x)
        x = ls[2](x)
        x
    end
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
        window(x, (3,), pads=(2,))
    end
    test_x = map(test_x) do x
        x = unshift!(x, 1, 1)
        window(x, (3,), pads=(2,))
    end
    train_c = sum(map(length, train_x))

    #test_x = map(x -> map(xx -> get(dict,xx,0), x), test_x)
    #foreach(x -> append!(dict,x), train_x)

    #trainurl = "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt"
    #testurl = "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt"
    #dict = IdDict(String["<BOS>","<EOS>"])
    #train_x, train_y = readdata(dict, trainurl)
    #test_x = readdata(dict, testurl)
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
