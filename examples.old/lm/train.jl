workspace()
using Merlin
using JukaiNLP
using HDF5

function readdata(dict::IdDict, url)
    sents = open(readlines, download(url))
    data_x = map(sents) do str
        words = split(str, " ")
        pop!(words) # remove "\n"
        append!(dict, words)
    end
    data_y = map(data_x) do x
        y = Int[]
        for i = 1:length(x)-1
            push!(y, x[i+1])
        end
        push!(y, dict["<EOS>"])
        y
    end
    data_x, data_y
end

function model(nvocabs)
    T = Float32
    #filename = joinpath(Pkg.dir("JukaiNLP"), ".corpus/nyt100.h5")
    #embed = Embedding(h5read(filename,"vec"))
    embed = Embedding(T, nvocabs, 100)
    ls = [Linear(T,300,100), Linear(T,100,nvocabs)]
    g = @graph begin
        x = embed(:x)
        x = ls[1](x)
        x = relu(x)
        x = ls[2](x)
        x
    end
    f = compile(g, :x)
    function forward(x::Vector{Int})
        ids = Array(Int, 3, length(x))
        for i = 1:length(x)
            c = 1
            for j = i-2:i
                ids[c,i] = j <= 0 ? 1 : x[j]
                c += 1
            end
        end
        f(Var(ids))
    end
    forward
end

function train()
    trainurl = "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt"
    testurl = "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt"
    dict = IdDict(String["<BOS>","<EOS>"])
    train_x, train_y = readdata(dict, trainurl)
    #test_x = readdata(dict, testurl)
    f = model(length(dict))

    opt = SGD(0.01)
    for epoch = 1:100
        println("epoch: $(epoch)")
        loss = fit(f, crossentropy, opt, train_x, train_y)
        println("loss: $(loss)")
        println("")
    end
end
