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
        x = embed(:x)
        x = ls[1](x)
        x = relu(x)
        x = ls[2](x)
        x
    end
end

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

function train()
    train_x, train_y = PTBLM.traindata()
    test_x, test_y = PTBLM.testdata()
    dict = IdDict{String}()
    foreach(x -> append!(dict,x), train_x)
    push!(dict, "<eos>")



    #trainurl = "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt"
    #testurl = "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt"
    #dict = IdDict(String["<BOS>","<EOS>"])
    #train_x, train_y = readdata(dict, trainurl)
    #test_x = readdata(dict, testurl)
    #f = model(length(dict))

    opt = SGD(0.01)
    for epoch = 1:10
        println("epoch: $(epoch)")
        #loss = fit(f, crossentropy, opt, train_x, train_y)
        #println("loss: $(loss)")
        println("")
    end
end

function perplexity(probs::Vector{Float64})
    mapreduce()
end

train()
