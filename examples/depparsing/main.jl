using Merlin
using MLDatasets
using HDF5
using ProgressMeter

const h5file = ".data/wordembeds_nyt100.h5"

function main()
    traindata, testdata, worddict, chardict = setup_data()
    traindata = traindata[1:5000]
    wordembeds = h5read(h5file, "v")
    charembeds = rand(Float32, 10, 100)
    model = Model(wordembeds)

    opt = SGD()
    for epoch = 1:10
        println("epoch: $epoch")
        prog = Progress(length(traindata))
        opt.rate = 0.0075 / epoch
        loss = 0.0
        for (w,cs,y) in shuffle(traindata)
            z = model(w, y)
            loss += sum(z.data)
            vars = gradient!(z)
            foreach(v -> opt(v.data,v.grad), vars)
            next!(prog)
        end
        loss /= length(traindata)
        println("loss: $loss")

        ys = cat(1, shift!(map(x -> x[3].data, testdata))...) # remove ROOT head
        zs = cat(1, shift!(map(x -> model(x[1]), testdata))...)
        acc = mean(i -> ys[i] == zs[i] ? 1.0 : 0.0, 1:length(ys))
        println("test acc.: $acc")
        println("")
    end
end

include("data.jl")
include("model.jl")
main()
