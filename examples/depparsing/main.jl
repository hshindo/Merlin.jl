using Merlin
using MLDatasets
using HDF5
using ProgressMeter

const h5file = ".data/wordembeds_nyt100.h5"

function main()
    traindata, testdata, worddict, chardict = setup_data()
    println(length(testdata))
    traindata = traindata[1:10000]
    wordembeds = h5read(h5file, "v")
    charembeds = rand(Float32, 10, 100)
    model = Model2(wordembeds)

    opt = SGD(0.0001, momentum=0.99, nesterov=true)
    #opt = SGD(0.001)
    for epoch = 1:30
        println("epoch: $epoch")
        prog = Progress(length(traindata))
        loss = 0.0
        for (w,cs,y) in shuffle(traindata)
            z = model(w, y)
            loss += sum(z.data)
            vars = gradient!(z)
            for v in vars
                #clipnorm!(v.grad, 0.1)
                opt(v.data, v.grad)
            end
            next!(prog)
        end
        loss /= length(traindata)
        println("loss: $loss")

        ys = cat(1, map(x -> x[3].data[2:end], testdata)...) # remove ROOT head
        zs = cat(1, map(x -> model(x[1])[2:end], testdata)...)
        @assert length(ys) == length(zs)
        acc = mean(i -> ys[i] == zs[i] ? 1.0 : 0.0, 1:length(ys))
        println("test acc.: $acc")
        println("")
    end
end

include("data.jl")
include("model.jl")
main()
