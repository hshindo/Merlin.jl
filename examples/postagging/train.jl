using Merlin
using HDF5

const wordembeds_file = "wordembeds_nyt100.h5"

function train()
    words = h5read(wordembeds_file, "s")
    worddict = Dict(words[i] => i for i=1:length(words))
    chardict = Dict{Char,Int}(' ' => 1)
    tagdict = Dict{String,Int}()
    #traindoc = UD_English.traindata()
    #testdoc = UD_English.testdata()
    train_w, train_c, train_t = readdata(".data/wsj_00-18.conll", worddict, chardict, tagdict)
    test_w, test_c, test_t = readdata(".data/wsj_22-24.conll", worddict, chardict, tagdict)
    info("# training sentences:\t$(length(train_w))")
    info("# testing sentences:\t$(length(test_w))")
    info("# words:\t$(length(worddict))")
    info("# chars:\t$(length(chardict))")
    info("# tags:\t$(length(tagdict))")

    nn = Model(length(tagdict))
    opt = SGD()
    for epoch = 1:10
        println("Epoch: $epoch")
        opt.rate = 0.0075 / epoch

        function train_f(data::Tuple)
            w, c, t = data
            y = nn(w, c)
            softmax_crossentropy(t, y)
        end
        train_data = collect(zip(train_w, train_c, train_t))
        loss = minimize!(train_f, opt, train_data)
        println("Average loss: $loss")

        # test
        function test_f(data::Tuple)
            w, c = data
            y = nn(w, c)
            vec(argmax(y.data,1))
        end
        test_data = collect(zip(test_w, test_c))
        ys = cat(1, map(t -> t.data, test_t)...)
        zs = cat(1, map(test_f, test_data)...)
        length(ys) == length(zs) || throw("Length mismatch.")

        acc = mean(i -> ys[i] == zs[i] ? 1.0 : 0.0, 1:length(ys))
        acc = round(acc, 5)
        println("Test acc.:\t$acc")
        println()
    end
    Merlin.save("postagger.h5", nn)
end

include("data.jl")
include("model.jl")
train()
