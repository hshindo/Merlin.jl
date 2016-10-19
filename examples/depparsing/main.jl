using Merlin
using MLDatasets

function main()
    traindata = CoNLL.read(x -> Token(x[2],x[5],), ".data/wsj_02-21.conll")
    testdata = CoNLL.read(".data/wsj_23.conll")
    train_x = map(s -> map(x -> x[2], s), traindata)
    train_y = map(s -> map(x -> x[5], s), traindata)
    test_x = map(s -> map(x -> x[2], s), testdata)
    test_y = map(s -> map(x -> x[5], s), testdata)

    worddict = IntDict{String}()
    catdict = IntDict{String}()
end

#include("beamsearch.jl")
include("state.jl")

main()
