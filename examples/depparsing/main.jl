using Merlin
using MLDatasets

export CoNLL
module CoNLL

function read(f, path::String)
    doc = []
    sent = []
    lines = open(readlines, path)
    for line in lines
        line = chomp(line)
        if length(line) == 0
            length(sent) > 0 && push!(doc, sent)
            sent = []
        else
            items = Vector{String}(split(line,'\t'))
            push!(sent, f(items))
        end
    end
    length(sent) > 0 && push!(doc, sent)
    T = typeof(doc[1][1])
    Vector{Vector{T}}(doc)
end
read(path::String) = read(identity, path)

end

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
