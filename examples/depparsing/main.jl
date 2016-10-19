using Merlin

type Token
    form::Int
    cat::Int
    head::Int
end

function main()
    worddict = IntDict{String}()
    catdict = IntDict{String}()
    #unkform = worddict["UNKNOWN"]
    function encode(items::Vector{String})
        form0 = replace(items[2], r"[0-9]", '0')
        #formid = get(worddict, lowercase(form0), unkform)
        formid = push!(worddict, lowercase(form0))
        catid = push!(catdict, items[5])
        head = parse(Int, items[7])
        Token(formid, catid, head)
    end
    train_x = CoNLL.read(encode, ".data/wsj_02-21.conll")
    test_x = CoNLL.read(encode, ".data/wsj_23.conll")

    train_x = map(State, train_x)
    test_x = map(State, test_x)
    for s in train_x
        beamsearch(s, 10, x->0.0)
    end
end

include("intdict.jl")
include("io.jl")
include("beamsearch.jl")
include("state.jl")

main()
