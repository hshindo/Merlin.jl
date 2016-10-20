using Merlin

type Token
    form::Int
    cat::Int
    head::Int
end

const nulltoken = Token(0, 0, 0)

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
    trainx = CoNLL.read(encode, ".data/wsj_02-21.conll")
    trainy = map(s -> map(x -> x.head, s), trainx)
    testx = CoNLL.read(encode, ".data/wsj_23.conll")
    testy = map(s -> map(x -> x.head, s), testx)

    trainstates = map(State, trainx)
    teststates = map(State, testx)

    function f()
        for i = 1:1
            for s in trainstates
                beamsearch(s, 16, nextpred, a->0.0)
            end
        end
    end
    @time f()
    #n = beamsearch(trainstates[1], 1, nextgold, a->0.0)
end

include("intdict.jl")
include("io.jl")
include("beamsearch.jl")
include("state.jl")

main()
