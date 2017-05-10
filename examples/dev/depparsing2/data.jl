function setup_data()
    words = h5read(h5file, "s")
    worddict = Dict(words[i] => i for i=1:length(words))
    chardict = Dict{Char,Int}()

    traindoc = CoNLL.read(".data/wsj_00-18.conll")
    testdoc = CoNLL.read(".data/wsj_22-24.conll")
    info("# sentences of train doc: $(length(traindoc))")
    info("# sentences of test doc: $(length(testdoc))")

    traindata = setup_data(traindoc, worddict, chardict)
    testdata = setup_data(testdoc, worddict, chardict)
    info("# words: $(length(worddict))")
    info("# chars: $(length(chardict))")
    traindata, testdata, worddict, chardict
end

function setup_data(doc::Vector, worddict, chardict)
    data = []
    unkword = worddict["UNKNOWN"]
    root = worddict["PADDING"]
    for sent in doc
        w = Int[root]
        cs = Var[Var([0])]
        h = Int[0]
        for items in sent
            word = items[2]
            word0 = replace(word, r"[0-9]", '0')
            wordid = get(worddict, lowercase(word0), unkword)

            chars = Vector{Char}(word0)
            charids = map(c -> get!(chardict,c,length(chardict)+1), chars)

            head = parse(Int, items[7]) + 1
            push!(w, wordid)
            push!(cs, Var(charids))
            push!(h, head)
        end
        push!(data, (Var(w),cs,Var(h)))
    end
    data
end
