function setup_data()
    words = h5read(h5file, "s")
    worddict = Dict(words[i] => i for i=1:length(words))
    chardict = Dict{Char,Int}()
    tagdict = Dict{String,Int}()

    #traindata = UD_English.traindata()
    #testdata = UD_English.testdata()
    traindoc = CoNLL.read(".data/wsj_00-18.conll")
    testdoc = CoNLL.read(".data/wsj_22-24.conll")
    info("# sentences of train doc: $(length(traindoc))")
    info("# sentences of test doc: $(length(testdoc))")

    traindata = setup_data(traindoc, worddict, chardict, tagdict)
    testdata = setup_data(testdoc, worddict, chardict, tagdict)
    info("# words: $(length(worddict))")
    info("# chars: $(length(chardict))")
    info("# tags: $(length(tagdict))")
    traindata, testdata, worddict, chardict, tagdict
end

function setup_data(doc::Vector, worddict, chardict, tagdict)
    data = []
    unkword = worddict["UNKNOWN"]
    for sent in doc
        w = Int[]
        cs = Var[]
        t = Int[]
        for items in sent
            word, tag = items[2], items[5]
            word0 = replace(word, r"[0-9]", '0')
            wordid = get(worddict, lowercase(word0), unkword)
            chars = Vector{Char}(word0)
            charids = map(c -> get!(chardict,c,length(chardict)+1), chars)
            tagid = get!(tagdict, tag, length(tagdict)+1)
            push!(w, wordid)
            push!(cs, Var(charids))
            push!(t, tagid)
        end
        push!(data, (Var(w),cs,Var(t)))
    end
    data
end
