function setup_nn(ntags::Int)
    T = Float32
    data_w, data_c = Var(), Var()

    wordembeds = h5read(wordembeds_file, "v")
    w = Lookup(wordembeds)(data_w)

    c = Lookup(T,100,10)(data_c)
    c = Conv1D(T,50,50,20,10)(c)
    c = max(c, 2)

    h = cat(1, w, c)
    h = Conv1D(T,750,300,300,150)(h)
    h = relu(h)
    h = Linear(T,300,ntags)(h)

    compile((data_w,data_c), h)
end

function setup_nn2(ntags::Int)
    T = Float32
    data_w, data_c = Var(), Var()

    wordembeds = h5read(wordembeds_file, "v")
    w = Lookup(wordembeds)(data_w)

    c = Lookup(T,100,10)(data_c)
    c = Conv1D(T,50,50,20,10)(c)
    c = max(c, 2)

    h = cat(1, w, c)
    h = Conv1D(T,750,300,300,150)(h)
    h = relu(h)
    h = Linear(T,300,ntags)(h)

    compile((data_w,data_c), h)
end
