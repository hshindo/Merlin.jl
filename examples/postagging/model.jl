function create_model(chardim::Int, worddim::Int, numtags::Int)
    T = Float32

    w = Lookup()(x)

    h = Lookup()(x)
    h = Conv1D(T,50,20,10))(h)
    c = max(h, 2)
    compile(c, x)

    h = cat(1, w, c)
    h = Conv1D(T,750,300,150)(h)
    h = relu(h)
    h = Linear(T,300,numtags)(h)

    compile(h, x, c)
end
