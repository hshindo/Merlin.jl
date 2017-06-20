function create_model(numtags::Int)
    T = Float32

    data_w = Var([1,2,3])
    data_c = Var([[1,2,3],[4,5,6]], [3,3])
    data_y = Var([1])

    w = Lookup(h5read(h5file,"v"))(data_w)

    c = Lookup(T,100,10)(data_c)
    c = Conv1D(T,50,20,10)(c)
    c = max(c, 2)

    h = cat(1, w, c)
    h = Conv1D(T,750,300,150)(h)
    h = relu(h)
    h = Linear(T,300,numtags)(h)
    crossentropy(data_y, h)
end
