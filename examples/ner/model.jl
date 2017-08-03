struct Model
    fw
    fc
    fs
end

function Model(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int) where T
    fw = @graph n begin
        n = Node
        Node(Lookup(wordembeds), n)
    end

    d = size(charembeds, 1)
    fc = @graph n begin
        n = Node(Lookup(charembeds), n)
        n = Node(Conv1D(T,5d,5d,2d,d), n)
        Node(max, n, 2)
    end

    d = size(wordembeds,1) + size(charembeds,1)*5
    # n1 = Node(Conv1D(T,10d,2d,4d,2d), n)
    fs = @graph (n,b) begin
        n = Node(dropout, n, 0.5, b)
        n = Node(Conv1D(T,5d,2d,2d,d), n)
        n = Node(relu, n)
        nn = Node(dropout, n, 0.5, b)
        nn = Node(Conv1D(T,6d,2d,2d,2d), nn)
        nn = Node(relu, nn)
        n = n + nn
        nn = Node(dropout, n, 0.5, b)
        nn = Node(Conv1D(T,6d,2d,2d,2d), nn)
        nn = Node(relu, nn)
        n = n + nn
        nn = Node(dropout, n, 0.5, b)
        nn = Node(Conv1D(T,6d,2d,2d,2d), nn)
        nn = Node(relu, nn)
        n = n + nn
        nn = Node(dropout, n, 0.5, b)
        nn = Node(Conv1D(T,6d,2d,2d,2d), nn)
        nn = Node(relu, nn)
        n = n + nn
        Node(Linear(T,2d,ntags), n)
    end
    Model(fw, fc, fs)
end

function (m::Model)(word::Var, chars::Vector{Var}, istrain::Bool)
    w = m.fw(word)
    cs = Var[]
    for i = 1:length(chars)
        push!(cs, m.fc(chars[i]))
    end
    c = cat(2, cs...)
    s = cat(1, w, c)
    m.fs(s,Var(istrain))
end
