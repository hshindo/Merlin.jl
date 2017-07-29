struct Model
    fw
    fc
    fs
end

function Model(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int) where T
    fw = @graph n begin
        Node(Lookup(wordembeds), n)
    end

    d = size(charembeds, 1)
    fc = @graph n begin
        n = Node(Lookup(charembeds), n)
        n = Node(Conv1D(T,3d,d,d,d), n)
        Node(max, n, 2)
    end

    d = size(wordembeds,1) + size(charembeds,1)
    fs = @graph n begin
        n = Node(BiLSTM(T,d,d), n)
        Node(Linear(T,2d,ntags), n)
    end
    Model(fw, fc, fs)
end

function (m::Model)(word::Var, chars::Vector{Var})
    w = m.fw(word)
    cs = Var[]
    for i = 1:length(chars)
        push!(cs, m.fc(chars[i]))
    end
    c = cat(2, cs...)
    s = cat(1, w, c)
    m.fs(s)
end
