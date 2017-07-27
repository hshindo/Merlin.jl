struct Model
    fw
    fc
    fs
end

function Model{T}(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int)
    fw = @graph n begin
        Node(Lookup(wordembeds), n)
    end
    fc = @graph n begin
        n = Node(Lookup(T,100,10), n)
        n = Node(Conv1D(T,50,50,20,10), n)
        Node(max, n, 2)
    end
    fs = @graph n begin
        n = Node(Conv1D(T,750,300,300,150), n)
        n = Node(relu, n)
        Node(Linear(T,300,ntags), n)
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
