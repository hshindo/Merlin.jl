struct Model
    fw
    fc
    fs
    O
    q0
    q1
    Y1
    Y2
    conv
end

function Model(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int) where T
    fw = @graph n begin
        Node(Lookup(wordembeds), n)
    end

    d = size(charembeds, 1)
    fc = @graph n begin
        n = Node(Lookup(charembeds), n)
        n = Node(Conv1D(T,5d,5d,2d,d), n)
        Node(max, n, 2)
    end

    d = size(wordembeds,1) + size(charembeds,1)*5
    fs = @graph n begin
        n = Node(Conv1D(T,5d,2d,2d,d), n)
        Node(relu, n)
    end
    O = Linear(T,2d,ntags)
    q0 = zerograd(rand(T,ntags,1))
    q1 = zerograd(rand(T,ntags,1))
    Y1 = zerograd(rand(T,ntags,ntags))
    Y2 = zerograd(rand(T,ntags,ntags))
    conv = Conv1D(T,6d,ntags,2d,2d)
    Model(fw, fc, fs, O, q0, q1, Y1, Y2, conv)
end

function (m::Model)(word::Var, chars::Vector{Var})
    w = m.fw(word)
    cs = Var[]
    for i = 1:length(chars)
        push!(cs, m.fc(chars[i]))
    end
    c = cat(2, cs...)
    s = cat(1, w, c)
    h = m.fs(s)
    u = m.O(h)
    F = m.conv(h)
    Q = softmax(-u)
    n = length(chars)
    for i = 1:5
        L = m.Y1 * cat(2, m.q0, Q)[:,1:n]
        R = m.Y2 * cat(2, Q, m.q1)[:,2:n+1]
        QQ = L + R
        u += QQ
        Q = softmax(-u)
    end
    Q
end
