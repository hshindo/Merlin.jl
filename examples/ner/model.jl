struct Model
    fw
    fc
    fs
end

function Model(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int) where {T}
    fw = @graph x begin
        Lookup(wordembeds)(x)
    end

    d = size(charembeds, 1)
    fc = @graph x begin
        x = Lookup(charembeds)(x)
        x = Conv1D(T,5d,5d,2d,d)(x)
        max(x, 2)
    end

    d = size(wordembeds,1) + size(charembeds,1)*5
    # n1 = Node(Conv1D(T,10d,2d,4d,2d), n)
    fs = @graph (x,b) begin
        x = Conv1D(T,5d,2d,2d,d)(x)
        x = relu(x)
        # xx = dropout(x, 0.5, b)
        x = Conv1D(T,6d,2d,2d,2d)(x)
        x = relu(x)
        Linear(T,2d,ntags)(x)
    end
    Model(fw, fc, fs)
end

function (m::Model)(word::Var, char::Var, istrain::Bool)
    w = m.fw(word)
    c = m.fc(char)
    c.batchdims = w.batchdims
    s = cat(1, w, c)
    m.fs(s,Var(istrain))
end
