struct Model
    fw
    fc
    fs
end

function Model(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int) where {T}
    fw = @graph x begin
        Lookup(wordembeds)(x)
    end
    fc = @graph x begin
        x = Lookup(T,100,10)(x)
        x = Conv1D(T,50,50,20,10)(x)
        max(x, 2)
    end
    fs = @graph x begin
        x = Conv1D(T,750,300,300,150)(x)
        x = relu(x)
        Linear(T,300,ntags)(x)
    end
    Model(fw, fc, fs)
end

function (m::Model)(word::Var, char::Var)
    w = m.fw(word)
    c = m.fc(char)
    c.batchdims = w.batchdims
    s = cat(1, w, c)
    m.fs(s)
end
