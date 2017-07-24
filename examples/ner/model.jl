type Model
    fw
    fc
    fs
end

function Model(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int) where T
    fw = @graph w begin
        Lookup(wordembeds)(w)
    end
    fc = @graph c begin
        c = Lookup(charembeds)(c)
        d = size(charembeds, 1)
        c = Conv1D(T,5d,5d,2d,d)(c)
        max(c, 2)
    end
    fs = @graph s begin
        d = size(wordembeds,1) + size(charembeds,1)*5
        s = Conv1D(T,5d,2d,2d,d)(s)
        s = relu(s)
        Linear(T,2d,ntags)(s)
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
