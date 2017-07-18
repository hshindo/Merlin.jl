struct Model
    fw
    fc
    fs
end

function Model(ntags::Int)
    T = Float32
    fw = @graph w begin
        wordembeds = h5read(wordembeds_file, "v")
        Lookup(wordembeds)(w)
    end
    fc = @graph c begin
        c = Lookup(T,100,10)(c)
        c = Conv1D(T,50,50,20,10)(c)
        max(c, 2)
    end
    fs = @graph s begin
        s = Conv1D(T,750,300,300,150)(s)
        s = relu(s)
        Linear(T,300,ntags)(s)
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
