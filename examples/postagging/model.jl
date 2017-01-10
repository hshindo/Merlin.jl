type Model
    wordfun
    charfun
    sentfun
end

function Model{T}(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int)
    x = Var()
    y = Lookup(charembeds)(x)
    y = window(y, (50,), pads=(20,), strides=(10,))
    y = Linear(T,50,50)(y)
    y = max(y, 2)
    charfun = compile(y, x)

    w = Var() # word vector
    c = Var() # chars vector
    y = concat(1, w, c)
    y = window(y, (750,), pads=(300,), strides=(150,))
    y = Linear(T,750,300)(y)
    y = relu(y)
    y = Linear(T,300,ntags)(y)
    sentfun = compile(y, w, c)

    Model(Lookup(wordembeds), charfun, sentfun)
end

function (m::Model)(w::Var, cs::Vector{Var}, y=nothing)
    wmat = m.wordfun(w)
    cvecs::Vector{Var} = map(m.charfun, cs)
    cmat = concat(2, cvecs)
    x = m.sentfun(wmat, cmat)
    if y == nothing
        argmax(x.data, 1)
    else
        crossentropy(y, x)
    end
end
