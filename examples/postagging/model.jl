type Model
    wordfun
    charfun
    sentfun
end

function Model{T}(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int)
    wordfun = Lookup(wordembeds)

    x = Var()
    h = Lookup(charembeds)(x)
    h = window(h, (50,), pads=(20,), strides=(10,))
    h = Linear(T,50,50)(h)
    h = max(h, 2)
    charfun = Graph()

    w = Var()
    c = Var()
    h = cat(1, w, c)
    h = window(h, (750,), pads=(300,), strides=(150,))
    h = Linear(T,750,300)(h)
    h = relu(h)
    h = Linear(T,300,ntags)(h)
    sentfun = Graph()

    Model(wordfun, charfun, sentfun)
end

function (m::Model)(x::Tuple{Var,Vector{Var}}, y=nothing)
    w, cs = x
    wmat = m.wordfun(w)
    cvecs = map(m.charfun, cs)
    cmat = cat(2, cvecs...)
    x = m.sentfun(wmat, cmat)
    y == nothing ? argmax(x.data,1) : crossentropy(y,x)
end
