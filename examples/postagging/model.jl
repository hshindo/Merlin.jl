type Model
    wordfun::Lookup
    charfun::Graph
    sentfun::Graph
end

function Model{T}(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int)
    wordfun = Lookup(wordembeds)

    x = Var([1,2,3])
    h = Lookup(charembeds)(x)
    h = window(h, (50,), pads=(20,), strides=(10,))
    h = Linear(T,50,50)(h)
    h = max(h, 2)
    charfun = Graph([x], [h])

    w = Var(rand(T,100,3))
    c = Var(rand(T,50,3))
    h = cat(1, w, c)
    h = window(h, (750,), pads=(300,), strides=(150,))
    h = Linear(T,750,300)(h)
    h = relu(h)
    h = Linear(T,300,ntags)(h)
    sentfun = Graph([w,c], [h])

    Model(wordfun, charfun, sentfun)
end

function (m::Model)(input::Tuple{Var,Vector{Var}}, y=nothing)
    w, cs = input
    wmat = m.wordfun(w)
    cvecs = map(m.charfun, cs)
    cmat = cat(2, cvecs)
    x = m.sentfun(wmat, cmat)
    y == nothing ? argmax(x,1) : crossentropy(y,x)
end
