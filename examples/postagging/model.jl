type Model
    wordfun
    charfun
    sentfun
end

function setup_charfun{T}(embeds::Matrix{T})
    lu = Lookup(embeds)
    f = Linear(T,50,50)
    x -> begin
        y = lu(x)
        y = window(y, (50,), pads=(20,), strides=(10,))
        y = f(y)
        y = max(y, 2)
        y
    end
end

function Model{T}(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int)
    wordfun = Lookup(wordembeds)

    charfs = [Lookup(charembeds), Linear(T,50,50)]
    function charfun(x::Var)
        y = charfs[1](x)
        y = window(y, (50,), pads=(20,), strides=(10,))
        y = charfs[2](y)
        y = max(y, 2)
        y
    end

    sentfs = [Linear(T,750,300), Linear(T,300,ntags)]
    function sentfun(w::Var, c::Var)
        y = cat(1, w, c)
        y = window(y, (750,), pads=(300,), strides=(150,))
        y = sentfs[1](y)
        y = relu(y)
        y = sentfs[2](y)
    end

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

#=
function Model{T}(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int)
    x = Var()
    y = Lookup(charembeds)(x)
    y = window(y, (50,), pads=(20,), strides=(10,))
    y = Linear(T,50,50)(y)
    y = max(y, 2)
    charfun = Graph(y, x)

    w = Var() # word vector
    c = Var() # chars vector
    y = cat(1, w, c)
    y = window(y, (750,), pads=(300,), strides=(150,))
    y = Linear(T,750,300)(y)
    y = relu(y)
    y = Linear(T,300,ntags)(y)
    sentfun = Graph(y, w, c)

    Model(Lookup(wordembeds), charfun, sentfun)
end
=#

#=
function (m::Model)(w::Var, cs::Vector{Var}, y=nothing)
    wmat = m.wordfun(w)
    cvecs = map(m.charfun, cs)
    cmat = cat(2, cvecs...)
    x = m.sentfun(wmat, cmat)
    if y == nothing
        argmax(x.data, 1)
    else
        crossentropy(y, x)
    end
end
=#
