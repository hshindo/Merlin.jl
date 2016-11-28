type Model
    wordfun
    charfun
    sentfun
end

function Model(wordembeds, charembeds, ntags::Int)
    T = Float32

    x = Var()
    y = charembeds(x)
    y = window(y, (50,), strides=(10,), pads=(20,))
    y = Linear(T,50,50)(y)
    y = max(y, 2)
    charfun = compile(y, x)

    w = Var() # word vector
    c = Var() # chars vector
    y = concat(1, w, c)
    y = window(y, (750,), strides=(150,), pads=(300,))
    y = Linear(T,750,300)(y)
    y = relu(y)
    y = Linear(T,300,ntags)(y)
    sentfun = compile(y, w, c)

    Model(wordembeds, charfun, sentfun)
end

function (m::Model)(tokens::Vector{Token})
    wordvec = map(t -> t.word, tokens)
    wordvec = reshape(wordvec, 1, length(wordvec))
    wordmat = m.wordfun(Var(wordvec))

    charvecs = map(tokens) do t
        #Var(zeros(Float32,50,1))
        charvec = reshape(t.chars, 1, length(t.chars))
        m.charfun(Var(charvec))
    end
    charmat = concat(2, charvecs)
    m.sentfun(wordmat, charmat)
end
