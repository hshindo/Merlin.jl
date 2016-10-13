type Model
    wordfun
    charfun
    sentfun
end

function Model(wordembeds, charembeds)
    T = Float32

    x = Var()
    y = charembeds(x)
    y = window(y, (50,), strides=(10,), pads=(20,))
    y = Linear(T,50,50)(y)
    y = max(y, 2)
    charfun = Graph(y, x)

    w = Var() # word vector
    c = Var() # chars vector
    y = concat(1, w, c)
    y = window(y, (750,), strides=(150,), pads=(300,))
    y = Linear(T,750,300)(y)
    y = relu(y)
    y = Linear(T,300,45)(y)
    sentfun = Graph(y, w, c)

    Model(wordembeds, charfun, sentfun)
end

function (m::Model)(tokens::Vector{Token})
    wordvec = map(t -> t.word, tokens)
    wordvec = reshape(wordvec, 1, length(wordvec))
    wordmat = m.wordfun(constant(wordvec))

    charvecs = map(tokens) do t
        charvec = reshape(t.chars, 1, length(t.chars))
        m.charfun(constant(charvec))
    end
    charmat = concat(2, charvecs)
    m.sentfun(wordmat, charmat)
end
