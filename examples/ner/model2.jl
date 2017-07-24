using Merlin

type Model
    wordfun::Lookup
    charfun::Graph
    sentfun::Graph
    outfun
    W
    M
    ntags::Int
end

function Model{T}(wordembeds::Matrix{T}, charembeds::Matrix{T}, ntags::Int)
    wordfun = Lookup(wordembeds)

    x = Var([1,2,3])
    h = Lookup(charembeds)(x)
    h = window(h, (30*5,), pads=(30*2,), strides=(30,))
    h = Linear(T,150,50)(h)
    h = max(h, 2)
    charfun = Graph([x], [h])

    w = Var(rand(T,100,3))
    c = Var(rand(T,50,3))
    h = cat(1, w, c)
    h = window(h, (150*5,), pads=(150*2,), strides=(150,))
    h = Linear(T,750,300)(h)
    h = relu(h)
    sentfun = Graph([w,c], [h])

    outfun = Linear(T,300,ntags)
    W = zerograd(uniform(T,-0.001,0.001,ntags,(ntags+300)*7))
    M = zerograd(diagm(ones(T,ntags)))

    Model(wordfun, charfun, sentfun, outfun, W, M, ntags)
end

function (m::Model)(input::Tuple{Var,Vector{Var}}, y=nothing)
    w, cs = input
    wmat = m.wordfun(w)
    cvecs = map(m.charfun, cs)
    cmat = cat(2, cvecs...)
    H = m.sentfun(wmat, cmat)
    U = m.outfun(H)
    Q = softmax(U)

    #=
    Hs = [H[:,i] for i = 1:size(H.data,2)]
    Ks = Var[]
    for j = 1:size(w.data,2)
        for i = 1:size(w.data,2)
            x = i == j ? Var(zeros(Float32,600)) : cat(1,Hs[i],Hs[j])
            push!(Ks, x)
        end
    end
    K = m.W * cat(2, Ks...)
    K = reshape(K, size(w.data,2), size(w.data,2))
    =#
    #K = Var(ones(Float32,size(Q.data,2),size(Q.data,2)))
    #for i = 1:size(K.data,1)
    #    K.data[i,i] = 0
    #end

    for i = 1:5
        Q = cat(1, Q, H)
        Q = window(Q, ((m.ntags+300)*7,), pads=((m.ntags+300)*3,), strides=((m.ntags+300),))
        Q = U + m.M * m.W * Q
        Q = softmax(-Q)
    end
    x = Q
    y == nothing ? argmax(x,1) : crossentropy(y,x,false)
end
