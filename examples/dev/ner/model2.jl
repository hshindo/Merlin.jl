using Merlin

type Model
    wordfun::Lookup
    charfun::Graph
    sentfun::Graph
    outfun
    W
    M
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
    sentfun = Graph([w,c], [h])

    outfun = Linear(T,300,ntags)
    W = zerograd(fill(T(-0.001),1,600))
    M = zerograd(uniform(T,-0.001,0.001,ntags,ntags))

    Model(wordfun, charfun, sentfun, outfun, W, M)
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

    K = Var(-ones(Float32,size(w.data,2),size(w.data,2))/100)
    for i = 1:size(K.data,1)
        K.data[i,i] = 0
    end

    for i = 1:1
        Q1 = Q * K
        Q2 = -U + Q1
        Q3 = softmax(Q2)
        #if any(isnan, Q3.data)
        #    println("Error")
        #    println(Q2.data)
        #    throw("")
        #end
        Q = Q3
    end

    x = Q
    y == nothing ? argmax(x,1) : crossentropy(y,x,false)
end
