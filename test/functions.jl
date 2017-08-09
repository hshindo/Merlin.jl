const T = Float32
const eps = 1e-3

@testset "activation" for i = 1:5
    x = Var(rand(T,10,5))
    relu(x)
    #clipped_relu(x)
    @testgrad eps sigmoid(x) x
    @testgrad eps tanh(x) x
end

@testset "cat" for i = 1:5
    x1 = Var(rand(T,10,5,2))
    x2 = Var(rand(T,10,5,2))
    for dim = 1:3
        @testgrad eps cat(dim,x1,x2) x1 x2
    end
end

@testset "cnn" for i = 1:5
    x = Var(rand(T,10,5))
    f = Conv1D(T, 30, 10, 10, 10, dilation=1)
    @testgrad eps f(x) x f.w f.b
end

@testset "loss" for i = 1:5
    p = Var(rand(1:10,5))
    q = Var(softmax(rand(T,10,5)))
    # @testgrad 1e-2 crossentropy(p,q) q

    # softmax_crossentropy
    p = Var(rand(1:10,5))
    q = Var(rand(T,10,5))
    @testgrad eps softmax_crossentropy(p,q) q
end

@testset "getindex" for i = 1:5
    x = Var(rand(T,10,5))
    @testgrad eps x[1:3,:] x
    @testgrad eps x[2:10,3] x
end

@testset "linear" for i = 1:5
    x = Var(rand(T,10,5))
    f = Linear(T, 10, 7)
    @testgrad eps f(x) x f.w f.b
end

@testset "lookup" for i = 1:5
    x = Var(rand(1:100,10))
    f = Lookup(T, 100, 10)
    y = f(x)
end

@testset "reduce" for i = 1:5
    x = Var(rand(T,10,5)+1)
    for dim = 1:ndims(x.data)
        max(x, dim)
        #@test checkgrad(()->sum(x,dim), x)
    end
end

@testset "softmax" for i = 1:5
    x1 = Var(rand(T,10)+1)
    x2 = Var(rand(T,10,5)+1)
    for x in (x1,x2)
        @testgrad eps softmax(x) x
        #@test checkgrad(()->logsoftmax(x), x, eps=1e-2)
    end
end

@testset "window" for i = 1:5
    x = Var(rand(T,10,5))
    @testgrad eps window1d(x,30,10,10,1) x
end
