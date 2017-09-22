const T = Float32

@testset "activation" for i = 1:5
    x = Var(randn(T,10,5))
    for i = 1:length(x.data)
        abs(x.data[i]) < 0.1 && (x.data[i] += 1)
    end
    @testgrad elu(x) x
    @testgrad relu(x) x
    @testgrad leaky_relu(x) x
    @testgrad crelu(x) x
    @testgrad selu(x) x
    @testgrad sigmoid(x) x
    @testgrad tanh(x) x
end

@testset "blas" for i = 1:5
    A = Var(randn(T,10,5))
    x = Var(randn(T,10))
    B = Var(randn(T,10,5))
    @testgrad BLAS.gemv('T',1,A,x) A x
    @testgrad BLAS.gemm('T','N',1,A,B) A B
end

@testset "concat" for i = 1:5
    x1 = Var(randn(T,10,5,2))
    x2 = Var(randn(T,10,5,2))
    for dim = 1:2
        @testgrad concat(dim,x1,x2) x1 x2
    end
end

@testset "conv" for i = 1:5
    x = Var(rand(T,10,15),[10,5])
    f = Conv1D(T, 5, 10, 20, 2, 1)
    @testgrad f(x) x f.w f.b
end

@testset "getindex" for i = 1:5
    x = Var(rand(T,10,5))
    #@testgrad x[1:3,:] x
    #@testgrad x[2:10,3] x
end

@testset "linear" for i = 1:5
    x = Var(rand(T,10,5))
    f = Linear(T, 10, 7)
    @testgrad f(x) x f.w f.b
end

@testset "loss" for i = 1:5
    p = Var(rand(1:10,5))
    q = Var(softmax(rand(T,10,5)))
    # @testgrad crossentropy(p,q) q

    # softmax_crossentropy
    p1 = Var(rand(1:10,5))
    p2 = Var(softmax(rand(T,10,5)))
    q = Var(rand(T,10,5))
    @testgrad softmax_crossentropy(p1,q) q
    @testgrad softmax_crossentropy(p2,q) q
end

@testset "math" for i = 1:5
    x = Var(rand(T,10,5))
    @testgrad exp(x) x
    # @testgrad log(x) x
    #@testgrad transpose(x) x
    @testgrad -x x
    #@testgrad x/2 x
    # @testgrad x^3 x

    x1 = Var(rand(T,10,5))
    x2 = Var(rand(T,10,5))
    x3 = Var(rand(T,10,1))
    x4 = Var(rand(T,5,4))
    @testgrad x1+x2 x1 x2
    @testgrad x1-x2 x1 x2
    #@testgrad x1.+x3 x1 x3
    #@testgrad x1.-x3 x1 x3
    #@testgrad x1.*x2 x1 x3
    #@testgrad x1*x4 x1 x4
end

@testset "reduction" for i = 1:5
    x = Var(rand(T,10,15)+1, [5,10])
    for dim = 1:ndims(x.data)
        max(x, dim)
        #@testgrad sum(x,dim) x
        #@testgrad mean(x,dim) x
    end
end

@testset "reshape" for i = 1:5
    x = Var(randn(T,10,5))
    #@testgrad reshape(x,5,10) x
end

@testset "resize" for i = 1:5
    x = Var(randn(T,10,5),[2,3])
    @testgrad resize(x,[4,1]) x
end

@testset "softmax" for i = 1:5
    x1 = Var(randn(T,10))
    x2 = Var(randn(T,10,5))
    for x in (x1,x2)
        @testgrad softmax(x) x
        #@test checkgrad(()->logsoftmax(x), x, eps=1e-2)
    end
end

@testset "split" for i = 1:5
    x = Var(randn(T,10,5))
    #@testgrad split(x,[2,3]) x
end

@testset "standardize" for i = 1:5
    x = Var(randn(T,1,5)*3+2)
    f = Standardize(T,size(x.data))
    @testgrad f(x) x f.scale f.bias
end
