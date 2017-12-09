const T = Float32

@testset "activation" for i = 1:5
    x = Var(randn(T,10,5))
    for i = 1:length(x.data)
        abs(x.data[i]) < 0.1 && (x.data[i] += 1)
    end
    beta = Var([T(1)])

    @testgrad crelu(x) x
    @testgrad elu(x) x
    @testgrad leaky_relu(x) x
    @testgrad relu(x) x
    @testgrad selu(x) x
    @testgrad sigmoid(x) x
    @testgrad swish(x,beta) x beta
    @testgrad tanh(x) x
end

@testset "blas" for i = 1:5
    A = Var(randn(T,10,5))
    x = Var(randn(T,10))
    B = Var(randn(T,10,5))
    @testgrad BLAS.gemm('T','N',1,A,B) A B
    @testgrad BLAS.gemv('T',1,A,x) A x
end

@testset "cat" for i = 1:5
    x1 = zerograd(randn(T,10,5,2))
    x2 = zerograd(randn(T,10,5,2))
    for dim = 1:3
        @testgrad cat(dim,x1,x2) x1 x2
    end
end

@testset "conv" for i = 1:5
    x = Var(rand(T,10,15))
    f = Conv1D(T, 5, 10, 20, 2, 1)
    @testgrad f(x,[10,5]) x f.W f.b
end

@testset "getindex" for i = 1:5
    x = Var(rand(T,10,10))
    @testgrad x[1:3,:] x
    @testgrad x[:,1:5] x
    @testgrad x[3:7,2:8] x
end

@testset "linear" for i = 1:5
    x = Var(rand(T,10,5))
    f = Linear(T, 10, 7)
    @testgrad f(x) x f.W f.b
end

@testset "loss" for i = 1:5
    # l2
    x = Var(rand(T,10,5))
    @testgrad l2(x,0.01) x

    # mse
    x1 = Var(rand(T,10,5))
    x2 = Var(rand(T,10,5))
    @testgrad mse(x1,x2) x1 x2

    p = Var(rand(1:10,5))
    q = Var(softmax(rand(T,10,5)))
    #@testgrad crossentropy(p,q) q

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
    @testgrad x1.+x3 x1 x3
    @testgrad x1.-x3 x1 x3
    @testgrad x1.*x2 x1 x3
    @testgrad x1*x4 x1 x4
end

@testset "recurrent" for i = 1:5
    x = Var(rand(T,20,15))
    f = LSTM(T, 20, 20)
    @testgrad f(x,[2,3,5]) x
    f = BiLSTM(T, 20, 20)
    @testgrad f(x,[2,3,5]) x
end

@testset "reduction" for i = 1:5
    x = Var(rand(T,10,15)+1)
    max_batch(x, [2,3,5])
    for dim = 1:ndims(x.data)
        max(x, dim)
        #@testgrad sum(x,dim) x
        #@testgrad mean(x,dim) x
    end
end

@testset "reshape" for i = 1:5
    x = Var(randn(T,10,5))
    @testgrad reshape(x,5,10) x
end

@testset "softmax" for i = 1:5
    x1 = Var(randn(T,10)+3)
    x2 = Var(randn(T,10,5)+3)
    for x in (x1,x2)
        @testgrad softmax(x) x
        #@testgrad logsoftmax(x) x
    end
end

@testset "standardize" for i = 1:5
    x = Var(randn(T,1,5)*3+2)
    f = Standardize(T,size(x.data))
    @testgrad f(x,true) x f.scale f.bias
end
