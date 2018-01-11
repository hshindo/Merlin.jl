const T = Float32
const cuda = CUDABackend(0)

function Base.isapprox(x::Array, y::CuArray)
    tol = 1e-3
    maximum(abs, x-Array(y)) <= tol
end

test_cuda(args...) = test_backend(cuda, args...)

@testset "activation" for i = 1:5
    x = zerograd(randn(T,10,5))
    for i = 1:length(x.data)
        abs(x.data[i]) < 0.1 && (x.data[i] += 1)
    end
    beta = Var([T(1)])

    #@testgrad crelu(x) x
    #@testgrad elu(x) x
    #@testgrad leaky_relu(x) x
    #test_gradient(relu, x)
    test_gradient(sigmoid, x)
    test_gradient(tanh, x)
    #@testgrad selu(x) x
    #@test @checkgrad tol sigmoid(x) x
    #@testgrad swish(x,beta) x beta
    #@test @checkgrad tol tanh(x) x
    test_cuda(relu, x)
    test_cuda(sigmoid, x)
    test_cuda(tanh, x)
end

@testset "concat" for i = 1:5
    x1 = zerograd(randn(T,10,5,2))
    x2 = zerograd(randn(T,10,5,2))
    for dim = 1:3
        test_gradient(concat, dim, x1, x2)
        test_cuda(concat, dim, x1, x2)
    end
end

@testset "conv" for i = 1:5
    x = zerograd(curandn(T,10,10,5,4))
    conv = Conv(T, (1,1,5,3))
    conv = convert(cuda, conv)
    y = conv(x)
    gradient!(y)
end

@testset "dropout" for i = 1:5
    x = zerograd(randn(T,10,5))
    y = dropout(x, 0.5)
    gradient!(y)
    x = convert(cuda, x)
    y = dropout(x, 0.5)
    gradient!(y)
end

@testset "linear" for i = 1:5
    x = zerograd(randn(T,10,5))
    f = Linear(T, 10, 7)
    test_gradient(linear, x, f.w, f.b)
    test_cuda(linear, x, f.w, f.b)
end

@testset "lookup" for i = 1:5
    w = zerograd(randn(T,10,15))
    x = rand(1:15, 10)
    #test_gradient(lookup, w, x)
    #test_cuda(lookup, w, x)
end

@testset "loss" for i = 1:5
    # l2
    x = Var(rand(T,10,5))
    #@testgrad l2(x,0.01) x

    # mse
    x1 = Var(rand(T,10,5))
    x2 = Var(rand(T,10,5))
    #@testgrad mse(x1,x2) x1 x2

    p = Var(rand(1:10,5))
    q = Var(softmax(rand(T,10,5)))
    ##@testgrad crossentropy(p,q) q

    p1 = Var(rand(1:10,5))
    p2 = zerograd(softmax(rand(T,10,5)))
    q = zerograd(rand(T,10,5))
    test_gradient(softmax_crossentropy, p1, q)
    test_cuda(softmax_crossentropy, p1, q)
    #@testgrad softmax_crossentropy(p2,q) q
end

@testset "reshape" for i = 1:5
    x = zerograd(randn(T,10,5))
    test_gradient(reshape, x, 5, 10)
    test_cuda(reshape, x, 5, 10)
end

@testset "rnn" for i = 1:5
    x = Var(rand(T,20,10))
    f = LSTM(T, 20, 20)
    @testgrad f(x,[2,3,5]) x f.WU f.b f.h0 f.c0
    @testgrad f(x,[2,3,5], rev=true) x f.WU f.b f.h0 f.c0
    #f = BiLSTM(T, 20, 20)
    #@testgrad f(x,[2,3,5]) x f.WU f.b f.h0 f.c0
end

@testset "softmax" for i = 1:5
    x = zerograd(rand(T,10,5))
    test_gradient(softmax, x)
    test_cuda(softmax, x)
end

@testset "standardize" for i = 1:5
    x = zerograd(randn(T,1,5)*3+2)
    #f = Standardize(T,size(x.data))
    #@testgrad f(x,true) x f.scale f.bias
end
