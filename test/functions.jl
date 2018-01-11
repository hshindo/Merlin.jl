const T = Float32

function Base.isapprox(x::Array, y::CuArray)
    tol = 1e-3
    maximum(abs, x-Array(y)) <= tol
end

test_cuda(args...) = test_backend(CUDABackend(0), args...)

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
    # gradient!(y)
    # test_cuda(conv, x)
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
    test_gradient(lookup, w, x)
    test_cuda(lookup, w, x)
end

@testset "softmax" for i = 1:5
    x = zerograd(rand(T,10,5))
    test_gradient(softmax, x)
    test_cuda(softmax, x)
end
