const T = Float32

@testset "functions" for i = 1:5

@testset "activation" begin
    x = parameter(randn(T,10,5))
    for i = 1:length(x.data)
        abs(x.data[i]) < 0.1 && (x.data[i] += 1)
    end
    for f in (relu,sigmoid,tanh)
        checkgrad(()->f(x), x)
    end
end

@testset "cnn" begin
    x = parameter(randn(T,10,10))
    f = Conv1d(T, 5, 10, 7, padding=2)
    checkgrad(()->f(x,[3,7]), x, f.W, f.b)
end

@testset "loss" begin
    @testset "softmax_crossentropy" begin
        p = Var(rand(1:10,5))
        q = parameter(rand(T,10,5))
        checkgrad(()->softmax_crossentropy(p,q), p, q)
    end

    @testset "l2" begin
        x = parameter(rand(T,10,5))
        # @test_grad l2(x,0.01) x
    end

    @testset "mse" begin
        x1 = parameter(rand(T,10,5))
        x2 = parameter(rand(T,10,5))
        #@test_grad mse(x1,x2) x1 x2
    end
end

@testset "math" begin
    x = parameter(rand(T,10,5) .+ T(1))
    #@test_function exp x
    #@test_function log x

    x1 = parameter(randn(T,10,5))
    x2 = parameter(randn(T,10,5))
    checkgrad(()->x1+x2, x1, x2)
    checkgrad(()->x1-x2, x1, x2)
    checkgrad(()->x1.*x2, x1, x2)

    A = parameter(randn(T,10,5))
    B = parameter(randn(T,5,7))
    checkgrad(()->A*B, A, B)

    x1 = parameter(randn(T,10,5))
    x2 = parameter(randn(T,10,1))
    checkgrad(()->x1.+x2, x1, x2)
    checkgrad(()->x1.-x2, x1, x2)
    checkgrad(()->x1.*x2, x1, x2)
end

@testset "reduction" begin
    x = parameter(randn(T,10,10)*T(10))
    @testset "max" begin
        for dim = 1:2
            checkgrad(()->max(x,dim), x)
        end
        checkgrad(()->max(x,[2,5,3]), x)
    end
end

@testset "rnn" begin
    x = parameter(randn(T,10,10))
    for nlayers = 1:1
        f = LSTM(T, 10, 5, nlayers, 0.0, true)
        checkgrad(()->f(x,[5,3,2]), x, f.weights...)
    end
end

@testset "blas" begin
    A = parameter(randn(T,10,5))
    x = parameter(randn(T,10))
    checkgrad(()->gemv('T',1,A,x), A, x)

    A = parameter(randn(T,10,5))
    B = parameter(randn(T,10,5))
    checkgrad(()->gemm('N','T',1,A,B), A, B)

    A = parameter(randn(T,10,5,7))
    B = parameter(randn(T,10,5,7))
    #test_gradient(gemm_batch, 'N', 'T', 1, A, B)
    #test_cuda(gemm_batch, 'N', 'T', 1, A, B)
end

@testset "concat" begin
    x1 = parameter(randn(T,10,5,2))
    x2 = parameter(randn(T,10,5,2))
    x3 = parameter(randn(T,10,5,2))
    for dim = 1:3
        checkgrad(()->concat(dim,x1,x2,x3), x1, x2, x3)
    end
end

@testset "dropout" begin
    x = parameter(randn(T,10,5))
    dropout(x, 0.5)
end

@testset "getindex" begin
    x = parameter(randn(T,10,5,4))
    checkgrad(()->x[2:7,:,1:3], x)
    x = parameter(randn(T,10,5))
    checkgrad(()->x[:,3:3], x)
end

@testset "linear" begin
    x = parameter(randn(T,10,5))
    f = Linear(T, 10, 7, init_b=Uniform(-1,1))
    checkgrad(()->f(x), x, f.W, f.b)
end

@testset "lookup" begin
    w = parameter(randn(T,10,15))
    x = Var(rand(1:15,10))
    checkgrad(()->lookup(w,x), w, x)
end

@testset "pack" begin
    x = parameter(randn(T,10,10))
    dims = [2, 5, 3]
    checkgrad(()->pack(x,[2,5,3],0), x)
end

@testset "reshape" begin
    x = parameter(randn(T,10,1,5))
    checkgrad(()->reshape(x,5,10), x)
    checkgrad(()->vec(x), x)
    checkgrad(()->dropdims(x), x)
end

@testset "softmax" begin
    x = parameter(rand(T,10,5))
    checkgrad(()->softmax(x), x)
    logsoftmax(x)
end

@testset "window1d" begin
    x = parameter(randn(T,10,10))
    ksize, padding, stride, dilation = 5, 2, 1, 1
    checkgrad(()->window1d(x,[3,7],ksize,padding,stride,dilation), x)
end

end
