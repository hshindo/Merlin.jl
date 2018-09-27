const T = Float32

@testset "functions" for i = 1:5

@testset "activation" begin
    x = param(randn(T,10,5))
    for i = 1:length(x.data)
        abs(x.data[i]) < 0.1 && (x.data[i] += 1)
    end
    for f in (relu,sigmoid,tanh)
        @test_grad f(x) x
        @test_cuda f(x) x
    end
end

@testset "cnn" begin
    x = param(randn(T,20,10))
    dims = [3, 7]
    conv = Conv1d(T, 5, 20, 15, padding=2)
    @test_grad conv(x,dims) x
    @test_cuda conv(x,dims) x
end

@testset "loss" begin
    @testset "softmax_crossentropy" begin
        p = Var(rand(1:10,5))
        q = param(rand(T,10,5))
        @test_grad softmax_crossentropy(p,q) q
        @test_cuda softmax_crossentropy(p,q) q
    end

    @testset "l2" begin
        x = param(rand(T,10,5))
        # @test_grad l2(x,0.01) x
    end

    @testset "mse" begin
        x1 = param(rand(T,10,5))
        x2 = param(rand(T,10,5))
        #@test_grad mse(x1,x2) x1 x2
    end
end

@testset "math" begin
    x = param(rand(T,10,5) .+ T(1))
    #@test_function exp x
    #@test_function log x

    x1 = param(randn(T,10,5))
    x2 = param(randn(T,10,5))
    @test_grad x1+x2 x1 x2
    @test_cuda x1+x2 x1 x2
    @test_grad x1-x2 x1 x2
    @test_cuda x1-x2 x1 x2
    @test_grad x1.*x2 x1 x2
    @test_cuda x1.*x2 x1 x2

    A = param(randn(T,10,5))
    B = param(randn(T,5,7))
    @test_grad A*B A B
    @test_cuda A*B A B

    x1 = param(randn(T,10,5))
    x2 = param(randn(T,10,1))
    @test_grad x1.+x2 x1 x2
    @test_cuda x1.+x2 x1 x2
    @test_grad x1.-x2 x1 x2
    @test_cuda x1.-x2 x1 x2
    @test_grad x1.*x2 x1 x2
    @test_cuda x1.*x2 x1 x2
end

@testset "reduction" begin
    x = param(randn(T,10,10)*T(10))
    @testset "max" begin
        for dim = 1:2
            #@test_grad max(x,dim) x
            #@test_cuda max(x,dim) x
        end
        #@test_grad max(x,[2,5,3]) x
        #@test_cuda max(x,[2,5,3]) x
    end
end

@testset "rnn" begin
    x = param(randn(T,20,10))
    dims = [5, 3, 2]
    for nlayers = 1:1
        lstm = LSTM(T, 20, 15, nlayers, 0.0, true)
        @test_grad lstm(x,dims) x
        @test_cuda lstm(x,dims) x
    end
end

@testset "blas" begin
    A = param(randn(T,10,5))
    x = param(randn(T,10))
    @test_grad gemv('T',1,A,x) A x
    @test_cuda gemv('T',1,A,x) A x

    A = param(randn(T,10,5))
    B = param(randn(T,10,5))
    @test_grad gemm('N','T',1,A,B) A B
    @test_cuda gemm('N','T',1,A,B) A B

    A = param(randn(T,10,5,7))
    B = param(randn(T,10,5,7))
    #test_gradient(gemm_batch, 'N', 'T', 1, A, B)
    #test_cuda(gemm_batch, 'N', 'T', 1, A, B)
end

@testset "concat" begin
    x1 = param(randn(T,10,5,2))
    x2 = param(randn(T,10,5,2))
    x3 = param(randn(T,10,5,2))
    for dim = 1:3
        @test_grad concat(dim,x1,x2,x3) x1 x2 x3
        @test_cuda concat(dim,x1,x2,x3) x1 x2 x3
    end
end

@testset "dropout" begin
    x = param(randn(T,10,5))
    dropout(x, 0.5)
end

@testset "getindex" begin
    x = param(randn(T,10,5,4))
    @test_grad x[2:7,:,1:3] x
    @test_cuda x[2:7,:,1:3] x
    x = param(randn(T,10,5))
    @test_grad x[:,3:3] x
    @test_cuda x[:,3:3] x
end

@testset "linear" begin
    x = param(randn(T,10,5))
    f = Linear(T, 10, 7, init_b=Uniform(-1,1))
    @test_grad f(x) x f.W f.b
    @test_cuda f(x) x f.W f.b
end

@testset "lookup" begin
    w = param(randn(T,10,15))
    x = Var(rand(1:15,10))
    @test_grad lookup(w,x) w
    @test_cuda lookup(w,x) w
end

@testset "pack" begin
    x = param(randn(T,10,10))
    dims = [2, 5, 3]
    @test_grad pack(x,dims,0) x
    @test_cuda pack(x,dims,0) x
end

@testset "reshape" begin
    x = param(randn(T,10,1,5))
    @test_grad reshape(x,5,10) x
    @test_cuda reshape(x,5,10) x
    @test_grad vec(x) x
    @test_cuda vec(x) x
    @test_grad dropdims(x) x
    #@test_cuda dropdims x
end

@testset "softmax" begin
    x = param(rand(T,10,5))
    @test_grad softmax(x) x
    @test_cuda softmax(x) x
    logsoftmax(x)
end

end
