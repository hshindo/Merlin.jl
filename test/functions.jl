const T = Float32

@testset "activation" for i = 1:5
    x = param(randn(T,10,5))
    for i = 1:length(x)
        abs(x.data[i]) < 0.1 && (x.data[i] += 1)
    end
    for f in (relu,sigmoid,tanh)
        @test_grad f x
        @test_cuda f x
    end
end

##### loss #####
# crossentropy
#p = Var(rand(1:10,5))
#q = param(softmax(rand(T,10,5)))
#test_gradient(crossentropy, p, q, tol=2e-3)
#p = Var(softmax(randn(T,10)))
#q = param(softmax(randn(T,10)))
# test_gradient(crossentropy, p, q, tol=2e-3)

# l2
#x = Var(rand(T,10,5))
#@testgrad l2(x,0.01) x

# mse
#x1 = param(rand(T,10,5))
#x2 = param(rand(T,10,5))
#test_gradient(mse, x1, x2)

@testset "loss" for i=1:5
    @testset "softmax_crossentropy" begin
        x = param(rand(T,10,5))
        y = Var(rand(1:10,5))
        @test_grad softmax_crossentropy x y
        @test_cuda softmax_crossentropy x y
    end
end

@testset "reduction" for i = 1:5
    x = param(randn(T,10,5)*T(10))
    xs = [param(randn(T,10,5)*T(10)) for i=1:5]
    @testset "max" begin
        for dim = 1:2
            @test_grad max x dim
            #@test_grad max xs dim
            @test_cuda max x dim
            #@test_cuda max xs dim
        end
    end
end

@testset "rnn" for i = 1:5
    xs = [param(randn(T,20,d)) for d in (5,3,2)]
    for nlayers = 1:1
        lstm = LSTM(T, 20, 15, nlayers, 0.0, true)
        #@test_grad lstm xs
        #@test_cuda lstm x batchdims
    end
end

@testset "blas" for i = 1:5
    A = param(randn(T,10,5))
    B = param(randn(T,10,5))
    @test_grad BLAS.gemm 'N' 'T' 1 A B
    @test_cuda BLAS.gemm 'N' 'T' 1 A B

    A = param(randn(T,10,5))
    B = param(randn(T,10))
    @test_grad BLAS.gemv 'T' 1 A B
    @test_cuda BLAS.gemv 'T' 1 A B

    A = param(randn(T,10,5,7))
    B = param(randn(T,10,5,7))
    #test_gradient(gemm_batch, 'N', 'T', 1, A, B)
    #test_cuda(gemm_batch, 'N', 'T', 1, A, B)
end

@testset "concat" for i = 1:5
    x1 = param(randn(T,10,5,2))
    x2 = param(randn(T,10,5,2))
    x3 = param(randn(T,10,5,2))
    for dim = 1:3
        @test_grad concat dim x1 x2 x3
        @test_cuda concat dim x1 x2 x3
    end
end

@testset "conv1d" for i = 1:5
    x = param(randn(T,20,10))
    batchdims = [3,7]
    conv = Conv1d(T, 5, 20, 15, padding=2)
    # @test_grad conv x batchdims
    # @test_cuda conv x batchdims
end

@testset "dropout" for i = 1:5
    x = param(randn(T,10,5))
    @test_function dropout x 0.5
end

@testset "getindex" for i = 1:5
    x = param(randn(T,10,5,4))
    @test_grad getindex x (2:7,:,1:3)
    @test_cuda getindex x (2:7,:,1:3)
end

@testset "linear" for i = 1:5
    x = param(randn(T,10,5))
    f = Linear(T, 10, 7, init_b=Uniform(-0.01,0.01))
    @test_grad linear x f.w f.b
    @test_cuda linear x f.w f.b
end

@testset "lookup" for i = 1:5
    w = param(randn(T,10,15))
    x = rand(1:15,10)
    @test_grad lookup w x
    @test_cuda lookup w x
end

@testset "math" for i = 1:5
    x = param(rand(T,10,5) + T(1))
    #@test_function exp x
    #@test_function log x

    x1 = param(randn(T,10,5))
    x2 = param(randn(T,10,5))
    @test_grad (+) x1 x2
    @test_cuda (+) x1 x2
    @test_grad (-) x1 x2
    @test_cuda (-) x1 x2
    @test_grad (-) x1
    @test_cuda (-) x1

    x1 = param(randn(T,10,5))
    x2 = param(randn(T,10,1))
    #@test_grad broadcast (+) x1 x2
    # test_gradient!(broadcast, -, x1, x2)
    #test_gradient!(broadcast, *, x1, x2)

    A = param(randn(T,10,5))
    B = param(randn(T,5,7))
    @test_grad (*) A B
    @test_cuda (*) A B
end

@testset "pad" for i = 1:5
    xs = [param(randn(T,10,5)) for i=1:5]
    @test_grad pad xs 0
end

@testset "reshape" for i = 1:5
    x = param(randn(T,10,1,5))
    @test_grad reshape x (5,10)
    @test_cuda reshape x (5,10)
    @test_grad vec x
    @test_cuda vec x
    @test_grad squeeze x
    @test_cuda squeeze x
end

@testset "softmax" for i = 1:5
    x = param(rand(T,10,5))
    @test_grad softmax x
    @test_cuda softmax x
    @test_function logsoftmax x
end

@testset "split" for i = 1:5
    x = param(rand(T,10,10))
    #@test_grad split x 2 [3,5,2]
end

##### split #####
#=
@testset "standardize" for i = 1:5
    x = param(randn(T,1,5)*3+2)
    #f = Standardize(T,size(x.data))
    #@testgrad f(x,true) x f.scale f.bias
end
=#
