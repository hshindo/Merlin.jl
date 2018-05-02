const T = Float32

@testset "functions" for i = 1:1

##### activation #####
x = param(randn(T,10,5))
for i = 1:length(x)
    abs(x.data[i]) < 0.1 && (x.data[i] += 1)
end
for f in (relu,sigmoid,tanh)
    @test_grad f x
    @test_cuda f x
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

# softmax_crossentropy
#=
p = Var(Array{Int32}(rand(1:10,5)))
q = param(rand(T,10,5))
test_gradient!(softmax_crossentropy, p, q)
test_cuda!(softmax_crossentropy, p, q)
p = Var(softmax(rand(T,10,5)))
test_gradient!(softmax_crossentropy, p, q)
test_cuda!(softmax_crossentropy, p, q)
=#

##### reduction #####
x = param(randn(T,10,5)*T(10))
for dim = 1:2
    @test_grad max x dim
    @test_cuda max x dim
end
# @test_grad max_batch x (3,2)
# @test_cuda max_batch x [3,2]

##### blas #####
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

##### concat #####
x1 = param(randn(T,10,5,2))
x2 = param(randn(T,10,5,2))
x3 = param(randn(T,10,5,2))
for dim = 1:3
    @test_grad concat dim x1 x2 x3
    @test_cuda concat dim x1 x2 x3
end

#=
@testset "conv" for i = 1:5
    #x = param(curandn(T,10,10,5,4))
    #conv = Conv(T, 1, 1, 5, 3)
    #conv = cuda(conv)
    #y = conv(x)
    #gradient!(y)
end
=#

##### dropout #####
x = param(randn(T,10,5))
@test_function dropout x 0.5

##### getindex #####
x = param(randn(T,10,5,4))
@test_grad getindex x (2:7,:,1:3)
@test_cuda getindex x (2:7,:,1:3)

##### linear #####
x = param(randn(T,10,5))
f = Linear(T, 10, 7, init_b=Uniform(-0.01,0.01))
@test_grad linear x f.w f.b
@test_cuda linear x f.w f.b

##### lookup #####
w = param(randn(T,10,15))
x = Var(Array{Int32}(rand(1:15,10)))
@test_grad lookup w x
@test_cuda lookup w x

##### math #####
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

##### reshape #####
x = param(randn(T,10,1,5))
@test_grad reshape x (5,10)
@test_cuda reshape x (5,10)
@test_grad vec x
@test_cuda vec x
@test_grad squeeze x
@test_cuda squeeze x

##### rnn #####
x = param(randn(T,20,10))
batchdims = [5,3,2]
for nlayers = 1:1
    #lstm = LSTM(T, 20, 15, nlayers, 0.0, true)
    #@test_grad lstm x batchdims
    #@test_cuda lstm x batchdims
end

##### softmax #####
x = param(rand(T,10,5))
@test_grad softmax x
@test_cuda softmax x
@test_function logsoftmax x

##### split #####

##### window1d #####
x = param(randn(T,10,10))
@test_grad window1d x 2 [5,3,2]

#=
@testset "standardize" for i = 1:5
    x = param(randn(T,1,5)*3+2)
    #f = Standardize(T,size(x.data))
    #@testgrad f(x,true) x f.scale f.bias
end

@testset "transpose_batch" for i = 1:5
    x = param(randn(T,10,5))

    #f = Standardize(T,size(x.data))
    #@testgrad f(x,true) x f.scale f.bias
end
=#

end
