const T = Float32

@testset "functions" for i = 1:5

# activation
x = zerograd(rand(T,10,5))
relu(x)
clipped_relu(x)
@test checkgrad(sigmoid, x)
@test checkgrad(tanh, x)

# argmax
x = rand(T,10,7,5,3)
for dim = 1:ndims(x)
    y = argmax(x, dim)
end

# blas
A = zerograd(rand(T,10,5))
x = zerograd(rand(T,5))
@test checkgrad(gemv, 'N', 0.3, A, x)
#@test checkgrad(gemv, 'T', 0.3, transpose(A), x)

x1 = zerograd(rand(T,10,5))
x2 = zerograd(rand(T,5,10))
x3 = zerograd(rand(T,10,5))
@test checkgrad(gemm, 'N', 'N', 0.2, x1, x2)
@test checkgrad(gemm, 'N', 'T', 0.3, x1, x3)
@test checkgrad(gemm, 'T', 'N', 0.4, x1, x3)
@test checkgrad(gemm, 'T', 'T', 0.5, x1, x2)

# cat
x1 = zerograd(rand(T,10,5,2))
x2 = zerograd(rand(T,10,5,2))
x3 = zerograd(rand(T,10,5,2))
for dim = 1:3
    @test checkgrad(cat, dim, x1, x2, x3)
end

# linear
x = zerograd(rand(T,10,5))
f = Linear(T, 10, 7)
@test checkgrad(linear, x, f.w, f.b)

# math
x1 = zerograd(rand(T,10,5))
x2 = zerograd(rand(T,10,5))
@test checkgrad(exp, x1)
#@test checkgrad(log, x1)
for op in (+, -, .*)
    @test checkgrad(op, x1, x2)
end
@test checkgrad(-, x1)
x3 = zerograd(rand(T,10,1))
x4 = zerograd(rand(T,1,5))
for op in (.+, .-, .*)
    @test checkgrad(op, x1, x3)
    @test checkgrad(op, x1, x4)
end
x5 = zerograd(rand(T,5,7))
@test checkgrad(*, x1, x5)


end
