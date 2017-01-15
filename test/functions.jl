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
@test checkgrad(gemv, 'N', T(0.3), A, x)

x1 = zerograd(rand(T,10,5))
x2 = zerograd(rand(T,5,10))
x3 = zerograd(rand(T,10,5))
@test checkgrad(gemm, 'N', 'N', T(0.2), x1, x2)
@test checkgrad(gemm, 'N', 'T', T(0.3), x1, x3)
@test checkgrad(gemm, 'T', 'N', T(0.4), x1, x3)
@test checkgrad(gemm, 'T', 'T', T(0.5), x1, x2)

# concat
x1 = zerograd(rand(T,10,5,2))
x2 = zerograd(rand(T,10,5,2))
x3 = zerograd(rand(T,10,5,2))
for dim = 1:3
    @test checkgrad(cat, dim, x1, x2, x3)
end

# crossentropy
p = Var(rand(0:10,5))
q = zerograd(rand(T,10,5))
@test checkgrad(crossentropy, p, q)

# getindex
x = zerograd(rand(T,10,5))
@test checkgrad(getindex, x, 1:3)
@test checkgrad(getindex, x, 10:10)

# linear
x = zerograd(rand(T,10,1))
f = Linear(T, 10, 7)
#f.w.grad = nothing
#f.b.grad = nothing
@test checkgrad(linear, x, f.w, f.b)
#@test checkcuda(linear, x, f.w, f.b)

# math
x1 = zerograd(rand(T,10,5))
x2 = zerograd(rand(T,10,5))
for op in (+, -, .*)
    @test checkgrad(op, x1, x2)
end
@test checkgrad(-, x1)
# broadcast
x3 = zerograd(rand(T,10,1))
x4 = zerograd(rand(T,1,5))
for op in (.+, .-, .*)
    @test checkgrad(op, x1, x3)
    @test checkgrad(op, x1, x4)
end
# multiply
x5 = zerograd(rand(T,5,7))
#@test checkgrad(*, x1, x5)

# normalize
x = zerograd(rand(T,10,5))
@test checkgrad(normalize, x)

# pairwise
x1 = zerograd(rand(T,10,5))
x2 = zerograd(rand(T,8,6))
@test checkgrad(pairwise, x1, x2)

# reduce
x = zerograd(rand(T,10,5))
for dim = 1:ndims(x.data)
    @test checkgrad(sum, x, dim)
end

# reshape
x = zerograd(rand(T,5,4,3))
@test checkgrad(reshape, x, 3, 4, 5)
@test checkgrad(reshape, x, 4, 3, 5)

# softmax
x = zerograd(rand(T,10,5))
@test checkgrad(softmax, x)
@test checkgrad(logsoftmax, x, eps=1e-2)

# view
x = zerograd(rand(T,10,5))
@test checkgrad(view, x, (3:6,1:4))

# window
x = zerograd(rand(T,10,5))
@test checkgrad(window, x, (10,))

end
