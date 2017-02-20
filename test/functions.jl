const T = Float32

@testset "functions" for i = 1:5

# activation
x = zerograd(rand(T,10,5))
relu(x)
clipped_relu(x)
@test checkgrad(sigmoid, x)
@test checkcuda(sigmoid, x)
@test checkgrad(tanh, x)
@test checkcuda(tanh, x)

# argmax
x = rand(T,10,7,5,3)
for dim = 1:ndims(x)
    y = argmax(x, dim)
end

# blas
A = zerograd(rand(T,10,5))
x = zerograd(rand(T,5))
@test checkgrad(gemv, 'N', 0.3, A, x)
@test checkgrad(gemv, 'T', 0.3, transpose(A), x)

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

# crossentropy
p = Var(rand(0:10,5))
q = zerograd(rand(T,10,5))
@test checkgrad(crossentropy, p, q)

# dropout
x = zerograd(rand(T,10,5))
dropout(x, 0.5)

# getindex
x = zerograd(rand(T,10,5))
@test checkgrad(getindex, x, 1:3)
@test checkgrad(getindex, x, 10:10)

# gru
x = zerograd(rand(T,100))
h = zerograd(rand(T,100))
f = GRU(T,100)
@test checkgrad(f, x, h)

# linear
x = zerograd(rand(T,10,1))
f = Linear(T, 10, 7)
@test checkgrad(linear, x, f.w, f.b)
#@test checkcuda(linear, x, f.w, f.b)

# lookup
x = Var(rand(1:10000,10,5))
f = Lookup(T, 10000, 100)
y = f(x)

# lstm
T = Float32
lstm = LSTM(T, 10, ()->rand()*0.02-0.01)
x = zerograd(rand(T,10,20))
@test checkgrad(lstm, x)

# math
x1 = zerograd(rand(T,10,5))
x2 = zerograd(rand(T,10,5))
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

# normalize
x = zerograd(rand(T,10,5))
@test checkgrad(normalize, x)

# pairwise
x1 = zerograd(rand(T,10,5))
x2 = zerograd(rand(T,8,6))
@test checkgrad(pairwise, x1, x2)

# pooling

# reduce
x = zerograd(rand(T,10,5))
for dim = 1:ndims(x.data)
    max(x, dim)
    @test checkgrad(sum, x, dim)
end

# reshape
x = zerograd(rand(T,5,4,3))
@test checkgrad(reshape, x, 3, 4, 5)
@test checkgrad(reshape, x, 4, 3, 5)

# softmax
x = zerograd(rand(T,10,5))
@test checkgrad(softmax, x)
@test checkcuda(softmax, x)
@test checkgrad(logsoftmax, x, eps=1e-2)

# view
x = zerograd(rand(T,10,5))
@test checkgrad(view, x, (3:6,1:4))

# window
x = zerograd(rand(T,10,5))
@test checkgrad(window, x, (10,))

end
