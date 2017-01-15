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

# linear
x = zerograd(rand(T,10,1))
f = Linear(T, 10, 7)
#f.w.grad = nothing
#f.b.grad = nothing
@test checkgrad(linear, x, f.w, f.b)
#@test checkcuda(linear, x, f.w, f.b)

# normalize
x = zerograd(rand(T,2,1))
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

# softmax
x = zerograd(rand(T,10,5))
@test checkgrad(softmax, x)
#@test checkgrad(logsoftmax, x)

# window
x = zerograd(rand(T,10,5))
@test checkgrad(window, x, (10,))

end
