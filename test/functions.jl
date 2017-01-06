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

# concat
x1 = zerograd(rand(T,10,5,2))
x2 = zerograd(rand(T,10,5,2))
x3 = zerograd(rand(T,10,5,2))
for dim = 1:3
    @test checkgrad(concat, dim, x1, x2, x3)
end

# conv
x = Var(rand(T,5,4,3,2))
#f = Conv(T, (2,2,3,4), pads=(0,0), strides=(1,1))
#@testgrad f(x) x f.w f.b

# crossentropy
p = Var([1:5;])
q = zerograd(rand(T,10,5))
#@test checkgrad(crossentropy, p, q)

# linear
x = zerograd(rand(T,10,5))
f = Linear(T, 10, 7)
#f.w.grad = nothing
@test checkgrad(linear, x, f.w, f.b)

# reduce
x = zerograd(rand(T,10,5))
for dim = 1:ndims(x.data)
    @test checkgrad(sum, x, dim)
end

# window
x = zerograd(rand(T,10,5))
@test checkgrad(window, x, (10,))

end
