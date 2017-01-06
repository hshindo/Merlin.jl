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
f.b = zerograd(rand(T,size(f.b.data)))
#f.b = Var(zeros(T,7,1))
@test checkgrad(linear, x, f.w, f.b)

end
