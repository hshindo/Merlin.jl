const T = Float32

@testset "functions" for i = 1:5

# activation
x = zerograd(rand(T,10,5))
relu(x)
clipped_relu(x)
@test checkgrad(sigmoid, x)
@test checkgrad(tanh, x)

# argmax
x = Var(rand(T,10,7,5,3))
for dim = 1:ndims(x.data)
    y = argmax(x.data, dim)
end

# conv
x = Var(rand(T,5,4,3,2))
#f = Conv(T, (2,2,3,4), pads=(0,0), strides=(1,1))
#@testgrad f(x) x f.w f.b

# linear
x = zerograd(rand(T,10,5))
f = Linear(T, 10, 7)
f.b = Var(rand(T,size(f.b.data)))
@test checkgrad(linear, x, f.w, f.b)

end
