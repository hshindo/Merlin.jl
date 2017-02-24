const T = Float32

@testset "functions" for i = 1:5

# activation
x = zerograd(rand(T,10,5))
relu(x)
clipped_relu(x)
@test checkgrad(sigmoid, x)
@test checkgrad(tanh, x)

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
    #@test checkgrad(op, x1, x3)
    #@test checkgrad(op, x1, x4)
end
#x5 = zerograd(rand(T,5,7))
#@test checkgrad(*, x1, x5)

x1 = zerograd(rand(T,7,2))
x2 = zerograd(rand(T,7,2))
@test checkgrad(.+, x1, x2)

end
