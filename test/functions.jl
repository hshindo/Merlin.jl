const T = Float32


@testset "functions" for i = 1:5

# activation
x = Var(rand(T,10,15), [1:5...])
relu(x)
#clipped_relu(x)
@test checkgrad(()->sigmoid(x), x)
@test checkgrad(()->tanh(x), x)

# cat
x1 = Var(rand(T,10,5,2))
x2 = Var(rand(T,10,5,2))
for dim = 1:3
    @test checkgrad(()->cat(dim,[x1,x2]), x1, x2)
end

# conv1d
x = Var(rand(T,10,15), [1:5...])
f = Conv1D(T, 10, 7, 0, 1)
@test checkgrad(()->f(x), x)

# linear
x = Var(rand(T,10,15), [1:5...])
f = Linear(T, 10, 7)
@test checkgrad(()->f(x), x, f.w, f.b)

# math
x = Var(rand(T,10,5)+1)
for op in (exp,log,transpose)
    @test checkgrad(()->op(x), x)
end

x1 = Var(rand(T,10,15), [1:5...])
x2 = Var(rand(T,10,15), [1:5...])
for op in (+,-)
    @test checkgrad(()->op(x1,x2), x1, x2)
end

end
