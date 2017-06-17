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
for dim = 1:3
    @test checkgrad(()->cat(dim,x1,x2), x1, x2)
end

end
