const T = Float32

@testset "functions" for i = 1:5

# activation
x = Var(rand(T,10,5))
relu(x)
#clipped_relu(x)
@test checkgrad(()->sigmoid(x), x)
@test checkgrad(()->tanh(x), x)

# cat
x1 = Var(rand(T,10,5,2))
x2 = Var(rand(T,10,5,2))
for dim = 1:3
    @test checkgrad(()->cat(dim,x1,x2), x1, x2)
end

# conv1d
x = Var(rand(T,10,5))
f = Conv1D(T, 10, 7, 0, 1)
@test checkgrad(()->f(x), x)

# linear
x = Var(rand(T,10,5))
f = Linear(T, 10, 7)
@test checkgrad(()->f(x), x, f.w, f.b)

# lookup
x = Var(rand(1:100,10))
f = Lookup(T, 100, 10)
y = f(x)

# reduce
x = Var(rand(T,10,5)+1)
for dim = 1:2
    max(x, dim)
    #@test checkgrad(()->sum(x,dim), x)
end

# softmax_crossentropy
p = Var(rand(1:10,5))
q = Var(rand(T,10,5))
@test checkgrad(()->softmax_crossentropy(p,q), q)

# window
x = Var(rand(T,10,5))
@test checkgrad(()->window1d(x,10,1,1), x)

# softmax
x1 = Var(rand(T,10)+1)
x2 = Var(rand(T,10,5)+1)
for x in (x1,x2)
    @test checkgrad(()->softmax(x), x)
    #@test checkgrad(()->logsoftmax(x), x, eps=1e-2)
end

### loss ###
x1 = Var(rand(T,10,5))
x2 = Var(rand(T,10,5))
@test checkgrad(()->mse(x1,x2), x1)

end
