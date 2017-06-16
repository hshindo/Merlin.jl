const T = Float32


@testset "functions" for i = 1:5

# activation
x = Var(BatchedArray([rand(T,10,i) for i=1:5]))
relu(x)
#clipped_relu(x)
@test checkgrad(()->sigmoid(x), x)
#@test checkgrad(()->tanh(x), x)

# cat
x1 = Var(BatchedArray([rand(T,10,5,2),rand(T,10,5,3)]))
x2 = Var(BatchedArray([rand(T,10,5,2),rand(T,10,5,3)]))
for dim = 1:3
    @test checkgrad(()->cat(dim,x1,x2), x1, x2)
end

# conv1d
x = Var(BatchedArray([rand(T,10,i) for i=1:5]))
f = Conv1D(T, 10, 5, 0, 1)
@test checkgrad(()->f(x), x)

# crossentropy
p = Var(rand(0:10,5))
q = Var(rand(T,10,5))
@test checkgrad(()->crossentropy(p,q), q)

# linear
x = Var(BatchedArray([rand(T,10,4),rand(T,10,5)]))
#x = Var(rand(T,10,10))
f = Linear(T, 10, 7)
@test checkgrad(()->f(x), x, f.w, f.b)

# lookup
x = Var(rand(1:10000,10,5))
f = Lookup(T, 10000, 100)
y = f(x)

# reduce
x = Var(BatchedArray([rand(T,10,4),rand(T,10,5)]))
#x = Var(rand(T,10,5))
for dim = 1:2
    max(x, dim)
    #@test checkgrad(()->sum(x,dim), x)
end

end
