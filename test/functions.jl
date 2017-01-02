const T = Float32

@testset "functions" for i = 1:5

# argmax
x = rand(T,10,5,2)
for d = 1:ndims(x)
    y = argmax(x, d)
end

# blas
A = Var(rand(T,10,5))
x = Var(rand(T,5))
@test checkgrad(()->gemv('N',T(0.3),A,x))

x1 = Var(rand(T,10,5))
x2 = Var(rand(T,5,10))
x3 = Var(rand(T,10,5))
@test checkgrad(()->gemm('N','N',T(0.2),x1,x2), x1, x2)
@test checkgrad(()->gemm('N','T',T(0.3),x1,x3), x1, x3)
@test checkgrad(()->gemm('T','N',T(0.4),x1,x3), x1, x3)
@test checkgrad(()->gemm('T','T',T(0.5),x1,x2), x1, x2)

# concat
x1 = Var(rand(T,10,5,2))
x2 = Var(rand(T,10,5,2))
x3 = Var(rand(T,10,5,2))
for dim = 1:3
    @test checkgrad(()->concat(dim,x1,x2,x3), x1, x2, x3)
    #@test_grad concat(dim,x1,x2,x3) x1 x2 x3
end

# conv
x = Var(rand(T,5,4,3,2))
f = Conv(T, (2,2,3,4), pads=(0,0), strides=(1,1))
@test checkgrad(()->f(x), x, f.w, f.b)

# crossentropy
p = Var([1:5;])
x = Var(rand(T,10,5))
@test checkgrad(()->crossentropy(p,x), x)

# dropout
x = Var(rand(T,10,5))
y = dropout(x, 0.5)
#@test checkgrad(()->dropout(x,0.5), x)

# getindex
x = Var(rand(T,10,5))
@test checkgrad(()->x[1:3], x)
@test checkgrad(()->x[10:10], x)

x = Var(rand(T,100,1))
h = Var(rand(T,100,1))
f = GRU(T, 100)
#@test checkgrad(()->f(x,h), x, h)

# linear
x = Var(rand(T,10,5))
f = Linear(T, 10, 7)
f.b = Var(rand(T,size(f.b)))
@test checkgrad(()->f(x), x, f.w, f.b)

# math
x1 = Var(rand(T,10,5))
x2 = Var(rand(T,10,5))
x3 = Var(rand(T,10,5))
for op in (+,-,.*)
    @test checkgrad(()->op(x1,x2), x1, x2)
    @test checkgrad(()->op(x1,x3), x1, x3)
    @test checkgrad(()->op(x3,x1), x3, x1)
end
x4 = Var(rand(T,5,7))
@test checkgrad(()->-x4, x4)
#@test checkgrad(()->x1*x4, x1, x4)

# pairwise
x1 = Var(rand(T,10,5))
x2 = Var(rand(T,8,6))
@test checkgrad(()->pairwise(x1,x2), x1, x2)

# pooling
x = Var(rand(T,5,4,3,2))
@test checkgrad(()->pooling(:average,x,(2,2)), x)

# reduce
x = Var(rand(T,5,4,3,2))
for dim = 1:ndims(x)
    @test checkgrad(()->sum(x,dim), x)
    # @test checkgrad(()->max(x,dim), x)
end

# reshape
x = Var(rand(T,5,4,3))
@test checkgrad(()->reshape(x,3,4,5))
@test checkgrad(()->reshape(x,4,3,5))

# sigmoid
x = Var(rand(T,10,5))
@test checkgrad(()->sigmoid(x), x)

# tanh
x = Var(rand(T,10,5))
@test checkgrad(()->tanh(x), x)

# transpose
x = Var(rand(T,10,5))
@test checkgrad(()->transpose(x), x)

# softmax
x = Var(rand(T,10,5,3,4))
for f in (softmax,logsoftmax)
    @test checkgrad(()->f(x), x)
end

# view
x = Var(rand(T,5,4,3))
#@test checkgrad(()->view(x,1:3,2:2,3:3), x)

end
